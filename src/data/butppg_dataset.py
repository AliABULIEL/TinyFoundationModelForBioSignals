"""BUT PPG Dataset with multi-modal support.

This dataset supports loading multiple modalities from BUT PPG:
- PPG (photoplethysmography) 
- ECG (electrocardiogram)
- ACC (accelerometer - 3 axes)

Features:
- Multi-modal loading (PPG + ECG + ACC)
- Participant-level positive pairs (same participant, different recordings)
- Unified preprocessing matching VitalDB for transfer learning
"""

import hashlib
import json
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import signal as scipy_signal
import wfdb



class BUTPPGDataset(Dataset):
    """BUT PPG dataset with multi-modal support.
    
    Supports loading PPG, ECG, and ACC simultaneously for multi-modal learning.
    
    Args:
        data_dir: Root directory containing BUT PPG data
        modality: Single modality string, list of modalities, or 'all' for PPG+ECG+ACC
                 Options: 'ppg', 'ecg', 'acc', ['ppg', 'ecg'], 'ppg,ecg,acc', 'all'
        split: 'train', 'val', or 'test'
        window_sec: Window duration in seconds (default: 10.0)
        fs: Target sampling rate in Hz (default: 125 to match VitalDB)
        quality_filter: If True, filter out low quality signals
        return_participant_id: If True, return participant ID with segments
        return_labels: If True, return demographic labels
        segment_overlap: Overlap between consecutive windows (0.0-1.0)
        random_seed: Random seed for reproducible splits
        train_ratio: Ratio of data for training (default: 0.7)
        val_ratio: Ratio of data for validation (default: 0.15)
    
    Returns:
        (seg1, seg2): Two segments from same participant
        Each segment has shape [C, T] where:
        - C = number of channels (1 for PPG/ECG, 3 for ACC, or sum for multi-modal)
        - T = window_sec * fs samples
    """
    
    SUPPORTED_MODALITIES = ['ppg', 'ecg', 'acc']
    
    # Original sampling rates in BUT PPG
    ORIGINAL_FS = {
        'ppg': 30,   # Smartphone camera PPG
        'ecg': 1000, # ECG sensor
        'acc': 100   # Accelerometer
    }
    
    # Band-pass filter settings per modality
    FILTER_BANDS = {
        'ppg': (0.5, 8.0),   # PPG heart rate band
        'ecg': (0.5, 40.0),  # ECG standard band
        'acc': (0.1, 20.0)   # ACC movement band
    }
    
    def __init__(
        self,
        data_dir: str,
        modality: Union[str, List[str]] = 'ppg',
        split: str = 'train',
        window_sec: float = 10.0,
        fs: int = 125,
        quality_filter: bool = False,
        return_participant_id: bool = False,
        return_labels: bool = False,
        segment_overlap: float = 0.5,
        random_seed: Optional[int] = 42,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        use_cache: bool = True,
        cache_size: int = 500
    ):
        super().__init__()
        
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {data_dir}")
            
        # Parse modalities
        self.modalities = self._parse_modalities(modality)
        self.n_channels = self._get_total_channels()
        self.is_multimodal = len(self.modalities) > 1
        
        self.split = split
        self.window_sec = window_sec
        self.fs = fs  # Target sampling rate
        self.T = int(window_sec * fs)  # Samples per window
        
        self.quality_filter = quality_filter
        self.return_participant_id = return_participant_id
        self.return_labels = return_labels
        self.segment_overlap = segment_overlap
        self.random_seed = random_seed
        
        # Caching
        self.use_cache = use_cache
        self.cache_size = cache_size
        self.signal_cache = OrderedDict()
        self._failed_records = set()
        
        # Load annotations
        self._load_annotations()
        
        # Create participant splits
        self._create_splits(train_ratio, val_ratio)
        
        # Build segment pairs for SSL
        self._build_segment_pairs()
        
        print(f"\nBUT PPG Dataset initialized for {split}:")
        print(f"  Modalities: {self.modalities} ({self.n_channels} channels)")
        print(f"  Participants: {len(self.split_participants)}")
        print(f"  Records: {len(self.split_records)}")
        print(f"  Positive pairs: {len(self.segment_pairs)}")
        print(f"  Window: {window_sec}s @ {fs}Hz = {self.T} samples")
        print(f"  Output shape: [{self.n_channels}, {self.T}]")
        
    def _parse_modalities(self, modality: Union[str, List[str]]) -> List[str]:
        """Parse modality specification into list."""
        if isinstance(modality, str):
            if modality.lower() == 'all':
                return self.SUPPORTED_MODALITIES.copy()
            elif ',' in modality:
                return [m.strip().lower() for m in modality.split(',')]
            else:
                return [modality.lower()]
        elif isinstance(modality, list):
            return [m.lower() for m in modality]
        else:
            return ['ppg']  # Default
            
    def _get_total_channels(self) -> int:
        """Get total number of channels across all modalities."""
        total = 0
        for mod in self.modalities:
            if mod == 'acc':
                total += 3  # 3-axis accelerometer
            else:
                total += 1  # Single channel for PPG/ECG
        return total
        
    def _load_annotations(self):
        """Load quality annotations and subject info."""
        # Load subject info
        subject_path = self.data_dir / 'subject-info.csv'
        if subject_path.exists():
            import pandas as pd
            self.subject_df = pd.read_csv(subject_path)
            print(f"  Loaded subject info: {len(self.subject_df)} entries")
        else:
            self.subject_df = None
            
        # Build participant mapping
        self.participant_records = {}
        
        # Find all records by looking for WFDB files
        for modality in self.modalities:
            pattern = f"*/*_{modality.upper()}.hea"
            for hea_file in self.data_dir.glob(pattern):
                record_id = hea_file.parent.name
                # Extract participant ID (first 3 digits)
                participant_id = record_id[:3] if len(record_id) >= 3 else record_id
                
                if participant_id not in self.participant_records:
                    self.participant_records[participant_id] = []
                if record_id not in self.participant_records[participant_id]:
                    self.participant_records[participant_id].append(record_id)
                    
        print(f"  Found {len(self.participant_records)} participants")
        
    def _create_splits(self, train_ratio: float, val_ratio: float):
        """Create train/val/test splits at participant level."""
        all_participants = list(self.participant_records.keys())
        
        if len(all_participants) == 0:
            print("  Warning: No participants found!")
            self.split_participants = []
            self.split_records = []
            return
            
        # Sort and shuffle
        all_participants.sort()
        np.random.seed(self.random_seed)
        np.random.shuffle(all_participants)
        
        # Calculate splits
        n_total = len(all_participants)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Assign participants to splits
        if self.split == 'train':
            self.split_participants = all_participants[:n_train]
        elif self.split == 'val':
            self.split_participants = all_participants[n_train:n_train + n_val]
        else:  # test
            self.split_participants = all_participants[n_train + n_val:]
            
        # Get all records for split participants
        self.split_records = []
        for participant_id in self.split_participants:
            self.split_records.extend(self.participant_records[participant_id])
            
    def _build_segment_pairs(self):
        """Build positive pairs from same participant."""
        self.segment_pairs = []
        
        for participant_id in self.split_participants:
            records = self.participant_records[participant_id]
            n_records = len(records)
            
            if n_records == 0:
                continue
            elif n_records == 1:
                # Self-pair for single recording
                self.segment_pairs.append({
                    'participant_id': participant_id,
                    'record1': records[0],
                    'record2': records[0],
                    'is_self_pair': True
                })
            else:
                # Create pairs from different recordings
                n_pairs = min(20, (n_records * (n_records - 1)) // 2)
                
                pairs_created = set()
                for _ in range(n_pairs):
                    attempts = 0
                    while attempts < 100:
                        idx1 = np.random.randint(0, n_records)
                        idx2 = np.random.randint(0, n_records)
                        if idx1 != idx2:
                            pair = (min(idx1, idx2), max(idx1, idx2))
                            if pair not in pairs_created:
                                pairs_created.add(pair)
                                self.segment_pairs.append({
                                    'participant_id': participant_id,
                                    'record1': records[idx1],
                                    'record2': records[idx2],
                                    'is_self_pair': False
                                })
                                break
                        attempts += 1
                        
        # Shuffle pairs
        if self.segment_pairs:
            np.random.seed(self.random_seed)
            np.random.shuffle(self.segment_pairs)
            
    def __len__(self):
        return len(self.segment_pairs) if self.segment_pairs else 1
        
    def __getitem__(self, idx):
        """Get pair of segments from same participant."""
        if not self.segment_pairs:
            # Return zeros if no pairs
            zeros = torch.zeros(self.n_channels, self.T, dtype=torch.float32)
            return zeros, zeros
            
        pair_info = self.segment_pairs[idx % len(self.segment_pairs)]
        participant_id = pair_info['participant_id']
        record1 = pair_info['record1']
        record2 = pair_info['record2']
        is_self_pair = pair_info.get('is_self_pair', False)
        
        # Load multi-modal signals
        signals1 = self._load_multimodal_signals(record1)
        
        if is_self_pair:
            signals2 = signals1  # Use same signals
        else:
            signals2 = self._load_multimodal_signals(record2)
            
        # Create segments
        seg1 = self._create_segment(signals1, seed=idx)
        seg2 = self._create_segment(signals2, seed=idx + 1000 if is_self_pair else idx)
        
        # Stack modalities into multi-channel tensor
        seg1 = self._stack_modalities(seg1)
        seg2 = self._stack_modalities(seg2)
        
        # Convert to tensors
        seg1 = torch.from_numpy(seg1).float()
        seg2 = torch.from_numpy(seg2).float()
        
        # Return with optional metadata
        if self.return_labels and self.return_participant_id:
            labels = self._get_participant_info(participant_id)
            return seg1, seg2, participant_id, labels
        elif self.return_participant_id:
            return seg1, seg2, participant_id
        elif self.return_labels:
            labels = self._get_participant_info(participant_id)
            return seg1, seg2, labels
        else:
            return seg1, seg2
            
    def _load_multimodal_signals(self, record_id: str) -> Dict[str, np.ndarray]:
        """Load all modality signals for a record."""
        signals = {}
        record_dir = self.data_dir / record_id
        
        if not record_dir.exists():
            return signals
            
        for modality in self.modalities:
            # Check cache first
            cache_key = f"{record_id}_{modality}"
            if cache_key in self.signal_cache:
                signals[modality] = self.signal_cache[cache_key]
                continue
                
            # Load from WFDB files
            try:
                record_path = record_dir / f"{record_id}_{modality.upper()}"
                if not record_path.with_suffix('.hea').exists():
                    signals[modality] = None
                    continue
                    
                # Read WFDB record
                record = wfdb.rdrecord(str(record_path))
                signal_data = record.p_signal.T  # [channels, samples]
                
                # Handle channel specifics
                if modality == 'ppg':
                    # PPG might have RGB channels, average them
                    if signal_data.shape[0] > 1:
                        signal_data = np.mean(signal_data, axis=0, keepdims=True)
                elif modality == 'ecg':
                    # ECG should be single channel
                    if signal_data.shape[0] > 1:
                        signal_data = signal_data[0:1, :]
                elif modality == 'acc':
                    # ACC should have 3 channels (X, Y, Z)
                    if signal_data.shape[0] != 3:
                        # Pad or trim to 3 channels
                        if signal_data.shape[0] < 3:
                            padding = np.zeros((3 - signal_data.shape[0], signal_data.shape[1]))
                            signal_data = np.vstack([signal_data, padding])
                        else:
                            signal_data = signal_data[:3, :]
                            
                # Preprocess
                processed = self._preprocess_signal(signal_data, modality)
                
                # Cache
                if self.use_cache and len(self.signal_cache) < self.cache_size:
                    self.signal_cache[cache_key] = processed
                    
                signals[modality] = processed
                
            except Exception as e:
                print(f"Error loading {modality} for {record_id}: {e}")
                signals[modality] = None
                
        return signals
        
    def _preprocess_signal(self, signal_data: np.ndarray, modality: str) -> Optional[np.ndarray]:
        """Preprocess signal with modality-specific parameters."""
        if signal_data is None or signal_data.size == 0:
            return None
            
        try:
            # Remove NaN/Inf
            signal_data = np.nan_to_num(signal_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Ensure 2D
            if signal_data.ndim == 1:
                signal_data = signal_data[np.newaxis, :]
                
            # Get modality-specific parameters
            original_fs = self.ORIGINAL_FS[modality]
            band_low, band_high = self.FILTER_BANDS[modality]
            
            # Check minimum length
            if signal_data.shape[1] < original_fs:
                return None
                
            # Bandpass filter
            if signal_data.shape[1] >= 100:
                nyquist = original_fs / 2
                if band_high < nyquist * 0.95:
                    sos = scipy_signal.butter(
                        4, [band_low, band_high],
                        btype='band', fs=original_fs, output='sos'
                    )
                    signal_data = scipy_signal.sosfiltfilt(sos, signal_data, axis=1)
                    
            # Resample to target fs
            if original_fs != self.fs:
                n_samples = signal_data.shape[1]
                n_resampled = int(n_samples * self.fs / original_fs)
                
                resampled = np.zeros((signal_data.shape[0], n_resampled))
                for i in range(signal_data.shape[0]):
                    resampled[i] = scipy_signal.resample(signal_data[i], n_resampled)
                signal_data = resampled
                
            # Z-score normalization (per channel)
            mean = np.mean(signal_data, axis=1, keepdims=True)
            std = np.std(signal_data, axis=1, keepdims=True)
            std = np.where(std < 1e-8, 1.0, std)
            signal_data = (signal_data - mean) / std
            
            return signal_data.astype(np.float32)
            
        except Exception as e:
            print(f"Preprocessing error for {modality}: {e}")
            return None
            
    def _create_segment(self, signals: Dict[str, np.ndarray], seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Create segments from signals."""
        if seed is not None:
            np.random.seed(seed)
            
        segments = {}
        
        for modality, signal in signals.items():
            if signal is None:
                # Create zeros for missing modality
                if modality == 'acc':
                    segments[modality] = np.zeros((3, self.T), dtype=np.float32)
                else:
                    segments[modality] = np.zeros((1, self.T), dtype=np.float32)
                continue
                
            n_samples = signal.shape[1]
            
            if n_samples < self.T:
                # Tile if too short
                n_repeats = (self.T // n_samples) + 1
                extended = np.tile(signal, (1, n_repeats))
                segment = extended[:, :self.T]
            elif n_samples == self.T:
                segment = signal
            else:
                # Random crop if longer
                max_start = n_samples - self.T
                start = np.random.randint(0, max_start + 1)
                segment = signal[:, start:start + self.T]
                
            segments[modality] = segment
            
        return segments
        
    def _stack_modalities(self, segments: Dict[str, np.ndarray]) -> np.ndarray:
        """Stack all modality segments into multi-channel array."""
        stacked = []
        
        for modality in self.modalities:
            if modality in segments:
                stacked.append(segments[modality])
            else:
                # Add zeros for missing modality
                if modality == 'acc':
                    stacked.append(np.zeros((3, self.T), dtype=np.float32))
                else:
                    stacked.append(np.zeros((1, self.T), dtype=np.float32))
                    
        # Stack all channels
        return np.vstack(stacked)
        
    def _get_participant_info(self, participant_id: str) -> Dict:
        """Get demographic info for participant."""
        if self.subject_df is None:
            return {'age': -1, 'sex': -1, 'bmi': -1}
            
        # Find records for this participant
        records = self.participant_records.get(participant_id, [])
        if not records:
            return {'age': -1, 'sex': -1, 'bmi': -1}
            
        # Get first record ID to look up demographics
        try:
            record_num = int(records[0])
            mask = self.subject_df['ID'] == record_num
            
            if mask.any():
                row = self.subject_df[mask].iloc[0]
                
                # Extract demographics
                age = row.get('Age [years]', row.get('Age', -1))
                gender = row.get('Gender', '')
                sex = 1 if gender == 'M' else (0 if gender == 'F' else -1)
                
                # Calculate BMI
                height = row.get('Height [cm]', row.get('Height', 0))
                weight = row.get('Weight [kg]', row.get('Weight', 0))
                
                if height > 0 and weight > 0:
                    bmi = weight / ((height / 100) ** 2)
                else:
                    bmi = -1
                    
                return {
                    'age': float(age) if age != -1 else -1,
                    'sex': sex,
                    'bmi': float(bmi) if bmi != -1 else -1
                }
                
        except Exception as e:
            print(f"Error getting participant info: {e}")
            
        return {'age': -1, 'sex': -1, 'bmi': -1}


def create_butppg_dataloaders(
    data_dir: str,
    modality: Union[str, List[str]] = 'ppg',
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    quality_filter: bool = False,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders for BUT PPG.
    
    Args:
        data_dir: Root directory containing BUT PPG data
        modality: Modalities to load ('ppg', 'ecg', 'acc', ['ppg', 'ecg'], 'all')
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for data loading
        pin_memory: Pin memory for GPU transfer
        quality_filter: If True, filter out low quality signals
        **dataset_kwargs: Additional arguments for BUTPPGDataset
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = BUTPPGDataset(
        data_dir=data_dir,
        modality=modality,
        split='train',
        quality_filter=quality_filter,
        **dataset_kwargs
    )
    
    val_dataset = BUTPPGDataset(
        data_dir=data_dir,
        modality=modality,
        split='val',
        quality_filter=quality_filter,
        **dataset_kwargs
    )
    
    test_dataset = BUTPPGDataset(
        data_dir=data_dir,
        modality=modality,
        split='test',
        quality_filter=quality_filter,
        **dataset_kwargs
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader
