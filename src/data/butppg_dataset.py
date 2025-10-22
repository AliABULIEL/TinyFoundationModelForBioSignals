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
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import signal as scipy_signal
import wfdb



class BUTPPGDataset(Dataset):
    """BUT PPG dataset with multi-modal support.

    Supports two modes:
    1. **RAW mode** (default): Load from WFDB files on-the-fly (flexible, slower)
    2. **PREPROCESSED mode**: Load from window_*.npz files (fast, requires pre-processing)

    Args:
        data_dir: Root directory containing BUT PPG data
                  - RAW mode: Path to WFDB files (e.g., 'data/but_ppg/dataset')
                  - PREPROCESSED mode: Path to window_*.npz files (e.g., 'data/processed/butppg/windows_with_labels')
        modality: Single modality string, list of modalities, or 'all' for PPG+ECG+ACC
                 Options: 'ppg', 'ecg', 'acc', ['ppg', 'ecg'], 'ppg,ecg,acc', 'all'
        split: 'train', 'val', or 'test'
        window_sec: Window duration in seconds (default: 10.0)
        fs: Target sampling rate in Hz (default: 125 to match VitalDB)
        quality_filter: If True, filter out low quality signals (RAW mode only)
        return_participant_id: If True, return participant ID with segments
        return_labels: If True, return clinical labels
        segment_overlap: Overlap between consecutive windows (0.0-1.0, RAW mode only)
        random_seed: Random seed for reproducible splits (RAW mode only)
        train_ratio: Ratio of data for training (default: 0.7, RAW mode only)
        val_ratio: Ratio of data for validation (default: 0.15, RAW mode only)
        mode: 'raw' (default) or 'preprocessed'
        task: Task name for label filtering (PREPROCESSED mode only)
              Options: 'quality', 'hr', 'blood_pressure', etc.
        filter_missing: Filter samples with missing labels (PREPROCESSED mode only)

    Returns:
        (seg1, seg2): Two segments (RAW mode: from same participant, PREPROCESSED mode: same window twice)
        Each segment has shape [C, T] where:
        - C = number of channels (1 for PPG/ECG, 3 for ACC, or sum for multi-modal)
        - T = window_sec * fs samples

    Examples:
        # RAW mode (existing behavior - on-the-fly processing)
        dataset = BUTPPGDataset(
            data_dir='data/but_ppg/dataset',
            modality='all',
            split='train',
            mode='raw'  # Default
        )

        # PREPROCESSED mode (fast - load from pre-processed windows)
        dataset = BUTPPGDataset(
            data_dir='data/processed/butppg/windows_with_labels',
            modality=['ppg', 'ecg'],
            split='train',
            mode='preprocessed',
            task='quality',
            return_labels=True
        )
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
        cache_size: int = 500,
        mode: str = 'raw',  # NEW: 'raw' or 'preprocessed'
        task: Optional[str] = None,  # NEW: for task-specific label loading
        filter_missing: bool = True  # NEW: filter samples with missing labels
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.mode = mode  # NEW
        self.task = task  # NEW
        self.filter_missing = filter_missing  # NEW

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

        # Caching (for raw mode only)
        self.use_cache = use_cache
        self.cache_size = cache_size
        self.signal_cache = OrderedDict()
        self._failed_records = set()

        # Initialize based on mode
        if self.mode == 'preprocessed':
            self._init_preprocessed_mode()
        else:  # raw mode (default)
            self._init_raw_mode(train_ratio, val_ratio)

        # Print summary
        self._print_summary()
        
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

    def _init_raw_mode(self, train_ratio: float, val_ratio: float):
        """Initialize dataset in RAW mode (load from WFDB files)."""
        # Load annotations
        self._load_annotations()

        # Create participant splits
        self._create_splits(train_ratio, val_ratio)

        # Build segment pairs for SSL
        self._build_segment_pairs()

    def _init_preprocessed_mode(self):
        """Initialize dataset in PREPROCESSED mode (load from window NPZ files)."""
        # Look for window_*.npz files in split directory
        split_dir = self.data_dir / self.split
        if not split_dir.exists():
            # Try without split subdirectory
            split_dir = self.data_dir

        # Find all window files
        self.window_files = sorted(split_dir.glob('window_*.npz'))

        if len(self.window_files) == 0:
            raise FileNotFoundError(
                f"No window_*.npz files found in {split_dir}. "
                f"Did you run create_butppg_windows_with_labels.py?"
            )

        # For preprocessed mode, we load individual windows (not pairs)
        # If task is specified, filter by valid labels
        if self.task and self.filter_missing:
            self.valid_indices = self._find_valid_samples_preprocessed()
        else:
            self.valid_indices = list(range(len(self.window_files)))

        # Set some dummy values for compatibility
        self.split_participants = []
        self.split_records = []
        self.segment_pairs = []  # No pairs in preprocessed mode

    def _find_valid_samples_preprocessed(self) -> List[int]:
        """Find samples with valid labels for specified task."""
        from src.data.unified_window_loader import UnifiedWindowDataset

        # Use UnifiedWindowDataset's task definitions
        task_configs = UnifiedWindowDataset.TASK_CONFIGS

        if self.task not in task_configs:
            print(f"Warning: Unknown task '{self.task}', loading all samples")
            return list(range(len(self.window_files)))

        task_config = task_configs[self.task]
        label_keys = task_config['label_keys']

        valid_indices = []
        for idx, window_file in enumerate(self.window_files):
            try:
                data = np.load(window_file)

                # Check if all required labels are valid
                labels_valid = True
                for label_key in label_keys:
                    if label_key not in data:
                        labels_valid = False
                        break

                    label_value = data[label_key]
                    if isinstance(label_value, (np.ndarray, np.generic)):
                        label_value = label_value.item()

                    if np.isnan(label_value) or np.isinf(label_value) or label_value == -1:
                        labels_valid = False
                        break

                if labels_valid:
                    valid_indices.append(idx)

            except Exception:
                continue

        return valid_indices

    def _print_summary(self):
        """Print dataset summary."""
        print(f"\nBUT PPG Dataset initialized for {self.split}:")
        print(f"  Mode: {self.mode}")
        print(f"  Modalities: {self.modalities} ({self.n_channels} channels)")

        if self.mode == 'raw':
            print(f"  Participants: {len(self.split_participants)}")
            print(f"  Records: {len(self.split_records)}")
            print(f"  Positive pairs: {len(self.segment_pairs)}")
        else:
            print(f"  Window files: {len(self.window_files)}")
            print(f"  Valid samples: {len(self.valid_indices)}")
            if self.task:
                print(f"  Task: {self.task}")

        print(f"  Window: {self.window_sec}s @ {self.fs}Hz = {self.T} samples")
        print(f"  Output shape: [{self.n_channels}, {self.T}]")

    def _load_annotations(self):
        """Load quality annotations and subject info."""
        # Load subject info (demographics, motion, BP, SpO2, glycaemia)
        subject_path = self.data_dir / 'subject-info.csv'
        if subject_path.exists():
            self.subject_df = pd.read_csv(subject_path)
            print(f"  Loaded subject info: {len(self.subject_df)} entries")
        else:
            self.subject_df = None

        # Load quality and HR annotations
        quality_path = self.data_dir / 'quality-hr-ann.csv'
        if quality_path.exists():
            self.quality_hr_df = pd.read_csv(quality_path)
            print(f"  Loaded quality-hr annotations: {len(self.quality_hr_df)} entries")
        else:
            self.quality_hr_df = None
            
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
        if self.mode == 'preprocessed':
            return len(self.valid_indices)
        else:
            return len(self.segment_pairs) if self.segment_pairs else 1

    def __getitem__(self, idx):
        """Get sample (mode-dependent behavior)."""
        if self.mode == 'preprocessed':
            return self._getitem_preprocessed(idx)
        else:
            return self._getitem_raw(idx)

    def _getitem_raw(self, idx):
        """Get pair of segments from same participant (RAW mode)."""
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

    def _getitem_preprocessed(self, idx):
        """Get single window from preprocessed NPZ file."""
        # Map to actual file index
        file_idx = self.valid_indices[idx]
        window_file = self.window_files[file_idx]

        # Load data
        data = np.load(window_file)

        # Extract signal
        signal = data['signal']  # [n_channels, T]

        # Select channels based on modality
        # BUT-PPG NPZ format: [PPG, ECG] = channels [0, 1]
        # NOTE: No accelerometer data in BUT-PPG dataset
        channel_map = {'ppg': [0], 'ecg': [1]}

        selected_channels = []
        for mod in self.modalities:
            if mod == 'acc':
                # Accelerometer not available in BUT-PPG dataset
                # print(f"Warning: Accelerometer data not available in BUT-PPG dataset")
                continue
            if mod in channel_map:
                selected_channels.extend(channel_map[mod])

        if not selected_channels:
            raise ValueError(f"No valid channels found for modalities: {self.modalities}")

        signal = signal[selected_channels, :]  # Select requested channels

        # Convert to tensor
        signal = torch.from_numpy(signal).float()

        # Return based on what's requested
        if self.return_labels:
            # Extract ALL labels
            labels = self._extract_labels_from_npz(data)

            if self.return_participant_id:
                record_id = str(data['record_id'])
                participant_id = record_id[:3] if len(record_id) >= 3 else record_id
                return signal, signal, participant_id, labels  # Return (seg1, seg2, ...) for compatibility
            else:
                return signal, signal, labels  # Return (seg1, seg2, labels) for compatibility
        else:
            if self.return_participant_id:
                record_id = str(data['record_id'])
                participant_id = record_id[:3] if len(record_id) >= 3 else record_id
                return signal, signal, participant_id
            else:
                return signal, signal  # Return (seg1, seg2) for compatibility

    def _extract_labels_from_npz(self, data) -> Dict:
        """Extract all available labels from NPZ file."""
        labels = {}

        label_keys = [
            'quality', 'hr', 'motion', 'bp_systolic', 'bp_diastolic', 'spo2', 'glycaemia',
            'age', 'sex', 'bmi', 'height', 'weight'
        ]

        for key in label_keys:
            if key in data:
                value = data[key]
                if isinstance(value, (np.ndarray, np.generic)):
                    value = value.item()
                labels[key] = value
            else:
                labels[key] = -1

        return labels
            
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
        
    def _get_record_labels(self, record_id: str) -> Dict:
        """
        Get all clinical labels for a specific recording.

        Args:
            record_id: Recording ID (e.g., "100001")

        Returns:
            Dict with all available labels:
            - Demographics: age, sex, bmi, height, weight
            - Quality: quality (binary), hr (float)
            - Clinical: motion (int), bp_systolic, bp_diastolic, spo2, glycaemia
        """
        labels = {
            # Demographics
            'age': -1,
            'sex': -1,
            'bmi': -1,
            'height': -1,
            'weight': -1,
            # Quality/HR from quality-hr-ann.csv
            'quality': -1,
            'hr': -1,
            # Clinical from subject-info.csv
            'motion': -1,
            'bp_systolic': -1,
            'bp_diastolic': -1,
            'spo2': -1,
            'glycaemia': -1
        }

        try:
            record_num = int(record_id)

            # Extract from subject-info.csv
            if self.subject_df is not None:
                mask = self.subject_df['ID'] == record_num

                if mask.any():
                    row = self.subject_df[mask].iloc[0]

                    # Demographics
                    age = row.get('Age [years]', row.get('Age', -1))
                    labels['age'] = float(age) if not pd.isna(age) and age != -1 else -1

                    gender = row.get('Gender', '')
                    labels['sex'] = 1 if gender == 'M' else (0 if gender == 'F' else -1)

                    height = row.get('Height [cm]', row.get('Height', -1))
                    weight = row.get('Weight [kg]', row.get('Weight', -1))
                    labels['height'] = float(height) if not pd.isna(height) and height > 0 else -1
                    labels['weight'] = float(weight) if not pd.isna(weight) and weight > 0 else -1

                    if labels['height'] > 0 and labels['weight'] > 0:
                        labels['bmi'] = labels['weight'] / ((labels['height'] / 100) ** 2)

                    # Motion
                    motion = row.get('Motion', -1)
                    labels['motion'] = int(motion) if not pd.isna(motion) else -1

                    # Blood pressure (parse "120/80" format)
                    bp = row.get('Blood pressure [mmHg]', '')
                    if not pd.isna(bp) and str(bp).strip():
                        bp_str = str(bp).strip()
                        if '/' in bp_str:
                            try:
                                parts = bp_str.split('/')
                                labels['bp_systolic'] = float(parts[0])
                                labels['bp_diastolic'] = float(parts[1])
                            except (ValueError, IndexError):
                                pass

                    # SpO2
                    spo2 = row.get('SpO2 [%]', -1)
                    labels['spo2'] = float(spo2) if not pd.isna(spo2) and spo2 != -1 else -1

                    # Glycaemia
                    glyc = row.get('Glycaemia [mmol/l]', -1)
                    labels['glycaemia'] = float(glyc) if not pd.isna(glyc) and glyc != -1 else -1

            # Extract from quality-hr-ann.csv
            if self.quality_hr_df is not None:
                mask = self.quality_hr_df['ID'] == record_num

                if mask.any():
                    row = self.quality_hr_df[mask].iloc[0]

                    # Quality (binary)
                    quality = row.get('Quality', -1)
                    labels['quality'] = int(quality) if not pd.isna(quality) else -1

                    # Heart rate
                    hr = row.get('HR', -1)
                    labels['hr'] = float(hr) if not pd.isna(hr) and hr != -1 else -1

        except Exception as e:
            print(f"Error getting labels for record {record_id}: {e}")

        return labels

    def _get_participant_info(self, participant_id: str) -> Dict:
        """
        Get clinical labels for participant (uses first recording as representative).

        Args:
            participant_id: Participant ID (first 3 digits of record ID)

        Returns:
            Dict with all available clinical labels
        """
        # Find records for this participant
        records = self.participant_records.get(participant_id, [])
        if not records:
            return self._get_record_labels('')  # Return default -1 values

        # Use first record as representative
        return self._get_record_labels(records[0])


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
