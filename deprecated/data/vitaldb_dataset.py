"""VitalDB Dataset for SSL pretraining with multi-modal support.

This dataset loads VitalDB cases with support for multiple modalities:
- PPG (PLETH track)
- ECG (ECG_II track)

Features:
- Multi-modal loading (PPG + ECG)
- Participant-level positive pairs (same patient, different time segments)
- Modality dropout for robust SSL pretraining
- Subject-level train/val/test splits (no leakage)
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import signal as scipy_signal


class VitalDBDataset(Dataset):
    """VitalDB dataset for SSL pretraining with multi-modal support.

    Supports TWO modes:
    1. **RAW mode** (default): Load from VitalDB API on-the-fly (flexible, slower)
    2. **PREPROCESSED mode**: Load from window_*.npz files (fast, requires pre-processing)

    Args:
        data_dir: Root directory containing train/val/test subdirectories or cache dir
        split: Split name ('train', 'val', 'test')
        channels: Single modality string, list of modalities, or 'all' for both PPG+ECG
                 Options: 'ppg', 'ecg', ['ppg', 'ecg'], 'ppg,ecg', 'all'
        window_sec: Window duration in seconds (default: 10.0)
        fs: Sampling rate in Hz (default: 125)
        return_pairs: If True, return two views (seg1, seg2) for contrastive learning
        apply_modality_dropout: If True, randomly zero entire channels during training
        modality_dropout_prob: Probability of dropping each channel (default: 0.25)
        transform: Optional transform to apply to signals
        cache_dir: Directory for caching VitalDB signals (RAW mode only)
        mode: 'raw' (load from VitalDB API) or 'preprocessed' (load from window NPZ files)
        task: Task name for label filtering (PREPROCESSED mode only)
              Options: 'mortality', 'icu_days', 'asa', etc.
        return_labels: If True, return (seg1, seg2, labels) (PREPROCESSED mode only)
        filter_missing: Filter samples with missing labels (PREPROCESSED mode only)
        use_raw_vitaldb: DEPRECATED - use mode='raw' instead (kept for backward compatibility)

    Returns:
        RAW mode / return_pairs=True: (seg1, seg2) each [C, T]
        RAW mode / return_pairs=False: seg [C, T]
        PREPROCESSED mode / return_labels=True: (seg1, seg2, labels)
        PREPROCESSED mode / return_labels=False: (seg1, seg2)

        Where T = window_sec * fs = 1250 for default params
        C = len(channels) (1 for single modality, 2 for PPG+ECG)

    Examples:
        # RAW mode (existing behavior - on-the-fly loading from VitalDB API)
        dataset = VitalDBDataset(
            cache_dir='data/vitaldb_cache',
            channels=['ppg', 'ecg'],
            split='train',
            mode='raw'  # Default
        )

        # PREPROCESSED mode (fast - load from pre-processed windows)
        dataset = VitalDBDataset(
            data_dir='data/processed/vitaldb/windows_with_labels',
            channels=['ppg', 'ecg'],
            split='train',
            mode='preprocessed',
            task='mortality',
            return_labels=True
        )
    """
    
    SUPPORTED_MODALITIES = ['ppg', 'ecg']
    TRACK_MAPPING = {
        'ppg': 'PLETH',
        'ecg': 'ECG_II'
    }
    
    def __init__(
        self,
        data_dir: str = None,
        split: str = 'train',
        channels: Union[str, List[str]] = 'ppg',
        window_sec: float = 10.0,
        fs: int = 125,
        return_pairs: bool = True,
        apply_modality_dropout: bool = True,
        modality_dropout_prob: float = 0.25,
        transform: Optional[callable] = None,
        cache_dir: str = None,
        mode: str = 'raw',  # NEW: 'raw' or 'preprocessed'
        task: Optional[str] = None,  # NEW: for task-specific label loading
        return_labels: bool = False,  # NEW: return labels in preprocessed mode
        filter_missing: bool = True,  # NEW: filter samples with missing labels
        use_raw_vitaldb: bool = False,  # DEPRECATED: kept for backward compatibility
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        max_cases: int = None,
        segments_per_case: int = 20
    ):
        super().__init__()

        # Backward compatibility: map use_raw_vitaldb to mode
        if use_raw_vitaldb:
            mode = 'raw'

        # Parse channels/modalities
        self.channels = self._parse_channels(channels)
        self.n_channels = len(self.channels)
        self.is_multimodal = self.n_channels > 1

        self.split = split
        self.window_sec = window_sec
        self.fs = fs
        self.return_pairs = return_pairs
        self.apply_modality_dropout = apply_modality_dropout and (split == 'train')
        self.modality_dropout_prob = modality_dropout_prob
        self.transform = transform
        self.mode = mode
        self.task = task
        self.return_labels = return_labels
        self.filter_missing = filter_missing
        self.segments_per_case = segments_per_case

        # Expected number of samples
        self.T = int(window_sec * fs)
        self.C = self.n_channels

        # Initialize based on mode
        if self.mode == 'preprocessed':
            self._init_preprocessed_mode(data_dir)
        else:  # raw mode (default)
            self._init_raw_mode(cache_dir, train_ratio, val_ratio, max_cases)

        # Print summary
        self._print_summary()
        
    def _parse_channels(self, channels: Union[str, List[str]]) -> List[str]:
        """Parse channel specification into list of modalities."""
        if isinstance(channels, str):
            if channels.lower() == 'all':
                return self.SUPPORTED_MODALITIES.copy()
            elif ',' in channels:
                return [ch.strip().lower() for ch in channels.split(',')]
            else:
                return [channels.lower()]
        elif isinstance(channels, list):
            return [ch.lower() for ch in channels]
        else:
            # Backward compatibility - treat as list if it looks like one
            if hasattr(channels, '__iter__'):
                return list(channels)
            return ['ppg']  # Default
            
    def _init_raw_mode(self, cache_dir: str, train_ratio: float, val_ratio: float, max_cases: int):
        """Initialize dataset in RAW mode (load from VitalDB API)."""
        try:
            import vitaldb
        except ImportError:
            raise ImportError("Please install vitaldb: pip install vitaldb")
            
        # Setup cache directory
        self.cache_dir = Path(cache_dir) if cache_dir else Path('data/vitaldb_cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Find cases that have ALL required tracks
        print(f"Finding VitalDB cases with tracks: {[self.TRACK_MAPPING[ch] for ch in self.channels]}")
        
        # Start with cases for first modality
        first_track = self.TRACK_MAPPING[self.channels[0]]
        cases = set(vitaldb.find_cases(first_track))
        
        # Intersect with other modalities if multi-modal
        if self.is_multimodal:
            for channel in self.channels[1:]:
                track = self.TRACK_MAPPING[channel]
                track_cases = set(vitaldb.find_cases(track))
                cases = cases.intersection(track_cases)
                
        cases = list(cases)
        
        # Limit cases if specified
        if max_cases:
            cases = cases[:max_cases]
            
        print(f"Found {len(cases)} cases with all required tracks")
        
        # Split cases
        n = len(cases)
        train_end = int(train_ratio * n)
        val_end = int((train_ratio + val_ratio) * n)
        
        if self.split == 'train':
            self.cases = cases[:train_end]
        elif self.split == 'val':
            self.cases = cases[train_end:val_end]
        else:
            self.cases = cases[val_end:]
            
        # Create segment pairs (same patient, different time)
        self.segment_pairs = []
        for case_id in self.cases:
            for _ in range(self.segments_per_case):
                self.segment_pairs.append({
                    'case_id': case_id,
                    'pair_type': 'same_patient'
                })
                
        self.vitaldb = vitaldb
        
    def _init_preprocessed_mode(self, data_dir: str):
        """Initialize dataset in PREPROCESSED mode (load from window NPZ files)."""
        if data_dir is None:
            raise ValueError("data_dir must be specified for mode='preprocessed'")

        self.data_dir = Path(data_dir)

        # Find all window files in split directory
        split_dir = self.data_dir / self.split
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")

        # Look for window_*.npz files
        self.window_files = sorted(list(split_dir.glob("window_*.npz")))

        if len(self.window_files) == 0:
            # Fallback to any *.npz files for backward compatibility
            self.window_files = sorted(list(split_dir.glob("*.npz")))

        if len(self.window_files) == 0:
            raise ValueError(f"No window files found in {split_dir}")

        # Filter by task if specified
        if self.task and self.filter_missing:
            self.valid_indices = self._find_valid_samples_preprocessed()
        else:
            self.valid_indices = list(range(len(self.window_files)))

        # For preprocessed windows, we don't have cases/pairs structure
        self.cases = None
        self.segment_pairs = None
        
    def __len__(self):
        if self.mode == 'preprocessed':
            return len(self.valid_indices)
        elif self.segment_pairs:
            return len(self.segment_pairs)
        else:
            return 0

    def __getitem__(self, idx):
        """Get sample (mode-dependent behavior)."""
        if self.mode == 'preprocessed':
            return self._getitem_preprocessed(idx)
        else:
            return self._getitem_raw(idx)
            
    def _getitem_raw(self, idx):
        """Get sample in RAW mode (load from VitalDB API)."""
        # Import vitaldb_loader functions
        from .vitaldb_loader import load_channel
        
        # Get case for this pair
        pair_info = self.segment_pairs[idx % len(self.segment_pairs)]
        case_id = pair_info['case_id']
        
        # Load and preprocess signals for all channels
        signals = {}
        
        for channel in self.channels:
            try:
                # Use the robust load_channel function from vitaldb_loader
                track_name = self.TRACK_MAPPING[channel]
                signal, fs_orig = load_channel(
                    case_id=str(case_id),
                    channel=track_name,
                    use_cache=True,
                    cache_dir=str(self.cache_dir),
                    auto_fix_alternating=True
                )
                
                # Remove NaNs if present
                if signal is not None and len(signal) > 0:
                    valid_mask = ~np.isnan(signal)
                    if np.any(valid_mask):
                        signal = signal[valid_mask]
                    
                    # Preprocess (filter, resample, normalize)
                    signal = self._preprocess_signal(signal, channel)
                else:
                    signal = None
                    
            except Exception as e:
                print(f"   Warning: Failed to load {channel} for case {case_id}: {e}")
                signal = None
                
            signals[channel] = signal
            
        # Create segments from signals
        seg1, seg2 = self._create_paired_segments(signals)
        
        # Apply modality dropout if enabled
        if self.apply_modality_dropout and self.is_multimodal:
            seg1 = self._apply_modality_dropout(seg1)
            seg2 = self._apply_modality_dropout(seg2)
            
        # Apply transforms if specified
        if self.transform:
            seg1 = self.transform(seg1)
            seg2 = self.transform(seg2)
            
        if self.return_pairs:
            return seg1, seg2
        else:
            return seg1
            
    def _getitem_preprocessed(self, idx):
        """Get sample in PREPROCESSED mode (load from window NPZ file)."""
        # Get file index from valid indices
        file_idx = self.valid_indices[idx]
        window_file = self.window_files[file_idx]

        # Load window data
        data = np.load(window_file)

        # Extract signal based on format
        if 'signal' in data:
            # New format: signal = [C, T] with C=2 (PPG, ECG)
            signal = data['signal']  # [2, T]

            # Map channels: [PPG, ECG]
            channel_map = {'ppg': 0, 'ecg': 1}
            selected_channels = []
            for ch in self.channels:
                if ch in channel_map:
                    selected_channels.append(channel_map[ch])

            signal = signal[selected_channels, :]
        else:
            # Old format: separate PPG and ECG keys
            multi_channel_signal = []
            for channel in self.channels:
                ch_upper = channel.upper()
                if ch_upper in data:
                    ch_signal = data[ch_upper]
                    if ch_signal.ndim == 1:
                        ch_signal = ch_signal[np.newaxis, :]
                    multi_channel_signal.append(ch_signal)
                else:
                    # Add zeros if channel not found
                    multi_channel_signal.append(np.zeros((1, self.T)))
            signal = np.vstack(multi_channel_signal).astype(np.float32)

        # Convert to tensor
        signal = torch.from_numpy(signal).float()

        if self.return_pairs:
            # Create two different augmented views
            seg1 = signal.clone()
            seg2 = signal.clone()

            # Apply modality dropout
            if self.apply_modality_dropout and self.is_multimodal:
                seg1 = self._apply_modality_dropout(seg1)
                seg2 = self._apply_modality_dropout(seg2)

            # Apply transforms
            if self.transform:
                seg1 = self.transform(seg1)
                seg2 = self.transform(seg2)

            # Return with labels if requested
            if self.return_labels:
                labels = self._extract_labels_from_npz(data)
                return seg1, seg2, labels

            return seg1, seg2
        else:
            if self.return_labels:
                labels = self._extract_labels_from_npz(data)
                return signal, labels
            return signal
            
    def _preprocess_signal(self, signal: np.ndarray, channel: str) -> Optional[np.ndarray]:
        """Preprocess raw signal."""
        if signal is None or len(signal) == 0:
            return None
            
        try:
            # Get channel-specific parameters
            if channel == 'ppg':
                band_low, band_high = 0.5, 8.0
                original_fs = 100  # VitalDB PPG is usually 100Hz
            else:  # ecg
                band_low, band_high = 0.5, 40.0
                original_fs = 250  # VitalDB ECG can be 250Hz or 500Hz
                
            # Bandpass filter
            if len(signal) >= 100:
                nyquist = original_fs / 2
                if band_high < nyquist * 0.95:
                    sos = scipy_signal.butter(
                        4, [band_low, band_high],
                        btype='band', fs=original_fs, output='sos'
                    )
                    signal = scipy_signal.sosfiltfilt(sos, signal)
                    
            # Resample to target fs
            if original_fs != self.fs:
                n_samples = int(len(signal) * self.fs / original_fs)
                signal = scipy_signal.resample(signal, n_samples)
                
            # Z-score normalization
            mean = np.mean(signal)
            std = np.std(signal)
            if std > 1e-8:
                signal = (signal - mean) / std
                
            return signal.astype(np.float32)
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None
            
    def _create_paired_segments(self, signals: Dict[str, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create paired segments from signals."""
        # Find minimum valid length
        valid_lengths = []
        for signal in signals.values():
            if signal is not None and len(signal) >= self.T:
                valid_lengths.append(len(signal))
                
        if not valid_lengths:
            # Return random noise if no valid signals (to avoid always returning zeros)
            noise1 = torch.randn(self.C, self.T, dtype=torch.float32) * 0.01
            noise2 = torch.randn(self.C, self.T, dtype=torch.float32) * 0.01
            return noise1, noise2
            
        min_len = min(valid_lengths)
        
        # If signal is too short for two non-overlapping segments
        if min_len < 2 * self.T:
            # Use overlapping segments
            max_start = max(0, min_len - self.T)
            if max_start == 0:
                start1 = 0
                start2 = 0
            else:
                start1 = np.random.randint(0, max_start + 1)
                start2 = np.random.randint(0, max_start + 1)
        else:
            # Use non-overlapping segments
            max_start = min_len - self.T
            start1 = np.random.randint(0, max(1, max_start // 2))
            start2 = np.random.randint(max_start // 2, max_start + 1)
            
        # Extract segments
        seg1_list = []
        seg2_list = []
        
        for channel in self.channels:
            if channel in signals and signals[channel] is not None:
                signal = signals[channel]
                seg1_list.append(signal[start1:start1 + self.T])
                seg2_list.append(signal[start2:start2 + self.T])
            else:
                seg1_list.append(np.zeros(self.T))
                seg2_list.append(np.zeros(self.T))
                
        # Stack and convert to tensors
        seg1 = np.vstack([s[np.newaxis, :] for s in seg1_list])
        seg2 = np.vstack([s[np.newaxis, :] for s in seg2_list])
        
        seg1 = torch.from_numpy(seg1).float()
        seg2 = torch.from_numpy(seg2).float()
        
        return seg1, seg2
        
    def _apply_modality_dropout(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply modality dropout - randomly zero entire channels."""
        if not self.is_multimodal or self.split != 'train':
            return signal

        # Randomly dropout each channel
        for c in range(self.C):
            if np.random.random() < self.modality_dropout_prob:
                signal[c, :] = 0

        # Ensure at least one channel remains
        if torch.all(signal == 0):
            # Restore random channel
            restore_idx = np.random.randint(0, self.C)
            signal[restore_idx, :] = torch.randn(self.T) * 0.1

        return signal

    def _find_valid_samples_preprocessed(self) -> List[int]:
        """Find samples with valid labels for the specified task (PREPROCESSED mode)."""
        # Task label mapping for VitalDB
        TASK_LABEL_MAP = {
            'mortality': ['death_inhosp'],
            'icu_days': ['icu_days'],
            'asa': ['asa'],
            'emergency': ['emergency'],
            'age': ['age'],
            'bmi': ['bmi'],
        }

        if self.task not in TASK_LABEL_MAP:
            warnings.warn(f"Unknown task '{self.task}', including all samples")
            return list(range(len(self.window_files)))

        label_keys = TASK_LABEL_MAP[self.task]
        valid_indices = []

        for idx, window_file in enumerate(self.window_files):
            try:
                data = np.load(window_file)

                # Check if all required labels are valid (not missing)
                all_valid = True
                for key in label_keys:
                    if key in data:
                        value = data[key]
                        # Check for missing values (NaN, -1, or np.nan)
                        if isinstance(value, (np.ndarray, np.generic)):
                            if np.isnan(value).any() or (value == -1).any():
                                all_valid = False
                                break
                        elif np.isnan(value) or value == -1:
                            all_valid = False
                            break
                    else:
                        all_valid = False
                        break

                if all_valid:
                    valid_indices.append(idx)

            except Exception as e:
                warnings.warn(f"Error loading {window_file}: {e}")
                continue

        return valid_indices

    def _extract_labels_from_npz(self, data: np.lib.npyio.NpzFile) -> Dict:
        """Extract all available labels from NPZ file."""
        labels = {}

        # Case-level labels (7)
        label_keys = ['age', 'sex', 'bmi', 'asa', 'emergency', 'death_inhosp', 'icu_days']

        for key in label_keys:
            if key in data:
                value = data[key]
                # Convert numpy types to Python types
                if isinstance(value, (np.ndarray, np.generic)):
                    value = value.item() if value.ndim == 0 else value.tolist()
                labels[key] = value
            else:
                # Set missing values
                labels[key] = -1 if key in ['sex', 'asa'] else np.nan

        # Quality metrics
        if 'ppg_quality' in data:
            labels['ppg_quality'] = float(data['ppg_quality'])
        if 'ecg_quality' in data:
            labels['ecg_quality'] = float(data['ecg_quality'])

        return labels

    def _print_summary(self):
        """Print dataset summary."""
        print(f"\nVitalDB Dataset initialized:")
        print(f"  Mode: {self.mode.upper()}")
        print(f"  Split: {self.split}")
        print(f"  Modalities: {self.channels} ({self.n_channels} channels)")
        print(f"  Samples: {len(self)}")
        print(f"  Window: {self.window_sec}s @ {self.fs}Hz = {self.T} samples")
        print(f"  Output shape: [{self.C}, {self.T}]")

        if self.mode == 'preprocessed':
            print(f"  Data dir: {self.data_dir}")
            print(f"  Task filtering: {self.task if self.task else 'None'}")
            print(f"  Return labels: {self.return_labels}")
            if self.task:
                total_files = len(self.window_files)
                valid_files = len(self.valid_indices)
                print(f"  Valid samples: {valid_files}/{total_files} ({100*valid_files/total_files:.1f}%)")
        else:
            print(f"  Cache dir: {self.cache_dir if hasattr(self, 'cache_dir') else 'None'}")
            if hasattr(self, 'cases'):
                print(f"  Cases: {len(self.cases) if self.cases else 0}")


def create_vitaldb_dataloaders(
    channels: Union[str, List[str]] = 'ppg',
    batch_size: int = 32,
    num_workers: int = 4,
    cache_dir: str = None,
    use_raw_vitaldb: bool = True,
    max_cases: int = None,
    pin_memory: bool = True,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders for VitalDB.
    
    Args:
        channels: Modalities to load ('ppg', 'ecg', ['ppg', 'ecg'], 'all')
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for data loading
        cache_dir: Directory for caching VitalDB data
        use_raw_vitaldb: If True, load from VitalDB API; if False, use preprocessed
        max_cases: Maximum number of cases to use (None for all)
        pin_memory: Pin memory for GPU transfer
        **dataset_kwargs: Additional arguments for VitalDBDataset
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = VitalDBDataset(
        cache_dir=cache_dir,
        channels=channels,
        split='train',
        use_raw_vitaldb=use_raw_vitaldb,
        max_cases=max_cases,
        **dataset_kwargs
    )
    
    val_dataset = VitalDBDataset(
        cache_dir=cache_dir,
        channels=channels,
        split='val',
        use_raw_vitaldb=use_raw_vitaldb,
        max_cases=max_cases,
        **dataset_kwargs
    )
    
    test_dataset = VitalDBDataset(
        cache_dir=cache_dir,
        channels=channels,
        split='test',
        use_raw_vitaldb=use_raw_vitaldb,
        max_cases=max_cases,
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
