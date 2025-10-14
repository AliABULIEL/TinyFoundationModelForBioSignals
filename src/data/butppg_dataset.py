"""BUT PPG Dataset with unified preprocessing matching VitalDB.

This dataset ensures BUT PPG uses IDENTICAL preprocessing as VitalDB
for proper transfer learning.
"""

import hashlib
import json
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import signal as scipy_signal

# Import unified preprocessing components
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    # Try to import from biosignal module if available
    from biosignal.data import WaveformQualityControl, SPEC_VERSION
    BIOSIGNAL_AVAILABLE = True
except ImportError:
    BIOSIGNAL_AVAILABLE = False
    SPEC_VERSION = "v1.0-2025-09"
    warnings.warn("biosignal module not available, using local WaveformQualityControl")

from .butppg_loader import BUTPPGLoader


# If biosignal not available, create local QC
if not BIOSIGNAL_AVAILABLE:
    class WaveformQualityControl:
        """Local copy of quality control for BUT PPG."""
        
        @staticmethod
        def check_flatline(signal: np.ndarray, variance_threshold: float = 0.01, 
                           consecutive_threshold: int = 50) -> bool:
            if len(signal) == 0:
                return True
            if np.var(signal) < variance_threshold:
                return True
            if len(signal) > consecutive_threshold:
                diff = np.diff(signal)
                change_indices = np.where(np.abs(diff) > 1e-10)[0]
                if len(change_indices) == 0:
                    return True
                elif len(change_indices) == 1:
                    return max(change_indices[0], len(signal) - change_indices[0] - 1) > consecutive_threshold
                else:
                    gaps = np.diff(change_indices)
                    if len(gaps) > 0 and np.max(gaps) > consecutive_threshold:
                        return True
            return False
        
        @staticmethod
        def check_spikes(signal: np.ndarray, z_threshold: float = 3.0, 
                         consecutive_samples: int = 5) -> np.ndarray:
            if len(signal) < consecutive_samples:
                return np.zeros(len(signal), dtype=bool)
            mean = np.mean(signal)
            std = np.std(signal)
            if std < 1e-8:
                return np.zeros(len(signal), dtype=bool)
            z_scores = np.abs((signal - mean) / std)
            large_spikes = z_scores > 6.0
            moderate_mask = z_scores > z_threshold
            consecutive_spikes = np.convolve(moderate_mask.astype(float), 
                                            np.ones(consecutive_samples), 'same') >= consecutive_samples
            return large_spikes | consecutive_spikes
        
        @staticmethod
        def check_physiologic_bounds(signal: np.ndarray, signal_type: str) -> np.ndarray:
            bounds = {
                'hr': (30, 200),
                'ppg': (-5, 5),
                'ecg': (-5, 5),
            }
            if signal_type not in bounds:
                return np.ones(len(signal), dtype=bool)
            min_val, max_val = bounds[signal_type]
            extreme_threshold = 1e-5
            is_extreme = (np.abs(signal) < extreme_threshold) | (np.abs(signal) > 1e5)
            return (signal >= min_val) & (signal <= max_val) & ~is_extreme
        
        @staticmethod
        def calculate_ppg_sqi(signal: np.ndarray) -> float:
            if len(signal) < 10:
                return 0.0
            if np.var(signal) < 1e-10:
                return 0.0
            from scipy import stats
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                skewness = stats.skew(signal)
            if np.isnan(skewness):
                return 0.0
            return skewness
        
        @staticmethod
        def get_quality_mask(signal: np.ndarray, signal_type: str = 'ppg',
                            min_valid_ratio: float = 0.7) -> Dict:
            qc_results = {
                'is_flatline': False,
                'has_spikes': False,
                'in_bounds': True,
                'sqi': 0.0,
                'valid_ratio': 0.0,
                'overall_valid': False,
                'mask': np.ones(len(signal), dtype=bool)
            }
            
            if len(signal) == 0:
                return qc_results
            
            qc_results['is_flatline'] = WaveformQualityControl.check_flatline(signal)
            spike_mask = WaveformQualityControl.check_spikes(signal)
            qc_results['has_spikes'] = np.any(spike_mask)
            bounds_mask = WaveformQualityControl.check_physiologic_bounds(signal, signal_type)
            qc_results['in_bounds'] = np.all(bounds_mask)
            
            if signal_type == 'ppg':
                qc_results['sqi'] = WaveformQualityControl.calculate_ppg_sqi(signal)
            
            valid_mask = bounds_mask & ~spike_mask
            qc_results['mask'] = valid_mask
            qc_results['valid_ratio'] = np.mean(valid_mask)
            
            if signal_type == 'ppg':
                qc_results['overall_valid'] = (
                    not qc_results['is_flatline'] and
                    qc_results['valid_ratio'] >= (min_valid_ratio * 0.8)
                )
            else:
                qc_results['overall_valid'] = (
                    not qc_results['is_flatline'] and
                    qc_results['valid_ratio'] >= min_valid_ratio
                )
            
            return qc_results


class BUTPPGDataset(Dataset):
    """
    BUT PPG dataset with UNIFIED preprocessing matching VitalDB.
    
    CRITICAL: Uses SAME preprocessing, filtering, QC, and windowing as VitalDB
    to ensure proper transfer learning.
    """
    
    # UNIFIED preprocessing configuration - MUST match VitalDB
    UNIFIED_CONFIG = {
        'ppg': {
            'filter_type': 'cheby2',
            'filter_order': 4,
            'filter_band': [0.5, 8.0],  # Article-aligned: 0.5-8.0 Hz
            'ripple': 20,  # Standard 20dB stopband ripple
            'target_fs': 125,  # SAME as VitalDB (was incorrectly 25)
            'window_sec': 10.0,
            'hop_sec': 5.0,
            'normalization': 'z-score',
            'min_valid_ratio': 0.7
        }
    }
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        modality: str = 'ppg',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        random_seed: int = 42,
        window_sec: float = 10.0,
        hop_sec: float = 5.0,
        enable_qc: bool = True,
        min_valid_ratio: float = 0.7,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        segments_per_subject: int = 20,
        return_labels: bool = False,
        return_participant_id: bool = False,
        **kwargs
    ):
        """Initialize BUT PPG dataset with unified preprocessing.
        
        Args:
            data_dir: Path to BUT PPG database
            split: 'train', 'val', or 'test'
            modality: Signal type (currently only 'ppg' supported)
            train_ratio: Training split ratio
            val_ratio: Validation split ratio
            random_seed: Random seed for reproducible splits
            window_sec: Window size in seconds (MUST be 10.0 for compatibility)
            hop_sec: Hop size in seconds (MUST be 5.0 for compatibility)
            enable_qc: Enable quality control
            min_valid_ratio: Minimum valid ratio for QC
            cache_dir: Cache directory
            use_cache: Whether to use caching
            segments_per_subject: Number of segments per subject
            return_labels: Whether to return labels
            return_participant_id: Whether to return participant ID
        """
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.split = split
        self.modality = modality.lower()
        self.random_seed = random_seed
        self.enable_qc = enable_qc
        self.min_valid_ratio = min_valid_ratio
        self.segments_per_subject = segments_per_subject
        self.return_labels = return_labels
        self.return_participant_id = return_participant_id
        
        # CRITICAL: Enforce unified configuration
        if window_sec != 10.0:
            warnings.warn(f"window_sec={window_sec} differs from VitalDB standard (10.0). Setting to 10.0 for compatibility.")
            window_sec = 10.0
        
        if hop_sec != 5.0:
            warnings.warn(f"hop_sec={hop_sec} differs from VitalDB standard (5.0). Setting to 5.0 for compatibility.")
            hop_sec = 5.0
        
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        
        # Get unified preprocessing config
        self.preprocessing_config = self.UNIFIED_CONFIG[self.modality]
        self.target_fs = self.preprocessing_config['target_fs']
        self.segment_length = int(self.window_sec * self.target_fs)
        
        # Setup cache
        if cache_dir is None:
            cache_dir = self.data_dir / 'cache'
        self.cache_dir = Path(cache_dir) / f"butppg_cache_{SPEC_VERSION}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache
        
        # Write cache metadata
        self._write_cache_metadata()
        
        # Initialize QC
        self.qc = WaveformQualityControl()
        
        # Initialize loader WITHOUT windowing (dataset handles windowing)
        self.loader = BUTPPGLoader(
            self.data_dir,
            fs=self.target_fs,
            window_duration=self.window_sec,
            window_stride=self.hop_sec,
            apply_windowing=False  # CRITICAL: Dataset handles windowing, not loader
        )
        
        # Get all subjects
        all_subjects = self.loader.get_subject_list()
        
        if len(all_subjects) == 0:
            raise ValueError(f"No subjects found in {self.data_dir}")
        
        # Create reproducible splits
        np.random.seed(self.random_seed)
        shuffled_subjects = np.random.permutation(all_subjects)
        
        n = len(shuffled_subjects)
        train_end = int(train_ratio * n)
        val_end = int((train_ratio + val_ratio) * n)
        
        if split == 'train':
            self.subjects = shuffled_subjects[:train_end].tolist()
        elif split == 'val':
            self.subjects = shuffled_subjects[train_end:val_end].tolist()
        else:  # test
            self.subjects = shuffled_subjects[val_end:].tolist()
        
        # Build segment pairs (for SSL compatibility)
        self.segment_pairs = self._build_segment_pairs()
        
        print(f"\nBUT PPG Dataset initialized (SPEC {SPEC_VERSION}):")
        print(f"  Split: {split}")
        print(f"  Modality: {modality}")
        print(f"  Subjects: {len(self.subjects)}")
        print(f"  Segments: {len(self.segment_pairs)}")
        print(f"  Filter: {self.preprocessing_config['filter_type']} "
              f"{self.preprocessing_config['filter_order']}th order")
        print(f"  Sampling: ANY → {self.target_fs}Hz (unified)")
        print(f"  Window: {self.window_sec}s, Hop: {self.hop_sec}s")
        print(f"  QC enabled: {self.enable_qc}")
        print(f"  ⚠️  UNIFIED PREPROCESSING: Matches VitalDB exactly")
    
    def _write_cache_metadata(self):
        """Write cache metadata for version tracking."""
        metadata = {
            'spec_version': SPEC_VERSION,
            'modality': self.modality,
            'preprocessing_config': self.preprocessing_config,
            'window_sec': self.window_sec,
            'hop_sec': self.hop_sec,
            'qc_enabled': self.enable_qc,
            'min_valid_ratio': self.min_valid_ratio,
            'note': 'UNIFIED preprocessing matching VitalDB for transfer learning'
        }
        
        metadata_file = self.cache_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _apply_unified_filter(self, signal: np.ndarray, original_fs: float) -> np.ndarray:
        """Apply UNIFIED filter - MUST match VitalDB exactly."""
        config = self.preprocessing_config
        
        if config['filter_type'] == 'cheby2':
            # Chebyshev Type-II (same as VitalDB PPG)
            sos = scipy_signal.cheby2(
                config['filter_order'],
                config['ripple'],
                config['filter_band'],
                btype='band',
                fs=original_fs,
                output='sos'
            )
            filtered = scipy_signal.sosfiltfilt(sos, signal)
        else:
            # Fallback to butterworth
            sos = scipy_signal.butter(
                config['filter_order'],
                config['filter_band'],
                btype='band',
                fs=original_fs,
                output='sos'
            )
            filtered = scipy_signal.sosfiltfilt(sos, signal)
        
        return filtered
    
    def _preprocess_with_qc(
        self,
        signal: np.ndarray,
        original_fs: float,
        subject_id: str
    ) -> Tuple[Optional[np.ndarray], Dict]:
        """Preprocess signal with UNIFIED pipeline matching VitalDB."""
        
        # Remove NaN
        signal = signal[~np.isnan(signal)]
        
        if len(signal) < original_fs * 2:  # Less than 2 seconds
            return None, {'overall_valid': False}
        
        # Apply UNIFIED filter
        filtered = self._apply_unified_filter(signal, original_fs)
        
        # Resample to UNIFIED target rate (25 Hz)
        if original_fs != self.target_fs:
            n_samples = int(len(filtered) * self.target_fs / original_fs)
            resampled = scipy_signal.resample(filtered, n_samples)
        else:
            resampled = filtered
        
        # Z-score normalization (UNIFIED)
        mean = np.mean(resampled)
        std = np.std(resampled)
        if std > 1e-8:
            normalized = (resampled - mean) / std
        else:
            normalized = resampled
        
        # Quality control (UNIFIED)
        if self.enable_qc:
            qc_results = self.qc.get_quality_mask(
                normalized,
                self.modality,
                self.min_valid_ratio
            )
        else:
            qc_results = {
                'overall_valid': True,
                'mask': np.ones(len(normalized), dtype=bool)
            }
        
        return normalized, qc_results
    
    def _extract_window(
        self,
        subject_id: str,
        window_idx: int
    ) -> Dict:
        """Extract a window with QC."""
        window_data = {
            'signal': None,
            'qc': None,
            'valid': False,
            'subject_id': subject_id
        }
        
        try:
            # Load signal WITHOUT windowing (we do windowing here)
            result = self.loader.load_subject(
                subject_id,
                self.modality,
                return_windows=False,  # CRITICAL: Get raw signal, not windows
                normalize=False,  # We handle normalization
                compute_quality=False  # Skip quality for now
            )
            if result is None:
                return window_data
            
            signal, metadata = result
            original_fs = metadata.get('fs', 100)
            
            # Ensure signal is 2D [T, C]
            if signal.ndim == 1:
                signal = signal.reshape(-1, 1)
            elif signal.ndim > 2:
                warnings.warn(f"Signal has {signal.ndim} dimensions, reshaping to 2D")
                signal = signal.reshape(signal.shape[0], -1)
            
            # For now, take only first channel if multi-channel
            if signal.shape[1] > 1:
                signal = signal[:, 0]
            
            # Flatten for preprocessing (expects 1D)
            signal = signal.flatten()
            
            # Preprocess with QC
            processed, qc_results = self._preprocess_with_qc(
                signal, original_fs, subject_id
            )
            
            if processed is None:
                return window_data
            
            # Extract window
            window_start = int(window_idx * self.hop_sec * self.target_fs)
            window_end = window_start + self.segment_length
            
            if window_end > len(processed):
                return window_data
            
            window_signal = processed[window_start:window_end]
            
            # Window-level QC
            window_qc = self.qc.get_quality_mask(
                window_signal,
                self.modality,
                self.min_valid_ratio
            )
            
            window_data = {
                'signal': window_signal,
                'qc': window_qc,
                'valid': window_qc['overall_valid'],
                'subject_id': subject_id
            }
            
        except Exception as e:
            warnings.warn(f"Error processing subject {subject_id}, window {window_idx}: {e}")
        
        return window_data
    
    def _build_segment_pairs(self) -> List[Dict]:
        """Build segment pairs (for SSL compatibility with VitalDB)."""
        pairs = []
        for subject_id in self.subjects:
            for _ in range(self.segments_per_subject):
                pairs.append({
                    'subject_id': subject_id,
                    'pair_type': 'same_subject'  # Same as VitalDB
                })
        return pairs
    
    def __len__(self) -> int:
        return len(self.segment_pairs)
    
    def __getitem__(self, idx: int) -> Tuple:
        """Get item with same interface as VitalDB."""
        # Get subject for this index
        subject_idx = idx % len(self.subjects)
        subject_id = self.subjects[subject_idx]
        
        # Extract two windows (for SSL compatibility)
        window1_data = self._extract_window(subject_id, 0)
        window2_data = self._extract_window(subject_id, 1)
        
        # Create tensors
        if window1_data['valid'] and window1_data['signal'] is not None:
            seg1 = torch.from_numpy(window1_data['signal']).float().unsqueeze(0)
        else:
            seg1 = torch.zeros(1, self.segment_length, dtype=torch.float32)
        
        if window2_data['valid'] and window2_data['signal'] is not None:
            seg2 = torch.from_numpy(window2_data['signal']).float().unsqueeze(0)
        else:
            seg2 = torch.zeros(1, self.segment_length, dtype=torch.float32)
        
        # Prepare context (for compatibility)
        context = {
            'subject_id': subject_id,
            'qc': {
                'seg1': window1_data['qc'],
                'seg2': window2_data['qc']
            },
            'dataset': 'butppg'  # Mark as BUT PPG
        }
        
        # Return based on flags (same as VitalDB)
        if self.return_labels and self.return_participant_id:
            return seg1, seg2, subject_id, context
        elif self.return_labels:
            return seg1, seg2, context
        elif self.return_participant_id:
            return seg1, seg2, subject_id
        else:
            return seg1, seg2
    
    def validate_compatibility_with_vitaldb(self, vitaldb_dataset) -> Dict:
        """Validate that preprocessing matches VitalDB exactly.
        
        Args:
            vitaldb_dataset: VitalDBDataset instance to compare against
            
        Returns:
            Dict with compatibility check results
        """
        checks = {}
        
        # Check filter type
        checks['filter_type'] = (
            self.preprocessing_config['filter_type'] ==
            vitaldb_dataset.filter_params.get('type', 'unknown')
        )
        
        # Check filter order
        checks['filter_order'] = (
            self.preprocessing_config['filter_order'] ==
            vitaldb_dataset.filter_params.get('order', -1)
        )
        
        # Check filter band
        checks['filter_band'] = (
            self.preprocessing_config['filter_band'] ==
            vitaldb_dataset.filter_params.get('band', [])
        )
        
        # Check target sampling rate
        checks['target_fs'] = (
            self.target_fs == vitaldb_dataset.target_fs
        )
        
        # Check window size
        checks['window_size'] = (
            self.segment_length == vitaldb_dataset.segment_length
        )
        
        # Check QC enabled
        checks['qc_enabled'] = (
            self.enable_qc == vitaldb_dataset.enable_qc
        )
        
        # Overall compatibility
        checks['overall_compatible'] = all(checks.values())
        
        return checks


def create_butppg_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    **dataset_kwargs
):
    """Create BUT PPG dataloaders with unified preprocessing.
    
    Returns train, val, test loaders compatible with VitalDB loaders.
    """
    train_dataset = BUTPPGDataset(
        data_dir=data_dir,
        split='train',
        **dataset_kwargs
    )
    
    val_dataset = BUTPPGDataset(
        data_dir=data_dir,
        split='val',
        **dataset_kwargs
    )
    
    test_dataset = BUTPPGDataset(
        data_dir=data_dir,
        split='test',
        **dataset_kwargs
    )
    
    from torch.utils.data import DataLoader
    
    # Custom collate for context
    def collate_fn_with_context(batch):
        """Collate that handles QC and context (same as VitalDB)."""
        valid_batch = [item for item in batch if item[0].numel() > 0]
        
        if len(valid_batch) == 0:
            dummy = torch.zeros(1, 1, train_dataset.segment_length)
            return dummy, dummy
        
        # Check format
        if len(valid_batch[0]) >= 3:  # With context
            segs1 = torch.stack([item[0] for item in valid_batch])
            segs2 = torch.stack([item[1] for item in valid_batch])
            
            contexts = []
            for item in valid_batch:
                if len(item) > 2:
                    contexts.append(item[2])
            
            return segs1, segs2, contexts
        else:
            return torch.utils.data.dataloader.default_collate(valid_batch)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_with_context,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_with_context
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_with_context
    )
    
    return train_loader, val_loader, test_loader
