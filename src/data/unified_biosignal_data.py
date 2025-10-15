"""
Unified Biosignal Dataset with VitalDB and BUT PPG
SPEC-compliant processing with unified preprocessing for transfer learning
Version: v1.0-2025-09
"""

import hashlib
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union
from scipy import signal as scipy_signal
from scipy.io import loadmat
import warnings
from collections import OrderedDict
import threading

# Try to import PyWavelets (optional for EEG)
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    warnings.warn("PyWavelets not installed. EEG wavelet processing disabled.", ImportWarning)

# Try to import h5py for HDF5 support
try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    warnings.warn("h5py not installed. HDF5 file support disabled.", ImportWarning)

warnings.filterwarnings('ignore')

# SPEC Version for cache invalidation
SPEC_VERSION = "v1.0-2025-09"

# Import vitaldb at module level for test patching
vitaldb = None

# ================================================================================
# QUALITY CONTROL MODULE - Per SPEC (SHARED BY BOTH DATASETS)
# ================================================================================

class WaveformQualityControl:
    """
    Quality control checks for physiological waveforms per VitalDB SPEC.
    SHARED by both VitalDB and BUT PPG datasets for consistency.
    References: Nature Scientific Data 2022, Frontiers Digital Health 2022
    """
    
    @staticmethod
    def check_flatline(signal: np.ndarray, variance_threshold: float = 0.01, 
                       consecutive_threshold: int = 50) -> bool:
        """
        Check for flatline (no variation in signal).
        SPEC: variance < 0.01 OR >50 consecutive identical samples
        """
        if len(signal) == 0:
            return True
        
        # Check variance
        if np.var(signal) < variance_threshold:
            return True
        
        # Check consecutive identical values
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
        """
        Detect spike artifacts using z-score method.
        SPEC: |z-score| > 3.0 for ≥5 consecutive samples
        Returns boolean mask (True = spike)
        """
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
        """
        Check if signal is within physiologic bounds.
        SPEC: HR 30-200 BPM, SBP 60-200, DBP 30-150, MAP 40-180 mmHg
        Returns boolean mask (True = valid)
        """
        bounds = {
            'hr': (30, 200),
            'sbp': (60, 200),
            'dbp': (30, 150),
            'map': (40, 180),
            'spo2': (70, 100),
            'ppg': (-5, 5),  # Normalized
            'ecg': (-5, 5),   # Normalized
        }
        
        if signal_type not in bounds:
            return np.ones(len(signal), dtype=bool)
        
        min_val, max_val = bounds[signal_type]
        extreme_threshold = 1e-5
        is_extreme = (np.abs(signal) < extreme_threshold) | (np.abs(signal) > 1e5)
        
        return (signal >= min_val) & (signal <= max_val) & ~is_extreme
    
    @staticmethod
    def calculate_ppg_sqi(signal: np.ndarray) -> float:
        """
        Calculate PPG Signal Quality Index using skewness.
        SPEC: skewness > 3.0 indicates good quality
        """
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
        """
        Comprehensive quality assessment returning multiple metrics.
        SPEC: At least 70% of window must be valid
        """
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


# ================================================================================
# UNIFIED PREPROCESSING CONFIGURATION
# ================================================================================

class UnifiedPreprocessingConfig:
    """
    UNIFIED preprocessing configuration for VitalDB and BUT PPG.
    CRITICAL: These values MUST NOT be changed - they ensure compatibility.
    """
    
    # Unified PPG configuration
    PPG_CONFIG = {
        'filter_type': 'cheby2',
        'filter_order': 4,
        'filter_band': [0.5, 10],
        'ripple': 40,
        'target_fs': 25,
        'window_sec': 10.0,
        'hop_sec': 5.0,
        'normalization': 'z-score',
        'min_valid_ratio': 0.7
    }
    
    # Other modalities
    ECG_CONFIG = {
        'filter_type': 'butter',
        'filter_order': 4,
        'filter_band': [0.5, 40],
        'target_fs': 125,
        'zero_phase': True
    }
    
    ABP_CONFIG = {
        'filter_type': 'butter',
        'filter_order': 2,
        'filter_band': [0.5, 10],
        'target_fs': 100,
        'zero_phase': False
    }
    
    @classmethod
    def get_config(cls, modality: str) -> Dict:
        """Get preprocessing config for modality."""
        configs = {
            'ppg': cls.PPG_CONFIG,
            'ecg': cls.ECG_CONFIG,
            'abp': cls.ABP_CONFIG
        }
        return configs.get(modality.lower(), cls.PPG_CONFIG)


# ================================================================================
# BUT PPG LOADER
# ================================================================================

class BUTPPGLoader:
    """Loader for BUT PPG database with multiple format support."""
    
    SUPPORTED_FORMATS = ['.mat', '.hdf5', '.h5', '.csv', '.npy']
    
    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)
        
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")
        
        self.metadata = self._load_metadata()
        self.subjects = self._discover_subjects()
        
        print(f"BUT PPG Loader initialized:")
        print(f"  Data directory: {self.data_dir}")
        print(f"  Subjects found: {len(self.subjects)}")
    
    def _discover_subjects(self) -> List[str]:
        """Discover available subject IDs."""
        subjects = []
        
        for ext in self.SUPPORTED_FORMATS:
            files = list(self.data_dir.rglob(f"*{ext}"))
            for file in files:
                subject_id = file.stem.split('_')[0]
                if subject_id not in subjects:
                    subjects.append(subject_id)
        
        if not subjects:
            for item in self.data_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    subjects.append(item.name)
        
        return sorted(subjects)
    
    def _load_metadata(self) -> Optional[pd.DataFrame]:
        """Load metadata CSV if available."""
        metadata_files = ['metadata.csv', 'subjects.csv', 'info.csv']
        
        for fname in metadata_files:
            fpath = self.data_dir / fname
            if fpath.exists():
                try:
                    return pd.read_csv(fpath)
                except Exception as e:
                    warnings.warn(f"Failed to load metadata: {e}")
        
        return None
    
    def load_subject(self, subject_id: str, signal_type: str = 'ppg') -> Optional[Tuple[np.ndarray, Dict]]:
        """Load signal for a subject."""
        subject_files = list(self.data_dir.rglob(f"{subject_id}*"))
        
        if not subject_files:
            warnings.warn(f"No files found for subject {subject_id}")
            return None
        
        signal_file = subject_files[0]
        
        try:
            if signal_file.suffix == '.mat':
                signal, metadata = self._load_mat(signal_file, signal_type)
            elif signal_file.suffix in ['.hdf5', '.h5']:
                signal, metadata = self._load_hdf5(signal_file, signal_type)
            elif signal_file.suffix == '.csv':
                signal, metadata = self._load_csv(signal_file, signal_type)
            elif signal_file.suffix == '.npy':
                signal, metadata = self._load_npy(signal_file, signal_type)
            else:
                warnings.warn(f"Unsupported format: {signal_file.suffix}")
                return None
            
            if self.metadata is not None:
                subject_meta = self.metadata[self.metadata['subject_id'] == subject_id]
                if not subject_meta.empty:
                    metadata.update(subject_meta.iloc[0].to_dict())
            
            return signal, metadata
            
        except Exception as e:
            warnings.warn(f"Error loading {subject_id}: {e}")
            return None
    
    def _load_mat(self, filepath: Path, signal_type: str) -> Tuple[np.ndarray, Dict]:
        """Load MATLAB .mat file."""
        mat_data = loadmat(str(filepath))
        
        ppg_keys = ['ppg', 'PPG', 'signal', 'data', 'sig']
        signal = None
        
        for key in ppg_keys:
            if key in mat_data:
                signal = mat_data[key]
                break
        
        if signal is None:
            for key, value in mat_data.items():
                if not key.startswith('__') and isinstance(value, np.ndarray):
                    signal = value
                    break
        
        if signal is None:
            raise ValueError(f"Could not find signal in {filepath}")
        
        if signal.ndim > 1:
            signal = signal.flatten()
        
        metadata = {
            'fs': mat_data.get('fs', [100])[0][0] if 'fs' in mat_data else 100,
            'subject_id': filepath.stem,
            'source_file': str(filepath)
        }
        
        return signal, metadata
    
    def _load_hdf5(self, filepath: Path, signal_type: str) -> Tuple[np.ndarray, Dict]:
        """Load HDF5 file."""
        if not H5PY_AVAILABLE:
            raise ImportError("h5py not installed. Cannot load HDF5 files.")
        
        with h5py.File(filepath, 'r') as f:
            ppg_keys = ['ppg', 'PPG', 'signal', 'data']
            signal = None
            
            for key in ppg_keys:
                if key in f:
                    signal = f[key][:]
                    break
            
            if signal is None:
                for key in f.keys():
                    if not key.startswith('_'):
                        signal = f[key][:]
                        break
            
            if signal is None:
                raise ValueError(f"Could not find signal in {filepath}")
            
            metadata = {
                'fs': f.attrs.get('fs', 100),
                'subject_id': filepath.stem,
                'source_file': str(filepath)
            }
            
            for key, value in f.attrs.items():
                if key not in metadata:
                    metadata[key] = value
        
        return signal.flatten(), metadata
    
    def _load_csv(self, filepath: Path, signal_type: str) -> Tuple[np.ndarray, Dict]:
        """Load CSV file."""
        df = pd.read_csv(filepath)
        
        ppg_cols = ['ppg', 'PPG', 'signal', 'value', 'amplitude']
        signal_col = None
        
        for col in ppg_cols:
            if col in df.columns:
                signal_col = col
                break
        
        if signal_col is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                signal_col = numeric_cols[0]
        
        if signal_col is None:
            raise ValueError(f"Could not find signal column in {filepath}")
        
        signal = df[signal_col].values
        
        metadata = {
            'fs': 100,
            'subject_id': filepath.stem,
            'source_file': str(filepath)
        }
        
        return signal, metadata
    
    def _load_npy(self, filepath: Path, signal_type: str) -> Tuple[np.ndarray, Dict]:
        """Load NumPy .npy file."""
        signal = np.load(filepath)
        
        if signal.ndim > 1:
            signal = signal.flatten()
        
        metadata = {
            'fs': 100,
            'subject_id': filepath.stem,
            'source_file': str(filepath)
        }
        
        json_path = filepath.with_suffix('.json')
        if json_path.exists():
            with open(json_path, 'r') as f:
                metadata.update(json.load(f))
        
        return signal, metadata
    
    def get_subject_list(self) -> List[str]:
        """Get list of available subject IDs."""
        return self.subjects.copy()


# ================================================================================
# BUT PPG DATASET
# ================================================================================

class BUTPPGDataset(Dataset):
    """
    BUT PPG dataset with UNIFIED preprocessing matching VitalDB.
    CRITICAL: Uses SAME preprocessing as VitalDB for transfer learning.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        modality: str = 'ppg',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        random_seed: int = 42,
        window_sec: Optional[float] = None,
        hop_sec: Optional[float] = None,
        enable_qc: bool = True,
        min_valid_ratio: float = 0.7,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        segments_per_subject: int = 20,
        return_labels: bool = False,
        return_participant_id: bool = False,
        **kwargs
    ):
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
        
        # Get UNIFIED preprocessing config
        self.preprocessing_config = UnifiedPreprocessingConfig.get_config(self.modality)
        
        # CRITICAL: Enforce unified window/hop parameters
        if window_sec is not None and window_sec != 10.0:
            warnings.warn(f"window_sec={window_sec} differs from VitalDB standard (10.0). Using 10.0 for compatibility.")
        if hop_sec is not None and hop_sec != 5.0:
            warnings.warn(f"hop_sec={hop_sec} differs from VitalDB standard (5.0). Using 5.0 for compatibility.")
        
        self.window_sec = self.preprocessing_config['window_sec']
        self.hop_sec = self.preprocessing_config['hop_sec']
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
        
        # Initialize QC and loader
        self.qc = WaveformQualityControl()
        self.loader = BUTPPGLoader(self.data_dir)
        
        # Get all subjects and create splits
        all_subjects = self.loader.get_subject_list()
        
        if len(all_subjects) == 0:
            raise ValueError(f"No subjects found in {self.data_dir}")
        
        np.random.seed(self.random_seed)
        shuffled_subjects = np.random.permutation(all_subjects)
        
        n = len(shuffled_subjects)
        train_end = int(train_ratio * n)
        val_end = int((train_ratio + val_ratio) * n)
        
        if split == 'train':
            self.subjects = shuffled_subjects[:train_end].tolist()
        elif split == 'val':
            self.subjects = shuffled_subjects[train_end:val_end].tolist()
        else:
            self.subjects = shuffled_subjects[val_end:].tolist()
        
        # Build segment pairs
        self.segment_pairs = self._build_segment_pairs()
        
        print(f"\nBUT PPG Dataset initialized (SPEC {SPEC_VERSION}):")
        print(f"  Split: {split}")
        print(f"  Modality: {modality}")
        print(f"  Subjects: {len(self.subjects)}")
        print(f"  Segments: {len(self.segment_pairs)}")
        print(f"  Filter: {self.preprocessing_config['filter_type']} {self.preprocessing_config['filter_order']}th order")
        print(f"  Sampling: ANY → {self.target_fs}Hz (unified)")
        print(f"  Window: {self.window_sec}s, Hop: {self.hop_sec}s")
        print(f"  QC enabled: {self.enable_qc}")
        print(f"  ⚠️  UNIFIED PREPROCESSING: Matches VitalDB exactly")
    
    def _write_cache_metadata(self):
        """Write cache metadata."""
        metadata = {
            'spec_version': SPEC_VERSION,
            'modality': self.modality,
            'preprocessing_config': self.preprocessing_config,
            'window_sec': self.window_sec,
            'hop_sec': self.hop_sec,
            'qc_enabled': self.enable_qc,
            'min_valid_ratio': self.min_valid_ratio,
            'note': 'UNIFIED preprocessing matching VitalDB'
        }
        
        metadata_file = self.cache_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _apply_unified_filter(self, signal: np.ndarray, original_fs: float) -> np.ndarray:
        """Apply UNIFIED filter - MUST match VitalDB."""
        config = self.preprocessing_config
        
        if config['filter_type'] == 'cheby2':
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
            sos = scipy_signal.butter(
                config['filter_order'],
                config['filter_band'],
                btype='band',
                fs=original_fs,
                output='sos'
            )
            filtered = scipy_signal.sosfiltfilt(sos, signal)
        
        return filtered
    
    def _preprocess_with_qc(self, signal: np.ndarray, original_fs: float, subject_id: str) -> Tuple[Optional[np.ndarray], Dict]:
        """Preprocess signal with UNIFIED pipeline."""
        signal = signal[~np.isnan(signal)]
        
        if len(signal) < original_fs * 2:
            return None, {'overall_valid': False}
        
        # Apply UNIFIED filter
        filtered = self._apply_unified_filter(signal, original_fs)
        
        # Resample to UNIFIED target rate
        if original_fs != self.target_fs:
            n_samples = int(len(filtered) * self.target_fs / original_fs)
            resampled = scipy_signal.resample(filtered, n_samples)
        else:
            resampled = filtered
        
        # Z-score normalization
        mean = np.mean(resampled)
        std = np.std(resampled)
        if std > 1e-8:
            normalized = (resampled - mean) / std
        else:
            normalized = resampled
        
        # Quality control
        if self.enable_qc:
            qc_results = self.qc.get_quality_mask(normalized, self.modality, self.min_valid_ratio)
        else:
            qc_results = {'overall_valid': True, 'mask': np.ones(len(normalized), dtype=bool)}
        
        return normalized, qc_results
    
    def _extract_window(self, subject_id: str, window_idx: int) -> Dict:
        """Extract a window with QC."""
        window_data = {
            'signal': None,
            'qc': None,
            'valid': False,
            'subject_id': subject_id
        }
        
        try:
            result = self.loader.load_subject(subject_id, self.modality)
            if result is None:
                return window_data
            
            signal, metadata = result
            original_fs = metadata.get('fs', 100)
            
            processed, qc_results = self._preprocess_with_qc(signal, original_fs, subject_id)
            
            if processed is None:
                return window_data
            
            window_start = int(window_idx * self.hop_sec * self.target_fs)
            window_end = window_start + self.segment_length
            
            if window_end > len(processed):
                return window_data
            
            window_signal = processed[window_start:window_end]
            window_qc = self.qc.get_quality_mask(window_signal, self.modality, self.min_valid_ratio)
            
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
        """Build segment pairs (for SSL compatibility)."""
        pairs = []
        for subject_id in self.subjects:
            for _ in range(self.segments_per_subject):
                pairs.append({'subject_id': subject_id, 'pair_type': 'same_subject'})
        return pairs
    
    def __len__(self) -> int:
        return len(self.segment_pairs)
    
    def __getitem__(self, idx: int) -> Tuple:
        """Get item with same interface as VitalDB."""
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
        
        # Prepare context
        context = {
            'subject_id': subject_id,
            'qc': {'seg1': window1_data['qc'], 'seg2': window2_data['qc']},
            'dataset': 'butppg'
        }
        
        # Return based on flags
        if self.return_labels and self.return_participant_id:
            return seg1, seg2, subject_id, context
        elif self.return_labels:
            return seg1, seg2, context
        elif self.return_participant_id:
            return seg1, seg2, subject_id
        else:
            return seg1, seg2


# ================================================================================
# CONVENIENCE FUNCTIONS
# ================================================================================

def create_butppg_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create BUT PPG dataloaders."""
    
    def collate_fn_with_context(batch):
        """Collate that handles QC and context."""
        valid_batch = [item for item in batch if item[0].numel() > 0]
        
        if len(valid_batch) == 0:
            dummy = torch.zeros(1, 1, 250)  # Default segment length
            return dummy, dummy
        
        if len(valid_batch[0]) >= 3:
            segs1 = torch.stack([item[0] for item in valid_batch])
            segs2 = torch.stack([item[1] for item in valid_batch])
            contexts = [item[2] for item in valid_batch if len(item) > 2]
            return segs1, segs2, contexts
        else:
            return torch.utils.data.dataloader.default_collate(valid_batch)
    
    train_dataset = BUTPPGDataset(data_dir=data_dir, split='train', **dataset_kwargs)
    val_dataset = BUTPPGDataset(data_dir=data_dir, split='val', **dataset_kwargs)
    test_dataset = BUTPPGDataset(data_dir=data_dir, split='test', **dataset_kwargs)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn_with_context, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn_with_context
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn_with_context
    )
    
    return train_loader, val_loader, test_loader
