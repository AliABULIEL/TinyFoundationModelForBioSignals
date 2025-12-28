"""BUT PPG Database Loader with SPEC-compliant processing.

The BUT PPG Database contains PPG recordings from wearable devices.
Reference: https://but.edu/ (Brno University of Technology)

This loader implements:
- Loading PPG signals from various formats
- Quality control using same QC as VitalDB
- Windowing with same parameters as VitalDB (10s windows)
- Unified preprocessing pipeline

Author: Senior Data Engineering Team
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.io import loadmat
import h5py

from .windows import make_windows, compute_normalization_stats, normalize_windows, NormalizationStats
from .detect import find_ppg_peaks
from .quality import compute_sqi

logger = logging.getLogger(__name__)


class BUTPPGLoader:
    """Loader for BUT PPG database with unified preprocessing."""
    
    # BUT PPG database structure
    SUPPORTED_FORMATS = ['.mat', '.hdf5', '.h5', '.csv', '.npy', '.dat']  # BUT-PPG uses .dat files
    
    # Metadata fields
    METADATA_FIELDS = {
        'demographics': ['subject_id', 'age', 'sex', 'height', 'weight'],
        'recording': ['fs', 'duration', 'device', 'location'],
        'quality': ['sqi_mean', 'sqi_std', 'artifact_ratio']
    }
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        fs: float = 125.0,
        window_duration: float = 10.0,
        window_stride: float = 10.0,
        apply_windowing: bool = True
    ):
        """Initialize BUT PPG loader.
        
        Args:
            data_dir: Path to BUT PPG database root directory
            fs: Sampling frequency in Hz (default 125 Hz to match VitalDB)
            window_duration: Window size in seconds (default 10s)
            window_stride: Stride between windows in seconds (default 10s, non-overlapping)
            apply_windowing: Whether to automatically apply windowing
        """
        self.data_dir = Path(data_dir)
        self.fs = fs
        self.window_duration = window_duration
        self.window_stride = window_stride
        self.apply_windowing = apply_windowing
        
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")
        
        # Load metadata if available
        self.metadata = self._load_metadata()
        
        # Discover available subjects
        self.subjects = self._discover_subjects()
        
        logger.info(f"BUT PPG Loader initialized:")
        logger.info(f"  Data directory: {self.data_dir}")
        logger.info(f"  Subjects found: {len(self.subjects)}")
        logger.info(f"  Windowing: {'enabled' if apply_windowing else 'disabled'} ({window_duration}s, stride {window_stride}s)")
    
    def _discover_subjects(self) -> List[str]:
        """Discover available subject IDs from directory structure."""
        subjects = set()

        # For BUT-PPG: Each subject has a directory with record files
        # Directory structure: data_dir/{record_id}/{record_id}_PPG.dat
        for item in self.data_dir.iterdir():
            if item.is_dir() and item.name.isdigit():
                # Directory name is the record ID
                subjects.add(item.name)

        # If no directories found, try file-based discovery
        if not subjects:
            for fmt in self.SUPPORTED_FORMATS:
                # Search in data_dir and one level deep
                for pattern in [f"*{fmt}", f"*/*{fmt}"]:
                    files = list(self.data_dir.glob(pattern))
                    for file in files:
                        # Extract subject ID from filename (everything before first extension)
                        subject_id = file.stem
                        subjects.add(subject_id)

        return sorted(list(subjects))
    
    def _load_metadata(self) -> Optional[pd.DataFrame]:
        """Load metadata CSV if available."""
        metadata_files = [
            self.data_dir / 'metadata.csv',
            self.data_dir / 'subjects.csv',
            self.data_dir / 'info.csv'
        ]
        
        for metadata_file in metadata_files:
            if metadata_file.exists():
                try:
                    df = pd.read_csv(metadata_file)
                    print(f"✓ Loaded metadata from {metadata_file.name}")
                    return df
                except Exception as e:
                    warnings.warn(f"Failed to load metadata: {e}")
        
        return None
    
    def load_subject(
        self,
        subject_id: str,
        signal_type: str = 'ppg',
        return_windows: Optional[bool] = None,
        normalize: bool = True,
        compute_quality: bool = True
    ) -> Optional[Union[Tuple[np.ndarray, Dict], Tuple[np.ndarray, Dict, List[int]]]]:
        """Load PPG signal for a subject with optional windowing.
        
        Args:
            subject_id: Subject identifier
            signal_type: Type of signal ('ppg', 'acc', 'ecg')
            return_windows: Whether to return windowed data (None = use class default)
            normalize: Apply z-score normalization
            compute_quality: Compute signal quality indices
            
        Returns:
            If return_windows=False:
                Tuple of (signal array [T, C], metadata dict)
            If return_windows=True:
                Tuple of (windowed signal [N, 1250, C], metadata dict, window_indices)
            Returns None if subject not found
        """
        if return_windows is None:
            return_windows = self.apply_windowing

        # For BUT-PPG: Look for signal-specific files in subject directory
        # Structure: data_dir/{record_id}/{record_id}_PPG.dat
        subject_dir = self.data_dir / subject_id

        if subject_dir.exists() and subject_dir.is_dir():
            # Build signal filename based on type
            signal_suffix_map = {
                'ppg': '_PPG',
                'ecg': '_ECG',
                'acc': '_ACC'
            }
            suffix = signal_suffix_map.get(signal_type.lower(), '_PPG')

            # Look for specific signal file
            signal_file = subject_dir / f"{subject_id}{suffix}.dat"

            if signal_file.exists():
                subject_files = [signal_file]
            else:
                # Fallback: find any matching file in directory
                subject_files = [
                    f for f in subject_dir.iterdir()
                    if f.is_file() and f.suffix in self.SUPPORTED_FORMATS
                ]
        else:
            # Original fallback: recursive search
            subject_files = list(self.data_dir.rglob(f"{subject_id}*"))

            # Filter to only signal files (exclude .json metadata)
            subject_files = [
                f for f in subject_files
                if f.is_file() and f.suffix in self.SUPPORTED_FORMATS
            ]
        
        if not subject_files:
            warnings.warn(f"No files found for subject {subject_id}")
            return None
        
        # Load from first matching file
        signal_file = subject_files[0]
        
        try:
            # Load based on file format
            if signal_file.suffix == '.mat':
                signal, metadata = self._load_mat(signal_file, signal_type)
            elif signal_file.suffix == '.hdf5' or signal_file.suffix == '.h5':
                signal, metadata = self._load_hdf5(signal_file, signal_type)
            elif signal_file.suffix == '.csv':
                signal, metadata = self._load_csv(signal_file, signal_type)
            elif signal_file.suffix == '.npy':
                signal, metadata = self._load_npy(signal_file, signal_type)
            elif signal_file.suffix == '.dat':
                signal, metadata = self._load_dat(signal_file, signal_type, subject_id)
            else:
                warnings.warn(f"Unsupported format: {signal_file.suffix}")
                return None
            
            # Add subject metadata if available
            if self.metadata is not None:
                subject_meta = self.metadata[self.metadata['subject_id'] == subject_id]
                if not subject_meta.empty:
                    metadata.update(subject_meta.iloc[0].to_dict())
            
            # Ensure signal is 2D [time, channels]
            if signal.ndim == 1:
                signal = signal.reshape(-1, 1)
            elif signal.ndim > 2:
                logger.warning(f"Signal has {signal.ndim} dimensions, flattening to 2D")
                signal = signal.reshape(signal.shape[0], -1)
            
            # Resample if needed (only if we know the original fs)
            original_fs = metadata.get('fs')
            if original_fs is not None and original_fs != self.fs:
                signal = self._resample_signal(signal, original_fs, self.fs)
                metadata['fs'] = self.fs
                metadata['resampled'] = True
            elif original_fs is None:
                # If fs is unknown, assume it matches target
                logger.debug(f"No sampling frequency metadata found, assuming {self.fs} Hz")
                metadata['fs'] = self.fs
            
            # Compute quality if requested
            if compute_quality:
                quality_metrics = self._compute_signal_quality(signal, metadata.get('fs', self.fs))
                metadata.update(quality_metrics)
            
            # Apply windowing if requested
            if return_windows:
                windowed_signal, window_indices = self._create_windows(
                    signal,
                    metadata.get('fs', self.fs),
                    normalize=normalize
                )
                metadata['n_windows'] = len(windowed_signal)
                metadata['window_duration'] = self.window_duration
                metadata['window_stride'] = self.window_stride
                return windowed_signal, metadata, window_indices
            
            return signal, metadata
            
        except Exception as e:
            warnings.warn(f"Error loading {subject_id}: {e}")
            return None
    
    def _load_mat(self, filepath: Path, signal_type: str) -> Tuple[np.ndarray, Dict]:
        """Load MATLAB .mat file."""
        mat_data = loadmat(str(filepath))
        
        # Try common variable names for PPG
        ppg_keys = ['ppg', 'PPG', 'signal', 'data', 'sig']
        signal = None
        
        for key in ppg_keys:
            if key in mat_data:
                signal = mat_data[key]
                break
        
        if signal is None:
            # Get first non-metadata variable
            for key, value in mat_data.items():
                if not key.startswith('__') and isinstance(value, np.ndarray):
                    signal = value
                    break
        
        if signal is None:
            raise ValueError(f"Could not find signal in {filepath}")
        
        # Flatten if needed
        if signal.ndim > 1:
            signal = signal.flatten()
        
        # Extract metadata
        metadata = {
            'fs': mat_data.get('fs', [100])[0][0] if 'fs' in mat_data else 100,
            'subject_id': filepath.stem,
            'source_file': str(filepath)
        }
        
        return signal, metadata
    
    def _load_hdf5(self, filepath: Path, signal_type: str) -> Tuple[np.ndarray, Dict]:
        """Load HDF5 file."""
        with h5py.File(filepath, 'r') as f:
            # Try common dataset names
            ppg_keys = ['ppg', 'PPG', 'signal', 'data']
            signal = None
            
            for key in ppg_keys:
                if key in f:
                    signal = f[key][:]
                    break
            
            if signal is None:
                # Get first dataset
                for key in f.keys():
                    if not key.startswith('_'):
                        signal = f[key][:]
                        break
            
            if signal is None:
                raise ValueError(f"Could not find signal in {filepath}")
            
            # Extract metadata
            metadata = {
                'fs': f.attrs.get('fs', 100),
                'subject_id': filepath.stem,
                'source_file': str(filepath)
            }
            
            # Add all HDF5 attributes
            for key, value in f.attrs.items():
                if key not in metadata:
                    metadata[key] = value
        
        return signal.flatten(), metadata
    
    def _load_csv(self, filepath: Path, signal_type: str) -> Tuple[np.ndarray, Dict]:
        """Load CSV file."""
        df = pd.read_csv(filepath)
        
        # Try common column names
        ppg_cols = ['ppg', 'PPG', 'signal', 'value', 'amplitude']
        signal_col = None
        
        for col in ppg_cols:
            if col in df.columns:
                signal_col = col
                break
        
        if signal_col is None:
            # Use first numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                signal_col = numeric_cols[0]
        
        if signal_col is None:
            raise ValueError(f"Could not find signal column in {filepath}")
        
        signal = df[signal_col].values
        
        metadata = {
            'fs': 100,  # Default, should be in metadata
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
            'fs': 100,  # Default
            'subject_id': filepath.stem,
            'source_file': str(filepath)
        }
        
        # Try to load companion metadata JSON
        json_path = filepath.with_suffix('.json')
        if json_path.exists():
            with open(json_path, 'r') as f:
                metadata.update(json.load(f))

        return signal, metadata

    def _load_dat(self, filepath: Path, signal_type: str, subject_id: str) -> Tuple[np.ndarray, Dict]:
        """Load BUT-PPG .dat file (binary format).

        BUT-PPG .dat files are binary files containing float32 samples.
        Each record has separate files: {record_id}_PPG.dat, {record_id}_ECG.dat, {record_id}_ACC.dat
        """
        # Load binary data as float32
        signal = np.fromfile(filepath, dtype=np.float32)

        # Determine signal type and sampling frequency from filename
        filename = filepath.stem
        if '_PPG' in filename:
            fs = 64  # BUT-PPG: PPG at 64 Hz
        elif '_ECG' in filename:
            fs = 250  # BUT-PPG: ECG at 250 Hz
        elif '_ACC' in filename:
            fs = 64  # BUT-PPG: ACC at 64 Hz
            # ACC has 3 channels (x, y, z), reshape
            if len(signal) % 3 == 0:
                signal = signal.reshape(-1, 3)
        else:
            fs = 64  # Default

        metadata = {
            'fs': fs,
            'subject_id': subject_id,
            'source_file': str(filepath),
            'signal_type': signal_type
        }

        return signal, metadata

    def get_subject_list(self) -> List[str]:
        """Get list of available subject IDs."""
        return self.subjects.copy()
    
    def get_subject_metadata(self, subject_id: str) -> Dict:
        """Get metadata for a specific subject."""
        if self.metadata is not None:
            subject_meta = self.metadata[self.metadata['subject_id'] == subject_id]
            if not subject_meta.empty:
                return subject_meta.iloc[0].to_dict()
        
        return {'subject_id': subject_id}
    
    def _resample_signal(self, signal: np.ndarray, fs_old: float, fs_new: float) -> np.ndarray:
        """Resample signal to target sampling frequency.
        
        Args:
            signal: Input signal [T, C]
            fs_old: Original sampling frequency
            fs_new: Target sampling frequency
            
        Returns:
            Resampled signal
        """
        from scipy import signal as scipy_signal
        
        if fs_old == fs_new:
            return signal
        
        T_old, C = signal.shape
        T_new = int(T_old * fs_new / fs_old)
        
        resampled = np.zeros((T_new, C))
        
        for c in range(C):
            resampled[:, c] = scipy_signal.resample(signal[:, c], T_new)
        
        logger.debug(f"Resampled signal from {fs_old} Hz to {fs_new} Hz ({T_old} -> {T_new} samples)")
        return resampled
    
    def _compute_signal_quality(self, signal: np.ndarray, fs: float) -> Dict[str, float]:
        """Compute signal quality metrics.
        
        Args:
            signal: Input signal [T, C]
            fs: Sampling frequency
            
        Returns:
            Dictionary of quality metrics
        """
        quality_metrics = {}
        
        try:
            # Compute SQI for each channel
            for c in range(signal.shape[1]):
                channel_signal = signal[:, c]
                
                # Find peaks for quality assessment
                try:
                    peaks = find_ppg_peaks(channel_signal, fs)
                    
                    if len(peaks) > 0:
                        # Compute basic quality metrics
                        sqi = compute_sqi(channel_signal, peaks, fs)
                        quality_metrics[f'sqi_ch{c}'] = float(sqi)
                        quality_metrics[f'n_peaks_ch{c}'] = len(peaks)
                        
                        # Peak rate
                        duration = len(channel_signal) / fs
                        peak_rate = len(peaks) / duration * 60  # peaks per minute
                        quality_metrics[f'peak_rate_ch{c}'] = float(peak_rate)
                except Exception as e:
                    logger.debug(f"Could not compute quality for channel {c}: {e}")
                    quality_metrics[f'sqi_ch{c}'] = 0.0
            
            # Overall quality (mean of channel qualities)
            sqi_values = [v for k, v in quality_metrics.items() if k.startswith('sqi_')]
            if sqi_values:
                quality_metrics['sqi_mean'] = float(np.mean(sqi_values))
                quality_metrics['sqi_std'] = float(np.std(sqi_values))
        
        except Exception as e:
            logger.warning(f"Error computing signal quality: {e}")
            quality_metrics['sqi_mean'] = 0.0
        
        return quality_metrics
    
    def _create_windows(
        self,
        signal: np.ndarray,
        fs: float,
        normalize: bool = True
    ) -> Tuple[np.ndarray, List[int]]:
        """Create fixed-size windows from signal.
        
        Args:
            signal: Input signal [T, C]
            fs: Sampling frequency
            normalize: Apply z-score normalization
            
        Returns:
            Tuple of (windowed_signal [N, window_samples, C], window_start_indices)
        """
        # Detect peaks for cycle validation
        peaks_dict = {}
        for c in range(signal.shape[1]):
            try:
                peaks = find_ppg_peaks(signal[:, c], fs)
                if len(peaks) > 0:
                    peaks_dict[c] = peaks
            except Exception as e:
                logger.debug(f"Could not detect peaks for channel {c}: {e}")
        
        # Create windows with minimum 3 cycles requirement
        # Note: make_windows doesn't have return_indices parameter
        windows, valid_mask = make_windows(
            X=signal,
            fs=fs,
            win_s=self.window_duration,
            stride_s=self.window_stride,
            min_cycles=3,
            signal_type='ppg'
        )
        
        # Get indices from valid mask
        indices = [i * int(self.window_stride * fs) for i in range(len(windows))]
        
        # Normalize if requested
        if normalize and len(windows) > 0:
            # Compute normalization stats from all windows
            channel_names = [f'ch{c}' for c in range(windows.shape[-1] if windows.ndim > 2 else 1)]
            norm_stats = compute_normalization_stats(windows, channel_names)
            windows = normalize_windows(windows, norm_stats, channel_names)
        
        logger.debug(f"Created {len(windows)} windows of {self.window_duration}s from signal")
        
        return windows, indices


def find_butppg_cases(data_dir: Union[str, Path]) -> List[str]:
    """Find all available BUT PPG cases (wrapper for compatibility)."""
    loader = BUTPPGLoader(data_dir)
    return loader.get_subject_list()


def load_butppg_signal(
    data_dir: Union[str, Path],
    subject_id: str,
    signal_type: str = 'ppg'
) -> Optional[Tuple[np.ndarray, Dict]]:
    """Load BUT PPG signal (wrapper for compatibility)."""
    loader = BUTPPGLoader(data_dir)
    return loader.load_subject(subject_id, signal_type)
