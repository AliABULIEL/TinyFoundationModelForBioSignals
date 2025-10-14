"""
Windowing and normalization utilities for biosignal data.

This module provides functions to:
- Create fixed-size windows from continuous signals
- Validate windows contain sufficient cardiac cycles  
- Normalize windows using various strategies
- Apply train-only statistics for proper test set evaluation

Key Functions:
    make_windows: Create 10-second windows from continuous signals
    validate_cardiac_cycles: Ensure windows have ≥3 cardiac cycles
    compute_normalization_stats: Calculate train set statistics
    normalize_windows: Apply normalization with train-only stats
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import signal as scipy_signal

from .detect import find_ecg_rpeaks, find_ppg_peaks


@dataclass
class NormalizationStats:
    """Statistics for normalization computed from training data only."""
    
    mean: Dict[str, float]
    std: Dict[str, float]
    min: Optional[Dict[str, float]] = None
    max: Optional[Dict[str, float]] = None
    patient_stats: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'mean': self.mean,
            'std': self.std,
            'min': self.min,
            'max': self.max,
            'patient_stats': self.patient_stats
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'NormalizationStats':
        """Load from dictionary."""
        return cls(**data)


def make_windows(
    X: np.ndarray,
    fs: float,
    win_s: float = 10.0,
    stride_s: float = 10.0,
    min_cycles: int = 3,
    signal_type: str = 'ecg'
) -> Tuple[np.ndarray, List[bool]]:
    """
    Create fixed-size windows from continuous signal.
    
    Args:
        X: Input signal array of shape (n_samples,) or (n_samples, n_channels)
        fs: Sampling frequency in Hz
        win_s: Window size in seconds (default: 10s)
        stride_s: Stride between windows in seconds (default: 10s, non-overlapping)
        min_cycles: Minimum cardiac cycles per window (default: 3)
        signal_type: Type of signal for peak detection ('ecg' or 'ppg')
        
    Returns:
        windows: Array of shape (n_windows, window_samples) or (n_windows, window_samples, n_channels)
        valid_mask: Boolean mask indicating which windows have sufficient cycles
        
    Example:
        >>> signal = np.random.randn(1250)  # 10s at 125 Hz
        >>> windows, valid = make_windows(signal, fs=125, win_s=10)
        >>> assert windows.shape[0] == 1
        >>> assert windows.shape[1] == 1250
    """
    # Ensure 2D array
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n_samples, n_channels = X.shape
    
    # Calculate window parameters
    win_samples = int(win_s * fs)
    stride_samples = int(stride_s * fs)
    
    # Calculate number of windows
    n_windows = (n_samples - win_samples) // stride_samples + 1
    
    if n_windows <= 0:
        return np.array([]), []
    
    # Create windows
    windows = []
    valid_mask = []
    
    for i in range(n_windows):
        start = i * stride_samples
        end = start + win_samples
        
        if end > n_samples:
            break
            
        window = X[start:end]
        windows.append(window)
        
        # Validate cardiac cycles for first channel (primary signal)
        is_valid = validate_cardiac_cycles(
            window[:, 0],
            fs,
            min_cycles=min_cycles,
            signal_type=signal_type
        )
        valid_mask.append(is_valid)
    
    # Stack windows
    windows = np.array(windows)
    
    # Reshape if single channel
    if n_channels == 1:
        windows = windows.squeeze(axis=-1)
    
    return windows, valid_mask


def validate_cardiac_cycles(
    window: np.ndarray,
    fs: float,
    min_cycles: int = 3,
    signal_type: str = 'ecg'
) -> bool:
    """
    Validate that a window contains sufficient cardiac cycles.
    
    Args:
        window: Signal window array
        fs: Sampling frequency in Hz
        min_cycles: Minimum required cardiac cycles
        signal_type: Type of signal ('ecg' or 'ppg')
        
    Returns:
        True if window has at least min_cycles cardiac cycles
        
    Example:
        >>> # Create synthetic signal with 5 R-peaks
        >>> fs = 125
        >>> t = np.linspace(0, 10, int(10 * fs))
        >>> signal = np.zeros_like(t)
        >>> for i in range(5):
        ...     peak_idx = int((i * 2 + 1) * fs)  # Peak every 2 seconds
        ...     if peak_idx < len(signal):
        ...         signal[peak_idx] = 1.0
        >>> is_valid = validate_cardiac_cycles(signal, fs, min_cycles=3)
        >>> assert is_valid  # Should have at least 3 cycles
    """
    try:
        # Find peaks based on signal type
        if signal_type.lower() == 'ecg':
            peaks = find_ecg_rpeaks(window, fs)
        elif signal_type.lower() == 'ppg':
            peaks = find_ppg_peaks(window, fs)
        else:
            # Default to simple peak finding
            height = np.percentile(np.abs(window), 75)
            peaks, _ = scipy_signal.find_peaks(
                np.abs(window),
                height=height,
                distance=int(0.4 * fs)  # Min 0.4s between peaks
            )
        
        # Check if we have enough peaks (cycles)
        return len(peaks) >= min_cycles
        
    except Exception:
        # If peak detection fails, be conservative
        return False


def compute_normalization_stats(
    windows: np.ndarray,
    channel_names: List[str],
    patient_ids: Optional[List[str]] = None,
    compute_patient_stats: bool = False
) -> NormalizationStats:
    """
    Compute normalization statistics from training windows.
    
    Args:
        windows: Array of shape (n_windows, window_samples, n_channels)
        channel_names: List of channel names
        patient_ids: Optional list of patient IDs for each window
        compute_patient_stats: Whether to compute per-patient statistics
        
    Returns:
        NormalizationStats object with computed statistics
        
    Example:
        >>> windows = np.random.randn(100, 1250, 2)  # 100 windows, 2 channels
        >>> stats = compute_normalization_stats(windows, ['ecg', 'ppg'])
        >>> assert 'ecg' in stats.mean
        >>> assert 'ppg' in stats.std
    """
    if windows.ndim == 2:
        windows = windows[:, :, np.newaxis]
    
    n_windows, window_samples, n_channels = windows.shape
    
    if len(channel_names) != n_channels:
        raise ValueError(f"Number of channel names ({len(channel_names)}) doesn't match channels ({n_channels})")
    
    # Initialize stats dictionaries
    mean = {}
    std = {}
    min_vals = {}
    max_vals = {}
    patient_stats = {} if compute_patient_stats and patient_ids else None
    
    # Compute global statistics per channel
    for i, channel in enumerate(channel_names):
        channel_data = windows[:, :, i].flatten()
        mean[channel] = float(np.mean(channel_data))
        std[channel] = float(np.std(channel_data))
        min_vals[channel] = float(np.min(channel_data))
        max_vals[channel] = float(np.max(channel_data))
    
    # Compute per-patient statistics if requested
    if compute_patient_stats and patient_ids:
        unique_patients = np.unique(patient_ids)
        for patient in unique_patients:
            patient_mask = np.array(patient_ids) == patient
            patient_windows = windows[patient_mask]
            
            patient_stats[patient] = {}
            for i, channel in enumerate(channel_names):
                patient_data = patient_windows[:, :, i].flatten()
                patient_mean = float(np.mean(patient_data))
                patient_std = float(np.std(patient_data))
                patient_stats[patient][channel] = (patient_mean, patient_std)
    
    return NormalizationStats(
        mean=mean,
        std=std,
        min=min_vals,
        max=max_vals,
        patient_stats=patient_stats
    )


def normalize_windows(
    windows: np.ndarray,
    stats: NormalizationStats,
    channel_names: List[str],
    method: str = 'zscore',
    patient_ids: Optional[List[str]] = None
) -> np.ndarray:
    """
    Normalize windows using pre-computed training statistics.
    
    Args:
        windows: Array of shape (n_windows, window_samples, n_channels)
        stats: Pre-computed normalization statistics from training set
        channel_names: List of channel names
        method: Normalization method ('zscore', 'patient_zscore', 'minmax', 'ppg_minmax')
        patient_ids: Patient IDs for patient-level normalization
        
    Returns:
        Normalized windows array
        
    Example:
        >>> windows = np.random.randn(10, 1250, 2) * 10 + 5
        >>> stats = NormalizationStats(
        ...     mean={'ecg': 5.0, 'ppg': 5.0},
        ...     std={'ecg': 10.0, 'ppg': 10.0},
        ...     min={'ecg': -30.0, 'ppg': -30.0},
        ...     max={'ecg': 35.0, 'ppg': 35.0}
        ... )
        >>> norm_windows = normalize_windows(windows, stats, ['ecg', 'ppg'], method='zscore')
        >>> assert np.abs(np.mean(norm_windows)) < 1.0  # Should be close to 0
    """
    if windows.ndim == 2:
        windows = windows[:, :, np.newaxis]
        squeeze_output = True
    else:
        squeeze_output = False
    
    n_windows, window_samples, n_channels = windows.shape
    normalized = windows.copy()
    
    if len(channel_names) != n_channels:
        raise ValueError(f"Number of channel names ({len(channel_names)}) doesn't match channels ({n_channels})")
    
    for i, channel in enumerate(channel_names):
        if method == 'zscore':
            # Standard z-score normalization using train stats
            normalized[:, :, i] = (windows[:, :, i] - stats.mean[channel]) / (stats.std[channel] + 1e-8)
            
        elif method == 'patient_zscore':
            # Patient-specific z-score normalization
            if not patient_ids or not stats.patient_stats:
                # Fall back to global zscore
                normalized[:, :, i] = (windows[:, :, i] - stats.mean[channel]) / (stats.std[channel] + 1e-8)
            else:
                for j, patient_id in enumerate(patient_ids):
                    if patient_id in stats.patient_stats and channel in stats.patient_stats[patient_id]:
                        p_mean, p_std = stats.patient_stats[patient_id][channel]
                        normalized[j, :, i] = (windows[j, :, i] - p_mean) / (p_std + 1e-8)
                    else:
                        # Use global stats as fallback
                        normalized[j, :, i] = (windows[j, :, i] - stats.mean[channel]) / (stats.std[channel] + 1e-8)
                        
        elif method == 'minmax':
            # Min-max normalization to [0, 1]
            range_val = stats.max[channel] - stats.min[channel]
            if range_val > 0:
                normalized[:, :, i] = (windows[:, :, i] - stats.min[channel]) / range_val
            else:
                normalized[:, :, i] = 0.0
                
        elif method == 'ppg_minmax' and channel.lower() in ['ppg', 'pleth']:
            # Special min-max for PPG signals
            range_val = stats.max[channel] - stats.min[channel]
            if range_val > 0:
                normalized[:, :, i] = (windows[:, :, i] - stats.min[channel]) / range_val
            else:
                normalized[:, :, i] = 0.0
        elif method == 'ppg_minmax':
            # For non-PPG channels when using ppg_minmax, use zscore
            normalized[:, :, i] = (windows[:, :, i] - stats.mean[channel]) / (stats.std[channel] + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    if squeeze_output:
        normalized = normalized.squeeze(axis=-1)
    
    return normalized


def save_normalization_stats(stats: NormalizationStats, path: str) -> None:
    """Save normalization statistics to a numpy file."""
    np.save(path, stats.to_dict())


def load_normalization_stats(path: str) -> NormalizationStats:
    """Load normalization statistics from a numpy file."""
    data = np.load(path, allow_pickle=True).item()
    return NormalizationStats.from_dict(data)
