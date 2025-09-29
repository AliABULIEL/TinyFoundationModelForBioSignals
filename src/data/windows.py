"""
Windowing and normalization for biosignals.

Evidence-aligned: 10-second non-overlapping windows with ≥3 cardiac cycles.
Normalization using train stats only to prevent leakage.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings

from ..utils.io import load_yaml, save_npz
from .detect import find_ecg_rpeaks, find_ppg_peaks


@dataclass
class NormalizationStats:
    """Statistics for normalization."""
    mean: np.ndarray
    std: np.ndarray
    min: Optional[np.ndarray] = None
    max: Optional[np.ndarray] = None
    method: str = "zscore"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for saving."""
        return {
            'mean': self.mean,
            'std': self.std,
            'min': self.min,
            'max': self.max,
            'method': self.method
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'NormalizationStats':
        """Create from dictionary."""
        return cls(**d)


def make_windows(
    X_tc: np.ndarray,
    fs: float,
    win_s: float = 10.0,
    stride_s: float = 10.0,
    min_cycles: int = 3,
    peaks_tc: Optional[Dict[int, np.ndarray]] = None,
    return_indices: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Create fixed-size windows from time series data.
    
    Args:
        X_tc: Input data [time, channels]
        fs: Sampling frequency in Hz
        win_s: Window size in seconds (default 10.0)
        stride_s: Stride in seconds (default 10.0 for non-overlapping)
        min_cycles: Minimum cardiac cycles per window (default 3)
        peaks_tc: Optional dict of channel -> peak indices for cycle validation
        return_indices: If True, also return start indices
        
    Returns:
        W_ntc: Windowed data [n_windows, time_samples, channels]
        indices: Start indices if return_indices=True
    """
    if X_tc.ndim != 2:
        raise ValueError(f"Expected 2D input [time, channels], got shape {X_tc.shape}")
    
    T, C = X_tc.shape
    win_samples = int(win_s * fs)
    stride_samples = int(stride_s * fs)
    
    # Calculate number of windows
    n_windows = (T - win_samples) // stride_samples + 1
    if n_windows <= 0:
        warnings.warn(f"Signal too short for windowing: {T} samples < {win_samples} window")
        return np.array([]).reshape(0, win_samples, C)
    
    # Create windows
    windows = []
    valid_indices = []
    
    for i in range(n_windows):
        start_idx = i * stride_samples
        end_idx = start_idx + win_samples
        
        if end_idx > T:
            break
            
        window = X_tc[start_idx:end_idx]
        
        # Check minimum cycles if peaks provided
        if peaks_tc is not None and min_cycles > 0:
            valid = False
            for ch_idx, peaks in peaks_tc.items():
                if ch_idx < C:
                    # Count peaks in this window
                    window_peaks = peaks[(peaks >= start_idx) & (peaks < end_idx)]
                    if len(window_peaks) >= min_cycles:
                        valid = True
                        break
            
            if not valid:
                continue
        
        windows.append(window)
        valid_indices.append(start_idx)
    
    if len(windows) == 0:
        warnings.warn(f"No valid windows found (min_cycles={min_cycles})")
        return np.array([]).reshape(0, win_samples, C)
    
    W_ntc = np.stack(windows, axis=0)
    
    if return_indices:
        return W_ntc, np.array(valid_indices)
    return W_ntc


def compute_normalization_stats(
    X: np.ndarray,
    method: str = "zscore",
    axis: Optional[Union[int, Tuple[int, ...]]] = (0,),
    robust: bool = False
) -> NormalizationStats:
    """
    Compute normalization statistics from training data.
    
    Args:
        X: Input data
        method: Normalization method ('zscore', 'minmax', 'patient_zscore')
        axis: Axis/axes to compute stats over (None = all)
        robust: Use median/MAD instead of mean/std for zscore
        
    Returns:
        NormalizationStats object
    """
    if method == "zscore" or method == "patient_zscore":
        if robust:
            # Use median and MAD for robust estimation
            median = np.median(X, axis=axis, keepdims=True)
            mad = np.median(np.abs(X - median), axis=axis, keepdims=True)
            # Scale MAD to approximate std
            std = mad * 1.4826
            mean = np.squeeze(median)
            std = np.squeeze(std)
        else:
            mean = np.mean(X, axis=axis, keepdims=False) if axis is not None else np.mean(X)
            std = np.std(X, axis=axis, keepdims=False) if axis is not None else np.std(X)
            # Avoid division by zero
            std = np.where(std < 1e-8, 1.0, std) if isinstance(std, np.ndarray) else max(std, 1e-8)
        
        return NormalizationStats(mean=mean, std=std, method=method)
    
    elif method == "minmax":
        min_val = np.min(X, axis=axis, keepdims=False) if axis is not None else np.min(X)
        max_val = np.max(X, axis=axis, keepdims=False) if axis is not None else np.max(X)
        # Avoid division by zero
        range_val = max_val - min_val
        range_val = np.where(range_val < 1e-8, 1.0, range_val) if isinstance(range_val, np.ndarray) else max(range_val, 1e-8)
        
        return NormalizationStats(
            mean=min_val,  # Store min as mean for consistency
            std=range_val,  # Store range as std
            min=min_val,
            max=max_val,
            method=method
        )
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def normalize_windows(
    W_ntc: np.ndarray,
    stats: NormalizationStats,
    baseline_correction: bool = True,
    per_channel: bool = True
) -> np.ndarray:
    """
    Normalize windows using precomputed statistics.
    
    Args:
        W_ntc: Windowed data [n_windows, time, channels]
        stats: Normalization statistics from training data
        baseline_correction: Apply baseline correction per window
        per_channel: Apply normalization per channel
        
    Returns:
        Normalized windows
    """
    W_norm = W_ntc.copy()
    
    # Baseline correction (per window)
    if baseline_correction:
        # Subtract mean of first 10% of samples
        baseline_samples = max(1, W_ntc.shape[1] // 10)
        baseline = np.mean(W_norm[:, :baseline_samples, :], axis=1, keepdims=True)
        W_norm = W_norm - baseline
    
    # Apply normalization
    if stats.method == "zscore":
        if per_channel:
            # Broadcast stats to match window dimensions
            mean = stats.mean.reshape(1, 1, -1) if stats.mean.ndim == 1 else stats.mean
            std = stats.std.reshape(1, 1, -1) if stats.std.ndim == 1 else stats.std
        else:
            mean = stats.mean
            std = stats.std
        
        W_norm = (W_norm - mean) / std
        
    elif stats.method == "minmax":
        min_val = stats.min.reshape(1, 1, -1) if per_channel and stats.min.ndim == 1 else stats.min
        range_val = stats.std.reshape(1, 1, -1) if per_channel and stats.std.ndim == 1 else stats.std
        
        W_norm = (W_norm - min_val) / range_val
        
    elif stats.method == "patient_zscore":
        # Compute patient-specific stats
        patient_mean = np.mean(W_norm, axis=(1, 2), keepdims=True)
        patient_std = np.std(W_norm, axis=(1, 2), keepdims=True)
        patient_std = np.where(patient_std < 1e-8, 1.0, patient_std)
        
        W_norm = (W_norm - patient_mean) / patient_std
    
    return W_norm


def validate_cardiac_cycles(
    signal: np.ndarray,
    fs: float,
    signal_type: str = "ecg",
    min_cycles: int = 3
) -> Tuple[bool, int]:
    """
    Validate that a signal contains minimum cardiac cycles.
    
    Args:
        signal: 1D signal array
        fs: Sampling frequency
        signal_type: Type of signal ('ecg' or 'ppg')
        min_cycles: Minimum required cycles
        
    Returns:
        (is_valid, n_cycles)
    """
    try:
        if signal_type.lower() == "ecg":
            peaks, _ = find_ecg_rpeaks(signal, fs)
        elif signal_type.lower() == "ppg":
            peaks = find_ppg_peaks(signal, fs)
        else:
            return True, -1  # Unknown type, assume valid
        
        n_cycles = len(peaks)
        return n_cycles >= min_cycles, n_cycles
        
    except Exception:
        # If detection fails, be conservative
        return False, 0


def create_sliding_windows(
    X_tc: np.ndarray,
    fs: float,
    window_s: float = 30.0,
    overlap: float = 0.5
) -> np.ndarray:
    """
    Create overlapping windows for inference.
    
    Args:
        X_tc: Input data [time, channels]
        fs: Sampling frequency
        window_s: Window size in seconds
        overlap: Overlap fraction (0.5 = 50% overlap)
        
    Returns:
        Windows [n_windows, samples, channels]
    """
    stride_s = window_s * (1 - overlap)
    return make_windows(X_tc, fs, window_s, stride_s, min_cycles=0)


def aggregate_window_predictions(
    predictions: np.ndarray,
    overlap: float = 0.5,
    method: str = "mean"
) -> np.ndarray:
    """
    Aggregate predictions from overlapping windows.
    
    Args:
        predictions: Window predictions [n_windows, ...]
        overlap: Overlap fraction used in windowing
        method: Aggregation method ('mean', 'median', 'max')
        
    Returns:
        Aggregated predictions
    """
    if method == "mean":
        return np.mean(predictions, axis=0)
    elif method == "median":
        return np.median(predictions, axis=0)
    elif method == "max":
        return np.max(predictions, axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


# Load window config
def load_window_config(config_path: str = "configs/windows.yaml") -> Dict:
    """Load window configuration."""
    return load_yaml(config_path)
