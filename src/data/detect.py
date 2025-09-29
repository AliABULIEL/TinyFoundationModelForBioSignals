"""Peak detection for ECG and PPG signals.

Wraps NeuroKit2 implementations of Pan-Tompkins (ECG) and Elgendi (PPG) algorithms.
"""

from typing import Optional, Tuple

import numpy as np

try:
    import neurokit2 as nk
    NEUROKIT_AVAILABLE = True
except ImportError:
    NEUROKIT_AVAILABLE = False
    # Mock for testing without neurokit2
    class MockNeuroKit:
        @staticmethod
        def ecg_peaks(signal, sampling_rate=1000, method="neurokit"):
            # Simple peak detection for testing
            threshold = np.mean(signal) + 2 * np.std(signal)
            peaks = np.where(signal > threshold)[0]
            # Ensure minimum distance between peaks
            if len(peaks) > 1:
                min_distance = int(0.6 * sampling_rate)  # 100 bpm max
                filtered_peaks = [peaks[0]]
                for p in peaks[1:]:
                    if p - filtered_peaks[-1] >= min_distance:
                        filtered_peaks.append(p)
                peaks = np.array(filtered_peaks)
            return peaks, {}
        
        @staticmethod
        def ppg_peaks(signal, sampling_rate=1000, method="elgendi"):
            # Simple peak detection for PPG
            from scipy import signal as scipy_signal
            # Smooth signal first
            window_size = int(0.1 * sampling_rate)
            if window_size % 2 == 0:
                window_size += 1
            smoothed = scipy_signal.savgol_filter(signal, window_size, 3)
            # Find peaks
            peaks, _ = scipy_signal.find_peaks(smoothed, distance=int(0.4 * sampling_rate))
            return peaks, {}
        
        @staticmethod
        def signal_rate(peaks, sampling_rate=1000, desired_length=None):
            if len(peaks) < 2:
                if desired_length:
                    return np.full(desired_length, np.nan)
                return np.array([np.nan])
            
            # Calculate instantaneous rate
            rr_intervals = np.diff(peaks) / sampling_rate  # in seconds
            rates = 60 / rr_intervals  # in bpm
            
            if desired_length:
                # Interpolate to desired length
                indices = peaks[1:]  # Rate corresponds to the second peak of each interval
                full_rate = np.zeros(desired_length)
                
                # Set rate at peak locations
                for i, idx in enumerate(indices):
                    if idx < desired_length:
                        full_rate[idx] = rates[i]
                
                # Forward fill
                last_rate = rates[0] if len(rates) > 0 else 60
                for i in range(desired_length):
                    if full_rate[i] == 0:
                        full_rate[i] = last_rate
                    else:
                        last_rate = full_rate[i]
                
                return full_rate
            
            return rates
    
    nk = MockNeuroKit()


def find_ecg_rpeaks(
    ecg: np.ndarray,
    fs: float,
    mode: str = "analysis",
    method: str = "neurokit"
) -> Tuple[np.ndarray, np.ndarray]:
    """Find R-peaks in ECG signal using Pan-Tompkins algorithm.
    
    Args:
        ecg: ECG signal array.
        fs: Sampling frequency in Hz.
        mode: Filter mode - "analysis" (0.5-40 Hz) or "rpeak" (LP 8 Hz).
        method: Peak detection method (default "neurokit" uses Pan-Tompkins).
        
    Returns:
        Tuple of (peak_indices, heart_rate_series).
        - peak_indices: Array of R-peak sample indices.
        - heart_rate_series: Instantaneous heart rate at each sample (bpm).
        
    Note:
        Uses NeuroKit2's implementation of Pan-Tompkins algorithm.
        The mode parameter affects pre-filtering before detection.
    """
    if not NEUROKIT_AVAILABLE and method != "simple":
        print("Warning: NeuroKit2 not available, using simple peak detection")
        method = "simple"
    
    # Pre-filter based on mode
    from .filters import filter_ecg
    ecg_filtered = filter_ecg(ecg, fs, mode=mode)
    
    if method == "simple" or not NEUROKIT_AVAILABLE:
        # Fallback simple detection
        peaks = _simple_ecg_peaks(ecg_filtered, fs)
        hr_series = _compute_hr_series(peaks, fs, len(ecg))
    else:
        # Use NeuroKit2's Pan-Tompkins implementation
        try:
            # Detect R-peaks
            result = nk.ecg_peaks(
                ecg_filtered,
                sampling_rate=fs,
                method=method  # "neurokit" uses enhanced Pan-Tompkins
            )
            
            # Extract peak indices from result
            # NeuroKit2 returns a tuple: (dataframe/dict, info_dict)
            if isinstance(result, tuple):
                peaks_data = result[0]
            else:
                peaks_data = result
            
            # Handle DataFrame or dict
            if hasattr(peaks_data, 'columns'):  # DataFrame
                if 'ECG_R_Peaks' in peaks_data.columns:
                    peaks = np.where(peaks_data['ECG_R_Peaks'].values)[0]
                else:
                    # Fallback
                    peaks = _simple_ecg_peaks(ecg_filtered, fs)
            elif isinstance(peaks_data, dict) and 'ECG_R_Peaks' in peaks_data:
                peaks = np.where(peaks_data['ECG_R_Peaks'])[0]
            else:
                peaks = np.array(peaks_data) if not isinstance(peaks_data, np.ndarray) else peaks_data
            
            # Compute heart rate series
            if len(peaks) > 1:
                hr_series = nk.signal_rate(
                    peaks,
                    sampling_rate=fs,
                    desired_length=len(ecg)
                )
                if not isinstance(hr_series, np.ndarray):
                    hr_series = np.array(hr_series)
            else:
                hr_series = _compute_hr_series(peaks, fs, len(ecg))
            
        except Exception as e:
            print(f"NeuroKit2 detection failed: {e}, using simple detection")
            peaks = _simple_ecg_peaks(ecg_filtered, fs)
            hr_series = _compute_hr_series(peaks, fs, len(ecg))
    
    return peaks.astype(np.int32), hr_series.astype(np.float32)


def find_ppg_peaks(
    ppg: np.ndarray,
    fs: float,
    method: str = "elgendi"
) -> np.ndarray:
    """Find systolic peaks in PPG signal using Elgendi algorithm.
    
    Args:
        ppg: PPG signal array.
        fs: Sampling frequency in Hz.
        method: Peak detection method (default "elgendi").
        
    Returns:
        Array of systolic peak sample indices.
        
    Note:
        Uses NeuroKit2's implementation of Elgendi algorithm,
        which is optimized for PPG signals.
    """
    if not NEUROKIT_AVAILABLE and method != "simple":
        print("Warning: NeuroKit2 not available, using simple peak detection")
        method = "simple"
    
    # Pre-filter PPG signal
    from .filters import filter_ppg
    ppg_filtered = filter_ppg(ppg, fs)
    
    if method == "simple" or not NEUROKIT_AVAILABLE:
        # Fallback simple detection
        peaks = _simple_ppg_peaks(ppg_filtered, fs)
    else:
        # Use NeuroKit2's Elgendi implementation
        try:
            result = nk.ppg_peaks(
                ppg_filtered,
                sampling_rate=fs,
                method=method
            )
            
            # Extract peak indices from result
            # NeuroKit2 returns a tuple: (dataframe/dict, info_dict)
            if isinstance(result, tuple):
                peaks_data = result[0]
            else:
                peaks_data = result
            
            # Handle DataFrame or dict
            if hasattr(peaks_data, 'columns'):  # DataFrame
                if 'PPG_Peaks' in peaks_data.columns:
                    peaks = np.where(peaks_data['PPG_Peaks'].values)[0]
                else:
                    # Fallback
                    peaks = _simple_ppg_peaks(ppg_filtered, fs)
            elif isinstance(peaks_data, dict) and 'PPG_Peaks' in peaks_data:
                peaks = np.where(peaks_data['PPG_Peaks'])[0]
            else:
                peaks = np.array(peaks_data) if not isinstance(peaks_data, np.ndarray) else peaks_data
                
        except Exception as e:
            print(f"NeuroKit2 detection failed: {e}, using simple detection")
            peaks = _simple_ppg_peaks(ppg_filtered, fs)
    
    return peaks.astype(np.int32)


def _simple_ecg_peaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    """Simple R-peak detection fallback.
    
    Args:
        ecg: Filtered ECG signal.
        fs: Sampling frequency.
        
    Returns:
        Array of R-peak indices.
    """
    from scipy import signal
    
    # Simple threshold-based detection
    # Square to emphasize R-peaks
    ecg_squared = ecg ** 2
    
    # Moving average to smooth
    window_size = int(0.15 * fs)  # 150ms window
    ecg_smoothed = signal.convolve(ecg_squared, np.ones(window_size) / window_size, mode='same')
    
    # Dynamic threshold
    threshold = np.mean(ecg_smoothed) + 0.5 * np.std(ecg_smoothed)
    
    # Find peaks above threshold
    min_distance = int(0.4 * fs)  # Minimum 400ms between R-peaks (150 bpm max)
    peaks, _ = signal.find_peaks(
        ecg_smoothed,
        height=threshold,
        distance=min_distance
    )
    
    return peaks


def _simple_ppg_peaks(ppg: np.ndarray, fs: float) -> np.ndarray:
    """Simple PPG peak detection fallback.
    
    Args:
        ppg: Filtered PPG signal.
        fs: Sampling frequency.
        
    Returns:
        Array of systolic peak indices.
    """
    from scipy import signal
    
    # Smooth the signal
    window_size = int(0.1 * fs)  # 100ms window
    if window_size % 2 == 0:
        window_size += 1
    ppg_smoothed = signal.savgol_filter(ppg, window_size, 3)
    
    # Find peaks
    min_distance = int(0.4 * fs)  # Minimum 400ms between peaks
    
    # Dynamic threshold based on signal statistics
    threshold = np.mean(ppg_smoothed) + 0.3 * np.std(ppg_smoothed)
    
    peaks, _ = signal.find_peaks(
        ppg_smoothed,
        height=threshold,
        distance=min_distance
    )
    
    return peaks


def _compute_hr_series(peaks: np.ndarray, fs: float, length: int) -> np.ndarray:
    """Compute instantaneous heart rate series from peaks.
    
    Args:
        peaks: R-peak indices.
        fs: Sampling frequency.
        length: Desired length of output series.
        
    Returns:
        Heart rate series in bpm.
    """
    if len(peaks) < 2:
        # Not enough peaks to compute rate
        return np.full(length, 60.0)  # Default 60 bpm
    
    # Compute RR intervals in samples
    rr_intervals = np.diff(peaks)
    
    # Convert to heart rate in bpm
    hr_values = 60.0 * fs / rr_intervals
    
    # Clip unrealistic values
    hr_values = np.clip(hr_values, 30, 200)
    
    # Create full series by interpolation
    hr_series = np.zeros(length)
    
    # Assign HR values at peak locations
    for i in range(1, len(peaks)):
        start_idx = peaks[i-1]
        end_idx = peaks[i] if i < len(peaks) else length
        
        if start_idx < length:
            end_idx = min(end_idx, length)
            hr_series[start_idx:end_idx] = hr_values[i-1]
    
    # Handle edges
    if peaks[0] > 0:
        hr_series[:peaks[0]] = hr_values[0] if len(hr_values) > 0 else 60.0
    
    return hr_series


def compute_peak_statistics(
    peaks: np.ndarray,
    fs: float,
    signal_length: int
) -> dict:
    """Compute statistics from detected peaks.
    
    Args:
        peaks: Peak indices.
        fs: Sampling frequency.
        signal_length: Total signal length in samples.
        
    Returns:
        Dictionary with peak statistics:
        - count: Number of peaks
        - mean_rate: Mean rate (bpm)
        - std_rate: Standard deviation of rate
        - mean_interval: Mean interval (seconds)
        - coverage: Fraction of signal with detected peaks
    """
    stats = {
        'count': len(peaks),
        'mean_rate': np.nan,
        'std_rate': np.nan,
        'mean_interval': np.nan,
        'coverage': 0.0
    }
    
    if len(peaks) < 2:
        return stats
    
    # Compute intervals
    intervals_samples = np.diff(peaks)
    intervals_seconds = intervals_samples / fs
    
    # Compute rates
    rates_bpm = 60.0 / intervals_seconds
    
    # Filter physiological rates (30-200 bpm)
    valid_rates = rates_bpm[(rates_bpm >= 30) & (rates_bpm <= 200)]
    
    if len(valid_rates) > 0:
        stats['mean_rate'] = np.mean(valid_rates)
        stats['std_rate'] = np.std(valid_rates)
        stats['mean_interval'] = np.mean(60.0 / valid_rates)
    
    # Coverage: fraction of signal spanned by peaks
    if len(peaks) > 0:
        span = peaks[-1] - peaks[0]
        stats['coverage'] = span / signal_length
    
    return stats


def compute_peak_statistics(
    peaks: np.ndarray,
    fs: float,
    signal_length: int = None
) -> dict:
    """Compute statistics from detected peaks.
    
    Args:
        peaks: Peak indices.
        fs: Sampling frequency.
        signal_length: Total signal length in samples.
        
    Returns:
        Dictionary with statistics.
    """
    stats = {
        'n_peaks': len(peaks),
        'mean_rate': 0.0,
        'std_rate': 0.0,
        'mean_interval': 0.0,
        'coverage': 0.0
    }
    
    if len(peaks) < 2:
        return stats
    
    # Calculate intervals
    intervals = np.diff(peaks) / fs  # in seconds
    rates = 60.0 / intervals  # in bpm
    
    stats['mean_rate'] = np.mean(rates)
    stats['std_rate'] = np.std(rates)
    stats['mean_interval'] = np.mean(intervals)
    
    # Coverage (time from first to last peak)
    if signal_length:
        stats['coverage'] = (peaks[-1] - peaks[0]) / fs
    else:
        stats['coverage'] = (peaks[-1] - peaks[0]) / fs
    
    return stats


def validate_peaks(
    peaks: np.ndarray,
    fs: float,
    signal_length: int,
    min_rate: float = 30,
    max_rate: float = 200
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and clean detected peaks.
    
    Args:
        peaks: Peak indices.
        fs: Sampling frequency.
        signal_length: Total signal length.
        min_rate: Minimum physiological rate (bpm).
        max_rate: Maximum physiological rate (bpm).
        
    Returns:
        Tuple of (valid_peaks, removed_indices).
    """
    if len(peaks) < 2:
        return peaks, np.array([])
    
    # Calculate intervals
    intervals = np.diff(peaks) / fs
    rates = 60.0 / intervals
    
    # Find valid intervals
    valid_mask = (rates >= min_rate) & (rates <= max_rate)
    
    # Keep peaks that form valid intervals
    valid_indices = [0]  # Keep first peak
    for i, is_valid in enumerate(valid_mask):
        if is_valid:
            valid_indices.append(i + 1)
    
    valid_peaks = peaks[valid_indices]
    removed_indices = np.setdiff1d(np.arange(len(peaks)), valid_indices)
    
    return valid_peaks, removed_indices


def align_peaks(
    ecg_peaks: np.ndarray,
    ppg_peaks: np.ndarray,
    max_delay_samples: int = None,
    fs: float = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Align ECG and PPG peaks accounting for pulse transit time.
    
    Args:
        ecg_peaks: ECG R-peak indices.
        ppg_peaks: PPG systolic peak indices.
        max_delay_samples: Maximum expected delay in samples.
        fs: Sampling frequency (for default delay calculation).
        
    Returns:
        Tuple of (aligned_ecg_peaks, aligned_ppg_peaks, mean_delay).
    """
    if len(ecg_peaks) == 0 or len(ppg_peaks) == 0:
        return np.array([]), np.array([]), 0.0
    
    # Default max delay: 500ms (typical PTT range 100-400ms)
    if max_delay_samples is None:
        if fs is not None:
            max_delay_samples = int(0.5 * fs)
        else:
            max_delay_samples = 100
    
    aligned_ecg = []
    aligned_ppg = []
    delays = []
    
    # For each ECG peak, find closest PPG peak within delay window
    for ecg_peak in ecg_peaks:
        # Define search window
        window_start = ecg_peak
        window_end = ecg_peak + max_delay_samples
        
        # Find PPG peaks in window
        ppg_in_window = ppg_peaks[
            (ppg_peaks >= window_start) & (ppg_peaks <= window_end)
        ]
        
        if len(ppg_in_window) > 0:
            # Take closest PPG peak
            closest_ppg = ppg_in_window[0]
            aligned_ecg.append(ecg_peak)
            aligned_ppg.append(closest_ppg)
            delays.append(closest_ppg - ecg_peak)
    
    aligned_ecg = np.array(aligned_ecg)
    aligned_ppg = np.array(aligned_ppg)
    mean_delay = np.mean(delays) if len(delays) > 0 else 0.0
    
    return aligned_ecg, aligned_ppg, mean_delay
