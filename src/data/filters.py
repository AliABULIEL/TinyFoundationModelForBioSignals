"""Digital filters for biosignal processing.

Evidence-aligned implementations of Butterworth and Chebyshev Type II filters
for ECG, PPG, ABP, and EEG signals.
"""

from typing import Optional, Tuple, Union

import numpy as np
from scipy import signal


def apply_bandpass_filter(
    data: np.ndarray,
    fs: float,
    lowcut: float,
    highcut: float,
    filter_type: str = 'butter',
    order: int = 4,
    axis: int = -1
) -> np.ndarray:
    """Apply bandpass filter to signal.
    
    Generic bandpass filter that can use Butterworth or Chebyshev Type II.
    
    Args:
        data: Input signal
        fs: Sampling frequency in Hz
        lowcut: Low cutoff frequency in Hz
        highcut: High cutoff frequency in Hz
        filter_type: 'butter' or 'cheby2'
        order: Filter order
        axis: Axis along which to filter
        
    Returns:
        Filtered signal
    """
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Ensure frequencies are within valid range
    low = max(0.001, min(low, 0.99))
    high = max(low + 0.001, min(high, 0.99))
    
    if filter_type == 'butter':
        b, a = signal.butter(
            N=order,
            Wn=[low, high],
            btype='band',
            analog=False,
            output='ba'
        )
    elif filter_type == 'cheby2':
        b, a = signal.cheby2(
            N=order,
            rs=20,  # Stopband ripple in dB
            Wn=[low, high],
            btype='band',
            analog=False,
            output='ba'
        )
    else:
        raise ValueError(f"Unknown filter type: {filter_type}. Use 'butter' or 'cheby2'")
    
    return apply_filter(data, b, a, axis=axis)


def design_ppg_filter(fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Design PPG filter: Chebyshev Type II order 4, bandpass 0.5-8.0 Hz.
    
    Args:
        fs: Sampling frequency in Hz.
        
    Returns:
        Tuple of (b, a) filter coefficients.
    """
    nyquist = fs / 2
    low = 0.5 / nyquist
    high = 8.0 / nyquist
    
    # Chebyshev Type II with 20 dB stopband ripple
    b, a = signal.cheby2(
        N=4,
        rs=20,  # Stopband ripple in dB
        Wn=[low, high],
        btype='band',
        analog=False,
        output='ba'
    )
    return b, a


def design_ecg_filter(fs: float, mode: str = 'analysis') -> Tuple[np.ndarray, np.ndarray]:
    """Design ECG filter based on analysis mode.
    
    Args:
        fs: Sampling frequency in Hz.
        mode: 'analysis' for general (0.5-40 Hz) or 'rpeak' for R-peak detection (LP 8 Hz).
        
    Returns:
        Tuple of (b, a) filter coefficients.
        
    Raises:
        ValueError: If mode is not 'analysis' or 'rpeak'.
    """
    nyquist = fs / 2
    
    if mode == 'analysis':
        # Butterworth order 4, bandpass 0.5-40.0 Hz
        low = 0.5 / nyquist
        high = 40.0 / nyquist
        
        # Ensure frequencies are within valid range
        high = min(high, 0.99)
        
        b, a = signal.butter(
            N=4,
            Wn=[low, high],
            btype='band',
            analog=False,
            output='ba'
        )
    elif mode == 'rpeak':
        # Chebyshev Type II lowpass 8.0 Hz for R-peak detection
        cutoff = 8.0 / nyquist
        cutoff = min(cutoff, 0.99)
        
        b, a = signal.cheby2(
            N=4,
            rs=20,  # Stopband ripple
            Wn=cutoff,
            btype='low',
            analog=False,
            output='ba'
        )
    else:
        raise ValueError(f"Unknown ECG filter mode: {mode}. Use 'analysis' or 'rpeak'")
    
    return b, a


def design_abp_filter(fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Design ABP filter: Butterworth order 4, bandpass 0.5-20.0 Hz.
    
    Args:
        fs: Sampling frequency in Hz.
        
    Returns:
        Tuple of (b, a) filter coefficients.
    """
    nyquist = fs / 2
    low = 0.5 / nyquist
    high = 20.0 / nyquist
    
    # Ensure frequencies are within valid range
    high = min(high, 0.99)
    
    b, a = signal.butter(
        N=4,
        Wn=[low, high],
        btype='band',
        analog=False,
        output='ba'
    )
    return b, a


def design_eeg_filter(fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Design EEG filter: Butterworth order 4, bandpass 1.0-45.0 Hz.
    
    Args:
        fs: Sampling frequency in Hz.
        
    Returns:
        Tuple of (b, a) filter coefficients.
    """
    nyquist = fs / 2
    low = 1.0 / nyquist
    high = 45.0 / nyquist
    
    # Ensure frequencies are within valid range
    high = min(high, 0.99)
    
    b, a = signal.butter(
        N=4,
        Wn=[low, high],
        btype='band',
        analog=False,
        output='ba'
    )
    return b, a


def apply_filter(
    data: np.ndarray,
    b: np.ndarray,
    a: np.ndarray,
    axis: int = -1,
    padlen: Optional[int] = None
) -> np.ndarray:
    """Apply filter using zero-phase filtering (filtfilt).
    
    Args:
        data: Input signal.
        b: Numerator coefficients.
        a: Denominator coefficients.
        axis: Axis along which to filter.
        padlen: Padding length for filtfilt (default: 3 * max(len(a), len(b))).
        
    Returns:
        Filtered signal with zero phase distortion.
    """
    # Use filtfilt for zero-phase filtering
    # This prevents phase distortion which is crucial for biosignals
    
    # Handle edge effects with padding
    if padlen is None:
        padlen = 3 * max(len(a), len(b))
    
    # Ensure minimum signal length
    min_length = 3 * padlen
    if data.shape[axis] < min_length:
        # For short signals, use regular filter with careful padding
        return signal.lfilter(b, a, data, axis=axis)
    
    try:
        # Apply zero-phase filtering
        filtered = signal.filtfilt(b, a, data, axis=axis, padlen=padlen)
    except ValueError:
        # Fallback for very short signals
        filtered = signal.lfilter(b, a, data, axis=axis)
    
    return filtered.astype(np.float32)


def filter_ppg(
    data: np.ndarray,
    fs: float,
    axis: int = -1
) -> np.ndarray:
    """Apply PPG filter to signal.
    
    Args:
        data: PPG signal.
        fs: Sampling frequency in Hz.
        axis: Axis along which to filter.
        
    Returns:
        Filtered PPG signal.
    """
    b, a = design_ppg_filter(fs)
    return apply_filter(data, b, a, axis=axis)


def filter_ecg(
    data: np.ndarray,
    fs: float,
    mode: str = 'analysis',
    axis: int = -1
) -> np.ndarray:
    """Apply ECG filter to signal.
    
    Args:
        data: ECG signal.
        fs: Sampling frequency in Hz.
        mode: 'analysis' or 'rpeak'.
        axis: Axis along which to filter.
        
    Returns:
        Filtered ECG signal.
    """
    b, a = design_ecg_filter(fs, mode=mode)
    return apply_filter(data, b, a, axis=axis)


def filter_abp(
    data: np.ndarray,
    fs: float,
    axis: int = -1
) -> np.ndarray:
    """Apply ABP filter to signal.
    
    Args:
        data: ABP signal.
        fs: Sampling frequency in Hz.
        axis: Axis along which to filter.
        
    Returns:
        Filtered ABP signal.
    """
    b, a = design_abp_filter(fs)
    return apply_filter(data, b, a, axis=axis)


def filter_eeg(
    data: np.ndarray,
    fs: float,
    axis: int = -1
) -> np.ndarray:
    """Apply EEG filter to signal.
    
    Args:
        data: EEG signal.
        fs: Sampling frequency in Hz.
        axis: Axis along which to filter.
        
    Returns:
        Filtered EEG signal.
    """
    b, a = design_eeg_filter(fs)
    return apply_filter(data, b, a, axis=axis)


def freqz_response(
    b: np.ndarray,
    a: np.ndarray,
    fs: float,
    n_points: int = 512
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute frequency response of digital filter.
    
    Args:
        b: Numerator coefficients.
        a: Denominator coefficients.
        fs: Sampling frequency in Hz.
        n_points: Number of frequency points.
        
    Returns:
        Tuple of (frequencies_hz, magnitude_db, phase_deg).
    """
    # Compute frequency response
    w, h = signal.freqz(b, a, worN=n_points, fs=fs)
    
    # Convert to magnitude in dB
    magnitude_db = 20 * np.log10(np.abs(h) + 1e-10)
    
    # Convert phase to degrees
    phase_deg = np.angle(h, deg=True)
    
    return w, magnitude_db, phase_deg


def get_filter_specs(filter_type: str) -> dict:
    """Get filter specifications for a given type.
    
    Args:
        filter_type: One of 'ppg', 'ecg', 'ecg_rpeak', 'abp', 'eeg'.
        
    Returns:
        Dictionary with filter specifications.
    """
    specs = {
        'ppg': {
            'type': 'cheby2',
            'order': 4,
            'passband': [0.5, 8.0],
            'stopband_ripple_db': 20,
            'description': 'PPG Chebyshev Type II bandpass 0.5-8.0 Hz'
        },
        'ecg': {
            'type': 'butter',
            'order': 4,
            'passband': [0.5, 40.0],
            'description': 'ECG Butterworth bandpass 0.5-40.0 Hz'
        },
        'ecg_rpeak': {
            'type': 'cheby2',
            'order': 4,
            'passband': [0.0, 8.0],
            'stopband_ripple_db': 20,
            'description': 'ECG R-peak Chebyshev Type II lowpass 8.0 Hz'
        },
        'abp': {
            'type': 'butter',
            'order': 4,
            'passband': [0.5, 20.0],
            'description': 'ABP Butterworth bandpass 0.5-20.0 Hz'
        },
        'eeg': {
            'type': 'butter',
            'order': 4,
            'passband': [1.0, 45.0],
            'description': 'EEG Butterworth bandpass 1.0-45.0 Hz'
        }
    }
    
    if filter_type not in specs:
        raise ValueError(f"Unknown filter type: {filter_type}")
    
    return specs[filter_type]


def validate_filter_stability(b: np.ndarray, a: np.ndarray) -> bool:
    """Check if filter is stable (all poles inside unit circle).
    
    Args:
        b: Numerator coefficients.
        a: Denominator coefficients.
        
    Returns:
        True if filter is stable, False otherwise.
    """
    # Get poles of the filter
    poles = np.roots(a)
    
    # Check if all poles are inside unit circle
    return np.all(np.abs(poles) < 1.0)


def design_custom_filter(
    filter_type: str,
    order: int,
    cutoff: Union[float, list],
    fs: float,
    btype: str = 'band',
    rs: float = 20.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Design custom Butterworth or Chebyshev Type II filter.
    
    Args:
        filter_type: 'butter' or 'cheby2'.
        order: Filter order.
        cutoff: Cutoff frequency/frequencies in Hz.
        fs: Sampling frequency in Hz.
        btype: 'low', 'high', 'band', or 'stop'.
        rs: Stopband ripple for Chebyshev (dB).
        
    Returns:
        Tuple of (b, a) filter coefficients.
    """
    nyquist = fs / 2
    
    # Normalize frequencies
    if isinstance(cutoff, list):
        wn = [f / nyquist for f in cutoff]
        wn = [min(f, 0.99) for f in wn]
    else:
        wn = cutoff / nyquist
        wn = min(wn, 0.99)
    
    if filter_type == 'butter':
        b, a = signal.butter(N=order, Wn=wn, btype=btype, analog=False, output='ba')
    elif filter_type == 'cheby2':
        b, a = signal.cheby2(N=order, rs=rs, Wn=wn, btype=btype, analog=False, output='ba')
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    
    return b, a
