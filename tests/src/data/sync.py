"""Signal synchronization and resampling utilities."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal
from scipy.signal import resample_poly


def resample_to_fs(
    x: np.ndarray,
    fs_in: float,
    fs_out: float,
    axis: int = -1,
    method: str = 'poly'
) -> np.ndarray:
    """Resample signal to target sampling frequency with anti-aliasing.
    
    Args:
        x: Input signal array.
        fs_in: Input sampling frequency (Hz).
        fs_out: Target sampling frequency (Hz).
        axis: Axis along which to resample.
        method: Resampling method ('poly', 'fourier', 'linear').
        
    Returns:
        Resampled signal at fs_out.
        
    Note:
        - Uses anti-aliasing filter when downsampling.
        - 'poly' method is preferred for most biosignals.
        - Automatically finds optimal up/down factors.
        
    Example:
        >>> # Downsample ECG from 500 Hz to 125 Hz
        >>> ecg_125 = resample_to_fs(ecg_500, 500, 125)
    """
    if fs_in == fs_out:
        return x.copy()
    
    if method == 'poly':
        # Use polyphase resampling (efficient and high quality)
        # Find optimal integer up/down factors
        up, down = _get_resampling_factors(fs_in, fs_out)
        
        # Apply anti-aliasing and resample
        y = resample_poly(x, up, down, axis=axis)
        
    elif method == 'fourier':
        # Fourier-domain resampling
        n_samples_in = x.shape[axis]
        n_samples_out = int(n_samples_in * fs_out / fs_in)
        y = signal.resample(x, n_samples_out, axis=axis)
        
    elif method == 'linear':
        # Simple linear interpolation (faster but lower quality)
        n_samples_in = x.shape[axis]
        n_samples_out = int(n_samples_in * fs_out / fs_in)
        
        # Create time axes
        t_in = np.arange(n_samples_in) / fs_in
        t_out = np.arange(n_samples_out) / fs_out
        
        # Interpolate
        y = np.interp(t_out, t_in, x)
    else:
        raise ValueError(f"Unknown resampling method: {method}")
    
    return y.astype(np.float32)


def align_streams(
    streams: Dict[str, Tuple[np.ndarray, float]],
    target_fs_hz: float = 125.0,
    max_duration_s: Optional[float] = None,
    start_offset_s: float = 0.0
) -> np.ndarray:
    """Align and resample multiple signal streams to common sampling rate.
    
    Args:
        streams: Dictionary mapping channel names to (signal, fs_hz) tuples.
        target_fs_hz: Target sampling frequency for all streams.
        max_duration_s: Maximum duration to process (seconds).
        start_offset_s: Starting offset in seconds.
        
    Returns:
        Aligned array of shape [n_samples, n_channels].
        Channels are ordered alphabetically by name.
        
    Raises:
        ValueError: If streams is empty or signals have incompatible lengths.
        
    Example:
        >>> streams = {
        ...     'ECG_II': (ecg_signal, 500.0),
        ...     'PLETH': (ppg_signal, 125.0),
        ...     'ART': (abp_signal, 125.0)
        ... }
        >>> aligned = align_streams(streams, target_fs_hz=125)
        >>> print(f"Aligned shape: {aligned.shape}")  # [n_samples, 3]
    """
    if not streams:
        raise ValueError("No streams provided for alignment")
    
    # Sort channel names for consistent ordering
    channel_names = sorted(streams.keys())
    n_channels = len(channel_names)
    
    # Resample each stream to target frequency
    resampled_streams = []
    min_duration_s = float('inf')
    
    for channel in channel_names:
        signal_data, fs_hz = streams[channel]
        
        # Skip if signal is empty
        if len(signal_data) == 0:
            raise ValueError(f"Empty signal for channel {channel}")
        
        # Calculate duration
        duration_s = len(signal_data) / fs_hz
        min_duration_s = min(min_duration_s, duration_s)
        
        # Resample to target frequency
        if fs_hz != target_fs_hz:
            signal_resampled = resample_to_fs(
                signal_data, fs_hz, target_fs_hz, axis=0
            )
        else:
            signal_resampled = signal_data.copy()
        
        resampled_streams.append(signal_resampled)
    
    # Determine final duration
    if max_duration_s is not None:
        final_duration_s = min(min_duration_s, max_duration_s)
    else:
        final_duration_s = min_duration_s
    
    # Calculate number of samples
    n_samples = int((final_duration_s - start_offset_s) * target_fs_hz)
    if n_samples <= 0:
        raise ValueError(f"Invalid duration or offset: {final_duration_s}s - {start_offset_s}s")
    
    # Calculate start sample
    start_sample = int(start_offset_s * target_fs_hz)
    
    # Align all streams to same length
    aligned = np.zeros((n_samples, n_channels), dtype=np.float32)
    
    for i, signal_data in enumerate(resampled_streams):
        # Trim or pad as needed
        available_samples = len(signal_data) - start_sample
        
        if available_samples <= 0:
            # Signal too short, pad with zeros
            continue
        
        copy_samples = min(n_samples, available_samples)
        aligned[:copy_samples, i] = signal_data[start_sample:start_sample + copy_samples]
    
    return aligned


def _get_resampling_factors(
    fs_in: float,
    fs_out: float,
    max_factor: int = 100
) -> Tuple[int, int]:
    """Find optimal integer up/down sampling factors.
    
    Args:
        fs_in: Input sampling frequency.
        fs_out: Output sampling frequency.
        max_factor: Maximum factor to consider.
        
    Returns:
        Tuple of (up_factor, down_factor).
    """
    # Try to find exact integer factors
    ratio = fs_out / fs_in
    
    # Check if ratio is already simple
    if ratio == int(ratio):
        return int(ratio), 1
    if 1/ratio == int(1/ratio):
        return 1, int(1/ratio)
    
    # Find best rational approximation
    best_error = float('inf')
    best_up = 1
    best_down = 1
    
    for down in range(1, min(max_factor, int(fs_in) + 1)):
        up = round(down * ratio)
        if up == 0 or up > max_factor:
            continue
        
        actual_ratio = up / down
        error = abs(actual_ratio - ratio)
        
        if error < best_error:
            best_error = error
            best_up = up
            best_down = down
            
            # Early exit if we found exact match
            if error < 1e-10:
                break
    
    return best_up, best_down


def synchronize_events(
    events: Dict[str, np.ndarray],
    fs_hz: float,
    window_s: float = 0.1
) -> Dict[str, np.ndarray]:
    """Synchronize event markers across channels.
    
    Args:
        events: Dictionary of channel -> event indices.
        fs_hz: Sampling frequency.
        window_s: Synchronization window in seconds.
        
    Returns:
        Dictionary of synchronized event indices.
    """
    if not events:
        return {}
    
    # Convert to timestamps
    timestamps = {
        ch: indices / fs_hz 
        for ch, indices in events.items()
    }
    
    # Find reference channel with most events
    ref_channel = max(timestamps.keys(), key=lambda k: len(timestamps[k]))
    ref_times = timestamps[ref_channel]
    
    # Align other channels to reference
    synchronized = {ref_channel: events[ref_channel]}
    
    for channel in timestamps:
        if channel == ref_channel:
            continue
        
        ch_times = timestamps[channel]
        aligned_indices = []
        
        for ref_t in ref_times:
            # Find closest event within window
            diffs = np.abs(ch_times - ref_t)
            min_idx = np.argmin(diffs)
            
            if diffs[min_idx] < window_s:
                aligned_indices.append(events[channel][min_idx])
        
        if aligned_indices:
            synchronized[channel] = np.array(aligned_indices)
    
    return synchronized


def compute_stream_delays(
    streams: Dict[str, Tuple[np.ndarray, float]],
    max_lag_s: float = 1.0,
    reference_channel: Optional[str] = None
) -> Dict[str, float]:
    """Compute relative delays between signal streams using cross-correlation.
    
    Args:
        streams: Dictionary mapping channel names to (signal, fs_hz) tuples.
        max_lag_s: Maximum lag to search in seconds.
        reference_channel: Reference channel for alignment (default: first).
        
    Returns:
        Dictionary mapping channel names to delays in seconds.
        Positive delay means the channel lags the reference.
    """
    if not streams:
        return {}
    
    channel_names = sorted(streams.keys())
    
    # Select reference channel
    if reference_channel is None:
        reference_channel = channel_names[0]
    elif reference_channel not in streams:
        raise ValueError(f"Reference channel {reference_channel} not found")
    
    # Get reference signal
    ref_signal, ref_fs = streams[reference_channel]
    
    # Compute delays
    delays = {reference_channel: 0.0}
    
    for channel in channel_names:
        if channel == reference_channel:
            continue
        
        ch_signal, ch_fs = streams[channel]
        
        # Resample to common frequency for correlation
        common_fs = min(ref_fs, ch_fs)
        
        if ref_fs != common_fs:
            ref_resampled = resample_to_fs(ref_signal, ref_fs, common_fs)
        else:
            ref_resampled = ref_signal
        
        if ch_fs != common_fs:
            ch_resampled = resample_to_fs(ch_signal, ch_fs, common_fs)
        else:
            ch_resampled = ch_signal
        
        # Compute cross-correlation
        max_lag_samples = int(max_lag_s * common_fs)
        
        # Use shorter signal for correlation
        min_len = min(len(ref_resampled), len(ch_resampled))
        corr_len = min(min_len, int(10 * common_fs))  # Use up to 10 seconds
        
        # Compute correlation
        correlation = signal.correlate(
            ref_resampled[:corr_len],
            ch_resampled[:corr_len],
            mode='same'
        )
        
        # Find peak
        center = len(correlation) // 2
        search_start = max(0, center - max_lag_samples)
        search_end = min(len(correlation), center + max_lag_samples)
        
        peak_idx = search_start + np.argmax(correlation[search_start:search_end])
        lag_samples = peak_idx - center
        
        # Convert to seconds
        delays[channel] = lag_samples / common_fs
    
    return delays
