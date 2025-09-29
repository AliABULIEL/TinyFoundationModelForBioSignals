"""VitalDB data loader with case listing and channel loading."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import vitaldb
    VITALDB_AVAILABLE = True
except ImportError:
    print("Warning: Vital db is not avaialble 1!!!!!!!")
    VITALDB_AVAILABLE = False
    # Mock for testing
    class MockVitalDB:
        @staticmethod
        def list_cases():
            return pd.DataFrame({
                'caseid': ['1', '2', '3'],
                'subject': [1, 2, 3],
                'age': [45, 52, 38]
            })
    vitaldb = MockVitalDB()


def list_cases(
    min_duration_hours: float = 0.5,
    required_channels: Optional[List[str]] = None,
    use_cache: bool = True,
    cache_dir: str = "data/cache"
) -> List[Dict[str, any]]:
    """List available VitalDB cases with metadata.
    
    Args:
        min_duration_hours: Minimum case duration in hours.
        required_channels: List of required channel names.
        use_cache: Whether to use cached case list.
        cache_dir: Directory for cache files.
        
    Returns:
        List of dictionaries with case metadata:
        - case_id: Case identifier
        - subject_id: Subject identifier
        - available_channels: List of available channel names
        - duration_s: Duration in seconds
        
    Example:
        >>> cases = list_cases(required_channels=['ART', 'PLETH'])
        >>> print(f"Found {len(cases)} cases with required channels")
    """
    # Check if we're in mock mode (for testing)
    if os.environ.get('VITALDB_MOCK', '0') == '1' or not VITALDB_AVAILABLE:
        return _mock_list_cases(min_duration_hours, required_channels)
    
    # Try to load cached case list
    cache_path = Path(cache_dir) / "vitaldb_cases.parquet"
    
    if use_cache and cache_path.exists():
        try:
            df_cases = pd.read_parquet(cache_path)
        except Exception:
            df_cases = vitaldb.list_cases()
            _save_case_cache(df_cases, cache_path)
    else:
        # Fetch case list from VitalDB
        df_cases = vitaldb.list_cases()
        if use_cache:
            _save_case_cache(df_cases, cache_path)
    
    # Process cases
    cases = []
    for _, row in df_cases.iterrows():
        # Extract case metadata
        case_id = str(row.get('caseid', row.name))
        subject_id = row.get('subject', case_id)
        
        # Parse available channels from track info
        channels = _parse_channels(row)
        
        # Calculate duration (assuming it's in the data or we fetch it)
        duration_s = _estimate_duration(row)
        
        # Filter by duration
        if duration_s < min_duration_hours * 3600:
            continue
        
        # Filter by required channels
        if required_channels:
            if not all(ch in channels for ch in required_channels):
                continue
        
        cases.append({
            'case_id': case_id,
            'subject_id': subject_id,
            'available_channels': channels,
            'duration_s': duration_s
        })
    
    return cases


def load_channel(
    case_id: str,
    channel: str,
    use_cache: bool = True,
    cache_dir: str = "data/cache"
) -> Tuple[np.ndarray, float]:
    """Load a single channel from VitalDB case.
    
    Args:
        case_id: Case identifier.
        channel: Channel name (e.g., 'ART', 'PLETH', 'ECG_II').
        use_cache: Whether to use cached data.
        cache_dir: Directory for cache files.
        
    Returns:
        Tuple of (signal_array, sampling_frequency_hz).
        
    Raises:
        ValueError: If case or channel not found.
        
    Example:
        >>> signal, fs = load_channel('1', 'PLETH')
        >>> print(f"Loaded {len(signal)/fs:.1f} seconds at {fs} Hz")
    """
    # Check if we're in mock mode
    if os.environ.get('VITALDB_MOCK', '0') == '1' or not VITALDB_AVAILABLE:
        return _mock_load_channel(case_id, channel)
    
    # Try cache first
    if use_cache:
        cache_path = Path(cache_dir) / f"case_{case_id}" / f"{channel}.npz"
        if cache_path.exists():
            try:
                data = np.load(cache_path)
                return data['signal'], float(data['fs'])
            except Exception:
                pass  # Fall back to loading from VitalDB
    
    # Load from VitalDB
    try:
        # Load the case
        vital_data = vitaldb.load_case(case_id)
        
        # Map channel names (handle aliases)
        channel_map = {
            'ART': ['ART', 'ABP', 'ARTERIAL'],
            'PLETH': ['PLETH', 'PPG', 'SpO2'],
            'ECG_II': ['ECG_II', 'ECG_LEAD_II', 'II']
        }
        
        # Find the actual channel name in the data
        actual_channel = channel
        if channel in channel_map:
            for alias in channel_map[channel]:
                if alias in vital_data:
                    actual_channel = alias
                    break
        
        if actual_channel not in vital_data:
            raise ValueError(f"Channel {channel} not found in case {case_id}")
        
        # Extract signal and sampling frequency
        track = vital_data[actual_channel]
        signal = track['vals']
        fs = track.get('srate', 100.0)  # Default to 100 Hz if not specified
        
        # Remove NaN values (mark as 0 or interpolate)
        if np.any(np.isnan(signal)):
            signal = _handle_nans(signal)
        
        # Cache the result
        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, signal=signal, fs=fs)
        
        return signal.astype(np.float32), float(fs)
        
    except Exception as e:
        raise ValueError(f"Failed to load channel {channel} from case {case_id}: {e}")


def _mock_list_cases(
    min_duration_hours: float = 0.5,
    required_channels: Optional[List[str]] = None
) -> List[Dict[str, any]]:
    """Mock case listing for testing."""
    mock_cases = [
        {
            'case_id': '1',
            'subject_id': 'subj_001',
            'available_channels': ['ART', 'PLETH', 'ECG_II'],
            'duration_s': 7200.0  # 2 hours
        },
        {
            'case_id': '2',
            'subject_id': 'subj_002',
            'available_channels': ['ART', 'PLETH', 'ECG_II', 'EEG'],
            'duration_s': 3600.0  # 1 hour
        },
        {
            'case_id': '3',
            'subject_id': 'subj_003',
            'available_channels': ['PLETH', 'ECG_II'],
            'duration_s': 1800.0  # 0.5 hours
        }
    ]
    
    # Filter by duration
    cases = [c for c in mock_cases if c['duration_s'] >= min_duration_hours * 3600]
    
    # Filter by required channels
    if required_channels:
        cases = [
            c for c in cases
            if all(ch in c['available_channels'] for ch in required_channels)
        ]
    
    return cases


def _mock_load_channel(case_id: str, channel: str) -> Tuple[np.ndarray, float]:
    """Mock channel loading for testing."""
    np.random.seed(hash((case_id, channel)) % 2**32)
    
    # Mock sampling frequencies
    fs_map = {
        'ART': 125.0,
        'PLETH': 125.0,
        'ECG_II': 500.0,  # Will be downsampled
        'EEG': 250.0
    }
    
    fs = fs_map.get(channel, 100.0)
    duration_s = {'1': 7200, '2': 3600, '3': 1800}.get(case_id, 3600)
    n_samples = int(duration_s * fs)
    
    # Generate synthetic signal based on channel type
    if channel in ['ART', 'ABP']:
        # Blood pressure: oscillating around 120/80
        t = np.linspace(0, duration_s, n_samples)
        signal = 100 + 20 * np.sin(2 * np.pi * 1.2 * t)  # ~72 bpm
        signal += 5 * np.random.randn(n_samples)
    elif channel in ['PLETH', 'PPG']:
        # PPG: cardiac pulses
        t = np.linspace(0, duration_s, n_samples)
        signal = np.sin(2 * np.pi * 1.2 * t) ** 2  # Pulse-like
        signal += 0.1 * np.random.randn(n_samples)
    elif channel == 'ECG_II':
        # ECG: R-peaks
        t = np.linspace(0, duration_s, n_samples)
        signal = np.zeros(n_samples)
        # Add R-peaks every ~0.83s (72 bpm)
        peak_interval = int(0.83 * fs)
        signal[::peak_interval] = 1.0
        # Smooth slightly
        from scipy.ndimage import gaussian_filter1d
        signal = gaussian_filter1d(signal, sigma=2)
        signal += 0.05 * np.random.randn(n_samples)
    else:
        # Generic signal
        signal = np.random.randn(n_samples) * 0.1
    
    return signal.astype(np.float32), fs


def _parse_channels(row: pd.Series) -> List[str]:
    """Parse available channels from case metadata."""
    # This would parse the actual VitalDB track information
    # For now, return common channels
    channels = []
    
    # Check for various track columns or parse track string
    track_info = str(row.get('track', ''))
    
    # Common channels in VitalDB
    channel_keywords = {
        'ART': ['ART', 'ABP', 'ARTERIAL'],
        'PLETH': ['PLETH', 'PPG', 'SpO2'],
        'ECG_II': ['ECG', 'II', 'LEAD'],
        'EEG': ['EEG', 'BIS'],
        'CO2': ['CO2', 'ETCO2', 'CAPNO']
    }
    
    for channel, keywords in channel_keywords.items():
        if any(kw in track_info.upper() for kw in keywords):
            channels.append(channel)
    
    # If no channels parsed, assume standard set
    if not channels:
        channels = ['ART', 'PLETH', 'ECG_II']
    
    return channels


def _estimate_duration(row: pd.Series) -> float:
    """Estimate case duration in seconds."""
    # Try to get duration from metadata
    duration = row.get('duration', row.get('time_len', None))
    
    if duration is not None:
        # Convert to seconds if needed
        if isinstance(duration, str):
            # Parse duration string (e.g., "02:30:00")
            try:
                parts = duration.split(':')
                if len(parts) == 3:
                    h, m, s = map(float, parts)
                    return h * 3600 + m * 60 + s
            except:
                pass
        else:
            # Assume it's already in seconds or minutes
            duration = float(duration)
            if duration < 1000:  # Likely in minutes
                return duration * 60
            return duration
    
    # Default to 1 hour if unknown
    return 3600.0


def _save_case_cache(df: pd.DataFrame, cache_path: Path) -> None:
    """Save case list to cache."""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, compression='snappy')
    except Exception:
        pass  # Ignore cache errors


def _handle_nans(signal: np.ndarray) -> np.ndarray:
    """Handle NaN values in signal."""
    if not np.any(np.isnan(signal)):
        return signal
    
    # Simple interpolation for NaN values
    nans = np.isnan(signal)
    if np.all(nans):
        return np.zeros_like(signal)
    
    # Linear interpolation
    x = np.arange(len(signal))
    signal[nans] = np.interp(x[nans], x[~nans], signal[~nans])
    
    return signal
