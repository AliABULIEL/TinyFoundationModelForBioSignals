"""Fixed VitalDB loader using robust API methods."""

import os
import ssl
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Union
import warnings

import numpy as np
import pandas as pd

# Configure SSL for VitalDB access
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass

try:
    import vitaldb
    VITALDB_AVAILABLE = True
except ImportError:
    print("Warning: VitalDB is not available!")
    VITALDB_AVAILABLE = False


def safe_to_numeric(data: Union[np.ndarray, list, pd.Series]) -> Optional[np.ndarray]:
    """Safely convert data to numeric numpy array."""
    if data is None:
        return None
    
    try:
        if isinstance(data, pd.Series):
            data = data.values
        
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        if data.dtype in [np.float32, np.float64, np.int32, np.int64]:
            return data.astype(np.float64)
        
        if data.dtype == object:
            numeric_data = []
            for item in data.flat:
                try:
                    if isinstance(item, (list, np.ndarray)):
                        if len(item) > 0:
                            val = float(item[0]) if item[0] is not None else np.nan
                        else:
                            val = np.nan
                    elif item is None or (isinstance(item, str) and item.strip() == ''):
                        val = np.nan
                    else:
                        val = float(item)
                    numeric_data.append(val)
                except (TypeError, ValueError):
                    numeric_data.append(np.nan)
            
            return np.array(numeric_data, dtype=np.float64)
        
        return data.astype(np.float64)
        
    except Exception as e:
        warnings.warn(f"Could not convert data to numeric: {e}")
        return None


def fix_alternating_nan_pattern(signal: np.ndarray) -> np.ndarray:
    """
    Fix the common VitalDB pattern where every other sample is NaN.
    """
    valid_mask = ~np.isnan(signal)
    valid_ratio = np.mean(valid_mask)
    
    if 0.45 <= valid_ratio <= 0.55:
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > 1:
            gaps = np.diff(valid_indices)
            if np.median(gaps) == 2 or np.mean(gaps) < 2.5:
                print(f"   Detected alternating NaN pattern, extracting valid samples...")
                clean_signal = signal[valid_mask]
                return clean_signal
    
    return signal


def get_available_case_sets() -> Dict[str, Set[int]]:
    """Get all available pre-filtered case sets from VitalDB."""
    if not VITALDB_AVAILABLE:
        return {}
    
    case_sets = {}
    
    # Standard case sets
    if hasattr(vitaldb, 'caseids_bis'):
        case_sets['bis'] = set(vitaldb.caseids_bis)
    if hasattr(vitaldb, 'caseids_des'):
        case_sets['desflurane'] = set(vitaldb.caseids_des)
    if hasattr(vitaldb, 'caseids_sevo'):
        case_sets['sevoflurane'] = set(vitaldb.caseids_sevo)
    if hasattr(vitaldb, 'caseids_rft20'):
        case_sets['remifentanil'] = set(vitaldb.caseids_rft20)
    if hasattr(vitaldb, 'caseids_ppf'):
        case_sets['propofol'] = set(vitaldb.caseids_ppf)
    if hasattr(vitaldb, 'caseids_tiva'):
        case_sets['tiva'] = set(vitaldb.caseids_tiva)
    
    return case_sets


def find_cases_with_track(track_name: str, max_cases: Optional[int] = None) -> List[int]:
    """
    Find VitalDB cases that have a specific track.
    Uses the robust vitaldb.find_cases() API.
    """
    if not VITALDB_AVAILABLE:
        return []
    
    try:
        # Map common names to VitalDB track names
        track_mapping = {
            'PLETH': 'PLETH',
            'PPG': 'PLETH',
            'ECG_II': 'ECG_II',
            'ECG': 'ECG_II',
            'ABP': 'ABP',
            'ART': 'ART',
            'EEG': 'BIS/BIS'
        }
        
        vitaldb_track = track_mapping.get(track_name.upper(), track_name)
        
        print(f"Finding cases with track: {vitaldb_track}...")
        cases = vitaldb.find_cases(vitaldb_track)
        
        if max_cases:
            cases = cases[:max_cases]
        
        print(f"✓ Found {len(cases)} cases with {vitaldb_track}")
        return cases
        
    except Exception as e:
        print(f"Error finding cases: {e}")
        return []


def list_cases(
    min_duration_hours: float = 0.5,
    required_channels: Optional[List[str]] = None,
    use_cache: bool = True,
    cache_dir: str = "data/cache",
    case_set: str = 'bis',
    max_cases: Optional[int] = None
) -> List[Dict[str, any]]:
    """List available VitalDB cases using robust method."""
    if os.environ.get('VITALDB_MOCK', '0') == '1' or not VITALDB_AVAILABLE:
        return _mock_list_cases(min_duration_hours, required_channels)
    
    # Try using find_cases for specific channels
    if required_channels and len(required_channels) > 0:
        # Find cases with first required channel
        case_ids = find_cases_with_track(required_channels[0], max_cases)
    else:
        # Use predefined case sets
        case_sets = get_available_case_sets()
        if case_set in case_sets:
            case_ids = list(case_sets[case_set])
        else:
            case_ids = list(case_sets.get('bis', []))
        
        if max_cases:
            case_ids = case_ids[:max_cases]
    
    cases = []
    for case_id in case_ids:
        cases.append({
            'case_id': str(case_id),
            'subject_id': str(case_id),
            'available_channels': required_channels if required_channels else [],
            'duration_s': 3600
        })
    
    print(f"Found {len(cases)} cases")
    return cases


def load_channel(
    case_id: str,
    channel: str,
    use_cache: bool = True,
    cache_dir: str = "data/cache",
    start_sec: float = 0,
    duration_sec: Optional[float] = None,
    auto_fix_alternating: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Load a single channel from VitalDB case using robust API.
    FIXED: Uses vitaldb.load_case() instead of VitalFile to avoid track errors.
    """
    if os.environ.get('VITALDB_MOCK', '0') == '1' or not VITALDB_AVAILABLE:
        return _mock_load_channel(case_id, channel)
    
    try:
        case_id_int = int(case_id)
    except:
        case_id_int = case_id
    
    # Try cache first
    if use_cache:
        cache_path = Path(cache_dir) / f"case_{case_id}_clean" / f"{channel}.npz"
        if cache_path.exists():
            try:
                data = np.load(cache_path)
                return data['signal'], float(data['fs'])
            except:
                pass
    
    # Load from VitalDB using robust API
    try:
        # Map channel name to VitalDB track
        track_mapping = {
            'PLETH': ['SNUADC/PLETH', 'Solar8000/PLETH', 'Intellivue/PLETH'],
            'PPG': ['SNUADC/PLETH', 'Solar8000/PLETH', 'Intellivue/PLETH'],
            'ECG_II': ['SNUADC/ECG_II', 'Solar8000/ECG_II'],
            'ECG': ['SNUADC/ECG_II', 'SNUADC/ECG_V5', 'Solar8000/ECG_II'],
            'ABP': ['SNUADC/ART', 'Solar8000/ART'],
            'ART': ['SNUADC/ART', 'Solar8000/ART']
        }
        
        channel_upper = channel.upper()
        possible_tracks = track_mapping.get(channel_upper, [channel])
        
        print(f"   Loading case {case_id}, channel {channel}...")
        
        # Try each possible track
        signal = None
        actual_track = None
        
        for track in possible_tracks:
            try:
                # Use vitaldb.load_case() - more robust than VitalFile
                data = vitaldb.load_case(case_id_int, [track])
                
                if data is not None and len(data) > 0:
                    signal = data
                    actual_track = track
                    print(f"   ✓ Loaded from {track}")
                    break
            except Exception as e:
                continue
        
        if signal is None:
            raise ValueError(f"Could not load {channel} from case {case_id}")
        
        # Convert to numeric
        signal = safe_to_numeric(signal)
        if signal is None:
            raise ValueError(f"Could not convert data to numeric")
        
        # Handle 2D arrays
        if signal.ndim == 2:
            signal = signal[:, 0]
        
        # Apply window if specified
        if duration_sec:
            fs = 100.0 if 'PLETH' in actual_track else 500.0
            end_sample = int((start_sec + duration_sec) * fs)
            start_sample = int(start_sec * fs)
            signal = signal[start_sample:end_sample]
        
        # Determine sampling rate
        if 'PLETH' in actual_track or 'PPG' in channel_upper:
            fs = 100.0
        elif 'ECG' in actual_track or 'ECG' in channel_upper:
            fs = 500.0 if 'SNUADC' in actual_track else 250.0
        elif 'ART' in actual_track or 'ABP' in channel_upper:
            fs = 100.0
        else:
            fs = 100.0
        
        # Fix alternating NaN pattern
        original_length = len(signal)
        if auto_fix_alternating:
            signal = fix_alternating_nan_pattern(signal)
            if len(signal) < original_length:
                fs = fs / 2
                print(f"   Adjusted sampling rate to {fs} Hz after removing alternating NaNs")
        
        # Final quality check
        valid_ratio = np.mean(~np.isnan(signal))
        print(f"   Final quality: {valid_ratio:.1%} valid samples ({len(signal)} total)")
        
        if valid_ratio < 0.9:
            # Interpolate remaining NaNs
            valid_mask = ~np.isnan(signal)
            if np.sum(valid_mask) > 100:
                valid_indices = np.where(valid_mask)[0]
                signal = np.interp(np.arange(len(signal)), valid_indices, signal[valid_indices])
                print(f"   Interpolated remaining NaN values")
        
        # Save to cache
        if use_cache and len(signal) > 0:
            cache_path = Path(cache_dir) / f"case_{case_id}_clean" / f"{channel}.npz"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                np.savez_compressed(cache_path, signal=signal, fs=fs)
            except:
                pass
        
        return signal, float(fs)
        
    except Exception as e:
        raise ValueError(f"Error loading channel '{channel}' from case {case_id}: {e}")


def _mock_list_cases(min_duration_hours: float = 0.5,
                     required_channels: Optional[List[str]] = None) -> List[Dict]:
    """Mock case list for testing."""
    return [{'case_id': str(i), 'subject_id': str(i), 
             'available_channels': ['PLETH'], 'duration_s': 3600} 
            for i in range(1, 6)]


def _mock_load_channel(case_id: str, channel: str) -> Tuple[np.ndarray, float]:
    """Mock channel loading."""
    fs = 100.0
    t = np.arange(0, 10, 1/fs)
    signal = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(4 * np.pi * 1.2 * t)
    return signal, fs
