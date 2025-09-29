"""VitalDB loader that handles the 50% NaN pattern in waveform data."""

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
    This happens when data is stored at half the reported sampling rate.
    """
    valid_mask = ~np.isnan(signal)
    valid_ratio = np.mean(valid_mask)
    
    # Check if we have the alternating pattern (around 50% valid)
    if 0.45 <= valid_ratio <= 0.55:
        # Check if it's actually alternating
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > 1:
            gaps = np.diff(valid_indices)
            # If most gaps are 2 (alternating pattern), extract valid samples
            if np.median(gaps) == 2 or np.mean(gaps) < 2.5:
                print(f"   Detected alternating NaN pattern, extracting valid samples...")
                # Extract only valid samples
                clean_signal = signal[valid_mask]
                return clean_signal
    
    return signal


def get_available_case_sets() -> Dict[str, Set[int]]:
    """Get all available pre-filtered case sets from VitalDB."""
    if not VITALDB_AVAILABLE:
        return {}
    
    case_sets = {}
    
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


def list_cases(
    min_duration_hours: float = 0.5,
    required_channels: Optional[List[str]] = None,
    use_cache: bool = True,
    cache_dir: str = "data/cache",
    case_set: str = 'bis',
    max_cases: Optional[int] = None
) -> List[Dict[str, any]]:
    """List available VitalDB cases."""
    if os.environ.get('VITALDB_MOCK', '0') == '1' or not VITALDB_AVAILABLE:
        return _mock_list_cases(min_duration_hours, required_channels)
    
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
            'available_channels': [],
            'duration_s': 3600
        })
    
    print(f"Found {len(cases)} cases from {case_set} set")
    return cases


def load_channel(
    case_id: str,
    channel: str,
    use_cache: bool = True,
    cache_dir: str = "data/cache",
    start_sec: float = 0,
    duration_sec: Optional[float] = None,
    auto_fix_alternating: bool = True  # NEW: Auto-fix alternating NaN pattern
) -> Tuple[np.ndarray, float]:
    """
    Load a single channel from VitalDB case.
    Handles the alternating NaN pattern common in VitalDB data.
    """
    if os.environ.get('VITALDB_MOCK', '0') == '1' or not VITALDB_AVAILABLE:
        return _mock_load_channel(case_id, channel)
    
    try:
        case_id = int(case_id)
    except:
        pass
    
    # Try cache first
    if use_cache:
        cache_path = Path(cache_dir) / f"case_{case_id}_clean" / f"{channel}.npz"
        if cache_path.exists():
            try:
                data = np.load(cache_path)
                return data['signal'], float(data['fs'])
            except:
                pass
    
    # Load from VitalDB
    try:
        vf = vitaldb.VitalFile(case_id)
        tracks = vf.get_track_names()
        
        # Find the right track (prioritize device-specific waveforms)
        actual_track = None
        channel_upper = channel.upper()
        
        # Priority tracks for each signal type
        priority_tracks = {
            'PLETH': ['SNUADC/PLETH', 'Solar8000/PLETH', 'Intellivue/PLETH'],
            'PPG': ['SNUADC/PLETH', 'Solar8000/PLETH', 'Intellivue/PLETH'],
            'ECG': ['SNUADC/ECG_II', 'SNUADC/ECG_V5', 'Solar8000/ECG_II'],
            'ECG_II': ['SNUADC/ECG_II', 'Solar8000/ECG_II'],
        }
        
        # Find track
        if channel_upper in priority_tracks:
            for priority in priority_tracks[channel_upper]:
                if priority in tracks:
                    actual_track = priority
                    break
        
        # Fallback search
        if not actual_track:
            for track in tracks:
                if 'EVENT' in track.upper():
                    continue
                track_upper = track.upper()
                if 'PLETH' in channel_upper and 'PLETH' in track_upper:
                    actual_track = track
                    break
                elif 'ECG' in channel_upper and 'ECG' in track_upper:
                    actual_track = track
                    break
        
        if not actual_track:
            raise ValueError(f"No track found for '{channel}' in case {case_id}")
        
        print(f"   Loading {actual_track}...")
        
        # Determine sampling rate
        if 'PLETH' in actual_track.upper():
            fs = 100.0
        elif 'ECG' in actual_track.upper():
            fs = 500.0 if 'SNUADC' in actual_track else 250.0
        else:
            fs = 100.0
        
        # Load data
        end_sec = start_sec + duration_sec if duration_sec else start_sec + 60
        data = vf.to_numpy([actual_track], start_sec, end_sec)
        
        if data is None or len(data) == 0:
            raise ValueError(f"No data returned for {actual_track}")
        
        # Convert to numeric
        data = safe_to_numeric(data)
        if data is None:
            raise ValueError(f"Could not convert data to numeric")
        
        # Extract signal
        if data.ndim == 2:
            signal = data[:, 0]
        else:
            signal = data
        
        # Check for alternating NaN pattern and fix if present
        original_length = len(signal)
        if auto_fix_alternating:
            signal = fix_alternating_nan_pattern(signal)
            if len(signal) < original_length:
                # Adjust sampling rate since we removed every other sample
                fs = fs / 2
                print(f"   Adjusted sampling rate to {fs} Hz after removing alternating NaNs")
        
        # Final quality check
        valid_ratio = np.mean(~np.isnan(signal))
        print(f"   Final quality: {valid_ratio:.1%} valid samples ({len(signal)} total)")
        
        if valid_ratio < 0.9:
            print(f"   Warning: Still some NaN values present")
            # Clean remaining NaNs
            valid_mask = ~np.isnan(signal)
            if np.sum(valid_mask) > 100:  # At least 100 valid samples
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
