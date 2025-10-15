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
    # This helps with SSL certificate issues on macOS
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # Also try to use certifi if available
    try:
        import certifi
        os.environ['SSL_CERT_FILE'] = certifi.where()
        os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
    except ImportError:
        pass
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
    CRITICAL FIX: Interpolate NaN values instead of removing them to preserve sampling rate!
    """
    if signal is None or len(signal) == 0:
        return signal
    
    nan_mask = np.isnan(signal)
    nan_ratio = np.mean(nan_mask)
    
    # Check if all NaN
    if nan_ratio >= 0.99:
        print(f"   ⚠️  WARNING: Signal is {nan_ratio*100:.1f}% NaN!")
        return signal
    
    # For alternating pattern or any NaN pattern, interpolate
    if nan_ratio > 0.01:  # If more than 1% NaN
        valid_mask = ~nan_mask
        if np.sum(valid_mask) > 1:
            print(f"   Interpolating {nan_ratio*100:.1f}% NaN values...")
            valid_indices = np.where(valid_mask)[0]
            valid_values = signal[valid_mask]
            # Linear interpolation
            interpolated = np.interp(
                np.arange(len(signal)),
                valid_indices,
                valid_values
            )
            return interpolated
    
    return signal


def resample_signal(signal: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
    """
    Resample signal from original sampling rate to target sampling rate.
    Uses scipy.signal.resample for high-quality resampling.
    
    CRITICAL FIX: Interpolate ANY remaining NaN values BEFORE resampling!
    
    Args:
        signal: Input signal
        orig_fs: Original sampling rate in Hz
        target_fs: Target sampling rate in Hz
    
    Returns:
        Resampled signal at target_fs
    """
    if signal is None or len(signal) == 0:
        return signal
    
    if abs(orig_fs - target_fs) < 0.01:  # Close enough, no need to resample
        return signal
    
    # CRITICAL: Remove ALL NaN before resampling to prevent propagation
    if np.any(np.isnan(signal)):
        nan_ratio = np.mean(np.isnan(signal))
        print(f"   ⚠️  Removing {nan_ratio*100:.1f}% NaN before resampling...")
        
        valid_mask = ~np.isnan(signal)
        if np.sum(valid_mask) < 2:
            print(f"   ❌ Not enough valid samples for resampling!")
            return np.full(int(len(signal) * target_fs / orig_fs), np.nan)
        
        # Interpolate NaN values
        valid_indices = np.where(valid_mask)[0]
        valid_values = signal[valid_mask]
        signal = np.interp(np.arange(len(signal)), valid_indices, valid_values)
        print(f"   ✓ Interpolated NaN values")
    
    # Calculate new number of samples
    duration = len(signal) / orig_fs
    new_length = int(duration * target_fs)
    
    # Use scipy's resample for high-quality resampling
    try:
        from scipy import signal as scipy_signal
        resampled = scipy_signal.resample(signal, new_length)
        print(f"   Resampled: {len(signal)} samples at {orig_fs}Hz → {len(resampled)} samples at {target_fs}Hz")
        return resampled
    except ImportError:
        # Fallback to linear interpolation if scipy not available
        print(f"   Using linear interpolation for resampling (scipy not available)...")
        x_old = np.linspace(0, duration, len(signal))
        x_new = np.linspace(0, duration, new_length)
        resampled = np.interp(x_new, x_old, signal)
        return resampled


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
    if not VITALDB_AVAILABLE:
        raise ImportError("VitalDB package not installed. Install with: pip install vitaldb")
    
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
    auto_fix_alternating: bool = True,
    target_fs: Optional[float] = None  # NEW: Target sampling rate for resampling
) -> Tuple[np.ndarray, float]:
    """
    Load a single channel from VitalDB case using robust API.
    FIXED: Uses vitaldb.load_case() instead of VitalFile to avoid track errors.
    """
    if not VITALDB_AVAILABLE:
        raise ImportError("VitalDB package not installed. Install with: pip install vitaldb")
    
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
                    # vitaldb.load_case returns a DataFrame - extract the column
                    if isinstance(data, pd.DataFrame):
                        if track in data.columns:
                            signal = data[track].values
                        else:
                            # Try finding column by name
                            matching_cols = [c for c in data.columns if track.split('/')[-1] in c]
                            if matching_cols:
                                signal = data[matching_cols[0]].values
                            else:
                                signal = data.iloc[:, 0].values
                    else:
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
        if auto_fix_alternating:
            signal = fix_alternating_nan_pattern(signal)
        
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
        
        # Resample to target fs if specified
        if target_fs is not None and abs(fs - target_fs) > 0.01:
            signal = resample_signal(signal, fs, target_fs)
            fs = target_fs
        
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
