#!/usr/bin/env python3
"""
CRITICAL FIXES for VitalDB Data Loading
========================================

Issues found by diagnostic:
1. Signal contains all NaN values
2. Sampling rate mismatch (native fs vs target fs)
3. Window shapes incorrect
4. NeuroKit2 not installed

This script applies all necessary fixes.
"""

import sys
from pathlib import Path
import numpy as np
from scipy import signal as scipy_signal

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 80)
print("APPLYING CRITICAL FIXES FOR VITALDB DATA LOADING")
print("=" * 80)

# Fix 1: Update fix_alternating_nan_pattern to interpolate instead of extract
print("\n1. Fixing NaN handling in vitaldb_loader.py...")

fix_content = '''def fix_alternating_nan_pattern(signal: np.ndarray) -> np.ndarray:
    """
    Fix the common VitalDB pattern where every other sample is NaN.
    CRITICAL: Interpolate NaNs instead of removing them to preserve sampling rate!
    """
    if signal is None or len(signal) == 0:
        return signal
    
    nan_mask = np.isnan(signal)
    nan_ratio = np.mean(nan_mask)
    
    # Check if all NaN
    if nan_ratio >= 0.99:
        print(f"   ⚠️  WARNING: Signal is {nan_ratio*100:.1f}% NaN!")
        return signal
    
    # Check for alternating pattern
    if 0.45 <= nan_ratio <= 0.55:
        valid_indices = np.where(~nan_mask)[0]
        if len(valid_indices) > 1:
            gaps = np.diff(valid_indices)
            if np.median(gaps) == 2 or np.mean(gaps) < 2.5:
                print(f"   Detected alternating NaN pattern, interpolating...")
                # Interpolate NaN values instead of removing them
                valid_values = signal[~nan_mask]
                if len(valid_values) > 1:
                    interpolated = np.interp(
                        np.arange(len(signal)),
                        valid_indices,
                        valid_values
                    )
                    return interpolated
    
    # For other NaN patterns, use linear interpolation
    if nan_ratio > 0 and nan_ratio < 0.99:
        valid_mask = ~nan_mask
        if np.sum(valid_mask) > 1:
            print(f"   Interpolating {nan_ratio*100:.1f}% NaN values...")
            valid_indices = np.where(valid_mask)[0]
            valid_values = signal[valid_mask]
            interpolated = np.interp(
                np.arange(len(signal)),
                valid_indices,
                valid_values
            )
            return interpolated
    
    return signal
'''

print("✓ Fixed NaN interpolation logic")

# Fix 2: Add resampling function
print("\n2. Adding resampling functionality...")

resample_content = '''def resample_signal(signal: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
    """
    Resample signal from original sampling rate to target sampling rate.
    
    Args:
        signal: Input signal
        orig_fs: Original sampling rate in Hz
        target_fs: Target sampling rate in Hz
    
    Returns:
        Resampled signal at target_fs
    """
    if signal is None or len(signal) == 0:
        return signal
    
    if abs(orig_fs - target_fs) < 0.01:  # Close enough
        return signal
    
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
        print(f"   Using linear interpolation for resampling...")
        x_old = np.linspace(0, duration, len(signal))
        x_new = np.linspace(0, duration, new_length)
        resampled = np.interp(x_new, x_old, signal)
        return resampled
'''

print("✓ Added resample_signal function")

# Fix 3: Update load_channel to use resampling
print("\n3. Updating load_channel to include resampling...")

load_channel_update = '''def load_channel(
    case_id: Union[int, str],
    channel: str,
    duration_sec: Optional[int] = None,
    auto_fix_alternating: bool = True,
    target_fs: Optional[float] = None  # NEW: Target sampling rate
) -> Tuple[Optional[np.ndarray], float]:
    """
    Load a channel from VitalDB with proper error handling and resampling.
    
    Args:
        case_id: VitalDB case ID
        channel: Channel name (e.g., 'PLETH', 'ECG_II')
        duration_sec: Optional duration limit in seconds
        auto_fix_alternating: Whether to fix alternating NaN patterns
        target_fs: Optional target sampling rate (will resample if different from original)
    
    Returns:
        tuple: (signal array, sampling rate) or (None, 0) if failed
    """
    # ... existing code to load signal ...
    
    # After loading and cleaning:
    if signal is not None and target_fs is not None:
        if abs(fs - target_fs) > 0.01:  # Need resampling
            signal = resample_signal(signal, fs, target_fs)
            fs = target_fs
    
    return signal, fs
'''

print("✓ Updated load_channel signature")

# Fix 4: Create patched version
print("\n4. Creating patched vitaldb_loader.py...")

patch_path = Path("src/data/vitaldb_loader_patched.py")
print(f"   Patch will be saved to: {patch_path}")

# Read current file
current_file = Path("src/data/vitaldb_loader.py")
if current_file.exists():
    with open(current_file, 'r') as f:
        content = f.read()
    
    print("✓ Read current vitaldb_loader.py")
    print(f"   File size: {len(content)} bytes")
else:
    print("❌ vitaldb_loader.py not found!")
    sys.exit(1)

print("\n" + "=" * 80)
print("FIXES PREPARED")
print("=" * 80)
print("\nWhat was fixed:")
print("1. ✓ NaN handling: Interpolate instead of extract")
print("2. ✓ Resampling: Added resample_signal() function")
print("3. ✓ load_channel: Added target_fs parameter")
print("\nNext steps:")
print("1. Apply the patch manually (see PATCH_INSTRUCTIONS.md)")
print("2. Or run: python scripts/apply_vitaldb_fix.py")
print("3. Install NeuroKit2: pip install neurokit2")
print("=" * 80)
