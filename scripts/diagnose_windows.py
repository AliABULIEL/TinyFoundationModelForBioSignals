#!/usr/bin/env python3
"""
Diagnostic script to debug VitalDB window creation issues.
Tests a single case end-to-end to see where windows are being rejected.
"""

import sys
from pathlib import Path
import numpy as np
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.vitaldb_loader import load_channel
from src.data.windows import make_windows
from src.data.filters import apply_bandpass_filter
from src.data.detect import find_ppg_peaks, find_ecg_rpeaks
from src.data.quality import compute_sqi

def load_config(config_path: str):
    """Load YAML config."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def test_single_case(case_id: str = "440", channel: str = "PPG"):
    """Test processing a single case with detailed logging."""
    
    print("=" * 80)
    print(f"DIAGNOSTIC TEST: Case {case_id}, Channel {channel}")
    print("=" * 80)
    
    # Load configs
    print("\n1. Loading configurations...")
    channels_config = load_config('configs/channels.yaml')
    windows_config = load_config('configs/windows.yaml')
    
    # Get channel config
    if 'pretrain' in channels_config:
        channels_dict = channels_config['pretrain']
    else:
        channels_dict = channels_config
    
    if channel not in channels_dict:
        print(f"❌ Channel '{channel}' not found!")
        print(f"   Available: {list(channels_dict.keys())}")
        return
    
    ch_config = channels_dict[channel]
    print(f"✓ Found config for {channel}")
    print(f"  VitalDB track: {ch_config.get('vitaldb_track')}")
    print(f"  Sampling rate: {ch_config.get('sampling_rate')}")
    
    # Window parameters
    if 'window' in windows_config:
        window_s = windows_config['window'].get('size_seconds', 4.096)
        stride_s = windows_config['window'].get('step_seconds', 4.096)
    else:
        window_s = windows_config.get('window_length_sec', 4.096)
        stride_s = windows_config.get('stride_sec', 4.096)
    
    min_cycles = windows_config.get('quality', {}).get('min_cycles', 3)
    
    print(f"  Window size: {window_s}s ({int(window_s * 125)} samples at 125Hz)")
    print(f"  Stride: {stride_s}s")
    print(f"  Min cycles: {min_cycles}")
    
    # Load signal
    print(f"\n2. Loading signal from VitalDB...")
    vitaldb_track = ch_config.get('vitaldb_track')
    duration_sec = 60
    target_fs = 125.0  # Target sampling rate
    
    signal, fs = load_channel(
        case_id=case_id,
        channel=vitaldb_track,
        duration_sec=duration_sec,
        auto_fix_alternating=True,
        target_fs=target_fs,  # Use resampling
        use_cache=False  # ← CRITICAL: Disable cache to test fresh data!
    )
    
    if signal is None:
        print(f"❌ Failed to load signal")
        return
    
    print(f"✓ Loaded signal:")
    print(f"  Shape: {signal.shape}")
    print(f"  Length: {len(signal) / fs:.1f}s")
    print(f"  Sampling rate: {fs} Hz")
    print(f"  Range: [{signal.min():.3f}, {signal.max():.3f}]")
    print(f"  Mean: {signal.mean():.3f}, Std: {signal.std():.3f}")
    
    # Apply filter
    print(f"\n3. Applying bandpass filter...")
    if 'filter' in ch_config:
        filt = ch_config['filter']
        filter_type = filt.get('type', 'cheby2')
        lowcut = filt.get('lowcut', 0.5)
        highcut = filt.get('highcut', 10)
        order = filt.get('order', 4)
        
        print(f"  Type: {filter_type}")
        print(f"  Passband: {lowcut}-{highcut} Hz")
        print(f"  Order: {order}")
        
        signal = apply_bandpass_filter(
            signal, fs,
            lowcut=lowcut,
            highcut=highcut,
            filter_type=filter_type,
            order=order
        )
        
        print(f"✓ Filtered signal:")
        print(f"  Range: [{signal.min():.3f}, {signal.max():.3f}]")
        print(f"  Mean: {signal.mean():.3f}, Std: {signal.std():.3f}")
    
    # Detect peaks
    print(f"\n4. Detecting peaks...")
    signal_type = channel.lower()
    peaks = None
    
    if signal_type in ['ppg', 'pleth']:
        peaks = find_ppg_peaks(signal, fs)
        print(f"  Method: PPG peak detection")
    elif signal_type in ['ecg']:
        peaks, _ = find_ecg_rpeaks(signal, fs)
        print(f"  Method: ECG R-peak detection")
    
    if peaks is not None:
        print(f"✓ Found {len(peaks)} peaks")
        if len(peaks) > 0:
            peak_intervals = np.diff(peaks) / fs
            print(f"  Peak intervals: {peak_intervals.mean():.3f}s ± {peak_intervals.std():.3f}s")
            hr = 60 / peak_intervals.mean() if peak_intervals.mean() > 0 else 0
            print(f"  Estimated HR: {hr:.1f} bpm")
    else:
        print(f"⚠️  No peaks detected")
    
    # Quality check
    print(f"\n5. Computing signal quality...")
    if peaks is not None and len(peaks) > 0:
        sqi = compute_sqi(signal, fs, peaks=peaks, signal_type=signal_type)
        print(f"  SQI: {sqi:.3f}")
        
        min_sqi = 0.7
        if sqi < min_sqi:
            print(f"❌ SQI below threshold ({min_sqi})")
            return
        else:
            print(f"✓ SQI acceptable (>= {min_sqi})")
    
    # Create windows
    print(f"\n6. Creating windows...")
    signal_tc = signal.reshape(-1, 1)
    print(f"  Input shape: {signal_tc.shape}")
    
    try:
        case_windows, valid_mask = make_windows(
            X=signal_tc,
            fs=fs,
            win_s=window_s,
            stride_s=stride_s,
            min_cycles=min_cycles if peaks is not None else 0,
            signal_type=signal_type
        )
        
        if case_windows is None:
            print(f"❌ make_windows returned None")
            return
        
        print(f"✓ Created windows:")
        print(f"  Output shape: {case_windows.shape}")
        print(f"  N windows: {len(case_windows)}")
        print(f"  Window shape: {case_windows[0].shape if len(case_windows) > 0 else 'N/A'}")
        
        # Convert valid_mask to array if it's a list
        if valid_mask is not None:
            if isinstance(valid_mask, list):
                valid_mask = np.array(valid_mask)
            print(f"  Valid mask: {valid_mask.sum()} / {len(valid_mask)}")
        else:
            print(f"  Valid mask: N/A")
        
        # Check window validation
        expected_samples = int(window_s * 125)
        print(f"\n7. Validating windows...")
        print(f"  Expected samples per window: {expected_samples}")
        
        valid_count = 0
        invalid_count = 0
        for i, w in enumerate(case_windows):
            # Check dimensionality
            if w.ndim == 1:
                w_test = w.reshape(-1, 1)
            elif w.ndim == 3:
                w_test = w.squeeze(0)
            else:
                w_test = w
            
            # Check shape
            if w_test.shape[0] == expected_samples:
                valid_count += 1
            else:
                invalid_count += 1
                if invalid_count <= 5:  # Show first 5 invalid
                    print(f"  ⚠️  Window {i}: shape {w.shape} -> {w_test.shape}, expected ({expected_samples}, 1)")
        
        print(f"\n  Valid windows: {valid_count}")
        print(f"  Invalid windows: {invalid_count}")
        
        if valid_count == 0:
            print(f"\n❌ NO VALID WINDOWS! All windows rejected by shape validation.")
            print(f"\n  This is the issue! Check:")
            print(f"  1. Window size in config ({window_s}s × 125Hz = {expected_samples} samples)")
            print(f"  2. Actual window shapes from make_windows: {case_windows[0].shape if len(case_windows) > 0 else 'N/A'}")
            print(f"  3. Shape validation logic in ttm_vitaldb.py")
        else:
            print(f"\n✅ SUCCESS! Would save {valid_count} valid windows.")
    
    except Exception as e:
        print(f"❌ Error creating windows: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose VitalDB window creation")
    parser.add_argument("--case-id", default="440", help="VitalDB case ID to test")
    parser.add_argument("--channel", default="PPG", choices=["PPG", "ECG"], help="Channel to test")
    args = parser.parse_args()
    
    test_single_case(args.case_id, args.channel)
