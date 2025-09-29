#!/usr/bin/env python
"""Quick test for filter implementations."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.data.filters import (
    design_ppg_filter, design_ecg_filter, design_abp_filter, design_eeg_filter,
    filter_ppg, filter_ecg, filter_abp, filter_eeg,
    freqz_response, validate_filter_stability
)


def test_filters():
    """Run basic filter tests."""
    print("Testing filter designs...")
    
    # Test each filter design
    filters_to_test = [
        ("PPG", 125.0, design_ppg_filter),
        ("ECG", 500.0, lambda fs: design_ecg_filter(fs, 'analysis')),
        ("ECG R-peak", 500.0, lambda fs: design_ecg_filter(fs, 'rpeak')),
        ("ABP", 125.0, design_abp_filter),
        ("EEG", 250.0, design_eeg_filter),
    ]
    
    for name, fs, design_func in filters_to_test:
        print(f"\nTesting {name} filter at {fs} Hz:")
        
        # Design filter
        b, a = design_func(fs)
        
        # Check stability
        is_stable = validate_filter_stability(b, a)
        print(f"  Stability: {'✅ Stable' if is_stable else '❌ Unstable'}")
        
        if not is_stable:
            return False
        
        # Check frequency response
        freqs, mag_db, _ = freqz_response(b, a, fs, n_points=256)
        
        # Find -3dB point
        idx_3db = np.where(mag_db > -3)[0]
        if len(idx_3db) > 0:
            passband_start = freqs[idx_3db[0]]
            passband_end = freqs[idx_3db[-1]]
            print(f"  Passband (-3dB): {passband_start:.1f} - {passband_end:.1f} Hz")
        
        # Find max attenuation
        max_atten = np.min(mag_db)
        print(f"  Max attenuation: {max_atten:.1f} dB")
    
    # Test filtering on synthetic signals
    print("\n\nTesting signal filtering...")
    
    duration = 2.0
    fs = 125.0
    t = np.arange(0, duration, 1/fs)
    
    # Create test signals
    test_signal = np.sin(2 * np.pi * 1.2 * t)  # 1.2 Hz sine wave
    noise = 0.1 * np.random.randn(len(t))
    noisy_signal = test_signal + noise
    
    # Test each filter function
    filter_funcs = [
        ("PPG", filter_ppg),
        ("ABP", filter_abp),
    ]
    
    for name, filter_func in filter_funcs:
        filtered = filter_func(noisy_signal, fs)
        
        # Check output validity
        is_finite = np.all(np.isfinite(filtered))
        print(f"  {name}: {'✅ Valid output' if is_finite else '❌ Invalid output'}")
        
        if not is_finite:
            return False
    
    # Test ECG at higher sampling rate
    fs_ecg = 500.0
    t_ecg = np.arange(0, duration, 1/fs_ecg)
    ecg_signal = np.zeros_like(t_ecg)
    ecg_signal[::int(fs_ecg)] = 1.0  # Spikes at 1 Hz
    
    filtered_ecg = filter_ecg(ecg_signal, fs_ecg, mode='analysis')
    is_finite = np.all(np.isfinite(filtered_ecg))
    print(f"  ECG: {'✅ Valid output' if is_finite else '❌ Invalid output'}")
    
    return True


if __name__ == "__main__":
    success = test_filters()
    
    if success:
        print("\n✅ All filter tests passed!")
    else:
        print("\n❌ Some filter tests failed")
        sys.exit(1)
