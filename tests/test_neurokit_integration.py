#!/usr/bin/env python
"""Quick test to verify NeuroKit2 integration."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.data.detect import find_ecg_rpeaks, find_ppg_peaks

def test_neurokit_detection():
    """Test NeuroKit2 detection works correctly."""
    print("Testing NeuroKit2 integration...")
    
    # Create synthetic signals
    fs = 500.0
    duration = 5.0
    t = np.arange(0, duration, 1/fs)
    
    # Create ECG with clear R-peaks
    ecg = np.zeros_like(t)
    peak_interval = int(fs * 60 / 72)  # 72 bpm
    for i in range(0, len(ecg), peak_interval):
        if i < len(ecg):
            ecg[i] = 1.0
            if i > 0:
                ecg[i-1] = 0.5
            if i < len(ecg) - 1:
                ecg[i+1] = 0.5
    
    # Add noise
    ecg += 0.05 * np.random.randn(len(ecg))
    
    # Test ECG detection
    print("\n1. Testing ECG R-peak detection...")
    try:
        peaks, hr_series = find_ecg_rpeaks(ecg, fs)
        print(f"   ECG peaks type: {type(peaks)}")
        print(f"   ECG peaks shape: {peaks.shape if hasattr(peaks, 'shape') else 'N/A'}")
        print(f"   Number of ECG peaks detected: {len(peaks)}")
        assert isinstance(peaks, np.ndarray), "Peaks should be numpy array"
        assert len(peaks) > 0, "Should detect some peaks"
        print("   ✓ ECG detection successful")
    except Exception as e:
        print(f"   ✗ ECG detection failed: {e}")
        return False
    
    # Create PPG signal
    fs = 125.0
    t = np.arange(0, duration, 1/fs)
    ppg = np.sin(2 * np.pi * 1.2 * t) ** 2
    ppg += 0.02 * np.random.randn(len(ppg))
    
    # Test PPG detection
    print("\n2. Testing PPG peak detection...")
    try:
        peaks = find_ppg_peaks(ppg, fs)
        print(f"   PPG peaks type: {type(peaks)}")
        print(f"   PPG peaks shape: {peaks.shape if hasattr(peaks, 'shape') else 'N/A'}")
        print(f"   Number of PPG peaks detected: {len(peaks)}")
        assert isinstance(peaks, np.ndarray), "Peaks should be numpy array"
        assert len(peaks) > 0, "Should detect some peaks"
        print("   ✓ PPG detection successful")
    except Exception as e:
        print(f"   ✗ PPG detection failed: {e}")
        return False
    
    print("\n✅ NeuroKit2 integration tests passed!")
    return True

if __name__ == "__main__":
    success = test_neurokit_detection()
    sys.exit(0 if success else 1)
