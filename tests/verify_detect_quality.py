#!/usr/bin/env python3
"""Quick verification of detection and quality modules."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.data.detect import find_ecg_rpeaks, find_ppg_peaks
from src.data.quality import (
    ecg_sqi, template_corr, ppg_ssqi, ppg_abp_corr,
    hard_artifacts, window_accept
)


def test_detection_quality():
    """Test detection and quality assessment."""
    print("Testing peak detection and quality assessment...")
    
    fs = 125.0
    duration = 10.0
    t = np.arange(0, duration, 1/fs)
    
    # Create synthetic signals
    print("\n1. Creating synthetic signals...")
    
    # ECG with R-peaks
    ecg = np.zeros_like(t)
    peak_interval = int(fs * 60 / 72)  # 72 bpm
    for i in range(0, len(ecg), peak_interval):
        if i < len(ecg):
            ecg[i] = 1.0
    ecg += 0.05 * np.random.randn(len(ecg))
    
    # PPG pulse wave
    ppg = np.sin(2 * np.pi * 1.2 * t) ** 2
    ppg += 0.02 * np.random.randn(len(ppg))
    
    # ABP
    abp = 100 + 20 * np.sin(2 * np.pi * 1.2 * t)
    abp += 2 * np.random.randn(len(abp))
    
    # Test detection
    print("\n2. Testing peak detection...")
    ecg_peaks, hr_series = find_ecg_rpeaks(ecg, fs)
    ppg_peaks = find_ppg_peaks(ppg, fs)
    
    print(f"  ECG peaks detected: {len(ecg_peaks)}")
    print(f"  PPG peaks detected: {len(ppg_peaks)}")
    print(f"  Mean HR: {np.mean(hr_series[hr_series > 0]):.1f} bpm")
    
    # Test quality metrics
    print("\n3. Testing quality metrics...")
    
    # ECG quality
    sqi = ecg_sqi(ecg, ecg_peaks, fs)
    tcorr = template_corr(ecg, ecg_peaks, fs)
    print(f"  ECG SQI: {sqi:.3f}")
    print(f"  ECG template correlation: {tcorr:.3f}")
    
    # PPG quality
    ssqi = ppg_ssqi(ppg, fs)
    pab_corr = ppg_abp_corr(ppg, abp, fs)
    print(f"  PPG sSQI: {ssqi:.3f}")
    print(f"  PPG-ABP correlation: {pab_corr:.3f}")
    
    # Test artifact detection
    print("\n4. Testing artifact detection...")
    
    # Clean signal
    clean_artifacts = hard_artifacts(ecg, fs)
    print(f"  Clean signal artifacts: {clean_artifacts}")
    
    # Signal with flatline
    flatline_signal = ecg.copy()
    flatline_signal[100:400] = 0.0  # 2.4 seconds flatline
    flatline_artifacts = hard_artifacts(flatline_signal, fs)
    print(f"  Flatline signal artifacts: {flatline_artifacts}")
    
    # Test window acceptance
    print("\n5. Testing window acceptance...")
    
    # Good window
    accept, reasons = window_accept(
        ecg=ecg,
        ppg=ppg,
        abp=abp,
        ecg_peaks=ecg_peaks,
        ppg_peaks=ppg_peaks,
        fs=fs,
        ecg_sqi_min=0.5,
        ppg_ssqi_min=0.5,
        min_cycles=3
    )
    print(f"  Good window accepted: {accept}")
    if not accept:
        print(f"  Rejection reasons: {reasons}")
    
    # Poor window (flatline)
    accept_poor, reasons_poor = window_accept(
        ecg=flatline_signal,
        ecg_peaks=ecg_peaks,
        fs=fs,
        check_artifacts=True
    )
    print(f"  Poor window accepted: {accept_poor}")
    if not accept_poor:
        print(f"  Rejection reasons: {reasons_poor}")
    
    print("\n✅ All detection and quality tests completed!")
    return True


if __name__ == "__main__":
    success = test_detection_quality()
    sys.exit(0 if success else 1)
