#!/usr/bin/env python
"""Debug template correlation issue."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.data.quality import template_corr

def test_template_correlation():
    """Debug why template correlation returns 0."""
    fs = 125.0
    
    # Create consistent beats with proper window size
    window_ms = 200  # 200ms window around R-peak
    window_samples = int(window_ms * fs / 1000)
    half_window = window_samples // 2
    
    print(f"Window: {window_ms}ms = {window_samples} samples")
    print(f"Half window: {half_window} samples")
    
    # Create beat template that fits in window
    beat_template = np.array([0, 0.1, 0.2, 0.5, 0.8, 1.0, 0.8, 0.5, 0.2, 0.1, 0, -0.1, -0.2, -0.1, 0])
    
    # Create signal with repeated template
    n_beats = 10
    ecg = np.zeros(n_beats * 125)  # 10 seconds at 125 Hz
    peaks = []
    
    for i in range(n_beats):
        peak_pos = i * 125 + half_window  # Ensure enough space before peak
        peaks.append(peak_pos)
        
        # Place template centered at peak
        template_half = len(beat_template) // 2
        start = peak_pos - template_half
        end = start + len(beat_template)
        
        if start >= 0 and end < len(ecg):
            ecg[start:end] = beat_template
            print(f"Beat {i}: peak at {peak_pos}, template at {start}:{end}")
    
    peaks = np.array(peaks)
    
    print(f"\nCreated {len(peaks)} peaks")
    print(f"Peak positions: {peaks[:5]}...")
    print(f"ECG signal length: {len(ecg)}")
    print(f"ECG non-zero values: {np.count_nonzero(ecg)}")
    
    # Test template correlation
    corr = template_corr(ecg, peaks, fs, window_ms=window_ms)
    
    print(f"\nTemplate correlation: {corr:.3f}")
    
    # Debug: Extract beats manually to see what's happening
    beats = []
    for peak in peaks:
        start = peak - half_window
        end = peak + half_window
        
        if start >= 0 and end < len(ecg):
            beat = ecg[start:end]
            if len(beat) == window_samples:
                beats.append(beat)
                print(f"Beat extracted: start={start}, end={end}, len={len(beat)}, max={np.max(beat):.2f}")
    
    print(f"\nExtracted {len(beats)} beats")
    
    if len(beats) > 0:
        beats = np.array(beats)
        print(f"Beats shape: {beats.shape}")
        
        # Compute template manually
        template = np.median(beats, axis=0)
        print(f"Template shape: {template.shape}")
        print(f"Template max: {np.max(template):.3f}")
        
        # Check correlations manually
        correlations = []
        for beat in beats[:3]:  # Check first 3 beats
            if np.std(beat) > 0 and np.std(template) > 0:
                corr_val = np.corrcoef(beat, template)[0, 1]
                correlations.append(corr_val)
                print(f"Beat correlation: {corr_val:.3f}")
    
    return corr > 0.5

if __name__ == "__main__":
    test_template_correlation()
