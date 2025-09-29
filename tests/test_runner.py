#!/usr/bin/env python
"""Quick test runner for sync module."""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set mock mode for VitalDB
os.environ['VITALDB_MOCK'] = '1'

# Run a simple test
if __name__ == "__main__":
    from src.data.vitaldb_loader import list_cases, load_channel
    from src.data.sync import resample_to_fs, align_streams
    import numpy as np
    
    print("Testing VitalDB loader (mock mode)...")
    cases = list_cases(required_channels=['ART', 'PLETH'])
    print(f"  Found {len(cases)} mock cases")
    
    if cases:
        case = cases[0]
        print(f"  Loading PPG from case {case['case_id']}...")
        signal, fs = load_channel(case['case_id'], 'PLETH')
        print(f"  Loaded {len(signal)} samples at {fs} Hz")
    
    print("\nTesting resampling...")
    x = np.random.randn(1000)
    y = resample_to_fs(x, 500, 125)
    print(f"  Resampled from 1000 samples to {len(y)} samples")
    
    print("\nTesting stream alignment...")
    streams = {
        'ECG': (np.random.randn(1000), 500.0),
        'PPG': (np.random.randn(250), 125.0),
        'ABP': (np.random.randn(250), 125.0)
    }
    aligned = align_streams(streams, target_fs_hz=125)
    print(f"  Aligned shape: {aligned.shape}")
    
    print("\n✅ All basic tests passed!")
