#!/usr/bin/env python3
"""Run tests to verify they pass."""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set mock mode for VitalDB
os.environ['VITALDB_MOCK'] = '1'

def run_unit_tests():
    """Run key unit tests to verify functionality."""
    print("Running unit tests...")
    
    # Test imports
    try:
        from src.data.sync import resample_to_fs, align_streams
        from src.data.vitaldb_loader import list_cases, load_channel
        print("✅ Imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test resampling
    try:
        import numpy as np
        x = np.random.randn(1000)
        y = resample_to_fs(x, 500, 125)
        assert len(y) == 250
        print("✅ Resampling test passed")
    except Exception as e:
        print(f"❌ Resampling test failed: {e}")
        return False
    
    # Test alignment
    try:
        streams = {
            'ECG': (np.random.randn(500), 500.0),
            'PPG': (np.random.randn(125), 125.0),
        }
        aligned = align_streams(streams, target_fs_hz=125)
        assert aligned.shape[1] == 2
        print("✅ Alignment test passed")
    except Exception as e:
        print(f"❌ Alignment test failed: {e}")
        return False
    
    # Test mock loader
    try:
        cases = list_cases(required_channels=['ART', 'PLETH'])
        assert len(cases) > 0
        signal, fs = load_channel(cases[0]['case_id'], 'PLETH')
        assert len(signal) > 0 and fs > 0
        print("✅ VitalDB loader test passed")
    except Exception as e:
        print(f"❌ VitalDB loader test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = run_unit_tests()
    
    if success:
        print("\n✅ All unit tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)
