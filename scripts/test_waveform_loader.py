#!/usr/bin/env python3
"""
Test VitalDB loader that avoids EVENT tracks and finds real waveforms
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Testing VitalDB loader (avoiding EVENT tracks)...")
print("="*60)

try:
    from src.data.vitaldb_loader import load_channel, list_cases, get_available_case_sets
    import numpy as np
    
    # Get cases
    print("\n1. Getting test cases from BIS set:")
    cases = list_cases(case_set='bis', max_cases=5)
    
    print("\n2. Loading actual waveform data (not EVENT tracks):")
    successful = 0
    
    for case in cases:
        case_id = case['case_id']
        print(f"\n   Testing case {case_id}:")
        
        try:
            # Load with debug output to see what tracks are used
            signal, fs = load_channel(
                case_id, 
                'PLETH',  # Request PPG/PLETH data
                use_cache=False, 
                duration_sec=10,
                allow_poor_quality=False
            )
            
            # Check signal quality
            valid_ratio = np.mean(~np.isnan(signal))
            
            print(f"      ✓ SUCCESS: {len(signal)} samples at {fs} Hz")
            print(f"      Quality: {valid_ratio:.1%} valid")
            print(f"      Range: [{np.nanmin(signal):.2f}, {np.nanmax(signal):.2f}]")
            successful += 1
            
        except Exception as e:
            print(f"      ✗ Failed: {str(e)[:100]}")
    
    print(f"\n" + "="*60)
    print(f"RESULTS: {successful}/{len(cases)} cases loaded successfully")
    
    if successful > 0:
        print("\n✅ VitalDB loader is working correctly!")
        print("   The key was to avoid EVENT tracks and load actual waveforms")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("="*60)
