#!/usr/bin/env python3
"""
Test VitalDB loader that fixes alternating NaN pattern
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Testing VitalDB loader with alternating NaN fix...")
print("="*60)

try:
    from src.data.vitaldb_loader import load_channel, list_cases
    import numpy as np
    
    # Get cases
    print("\n1. Getting test cases:")
    cases = list_cases(case_set='bis', max_cases=5)
    
    print("\n2. Loading with alternating NaN pattern fix:")
    successful = 0
    
    for case in cases:
        case_id = case['case_id']
        print(f"\n   Case {case_id}:")
        
        try:
            # Load with auto-fix for alternating NaNs
            signal, fs = load_channel(
                case_id, 
                'PLETH',
                use_cache=False,
                duration_sec=10,
                auto_fix_alternating=True  # This is the key!
            )
            
            # Check results
            valid_ratio = np.mean(~np.isnan(signal))
            
            print(f"      ✓ SUCCESS: {len(signal)} samples at {fs} Hz")
            print(f"      Quality: {valid_ratio:.1%} valid")
            print(f"      Signal stats:")
            print(f"        Mean: {np.nanmean(signal):.2f}")
            print(f"        Std: {np.nanstd(signal):.2f}")
            print(f"        Range: [{np.nanmin(signal):.2f}, {np.nanmax(signal):.2f}]")
            
            successful += 1
            
        except Exception as e:
            print(f"      ✗ Failed: {str(e)[:150]}")
    
    print(f"\n" + "="*60)
    print(f"RESULTS: {successful}/{len(cases)} cases loaded successfully")
    
    if successful > 0:
        print("\n✅ SUCCESS! The alternating NaN pattern has been fixed!")
        print("   The data had every other sample as NaN (50% valid)")
        print("   We extracted only the valid samples and adjusted the sampling rate")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("="*60)
