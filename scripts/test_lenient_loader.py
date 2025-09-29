#!/usr/bin/env python3
"""
Test the fixed VitalDB loader with lenient quality settings
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Testing VitalDB loader with lenient quality settings...")
print("="*60)

try:
    from src.data.vitaldb_loader import load_channel, list_cases, get_available_case_sets
    
    # 1. Check available case sets
    print("\n1. Available case sets:")
    case_sets = get_available_case_sets()
    for name, cases in case_sets.items():
        print(f"   - {name}: {len(cases)} cases")
    
    # 2. List some cases
    print("\n2. Getting cases from BIS set:")
    cases = list_cases(case_set='bis', max_cases=10)
    print(f"   Found {len(cases)} cases")
    
    # 3. Try loading data with LENIENT quality settings
    print("\n3. Testing data loading with allow_poor_quality=True:")
    successful = 0
    partial = 0
    failed = 0
    
    for i, case in enumerate(cases[:10]):
        case_id = case['case_id']
        print(f"\n   Case {case_id}:")
        
        try:
            # Try with lenient quality settings
            signal, fs = load_channel(
                case_id, 
                'PLETH', 
                use_cache=False, 
                duration_sec=10,
                allow_poor_quality=True  # KEY: Allow poor quality
            )
            
            # Check signal statistics
            import numpy as np
            valid_ratio = np.mean(~np.isnan(signal))
            
            print(f"      ✓ Loaded {len(signal)} samples at {fs} Hz")
            print(f"      Valid data: {valid_ratio:.1%}")
            print(f"      Signal range: [{np.nanmin(signal):.2f}, {np.nanmax(signal):.2f}]")
            
            if valid_ratio < 0.5:
                print(f"      ⚠️ Poor quality signal accepted")
                partial += 1
            else:
                successful += 1
                
        except Exception as e:
            error_msg = str(e)
            if "too poor" in error_msg.lower():
                print(f"      ✗ Signal quality too poor to use (<10% valid)")
            else:
                print(f"      ✗ Failed: {error_msg[:100]}")
            failed += 1
    
    print(f"\n" + "="*60)
    print("RESULTS:")
    print(f"  ✓ Good quality: {successful}/10 cases")
    print(f"  ⚠️ Poor quality (salvaged): {partial}/10 cases")
    print(f"  ✗ Failed: {failed}/10 cases")
    
    total_usable = successful + partial
    if total_usable > 0:
        print(f"\n✅ Successfully loaded {total_usable}/10 cases (with lenient settings)")
        print("   VitalDB loader is working with quality handling!")
    else:
        print("\n⚠️ No cases loaded - check VitalDB connection or try different cases")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
