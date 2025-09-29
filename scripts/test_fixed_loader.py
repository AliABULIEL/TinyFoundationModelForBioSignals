#!/usr/bin/env python3
"""
Simple test to verify the fixed VitalDB loader works
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test the fixed loader
print("Testing fixed VitalDB loader...")
print("="*50)

try:
    from src.data.vitaldb_loader import load_channel, list_cases, get_available_case_sets
    
    # 1. Check available case sets
    print("\n1. Available case sets:")
    case_sets = get_available_case_sets()
    for name, cases in case_sets.items():
        print(f"   - {name}: {len(cases)} cases")
    
    # 2. List some cases
    print("\n2. Getting cases from BIS set:")
    cases = list_cases(case_set='bis', max_cases=5)
    print(f"   Found {len(cases)} cases")
    
    # 3. Try loading data from first few cases
    print("\n3. Testing data loading:")
    successful = 0
    for case in cases[:5]:
        case_id = case['case_id']
        try:
            print(f"   Case {case_id}:")
            signal, fs = load_channel(case_id, 'PLETH', use_cache=False, duration_sec=10)
            print(f"      ✓ Loaded {len(signal)} samples at {fs} Hz")
            print(f"      Signal range: [{signal.min():.2f}, {signal.max():.2f}]")
            successful += 1
        except Exception as e:
            print(f"      ✗ Failed: {e}")
    
    print(f"\n✓ Successfully loaded {successful}/5 cases")
    
    if successful > 0:
        print("\n✅ VitalDB loader is working!")
    else:
        print("\n⚠️ No cases loaded successfully - check VitalDB connection")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
except Exception as e:
    print(f"✗ Unexpected error: {e}")

print("\n" + "="*50)
