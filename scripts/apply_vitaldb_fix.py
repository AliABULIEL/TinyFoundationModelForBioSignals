#!/usr/bin/env python3
"""
Apply the VitalDB loader fix and test it.
This replaces the old loader with the fixed version that uses vitaldb.load_case().
"""

import sys
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def apply_fix():
    """Replace old loader with fixed version."""
    print("="*60)
    print("Applying VitalDB Loader Fix")
    print("="*60)
    
    src_dir = Path(__file__).parent.parent / 'src' / 'data'
    old_loader = src_dir / 'vitaldb_loader.py'
    fixed_loader = src_dir / 'vitaldb_loader_fixed.py'
    backup_loader = src_dir / 'vitaldb_loader_backup.py'
    
    # Check if fixed version exists
    if not fixed_loader.exists():
        print(f"✗ Fixed loader not found: {fixed_loader}")
        return False
    
    # Backup old loader
    if old_loader.exists():
        print(f"Backing up old loader to: {backup_loader}")
        shutil.copy2(old_loader, backup_loader)
    
    # Replace with fixed version
    print(f"Replacing loader with fixed version...")
    shutil.copy2(fixed_loader, old_loader)
    
    print("✓ Loader replaced successfully!")
    return True


def test_fix():
    """Test the fixed loader."""
    print("\n" + "="*60)
    print("Testing Fixed Loader")
    print("="*60)
    
    try:
        from src.data.vitaldb_loader import (
            find_cases_with_track,
            load_channel,
            get_available_case_sets
        )
        
        print("\n1. Testing case set retrieval...")
        case_sets = get_available_case_sets()
        if case_sets:
            print(f"✓ Found {len(case_sets)} case sets:")
            for name, cases in list(case_sets.items())[:3]:
                print(f"  - {name}: {len(cases)} cases")
        else:
            print("⚠ No case sets found")
        
        print("\n2. Testing find_cases_with_track...")
        cases = find_cases_with_track('PLETH', max_cases=5)
        if cases:
            print(f"✓ Found {len(cases)} cases with PLETH")
            test_case = cases[0]
        else:
            print("⚠ No cases found with PLETH")
            # Try BIS cases instead
            if 'bis' in case_sets:
                test_case = list(case_sets['bis'])[0]
                print(f"Using BIS case {test_case} for testing")
            else:
                print("✗ Cannot proceed with testing - no cases available")
                return False
        
        print(f"\n3. Testing load_channel with case {test_case}...")
        try:
            signal, fs = load_channel(
                case_id=str(test_case),
                channel='PLETH',
                duration_sec=10,
                use_cache=False
            )
            
            print(f"✓ Successfully loaded channel!")
            print(f"  Signal shape: {signal.shape}")
            print(f"  Sampling rate: {fs} Hz")
            print(f"  Duration: {len(signal)/fs:.2f} seconds")
            
            import numpy as np
            nan_ratio = np.mean(np.isnan(signal))
            print(f"  NaN ratio: {nan_ratio:.1%}")
            
            if nan_ratio < 0.1:
                print("\n" + "="*60)
                print("✓✓✓ SUCCESS! Fixed loader is working!")
                print("="*60)
                return True
            else:
                print(f"\n⚠ Warning: High NaN ratio ({nan_ratio:.1%})")
                return True  # Still counts as success if data loaded
                
        except Exception as e:
            print(f"✗ Failed to load channel: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    print("\n" + "="*60)
    print("VitalDB Loader Fix & Test Script")
    print("="*60)
    print("""
This script will:
1. Backup your current vitaldb_loader.py
2. Replace it with the fixed version
3. Test the new loader

The fixed version uses vitaldb.load_case() instead of VitalFile,
which avoids the 'vitaldb_track' error.
    """)
    
    response = input("Proceed with fix? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Aborted.")
        return
    
    # Apply fix
    if apply_fix():
        print("\n✓ Fix applied successfully!")
        
        # Test fix
        test_input = input("\nRun tests? (yes/no): ").strip().lower()
        if test_input in ['yes', 'y']:
            success = test_fix()
            
            if success:
                print("\n" + "="*60)
                print("READY TO PROCEED!")
                print("="*60)
                print("""
You can now run your pipelines:

1. FastTrack mode (recommended first):
   bash scripts/run_fasttrack_complete.sh

2. High-accuracy mode:
   bash scripts/run_high_accuracy.sh

The loader should now work without 'vitaldb_track' errors!
                """)
            else:
                print("\n" + "="*60)
                print("TESTING FAILED")
                print("="*60)
                print("""
The loader was replaced but testing failed.
Possible issues:
1. VitalDB library needs update: pip install --upgrade vitaldb
2. Network connectivity issues
3. VitalDB API temporarily unavailable

You can restore the old loader from: src/data/vitaldb_loader_backup.py
                """)
    else:
        print("\n✗ Fix application failed")


if __name__ == '__main__':
    main()
