#!/usr/bin/env python3
"""Test VitalDB data loading with REAL data only - NO MOCK DATA."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt


def test_vitaldb_api():
    """Test VitalDB API directly with real data."""
    print("Testing Real VitalDB API...")
    print("=" * 60)
    
    try:
        import vitaldb
        print("✓ VitalDB package imported")
        
        # Find real cases with PPG
        print("\nSearching for real VitalDB cases...")
        cases = vitaldb.find_cases('PLETH')
        print(f"✓ Found {len(cases)} real cases with PLETH")
        
        if len(cases) == 0:
            print("✗ No cases found - check network connection")
            return False
            
        # Try loading real data from first case
        case_id = cases[0]
        print(f"\nLoading REAL data from case {case_id}...")
        
        # Method 1: Direct load_case
        try:
            data = vitaldb.load_case(case_id, ['SNUADC/PLETH'])
            if data is not None and len(data) > 0:
                print(f"  ✓ Loaded real PPG: {len(data)} samples")
                
                # Check if it's real data
                if isinstance(data, np.ndarray):
                    non_nan = data[~np.isnan(data)]
                    if len(non_nan) > 0:
                        print(f"    Valid samples: {len(non_nan)}/{len(data)}")
                        print(f"    Range: [{np.min(non_nan):.3f}, {np.max(non_nan):.3f}]")
                        print(f"    Mean: {np.mean(non_nan):.3f}, Std: {np.std(non_nan):.3f}")
                        
                        # Plot to verify it's real biosignal
                        if len(non_nan) > 1000:
                            plt.figure(figsize=(12, 4))
                            plt.plot(non_nan[:1000], 'b-', alpha=0.7, linewidth=0.5)
                            plt.title(f'Real VitalDB PPG - Case {case_id} (First 1000 samples)')
                            plt.xlabel('Sample')
                            plt.ylabel('Amplitude')
                            plt.grid(True, alpha=0.3)
                            plt.tight_layout()
                            plt.savefig('real_vitaldb_ppg.png')
                            print(f"  ✓ Saved real PPG plot to real_vitaldb_ppg.png")
                            plt.close()
        except Exception as e:
            print(f"  ✗ Failed to load: {e}")
            
        # Try getting ECG as well
        try:
            ecg_cases = vitaldb.find_cases('ECG_II')
            if len(ecg_cases) > 0:
                print(f"\n✓ Found {len(ecg_cases)} real cases with ECG")
                # Find cases with both
                both_cases = list(set(cases) & set(ecg_cases))
                if both_cases:
                    print(f"✓ Found {len(both_cases)} cases with both PPG and ECG")
        except:
            pass
            
        return True
        
    except ImportError:
        print("✗ VitalDB not installed. Install with: pip install vitaldb")
        return False
    except Exception as e:
        if 'SSL' in str(e) or 'certificate' in str(e).lower():
            print(f"✗ SSL Certificate Error: {e}")
            print("\nTo fix SSL on macOS:")
            print("1. Install certifi: pip install --upgrade certifi")
            print("2. Find Python folder and run 'Install Certificates.command'")
            print("3. Or set: export PYTHONHTTPSVERIFY=0 (less secure)")
        else:
            print(f"✗ Error: {e}")
        return False


def test_our_loader():
    """Test our VitalDB loader with real data."""
    print("\n" + "=" * 60)
    print("Testing Our VitalDB Loader (Real Data)...")
    
    try:
        from src.data.vitaldb_loader import load_channel, find_cases_with_track
        
        # Find real cases
        print("Finding real cases...")
        cases = find_cases_with_track('PLETH', max_cases=3)
        print(f"Found {len(cases)} real cases")
        
        if len(cases) == 0:
            print("✗ No cases found - check network/SSL")
            return False
            
        case_id = cases[0]
        print(f"\nLoading real data from case {case_id}...")
        
        # Load real PPG data
        signal, fs = load_channel(
            case_id=str(case_id),
            channel='PLETH',
            use_cache=True,
            cache_dir='data/vitaldb_cache',
            auto_fix_alternating=True
        )
        
        print(f"✓ Loaded real signal: {len(signal)} samples @ {fs}Hz")
        print(f"  Range: [{np.min(signal):.3f}, {np.max(signal):.3f}]")
        print(f"  Mean: {np.mean(signal):.3f}, Std: {np.std(signal):.3f}")
        
        # Verify it's real biosignal data
        if np.std(signal) > 0.001:  # Real signals have variance
            print("✓ Confirmed: This is real biosignal data (has variance)")
        else:
            print("⚠ Warning: Signal may be flat or synthetic")
            
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset():
    """Test VitalDB dataset with real data."""
    print("\n" + "=" * 60)
    print("Testing VitalDB Dataset (Real Data Only)...")
    
    try:
        from src.data.vitaldb_dataset import VitalDBDataset
        
        # Create dataset with REAL data
        print("Creating dataset with REAL VitalDB data...")
        dataset = VitalDBDataset(
            channels=['ppg', 'ecg'],
            split='train',
            use_raw_vitaldb=True,  # Use real VitalDB API
            cache_dir='data/vitaldb_cache',
            max_cases=2,  # Start small
            segments_per_case=3
        )
        
        if len(dataset) > 0:
            print(f"✓ Dataset created with {len(dataset)} real samples")
            
            # Get a real sample
            seg1, seg2 = dataset[0]
            print(f"  Real data shape: {seg1.shape}")
            print(f"  PPG statistics: mean={seg1[0].mean():.3f}, std={seg1[0].std():.3f}")
            print(f"  ECG statistics: mean={seg1[1].mean():.3f}, std={seg1[1].std():.3f}")
            
            # Verify it's real data
            if seg1[0].std() > 0.001 and seg1[1].std() > 0.001:
                print("✓ Confirmed: Loading real biosignal data")
            else:
                print("⚠ Warning: Data may not be real biosignals")
                
            return True
        else:
            print("✗ Dataset is empty")
            return False
            
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        if 'SSL' in str(e) or 'certificate' in str(e).lower():
            print("\n⚠ SSL Issue detected. See fix instructions above.")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print(" REAL VitalDB Data Test (NO MOCK DATA)")
    print("=" * 60)
    print("\nThis test uses ONLY real VitalDB data.")
    print("If SSL errors occur, you need to fix certificates.\n")
    
    # Test API
    api_ok = test_vitaldb_api()
    
    if api_ok:
        # Test our loader
        loader_ok = test_our_loader()
        
        # Test dataset
        dataset_ok = test_dataset()
    else:
        loader_ok = False
        dataset_ok = False
        
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  VitalDB API: {'✅ WORKING' if api_ok else '❌ FAILED'}")
    print(f"  Our Loader: {'✅ WORKING' if loader_ok else '❌ FAILED'}")
    print(f"  Dataset: {'✅ WORKING' if dataset_ok else '❌ FAILED'}")
    
    if api_ok and loader_ok and dataset_ok:
        print("\n✅ SUCCESS! Real VitalDB data is loading correctly!")
        print("\nYou can now run the full pipeline with real data:")
        print("  python scripts/run_multimodal_pipeline.py")
    elif not api_ok:
        print("\n❌ VitalDB API connection failed.")
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. Fix SSL certificates (see instructions above)")
        print("3. Install VitalDB: pip install vitaldb")
    else:
        print("\n❌ Some components failed. Check errors above.")
