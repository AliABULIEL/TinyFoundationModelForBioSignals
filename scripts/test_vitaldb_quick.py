#!/usr/bin/env python3
"""Quick test to verify VitalDB is loading real data."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt


def test_vitaldb_api():
    """Test VitalDB API directly."""
    print("Testing VitalDB API directly...")
    
    try:
        import vitaldb
        print("✓ VitalDB package imported")
        
        # Find some cases with PPG
        cases = vitaldb.find_cases('PLETH')
        print(f"✓ Found {len(cases)} cases with PLETH")
        
        if len(cases) > 0:
            # Load first case
            case_id = cases[0]
            print(f"\nLoading case {case_id}...")
            
            # Try different loading methods
            # Method 1: Direct load_case
            try:
                data = vitaldb.load_case(case_id, ['SNUADC/PLETH'])
                if data is not None and len(data) > 0:
                    print(f"  ✓ Loaded via SNUADC/PLETH: {len(data)} samples")
                    print(f"    Data type: {type(data)}, shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")
                    if isinstance(data, np.ndarray):
                        non_nan = data[~np.isnan(data)]
                        if len(non_nan) > 0:
                            print(f"    Valid samples: {len(non_nan)}/{len(data)}")
                            print(f"    Range: [{np.min(non_nan):.3f}, {np.max(non_nan):.3f}]")
            except Exception as e:
                print(f"  ✗ SNUADC/PLETH failed: {e}")
                
            # Method 2: Try Solar8000
            try:
                data = vitaldb.load_case(case_id, ['Solar8000/PLETH'])
                if data is not None and len(data) > 0:
                    print(f"  ✓ Loaded via Solar8000/PLETH: {len(data)} samples")
            except:
                pass
                
            # Method 3: Use VitalFile
            try:
                vf = vitaldb.VitalFile(case_id)
                tracks = vf.get_track_names()
                print(f"\n  Available tracks for case {case_id}:")
                pleth_tracks = [t for t in tracks if 'PLETH' in t.upper()]
                ecg_tracks = [t for t in tracks if 'ECG' in t.upper()]
                print(f"    PPG tracks: {pleth_tracks}")
                print(f"    ECG tracks: {ecg_tracks}")
                
                # Try to load PLETH
                for track in pleth_tracks:
                    try:
                        data = vf.get_samples(track, 0, 10)  # First 10 seconds
                        if data is not None and len(data) > 0:
                            print(f"    ✓ {track}: Got {len(data)} samples")
                            break
                    except:
                        pass
                        
            except Exception as e:
                print(f"  VitalFile method error: {e}")
                
    except ImportError:
        print("✗ VitalDB not installed. Install with: pip install vitaldb")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
        
    return True


def test_our_loader():
    """Test our vitaldb_loader module."""
    print("\n" + "=" * 60)
    print("Testing our VitalDB loader...")
    
    try:
        from src.data.vitaldb_loader import load_channel, find_cases_with_track
        
        # Find cases
        cases = find_cases_with_track('PLETH', max_cases=3)
        print(f"Found {len(cases)} cases")
        
        if len(cases) > 0:
            case_id = cases[0]
            print(f"\nLoading case {case_id} with our loader...")
            
            signal, fs = load_channel(
                case_id=str(case_id),
                channel='PLETH',
                use_cache=False,  # Force fresh load
                auto_fix_alternating=True
            )
            
            print(f"✓ Loaded: {len(signal)} samples @ {fs}Hz")
            print(f"  Range: [{np.min(signal):.3f}, {np.max(signal):.3f}]")
            print(f"  Mean: {np.mean(signal):.3f}, Std: {np.std(signal):.3f}")
            
            # Quick plot
            if len(signal) > 1000:
                plt.figure(figsize=(12, 4))
                plt.plot(signal[:1000], 'b-', alpha=0.7)
                plt.title(f'First 1000 samples from case {case_id}')
                plt.xlabel('Sample')
                plt.ylabel('Amplitude')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig('vitaldb_quick_test.png')
                print(f"  ✓ Saved plot to vitaldb_quick_test.png")
                plt.close()
                
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Quick VitalDB Test")
    print("=" * 60)
    
    # Test API
    api_ok = test_vitaldb_api()
    
    # Test our loader
    loader_ok = test_our_loader()
    
    print("\n" + "=" * 60)
    if api_ok and loader_ok:
        print("✅ All tests passed! VitalDB is working.")
    else:
        print("❌ Some tests failed. Check output above.")
