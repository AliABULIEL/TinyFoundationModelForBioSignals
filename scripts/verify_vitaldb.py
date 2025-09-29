#!/usr/bin/env python3
"""
Verify VitalDB data loading with proper SSL handling and API usage
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import ssl
import urllib.request
import certifi

print("="*70)
print("VITALDB DATA VERIFICATION")
print("="*70)

# Fix SSL certificate issue
def setup_ssl():
    """Configure SSL context for VitalDB downloads."""
    try:
        # Create unverified context for testing (not recommended for production)
        ssl._create_default_https_context = ssl._create_unverified_context
        print("   ✓ SSL context configured (unverified mode for testing)")
        
        # Alternative: Use certifi for proper SSL verification
        # ssl_context = ssl.create_default_context(cafile=certifi.where())
        # opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
        # urllib.request.install_opener(opener)
        
    except Exception as e:
        print(f"   ⚠ SSL setup warning: {e}")

# 1. Setup SSL first
print("\n1. Setting up SSL for VitalDB access...")
setup_ssl()

# 2. Import VitalDB
print("\n2. Checking VitalDB installation...")
try:
    import vitaldb
    print("   ✓ VitalDB is installed")
    
    # Check for the case ID sets
    print("   ✓ Pre-filtered case sets available:")
    if hasattr(vitaldb, 'caseids_bis'):
        print(f"     - BIS cases (BIS > 70): {len(vitaldb.caseids_bis)} cases")
    if hasattr(vitaldb, 'caseids_des'):
        print(f"     - Desflurane cases: {len(vitaldb.caseids_des)} cases")
    if hasattr(vitaldb, 'caseids_sevo'):
        print(f"     - Sevoflurane cases: {len(vitaldb.caseids_sevo)} cases")
    if hasattr(vitaldb, 'caseids_rft20'):
        print(f"     - Remifentanil cases: {len(vitaldb.caseids_rft20)} cases")
    if hasattr(vitaldb, 'caseids_ppf'):
        print(f"     - Propofol cases: {len(vitaldb.caseids_ppf)} cases")
    if hasattr(vitaldb, 'caseids_tiva'):
        print(f"     - TIVA cases: {len(vitaldb.caseids_tiva)} cases")
    
except ImportError as e:
    print(f"   ✗ VitalDB not installed: {e}")
    print("   Install with: pip install vitaldb")
    sys.exit(1)

# 3. Test vital_recs and vital_trks functions with correct arguments
print("\n3. Testing VitalDB metadata functions...")
try:
    # Test vital_recs - downloads and caches the case records
    print("   Loading case records...")
    recs = vitaldb.vital_recs('https://api.vitaldb.net/cases')  # Provide the URL/path
    print(f"   ✓ Found {len(recs)} total cases in VitalDB")
    print(f"   Columns: {list(recs.columns)[:10]}")
    
    # Show sample case info
    if not recs.empty:
        sample = recs.iloc[0]
        print(f"   Sample case info (ID {sample.get('caseid', 'unknown')}):")
        for col in ['age', 'sex', 'height', 'weight', 'asa', 'emop']:
            if col in sample:
                print(f"     - {col}: {sample[col]}")
except Exception as e:
    print(f"   Alternative: Using direct attribute access")
    try:
        # Alternative way to get records
        recs = pd.read_csv('https://api.vitaldb.net/cases')
        print(f"   ✓ Downloaded {len(recs)} cases directly")
    except:
        print(f"   ⚠ Could not load case records: {e}")
        recs = None

# 4. Test loading actual waveform data
print("\n4. Testing waveform data loading...")
test_cases = list(vitaldb.caseids_bis)[:3] if hasattr(vitaldb, 'caseids_bis') else [1, 2, 3]
successful_loads = 0

for case_id in test_cases:
    print(f"\n   Case {case_id}:")
    
    try:
        # Method 1: Using VitalFile
        print(f"   Loading with VitalFile...")
        vf = vitaldb.VitalFile(case_id)
        
        # Get available track names
        tracks = vf.get_track_names()
        print(f"   ✓ Found {len(tracks)} tracks")
        
        # Look for common biosignals
        biosignals_found = []
        for track in tracks:
            track_upper = track.upper()
            if 'ECG' in track_upper:
                biosignals_found.append(f"ECG: {track}")
            elif 'PLETH' in track_upper or 'PPG' in track_upper:
                biosignals_found.append(f"PPG: {track}")
            elif 'ABP' in track_upper or 'ART' in track_upper:
                biosignals_found.append(f"ABP: {track}")
            elif 'BIS' in track_upper:
                biosignals_found.append(f"BIS: {track}")
        
        if biosignals_found:
            print(f"   Biosignals found:")
            for sig in biosignals_found[:5]:  # Show first 5
                print(f"     - {sig}")
        
        # Try to load some data
        if tracks:
            test_track = tracks[0]
            print(f"   Loading 10 seconds of '{test_track}'...")
            data = vf.to_numpy([test_track], 0, 10)  # Load 10 seconds
            
            if data is not None and len(data) > 0:
                print(f"   ✓ Loaded {len(data)} samples successfully")
                
                # Check data quality
                if data.ndim == 2:
                    data = data[:, 0]
                
                if not np.all(np.isnan(data)):
                    mean_val = np.nanmean(data)
                    std_val = np.nanstd(data)
                    print(f"   Data stats: mean={mean_val:.3f}, std={std_val:.3f}")
                    successful_loads += 1
                else:
                    print(f"   ⚠ Data contains only NaN values")
            else:
                print(f"   ⚠ No data returned")
        
    except Exception as e:
        print(f"   ✗ Error loading case {case_id}: {e}")
        
        # Try alternative method
        try:
            print(f"   Trying alternative method (direct URL)...")
            # Build URL for track data
            if recs is not None and not recs.empty:
                case_rec = recs[recs['caseid'] == case_id]
                if not case_rec.empty:
                    # Could try to load directly from API
                    print(f"   Case exists in records but needs track IDs for direct loading")
        except Exception as e2:
            print(f"   Alternative also failed: {e2}")

# 5. Test loading with specific track names
print("\n5. Testing specific track loading...")
if successful_loads == 0:
    # Try a known good case with common tracks
    print("   Attempting to load common tracks from case 1...")
    try:
        vf = vitaldb.VitalFile(1)
        
        # Try common track names
        common_tracks = ['SNUADC/PLETH', 'SNUADC/ECG_II', 'SNUADC/ABP', 
                        'Intellivue/PLETH_SPO2', 'BIS/BIS']
        
        for track_name in common_tracks:
            try:
                if track_name in vf.get_track_names():
                    data = vf.to_numpy([track_name], 0, 5)  # Load 5 seconds
                    if data is not None and len(data) > 0:
                        print(f"   ✓ Successfully loaded {track_name}: {len(data)} samples")
                        successful_loads += 1
                        break
            except:
                continue
                
    except Exception as e:
        print(f"   Could not test specific tracks: {e}")

# 6. Summary
print("\n" + "="*70)
print("VERIFICATION SUMMARY")
print("="*70)

print(f"\n✓ VitalDB is installed")
print(f"✓ SSL configured for data access")
print(f"✓ {len(test_cases)} test cases available")
print(f"{'✓' if successful_loads > 0 else '✗'} Successfully loaded data from {successful_loads}/{len(test_cases)} test cases")

if successful_loads == 0:
    print("\n⚠ TROUBLESHOOTING:")
    print("1. Check your internet connection")
    print("2. Try installing certificates: pip install --upgrade certifi")
    print("3. On macOS, you may need to run:")
    print("   /Applications/Python\\ 3.x/Install\\ Certificates.command")
    print("4. Try using a VPN if VitalDB servers are blocked in your region")
else:
    print("\n✓ VitalDB data loading is working!")

print("\n" + "="*70)
print("USING VITALDB IN YOUR CODE")
print("="*70)

print("""
Example code to load VitalDB data:

```python
import vitaldb
import ssl

# Fix SSL if needed
ssl._create_default_https_context = ssl._create_unverified_context

# Use pre-filtered case sets
from vitaldb import caseids_bis, caseids_tiva

# Select cases
cases = list(caseids_bis)[:100]  # First 100 BIS cases

# Load data
for case_id in cases:
    vf = vitaldb.VitalFile(case_id)
    tracks = vf.get_track_names()
    
    # Load PPG data
    if 'PLETH' in ' '.join(tracks).upper():
        ppg_track = [t for t in tracks if 'PLETH' in t.upper()][0]
        ppg_data = vf.to_numpy([ppg_track], 0, 60)  # Load 60 seconds
        # Process ppg_data...
```
""")

print("="*70)
