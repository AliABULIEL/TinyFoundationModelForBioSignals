#!/usr/bin/env python3
"""
Fixed VitalDB verification script with proper type handling
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import ssl
import urllib.request
import certifi
import warnings

print("="*70)
print("VITALDB DATA VERIFICATION (FIXED)")
print("="*70)

# Fix SSL certificate issue
def setup_ssl():
    """Configure SSL context for VitalDB downloads."""
    try:
        # Create unverified context for testing
        ssl._create_default_https_context = ssl._create_unverified_context
        print("   ✓ SSL context configured (unverified mode for testing)")
        return True
    except Exception as e:
        print(f"   ⚠ SSL setup warning: {e}")
        return False

# Helper function to safely convert VitalDB data to numeric
def safe_to_numeric(data):
    """Convert data to numeric numpy array, handling VitalDB's object arrays."""
    if data is None:
        return None
    
    try:
        # Convert to numpy array if needed
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        # If already numeric, just ensure float64
        if data.dtype in [np.float32, np.float64, np.int32, np.int64]:
            return data.astype(np.float64)
        
        # Handle object arrays (common VitalDB issue)
        if data.dtype == object:
            numeric_data = []
            for item in data.flat:
                try:
                    if isinstance(item, (list, np.ndarray)):
                        if len(item) > 0:
                            numeric_data.append(float(item[0]))
                        else:
                            numeric_data.append(np.nan)
                    else:
                        val = float(item) if item is not None else np.nan
                        numeric_data.append(val)
                except (TypeError, ValueError):
                    numeric_data.append(np.nan)
            
            return np.array(numeric_data, dtype=np.float64)
        
        # Try direct conversion
        return data.astype(np.float64)
        
    except Exception as e:
        print(f"   Warning: Could not convert to numeric: {e}")
        return None

# 1. Setup SSL first
print("\n1. Setting up SSL for VitalDB access...")
ssl_ok = setup_ssl()

# 2. Import VitalDB
print("\n2. Checking VitalDB installation...")
try:
    import vitaldb
    print("   ✓ VitalDB is installed")
    
    # Check for the case ID sets
    print("   ✓ Pre-filtered case sets available:")
    case_sets = {}
    if hasattr(vitaldb, 'caseids_bis'):
        case_sets['BIS'] = vitaldb.caseids_bis
        print(f"     - BIS cases (BIS > 70): {len(vitaldb.caseids_bis)} cases")
    if hasattr(vitaldb, 'caseids_des'):
        case_sets['Desflurane'] = vitaldb.caseids_des
        print(f"     - Desflurane cases: {len(vitaldb.caseids_des)} cases")
    if hasattr(vitaldb, 'caseids_sevo'):
        case_sets['Sevoflurane'] = vitaldb.caseids_sevo
        print(f"     - Sevoflurane cases: {len(vitaldb.caseids_sevo)} cases")
    if hasattr(vitaldb, 'caseids_rft20'):
        case_sets['Remifentanil'] = vitaldb.caseids_rft20
        print(f"     - Remifentanil cases: {len(vitaldb.caseids_rft20)} cases")
    if hasattr(vitaldb, 'caseids_ppf'):
        case_sets['Propofol'] = vitaldb.caseids_ppf
        print(f"     - Propofol cases: {len(vitaldb.caseids_ppf)} cases")
    if hasattr(vitaldb, 'caseids_tiva'):
        case_sets['TIVA'] = vitaldb.caseids_tiva
        print(f"     - TIVA cases: {len(vitaldb.caseids_tiva)} cases")
    
except ImportError as e:
    print(f"   ✗ VitalDB not installed: {e}")
    print("   Install with: pip install vitaldb")
    sys.exit(1)

# 3. Test metadata functions
print("\n3. Testing VitalDB metadata functions...")
recs = None
try:
    # Try to get case records
    print("   Loading case records...")
    
    # Method 1: Using vital_recs
    if hasattr(vitaldb, 'vital_recs'):
        recs = vitaldb.vital_recs('https://api.vitaldb.net/cases')
    else:
        # Method 2: Direct download
        recs = pd.read_csv('https://api.vitaldb.net/cases')
    
    print(f"   ✓ Found {len(recs)} total cases in VitalDB")
    
    if not recs.empty:
        print(f"   Columns available: {', '.join(list(recs.columns)[:10])}")
        sample = recs.iloc[0]
        print(f"   Sample case info (ID {sample.get('caseid', 'unknown')}):")
        for col in ['age', 'sex', 'height', 'weight', 'asa', 'emop']:
            if col in sample:
                print(f"     - {col}: {sample[col]}")
                
except Exception as e:
    print(f"   ⚠ Could not load case records: {e}")

# 4. Test loading actual waveform data with type fixing
print("\n4. Testing waveform data loading with type fixes...")
test_cases = list(vitaldb.caseids_bis)[:5] if hasattr(vitaldb, 'caseids_bis') else [1, 2, 3, 4, 5]
successful_loads = 0
failed_cases = []

for case_id in test_cases:
    print(f"\n   Case {case_id}:")
    
    try:
        # Load case with VitalFile
        vf = vitaldb.VitalFile(case_id)
        
        # Get available track names
        tracks = vf.get_track_names()
        print(f"   ✓ Found {len(tracks)} tracks")
        
        # Look for common biosignals
        ppg_track = None
        ecg_track = None
        for track in tracks:
            track_upper = track.upper()
            if not ppg_track and ('PLETH' in track_upper or 'PPG' in track_upper):
                ppg_track = track
            if not ecg_track and 'ECG' in track_upper:
                ecg_track = track
        
        # Try to load PPG data first (most common)
        test_track = ppg_track or ecg_track or (tracks[0] if tracks else None)
        
        if test_track:
            print(f"   Loading 10 seconds of '{test_track}'...")
            
            # Load data
            data = vf.to_numpy([test_track], 0, 10)  # Load 10 seconds
            
            if data is not None and len(data) > 0:
                print(f"   Raw data shape: {data.shape}, dtype: {data.dtype}")
                
                # CRITICAL FIX: Convert to numeric
                data = safe_to_numeric(data)
                
                if data is not None:
                    # Handle 2D arrays
                    if data.ndim == 2:
                        data = data[:, 0]
                    
                    print(f"   ✓ Converted to numeric: {len(data)} samples")
                    
                    # Calculate statistics safely
                    valid_data = data[~np.isnan(data)]
                    if len(valid_data) > 0:
                        mean_val = np.mean(valid_data)
                        std_val = np.std(valid_data)
                        nan_ratio = np.mean(np.isnan(data))
                        
                        print(f"   Data stats: mean={mean_val:.3f}, std={std_val:.3f}, NaN ratio={nan_ratio:.2%}")
                        
                        if nan_ratio < 0.5:  # Less than 50% NaN
                            successful_loads += 1
                            print(f"   ✓ Successfully loaded and processed data")
                        else:
                            print(f"   ⚠ Too many NaN values")
                    else:
                        print(f"   ⚠ All data is NaN")
                else:
                    print(f"   ⚠ Could not convert data to numeric")
            else:
                print(f"   ⚠ No data returned")
        else:
            print(f"   ⚠ No tracks available")
        
    except Exception as e:
        print(f"   ✗ Error loading case {case_id}: {e}")
        failed_cases.append(case_id)

# 5. Test with our fixed loader
print("\n5. Testing with fixed VitalDB loader...")
try:
    # Import our fixed loader
    from src.data.vitaldb_loader_fixed import load_channel, list_cases, get_available_case_sets
    
    print("   ✓ Fixed loader imported successfully")
    
    # Get available case sets
    available_sets = get_available_case_sets()
    if available_sets:
        print(f"   Available case sets: {', '.join(available_sets.keys())}")
    
    # Try loading with fixed loader
    test_case = test_cases[0] if test_cases else 1
    print(f"\n   Testing fixed loader with case {test_case}...")
    
    try:
        signal, fs = load_channel(str(test_case), 'PLETH', use_cache=False)
        print(f"   ✓ Fixed loader succeeded: {len(signal)} samples at {fs} Hz")
        
        # Check signal quality
        valid_ratio = np.mean(~np.isnan(signal))
        print(f"   Signal quality: {valid_ratio:.1%} valid samples")
        
    except Exception as e:
        print(f"   Fixed loader error: {e}")
        
except ImportError as e:
    print(f"   Could not import fixed loader: {e}")

# 6. Summary and recommendations
print("\n" + "="*70)
print("VERIFICATION SUMMARY")
print("="*70)

print(f"\n✓ VitalDB is installed")
print(f"{'✓' if ssl_ok else '⚠'} SSL configured")
print(f"✓ {len(test_cases)} test cases checked")
print(f"{'✓' if successful_loads > 0 else '✗'} Successfully loaded data from {successful_loads}/{len(test_cases)} cases")

if failed_cases:
    print(f"\n⚠ Failed cases: {failed_cases}")

if successful_loads < len(test_cases):
    print("\n📋 RECOMMENDATIONS:")
    print("1. Use the fixed loader (vitaldb_loader_fixed.py) which handles type conversion")
    print("2. The main issue is VitalDB returns object arrays that need conversion")
    print("3. Always use safe_to_numeric() before any numpy operations")
    print("4. Check for NaN values after loading")
else:
    print("\n✓ All tests passed! VitalDB data loading is working correctly")

print("\n" + "="*70)
print("EXAMPLE USAGE WITH FIXED LOADER")
print("="*70)

print("""
from src.data.vitaldb_loader_fixed import load_channel, list_cases

# Load PPG data from a case
signal, fs = load_channel('1', 'PLETH')
print(f"Loaded {len(signal)} samples at {fs} Hz")

# List available cases
cases = list_cases(
    required_channels=['PLETH'],
    case_set='bis',  # Use high-quality BIS cases
    max_cases=100
)
print(f"Found {len(cases)} cases with PPG data")
""")

print("="*70)
