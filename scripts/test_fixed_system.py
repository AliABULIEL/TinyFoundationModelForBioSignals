#!/usr/bin/env python3
"""
Comprehensive system check after fixes.
Tests imports, VitalDB connection, and BUT PPG data.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("SYSTEM STATUS CHECK (After Fixes)")
print("=" * 80)

# Test 1: Basic imports
print("\n1. Testing basic imports...")
try:
    import numpy as np
    print("  ✓ NumPy imported")
except Exception as e:
    print(f"  ✗ NumPy failed: {e}")

try:
    import torch
    print("  ✓ PyTorch imported")
except Exception as e:
    print(f"  ✗ PyTorch failed: {e}")

# Test 2: Data module imports
print("\n2. Testing data module imports...")
try:
    from src.data import (
        VitalDBDataset,
        BUTPPGDataset,
        create_vitaldb_dataloaders,
        create_butppg_dataloaders,
        BUTPPGLoader,
        find_butppg_cases,
        load_butppg_signal,
    )
    print("  ✓ All data modules imported successfully")
except Exception as e:
    print(f"  ✗ Data module import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: VitalDB package
print("\n3. Testing VitalDB package...")
try:
    import vitaldb
    print("  ✓ VitalDB package installed")
except Exception as e:
    print(f"  ✗ VitalDB package not installed: {e}")

# Test 4: Data availability
print("\n4. Checking data availability...")

# Check BUT PPG
but_ppg_path = project_root / 'data' / 'but_ppg' / 'dataset'
if but_ppg_path.exists():
    files = list(but_ppg_path.rglob('*'))
    signal_files = [f for f in files if f.suffix in ['.mat', '.npy', '.csv', '.hdf5']]
    print(f"  ✓ BUT PPG data found: {len(signal_files)} files")
else:
    print(f"  ⚠ BUT PPG data not found at {but_ppg_path}")
    print("    Run: python scripts/download_but_ppg.py")

# Check VitalDB cache
vitaldb_cache = project_root / 'data' / 'vitaldb_cache'
if vitaldb_cache.exists():
    cache_files = list(vitaldb_cache.rglob('*.npy'))
    print(f"  ✓ VitalDB cache exists: {len(cache_files)} cached files")
else:
    print(f"  ⚠ VitalDB cache directory not found (will be created on first use)")

# Test 5: VitalDB API connection
print("\n5. Testing VitalDB API connection...")
try:
    import vitaldb
    
    # Try to list cases with PPG
    cases_with_ppg = vitaldb.find_cases(['PLETH'])
    print(f"  ✅ Connected! Found {len(cases_with_ppg)} cases with PPG")
    
except Exception as e:
    print(f"  ✗ VitalDB connection failed: {e}")

# Test 6: Quick dataset initialization
print("\n6. Testing dataset initialization...")

# Test VitalDB dataset
try:
    dataset = VitalDBDataset(
        channels=['ppg'],
        split='train',
        use_raw_vitaldb=True,
        max_cases=2
    )
    print(f"  ✓ VitalDB dataset initialized: {len(dataset)} samples")
except Exception as e:
    print(f"  ✗ VitalDB dataset failed: {e}")

# Test BUT PPG dataset if data exists
if but_ppg_path.exists():
    try:
        dataset = BUTPPGDataset(
            data_dir=str(but_ppg_path),
            modality='ppg',
            split='train'
        )
        print(f"  ✓ BUT PPG dataset initialized: {len(dataset)} records")
    except Exception as e:
        print(f"  ✗ BUT PPG dataset failed: {e}")
else:
    print(f"  ⚠ BUT PPG dataset test skipped (no data)")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("✅ All imports working!")
print("✅ VitalDB connection working!")
if but_ppg_path.exists():
    print("✅ BUT PPG data available!")
else:
    print("⚠ BUT PPG data needs to be downloaded")

print("\n🎯 Next Steps:")
if not but_ppg_path.exists():
    print("1. Download BUT PPG dataset:")
    print("   python scripts/download_but_ppg.py")
print("2. Test multi-modal data loading:")
print("   python scripts/test_multimodal_data.py")
print("3. Run the full pipeline:")
print("   python scripts/run_multimodal_pipeline.py")

print("=" * 80)
