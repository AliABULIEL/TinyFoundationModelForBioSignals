#!/usr/bin/env python3
"""
Clear VitalDB cache to force fresh data loading.
Use this after bug fixes to ensure old cached data isn't used.
"""

import shutil
from pathlib import Path

cache_dirs = [
    Path('data/vitaldb_cache'),
    Path('data/cache'),
    Path('/tmp/vitaldb_cache'),
]

print("Clearing VitalDB cache directories...")
for cache_dir in cache_dirs:
    if cache_dir.exists():
        print(f"  Removing {cache_dir}...")
        try:
            shutil.rmtree(cache_dir)
            print(f"  ✓ Cleared {cache_dir}")
        except Exception as e:
            print(f"  ✗ Failed to clear {cache_dir}: {e}")
    else:
        print(f"  ℹ️  {cache_dir} doesn't exist")

print("\n✓ Cache cleared! Now run your data preparation scripts.")
print("\nNext steps:")
print("  1. python scripts/diagnose_windows.py --case-id 440 --channel PPG")
print("  2. python scripts/diagnose_windows.py --case-id 440 --channel ECG")
print("  3. python scripts/prepare_all_data.py --mode fasttrack --dataset vitaldb")
