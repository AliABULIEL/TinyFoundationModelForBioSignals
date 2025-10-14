#!/usr/bin/env python3
"""Debug import issues"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("DEBUGGING IMPORT ISSUES")
print("=" * 60)

print(f"\nPython path:")
for p in sys.path[:5]:
    print(f"  {p}")

print(f"\nProject root: {project_root}")
print(f"Project root exists: {project_root.exists()}")

# Check if src exists
src_dir = project_root / "src"
print(f"\nsrc directory: {src_dir}")
print(f"src exists: {src_dir.exists()}")

# Check if src/data exists
data_dir = src_dir / "data"
print(f"\nsrc/data directory: {data_dir}")
print(f"src/data exists: {data_dir.exists()}")

# Check if critical files exist
critical_files = [
    "src/__init__.py",
    "src/data/__init__.py",
    "src/data/butppg_loader.py",
    "src/data/butppg_dataset.py",
    "src/data/vitaldb_loader.py",
    "src/data/vitaldb_dataset.py",
]

print(f"\nChecking critical files:")
for f in critical_files:
    file_path = project_root / f
    exists = "✓" if file_path.exists() else "✗"
    print(f"  {exists} {f}")

# Try step-by-step imports
print("\n" + "=" * 60)
print("STEP-BY-STEP IMPORT TESTS")
print("=" * 60)

# Test 1: Import src
print("\n1. Importing src...")
try:
    import src
    print("   ✓ src imported successfully")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 2: Import src.data
print("\n2. Importing src.data...")
try:
    import src.data
    print("   ✓ src.data imported successfully")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Import butppg_loader directly
print("\n3. Importing src.data.butppg_loader...")
try:
    from src.data import butppg_loader
    print("   ✓ butppg_loader imported successfully")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Import vitaldb_loader
print("\n4. Importing src.data.vitaldb_loader...")
try:
    from src.data import vitaldb_loader
    print("   ✓ vitaldb_loader imported successfully")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Import BUTPPGDataset
print("\n5. Importing BUTPPGDataset...")
try:
    from src.data.butppg_dataset import BUTPPGDataset
    print("   ✓ BUTPPGDataset imported successfully")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Import VitalDBDataset
print("\n6. Importing VitalDBDataset...")
try:
    from src.data.vitaldb_dataset import VitalDBDataset
    print("   ✓ VitalDBDataset imported successfully")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
