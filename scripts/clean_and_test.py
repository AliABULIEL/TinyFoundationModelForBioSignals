#!/usr/bin/env python3
"""Clean pycache and test imports."""

import os
import shutil
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("CLEANING PYCACHE AND TESTING IMPORTS")
print("=" * 80)

# Step 1: Remove __pycache__ directories
print("\n1. Cleaning __pycache__ directories...")
pycache_dirs = list(project_root.rglob('__pycache__'))
for pycache_dir in pycache_dirs:
    try:
        shutil.rmtree(pycache_dir)
        print(f"  ✓ Removed {pycache_dir.relative_to(project_root)}")
    except Exception as e:
        print(f"  ✗ Failed to remove {pycache_dir}: {e}")

print(f"\n  Cleaned {len(pycache_dirs)} __pycache__ directories")

# Step 2: Test individual module imports
print("\n2. Testing module imports...")

modules_to_test = [
    'src.data.filters',
    'src.data.detect', 
    'src.data.quality',
    'src.data.windows',
    'src.data.sync',
    'src.data.splits',
    'src.data.vitaldb_loader',
    'src.data.manifests',
    'src.data.butppg_loader',
    'src.data.butppg_dataset',
    'src.data.vitaldb_dataset',
]

failed = []
for module_name in modules_to_test:
    try:
        __import__(module_name)
        print(f"  ✓ {module_name}")
    except Exception as e:
        print(f"  ✗ {module_name}: {e}")
        failed.append((module_name, str(e)))

# Step 3: Test full package import
print("\n3. Testing full src.data package import...")
try:
    import src.data
    print("  ✓ src.data package imported successfully")
    
    # Test key imports
    print("\n4. Verifying key components...")
    checks = [
        ('VitalDBDataset', hasattr(src.data, 'VitalDBDataset')),
        ('BUTPPGDataset', hasattr(src.data, 'BUTPPGDataset')),
        ('create_vitaldb_dataloaders', hasattr(src.data, 'create_vitaldb_dataloaders')),
        ('create_butppg_dataloaders', hasattr(src.data, 'create_butppg_dataloaders')),
        ('BUTPPGLoader', hasattr(src.data, 'BUTPPGLoader')),
        ('find_butppg_cases', hasattr(src.data, 'find_butppg_cases')),
    ]
    
    for name, exists in checks:
        if exists:
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} missing")
            failed.append((name, "Not found in src.data"))
            
except Exception as e:
    print(f"  ✗ Failed to import src.data: {e}")
    import traceback
    traceback.print_exc()
    failed.append(('src.data', str(e)))

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

if failed:
    print(f"✗ {len(failed)} issues found:")
    for name, error in failed:
        print(f"  - {name}: {error}")
    sys.exit(1)
else:
    print("✅ All imports working correctly!")
    sys.exit(0)
