#!/usr/bin/env python3
"""Debug script to find import issues."""

import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("DEBUGGING IMPORT ERRORS")
print("=" * 80)

# Test individual module imports
modules_to_test = [
    ('src.data.filters', 'Testing filters module'),
    ('src.data.detect', 'Testing detect module'),
    ('src.data.quality', 'Testing quality module'),
    ('src.data.windows', 'Testing windows module'),
    ('src.data.sync', 'Testing sync module'),
    ('src.data.splits', 'Testing splits module'),
    ('src.data.vitaldb_loader', 'Testing vitaldb_loader module'),
    ('src.data.manifests', 'Testing manifests module'),
    ('src.data.butppg_loader', 'Testing butppg_loader module'),
    ('src.data.butppg_dataset', 'Testing butppg_dataset module'),
    ('src.data.vitaldb_dataset', 'Testing vitaldb_dataset module'),
    ('src.data.dataset_compatibility', 'Testing dataset_compatibility module'),
]

failed_modules = []

for module_name, description in modules_to_test:
    print(f"\n{description}...")
    try:
        module = __import__(module_name, fromlist=[''])
        print(f"✓ {module_name} imported successfully")
    except Exception as e:
        print(f"✗ {module_name} failed:")
        print(f"  Error: {e}")
        print("  Traceback:")
        traceback.print_exc()
        failed_modules.append((module_name, e))

print("\n" + "=" * 80)
print("TRYING FULL PACKAGE IMPORT")
print("=" * 80)

try:
    import src.data
    print("✓ Full src.data package imported successfully")
except Exception as e:
    print("✗ Full src.data import failed:")
    print(f"  Error: {e}")
    print("  Traceback:")
    traceback.print_exc()

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

if failed_modules:
    print(f"✗ {len(failed_modules)} modules failed to import:")
    for module_name, error in failed_modules:
        print(f"  - {module_name}: {error}")
else:
    print("✅ All modules imported successfully!")
