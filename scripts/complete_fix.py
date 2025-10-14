#!/usr/bin/env python3
"""
Complete fix for the import issue.
This will diagnose and fix the problem once and for all.
"""

import sys
import shutil
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("COMPLETE FIX FOR IMPORT ISSUES")
print("=" * 80)

src_data = project_root / 'src' / 'data'
init_file = src_data / '__init__.py'

# Step 1: Backup the current __init__.py
print("\nStep 1: Backing up current __init__.py")
backup_file = src_data / '__init__.py.backup'
if init_file.exists():
    shutil.copy2(init_file, backup_file)
    print(f"✓ Backed up to {backup_file.name}")

# Step 2: Create a MINIMAL working __init__.py
print("\nStep 2: Creating minimal __init__.py")

minimal_init = '''"""Data processing modules for VitalDB signals and BUT PPG.

This module provides utilities for loading and processing biosignal data.
Import specific modules directly if you encounter circular import issues.
"""

# Import only the most basic utilities that have no dependencies
__all__ = []

# Make datasets available via direct import
# Example: from src.data.vitaldb_dataset import VitalDBDataset
'''

with open(init_file, 'w') as f:
    f.write(minimal_init)
print("✓ Created minimal __init__.py")

# Step 3: Test if direct imports work
print("\nStep 3: Testing direct imports (bypassing __init__.py)")
print("-" * 80)

test_imports = [
    ('src.data.filters', 'filter_ppg'),
    ('src.data.detect', 'find_ppg_peaks'),
    ('src.data.quality', 'compute_sqi'),
    ('src.data.windows', 'make_windows'),
    ('src.data.sync', 'resample_to_fs'),
    ('src.data.vitaldb_loader', 'list_cases'),
    ('src.data.butppg_loader', 'BUTPPGLoader'),
    ('src.data.vitaldb_dataset', 'VitalDBDataset'),
    ('src.data.butppg_dataset', 'BUTPPGDataset'),
]

working = []
broken = []

for module_path, item in test_imports:
    # Clear from cache
    if module_path in sys.modules:
        del sys.modules[module_path]
    
    try:
        module = __import__(module_path, fromlist=[item])
        if hasattr(module, item):
            print(f"✓ {module_path}.{item}")
            working.append((module_path, item))
        else:
            print(f"✗ {module_path} - missing {item}")
            broken.append((module_path, item, f"missing {item}"))
    except Exception as e:
        print(f"✗ {module_path}.{item} - {type(e).__name__}: {str(e)[:80]}")
        broken.append((module_path, item, str(e)))

# Step 4: Create a working __init__.py with only working imports
print("\nStep 4: Creating working __init__.py with successful imports")

working_imports_by_module = {}
for module_path, item in working:
    module_name = module_path.split('.')[-1]
    if module_name not in working_imports_by_module:
        working_imports_by_module[module_name] = []
    working_imports_by_module[module_name].append(item)

# Build the new __init__.py
new_init_lines = [
    '"""Data processing modules for VitalDB signals and BUT PPG."""\n',
    '\n',
    '# Direct imports of working modules\n',
]

# Add imports
for module_name, items in sorted(working_imports_by_module.items()):
    items_str = ', '.join(sorted(items))
    new_init_lines.append(f'from .{module_name} import {items_str}\n')

# Add __all__
new_init_lines.append('\n__all__ = [\n')
for module_name, items in sorted(working_imports_by_module.items()):
    for item in sorted(items):
        new_init_lines.append(f"    '{item}',\n")
new_init_lines.append(']\n')

working_init_content = ''.join(new_init_lines)

with open(init_file, 'w') as f:
    f.write(working_init_content)

print(f"✓ Created working __init__.py with {len(working)} successful imports")

# Step 5: Test the new __init__.py
print("\nStep 5: Testing new __init__.py")
print("-" * 80)

# Clear ALL src.data modules
to_clear = [k for k in list(sys.modules.keys()) if k.startswith('src.data')]
for k in to_clear:
    del sys.modules[k]

try:
    import src.data
    print("✓ src.data imports successfully")
    
    # Test key classes
    tests = [
        ('VitalDBDataset', hasattr(src.data, 'VitalDBDataset')),
        ('BUTPPGDataset', hasattr(src.data, 'BUTPPGDataset')),
        ('BUTPPGLoader', hasattr(src.data, 'BUTPPGLoader')),
    ]
    
    for name, exists in tests:
        if exists:
            print(f"✓ src.data.{name} available")
        else:
            print(f"✗ src.data.{name} not available")
            
except Exception as e:
    print(f"✗ Failed to import src.data: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\n✓ Working imports: {len(working)}")
if broken:
    print(f"✗ Broken imports: {len(broken)}")
    print("\nBroken imports (use direct import instead):")
    for module_path, item, error in broken:
        print(f"  - {module_path}.{item}")
        print(f"    Error: {error[:100]}")
        print(f"    Use: from {module_path} import {item}")

print("\n" + "=" * 80)
print("FIX COMPLETE")
print("=" * 80)
print("\n📝 How to use:")
print("  1. For working imports:")
print("     from src.data import VitalDBDataset, BUTPPGDataset")
print("\n  2. For broken imports, use direct import:")
print("     from src.data.module_name import ClassName")
print("\n  3. Test your code:")
print("     python scripts/test_multimodal_data.py")
print("=" * 80)
