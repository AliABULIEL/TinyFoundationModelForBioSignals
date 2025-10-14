#!/usr/bin/env python3
"""
Root cause analysis - trace the exact import failure.
"""

import sys
from pathlib import Path
import importlib.util
import ast

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("ROOT CAUSE ANALYSIS")
print("=" * 80)

src_data = project_root / 'src' / 'data'

# Step 1: Check all files exist
print("\nStep 1: File existence check")
print("-" * 80)
files = ['butppg_loader.py', 'quality.py', 'windows.py', 'detect.py', 'filters.py', 'sync.py', 'splits.py']
for fname in files:
    fpath = src_data / fname
    if fpath.exists():
        print(f"✓ {fname} ({fpath.stat().st_size:,} bytes)")
    else:
        print(f"✗ {fname} MISSING")

# Step 2: Parse butppg_loader.py to see what it imports
print("\nStep 2: Analyzing butppg_loader.py imports")
print("-" * 80)

butppg_loader_file = src_data / 'butppg_loader.py'
with open(butppg_loader_file, 'r') as f:
    tree = ast.parse(f.read(), filename='butppg_loader.py')

imports = []
for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom):
        module = node.module
        names = [alias.name for alias in node.names]
        imports.append((module, names))
        
print("Imports in butppg_loader.py:")
for module, names in imports:
    print(f"  from {module} import {', '.join(names)}")

# Step 3: Try to import each dependency individually
print("\nStep 3: Testing each dependency")
print("-" * 80)

# Test imports that butppg_loader needs
test_sequence = [
    ('src.data.detect', 'find_ppg_peaks'),
    ('src.data.quality', 'compute_sqi'),
    ('src.data.windows', 'make_windows'),
    ('src.data.windows', 'compute_normalization_stats'),
    ('src.data.windows', 'normalize_windows'),
    ('src.data.windows', 'NormalizationStats'),
]

failed = []
for module_name, item_name in test_sequence:
    try:
        # Clear cache
        if module_name in sys.modules:
            del sys.modules[module_name]
        
        module = __import__(module_name, fromlist=[item_name])
        if hasattr(module, item_name):
            print(f"✓ {module_name}.{item_name}")
        else:
            print(f"✗ {module_name}.{item_name} - exists but doesn't have {item_name}")
            failed.append(f"{module_name}.{item_name}")
    except Exception as e:
        print(f"✗ {module_name}.{item_name} - {type(e).__name__}: {e}")
        failed.append(f"{module_name}.{item_name}")

# Step 4: Try direct module import with detailed error
print("\nStep 4: Attempting direct import of butppg_loader module")
print("-" * 80)

try:
    spec = importlib.util.spec_from_file_location("butppg_loader", butppg_loader_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules['butppg_loader'] = module
    spec.loader.exec_module(module)
    print("✓ Direct import successful!")
except Exception as e:
    print(f"✗ Direct import failed: {type(e).__name__}")
    print(f"   {e}")
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()
    
# Step 5: Try importing via src.data
print("\nStep 5: Attempting import via src.data")
print("-" * 80)

try:
    # Clear all src.data modules from cache
    to_clear = [k for k in sys.modules.keys() if k.startswith('src.data')]
    for k in to_clear:
        del sys.modules[k]
    
    from src.data import butppg_loader
    print("✓ Import via src.data successful!")
except Exception as e:
    print(f"✗ Import via src.data failed: {type(e).__name__}")
    print(f"   {e}")
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)
if failed:
    print(f"\n✗ {len(failed)} dependencies failed:")
    for item in failed:
        print(f"  - {item}")
else:
    print("\n✓ All dependencies OK")
