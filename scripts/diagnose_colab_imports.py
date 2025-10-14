#!/usr/bin/env python3
"""
Diagnostic script to identify import issues in Google Colab.
"""

import os
import sys
from pathlib import Path
import traceback

print("=" * 80)
print("COLAB IMPORT DIAGNOSTICS")
print("=" * 80)

# Determine if we're in Colab or local
is_colab = 'google.colab' in sys.modules
if is_colab:
    print("✓ Running in Google Colab")
    project_root = Path('/content/drive/MyDrive/TinyFoundationModelForBioSignals')
else:
    print("✓ Running locally")
    project_root = Path(__file__).parent.parent

print(f"Project root: {project_root}")

# Add to path
sys.path.insert(0, str(project_root))

# Step 1: Check if files exist
print("\n" + "=" * 80)
print("STEP 1: Checking File Existence")
print("=" * 80)

src_data_dir = project_root / 'src' / 'data'
print(f"\nChecking directory: {src_data_dir}")

if not src_data_dir.exists():
    print(f"✗ CRITICAL: {src_data_dir} does not exist!")
    sys.exit(1)
else:
    print(f"✓ Directory exists")

# List all Python files
py_files = list(src_data_dir.glob('*.py'))
print(f"\nPython files in src/data/ ({len(py_files)} files):")
for f in sorted(py_files):
    size = f.stat().st_size
    print(f"  {'✓' if size > 0 else '✗'} {f.name} ({size:,} bytes)")

# Check specific files
required_files = [
    'butppg_loader.py',
    'butppg_dataset.py',
    'vitaldb_loader.py',
    'vitaldb_dataset.py',
    '__init__.py',
]

print("\nRequired files:")
for filename in required_files:
    filepath = src_data_dir / filename
    if filepath.exists():
        size = filepath.stat().st_size
        print(f"  ✓ {filename} ({size:,} bytes)")
    else:
        print(f"  ✗ {filename} MISSING")

# Step 2: Check for syntax errors
print("\n" + "=" * 80)
print("STEP 2: Checking for Syntax Errors")
print("=" * 80)

import ast

for filename in required_files:
    if filename == '__init__.py':
        continue
    
    filepath = src_data_dir / filename
    if not filepath.exists():
        continue
    
    print(f"\nChecking {filename}...")
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        ast.parse(code)
        print(f"  ✓ No syntax errors")
    except SyntaxError as e:
        print(f"  ✗ SYNTAX ERROR at line {e.lineno}: {e.msg}")
    except Exception as e:
        print(f"  ✗ Error: {e}")

# Step 3: Test individual imports
print("\n" + "=" * 80)
print("STEP 3: Testing Individual Module Imports")
print("=" * 80)

modules = [
    'src.data.filters',
    'src.data.detect',
    'src.data.quality',
    'src.data.sync',
    'src.data.windows',
    'src.data.splits',
    'src.data.vitaldb_loader',
    'src.data.butppg_loader',  # This is the problematic one
    'src.data.butppg_dataset',
    'src.data.vitaldb_dataset',
]

failed_imports = []

for module_name in modules:
    print(f"\nImporting {module_name}...")
    try:
        module = __import__(module_name, fromlist=[''])
        print(f"  ✓ Success")
    except ModuleNotFoundError as e:
        print(f"  ✗ ModuleNotFoundError: {e}")
        failed_imports.append((module_name, 'ModuleNotFoundError', str(e)))
    except ImportError as e:
        print(f"  ✗ ImportError: {e}")
        failed_imports.append((module_name, 'ImportError', str(e)))
    except Exception as e:
        print(f"  ✗ {type(e).__name__}: {e}")
        print("\n  Full traceback:")
        traceback.print_exc()
        failed_imports.append((module_name, type(e).__name__, str(e)))

# Step 4: Check dependencies
print("\n" + "=" * 80)
print("STEP 4: Checking Dependencies")
print("=" * 80)

dependencies = [
    'numpy',
    'pandas',
    'scipy',
    'torch',
    'h5py',
]

print("\nRequired packages:")
for pkg in dependencies:
    try:
        __import__(pkg)
        print(f"  ✓ {pkg}")
    except ImportError:
        print(f"  ✗ {pkg} (MISSING)")

# Step 5: Try direct file import
print("\n" + "=" * 80)
print("STEP 5: Attempting Direct Import of butppg_loader")
print("=" * 80)

butppg_loader_path = src_data_dir / 'butppg_loader.py'

if butppg_loader_path.exists():
    print(f"\nFile exists: {butppg_loader_path}")
    print(f"Size: {butppg_loader_path.stat().st_size:,} bytes")
    
    # Read first 50 lines
    print("\nFirst 50 lines of butppg_loader.py:")
    print("-" * 80)
    with open(butppg_loader_path, 'r') as f:
        for i, line in enumerate(f, 1):
            if i > 50:
                break
            print(f"{i:3d}: {line.rstrip()}")
    print("-" * 80)
    
    # Try to import directly
    print("\nAttempting direct import...")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("butppg_loader", butppg_loader_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print("  ✓ Direct import successful")
    except Exception as e:
        print(f"  ✗ Direct import failed: {e}")
        print("\n  Full traceback:")
        traceback.print_exc()
else:
    print(f"✗ File does not exist: {butppg_loader_path}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

if failed_imports:
    print(f"\n✗ {len(failed_imports)} modules failed to import:")
    for module, error_type, error_msg in failed_imports:
        print(f"\n  Module: {module}")
        print(f"  Error: {error_type}")
        print(f"  Message: {error_msg}")
else:
    print("\n✅ All modules imported successfully!")

print("\n" + "=" * 80)
