#!/usr/bin/env python3
"""
Fix all scripts to use direct imports instead of src.data package imports.
This is the most reliable solution.
"""

import re
from pathlib import Path

project_root = Path(__file__).parent.parent

print("=" * 80)
print("FIXING ALL SCRIPTS TO USE DIRECT IMPORTS")
print("=" * 80)

# Define the import replacements
replacements = {
    'from src.data import VitalDBDataset': 'from src.data.vitaldb_dataset import VitalDBDataset',
    'from src.data import BUTPPGDataset': 'from src.data.butppg_dataset import BUTPPGDataset',
    'from src.data import create_vitaldb_dataloaders': 'from src.data.vitaldb_dataset import create_vitaldb_dataloaders',
    'from src.data import create_butppg_dataloaders': 'from src.data.butppg_dataset import create_butppg_dataloaders',
    'from src.data import BUTPPGLoader': 'from src.data.butppg_loader import BUTPPGLoader',
    'from src.data.vitaldb_dataset import VitalDBDataset': 'from src.data.vitaldb_dataset import VitalDBDataset',  # Already correct
    'from src.data.butppg_dataset import BUTPPGDataset': 'from src.data.butppg_dataset import BUTPPGDataset',  # Already correct
}

# Find all Python scripts in the scripts directory
scripts_dir = project_root / 'scripts'
py_files = list(scripts_dir.glob('*.py'))

print(f"\nFound {len(py_files)} Python files in scripts/")
print("-" * 80)

fixed_files = []

for py_file in py_files:
    try:
        with open(py_file, 'r') as f:
            content = f.read()
        
        original_content = content
        modified = False
        
        # Apply each replacement
        for old_import, new_import in replacements.items():
            if old_import in content and old_import != new_import:
                content = content.replace(old_import, new_import)
                modified = True
        
        if modified:
            # Backup original
            backup_file = py_file.with_suffix('.py.bak')
            with open(backup_file, 'w') as f:
                f.write(original_content)
            
            # Write fixed version
            with open(py_file, 'w') as f:
                f.write(content)
            
            print(f"✓ Fixed {py_file.name}")
            fixed_files.append(py_file.name)
        else:
            print(f"  {py_file.name} (no changes needed)")
            
    except Exception as e:
        print(f"✗ Error processing {py_file.name}: {e}")

# Create a working __init__.py
print("\n" + "-" * 80)
print("Creating minimal __init__.py that won't break imports")
print("-" * 80)

init_file = project_root / 'src' / 'data' / '__init__.py'

# Backup
backup_init = init_file.with_suffix('.py.original')
if init_file.exists() and not backup_init.exists():
    import shutil
    shutil.copy2(init_file, backup_init)
    print(f"✓ Backed up original to {backup_init.name}")

# Create minimal __init__.py
minimal_init = '''"""Data processing modules for VitalDB signals and BUT PPG.

IMPORTANT: Due to circular import issues, import modules directly:
    from src.data.vitaldb_dataset import VitalDBDataset
    from src.data.butppg_dataset import BUTPPGDataset
    from src.data.butppg_loader import BUTPPGLoader

Do NOT use: from src.data import VitalDBDataset
"""

# Leave empty to avoid circular import issues
# All imports should be done directly from submodules
__all__ = []

# Optional: Re-export if needed, but may cause circular imports
try:
    from .vitaldb_dataset import VitalDBDataset, create_vitaldb_dataloaders
    from .butppg_dataset import BUTPPGDataset, create_butppg_dataloaders
    from .butppg_loader import BUTPPGLoader, find_butppg_cases, load_butppg_signal
    __all__.extend([
        'VitalDBDataset', 'create_vitaldb_dataloaders',
        'BUTPPGDataset', 'create_butppg_dataloaders',
        'BUTPPGLoader', 'find_butppg_cases', 'load_butppg_signal'
    ])
except ImportError as e:
    # If imports fail, that's okay - use direct imports instead
    import warnings
    warnings.warn(f"Could not import all modules: {e}. Use direct imports from submodules.")
'''

with open(init_file, 'w') as f:
    f.write(minimal_init)
print("✓ Created minimal __init__.py")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

if fixed_files:
    print(f"\n✓ Fixed {len(fixed_files)} files:")
    for fname in fixed_files:
        print(f"  - {fname}")
    print("\n  Backups saved with .bak extension")
else:
    print("\n✓ All files already using correct imports")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)
print("\n1. Test the fixes:")
print("   python scripts/test_multimodal_data.py")
print("\n2. If you still have issues, try:")
print("   python scripts/find_root_cause.py")
print("\n3. For Colab, run this first:")
print("   !python3 scripts/complete_fix.py")
print("\n=" * 80)
