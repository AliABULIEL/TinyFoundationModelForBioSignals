#!/usr/bin/env python3
"""
MASTER FIX SCRIPT - Run this once to fix all import issues.
Works for both local and Google Colab environments.
"""

import sys
import shutil
from pathlib import Path
import os

# Detect environment
IS_COLAB = 'google.colab' in sys.modules

if IS_COLAB:
    print("Running in Google Colab")
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
    except:
        pass
    PROJECT_ROOT = Path('/content/drive/MyDrive/TinyFoundationModelForBioSignals')
else:
    print("Running locally")
    PROJECT_ROOT = Path(__file__).parent.parent

os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 80)
print("MASTER FIX SCRIPT - COMPLETE IMPORT REPAIR")
print("=" * 80)
print(f"Project root: {PROJECT_ROOT}")

# ============================================================================
# STEP 1: Clean cache
# ============================================================================
print("\n[1/5] Cleaning cache...")
pycache_dirs = list(PROJECT_ROOT.rglob('__pycache__'))
for pdir in pycache_dirs:
    try:
        shutil.rmtree(pdir)
        print(f"  ✓ Removed {pdir.relative_to(PROJECT_ROOT)}")
    except:
        pass
print(f"  Cleaned {len(pycache_dirs)} cache directories")

# ============================================================================
# STEP 2: Fix src/data/__init__.py
# ============================================================================
print("\n[2/5] Fixing src/data/__init__.py...")

init_file = PROJECT_ROOT / 'src' / 'data' / '__init__.py'

# Backup
backup = init_file.with_suffix('.py.backup_master')
if init_file.exists():
    shutil.copy2(init_file, backup)
    print(f"  ✓ Backed up to {backup.name}")

# Create working version
fixed_init = '''"""Data processing modules for VitalDB signals and BUT PPG."""

# Core utilities - these have no circular dependencies
try:
    from .detect import find_ecg_rpeaks, find_ppg_peaks
    from .filters import filter_ppg, filter_ecg, design_ppg_filter, design_ecg_filter
    from .quality import compute_sqi, ecg_sqi, ppg_ssqi
    from .sync import resample_to_fs, align_streams
    from .windows import (
        make_windows, 
        validate_cardiac_cycles,
        compute_normalization_stats,
        normalize_windows,
        NormalizationStats
    )
    from .splits import (
        make_patient_level_splits,
        verify_no_subject_leakage,
        save_splits,
        load_splits
    )
    from .vitaldb_loader import list_cases, load_channel
except ImportError as e:
    import warnings
    warnings.warn(f"Some core utilities could not be imported: {e}")

# Dataset classes - import these DIRECTLY to avoid circular imports
# Example: from src.data.vitaldb_dataset import VitalDBDataset
# DO NOT: from src.data import VitalDBDataset (may cause circular import)

__all__ = [
    'find_ecg_rpeaks', 'find_ppg_peaks',
    'filter_ppg', 'filter_ecg',
    'compute_sqi', 'ecg_sqi', 'ppg_ssqi',
    'resample_to_fs', 'align_streams',
    'make_windows', 'validate_cardiac_cycles',
    'list_cases', 'load_channel',
]

# Note: Import datasets directly:
# from src.data.vitaldb_dataset import VitalDBDataset
# from src.data.butppg_dataset import BUTPPGDataset
# from src.data.butppg_loader import BUTPPGLoader
'''

with open(init_file, 'w') as f:
    f.write(fixed_init)
print("  ✓ Created fixed __init__.py")

# ============================================================================
# STEP 3: Fix all test scripts
# ============================================================================
print("\n[3/5] Fixing test scripts...")

scripts_dir = PROJECT_ROOT / 'scripts'
test_files = [
    'test_multimodal_data.py',
    'test_after_fixes.py',
    'test_final.py',
    'test_dataloader_creation.py',
    'run_multimodal_pipeline.py',
]

for test_file in test_files:
    fpath = scripts_dir / test_file
    if not fpath.exists():
        continue
    
    try:
        with open(fpath, 'r') as f:
            content = f.read()
        
        # Replace package imports with direct imports
        replacements = [
            ('from src.data import VitalDBDataset', 
             'from src.data.vitaldb_dataset import VitalDBDataset'),
            ('from src.data import BUTPPGDataset', 
             'from src.data.butppg_dataset import BUTPPGDataset'),
            ('from src.data import create_vitaldb_dataloaders',
             'from src.data.vitaldb_dataset import create_vitaldb_dataloaders'),
            ('from src.data import create_butppg_dataloaders',
             'from src.data.butppg_dataset import create_butppg_dataloaders'),
            ('from src.data import BUTPPGLoader',
             'from src.data.butppg_loader import BUTPPGLoader'),
        ]
        
        modified = False
        for old, new in replacements:
            if old in content:
                content = content.replace(old, new)
                modified = True
        
        if modified:
            # Backup
            backup = fpath.with_suffix('.py.bak')
            with open(backup, 'w') as f:
                f.write(content)
            
            # Write fixed
            with open(fpath, 'w') as f:
                content_new = content.replace(old, new)
                f.write(content_new)
            
            print(f"  ✓ Fixed {test_file}")
    except Exception as e:
        print(f"  ✗ Error fixing {test_file}: {e}")

# ============================================================================
# STEP 4: Test imports
# ============================================================================
print("\n[4/5] Testing imports...")

# Clear cache
for key in list(sys.modules.keys()):
    if key.startswith('src.data'):
        del sys.modules[key]

tests = [
    ('src.data.vitaldb_dataset', 'VitalDBDataset'),
    ('src.data.butppg_dataset', 'BUTPPGDataset'),
    ('src.data.butppg_loader', 'BUTPPGLoader'),
]

all_pass = True
for module_name, class_name in tests:
    try:
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        print(f"  ✓ {module_name}.{class_name}")
    except Exception as e:
        print(f"  ✗ {module_name}.{class_name}: {e}")
        all_pass = False

# ============================================================================
# STEP 5: Create test script
# ============================================================================
print("\n[5/5] Creating verification test...")

test_script = PROJECT_ROOT / 'scripts' / 'verify_fix.py'
test_content = f'''#!/usr/bin/env python3
"""Verify that all imports work after fix."""

import sys
from pathlib import Path

project_root = Path(r"{PROJECT_ROOT}")
sys.path.insert(0, str(project_root))

print("Testing imports after fix...")
print("-" * 80)

try:
    from src.data.vitaldb_dataset import VitalDBDataset
    print("✓ VitalDBDataset")
except Exception as e:
    print(f"✗ VitalDBDataset: {{e}}")

try:
    from src.data.butppg_dataset import BUTPPGDataset
    print("✓ BUTPPGDataset")
except Exception as e:
    print(f"✗ BUTPPGDataset: {{e}}")

try:
    from src.data.butppg_loader import BUTPPGLoader
    print("✓ BUTPPGLoader")
except Exception as e:
    print(f"✗ BUTPPGLoader: {{e}}")

print("-" * 80)
print("Import test complete!")
'''

with open(test_script, 'w') as f:
    f.write(test_content)

print(f"  ✓ Created verify_fix.py")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("FIX COMPLETE!")
print("=" * 80)

if all_pass:
    print("\n✅ SUCCESS - All imports working!")
else:
    print("\n⚠ Some imports failed - but test scripts are fixed")
    print("   Try running: python scripts/verify_fix.py")

print("\n📝 Next steps:")
print("  1. Verify: python scripts/verify_fix.py")
print("  2. Test: python scripts/test_multimodal_data.py")
print("  3. Run: python scripts/run_multimodal_pipeline.py")

print("\n💡 Remember: Always use DIRECT imports:")
print("   ✓ from src.data.vitaldb_dataset import VitalDBDataset")
print("   ✗ from src.data import VitalDBDataset")

print("\n" + "=" * 80)
