#!/usr/bin/env python3
"""
Pre-flight Check for SSL Pre-training
======================================

Verifies that everything is ready for SSL pre-training:
- Data prepared and accessible
- Configurations correct
- Dependencies installed
- GPU available (if requested)
- Directory structure correct

Run this before: python scripts/pretrain_vitaldb_ssl.py

Author: Senior ML Engineer
"""

import sys
from pathlib import Path
import subprocess

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def check_mark(passed: bool) -> str:
    return "✅" if passed else "❌"

def print_section(title: str):
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")

def check_dependencies():
    """Check if required packages are installed."""
    print_section("1. Checking Dependencies")
    
    required = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'yaml': 'PyYAML',
        'tqdm': 'tqdm'
    }
    
    all_ok = True
    for module, name in required.items():
        try:
            __import__(module)
            print(f"  {check_mark(True)} {name}")
        except ImportError:
            print(f"  {check_mark(False)} {name} - NOT INSTALLED")
            all_ok = False
    
    # Check for optional TTM
    try:
        from tsfm_public import get_model
        print(f"  {check_mark(True)} tsfm_public (IBM TTM) - Real TTM available!")
    except ImportError:
        print(f"  ⚠️  tsfm_public (IBM TTM) - Not installed, will use fallback")
    
    return all_ok

def check_data():
    """Check if preprocessed data exists."""
    print_section("2. Checking Preprocessed Data")
    
    # Possible data locations
    locations = [
        Path('data/processed/vitaldb/windows/train/train_windows.npz'),
        Path('artifacts/raw_windows/train/train_windows.npz'),
        Path('data/vitaldb_windows/train/train_windows.npz')
    ]
    
    found = False
    data_path = None
    
    for loc in locations:
        if loc.exists():
            found = True
            data_path = loc
            break
    
    if found:
        print(f"  {check_mark(True)} Training data found: {data_path}")
        
        # Check data quality
        try:
            import numpy as np
            data = np.load(data_path)
            shape = data['data'].shape
            
            print(f"  {check_mark(True)} Shape: {shape}")
            
            # Verify shape
            if len(shape) == 3:
                N, dim1, dim2 = shape
                
                # Check if [N, C, T] or [N, T, C]
                if dim2 == 1250 or dim1 == 1250:
                    print(f"  {check_mark(True)} Time dimension: {max(dim1, dim2)} (expected 1250)")
                else:
                    print(f"  ⚠️  Time dimension: {max(dim1, dim2)} (expected 1250)")
                
                # Check channels
                channels = min(dim1, dim2)
                if channels == 2:
                    print(f"  {check_mark(True)} Channels: {channels} (PPG + ECG)")
                else:
                    print(f"  ⚠️  Channels: {channels} (expected 2 for VitalDB)")
                
                # Check size
                print(f"  {check_mark(True)} Windows: {N:,}")
                
                # Check for NaN/Inf
                has_nan = np.any(np.isnan(data['data']))
                has_inf = np.any(np.isinf(data['data']))
                
                if not has_nan and not has_inf:
                    print(f"  {check_mark(True)} No NaN/Inf detected")
                else:
                    print(f"  {check_mark(False)} Data quality issue: NaN={has_nan}, Inf={has_inf}")
                    found = False
            else:
                print(f"  {check_mark(False)} Unexpected shape: {shape}")
                found = False
                
        except Exception as e:
            print(f"  {check_mark(False)} Error loading data: {e}")
            found = False
    else:
        print(f"  {check_mark(False)} Training data NOT found")
        print(f"\n  Searched locations:")
        for loc in locations:
            print(f"    - {loc}")
        print(f"\n  To create data, run:")
        print(f"    python scripts/prepare_all_data.py --dataset vitaldb --mode fasttrack")
    
    return found

def check_configs():
    """Check if configuration files exist and are correct."""
    print_section("3. Checking Configuration Files")
    
    configs = {
        'configs/ssl_pretrain.yaml': 'SSL pre-training config',
        'configs/channels.yaml': 'Channel specifications',
        'configs/windows.yaml': 'Window specifications'
    }
    
    all_ok = True
    for config_path, desc in configs.items():
        path = Path(config_path)
        if path.exists():
            print(f"  {check_mark(True)} {desc}: {config_path}")
            
            # Check SSL config specifically
            if 'ssl_pretrain' in config_path:
                try:
                    import yaml
                    with open(path, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    lr = config['training']['lr']
                    if lr == 1e-4:
                        print(f"    {check_mark(True)} Learning rate: {lr} (correct)")
                    elif lr == 5e-4:
                        print(f"    ⚠️  Learning rate: {lr} (should be 1e-4)")
                        print(f"       Article recommends 1e-4, not 5e-4")
                    else:
                        print(f"    ⚠️  Learning rate: {lr}")
                    
                    mask_ratio = config['ssl']['mask_ratio']
                    if mask_ratio == 0.4:
                        print(f"    {check_mark(True)} Mask ratio: {mask_ratio}")
                    else:
                        print(f"    ⚠️  Mask ratio: {mask_ratio} (expected 0.4)")
                        
                except Exception as e:
                    print(f"    ⚠️  Error reading config: {e}")
        else:
            print(f"  {check_mark(False)} {desc}: NOT FOUND")
            all_ok = False
    
    return all_ok

def check_gpu():
    """Check GPU availability."""
    print_section("4. Checking GPU")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"  {check_mark(True)} GPU Available: {gpu_name}")
            print(f"  {check_mark(True)} GPU Memory: {gpu_memory:.1f} GB")
            
            # Test allocation
            try:
                test_tensor = torch.randn(128, 2, 1250).cuda()
                print(f"  {check_mark(True)} GPU Test: OK (allocated 128×2×1250 tensor)")
                del test_tensor
                torch.cuda.empty_cache()
                return True
            except Exception as e:
                print(f"  {check_mark(False)} GPU Test: FAILED ({e})")
                return False
        else:
            print(f"  ⚠️  GPU Not Available (will use CPU)")
            print(f"      Training will be much slower on CPU")
            return False
            
    except ImportError:
        print(f"  {check_mark(False)} PyTorch not installed")
        return False

def check_directories():
    """Check directory structure."""
    print_section("5. Checking Directory Structure")
    
    dirs = {
        'src/ssl': 'SSL components',
        'src/models': 'Model components',
        'scripts': 'Scripts directory',
        'configs': 'Configuration files'
    }
    
    all_ok = True
    for dir_path, desc in dirs.items():
        path = Path(dir_path)
        if path.exists() and path.is_dir():
            print(f"  {check_mark(True)} {desc}: {dir_path}")
        else:
            print(f"  {check_mark(False)} {desc}: NOT FOUND")
            all_ok = False
    
    # Check critical files
    files = {
        'src/ssl/pretrainer.py': 'SSL trainer',
        'src/models/ttm_adapter.py': 'TTM adapter',
        'src/models/decoders.py': 'Decoder',
        'scripts/pretrain_vitaldb_ssl.py': 'Pre-training script'
    }
    
    for file_path, desc in files.items():
        path = Path(file_path)
        if path.exists():
            print(f"  {check_mark(True)} {desc}: {file_path}")
        else:
            print(f"  {check_mark(False)} {desc}: NOT FOUND")
            all_ok = False
    
    return all_ok

def check_disk_space():
    """Check available disk space."""
    print_section("6. Checking Disk Space")
    
    try:
        import shutil
        
        total, used, free = shutil.disk_usage('.')
        free_gb = free / (1024**3)
        
        if free_gb > 5:
            print(f"  {check_mark(True)} Free space: {free_gb:.1f} GB")
            print(f"      (Checkpoints will need ~500 MB)")
            return True
        else:
            print(f"  ⚠️  Free space: {free_gb:.1f} GB (low)")
            print(f"      SSL training needs at least 5 GB free")
            return False
            
    except Exception as e:
        print(f"  ⚠️  Could not check disk space: {e}")
        return True

def main():
    """Run all checks."""
    print("\n" + "="*70)
    print("SSL PRE-TRAINING PRE-FLIGHT CHECK")
    print("="*70)
    print("\nVerifying that everything is ready for SSL pre-training...")
    
    checks = {
        'Dependencies': check_dependencies(),
        'Preprocessed Data': check_data(),
        'Configuration Files': check_configs(),
        'GPU': check_gpu(),
        'Directory Structure': check_directories(),
        'Disk Space': check_disk_space()
    }
    
    # Summary
    print_section("SUMMARY")
    
    all_passed = True
    for check_name, passed in checks.items():
        status = check_mark(passed) if passed else "⚠️ "
        print(f"  {status} {check_name}")
        if not passed and check_name in ['Dependencies', 'Preprocessed Data', 'Configuration Files', 'Directory Structure']:
            all_passed = False
    
    print(f"\n{'='*70}")
    
    if all_passed:
        print("✅ ALL CRITICAL CHECKS PASSED!")
        print("\nYou're ready to run SSL pre-training:")
        print("  python scripts/pretrain_vitaldb_ssl.py")
        print(f"{'='*70}\n")
        return 0
    else:
        print("❌ SOME CHECKS FAILED")
        print("\nPlease fix the issues above before running SSL pre-training.")
        print("\nCommon fixes:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Prepare data: python scripts/prepare_all_data.py --dataset vitaldb")
        print("  3. Check config files exist in configs/")
        print(f"{'='*70}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
