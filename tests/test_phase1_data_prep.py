#!/usr/bin/env python3
"""
Phase 1: Data Preparation Test

Tests that VitalDB windows are correctly preprocessed and ready for SSL.

Run:
    python tests/test_phase1_data_prep.py
    
    # Or specify custom data dir:
    python tests/test_phase1_data_prep.py --data-dir data/vitaldb_windows
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import torch
from typing import Dict, Tuple


def check_file_exists(filepath: Path) -> Tuple[bool, str]:
    """Check if a file exists and return status."""
    if filepath.exists():
        size_mb = filepath.stat().st_size / (1024 * 1024)
        return True, f"{size_mb:.1f} MB"
    return False, "NOT FOUND"


def load_and_validate_split(filepath: Path, split_name: str) -> Dict:
    """Load and validate a data split."""
    print(f"\n[{split_name.upper()}] Loading {filepath.name}...")
    
    if not filepath.exists():
        print(f"  ✗ File not found: {filepath}")
        return {'success': False}
    
    # Load data
    data = np.load(filepath)
    
    # Check required keys
    if 'signals' not in data:
        print(f"  ✗ Missing 'signals' key. Found: {list(data.keys())}")
        return {'success': False}
    
    signals = data['signals']
    N, C, T = signals.shape
    
    print(f"  Shape: {signals.shape}")
    print(f"    N = {N:,} windows")
    print(f"    C = {C} channels")
    print(f"    T = {T} timesteps")
    
    # Validate shape
    issues = []
    
    if C != 2:
        issues.append(f"Expected 2 channels (PPG+ECG), got {C}")
    
    if T != 1250:
        issues.append(f"Expected 1250 timesteps (10s @ 125Hz), got {T}")
    
    # Check for NaNs/Infs
    n_nans = np.isnan(signals).sum()
    n_infs = np.isinf(signals).sum()
    
    if n_nans > 0:
        issues.append(f"{n_nans:,} NaN values found")
    
    if n_infs > 0:
        issues.append(f"{n_infs:,} Inf values found")
    
    # Check normalization (should be roughly N(0,1) per window)
    means = signals.mean(axis=2)  # [N, C]
    stds = signals.std(axis=2)    # [N, C]
    
    mean_mean = np.abs(means.mean())
    std_mean = stds.mean()
    
    print(f"  Stats:")
    print(f"    Mean: {mean_mean:.4f} (expect ~0)")
    print(f"    Std:  {std_mean:.4f} (expect ~1)")
    print(f"    NaNs: {n_nans}")
    print(f"    Infs: {n_infs}")
    
    # Normalization check
    if mean_mean > 0.1:
        issues.append(f"Data may not be normalized (mean={mean_mean:.3f})")
    
    if std_mean < 0.5 or std_mean > 2.0:
        issues.append(f"Suspicious std deviation (std={std_mean:.3f})")
    
    # Report
    if issues:
        print(f"  ⚠️  Issues found:")
        for issue in issues:
            print(f"    - {issue}")
        return {
            'success': False,
            'n_windows': N,
            'issues': issues
        }
    else:
        print(f"  ✓ All checks passed")
        return {
            'success': True,
            'n_windows': N,
            'channels': C,
            'timesteps': T,
            'mean': mean_mean,
            'std': std_mean
        }


def test_vitaldb_windows(data_dir: Path) -> bool:
    """Test VitalDB preprocessed windows."""
    print("="*70)
    print("TESTING VITALDB WINDOWS")
    print("="*70)
    print(f"Data directory: {data_dir}")
    
    # Check files exist
    files = {
        'train': data_dir / 'train_windows.npz',
        'val': data_dir / 'val_windows.npz',
        'test': data_dir / 'test_windows.npz'
    }
    
    print("\n[FILE CHECK]")
    file_status = {}
    for split, filepath in files.items():
        exists, info = check_file_exists(filepath)
        file_status[split] = exists
        status = "✓" if exists else "✗"
        print(f"  {status} {split:5} | {info:12} | {filepath.name}")
    
    if not any(file_status.values()):
        print("\n✗ No data files found!")
        print("\nTo create VitalDB windows:")
        print("  1. python scripts/ttm_vitaldb.py prepare-splits \\")
        print("       --mode fasttrack --output configs/splits")
        print("  2. python scripts/build_windows_quiet.py train")
        print("  3. python scripts/build_windows_quiet.py val")
        return False
    
    # Load and validate each split
    results = {}
    for split, filepath in files.items():
        if file_status[split]:
            results[split] = load_and_validate_split(filepath, split)
        else:
            results[split] = {'success': False}
    
    # Summary
    print("\n" + "="*70)
    print("DATA VALIDATION SUMMARY")
    print("="*70)
    
    all_ok = True
    for split, result in results.items():
        if result['success']:
            n = result['n_windows']
            print(f"✓ {split:5} | {n:,} windows | VALID")
        else:
            print(f"✗ {split:5} | INVALID")
            all_ok = False
    
    # Article recommendation check
    if 'train' in results and results['train']['success']:
        n_train = results['train']['n_windows']
        print(f"\nWindow count analysis:")
        print(f"  Train: {n_train:,} windows")
        
        if n_train >= 400000:
            print(f"  ✓ Excellent! Article recommends ~500K windows")
        elif n_train >= 100000:
            print(f"  ⚠️  Acceptable, but article recommends ~500K windows")
        else:
            print(f"  ⚠️  Low window count. Article recommends ~500K windows")
    
    return all_ok


def test_quick_dataloader(data_dir: Path) -> bool:
    """Test that data can be loaded with PyTorch."""
    print("\n" + "="*70)
    print("TESTING PYTORCH DATALOADER")
    print("="*70)
    
    train_file = data_dir / 'train_windows.npz'
    
    if not train_file.exists():
        print("⚠️  Skipping (no train data)")
        return True
    
    try:
        # Load data
        data = np.load(train_file)
        signals = torch.from_numpy(data['signals']).float()
        
        # Create simple dataset
        from torch.utils.data import TensorDataset, DataLoader
        
        dataset = TensorDataset(signals[:64])  # Just use 64 samples
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        # Try to get one batch
        batch = next(iter(loader))
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        
        print(f"Batch shape: {batch.shape}")
        print(f"Batch dtype: {batch.dtype}")
        print(f"Batch device: {batch.device}")
        
        # Check values
        if torch.isnan(batch).any():
            print("✗ Batch contains NaNs")
            return False
        
        if torch.isinf(batch).any():
            print("✗ Batch contains Infs")
            return False
        
        print("✓ DataLoader working correctly")
        return True
        
    except Exception as e:
        print(f"✗ DataLoader failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Data preparation test"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/vitaldb_windows',
        help='Path to VitalDB windows directory'
    )
    args = parser.parse_args()
    
    print("="*70)
    print("PHASE 1: DATA PREPARATION TEST")
    print("="*70)
    print("\nThis test verifies VitalDB windows are ready for SSL pretraining.\n")
    
    data_dir = Path(args.data_dir)
    
    # Run tests
    test1 = test_vitaldb_windows(data_dir)
    test2 = test_quick_dataloader(data_dir)
    
    # Final summary
    print("\n" + "="*70)
    print("PHASE 1 SUMMARY")
    print("="*70)
    
    tests = [
        ("VitalDB windows valid", test1),
        ("PyTorch DataLoader", test2)
    ]
    
    passed = sum(1 for _, ok in tests if ok)
    total = len(tests)
    
    for name, ok in tests:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"{status:8} | {name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 Data is ready for SSL pretraining!")
        print("\nNext step:")
        print("  → Run Phase 2 SSL pretraining smoke test")
        print("  → python tests/test_phase2_ssl_smoke.py")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        return 1


if __name__ == '__main__':
    exit(main())
