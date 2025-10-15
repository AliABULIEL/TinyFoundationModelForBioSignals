#!/usr/bin/env python3
"""
Quick script to check BUT-PPG data format and compatibility

Usage:
    python scripts/check_butppg_data.py
    python scripts/check_butppg_data.py --data-dir data/processed/butppg/windows
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import argparse


def check_data(data_path: Path) -> bool:
    """Check a single .npz file"""
    print(f"\nChecking: {data_path}")

    if not data_path.exists():
        print(f"  ❌ File not found!")
        return False

    try:
        data = np.load(data_path)
        print(f"  ✓ File loads successfully")
        print(f"  Keys: {list(data.keys())}")

        if 'signals' not in data or 'labels' not in data:
            print(f"  ❌ Missing required keys (signals, labels)")
            return False

        signals = data['signals']
        labels = data['labels']

        print(f"  Signals shape: {signals.shape}")
        print(f"  Labels shape: {labels.shape}")

        # Check shape
        if len(signals.shape) != 3:
            print(f"  ❌ Signals should be 3D [N, C, T], got {len(signals.shape)}D")
            return False

        N, C, T = signals.shape
        print(f"    N={N} samples, C={C} channels, T={T} timesteps")

        # Check labels
        unique_labels = np.unique(labels)
        print(f"    Unique labels: {unique_labels}")

        # Validate
        issues = []
        warnings = []

        if C != 5:
            issues.append(f"Expected 5 channels (ACC_X/Y/Z + PPG + ECG), got {C}")
        if T != 1024:
            warnings.append(f"Expected 1024 timesteps, got {T} (model uses 1024)")
        if len(labels) != N:
            issues.append(f"Labels count mismatch: {len(labels)} vs {N}")
        if len(unique_labels) != 2:
            warnings.append(f"Expected binary labels (0,1), found {unique_labels}")
        if not np.isfinite(signals).all():
            issues.append("Signals contain NaN or Inf values")

        # Print issues
        if warnings:
            print(f"  ⚠️  Warnings:")
            for warning in warnings:
                print(f"    - {warning}")

        if issues:
            print(f"  ❌ Issues found:")
            for issue in issues:
                print(f"    - {issue}")
            return False
        else:
            print(f"  ✅ All checks passed!")
            return True

    except Exception as e:
        print(f"  ❌ Error loading file: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Check BUT-PPG data format and compatibility'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed/butppg/windows',
        help='Directory containing BUT-PPG data'
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("=" * 70)
    print("BUT-PPG DATA COMPATIBILITY CHECK")
    print("=" * 70)
    print(f"Data directory: {data_dir}")

    if not data_dir.exists():
        print(f"\n❌ Directory not found: {data_dir}")
        print(f"\nMake sure to run data preparation first:")
        print(f"  python scripts/prepare_all_data.py --dataset butppg --mode fasttrack")
        return 1

    # Find all .npz files
    npz_files = sorted(data_dir.rglob("*.npz"))

    if not npz_files:
        print(f"\n❌ No .npz files found in {data_dir}")
        print(f"\nDirectory structure:")
        for item in data_dir.rglob("*"):
            if item.is_file() or item.is_dir():
                print(f"  {item.relative_to(data_dir)}")
        return 1

    print(f"\nFound {len(npz_files)} .npz files:")
    for npz_file in npz_files:
        print(f"  - {npz_file.relative_to(data_dir)}")

    # Check each file
    results = []
    for npz_file in npz_files:
        result = check_data(npz_file)
        results.append((npz_file.name, result))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    for name, result in results:
        status = "✅" if result else "❌"
        print(f"  {status} {name}")

    if passed == total:
        print("\n✅ All data files are compatible!")
        print("   You can proceed with fine-tuning:")
        print(f"     python scripts/finetune_butppg.py \\")
        print(f"       --pretrained artifacts/foundation_model/best_model.pt \\")
        print(f"       --data-dir {data_dir} \\")
        print(f"       --epochs 1")
        return 0
    else:
        print("\n❌ Some files have issues. Fix them before fine-tuning.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
