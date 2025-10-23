"""Verify that SSL and fine-tuning use the same quality labels.

CRITICAL: This script verifies the fix for the SSL collapse issue.
- SSL was using ternary labels (Low/Med/High) with 98.8% as "Medium"
- Fine-tuning was using binary labels (Poor/Good) with 78.6% Poor, 21.4% Good
- This mismatch prevented SSL from learning useful features

This script verifies that BOTH now use binary labels (Poor/Good).

Usage:
    python scripts/verify_quality_labels.py \
        --data-dir data/processed/butppg/windows_with_labels

Expected output:
    ✓ SSL and fine-tuning labels MATCH
    ✓ Distribution: ~78.6% Poor, ~21.4% Good
    ✓ Ready for SSL training
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.butppg_quality_dataset import BinaryQualityBUTPPGDataset
from src.data.butppg_dataset import BUTPPGDataset


def verify_labels(data_dir: str, split: str = 'train'):
    """Verify that SSL and fine-tuning use identical labels."""

    print(f"\n{'='*80}")
    print(f"Verifying Quality Labels for {split.upper()} split")
    print(f"{'='*80}\n")

    # 1. Load SSL dataset (new binary quality dataset)
    print(f"1. Loading SSL dataset (BinaryQualityBUTPPGDataset)...")
    ssl_dataset = BinaryQualityBUTPPGDataset(
        data_dir=data_dir,
        split=split,
        modality=['ppg', 'ecg']
    )

    ssl_labels = ssl_dataset.quality_labels.numpy()
    print(f"   Loaded {len(ssl_labels)} samples")

    # 2. Load fine-tuning dataset
    print(f"\n2. Loading fine-tuning dataset (BUTPPGDataset with task='quality')...")
    ft_dataset = BUTPPGDataset(
        data_dir=data_dir,
        split=split,
        modality=['ppg', 'ecg'],
        mode='preprocessed',
        task='quality',
        filter_missing=True,
        return_labels=True
    )

    # Extract labels from fine-tuning dataset
    ft_labels = []
    for idx in range(len(ft_dataset)):
        # Load window file directly
        window_file = ft_dataset.window_files[ft_dataset.valid_indices[idx]]
        data = np.load(window_file)
        if 'quality' in data:
            ft_labels.append(int(data['quality'].item()))

    ft_labels = np.array(ft_labels)
    print(f"   Loaded {len(ft_labels)} samples")

    # 3. Verify counts match
    print(f"\n3. Verifying sample counts...")
    if len(ssl_labels) != len(ft_labels):
        print(f"   ❌ MISMATCH: SSL has {len(ssl_labels)} samples, fine-tuning has {len(ft_labels)} samples")
        return False

    print(f"   ✓ Sample counts match: {len(ssl_labels)} samples")

    # 4. Verify distributions match
    print(f"\n4. Verifying label distributions...")

    ssl_poor = (ssl_labels == 0).sum()
    ssl_good = (ssl_labels == 1).sum()
    ssl_poor_pct = 100 * ssl_poor / len(ssl_labels)
    ssl_good_pct = 100 * ssl_good / len(ssl_labels)

    ft_poor = (ft_labels == 0).sum()
    ft_good = (ft_labels == 1).sum()
    ft_poor_pct = 100 * ft_poor / len(ft_labels)
    ft_good_pct = 100 * ft_good / len(ft_labels)

    print(f"\n   SSL Distribution:")
    print(f"     Poor (0): {ssl_poor} ({ssl_poor_pct:.1f}%)")
    print(f"     Good (1): {ssl_good} ({ssl_good_pct:.1f}%)")

    print(f"\n   Fine-tuning Distribution:")
    print(f"     Poor (0): {ft_poor} ({ft_poor_pct:.1f}%)")
    print(f"     Good (1): {ft_good} ({ft_good_pct:.1f}%)")

    if ssl_poor != ft_poor or ssl_good != ft_good:
        print(f"\n   ❌ MISMATCH: Distributions differ!")
        return False

    print(f"\n   ✓ Distributions match!")

    # 5. Verify label-by-label match
    print(f"\n5. Verifying label-by-label correspondence...")

    if not np.array_equal(ssl_labels, ft_labels):
        # Find first mismatch
        mismatches = np.where(ssl_labels != ft_labels)[0]
        print(f"   ❌ MISMATCH: {len(mismatches)} samples have different labels")
        print(f"   First mismatch at index {mismatches[0]}: SSL={ssl_labels[mismatches[0]]}, FT={ft_labels[mismatches[0]]}")
        return False

    print(f"   ✓ All {len(ssl_labels)} labels match exactly!")

    # 6. Verify expected distribution
    print(f"\n6. Verifying expected distribution...")

    expected_poor_pct = 78.6
    expected_good_pct = 21.4

    if abs(ssl_poor_pct - expected_poor_pct) < 5 and abs(ssl_good_pct - expected_good_pct) < 5:
        print(f"   ✓ Distribution matches expected (~{expected_poor_pct:.0f}% Poor, ~{expected_good_pct:.0f}% Good)")
    else:
        print(f"   ⚠️  Distribution differs from expected ({expected_poor_pct:.0f}% Poor, {expected_good_pct:.0f}% Good)")
        print(f"   This is OK if data changed, but verify this is intentional.")

    # 7. Verify NOT ternary labels
    print(f"\n7. Verifying NOT using ternary labels (Low/Med/High)...")

    unique_labels = np.unique(ssl_labels)
    if len(unique_labels) > 2:
        print(f"   ❌ PROBLEM: Found {len(unique_labels)} unique labels: {unique_labels}")
        print(f"   Expected only 2 labels (0=Poor, 1=Good)")
        return False

    if not np.array_equal(unique_labels, np.array([0, 1])):
        print(f"   ❌ PROBLEM: Labels are {unique_labels}, expected [0, 1]")
        return False

    print(f"   ✓ Using binary labels only (0=Poor, 1=Good)")

    # Success!
    print(f"\n{'='*80}")
    print(f"✅ ALL CHECKS PASSED!")
    print(f"{'='*80}")
    print(f"\nSummary:")
    print(f"  - SSL and fine-tuning datasets use IDENTICAL labels")
    print(f"  - Binary labels (0=Poor, 1=Good) with {len(ssl_labels)} samples")
    print(f"  - Distribution: {ssl_poor_pct:.1f}% Poor, {ssl_good_pct:.1f}% Good")
    print(f"  - NOT using ternary labels (no 98.8% Medium issue)")
    print(f"\n✅ Ready for SSL training!")
    print(f"{'='*80}\n")

    return True


def main():
    parser = argparse.ArgumentParser(description="Verify SSL and fine-tuning label consistency")

    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed/butppg/windows_with_labels',
        help='Path to preprocessed BUT-PPG data'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val', 'test'],
        help='Which split to verify'
    )

    args = parser.parse_args()

    # Verify
    success = verify_labels(args.data_dir, args.split)

    if success:
        print(f"\n✅ Verification PASSED\n")
        sys.exit(0)
    else:
        print(f"\n❌ Verification FAILED\n")
        sys.exit(1)


if __name__ == '__main__':
    main()
