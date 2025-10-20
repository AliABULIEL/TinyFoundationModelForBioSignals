#!/usr/bin/env python3
"""
Check what labels are actually stored in BUT-PPG NPZ files.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from collections import Counter
import argparse


def check_npz_contents(npz_dir: Path, max_files: int = 100):
    """Check NPZ file contents and label distribution."""

    if not npz_dir.exists():
        print(f"✗ Directory not found: {npz_dir}")
        return

    window_files = sorted(npz_dir.glob('window_*.npz'))

    if not window_files:
        print(f"✗ No window_*.npz files found in {npz_dir}")
        return

    print(f"\n{'='*80}")
    print(f"CHECKING: {npz_dir}")
    print(f"{'='*80}")
    print(f"Total files: {len(window_files)}")

    # Load first file to see structure
    print(f"\n--- First file structure ---")
    first_data = np.load(window_files[0])
    print(f"File: {window_files[0].name}")
    print(f"\nAll keys: {list(first_data.keys())}")

    print(f"\n--- Key details ---")
    for key in sorted(first_data.keys()):
        val = first_data[key]
        if hasattr(val, 'shape'):
            print(f"  {key:20s}: shape={val.shape}, dtype={val.dtype}")
        else:
            print(f"  {key:20s}: type={type(val).__name__}, value={val}")

    # Check quality-related fields
    print(f"\n--- Quality-related fields (first file) ---")
    quality_keys = ['quality', 'ppg_quality', 'ecg_quality', 'hr']
    for key in quality_keys:
        if key in first_data:
            val = first_data[key]
            if hasattr(val, 'item'):
                val_scalar = val.item()
            else:
                val_scalar = val
            print(f"  {key}: {val_scalar} (is_nan: {np.isnan(val_scalar) if isinstance(val_scalar, (float, np.floating)) else False})")
        else:
            print(f"  {key}: NOT FOUND")

    # Analyze distribution across multiple files
    print(f"\n--- Analyzing {min(max_files, len(window_files))} files ---")

    qualities = []
    ppg_sqis = []
    ecg_sqis = []
    has_quality = 0
    has_ppg_sqi = 0
    has_ecg_sqi = 0

    for f in window_files[:max_files]:
        data = np.load(f)

        # Check 'quality' field
        if 'quality' in data:
            has_quality += 1
            q = data['quality']
            if hasattr(q, 'item'):
                q = q.item()
            qualities.append(q)

        # Check SQI fields
        if 'ppg_quality' in data:
            has_ppg_sqi += 1
            ppg_sqi = data['ppg_quality']
            if hasattr(ppg_sqi, 'item'):
                ppg_sqi = ppg_sqi.item()
            ppg_sqis.append(ppg_sqi)

        if 'ecg_quality' in data:
            has_ecg_sqi += 1
            ecg_sqi = data['ecg_quality']
            if hasattr(ecg_sqi, 'item'):
                ecg_sqi = ecg_sqi.item()
            ecg_sqis.append(ecg_sqi)

    # Report findings
    print(f"\nFiles with 'quality' field: {has_quality}/{max_files}")
    print(f"Files with 'ppg_quality' (SQI): {has_ppg_sqi}/{max_files}")
    print(f"Files with 'ecg_quality' (SQI): {has_ecg_sqi}/{max_files}")

    # Analyze 'quality' labels
    if qualities:
        print(f"\n--- 'quality' label distribution ---")
        valid_qualities = [q for q in qualities if not np.isnan(q)]
        nan_qualities = [q for q in qualities if np.isnan(q)]

        print(f"Valid labels: {len(valid_qualities)}/{len(qualities)}")
        print(f"NaN labels: {len(nan_qualities)}/{len(qualities)}")

        if valid_qualities:
            counter = Counter([int(q) for q in valid_qualities])
            print(f"\nValid label distribution:")
            for val, count in sorted(counter.items()):
                print(f"  {val}: {count} ({count/len(valid_qualities)*100:.1f}%)")
        else:
            print(f"\n✗ ALL 'quality' labels are NaN!")
            print(f"   Reason: quality-hr-ann.csv was missing when prepare script ran")

    # Analyze SQI metrics
    if ppg_sqis:
        print(f"\n--- PPG Quality (SQI) statistics ---")
        ppg_sqis = np.array(ppg_sqis)
        print(f"  Min: {ppg_sqis.min():.3f}")
        print(f"  Max: {ppg_sqis.max():.3f}")
        print(f"  Mean: {ppg_sqis.mean():.3f}")
        print(f"  Median: {np.median(ppg_sqis):.3f}")
        print(f"  > 0.4 (good): {np.sum(ppg_sqis > 0.4)}/{len(ppg_sqis)} ({np.sum(ppg_sqis > 0.4)/len(ppg_sqis)*100:.1f}%)")
        print(f"  ≤ 0.4 (poor): {np.sum(ppg_sqis <= 0.4)}/{len(ppg_sqis)} ({np.sum(ppg_sqis <= 0.4)/len(ppg_sqis)*100:.1f}%)")

    if ecg_sqis:
        print(f"\n--- ECG Quality (SQI) statistics ---")
        ecg_sqis = np.array(ecg_sqis)
        print(f"  Min: {ecg_sqis.min():.3f}")
        print(f"  Max: {ecg_sqis.max():.3f}")
        print(f"  Mean: {ecg_sqis.mean():.3f}")
        print(f"  Median: {np.median(ecg_sqis):.3f}")
        print(f"  > 0.4 (good): {np.sum(ecg_sqis > 0.4)}/{len(ecg_sqis)} ({np.sum(ecg_sqis > 0.4)/len(ecg_sqis)*100:.1f}%)")
        print(f"  ≤ 0.4 (poor): {np.sum(ecg_sqis <= 0.4)}/{len(ecg_sqis)} ({np.sum(ecg_sqis <= 0.4)/len(ecg_sqis)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Check BUT-PPG NPZ file contents")
    parser.add_argument('--data-dir', type=str,
                       default='data/processed/butppg/windows_with_labels',
                       help='Directory containing train/val/test subdirectories')
    parser.add_argument('--max-files', type=int, default=100,
                       help='Maximum files to check per split')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print(f"\nBUT-PPG NPZ FILE CONTENTS CHECK")
    print(f"{'='*80}")
    print(f"Data directory: {data_dir}")

    # Check each split
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        check_npz_contents(split_dir, args.max_files)

    # Provide recommendations
    print(f"\n{'='*80}")
    print(f"RECOMMENDATIONS")
    print(f"{'='*80}")

    print("""
If you see:
  ✗ ALL 'quality' labels are NaN

This means the prepare script couldn't find quality-hr-ann.csv.

SOLUTION 1: Generate labels from SQI (Signal Quality Index)
  Use the SQI values (ppg_quality, ecg_quality) to create binary labels:

  python scripts/patch_npz_labels_from_sqi.py \\
      --data-dir data/processed/butppg/windows_with_labels \\
      --sqi-threshold 0.4

  This will add 'quality' labels: 1 if both PPG and ECG SQI > 0.4, else 0

SOLUTION 2: Use annotation CSV
  Download the complete BUT-PPG dataset and re-run preparation:

  python scripts/download_butppg_dataset.py --output data/but_ppg/dataset

  Then re-run:
  python scripts/create_butppg_windows_with_labels.py \\
      --data-dir data/but_ppg/dataset \\
      --output-dir data/processed/butppg/windows_with_labels \\
      --splits-file configs/splits/butppg_splits.json \\
      --window-sec 8.192 \\
      --fs 125
""")


if __name__ == '__main__':
    main()
