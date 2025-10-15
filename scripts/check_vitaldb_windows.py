#!/usr/bin/env python3
"""
Quick script to check VitalDB paired dataset window counts.

Usage:
    python scripts/check_vitaldb_windows.py
    python scripts/check_vitaldb_windows.py --path data/processed/vitaldb/paired_1024
"""

import argparse
import json
from pathlib import Path
import numpy as np


def check_dataset(base_path):
    """Check window counts in a VitalDB paired dataset."""
    base_path = Path(base_path)

    if not base_path.exists():
        print(f"❌ Dataset not found: {base_path}")
        return

    print("="*80)
    print(f"📊 VitalDB Paired Dataset Statistics")
    print("="*80)
    print(f"Path: {base_path}")
    print()

    # Check summary file if exists
    summary_file = base_path / 'dataset_summary.json'
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)

        print("📄 Dataset Summary:")
        if 'summary' in summary:
            for split, stats in summary['summary'].items():
                print(f"  {split:5s}: {stats['cases_successful']:3d} cases, "
                      f"{stats['windows']:6,d} windows "
                      f"({stats['cases_successful']}/{stats['cases_attempted']} success rate)")
        print()

    # Check each split directory
    total_cases = 0
    total_windows = 0
    total_size_mb = 0

    for split_name in ['train', 'val', 'test']:
        split_dir = base_path / split_name

        if not split_dir.exists():
            print(f"⚠️  {split_name:5s}: Directory not found")
            continue

        # Find all case files
        case_files = sorted(list(split_dir.glob('case_*.npz')))

        if len(case_files) == 0:
            print(f"⚠️  {split_name:5s}: No case files found")
            continue

        # Count windows
        split_windows = 0
        split_size = 0

        for case_file in case_files:
            try:
                data = np.load(case_file)
                split_windows += data['data'].shape[0]
                split_size += case_file.stat().st_size
            except Exception as e:
                print(f"    ⚠️  Error reading {case_file.name}: {e}")

        split_size_mb = split_size / 1024 / 1024

        total_cases += len(case_files)
        total_windows += split_windows
        total_size_mb += split_size_mb

        # Check sample shape
        sample_data = np.load(case_files[0])
        sample_shape = sample_data['data'].shape

        print(f"✓ {split_name:5s}: {len(case_files):3d} cases, "
              f"{split_windows:7,d} windows, "
              f"{split_size_mb:6.1f} MB "
              f"(avg {split_windows//len(case_files):,} windows/case)")
        print(f"         Shape: {sample_shape} "
              f"[N, Channels={sample_shape[1]}, Samples={sample_shape[2]}]")

    print()
    print("="*80)
    print("📈 TOTALS")
    print("="*80)
    print(f"Total cases:   {total_cases:,}")
    print(f"Total windows: {total_windows:,}")
    print(f"Total size:    {total_size_mb:.1f} MB")

    if total_cases > 0:
        print(f"Average:       {total_windows//total_cases:,} windows per case")

    print()

    # Validation checks
    print("="*80)
    print("✅ VALIDATION CHECKS")
    print("="*80)

    if total_windows < 100:
        print("❌ FAILED: Very few windows detected (<100)")
        print("   → This indicates the interval parameter bug is NOT fixed")
        print("   → Check that rebuild_vitaldb_paired.py line 62 has: interval = 1.0 / default_fs")
    elif total_windows < 1000:
        print("⚠️  WARNING: Low window count (<1,000)")
        print("   → Dataset may be using very short recordings or strict quality filters")
    elif total_windows < 10000:
        print("✓ GOOD: Reasonable window count for small dataset")
    else:
        print("✓ EXCELLENT: Large dataset suitable for SSL pre-training")

    # Check for fix
    expected_windows_per_case = 1000  # Rough estimate for good data
    actual_windows_per_case = total_windows // total_cases if total_cases > 0 else 0

    if actual_windows_per_case < 10:
        print(f"❌ FAILED: Only {actual_windows_per_case} windows/case (expected ~{expected_windows_per_case})")
        print("   → The interval parameter bug is likely NOT fixed")
    else:
        print(f"✓ PASSED: {actual_windows_per_case:,} windows/case indicates fix is working")

    print()

    # Load sample and check stats
    print("="*80)
    print("📊 SAMPLE DATA STATISTICS")
    print("="*80)

    for split_name in ['train']:
        split_dir = base_path / split_name
        if not split_dir.exists():
            continue

        case_files = list(split_dir.glob('case_*.npz'))
        if len(case_files) == 0:
            continue

        # Load first case
        data = np.load(case_files[0])
        windows = data['data']

        print(f"Sample from {split_name} (first case):")
        print(f"  Shape: {windows.shape}")
        print(f"  PPG channel (ch 0):")
        print(f"    Mean: {windows[:, 0, :].mean():7.4f}, Std: {windows[:, 0, :].std():7.4f}")
        print(f"    Range: [{windows[:, 0, :].min():7.2f}, {windows[:, 0, :].max():7.2f}]")
        print(f"  ECG channel (ch 1):")
        print(f"    Mean: {windows[:, 1, :].mean():7.4f}, Std: {windows[:, 1, :].std():7.4f}")
        print(f"    Range: [{windows[:, 1, :].min():7.2f}, {windows[:, 1, :].max():7.2f}]")
        print(f"  NaN check: {'❌ Contains NaN' if np.any(np.isnan(windows)) else '✓ No NaN'}")
        print(f"  Inf check: {'❌ Contains Inf' if np.any(np.isinf(windows)) else '✓ No Inf'}")

    print()
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Check VitalDB paired dataset window counts and statistics"
    )
    parser.add_argument(
        '--path',
        type=str,
        default='data/processed/vitaldb/paired_1024',
        help='Path to paired dataset directory'
    )

    args = parser.parse_args()
    check_dataset(args.path)


if __name__ == '__main__':
    main()
