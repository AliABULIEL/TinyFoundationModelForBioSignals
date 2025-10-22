#!/usr/bin/env python3
"""
Fix corrupted VitalDB data from failed JSON serialization.

This script:
1. Identifies corrupted/empty NPZ files
2. Cleans them up
3. Provides summary of what was processed
4. Re-runs the final steps if needed
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


def check_npz_file(file_path):
    """Check if an NPZ file is valid and get its stats."""
    try:
        data = np.load(file_path)
        if 'data' not in data:
            return 'missing_data_key', 0, None

        arr = data['data']
        if arr.size == 0:
            return 'empty_array', 0, None

        # Check shape
        if arr.ndim != 3:
            return 'wrong_dims', arr.size, arr.shape

        if arr.shape[1] != 2:
            return 'wrong_channels', arr.size, arr.shape

        if arr.shape[2] != 1024:
            return 'wrong_samples', arr.size, arr.shape

        # Check for NaN/Inf
        if np.any(np.isnan(arr)):
            return 'has_nan', arr.size, arr.shape

        if np.any(np.isinf(arr)):
            return 'has_inf', arr.size, arr.shape

        # Valid file
        return 'valid', arr.size, arr.shape

    except EOFError:
        return 'eof_error', 0, None
    except Exception as e:
        return f'error: {e}', 0, None


def scan_directory(directory):
    """Scan a directory for NPZ files and check their validity."""
    npz_files = list(Path(directory).glob('case_*.npz'))

    results = {
        'total': 0,
        'valid': 0,
        'corrupted': 0,
        'issues': {},
        'file_details': []
    }

    print(f"\n📊 Scanning {len(npz_files)} NPZ files in {directory}...")

    for file_path in tqdm(npz_files, desc="Checking files"):
        status, size, shape = check_npz_file(file_path)

        results['total'] += 1

        if status == 'valid':
            results['valid'] += 1
            results['file_details'].append({
                'file': file_path.name,
                'status': 'valid',
                'windows': shape[0] if shape else 0,
                'shape': str(shape) if shape else 'N/A'
            })
        else:
            results['corrupted'] += 1
            if status not in results['issues']:
                results['issues'][status] = []
            results['issues'][status].append(file_path.name)

            results['file_details'].append({
                'file': file_path.name,
                'status': status,
                'windows': 0,
                'shape': str(shape) if shape else 'N/A'
            })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Fix corrupted VitalDB paired data",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed/vitaldb/windows/paired',
        help='Directory containing paired windows'
    )

    parser.add_argument(
        '--clean',
        action='store_true',
        help='Delete corrupted files (backup recommended!)'
    )

    parser.add_argument(
        '--summary',
        action='store_true',
        help='Print detailed summary'
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"❌ Directory not found: {data_dir}")
        return 1

    print("="*80)
    print("VITALDB DATA DIAGNOSTIC & FIX")
    print("="*80)

    # Scan train, val, test directories
    all_results = {}

    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        if split_dir.exists():
            all_results[split] = scan_directory(split_dir)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    total_windows = 0
    total_files = 0
    total_corrupted = 0

    for split, results in all_results.items():
        valid_windows = sum(
            d['windows'] for d in results['file_details']
            if d['status'] == 'valid'
        )
        total_windows += valid_windows
        total_files += results['total']
        total_corrupted += results['corrupted']

        print(f"\n{split.upper()} Split:")
        print(f"  Total files: {results['total']}")
        print(f"  Valid files: {results['valid']}")
        print(f"  Valid windows: {valid_windows:,}")
        print(f"  Corrupted files: {results['corrupted']}")

        if results['issues']:
            print(f"  Issues:")
            for issue, files in results['issues'].items():
                print(f"    - {issue}: {len(files)} files")
                if args.summary and len(files) <= 10:
                    for f in files:
                        print(f"      • {f}")

    print(f"\n" + "="*80)
    print(f"TOTAL SUMMARY:")
    print(f"  Total files: {total_files}")
    print(f"  Total windows: {total_windows:,}")
    print(f"  Corrupted files: {total_corrupted}")
    print("="*80)

    # Expected windows per split (approximate)
    expected = {
        'train': '~15,000-25,000 windows',
        'val': '~3,000-5,000 windows',
        'test': '~3,000-5,000 windows'
    }

    print(f"\n📝 Expected ranges:")
    for split, exp in expected.items():
        if split in all_results:
            valid_windows = sum(
                d['windows'] for d in all_results[split]['file_details']
                if d['status'] == 'valid'
            )
            print(f"  {split}: {valid_windows:,} windows (expected {exp})")

    # Cleanup if requested
    if args.clean:
        print(f"\n⚠️  CLEANUP MODE: Deleting corrupted files...")

        deleted_count = 0
        for split, results in all_results.items():
            split_dir = data_dir / split

            for detail in results['file_details']:
                if detail['status'] != 'valid':
                    file_path = split_dir / detail['file']:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        print(f"  ✓ Deleted: {file_path.name}")
                    except Exception as e:
                        print(f"  ✗ Failed to delete {file_path.name}: {e}")

        print(f"\n✅ Deleted {deleted_count} corrupted files")

    elif total_corrupted > 0:
        print(f"\n⚠️  Found {total_corrupted} corrupted files")
        print(f"    Run with --clean to delete them")

    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)

    if total_corrupted > 0:
        print("1. Run with --clean to remove corrupted files")
        print("2. Re-run prepare_all_data.py to regenerate missing files")
    else:
        print("✅ All files are valid!")
        print(f"✅ Total: {total_windows:,} windows across {total_files} files")

        if total_windows < 10000:
            print("\n⚠️  WARNING: Total window count seems low!")
            print("   Expected: ~20,000-35,000 windows for full VitalDB")
            print("   Actual: {:,} windows".format(total_windows))
            print("\n   This may be because:")
            print("   - Processing was interrupted")
            print("   - Many cases failed quality checks")
            print("   - You're using a subset of VitalDB")

    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
