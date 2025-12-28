#!/usr/bin/env python3
"""Clean up corrupted/empty NPZ files from failed data preparation runs.

This script removes:
- Empty NPZ files (0 bytes)
- Corrupted NPZ files (cannot be loaded)
- Invalid data files (wrong shape, all NaN, etc.)

Run this before retrying data preparation to avoid "No data left in file" errors.
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


def is_file_corrupted(file_path: Path) -> tuple[bool, str]:
    """Check if an NPZ file is corrupted.

    Returns:
        (is_corrupted, reason)
    """
    # Check if file is empty
    if file_path.stat().st_size == 0:
        return True, "empty file (0 bytes)"

    # Check if file is very small (likely incomplete)
    if file_path.stat().st_size < 1000:  # Less than 1KB
        return True, f"suspiciously small ({file_path.stat().st_size} bytes)"

    # Try to load the file
    try:
        data = np.load(file_path)

        # Check if it has expected keys
        if 'data' not in data:
            return True, "missing 'data' key"

        # Check if data is empty
        if data['data'].shape[0] == 0:
            return True, "empty data array"

        # Check for all NaN
        if np.all(np.isnan(data['data'])):
            return True, "all NaN values"

        # File seems OK
        return False, "OK"

    except Exception as e:
        return True, f"load error: {str(e)}"


def cleanup_directory(directory: Path, dry_run: bool = True) -> dict:
    """Clean up corrupted files in a directory.

    Args:
        directory: Directory to clean
        dry_run: If True, only report what would be deleted

    Returns:
        dict with statistics
    """
    stats = {
        'total_files': 0,
        'corrupted_files': 0,
        'deleted_files': 0,
        'freed_bytes': 0,
        'reasons': {}
    }

    if not directory.exists():
        print(f"Directory does not exist: {directory}")
        return stats

    # Find all NPZ files
    npz_files = list(directory.glob('**/*.npz'))
    stats['total_files'] = len(npz_files)

    print(f"\nScanning {len(npz_files)} NPZ files in {directory}...")

    corrupted_files = []

    for file_path in tqdm(npz_files, desc="Checking files"):
        is_corrupted, reason = is_file_corrupted(file_path)

        if is_corrupted:
            stats['corrupted_files'] += 1
            stats['freed_bytes'] += file_path.stat().st_size

            # Track reasons
            if reason not in stats['reasons']:
                stats['reasons'][reason] = 0
            stats['reasons'][reason] += 1

            corrupted_files.append((file_path, reason))

            if not dry_run:
                file_path.unlink()
                stats['deleted_files'] += 1

    # Print results
    print(f"\n{'='*70}")
    print(f"Scan Results for: {directory}")
    print(f"{'='*70}")
    print(f"Total files scanned: {stats['total_files']}")
    print(f"Corrupted files found: {stats['corrupted_files']}")

    if stats['corrupted_files'] > 0:
        print(f"\nCorruption reasons:")
        for reason, count in sorted(stats['reasons'].items(), key=lambda x: x[1], reverse=True):
            print(f"  - {reason}: {count} files")

        print(f"\nSpace to be freed: {stats['freed_bytes'] / 1024 / 1024:.2f} MB")

        if dry_run:
            print(f"\n⚠️  DRY RUN MODE - No files were deleted")
            print(f"\nTo actually delete these files, run with --delete flag:")
            print(f"  python scripts/cleanup_corrupted_data.py --directory {directory} --delete")
        else:
            print(f"\n✓ Deleted {stats['deleted_files']} corrupted files")
            print(f"✓ Freed {stats['freed_bytes'] / 1024 / 1024:.2f} MB")
    else:
        print(f"\n✓ No corrupted files found!")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Clean up corrupted NPZ files from failed data preparation runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check for corrupted files (dry run - safe)
  python scripts/cleanup_corrupted_data.py

  # Check specific directory
  python scripts/cleanup_corrupted_data.py --directory data/processed/vitaldb/windows/paired

  # Actually delete corrupted files
  python scripts/cleanup_corrupted_data.py --delete

  # Clean all data directories
  python scripts/cleanup_corrupted_data.py --all --delete
        """
    )

    parser.add_argument(
        '--directory',
        type=str,
        default='data/processed/vitaldb/windows/paired',
        help='Directory to clean (default: VitalDB paired windows)'
    )

    parser.add_argument(
        '--delete',
        action='store_true',
        help='Actually delete corrupted files (default: dry run)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Clean all data directories (VitalDB + BUT-PPG)'
    )

    args = parser.parse_args()

    if args.all:
        directories = [
            Path('data/processed/vitaldb/windows/paired'),
            Path('data/processed/butppg/windows'),
            Path('data/processed/butppg/windows_with_labels'),
        ]
    else:
        directories = [Path(args.directory)]

    # Run cleanup
    total_stats = {
        'total_files': 0,
        'corrupted_files': 0,
        'deleted_files': 0,
        'freed_bytes': 0
    }

    for directory in directories:
        stats = cleanup_directory(directory, dry_run=not args.delete)

        total_stats['total_files'] += stats['total_files']
        total_stats['corrupted_files'] += stats['corrupted_files']
        total_stats['deleted_files'] += stats['deleted_files']
        total_stats['freed_bytes'] += stats['freed_bytes']

    # Print summary if multiple directories
    if len(directories) > 1:
        print(f"\n{'='*70}")
        print(f"OVERALL SUMMARY")
        print(f"{'='*70}")
        print(f"Total files scanned: {total_stats['total_files']}")
        print(f"Total corrupted files: {total_stats['corrupted_files']}")

        if args.delete:
            print(f"Total deleted: {total_stats['deleted_files']}")
            print(f"Total space freed: {total_stats['freed_bytes'] / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
