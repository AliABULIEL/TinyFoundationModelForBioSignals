#!/usr/bin/env python3
"""
Debug script to check BUT-PPG dataset structure after extraction

Usage:
    python scripts/debug_butppg_structure.py --data-dir data/but_ppg/raw
"""

import argparse
from pathlib import Path


def check_directory_structure(root_dir: Path):
    """Check what's actually in the extracted directory"""

    print("="*80)
    print("BUT-PPG DATASET STRUCTURE DIAGNOSTIC")
    print("="*80)
    print(f"Checking: {root_dir}")
    print()

    if not root_dir.exists():
        print(f"❌ Directory not found: {root_dir}")
        return

    # Find all subdirectories
    print("📁 Directory structure:")
    subdirs = sorted([d for d in root_dir.rglob("*") if d.is_dir()])
    for d in subdirs[:20]:  # Show first 20
        rel_path = d.relative_to(root_dir)
        print(f"  {rel_path}/")

    if len(subdirs) > 20:
        print(f"  ... and {len(subdirs) - 20} more directories")

    print()

    # Find all file types
    print("📄 File types found:")
    extensions = {}
    for f in root_dir.rglob("*"):
        if f.is_file():
            ext = f.suffix.lower()
            extensions[ext] = extensions.get(ext, 0) + 1

    for ext, count in sorted(extensions.items()):
        print(f"  {ext if ext else '(no extension)'}: {count} files")

    print()

    # Look for specific patterns
    print("🔍 Looking for PhysioNet WFDB files...")
    dat_files = list(root_dir.rglob("*.dat"))
    hea_files = list(root_dir.rglob("*.hea"))

    print(f"  .dat files: {len(dat_files)}")
    print(f"  .hea files: {len(hea_files)}")

    if dat_files:
        print(f"\n  Sample .dat files:")
        for f in dat_files[:10]:
            print(f"    {f.relative_to(root_dir)}")

    if hea_files:
        print(f"\n  Sample .hea files:")
        for f in hea_files[:10]:
            print(f"    {f.relative_to(root_dir)}")

    print()

    # Look for BUT-PPG specific patterns
    print("🔍 Looking for BUT-PPG specific patterns...")
    ppg_files = list(root_dir.rglob("*PPG*"))
    ecg_files = list(root_dir.rglob("*ECG*"))
    acc_files = list(root_dir.rglob("*ACC*"))

    print(f"  Files with 'PPG' in name: {len(ppg_files)}")
    print(f"  Files with 'ECG' in name: {len(ecg_files)}")
    print(f"  Files with 'ACC' in name: {len(acc_files)}")

    if ppg_files:
        print(f"\n  Sample PPG files:")
        for f in ppg_files[:10]:
            print(f"    {f.relative_to(root_dir)} ({f.stat().st_size} bytes)")

    print()

    # Look for annotation files
    print("🔍 Looking for annotation files...")
    csv_files = list(root_dir.rglob("*.csv"))

    print(f"  CSV files found: {len(csv_files)}")
    for f in csv_files:
        print(f"    {f.relative_to(root_dir)} ({f.stat().st_size} bytes)")

        # Show first line
        try:
            with open(f) as file:
                first_line = file.readline().strip()
                print(f"      Header: {first_line[:100]}")
        except:
            pass

    print()

    # Check for specific BUT-PPG directory
    print("🔍 Looking for BUT-PPG dataset directory...")
    possible_dirs = [
        'but-ppg-an-annotated-photoplethysmography-dataset-2.0.0',
        'brno-university-of-technology-smartphone-ppg-database-but-ppg-2.0.0',
        'butppg',
        'BUT-PPG'
    ]

    for dirname in possible_dirs:
        test_dir = root_dir / dirname
        if test_dir.exists():
            print(f"  ✓ Found: {dirname}/")

            # Check what's inside
            contents = list(test_dir.iterdir())[:20]
            print(f"    Contents ({len(list(test_dir.iterdir()))} items total):")
            for item in contents:
                if item.is_dir():
                    print(f"      {item.name}/")
                else:
                    print(f"      {item.name} ({item.stat().st_size} bytes)")

    print()
    print("="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)
    print()
    print("💡 Next steps:")
    print("   1. Check the sample file paths above")
    print("   2. Look for the pattern of actual data files")
    print("   3. Update download_butppg_dataset.py with correct paths")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/but_ppg/raw',
        help='Root directory to check'
    )

    args = parser.parse_args()

    root_dir = Path(args.data_dir)
    check_directory_structure(root_dir)


if __name__ == '__main__':
    main()
