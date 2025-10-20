#!/usr/bin/env python3
"""
Diagnostic script to check BUT-PPG labels in NPZ files and annotation CSVs.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from collections import Counter

def check_annotation_files(data_dir: Path):
    """Check if annotation CSV files exist and are valid."""
    print("=" * 80)
    print("CHECKING ANNOTATION FILES")
    print("=" * 80)

    quality_hr_path = data_dir / 'quality-hr-ann.csv'
    subject_info_path = data_dir / 'subject-info.csv'

    # Check quality-hr-ann.csv
    if quality_hr_path.exists():
        df = pd.read_csv(quality_hr_path)
        print(f"\n✓ quality-hr-ann.csv found:")
        print(f"  - Rows: {len(df)}")
        print(f"  - Columns: {list(df.columns)}")

        if 'quality' in df.columns:
            quality_counts = df['quality'].value_counts()
            print(f"  - Quality distribution:")
            for val, count in quality_counts.items():
                print(f"      {val}: {count} ({count/len(df)*100:.1f}%)")
        else:
            print(f"  ✗ ERROR: No 'quality' column found!")
            print(f"    Available columns: {list(df.columns)}")

        print(f"\n  First 5 rows:")
        print(df.head())
    else:
        print(f"\n✗ quality-hr-ann.csv NOT FOUND at: {quality_hr_path}")
        print(f"  This is why all labels are NaN!")

    # Check subject-info.csv
    if subject_info_path.exists():
        df = pd.read_csv(subject_info_path)
        print(f"\n✓ subject-info.csv found:")
        print(f"  - Rows: {len(df)}")
        print(f"  - Columns: {list(df.columns)}")
    else:
        print(f"\n✗ subject-info.csv NOT FOUND at: {subject_info_path}")


def check_npz_files(output_dir: Path):
    """Check quality labels in NPZ files."""
    print("\n" + "=" * 80)
    print("CHECKING NPZ FILES")
    print("=" * 80)

    for split in ['train', 'val', 'test']:
        split_dir = output_dir / split
        if not split_dir.exists():
            print(f"\n✗ {split} directory not found: {split_dir}")
            continue

        window_files = sorted(split_dir.glob('window_*.npz'))
        if not window_files:
            print(f"\n✗ No window files found in {split}")
            continue

        print(f"\n{split.upper()} Split:")
        print(f"  Total files: {len(window_files)}")

        # Check first 100 files
        qualities = []
        has_quality_key = []

        for f in window_files[:100]:
            data = np.load(f)

            if 'quality' in data:
                has_quality_key.append(True)
                q = data['quality']

                # Handle different formats
                if hasattr(q, 'shape') and q.shape == ():
                    q = q.item()

                qualities.append(q)
            else:
                has_quality_key.append(False)

        # Report results
        print(f"  Has 'quality' key: {sum(has_quality_key)}/{len(has_quality_key)}")

        if qualities:
            # Count valid vs NaN
            valid_qualities = [q for q in qualities if not np.isnan(q)]
            nan_qualities = [q for q in qualities if np.isnan(q)]

            print(f"  Valid quality labels: {len(valid_qualities)}/{len(qualities)}")
            print(f"  NaN quality labels: {len(nan_qualities)}/{len(qualities)}")

            if valid_qualities:
                counter = Counter(valid_qualities)
                print(f"  Quality distribution (valid only):")
                for val, count in sorted(counter.items()):
                    print(f"      {int(val)}: {count} ({count/len(valid_qualities)*100:.1f}%)")
            else:
                print(f"  ✗ ALL QUALITY LABELS ARE NaN!")
                print(f"     This means quality-hr-ann.csv was missing when you ran prepare script")

        # Show sample NPZ contents
        if window_files:
            print(f"\n  Sample NPZ keys (first file):")
            sample_data = np.load(window_files[0])
            print(f"    {list(sample_data.keys())}")

            if 'quality' in sample_data:
                print(f"    quality value: {sample_data['quality']}")
                print(f"    quality type: {type(sample_data['quality'])}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Diagnose BUT-PPG label issues")
    parser.add_argument('--data-dir', type=str, default='data/but_ppg/dataset',
                       help='BUT-PPG raw dataset directory (with CSV files)')
    parser.add_argument('--output-dir', type=str, default='data/processed/butppg/windows_with_labels',
                       help='Processed windows directory')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    print("\nBUT-PPG LABEL DIAGNOSTICS")
    print("=" * 80)
    print(f"Raw data dir: {data_dir}")
    print(f"Output dir: {output_dir}")

    # Check annotation files
    check_annotation_files(data_dir)

    # Check NPZ files
    check_npz_files(output_dir)

    # Summary and recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    quality_hr_path = data_dir / 'quality-hr-ann.csv'

    if not quality_hr_path.exists():
        print("""
✗ CRITICAL: quality-hr-ann.csv is MISSING!

To fix:
1. Download complete BUT-PPG dataset with annotations:

   python scripts/download_butppg_dataset.py --output data/but_ppg/dataset

2. Re-run window preparation:

   rm -rf data/processed/butppg/windows_with_labels
   python scripts/create_butppg_windows_with_labels.py \\
       --data-dir data/but_ppg/dataset \\
       --output-dir data/processed/butppg/windows_with_labels \\
       --splits-file configs/splits/butppg_splits.json \\
       --window-sec 8.192 \\
       --fs 125

3. Re-run fine-tuning:

   python scripts/finetune_butppg.py \\
       --pretrained artifacts/foundation_model/best_model.pt \\
       --data-dir data/processed/butppg/windows_with_labels \\
       --epochs 30 \\
       --batch-size 64
""")
    else:
        print("""
✓ Annotation files exist!

If NPZ files still have NaN labels, you need to RE-RUN the preparation script:

   rm -rf data/processed/butppg/windows_with_labels
   python scripts/create_butppg_windows_with_labels.py \\
       --data-dir data/but_ppg/dataset \\
       --output-dir data/processed/butppg/windows_with_labels \\
       --splits-file configs/splits/butppg_splits.json \\
       --window-sec 8.192 \\
       --fs 125
""")


if __name__ == '__main__':
    main()
