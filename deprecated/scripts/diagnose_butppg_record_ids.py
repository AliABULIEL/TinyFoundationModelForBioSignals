#!/usr/bin/env python3
"""
Diagnostic script to find record ID mismatch issues in BUT-PPG dataset.

This script will:
1. Check CSV file structure
2. List sample record IDs from CSVs
3. List sample record IDs from data files
4. Compare formats and find mismatches
5. Suggest fixes
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np


def diagnose_csv_files(data_dir: Path):
    """Diagnose CSV annotation files."""

    print("="*80)
    print("1. CHECKING CSV FILES")
    print("="*80)

    # Check quality-hr-ann.csv
    quality_hr_path = data_dir / 'quality-hr-ann.csv'

    if not quality_hr_path.exists():
        print(f"\n❌ quality-hr-ann.csv NOT FOUND at: {quality_hr_path}")
        print(f"\nThis is why all labels are NaN!")
        print(f"\nSOLUTION: Download complete dataset:")
        print(f"  python scripts/download_butppg_dataset.py --output data/but_ppg/dataset")
        return None, None

    print(f"\n✓ Found: {quality_hr_path}")

    # Load and inspect
    df = pd.read_csv(quality_hr_path)

    print(f"\nColumns: {list(df.columns)}")
    print(f"Total rows: {len(df)}")

    # Show first few rows
    print(f"\nFirst 5 rows:")
    print(df.head())

    # Check record ID column
    if 'record' in df.columns:
        print(f"\n✓ Has 'record' column")
        print(f"  Record ID type: {df['record'].dtype}")
        print(f"  Sample IDs: {df['record'].head(10).tolist()}")

        # Check for NaN
        nan_count = df['record'].isna().sum()
        if nan_count > 0:
            print(f"  ⚠️  {nan_count} NaN record IDs")
    else:
        print(f"\n❌ NO 'record' column! Found: {list(df.columns)}")

    # Check quality column
    if 'quality' in df.columns:
        print(f"\n✓ Has 'quality' column")
        print(f"  Quality values: {df['quality'].unique()}")
        print(f"  Distribution:")
        print(df['quality'].value_counts())
    else:
        print(f"\n❌ NO 'quality' column! Found: {list(df.columns)}")

    # Check hr column
    if 'hr' in df.columns:
        print(f"\n✓ Has 'hr' column")
        print(f"  HR range: {df['hr'].min():.1f} - {df['hr'].max():.1f} BPM")
    else:
        print(f"\n❌ NO 'hr' column! Found: {list(df.columns)}")

    # Check subject-info.csv
    print("\n" + "="*80)
    subject_info_path = data_dir / 'subject-info.csv'

    if subject_info_path.exists():
        print(f"✓ Found: {subject_info_path}")

        df_subj = pd.read_csv(subject_info_path)

        print(f"\nColumns: {list(df_subj.columns)}")
        print(f"Total rows: {len(df_subj)}")
        print(f"\nFirst 5 rows:")
        print(df_subj.head())
    else:
        print(f"❌ subject-info.csv NOT FOUND")
        df_subj = None

    return df, df_subj


def diagnose_data_files(data_dir: Path):
    """Diagnose actual data files."""

    print("\n" + "="*80)
    print("2. CHECKING DATA FILES")
    print("="*80)

    # Find PPG files (in subdirectories)
    ppg_files = list(data_dir.glob("*/*_PPG.dat"))

    if len(ppg_files) == 0:
        # Try alternate structure
        ppg_files = list(data_dir.glob("*_PPG.dat"))

    if len(ppg_files) == 0:
        print(f"\n❌ NO DATA FILES FOUND in {data_dir}")
        print(f"\nExpected structure:")
        print(f"  {data_dir}/")
        print(f"    100001/")
        print(f"      100001_PPG.dat")
        print(f"      100001_ECG.dat")
        print(f"    100002/")
        print(f"      100002_PPG.dat")
        print(f"      ...")
        return []

    print(f"\n✓ Found {len(ppg_files)} PPG files")

    # Extract record IDs
    record_ids = []
    for f in ppg_files[:10]:  # Sample first 10
        # Extract ID from filename (e.g., "100001" from "100001_PPG.dat")
        record_id = f.stem.split('_')[0]
        record_ids.append(record_id)

    print(f"\nSample record IDs from files (first 10):")
    for rid in record_ids:
        print(f"  {rid} (type: {type(rid).__name__})")

    return record_ids


def compare_record_ids(csv_df: pd.DataFrame, file_record_ids: list):
    """Compare record IDs between CSV and files."""

    print("\n" + "="*80)
    print("3. COMPARING RECORD IDs")
    print("="*80)

    if csv_df is None or 'record' not in csv_df.columns:
        print("\n❌ Cannot compare - CSV missing or no 'record' column")
        return

    if not file_record_ids:
        print("\n❌ Cannot compare - no data files found")
        return

    # Get CSV record IDs
    csv_records = csv_df['record'].tolist()

    print(f"\nCSV record IDs (first 10):")
    for rid in csv_records[:10]:
        print(f"  {rid} (type: {type(rid).__name__})")

    print(f"\nFile record IDs (first 10):")
    for rid in file_record_ids:
        print(f"  {rid} (type: {type(rid).__name__})")

    # Try matching
    print(f"\n" + "-"*80)
    print("MATCHING TEST")
    print("-"*80)

    # Set index for fast lookup
    csv_indexed = csv_df.set_index('record')

    # Test each file record ID
    matches = 0
    mismatches = 0

    for file_id in file_record_ids[:10]:
        # Try different formats
        str_id = str(file_id)
        try:
            int_id = int(file_id)
        except:
            int_id = None

        # Check if in CSV
        in_csv_str = str_id in csv_indexed.index
        in_csv_int = int_id in csv_indexed.index if int_id is not None else False

        print(f"\nFile ID: '{file_id}'")
        print(f"  As string '{str_id}': {'✓ FOUND' if in_csv_str else '✗ NOT FOUND'} in CSV")
        if int_id is not None:
            print(f"  As int {int_id}: {'✓ FOUND' if in_csv_int else '✗ NOT FOUND'} in CSV")

        if in_csv_str or in_csv_int:
            matches += 1

            # Show the label
            if in_csv_str:
                row = csv_indexed.loc[str_id]
            else:
                row = csv_indexed.loc[int_id]

            if 'quality' in row:
                print(f"  Quality: {row['quality']}")
            if 'hr' in row:
                print(f"  HR: {row['hr']}")
        else:
            mismatches += 1

    print(f"\n" + "="*80)
    print(f"RESULTS: {matches} matches, {mismatches} mismatches out of {len(file_record_ids[:10])} tested")
    print("="*80)

    if mismatches > 0:
        print(f"\n⚠️  RECORD ID MISMATCH DETECTED!")
        print(f"\nPossible causes:")
        print(f"  1. CSV has integer IDs, code expects strings (or vice versa)")
        print(f"  2. Different ID format (e.g., '100001' vs '100-001')")
        print(f"  3. CSV IDs don't match actual data files")

        # Suggest fix
        print(f"\nSUGGESTED FIX:")
        print(f"  Check the actual CSV content and ensure IDs match:")
        print(f"  1. CSV record column: {csv_df['record'].dtype}")
        print(f"  2. File IDs are strings")
        print(f"  3. Update create_butppg_windows_with_labels.py line 270:")
        print(f"     Change: record_id in quality_hr_df.index")
        print(f"     To: str(record_id) in quality_hr_df.index OR int(record_id) in quality_hr_df.index")


def test_actual_lookup():
    """Test the actual lookup logic from the script."""

    print("\n" + "="*80)
    print("4. TESTING ACTUAL LOOKUP LOGIC")
    print("="*80)

    from deprecated.scripts.create_butppg_windows_with_labels import (
        load_csv_annotations,
        get_recording_labels
    )

    # Try with user's data
    data_dir = Path('data/but_ppg/dataset')

    if not data_dir.exists():
        print(f"\n⚠️  Cannot test - data directory not found: {data_dir}")
        return

    quality_hr_df, subject_info_df = load_csv_annotations(data_dir)

    if quality_hr_df is None:
        print(f"\n❌ CSV files not loaded")
        return

    # Test with sample IDs
    test_ids = ['100001', '100002', '100003', 100001, 100002]

    for test_id in test_ids:
        labels = get_recording_labels(test_id, quality_hr_df, subject_info_df)

        print(f"\nLookup '{test_id}' (type: {type(test_id).__name__}):")
        print(f"  Quality: {labels['quality']} (NaN: {np.isnan(labels['quality'])})")
        print(f"  HR: {labels['hr']} (NaN: {np.isnan(labels['hr'])})")
        print(f"  Motion: {labels['motion']} (NaN: {np.isnan(labels['motion'])})")


def main():
    parser = argparse.ArgumentParser(description="Diagnose BUT-PPG record ID issues")
    parser.add_argument('--data-dir', type=str, default='data/but_ppg/dataset',
                       help='BUT-PPG dataset directory')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("="*80)
    print("BUT-PPG RECORD ID DIAGNOSTIC TOOL")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print()

    # Diagnose CSVs
    csv_df, subject_df = diagnose_csv_files(data_dir)

    # Diagnose data files
    file_record_ids = diagnose_data_files(data_dir)

    # Compare
    if csv_df is not None and file_record_ids:
        compare_record_ids(csv_df, file_record_ids)

    # Test actual lookup
    test_actual_lookup()

    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
