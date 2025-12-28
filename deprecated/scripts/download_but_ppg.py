#!/usr/bin/env python3
"""
BUT-PPG Database Download Script

Downloads the complete BUT-PPG v2.0.0 dataset from PhysioNet:
- 50 subjects (25 male, 25 female), ages 19-76 years
- 3,888 10-second recordings
- Signals: PPG, ECG, ACC (3-axis accelerometer)
- Annotations for 3 downstream clinical tasks:
  1. Signal Quality Classification (binary: good/poor)
  2. Heart Rate Estimation (regression: BPM)
  3. Motion Type Classification (8 classes)

Dataset URL: https://physionet.org/content/butppg/2.0.0/
License: CC BY 4.0 (Open Access)

Usage:
    # Download complete dataset
    python scripts/download_but_ppg.py

    # Inspect annotations after download
    python scripts/download_but_ppg.py --inspect

    # Verify dataset completeness for downstream tasks
    python scripts/download_but_ppg.py --verify-tasks
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import requests
import zipfile
import shutil
from tqdm import tqdm
import json
import time
import warnings

warnings.filterwarnings('ignore')

# ==================== Configuration ====================

# PhysioNet BUT PPG Database URL
PHYSIONET_ZIP = "https://physionet.org/static/published-projects/butppg/brno-university-of-technology-smartphone-ppg-database-but-ppg-2.0.0.zip"

# Alternative: Use wget command if download fails
WGET_COMMAND = "wget -r -N -c -np https://physionet.org/files/butppg/2.0.0/"

# Output directories
DATA_DIR = Path("data/but_ppg")
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = Path("data/outputs")


# ==================== Download Functions ====================

def download_file(url: str, dest_path: Path, desc: str = "Downloading") -> bool:
    """Download file with progress bar."""
    try:
        # Create parent directory if needed
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Stream download with progress
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"❌ Error downloading: {e}")
        return False


def extract_database(zip_path: Path, extract_dir: Path) -> bool:
    """Extract the downloaded database."""
    try:
        print("\n📦 Extracting database...")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get all members
            members = zip_ref.namelist()

            # Extract with progress bar
            with tqdm(total=len(members), desc="Extracting files") as pbar:
                for member in members:
                    zip_ref.extract(member, extract_dir)
                    pbar.update(1)

        print(f"  ✓ Extracted {len(members)} files")
        return True

    except Exception as e:
        print(f"  ❌ Extraction failed: {e}")
        return False


def organize_data(extract_dir: Path) -> Path:
    """Organize extracted data into clean structure."""
    print("\n🗂️ Organizing data...")

    # Find the actual data directory (should be one level down)
    data_dirs = list(extract_dir.glob("*"))
    if not data_dirs:
        raise FileNotFoundError("No extracted data found")

    # The extracted folder structure might have nested directories
    actual_data_dir = None
    for d in data_dirs:
        if d.is_dir():
            # Check if this directory contains the data
            if (d / "quality-hr-ann.csv").exists():
                actual_data_dir = d
                break
            # Check one level deeper
            subdirs = list(d.glob("*"))
            for sd in subdirs:
                if sd.is_dir() and (sd / "quality-hr-ann.csv").exists():
                    actual_data_dir = sd
                    break

    if actual_data_dir is None:
        raise FileNotFoundError("Could not find data files in extracted directory")

    print(f"  Found data in: {actual_data_dir.name}")

    # Move to cleaner location if needed
    final_data_dir = DATA_DIR / "dataset"
    if actual_data_dir != final_data_dir:
        if final_data_dir.exists():
            shutil.rmtree(final_data_dir)
        shutil.move(str(actual_data_dir), str(final_data_dir))
        actual_data_dir = final_data_dir

    return actual_data_dir


def inspect_annotations(data_dir: Path) -> dict:
    """
    Inspect annotation files and validate for downstream tasks.

    Returns dict with task readiness status.
    """
    print("\n" + "="*80)
    print("📋 ANNOTATION INSPECTION")
    print("="*80)

    task_status = {
        'quality_classification': {'ready': False, 'issues': []},
        'heart_rate_regression': {'ready': False, 'issues': []},
        'motion_classification': {'ready': False, 'issues': []}
    }

    # Check quality-hr-ann.csv
    quality_file = data_dir / "quality-hr-ann.csv"
    if quality_file.exists():
        print(f"\n✅ Found: {quality_file.name}")

        # Read with actual headers
        quality_df = pd.read_csv(quality_file)
        print(f"   Records: {len(quality_df)}")
        print(f"   Columns: {list(quality_df.columns)}")

        # Show first few rows
        print(f"\n   Sample data:")
        print(quality_df.head(3).to_string(index=False, max_colwidth=20))

        # Detect quality column (flexible matching)
        quality_col = None
        for col in quality_df.columns:
            if col.lower() in ['quality', 'ppg_quality', 'signal_quality', 'ppgquality']:
                quality_col = col
                break

        if quality_col:
            quality_counts = quality_df[quality_col].value_counts()
            print(f"\n   Quality distribution ({quality_col}):")
            for val, count in quality_counts.items():
                print(f"     {val}: {count}")
            task_status['quality_classification']['ready'] = True
        else:
            task_status['quality_classification']['issues'].append(
                f"No quality column found. Columns: {list(quality_df.columns)}"
            )

        # Detect HR column (flexible matching)
        hr_col = None
        for col in quality_df.columns:
            if col.lower() in ['hr', 'heart_rate', 'heartrate', 'reference_hr', 'hr_reference']:
                hr_col = col
                break

        if hr_col:
            hr_stats = quality_df[hr_col].describe()
            print(f"\n   Heart Rate statistics ({hr_col}):")
            print(f"     Min: {hr_stats['min']:.1f} BPM")
            print(f"     Max: {hr_stats['max']:.1f} BPM")
            print(f"     Mean: {hr_stats['mean']:.1f} BPM")
            print(f"     Std: {hr_stats['std']:.1f} BPM")
            task_status['heart_rate_regression']['ready'] = True
        else:
            task_status['heart_rate_regression']['issues'].append(
                f"No HR column found. Columns: {list(quality_df.columns)}"
            )
    else:
        print(f"\n❌ Missing: quality-hr-ann.csv")
        task_status['quality_classification']['issues'].append("File not found")
        task_status['heart_rate_regression']['issues'].append("File not found")

    # Check subject-info.csv
    subject_file = data_dir / "subject-info.csv"
    if subject_file.exists():
        print(f"\n✅ Found: {subject_file.name}")

        subject_df = pd.read_csv(subject_file)
        print(f"   Records: {len(subject_df)}")
        print(f"   Columns: {list(subject_df.columns)}")

        # Show first few rows
        print(f"\n   Sample data:")
        print(subject_df.head(3).to_string(index=False, max_colwidth=20))

        # Detect motion column (flexible matching)
        motion_col = None
        for col in subject_df.columns:
            if col.lower() in ['motion', 'motion_type', 'motiontype', 'activity', 'activity_type']:
                motion_col = col
                break

        if motion_col:
            motion_counts = subject_df[motion_col].value_counts()
            print(f"\n   Motion distribution ({motion_col}):")
            for motion, count in motion_counts.items():
                print(f"     {motion}: {count}")
            print(f"   Total classes: {subject_df[motion_col].nunique()}")
            task_status['motion_classification']['ready'] = True
        else:
            task_status['motion_classification']['issues'].append(
                f"No motion column found. Columns: {list(subject_df.columns)}"
            )

        # Show demographics if available
        demo_cols = []
        for col in subject_df.columns:
            if col.lower() in ['age', 'gender', 'height', 'weight', 'bmi']:
                demo_cols.append(col)

        if demo_cols:
            print(f"\n   Demographics available: {demo_cols}")
            for col in demo_cols:
                if col.lower() == 'gender':
                    print(f"     {col}: {subject_df[col].value_counts().to_dict()}")
                elif subject_df[col].dtype in ['int64', 'float64']:
                    print(f"     {col}: {subject_df[col].min():.1f} - {subject_df[col].max():.1f}")
    else:
        print(f"\n❌ Missing: subject-info.csv")
        task_status['motion_classification']['issues'].append("File not found")

    print("\n" + "="*80)

    return task_status


def verify_downstream_tasks(data_dir: Path) -> bool:
    """
    Verify that all 3 downstream tasks are possible with this dataset.

    Returns True if all tasks ready, False otherwise.
    """
    print("\n" + "="*80)
    print("🎯 DOWNSTREAM TASK VERIFICATION")
    print("="*80)

    task_status = inspect_annotations(data_dir)

    print("\n📊 Task Readiness Summary:")
    print("-" * 80)

    all_ready = True

    for task_name, status in task_status.items():
        task_display = task_name.replace('_', ' ').title()

        if status['ready']:
            print(f"✅ {task_display}: READY")
        else:
            print(f"❌ {task_display}: NOT READY")
            for issue in status['issues']:
                print(f"   Issue: {issue}")
            all_ready = False

    print("-" * 80)

    if all_ready:
        print("\n✅ ALL DOWNSTREAM TASKS READY!")
        print("   You can proceed with:")
        print("   1. Data preprocessing: python scripts/process_butppg_clinical.py")
        print("   2. Fine-tuning: python scripts/finetune_butppg.py")
        print("   3. Evaluation: python scripts/run_downstream_evaluation.py")
    else:
        print("\n⚠️  SOME TASKS NOT READY")
        print("   Check annotation files for missing columns")
        print("   Expected files:")
        print("     - quality-hr-ann.csv (quality + heart rate)")
        print("     - subject-info.csv (motion + demographics)")

    print("="*80)

    return all_ready


def verify_dataset(data_dir: Path) -> dict:
    """Verify the downloaded dataset and gather statistics."""
    print("\n✅ Verifying dataset...")

    stats = {
        'data_dir': str(data_dir),
        'total_records': 0,
        'has_ppg': 0,
        'has_ecg': 0,
        'has_acc': 0,
        'has_annotations': False,
        'has_subject_info': False,
        'subjects': set(),
        'quality_good': 0,
        'quality_poor': 0
    }

    # Check annotation files
    quality_file = data_dir / "quality-hr-ann.csv"
    subject_file = data_dir / "subject-info.csv"

    if quality_file.exists():
        stats['has_annotations'] = True
        # Read with actual headers (don't assume column names)
        quality_df = pd.read_csv(quality_file)

        # Flexible quality column detection
        quality_col = None
        for col in quality_df.columns:
            if col.lower() in ['quality', 'ppg_quality', 'signal_quality']:
                quality_col = col
                break

        if quality_col:
            stats['quality_good'] = int((quality_df[quality_col] == 1).sum())
            stats['quality_poor'] = int((quality_df[quality_col] == 0).sum())
            print(f"  ✓ Quality annotations: {len(quality_df)} records")
            print(f"    - Good quality: {stats['quality_good']}")
            print(f"    - Poor quality: {stats['quality_poor']}")

    if subject_file.exists():
        stats['has_subject_info'] = True
        subject_df = pd.read_csv(subject_file)
        print(f"  ✓ Subject info: {len(subject_df)} records")

        # Print demographic summary
        if 'Age' in subject_df.columns:
            print(f"    - Age range: {subject_df['Age'].min()}-{subject_df['Age'].max()} years")
        if 'Gender' in subject_df.columns:
            gender_counts = subject_df['Gender'].value_counts()
            print(f"    - Gender distribution: {gender_counts.to_dict()}")

    # Count record directories
    record_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    stats['total_records'] = len(record_dirs)

    # Sample check for different signal types
    print(f"\n  Checking {min(10, len(record_dirs))} sample records...")
    for i, record_dir in enumerate(record_dirs[:10]):
        record_id = record_dir.name
        stats['subjects'].add(record_id[:3])  # First 3 digits are subject ID

        # Check for signal files
        if (record_dir / f"{record_id}_PPG.dat").exists():
            stats['has_ppg'] += 1
        if (record_dir / f"{record_id}_ECG.dat").exists():
            stats['has_ecg'] += 1
        if (record_dir / f"{record_id}_ACC.dat").exists():
            stats['has_acc'] += 1

    # Extrapolate from sample
    if len(record_dirs) > 10:
        scale = len(record_dirs) / 10
        stats['has_ppg'] = int(stats['has_ppg'] * scale)
        stats['has_ecg'] = int(stats['has_ecg'] * scale)
        stats['has_acc'] = int(stats['has_acc'] * scale)

    stats['subjects'] = len(stats['subjects'])

    print(f"\n  📊 Dataset Statistics:")
    print(f"    - Total records: {stats['total_records']}")
    print(f"    - Estimated subjects: ~{stats['subjects'] * 5}")  # Rough estimate
    print(f"    - Records with PPG: ~{stats['has_ppg']}")
    print(f"    - Records with ECG: ~{stats['has_ecg']}")
    print(f"    - Records with ACC: ~{stats['has_acc']}")

    return stats


def create_index_files(data_dir: Path, stats: dict):
    """Create index CSV files for easy data loading."""
    print("\n📝 Creating index files...")
    
    # Get all record directories
    record_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    
    # Create waveform index
    waveform_records = []
    
    for record_dir in record_dirs:
        record_id = record_dir.name
        subject_id = record_id[:3]  # First 3 digits
        
        # Check available modalities
        has_ppg = (record_dir / f"{record_id}_PPG.dat").exists()
        has_ecg = (record_dir / f"{record_id}_ECG.dat").exists() 
        has_acc = (record_dir / f"{record_id}_ACC.dat").exists()
        
        if has_ppg or has_ecg or has_acc:
            waveform_records.append({
                'record_id': record_id,
                'subject_id': subject_id,
                'has_ppg': has_ppg,
                'has_ecg': has_ecg,
                'has_acc': has_acc,
                'path': str(record_dir)
            })
    
    # Save waveform index
    waveform_df = pd.DataFrame(waveform_records)
    waveform_path = OUTPUT_DIR / "waveform_index.csv"
    waveform_df.to_csv(waveform_path, index=False)
    print(f"  ✓ Created {waveform_path} with {len(waveform_df)} records")
    
    # Load and process subject info if available
    subject_file = data_dir / "subject-info.csv"
    if subject_file.exists():
        subject_df = pd.read_csv(subject_file)
        
        # Rename columns to standard names
        rename_map = {
            'ID': 'record_id',
            'Age [years]': 'age',
            'Gender': 'gender',
            'Height [cm]': 'height',
            'Weight [kg]': 'weight'
        }
        
        # Only rename columns that exist
        rename_map = {k: v for k, v in rename_map.items() if k in subject_df.columns}
        subject_df = subject_df.rename(columns=rename_map)
        
        # Add subject_id
        if 'record_id' in subject_df.columns:
            subject_df['subject_id'] = subject_df['record_id'].astype(str).str[:3]
        
        # Calculate BMI if height and weight available
        if 'height' in subject_df.columns and 'weight' in subject_df.columns:
            subject_df['bmi'] = subject_df['weight'] / ((subject_df['height'] / 100) ** 2)
        
        # Save labels
        labels_path = OUTPUT_DIR / "labels.csv"
        subject_df.to_csv(labels_path, index=False)
        print(f"  ✓ Created {labels_path} with {len(subject_df)} subjects")
    
    # Save dataset metadata
    metadata = {
        'dataset': 'BUT PPG',
        'version': '2.0.0',
        'download_date': time.strftime('%Y-%m-%d'),
        'stats': stats,
        'files': {
            'waveform_index': str(waveform_path),
            'labels': str(labels_path) if subject_file.exists() else None,
            'data_dir': str(data_dir)
        }
    }
    
    metadata_path = OUTPUT_DIR / "dataset_info.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Created {metadata_path}")


# ==================== Main Function ====================

def main():
    """Main download function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and verify BUT-PPG dataset for downstream clinical tasks"
    )
    parser.add_argument(
        '--inspect',
        action='store_true',
        help='Inspect annotation files and exit (requires dataset to be downloaded)'
    )
    parser.add_argument(
        '--verify-tasks',
        action='store_true',
        help='Verify all 3 downstream tasks are ready and exit'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip download, only run inspection/verification'
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("📥 BUT PPG DATABASE DOWNLOAD")
    print("=" * 60)
    print("Dataset: Smartphone PPG with ECG, ACC, and demographics")
    print("Source: PhysioNet")

    # If inspection/verification only, find existing dataset
    if args.inspect or args.verify_tasks or args.skip_download:
        # Try to find dataset
        possible_dirs = [
            DATA_DIR / "dataset",
            RAW_DIR / "extracted",
            Path("data/but_ppg/raw/brno-university-of-technology-smartphone-ppg-database-but-ppg-2.0.0")
        ]

        data_dir = None
        for d in possible_dirs:
            if d.exists() and (d / "quality-hr-ann.csv").exists():
                data_dir = d
                break

        if data_dir is None:
            print("\n❌ Dataset not found!")
            print("   Please run without --inspect or --verify-tasks to download first")
            print(f"   Checked: {possible_dirs}")
            return False

        print(f"\nFound dataset at: {data_dir}")

        # Run requested inspection
        if args.inspect:
            inspect_annotations(data_dir)
            return True

        if args.verify_tasks:
            return verify_downstream_tasks(data_dir)

        # If skip-download, just verify and exit
        if args.skip_download:
            verify_dataset(data_dir)
            return True

    start_time = time.time()

    # Create directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Download
        print(f"\n📥 Step 1: Downloading database (~87 MB)...")
        zip_path = RAW_DIR / "but_ppg.zip"

        if zip_path.exists():
            print("  ✓ Already downloaded")
        else:
            success = download_file(PHYSIONET_ZIP, zip_path, "BUT PPG Database")
            if not success:
                print("\n💡 Alternative: Try running this command:")
                print(f"  {WGET_COMMAND}")
                return False

        # Step 2: Extract
        print(f"\n📦 Step 2: Extracting database...")
        extract_dir = RAW_DIR / "extracted"

        if extract_dir.exists() and any(extract_dir.iterdir()):
            print("  ✓ Already extracted")
            # Find the data directory
            data_dir = organize_data(extract_dir)
        else:
            success = extract_database(zip_path, extract_dir)
            if not success:
                return False
            data_dir = organize_data(extract_dir)

        # Step 3: Verify
        print(f"\n✅ Step 3: Verifying dataset...")
        stats = verify_dataset(data_dir)

        # Step 4: Create index files
        print(f"\n📝 Step 4: Creating index files...")
        create_index_files(data_dir, stats)

        # Calculate time
        elapsed = time.time() - start_time

        # Print summary
        print("\n" + "=" * 60)
        print("✅ DOWNLOAD COMPLETE!")
        print("=" * 60)

        print(f"\n⏱️ Total time: {elapsed:.1f} seconds")

        print("\n📁 Output files created:")
        print(f"  • {OUTPUT_DIR}/waveform_index.csv - Index of all records")
        print(f"  • {OUTPUT_DIR}/labels.csv - Demographics and health data")
        print(f"  • {OUTPUT_DIR}/dataset_info.json - Dataset metadata")
        print(f"  • {data_dir}/ - Raw signal data")

        # Step 5: Verify downstream tasks
        print(f"\n🎯 Step 5: Verifying downstream tasks...")
        all_tasks_ready = verify_downstream_tasks(data_dir)

        print("\n🚀 Next Steps:")
        if all_tasks_ready:
            print("1. ✅ All clinical tasks ready!")
            print("2. Process data: python scripts/process_butppg_clinical.py")
            print("3. Fine-tune model: python scripts/finetune_butppg.py")
            print("4. Run evaluation: python scripts/run_downstream_evaluation.py")
        else:
            print("1. ⚠️  Some tasks not ready - check annotation files")
            print("2. Re-run with --inspect to see details")
            print("3. Use waveform_index.csv for signals")
            print("4. Use labels.csv for available tasks")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
