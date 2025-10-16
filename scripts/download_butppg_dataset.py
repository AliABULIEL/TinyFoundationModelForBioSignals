#!/usr/bin/env python3
"""
BUT-PPG Dataset Downloader from PhysioNet

Downloads the complete BUT-PPG v2.0.0 dataset from PhysioNet.
Requires: wfdb package for PhysioNet data access

Usage:
    python scripts/download_butppg_dataset.py \
        --output-dir data/but_ppg/raw \
        --subjects all
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import wfdb
import pandas as pd
import numpy as np
from tqdm import tqdm


def download_butppg_metadata():
    """
    Download and parse BUT-PPG metadata

    The dataset structure:
    - 50 subjects (IDs: 100-149)
    - Each subject has multiple recordings (e.g., 100001, 100002, ...)
    - Annotations in CSV files
    """

    print("\n" + "="*80)
    print("BUT-PPG DATASET METADATA")
    print("="*80)

    # BUT-PPG subject IDs
    subject_ids = list(range(100, 150))  # Subjects 100-149

    print(f"Total subjects: {len(subject_ids)}")
    print(f"Subject ID range: {subject_ids[0]}-{subject_ids[-1]}")

    return subject_ids


def download_subject_recordings(
    subject_id: int,
    output_dir: Path,
    verbose: bool = True
):
    """
    Download all recordings for a specific subject

    Args:
        subject_id: Subject ID (100-149)
        output_dir: Output directory for raw data
        verbose: Print progress

    Returns:
        Number of recordings downloaded
    """
    output_dir = Path(output_dir) / f"subject_{subject_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # BUT-PPG records have format: SUBJECTID + recording number
    # e.g., subject 100 has recordings: 100001, 100002, ..., 100078

    downloaded = 0
    record_num = 1
    max_attempts = 200  # Max recordings per subject

    if verbose:
        print(f"\n📥 Downloading subject {subject_id}...")

    pbar = tqdm(total=max_attempts, desc=f"Subject {subject_id}", disable=not verbose, leave=False)

    while record_num < max_attempts:
        record_id = f"{subject_id}{record_num:03d}"

        try:
            # Download from PhysioNet
            # Format: database/version/record
            record = wfdb.rdrecord(
                record_name=record_id,
                pn_dir='butppg/2.0.0'
            )

            # Save to local directory
            record_path = output_dir / record_id
            wfdb.wrsamp(
                record_name=str(record_path),
                fs=record.fs,
                units=record.units,
                sig_name=record.sig_name,
                p_signal=record.p_signal,
                fmt=record.fmt
            )

            downloaded += 1
            pbar.update(1)

        except Exception as e:
            # No more recordings for this subject
            if "404" in str(e) or "not found" in str(e).lower():
                break
            else:
                if verbose:
                    print(f"  ⚠️  Error downloading {record_id}: {e}")

        record_num += 1

    pbar.close()

    if verbose:
        print(f"  ✓ Downloaded {downloaded} recordings")

    return downloaded


def download_annotations(output_dir: Path):
    """
    Download BUT-PPG annotation files

    Annotations include:
    - Quality labels (expert consensus)
    - Heart rate reference values
    - Motion type labels
    """

    print("\n📥 Downloading annotation files...")

    annotations_dir = Path(output_dir) / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    # Annotation files from PhysioNet
    annotation_files = [
        'PPGQualityLabels.csv',  # Quality labels
        'HRReference.csv',        # Heart rate reference
        'MotionLabels.csv'        # Motion type labels (if available)
    ]

    for ann_file in annotation_files:
        try:
            # Download using wget or requests
            url = f"https://physionet.org/files/butppg/2.0.0/{ann_file}"

            import urllib.request
            save_path = annotations_dir / ann_file

            print(f"  Downloading {ann_file}...")
            urllib.request.urlretrieve(url, save_path)
            print(f"  ✓ Saved to: {save_path}")

        except Exception as e:
            print(f"  ⚠️  Could not download {ann_file}: {e}")

    return annotations_dir


def verify_download(raw_dir: Path):
    """Verify downloaded data completeness"""

    print("\n" + "="*80)
    print("DOWNLOAD VERIFICATION")
    print("="*80)

    total_subjects = 0
    total_recordings = 0

    # Count subjects and recordings
    for subject_dir in sorted(raw_dir.glob("subject_*")):
        subject_id = subject_dir.name.replace("subject_", "")
        recordings = list(subject_dir.glob("*.dat"))

        if recordings:
            total_subjects += 1
            total_recordings += len(recordings)
            print(f"  Subject {subject_id}: {len(recordings)} recordings")

    print(f"\n✓ Total subjects: {total_subjects}")
    print(f"✓ Total recordings: {total_recordings}")

    # Check annotations
    ann_dir = raw_dir / "annotations"
    if ann_dir.exists():
        ann_files = list(ann_dir.glob("*.csv"))
        print(f"✓ Annotation files: {len(ann_files)}")
        for ann_file in ann_files:
            print(f"  - {ann_file.name}")
    else:
        print("⚠️  No annotation files found")

    return total_subjects, total_recordings


def main():
    parser = argparse.ArgumentParser(
        description="Download BUT-PPG dataset from PhysioNet"
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/but_ppg/raw',
        help='Output directory for raw data'
    )

    parser.add_argument(
        '--subjects',
        type=str,
        default='all',
        help='Subject IDs to download (comma-separated or "all")'
    )

    parser.add_argument(
        '--skip-if-exists',
        action='store_true',
        help='Skip downloading if data already exists'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("BUT-PPG DATASET DOWNLOADER")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"PhysioNet URL: https://physionet.org/content/butppg/2.0.0/")

    # Check if already downloaded
    if args.skip_if_exists and (output_dir / "annotations").exists():
        print("\n✓ Data already exists, skipping download")
        verify_download(output_dir)
        return

    # Get subject IDs
    if args.subjects == 'all':
        subject_ids = list(range(100, 150))
    else:
        subject_ids = [int(s.strip()) for s in args.subjects.split(',')]

    print(f"\nSubjects to download: {len(subject_ids)}")

    # Download annotations first
    download_annotations(output_dir)

    # Download subject recordings
    print("\n" + "="*80)
    print("DOWNLOADING SUBJECT RECORDINGS")
    print("="*80)

    total_downloaded = 0

    for subject_id in subject_ids:
        downloaded = download_subject_recordings(
            subject_id,
            output_dir,
            verbose=True
        )
        total_downloaded += downloaded

    # Verify download
    verify_download(output_dir)

    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE!")
    print("="*80)
    print(f"✓ Downloaded {total_downloaded} recordings")
    print(f"✓ Data saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Process raw data: python scripts/process_butppg_clinical.py")
    print("  2. Create task datasets: python scripts/prepare_butppg_tasks.py")


if __name__ == '__main__':
    main()
