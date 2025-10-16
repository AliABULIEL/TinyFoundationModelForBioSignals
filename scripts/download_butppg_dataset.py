#!/usr/bin/env python3
"""
BUT-PPG Dataset Downloader from PhysioNet

Downloads the complete BUT-PPG v2.0.0 dataset (86.7 MB) from PhysioNet.
Dataset contains 3,888 10-second recordings from 50 subjects.

Usage:
    python scripts/download_butppg_dataset.py \
        --output-dir data/but_ppg/raw \
        --method zip
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import urllib.request
import zipfile
import shutil
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path, desc="Downloading"):
    """Download file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_butppg_zip(output_dir: Path, skip_if_exists: bool = False):
    """
    Download BUT-PPG dataset as ZIP file

    Args:
        output_dir: Output directory for raw data
        skip_if_exists: Skip if already downloaded

    Returns:
        Path to downloaded ZIP file
    """

    print("\n" + "="*80)
    print("DOWNLOADING BUT-PPG DATASET")
    print("="*80)

    # PhysioNet ZIP URL
    zip_url = "https://physionet.org/static/published-projects/butppg/but-ppg-an-annotated-photoplethysmography-dataset-2.0.0.zip"
    zip_filename = "but-ppg-2.0.0.zip"
    zip_path = output_dir / zip_filename

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if skip_if_exists and zip_path.exists():
        print(f"✓ ZIP file already exists: {zip_path}")
        return zip_path

    print(f"Source: {zip_url}")
    print(f"Destination: {zip_path}")
    print(f"Size: 86.7 MB")
    print()

    try:
        download_url(zip_url, zip_path, desc="Downloading BUT-PPG ZIP")
        print(f"\n✓ Downloaded: {zip_path}")
        print(f"  Size: {zip_path.stat().st_size / 1024 / 1024:.1f} MB")
        return zip_path

    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\n💡 Alternative download methods:")
        print(f"   1. wget: wget -O {zip_path} {zip_url}")
        print(f"   2. curl: curl -o {zip_path} {zip_url}")
        print(f"   3. Browser: Download from https://physionet.org/content/butppg/2.0.0/")
        raise


def extract_butppg_zip(zip_path: Path, output_dir: Path):
    """
    Extract BUT-PPG ZIP file

    Args:
        zip_path: Path to ZIP file
        output_dir: Output directory

    Returns:
        Path to extracted dataset
    """

    print("\n" + "="*80)
    print("EXTRACTING BUT-PPG DATASET")
    print("="*80)

    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    print(f"Extracting: {zip_path}")
    print(f"Destination: {output_dir}")
    print()

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files
            file_list = zip_ref.namelist()
            print(f"Total files in ZIP: {len(file_list)}")

            # Extract with progress bar
            for file in tqdm(file_list, desc="Extracting files"):
                zip_ref.extract(file, output_dir)

        print(f"\n✓ Extraction complete!")

        # Find the extracted dataset directory
        # PhysioNet ZIPs typically extract to: but-ppg-an-annotated-photoplethysmography-dataset-2.0.0/
        extracted_dirs = [d for d in output_dir.iterdir() if d.is_dir() and 'but-ppg' in d.name.lower()]

        if extracted_dirs:
            dataset_dir = extracted_dirs[0]
            print(f"  Dataset extracted to: {dataset_dir}")
            return dataset_dir
        else:
            print(f"  Dataset extracted to: {output_dir}")
            return output_dir

    except Exception as e:
        print(f"\n❌ Extraction failed: {e}")
        raise


def organize_dataset(dataset_dir: Path, output_dir: Path):
    """
    Organize extracted dataset into cleaner structure

    Args:
        dataset_dir: Extracted dataset directory
        output_dir: Organized output directory
    """

    print("\n" + "="*80)
    print("ORGANIZING DATASET")
    print("="*80)

    # Expected structure after extraction:
    # but-ppg-an-annotated-photoplethysmography-dataset-2.0.0/
    # ├── 100001_ACC.dat, 100001_ACC.hea
    # ├── 100001_ECG.dat, 100001_ECG.hea
    # ├── 100001_PPG.dat, 100001_PPG.hea
    # ├── quality-hr-ann.csv
    # ├── subject-info.csv
    # └── ...

    # Create organized structure
    recordings_dir = output_dir / "recordings"
    annotations_dir = output_dir / "annotations"
    recordings_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    # Move annotation files
    ann_files = ['quality-hr-ann.csv', 'subject-info.csv']
    for ann_file in ann_files:
        src = dataset_dir / ann_file
        dst = annotations_dir / ann_file

        if src.exists():
            shutil.copy2(src, dst)
            print(f"  ✓ Copied: {ann_file}")
        else:
            print(f"  ⚠️  Not found: {ann_file}")

    # Count and organize recording files
    dat_files = list(dataset_dir.glob("*.dat"))
    hea_files = list(dataset_dir.glob("*.hea"))
    qrs_files = list(dataset_dir.glob("*.qrs"))

    print(f"\n  Total .dat files: {len(dat_files)}")
    print(f"  Total .hea files: {len(hea_files)}")
    print(f"  Total .qrs files: {len(qrs_files)}")

    # Copy recording files (optional - can just point to dataset_dir)
    print(f"\n  Recording files remain in: {dataset_dir}")
    print(f"  Annotations copied to: {annotations_dir}")

    # Create a symlink or note about recordings location
    readme_path = output_dir / "README.txt"
    with open(readme_path, 'w') as f:
        f.write("BUT-PPG Dataset Structure\n")
        f.write("="*80 + "\n\n")
        f.write(f"Recordings: {dataset_dir.relative_to(output_dir.parent) if dataset_dir.is_relative_to(output_dir.parent) else dataset_dir}\n")
        f.write(f"Annotations: {annotations_dir.relative_to(output_dir.parent)}\n\n")
        f.write("Dataset Contents:\n")
        f.write(f"  - 3,888 recordings (10-second signals)\n")
        f.write(f"  - 50 subjects (25 male, 25 female)\n")
        f.write(f"  - Signals: PPG, ECG, ACC (3-axis)\n")
        f.write(f"  - Annotations: quality-hr-ann.csv, subject-info.csv\n")

    print(f"  ✓ Created: {readme_path}")

    return dataset_dir, annotations_dir


def verify_dataset(dataset_dir: Path, annotations_dir: Path):
    """Verify downloaded dataset completeness"""

    print("\n" + "="*80)
    print("DATASET VERIFICATION")
    print("="*80)

    # Count files
    ppg_files = list(dataset_dir.glob("*_PPG.dat"))
    ecg_files = list(dataset_dir.glob("*_ECG.dat"))
    acc_files = list(dataset_dir.glob("*_ACC.dat"))

    print(f"\n📊 Recording files:")
    print(f"  PPG files: {len(ppg_files)}")
    print(f"  ECG files: {len(ecg_files)}")
    print(f"  ACC files: {len(acc_files)}")

    # Count unique recordings
    unique_records = set()
    for f in ppg_files:
        record_id = f.stem.split('_')[0]  # e.g., "100001" from "100001_PPG"
        unique_records.add(record_id)

    print(f"\n  Total unique recordings: {len(unique_records)}")
    print(f"  Expected: 3,888 recordings")

    # Check annotations
    print(f"\n📋 Annotation files:")
    for ann_file in ['quality-hr-ann.csv', 'subject-info.csv']:
        ann_path = annotations_dir / ann_file
        if ann_path.exists():
            # Count lines
            with open(ann_path) as f:
                num_lines = sum(1 for _ in f) - 1  # Subtract header
            print(f"  ✓ {ann_file}: {num_lines} entries")
        else:
            print(f"  ❌ {ann_file}: NOT FOUND")

    # Success check
    if len(unique_records) >= 3800:  # Allow some margin
        print(f"\n✅ Dataset verification PASSED")
        print(f"   Downloaded {len(unique_records)} recordings")
        return True
    else:
        print(f"\n⚠️  Dataset may be incomplete")
        print(f"   Expected ~3,888 recordings, found {len(unique_records)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download BUT-PPG v2.0.0 dataset from PhysioNet"
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/but_ppg/raw',
        help='Output directory for raw data (default: data/but_ppg/raw)'
    )

    parser.add_argument(
        '--method',
        type=str,
        default='zip',
        choices=['zip'],
        help='Download method (default: zip)'
    )

    parser.add_argument(
        '--skip-if-exists',
        action='store_true',
        help='Skip downloading if ZIP file already exists'
    )

    parser.add_argument(
        '--keep-zip',
        action='store_true',
        help='Keep ZIP file after extraction'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("="*80)
    print("BUT-PPG DATASET DOWNLOADER")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"PhysioNet URL: https://physionet.org/content/butppg/2.0.0/")
    print(f"Dataset size: 86.7 MB")
    print(f"Download method: {args.method}")

    try:
        # Step 1: Download ZIP
        zip_path = download_butppg_zip(output_dir, skip_if_exists=args.skip_if_exists)

        # Step 2: Extract ZIP
        dataset_dir = extract_butppg_zip(zip_path, output_dir)

        # Step 3: Organize dataset
        recordings_dir, annotations_dir = organize_dataset(dataset_dir, output_dir)

        # Step 4: Verify dataset
        verify_dataset(recordings_dir, annotations_dir)

        # Clean up ZIP file
        if not args.keep_zip and zip_path.exists():
            print(f"\n🗑️  Removing ZIP file: {zip_path}")
            zip_path.unlink()

        print("\n" + "="*80)
        print("✅ DOWNLOAD COMPLETE!")
        print("="*80)
        print(f"📁 Dataset location: {output_dir}")
        print(f"   - Recordings: {recordings_dir}")
        print(f"   - Annotations: {annotations_dir}")
        print("\n📖 Next steps:")
        print("   1. Process clinical data:")
        print("      python scripts/process_butppg_clinical.py \\")
        print(f"        --raw-dir {recordings_dir} \\")
        print(f"        --annotations-dir {annotations_dir} \\")
        print("        --output-dir data/processed/butppg")
        print()

    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Download failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
