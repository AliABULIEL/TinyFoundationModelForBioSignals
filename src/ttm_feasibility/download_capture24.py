"""
Script to download and prepare Capture-24 dataset for TTM feasibility study.

Dataset: CAPTURE-24 - Activity tracker dataset from Oxford University
URL: https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001
License: CC BY 4.0
"""

import os
import sys
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm


def download_file(url, destination):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(destination, 'wb') as file, tqdm(
        desc=destination.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)


def main():
    # Define paths
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data" / "capture24"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download URL (direct link to capture24.zip)
    download_url = "https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001/files/rpr76f381b"
    zip_path = data_dir / "capture24.zip"

    print("="*70)
    print("CAPTURE-24 Dataset Download")
    print("="*70)
    print(f"Source: Oxford University Research Archive")
    print(f"License: CC BY 4.0")
    print(f"Size: ~6.5 GB")
    print(f"Destination: {data_dir}")
    print("="*70)

    # Check if already downloaded
    if zip_path.exists():
        print(f"\n✓ File already downloaded: {zip_path}")
        user_input = input("Re-download? (y/n): ")
        if user_input.lower() != 'y':
            print("Skipping download.")
        else:
            print("\nDownloading...")
            download_file(download_url, zip_path)
    else:
        print("\nDownloading...")
        download_file(download_url, zip_path)

    # Extract
    print(f"\nExtracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    print(f"\n✓ Dataset extracted to: {data_dir}")

    # List contents
    print("\nDataset contents:")
    for item in sorted(data_dir.iterdir()):
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"  {item.name} ({size_mb:.2f} MB)")
        elif item.is_dir() and item.name != "__pycache__":
            num_files = len(list(item.glob("*")))
            print(f"  {item.name}/ ({num_files} files)")

    print("\n✓ Download complete!")
    print("\nNext steps:")
    print("1. Explore the data structure")
    print("2. Run data analysis script")
    print("3. Continue with TTM feasibility study")


if __name__ == "__main__":
    main()
