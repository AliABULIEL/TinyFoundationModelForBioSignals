#!/usr/bin/env python3
"""
Patch BUT-PPG NPZ files to add 'quality' labels based on SQI metrics.

This is useful when you don't have quality-hr-ann.csv but have SQI metrics
(ppg_quality, ecg_quality) computed by the prepare script.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tqdm import tqdm
import argparse


def patch_quality_labels(
    npz_dir: Path,
    sqi_threshold: float = 0.4,
    require_both: bool = True,
    dry_run: bool = False
):
    """Add quality labels to NPZ files based on SQI.

    Args:
        npz_dir: Directory containing window_*.npz files
        sqi_threshold: SQI threshold (above = good, below = poor)
        require_both: If True, require both PPG and ECG SQI > threshold for quality=1
        dry_run: If True, don't actually modify files
    """

    if not npz_dir.exists():
        print(f"✗ Directory not found: {npz_dir}")
        return 0, 0

    window_files = sorted(npz_dir.glob('window_*.npz'))

    if not window_files:
        print(f"✗ No window files found in {npz_dir}")
        return 0, 0

    print(f"\nProcessing {len(window_files)} files in {npz_dir.name}...")

    patched = 0
    skipped = 0
    quality_dist = {0: 0, 1: 0}

    for window_file in tqdm(window_files, desc=f"{npz_dir.name}"):
        # Load existing data
        data = dict(np.load(window_file))

        # Check if SQI fields exist
        if 'ppg_quality' not in data or 'ecg_quality' not in data:
            print(f"  ⚠️  {window_file.name}: Missing SQI fields, skipping")
            skipped += 1
            continue

        # Get SQI values
        ppg_sqi = float(data['ppg_quality'])
        ecg_sqi = float(data['ecg_quality'])

        # Compute quality label based on SQI
        if require_both:
            # Both PPG and ECG must be good
            quality = 1 if (ppg_sqi > sqi_threshold and ecg_sqi > sqi_threshold) else 0
        else:
            # Either PPG or ECG being good is sufficient
            quality = 1 if (ppg_sqi > sqi_threshold or ecg_sqi > sqi_threshold) else 0

        # Add quality label
        data['quality'] = np.array(quality, dtype=np.int64)
        quality_dist[quality] += 1

        # Save back (unless dry run)
        if not dry_run:
            np.savez(window_file, **data)
            patched += 1
        else:
            patched += 1  # Count for dry run reporting

    return patched, quality_dist


def main():
    parser = argparse.ArgumentParser(
        description="Patch NPZ files with quality labels from SQI metrics"
    )
    parser.add_argument('--data-dir', type=str,
                       default='data/processed/butppg/windows_with_labels',
                       help='Directory containing train/val/test subdirectories')
    parser.add_argument('--sqi-threshold', type=float, default=0.4,
                       help='SQI threshold for good quality (default: 0.4)')
    parser.add_argument('--require-both', action='store_true', default=True,
                       help='Require both PPG and ECG SQI > threshold (default: True)')
    parser.add_argument('--require-either', dest='require_both', action='store_false',
                       help='Accept if either PPG or ECG SQI > threshold')
    parser.add_argument('--dry-run', action='store_true',
                       help='Simulate without actually modifying files')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print(f"\n{'='*80}")
    print(f"PATCHING NPZ FILES WITH QUALITY LABELS FROM SQI")
    print(f"{'='*80}")
    print(f"Data directory: {data_dir}")
    print(f"SQI threshold: {args.sqi_threshold}")
    print(f"Logic: {'Both PPG AND ECG > threshold' if args.require_both else 'Either PPG OR ECG > threshold'}")
    print(f"Dry run: {args.dry_run}")

    if args.dry_run:
        print(f"\n⚠️  DRY RUN MODE - No files will be modified")

    # Process each split
    total_patched = 0
    total_quality = {0: 0, 1: 0}

    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"\n⚠️  {split} directory not found, skipping")
            continue

        patched, quality_dist = patch_quality_labels(
            split_dir,
            sqi_threshold=args.sqi_threshold,
            require_both=args.require_both,
            dry_run=args.dry_run
        )

        print(f"\n{split.upper()} Results:")
        print(f"  Patched: {patched} files")
        print(f"  Quality 0 (poor): {quality_dist[0]} ({quality_dist[0]/patched*100:.1f}%)")
        print(f"  Quality 1 (good): {quality_dist[1]} ({quality_dist[1]/patched*100:.1f}%)")

        total_patched += patched
        total_quality[0] += quality_dist[0]
        total_quality[1] += quality_dist[1]

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total files patched: {total_patched}")
    print(f"Overall quality distribution:")
    print(f"  Quality 0 (poor): {total_quality[0]} ({total_quality[0]/total_patched*100:.1f}%)")
    print(f"  Quality 1 (good): {total_quality[1]} ({total_quality[1]/total_patched*100:.1f}%)")

    if args.dry_run:
        print(f"\n⚠️  This was a DRY RUN. Run without --dry-run to actually patch files.")
    else:
        print(f"\n✓ Files successfully patched!")
        print(f"\nNext step: Re-run fine-tuning:")
        print(f"  python scripts/finetune_butppg.py \\")
        print(f"      --pretrained artifacts/foundation_model/best_model.pt \\")
        print(f"      --data-dir {data_dir} \\")
        print(f"      --epochs 30 \\")
        print(f"      --batch-size 64")


if __name__ == '__main__':
    main()
