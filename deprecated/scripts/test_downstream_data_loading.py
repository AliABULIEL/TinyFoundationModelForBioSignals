#!/usr/bin/env python3
"""
Test script to verify downstream evaluation data loading
with the new windowed format (labels embedded in NPZ files).

Usage:
    python scripts/test_downstream_data_loading.py \
        --data-dir data/processed/butppg/windows_with_labels
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def test_windowed_format(data_dir: Path):
    """Test loading BUT-PPG data from windowed format."""

    print("="*80)
    print("TESTING WINDOWED FORMAT DATA LOADING")
    print("="*80)
    print(f"Data directory: {data_dir}\n")

    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split

        if not split_dir.exists():
            print(f"❌ {split.upper()}: Directory not found")
            continue

        window_files = sorted(split_dir.glob('window_*.npz'))

        if len(window_files) == 0:
            print(f"❌ {split.upper()}: No window files found")
            continue

        print(f"{split.upper()} Split:")
        print(f"  Total windows: {len(window_files)}")

        # Load first file to check structure
        first_data = np.load(window_files[0])
        print(f"\n  First file structure:")
        print(f"    File: {window_files[0].name}")
        print(f"    Keys: {list(first_data.keys())}")

        for key in ['signal', 'quality', 'hr', 'motion']:
            if key in first_data:
                val = first_data[key]
                if hasattr(val, 'shape'):
                    print(f"    {key}: shape={val.shape}, dtype={val.dtype}, value={val.item() if val.size == 1 else val}")
                else:
                    print(f"    {key}: {val}")
            else:
                print(f"    {key}: MISSING")

        # Load all windows and check label distribution
        signals_list = []
        quality_labels = []
        hr_labels = []
        motion_labels = []

        for window_file in window_files:
            data = np.load(window_file)

            # Load signal
            signal = data['signal']
            signals_list.append(signal)

            # Load labels (handle NaN)
            if 'quality' in data:
                q = data['quality']
                quality_labels.append(q if not np.isnan(q) else -1)
            else:
                quality_labels.append(-1)

            if 'hr' in data:
                h = data['hr']
                hr_labels.append(h if not np.isnan(h) else -1)
            else:
                hr_labels.append(-1)

            if 'motion' in data:
                m = data['motion']
                motion_labels.append(m if not np.isnan(m) else -1)
            else:
                motion_labels.append(-1)

        # Stack and analyze
        signals = np.stack(signals_list, axis=0)
        quality_labels = np.array(quality_labels)
        hr_labels = np.array(hr_labels)
        motion_labels = np.array(motion_labels)

        print(f"\n  Data shape:")
        print(f"    Signals: {signals.shape}")
        print(f"    Quality labels: {quality_labels.shape}")
        print(f"    HR labels: {hr_labels.shape}")
        print(f"    Motion labels: {motion_labels.shape}")

        # Quality labels
        print(f"\n  Quality labels:")
        valid_quality = quality_labels != -1
        print(f"    Valid: {valid_quality.sum()}/{len(quality_labels)} ({valid_quality.sum()/len(quality_labels)*100:.1f}%)")
        if valid_quality.sum() > 0:
            unique, counts = np.unique(quality_labels[valid_quality], return_counts=True)
            for val, count in zip(unique, counts):
                print(f"      Label {int(val)}: {count} ({count/valid_quality.sum()*100:.1f}%)")
        else:
            print(f"    ❌ ALL LABELS ARE NaN or -1!")

        # HR labels
        print(f"\n  Heart Rate labels:")
        valid_hr = hr_labels != -1
        print(f"    Valid: {valid_hr.sum()}/{len(hr_labels)} ({valid_hr.sum()/len(hr_labels)*100:.1f}%)")
        if valid_hr.sum() > 0:
            print(f"      Min: {hr_labels[valid_hr].min():.1f}")
            print(f"      Max: {hr_labels[valid_hr].max():.1f}")
            print(f"      Mean: {hr_labels[valid_hr].mean():.1f}")
            print(f"      Median: {np.median(hr_labels[valid_hr]):.1f}")

        # Motion labels
        print(f"\n  Motion labels:")
        valid_motion = motion_labels != -1
        print(f"    Valid: {valid_motion.sum()}/{len(motion_labels)} ({valid_motion.sum()/len(motion_labels)*100:.1f}%)")
        if valid_motion.sum() > 0:
            unique, counts = np.unique(motion_labels[valid_motion], return_counts=True)
            for val, count in zip(unique, counts):
                print(f"      Label {int(val)}: {count} ({count/valid_motion.sum()*100:.1f}%)")

        # Test DataLoader creation
        print(f"\n  Testing DataLoader creation:")

        if valid_quality.sum() > 0:
            signals_tensor = torch.from_numpy(signals[valid_quality]).float()
            labels_tensor = torch.tensor(quality_labels[valid_quality], dtype=torch.long)
            dataset = TensorDataset(signals_tensor, labels_tensor)
            loader = DataLoader(dataset, batch_size=32, shuffle=False)

            batch = next(iter(loader))
            print(f"    ✓ Quality DataLoader: {len(dataset)} samples")
            print(f"      Batch shape: {batch[0].shape}")
            print(f"      Labels shape: {batch[1].shape}")
            print(f"      Sample labels: {batch[1][:5].tolist()}")
        else:
            print(f"    ❌ Cannot create Quality DataLoader (no valid labels)")

        print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Test downstream evaluation data loading")
    parser.add_argument('--data-dir', type=str,
                       default='data/processed/butppg/windows_with_labels',
                       help='Path to BUT-PPG windowed data directory')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print(f"\nExpected structure:")
        print(f"  {data_dir}/")
        print(f"    train/window_*.npz")
        print(f"    val/window_*.npz")
        print(f"    test/window_*.npz")
        return 1

    test_windowed_format(data_dir)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\n✓ Data loading test complete!")
    print("\nNext step: Run downstream evaluation with:")
    print(f"\n  python scripts/run_downstream_evaluation.py \\")
    print(f"      --butppg-checkpoint artifacts/butppg_ssl/best_model_butppg.pt \\")
    print(f"      --butppg-data {data_dir.parent} \\")
    print(f"      --output-dir artifacts/downstream_evaluation")

    return 0


if __name__ == '__main__':
    sys.exit(main())
