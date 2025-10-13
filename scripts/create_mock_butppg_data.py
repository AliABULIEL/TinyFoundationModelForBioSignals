#!/usr/bin/env python3
"""Create mock BUT-PPG data for testing fine-tuning pipeline.

This generates synthetic 5-channel data (ACC_X, ACC_Y, ACC_Z, PPG, ECG)
with binary quality labels for testing the fine-tuning script.

Usage:
    python scripts/create_mock_butppg_data.py --output data/but_ppg --samples 100

This will create:
    data/but_ppg/train.npz (80 samples)
    data/but_ppg/val.npz (10 samples)
    data/but_ppg/test.npz (10 samples)
"""

import argparse
from pathlib import Path
import numpy as np


def generate_ppg_signal(length: int, quality: str = 'good') -> np.ndarray:
    """Generate synthetic PPG signal.
    
    Args:
        length: Signal length in samples
        quality: 'good' or 'poor'
    
    Returns:
        PPG signal [length]
    """
    t = np.linspace(0, 10, length)
    
    if quality == 'good':
        # Clean PPG with clear peaks
        heart_rate = 75  # BPM
        ppg = np.sin(2 * np.pi * heart_rate / 60 * t)
        ppg += 0.3 * np.sin(4 * np.pi * heart_rate / 60 * t)  # Harmonics
        ppg += 0.05 * np.random.randn(length)  # Small noise
    else:
        # Noisy PPG with artifacts
        heart_rate = 75
        ppg = np.sin(2 * np.pi * heart_rate / 60 * t)
        ppg += 0.5 * np.random.randn(length)  # Large noise
        ppg += 2.0 * np.sin(2 * np.pi * 0.5 * t)  # Motion artifact
    
    return ppg


def generate_ecg_signal(length: int, quality: str = 'good') -> np.ndarray:
    """Generate synthetic ECG signal.
    
    Args:
        length: Signal length in samples
        quality: 'good' or 'poor'
    
    Returns:
        ECG signal [length]
    """
    t = np.linspace(0, 10, length)
    
    if quality == 'good':
        # Clean ECG with QRS complexes
        heart_rate = 75  # BPM
        ecg = 0.5 * np.sin(2 * np.pi * heart_rate / 60 * t)
        # Add QRS spikes
        beat_interval = 60 / heart_rate
        for i in range(int(10 / beat_interval)):
            peak_time = i * beat_interval
            ecg += 2.0 * np.exp(-((t - peak_time) ** 2) / 0.01)
        ecg += 0.05 * np.random.randn(length)  # Small noise
    else:
        # Noisy ECG
        heart_rate = 75
        ecg = 0.5 * np.sin(2 * np.pi * heart_rate / 60 * t)
        ecg += 0.3 * np.random.randn(length)  # Large noise
        # Baseline wander
        ecg += 0.5 * np.sin(2 * np.pi * 0.2 * t)
    
    return ecg


def generate_acc_signal(length: int, quality: str = 'good') -> np.ndarray:
    """Generate synthetic accelerometer signal.
    
    Args:
        length: Signal length in samples
        quality: 'good' or 'poor'
    
    Returns:
        ACC signal [length]
    """
    t = np.linspace(0, 10, length)
    
    if quality == 'good':
        # Minimal movement
        acc = 0.1 * np.sin(2 * np.pi * 0.3 * t)  # Breathing
        acc += 0.02 * np.random.randn(length)  # Sensor noise
    else:
        # High movement (motion artifacts)
        acc = 0.5 * np.sin(2 * np.pi * 1.5 * t)  # Movement
        acc += 0.3 * np.random.randn(length)  # Large noise
        # Random spikes
        num_spikes = np.random.randint(3, 8)
        for _ in range(num_spikes):
            spike_pos = np.random.randint(0, length)
            spike_width = np.random.randint(10, 50)
            start = max(0, spike_pos - spike_width // 2)
            end = min(length, spike_pos + spike_width // 2)
            acc[start:end] += np.random.randn() * 2.0
    
    return acc


def generate_sample(sample_idx: int, quality_label: int) -> np.ndarray:
    """Generate one 5-channel sample.
    
    Args:
        sample_idx: Sample index (for random seed)
        quality_label: 0=poor, 1=good
    
    Returns:
        5-channel signal [5, 1250]
    """
    np.random.seed(sample_idx)
    length = 1250  # 10s at 125Hz
    
    quality = 'good' if quality_label == 1 else 'poor'
    
    # Generate all 5 channels
    acc_x = generate_acc_signal(length, quality)
    acc_y = generate_acc_signal(length, quality)
    acc_z = generate_acc_signal(length, quality)
    ppg = generate_ppg_signal(length, quality)
    ecg = generate_ecg_signal(length, quality)
    
    # Stack channels: [5, 1250]
    signal = np.stack([acc_x, acc_y, acc_z, ppg, ecg], axis=0)
    
    return signal.astype(np.float32)


def create_dataset(num_samples: int, start_idx: int = 0) -> tuple:
    """Create a dataset with balanced classes.
    
    Args:
        num_samples: Total number of samples
        start_idx: Starting index for random seed
    
    Returns:
        signals [N, 5, 1250], labels [N]
    """
    # Balanced classes
    num_good = num_samples // 2
    num_poor = num_samples - num_good
    
    signals = []
    labels = []
    
    # Generate good quality samples
    for i in range(num_good):
        signal = generate_sample(start_idx + i, quality_label=1)
        signals.append(signal)
        labels.append(1)
    
    # Generate poor quality samples
    for i in range(num_poor):
        signal = generate_sample(start_idx + num_good + i, quality_label=0)
        signals.append(signal)
        labels.append(0)
    
    # Shuffle
    indices = np.random.permutation(num_samples)
    signals = np.array(signals)[indices]
    labels = np.array(labels)[indices]
    
    return signals, labels


def main():
    parser = argparse.ArgumentParser(
        description="Create mock BUT-PPG data for testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/but_ppg',
        help='Output directory'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=100,
        help='Total number of samples'
    )
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.8,
        help='Training set fraction'
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.1,
        help='Validation set fraction (rest goes to test)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate splits
    n_train = int(args.samples * args.train_split)
    n_val = int(args.samples * args.val_split)
    n_test = args.samples - n_train - n_val
    
    print("=" * 70)
    print("CREATING MOCK BUT-PPG DATA")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Total samples: {args.samples}")
    print(f"  Train: {n_train} samples")
    print(f"  Val:   {n_val} samples")
    print(f"  Test:  {n_test} samples")
    print("=" * 70)
    
    # Generate datasets
    print("\nGenerating training set...")
    train_signals, train_labels = create_dataset(n_train, start_idx=0)
    np.savez(
        output_dir / 'train.npz',
        signals=train_signals,
        labels=train_labels
    )
    print(f"  ✓ Saved train.npz: {train_signals.shape}")
    print(f"    Good: {(train_labels == 1).sum()}, Poor: {(train_labels == 0).sum()}")
    
    print("\nGenerating validation set...")
    val_signals, val_labels = create_dataset(n_val, start_idx=n_train)
    np.savez(
        output_dir / 'val.npz',
        signals=val_signals,
        labels=val_labels
    )
    print(f"  ✓ Saved val.npz: {val_signals.shape}")
    print(f"    Good: {(val_labels == 1).sum()}, Poor: {(val_labels == 0).sum()}")
    
    print("\nGenerating test set...")
    test_signals, test_labels = create_dataset(n_test, start_idx=n_train + n_val)
    np.savez(
        output_dir / 'test.npz',
        signals=test_signals,
        labels=test_labels
    )
    print(f"  ✓ Saved test.npz: {test_signals.shape}")
    print(f"    Good: {(test_labels == 1).sum()}, Poor: {(test_labels == 0).sum()}")
    
    print("\n" + "=" * 70)
    print("MOCK DATA CREATED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nYou can now test the fine-tuning script with:")
    print(f"  python scripts/finetune_butppg.py \\")
    print(f"    --data-dir {output_dir} \\")
    print(f"    --epochs 1 \\")
    print(f"    --batch-size 8")
    print("=" * 70)


if __name__ == '__main__':
    main()
