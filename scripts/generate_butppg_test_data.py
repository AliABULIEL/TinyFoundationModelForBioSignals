#!/usr/bin/env python3
"""Generate synthetic BUT-PPG data for testing fine-tuning pipeline.

This script creates synthetic BUT-PPG data files that can be used to test
the fine-tuning pipeline without requiring real BUT-PPG data.

Usage:
    python scripts/generate_butppg_test_data.py --output-dir data/but_ppg --samples 100
"""

import argparse
from pathlib import Path
import numpy as np


def generate_synthetic_ppg_signal(
    length: int = 1250,
    fs: int = 125,
    hr: float = 75,
    noise_level: float = 0.1
) -> np.ndarray:
    """Generate synthetic PPG signal.
    
    Args:
        length: Signal length in samples
        fs: Sampling frequency in Hz
        hr: Heart rate in BPM
        noise_level: Noise level (0-1)
    
    Returns:
        Synthetic PPG signal
    """
    t = np.arange(length) / fs
    
    # Generate cardiac pulses
    pulse_freq = hr / 60  # Hz
    signal = np.zeros(length)
    
    # Add multiple harmonics for realistic PPG shape
    for harmonic in range(1, 4):
        amplitude = 1.0 / harmonic
        signal += amplitude * np.sin(2 * np.pi * harmonic * pulse_freq * t)
    
    # Add respiratory modulation
    resp_freq = 0.25  # 15 breaths per minute
    signal *= (1 + 0.1 * np.sin(2 * np.pi * resp_freq * t))
    
    # Add noise
    signal += np.random.randn(length) * noise_level
    
    # Normalize
    signal = (signal - signal.mean()) / signal.std()
    
    return signal


def generate_synthetic_ecg_signal(
    length: int = 1250,
    fs: int = 125,
    hr: float = 75,
    noise_level: float = 0.05
) -> np.ndarray:
    """Generate synthetic ECG signal.
    
    Args:
        length: Signal length in samples
        fs: Sampling frequency in Hz
        hr: Heart rate in BPM
        noise_level: Noise level (0-1)
    
    Returns:
        Synthetic ECG signal
    """
    t = np.arange(length) / fs
    
    # Generate QRS complexes
    pulse_freq = hr / 60  # Hz
    signal = np.zeros(length)
    
    # Simplified ECG: dominant R-peak + P and T waves
    signal += 1.0 * np.sin(2 * np.pi * pulse_freq * t)  # R-peak
    signal += 0.2 * np.sin(2 * np.pi * pulse_freq * t - 0.3)  # P-wave
    signal += 0.3 * np.sin(2 * np.pi * pulse_freq * t + 0.5)  # T-wave
    
    # Add noise
    signal += np.random.randn(length) * noise_level
    
    # Normalize
    signal = (signal - signal.mean()) / signal.std()
    
    return signal


def generate_synthetic_acc_signal(
    length: int = 1250,
    fs: int = 125,
    activity_level: float = 0.5,
    noise_level: float = 0.2
) -> np.ndarray:
    """Generate synthetic accelerometer signal.
    
    Args:
        length: Signal length in samples
        fs: Sampling frequency in Hz
        activity_level: Activity level (0=stationary, 1=active)
        noise_level: Noise level (0-1)
    
    Returns:
        Synthetic accelerometer signal
    """
    t = np.arange(length) / fs
    
    # Base gravity component
    signal = np.ones(length)
    
    # Add movement artifacts (low frequency)
    if activity_level > 0:
        movement_freq = 2.0  # ~2 Hz movement
        signal += activity_level * 0.5 * np.sin(2 * np.pi * movement_freq * t)
        signal += activity_level * 0.3 * np.sin(2 * np.pi * movement_freq * 1.5 * t)
    
    # Add noise
    signal += np.random.randn(length) * noise_level
    
    # Normalize
    signal = (signal - signal.mean()) / signal.std()
    
    return signal


def generate_butppg_sample(
    quality: str = 'good',
    length: int = 1250,
    fs: int = 125
) -> np.ndarray:
    """Generate a single BUT-PPG sample.
    
    Args:
        quality: Signal quality ('good' or 'poor')
        length: Signal length in samples
        fs: Sampling frequency in Hz
    
    Returns:
        Signal array of shape [5, length]
        Channels: ACC_X, ACC_Y, ACC_Z, PPG, ECG
    """
    if quality == 'good':
        # Good quality: low noise, low motion
        hr = np.random.uniform(60, 90)
        ppg_noise = 0.05
        ecg_noise = 0.03
        activity = 0.1
    else:
        # Poor quality: high noise, high motion
        hr = np.random.uniform(50, 110)
        ppg_noise = 0.3
        ecg_noise = 0.2
        activity = 0.8
    
    # Generate signals
    signal = np.zeros((5, length), dtype=np.float32)
    
    # ACC_X, ACC_Y, ACC_Z
    for i in range(3):
        signal[i] = generate_synthetic_acc_signal(
            length, fs, activity, noise_level=0.2
        )
    
    # PPG
    signal[3] = generate_synthetic_ppg_signal(
        length, fs, hr, noise_level=ppg_noise
    )
    
    # ECG
    signal[4] = generate_synthetic_ecg_signal(
        length, fs, hr, noise_level=ecg_noise
    )
    
    return signal


def generate_dataset(
    n_samples: int,
    quality_ratio: float = 0.5,
    length: int = 1250,
    fs: int = 125
) -> tuple:
    """Generate a dataset of BUT-PPG samples.
    
    Args:
        n_samples: Number of samples to generate
        quality_ratio: Ratio of good quality samples (0-1)
        length: Signal length in samples
        fs: Sampling frequency in Hz
    
    Returns:
        signals: Array of shape [n_samples, 5, length]
        labels: Array of shape [n_samples] (0=poor, 1=good)
    """
    print(f"Generating {n_samples} samples...")
    
    signals = np.zeros((n_samples, 5, length), dtype=np.float32)
    labels = np.zeros(n_samples, dtype=np.int64)
    
    n_good = int(n_samples * quality_ratio)
    n_poor = n_samples - n_good
    
    # Generate good quality samples
    for i in range(n_good):
        signals[i] = generate_butppg_sample('good', length, fs)
        labels[i] = 1
    
    # Generate poor quality samples
    for i in range(n_good, n_samples):
        signals[i] = generate_butppg_sample('poor', length, fs)
        labels[i] = 0
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    signals = signals[indices]
    labels = labels[indices]
    
    print(f"  Generated {n_good} good samples and {n_poor} poor samples")
    
    return signals, labels


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic BUT-PPG test data")
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/but_ppg',
        help='Output directory for data files'
    )
    parser.add_argument(
        '--train-samples',
        type=int,
        default=200,
        help='Number of training samples'
    )
    parser.add_argument(
        '--val-samples',
        type=int,
        default=50,
        help='Number of validation samples'
    )
    parser.add_argument(
        '--test-samples',
        type=int,
        default=100,
        help='Number of test samples'
    )
    parser.add_argument(
        '--quality-ratio',
        type=float,
        default=0.6,
        help='Ratio of good quality samples (0-1)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("GENERATING SYNTHETIC BUT-PPG DATA")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Quality ratio: {args.quality_ratio:.1%}")
    print()
    
    # Generate train set
    print("Train set:")
    train_signals, train_labels = generate_dataset(
        args.train_samples, args.quality_ratio
    )
    np.savez(
        output_dir / 'train.npz',
        signals=train_signals,
        labels=train_labels
    )
    print(f"  ✓ Saved to {output_dir / 'train.npz'}")
    
    # Generate validation set
    print("\nValidation set:")
    val_signals, val_labels = generate_dataset(
        args.val_samples, args.quality_ratio
    )
    np.savez(
        output_dir / 'val.npz',
        signals=val_signals,
        labels=val_labels
    )
    print(f"  ✓ Saved to {output_dir / 'val.npz'}")
    
    # Generate test set
    print("\nTest set:")
    test_signals, test_labels = generate_dataset(
        args.test_samples, args.quality_ratio
    )
    np.savez(
        output_dir / 'test.npz',
        signals=test_signals,
        labels=test_labels
    )
    print(f"  ✓ Saved to {output_dir / 'test.npz'}")
    
    print("\n" + "=" * 70)
    print("DATA GENERATION COMPLETE")
    print("=" * 70)
    print(f"Total samples: {args.train_samples + args.val_samples + args.test_samples}")
    print(f"Files saved to: {output_dir}")


if __name__ == "__main__":
    main()
