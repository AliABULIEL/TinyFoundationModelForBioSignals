#!/usr/bin/env python3
"""
Migration Script: Convert Multi-Window NPZ to Individual Window NPZ

Converts existing data formats to new one-window-per-NPZ format:
- Old: train_windows.npz with data=[N, T, C], labels=[N]
- New: window_000001.npz, window_000002.npz, ... each with signal=[C, T]

Supports both VitalDB and BUT-PPG datasets.

Usage:
    # Migrate BUT-PPG data
    python scripts/migrate_to_windowed_npz.py \
        --input data/processed/butppg/windows/train_windows.npz \
        --output data/processed/butppg/windows_with_labels/train \
        --dataset butppg \
        --fs 125

    # Migrate VitalDB data
    python scripts/migrate_to_windowed_npz.py \
        --input data/processed/vitaldb/windows/train_windows.npz \
        --output data/processed/vitaldb/windows_with_labels/train \
        --dataset vitaldb \
        --fs 125
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional
from scipy import signal as scipy_signal


def compute_sqi_ppg(signal: np.ndarray, fs: int) -> float:
    """Compute Signal Quality Index for PPG."""
    signal_range = signal.max() - signal.min()
    if signal_range < 0.1:
        return 0.0

    noise_ratio = signal.std() / signal_range
    if noise_ratio > 2.0:
        return 0.0

    try:
        freqs, psd = scipy_signal.welch(signal, fs=fs, nperseg=min(256, len(signal)))
        ppg_band = (freqs >= 0.5) & (freqs <= 3.0)
        signal_power = psd[ppg_band].sum()
        total_power = psd.sum()

        if total_power == 0:
            return 0.0

        sqi = signal_power / total_power
        return float(np.clip(sqi, 0, 1))
    except:
        return 0.5


def compute_sqi_ecg(signal: np.ndarray, fs: int) -> float:
    """Compute Signal Quality Index for ECG."""
    signal_range = signal.max() - signal.min()
    if signal_range < 0.1:
        return 0.0

    noise_ratio = signal.std() / signal_range
    if noise_ratio > 3.0:
        return 0.0

    try:
        freqs, psd = scipy_signal.welch(signal, fs=fs, nperseg=min(256, len(signal)))
        ecg_band = (freqs >= 0.5) & (freqs <= 40.0)
        signal_power = psd[ecg_band].sum()
        total_power = psd.sum()

        if total_power == 0:
            return 0.0

        sqi = signal_power / total_power
        return float(np.clip(sqi, 0, 1))
    except:
        return 0.5


def migrate_butppg_data(
    input_file: Path,
    output_dir: Path,
    fs: int
) -> int:
    """Migrate BUT-PPG multi-window NPZ to individual window NPZs.

    Args:
        input_file: Input NPZ file (train_windows.npz)
        output_dir: Output directory for individual windows
        fs: Sampling rate

    Returns:
        Number of windows migrated
    """
    print(f"\nMigrating BUT-PPG data from: {input_file}")

    # Load input file
    data = np.load(input_file)

    # Check format
    if 'data' in data:
        signals = data['data']  # [N, T, C] or [N, C, T]
    elif 'signals' in data:
        signals = data['signals']
    else:
        raise ValueError("No 'data' or 'signals' array found in input file")

    print(f"  Signal shape: {signals.shape}")

    # Detect format and transpose if needed
    if len(signals.shape) == 3:
        if signals.shape[1] == 1024:  # [N, T, C]
            print("  Detected [N, T, C] format, transposing to [N, C, T]")
            signals = np.transpose(signals, (0, 2, 1))
        elif signals.shape[2] == 1024:  # [N, C, T]
            print("  Detected [N, C, T] format")
        else:
            raise ValueError(f"Unexpected signal shape: {signals.shape}")

    n_samples, n_channels, n_timesteps = signals.shape

    # Load all available labels
    label_keys = ['quality', 'hr', 'motion', 'bp_systolic', 'bp_diastolic', 'spo2', 'glycaemia']
    demographic_keys = ['age', 'sex', 'bmi', 'height', 'weight']

    labels_dict = {}
    for key in label_keys + demographic_keys:
        if key in data:
            labels_dict[key] = data[key]
        else:
            labels_dict[key] = np.full(n_samples, np.nan)

    # Handle backwards compatibility: 'labels' -> 'quality'
    if 'labels' in data and 'quality' not in labels_dict:
        labels_dict['quality'] = data['labels']

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCreating individual window files...")
    window_counter = 0

    for idx in tqdm(range(n_samples)):
        signal = signals[idx]  # [C, T]

        # Compute quality metrics
        # For BUT-PPG: channels are [ACC_X, ACC_Y, ACC_Z, PPG, ECG]
        if n_channels == 5:
            ppg_idx = 3
            ecg_idx = 4
        elif n_channels == 2:
            ppg_idx = 0
            ecg_idx = 1
        else:
            raise ValueError(f"Unexpected number of channels: {n_channels}")

        ppg_sqi = compute_sqi_ppg(signal[ppg_idx], fs)
        ecg_sqi = compute_sqi_ecg(signal[ecg_idx], fs)

        # Compute normalization stats
        signal_mean = signal.mean(axis=1)
        signal_std = signal.std(axis=1)

        # Create window file
        output_file = output_dir / f"window_{window_counter:06d}.npz"

        # Build save dict
        save_dict = {
            'signal': signal.astype(np.float32),
            'record_id': f'unknown_{idx}',  # Unknown record ID
            'window_idx': idx,
            'start_time': 0.0,  # Unknown start time
            'fs': fs,
            'ppg_quality': ppg_sqi,
            'ecg_quality': ecg_sqi,
        }

        # Add labels
        for key, values in labels_dict.items():
            if idx < len(values):
                save_dict[key] = values[idx]
            else:
                save_dict[key] = np.nan

        # Add normalization stats
        if n_channels == 5:
            save_dict['ppg_mean'] = signal_mean[ppg_idx]
            save_dict['ppg_std'] = signal_std[ppg_idx]
            save_dict['ecg_mean'] = signal_mean[ecg_idx]
            save_dict['ecg_std'] = signal_std[ecg_idx]
            save_dict['acc_mean'] = signal_mean[:3]
            save_dict['acc_std'] = signal_std[:3]
        else:
            save_dict['ppg_mean'] = signal_mean[0]
            save_dict['ppg_std'] = signal_std[0]
            save_dict['ecg_mean'] = signal_mean[1]
            save_dict['ecg_std'] = signal_std[1]

        np.savez_compressed(output_file, **save_dict)
        window_counter += 1

    print(f"\n✓ Migrated {window_counter} windows")
    return window_counter


def migrate_vitaldb_data(
    input_file: Path,
    output_dir: Path,
    fs: int
) -> int:
    """Migrate VitalDB multi-window NPZ to individual window NPZs.

    Args:
        input_file: Input NPZ file (train_windows.npz)
        output_dir: Output directory for individual windows
        fs: Sampling rate

    Returns:
        Number of windows migrated
    """
    print(f"\nMigrating VitalDB data from: {input_file}")

    # Load input file
    data = np.load(input_file)

    # Check format
    if 'data' in data:
        signals = data['data']  # [N, T, C] or [N, C, T]
    elif 'signals' in data:
        signals = data['signals']
    else:
        raise ValueError("No 'data' or 'signals' array found in input file")

    print(f"  Signal shape: {signals.shape}")

    # Detect format and transpose if needed
    if len(signals.shape) == 3:
        if signals.shape[1] == 1024:  # [N, T, C]
            print("  Detected [N, T, C] format, transposing to [N, C, T]")
            signals = np.transpose(signals, (0, 2, 1))
        elif signals.shape[2] == 1024:  # [N, C, T]
            print("  Detected [N, C, T] format")
        else:
            raise ValueError(f"Unexpected signal shape: {signals.shape}")

    n_samples, n_channels, n_timesteps = signals.shape

    if n_channels != 2:
        raise ValueError(f"Expected 2 channels for VitalDB, got {n_channels}")

    # Load case-level labels if available
    label_keys = ['death_inhosp', 'icu_days', 'emergency', 'asa', 'age', 'sex', 'bmi']
    labels_dict = {}
    for key in label_keys:
        if key in data:
            labels_dict[key] = data[key]
        else:
            labels_dict[key] = np.full(n_samples, np.nan)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCreating individual window files...")
    window_counter = 0

    for idx in tqdm(range(n_samples)):
        signal = signals[idx]  # [2, T] - [PPG, ECG]

        # Compute quality metrics
        ppg_sqi = compute_sqi_ppg(signal[0], fs)
        ecg_sqi = compute_sqi_ecg(signal[1], fs)

        # Compute normalization stats
        ppg_mean, ppg_std = signal[0].mean(), signal[0].std()
        ecg_mean, ecg_std = signal[1].mean(), signal[1].std()

        # Create window file
        output_file = output_dir / f"window_{window_counter:06d}.npz"

        # Build save dict
        save_dict = {
            'signal': signal.astype(np.float32),
            'case_id': 0,  # Unknown case ID
            'window_idx': idx,
            'start_time': 0.0,  # Unknown start time
            'fs': fs,
            'ppg_quality': ppg_sqi,
            'ecg_quality': ecg_sqi,
            'ppg_mean': ppg_mean,
            'ppg_std': ppg_std,
            'ecg_mean': ecg_mean,
            'ecg_std': ecg_std,
        }

        # Add case-level labels
        for key, values in labels_dict.items():
            if idx < len(values):
                save_dict[key] = values[idx]
            else:
                save_dict[key] = np.nan

        np.savez_compressed(output_file, **save_dict)
        window_counter += 1

    print(f"\n✓ Migrated {window_counter} windows")
    return window_counter


def main():
    parser = argparse.ArgumentParser(description="Migrate to windowed NPZ format")
    parser.add_argument('--input', type=str, required=True,
                       help='Input NPZ file (e.g., train_windows.npz)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for individual windows')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['vitaldb', 'butppg'],
                       help='Dataset type')
    parser.add_argument('--fs', type=int, default=125,
                       help='Sampling rate (Hz)')

    args = parser.parse_args()

    input_file = Path(args.input)
    output_dir = Path(args.output)

    # Check input exists
    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        sys.exit(1)

    print("="*80)
    print("MIGRATE TO WINDOWED NPZ FORMAT")
    print("="*80)
    print(f"Input: {input_file}")
    print(f"Output: {output_dir}")
    print(f"Dataset: {args.dataset}")
    print(f"Sampling rate: {args.fs} Hz")

    # Migrate data
    if args.dataset == 'butppg':
        num_windows = migrate_butppg_data(input_file, output_dir, args.fs)
    else:
        num_windows = migrate_vitaldb_data(input_file, output_dir, args.fs)

    # Create metadata.json
    metadata = {
        'window_sec': 1024 / args.fs,
        'fs': args.fs,
        'window_samples': 1024,
        'n_channels': 5 if args.dataset == 'butppg' else 2,
        'channel_names': ['ACC_X', 'ACC_Y', 'ACC_Z', 'PPG', 'ECG'] if args.dataset == 'butppg' else ['PPG', 'ECG'],
        'total_windows': num_windows,
        'source_file': str(input_file.name)
    }

    metadata_file = output_dir.parent / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*80)
    print("MIGRATION COMPLETE!")
    print("="*80)
    print(f"✓ Created {num_windows} individual window files")
    print(f"✓ Saved metadata to {metadata_file}")
    print("\nNext steps:")
    print("  1. Verify migration with: python scripts/test_window_npz_storage.py")
    print(f"  2. Test unified loader: python src/data/unified_window_loader.py {output_dir}")


if __name__ == '__main__':
    main()
