#!/usr/bin/env python3
"""
BUT-PPG Window Processor with Embedded Labels

Creates one NPZ file per window with:
- Synchronized PPG + ECG + ACC signals (5 channels)
- Recording-level clinical labels (quality, HR, BP, SpO2, etc.)
- Quality metrics
- Normalization statistics

Output: One window = One NPZ file with all metadata

Usage:
    python scripts/create_butppg_windows_with_labels.py \
        --data-dir data/but_ppg/dataset/but-ppg-an-annotated-photoplethysmography-dataset-2.0.0 \
        --output-dir data/processed/butppg/windows_with_labels \
        --splits-file configs/splits/butppg_splits.json \
        --window-sec 8.192 \
        --fs 125 \
        --max-recordings 100
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Optional, Tuple
from scipy import signal as scipy_signal
import wfdb


def load_and_sync_signals(
    record_path: Path,
    fs_target: int = 125
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Load PPG, ECG, and ACC with proper temporal synchronization.

    CRITICAL: Ensures all modalities come from EXACT same time points.

    Args:
        record_path: Path to WFDB record (without extension)
        fs_target: Target sampling rate (Hz)

    Returns:
        ppg_sync: Synchronized PPG signal [N]
        ecg_sync: Synchronized ECG signal [N]
        acc_sync: Synchronized ACC signal [N, 3]
    """
    try:
        # Load WFDB record
        record = wfdb.rdrecord(str(record_path))

        # Extract signals
        # BUT-PPG channels: [ACC_X, ACC_Y, ACC_Z, PPG, ECG]
        if record.n_sig != 5:
            print(f"  Warning: Expected 5 channels, got {record.n_sig}")
            return None, None, None

        acc_x = record.p_signal[:, 0]
        acc_y = record.p_signal[:, 1]
        acc_z = record.p_signal[:, 2]
        ppg = record.p_signal[:, 3]
        ecg = record.p_signal[:, 4]

        # Original sampling rates
        # ACC: 64 Hz, PPG: 64 Hz, ECG: 250 Hz
        acc_fs = 64
        ppg_fs = 64
        ecg_fs = 250

        # Resample to target frequency
        ppg_resampled = scipy_signal.resample_poly(ppg, fs_target, ppg_fs)
        ecg_resampled = scipy_signal.resample_poly(ecg, fs_target, ecg_fs)
        acc_x_resampled = scipy_signal.resample_poly(acc_x, fs_target, acc_fs)
        acc_y_resampled = scipy_signal.resample_poly(acc_y, fs_target, acc_fs)
        acc_z_resampled = scipy_signal.resample_poly(acc_z, fs_target, acc_fs)

        # CRITICAL: Synchronize to same length
        min_len = min(
            len(ppg_resampled),
            len(ecg_resampled),
            len(acc_x_resampled),
            len(acc_y_resampled),
            len(acc_z_resampled)
        )

        ppg_sync = ppg_resampled[:min_len]
        ecg_sync = ecg_resampled[:min_len]
        acc_x_sync = acc_x_resampled[:min_len]
        acc_y_sync = acc_y_resampled[:min_len]
        acc_z_sync = acc_z_resampled[:min_len]

        # Stack ACC channels
        acc_sync = np.stack([acc_x_sync, acc_y_sync, acc_z_sync], axis=1)

        # Remove NaN values
        ppg_valid = ~np.isnan(ppg_sync)
        ecg_valid = ~np.isnan(ecg_sync)
        acc_valid = ~np.any(np.isnan(acc_sync), axis=1)
        valid_mask = ppg_valid & ecg_valid & acc_valid

        if not np.any(valid_mask):
            return None, None, None

        ppg_sync = ppg_sync[valid_mask]
        ecg_sync = ecg_sync[valid_mask]
        acc_sync = acc_sync[valid_mask]

        return ppg_sync, ecg_sync, acc_sync

    except Exception as e:
        print(f"  Error loading record {record_path}: {e}")
        return None, None, None


def compute_sqi_ppg(signal: np.ndarray, fs: int) -> float:
    """Compute Signal Quality Index for PPG.

    Args:
        signal: PPG signal [N]
        fs: Sampling rate

    Returns:
        SQI score (0-1, higher = better quality)
    """
    # Simple quality metrics
    signal_range = signal.max() - signal.min()
    if signal_range < 0.1:
        return 0.0  # Flat signal

    noise_ratio = signal.std() / signal_range
    if noise_ratio > 2.0:
        return 0.0  # Too noisy

    # Check for reasonable frequency content (0.5-3 Hz for PPG)
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
        return 0.5  # Default moderate quality


def compute_sqi_ecg(signal: np.ndarray, fs: int) -> float:
    """Compute Signal Quality Index for ECG.

    Args:
        signal: ECG signal [N]
        fs: Sampling rate

    Returns:
        SQI score (0-1, higher = better quality)
    """
    signal_range = signal.max() - signal.min()
    if signal_range < 0.1:
        return 0.0

    noise_ratio = signal.std() / signal_range
    if noise_ratio > 3.0:
        return 0.0

    # Check for reasonable frequency content (0.5-40 Hz for ECG)
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


def check_window_quality(
    ppg_window: np.ndarray,
    ecg_window: np.ndarray,
    fs: int = 125
) -> Tuple[bool, float, float]:
    """Check if window passes quality criteria.

    Args:
        ppg_window: PPG window [T]
        ecg_window: ECG window [T]
        fs: Sampling rate

    Returns:
        passes: Whether window passes quality checks
        ppg_sqi: PPG quality score
        ecg_sqi: ECG quality score
    """
    ppg_sqi = compute_sqi_ppg(ppg_window, fs)
    ecg_sqi = compute_sqi_ecg(ecg_window, fs)

    # Moderate thresholds for general recordings
    passes = ppg_sqi > 0.4 and ecg_sqi > 0.4

    return passes, ppg_sqi, ecg_sqi


def load_csv_annotations(data_dir: Path) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load annotation CSV files.

    Args:
        data_dir: BUT-PPG dataset directory

    Returns:
        quality_hr_df: DataFrame with quality and HR annotations
        subject_info_df: DataFrame with subject info and clinical labels
    """
    # Load quality-hr-ann.csv
    quality_hr_path = data_dir / 'quality-hr-ann.csv'
    quality_hr_df = None
    if quality_hr_path.exists():
        quality_hr_df = pd.read_csv(quality_hr_path)
        # Set record ID as index for fast lookup
        if 'record' in quality_hr_df.columns:
            quality_hr_df.set_index('record', inplace=True)

    # Load subject-info.csv
    subject_info_path = data_dir / 'subject-info.csv'
    subject_info_df = None
    if subject_info_path.exists():
        subject_info_df = pd.read_csv(subject_info_path)
        if 'participant' in subject_info_df.columns:
            subject_info_df.set_index('participant', inplace=True)

    return quality_hr_df, subject_info_df


def get_recording_labels(
    record_id: str,
    quality_hr_df: Optional[pd.DataFrame],
    subject_info_df: Optional[pd.DataFrame]
) -> Dict:
    """Get all clinical labels for a recording.

    Args:
        record_id: Recording ID (e.g., '100001')
        quality_hr_df: Quality/HR annotations
        subject_info_df: Subject info annotations

    Returns:
        Dictionary of clinical labels
    """
    labels = {
        # Primary clinical labels
        'quality': np.nan,
        'hr': np.nan,
        'motion': np.nan,
        'bp_systolic': np.nan,
        'bp_diastolic': np.nan,
        'spo2': np.nan,
        'glycaemia': np.nan,
        # Demographics
        'age': np.nan,
        'sex': 'U',
        'bmi': np.nan,
        'height': np.nan,
        'weight': np.nan
    }

    # Load quality and HR from quality-hr-ann.csv
    if quality_hr_df is not None and record_id in quality_hr_df.index:
        row = quality_hr_df.loc[record_id]

        # Quality (0=poor, 1=good)
        if 'quality' in row:
            labels['quality'] = int(row['quality'])

        # Heart rate (BPM)
        if 'hr' in row:
            labels['hr'] = float(row['hr'])

    # Load subject-level labels from subject-info.csv
    # Extract participant ID (first 3 digits)
    participant_id = int(record_id[:3])

    if subject_info_df is not None and participant_id in subject_info_df.index:
        row = subject_info_df.loc[participant_id]

        # Motion class (0-7)
        if 'motion' in row and not pd.isna(row['motion']):
            labels['motion'] = int(row['motion'])

        # Blood pressure (parse "120/80" format)
        if 'bp' in row and pd.notna(row['bp']):
            bp_str = str(row['bp']).strip()
            if '/' in bp_str:
                try:
                    sys, dia = bp_str.split('/')
                    labels['bp_systolic'] = float(sys.strip())
                    labels['bp_diastolic'] = float(dia.strip())
                except:
                    pass

        # SpO2 percentage
        if 'spo2' in row and not pd.isna(row['spo2']):
            labels['spo2'] = float(row['spo2'])

        # Glycaemia (blood glucose)
        if 'glycaemia' in row and not pd.isna(row['glycaemia']):
            labels['glycaemia'] = float(row['glycaemia'])

        # Demographics
        if 'age' in row and not pd.isna(row['age']):
            labels['age'] = float(row['age'])

        if 'sex' in row and not pd.isna(row['sex']):
            labels['sex'] = str(row['sex'])

        if 'bmi' in row and not pd.isna(row['bmi']):
            labels['bmi'] = float(row['bmi'])

        if 'height' in row and not pd.isna(row['height']):
            labels['height'] = float(row['height'])

        if 'weight' in row and not pd.isna(row['weight']):
            labels['weight'] = float(row['weight'])

    # Encode sex as integer
    sex_code = {'M': 0, 'F': 1, 'U': -1}.get(labels['sex'], -1)
    labels['sex_code'] = sex_code

    return labels


def process_recording(
    record_id: str,
    data_dir: Path,
    output_dir: Path,
    split: str,
    window_sec: float,
    fs: int,
    quality_hr_df: Optional[pd.DataFrame],
    subject_info_df: Optional[pd.DataFrame],
    window_counter: int,
    quality_filter: bool = True,
    overlap_ratio: float = 0.25
) -> Tuple[int, int, int]:
    """Process single recording and save windows with overlap.

    Args:
        record_id: Recording ID
        data_dir: BUT-PPG data directory
        output_dir: Output directory
        split: 'train', 'val', or 'test'
        window_sec: Window duration in seconds
        fs: Sampling rate
        quality_hr_df: Quality/HR annotations
        subject_info_df: Subject info annotations
        window_counter: Starting window index
        quality_filter: Apply quality filtering
        overlap_ratio: Overlap ratio (0.25 = 25% overlap between consecutive windows)

    Returns:
        num_windows: Number of windows created
        num_rejected: Number of rejected windows
        next_counter: Next window counter value
    """
    # Find record file (navigate to subdirectory)
    record_dir = data_dir / record_id
    record_path = record_dir / f"{record_id}_PPG"

    if not (record_path.parent / f"{record_path.name}.dat").exists():
        return 0, 0, window_counter

    # Load and synchronize signals
    ppg, ecg, acc = load_and_sync_signals(record_path, fs_target=fs)

    if ppg is None or ecg is None or acc is None:
        return 0, 0, window_counter

    # Normalize per-recording (z-score)
    ppg_mean, ppg_std = ppg.mean(), ppg.std()
    ecg_mean, ecg_std = ecg.mean(), ecg.std()
    acc_mean = acc.mean(axis=0)
    acc_std = acc.std(axis=0)

    if ppg_std > 0:
        ppg = (ppg - ppg_mean) / ppg_std
    if ecg_std > 0:
        ecg = (ecg - ecg_mean) / ecg_std
    for i in range(3):
        if acc_std[i] > 0:
            acc[:, i] = (acc[:, i] - acc_mean[i]) / acc_std[i]

    # Load recording labels
    recording_labels = get_recording_labels(record_id, quality_hr_df, subject_info_df)

    # Create overlapping windows within the same recording
    window_samples = int(window_sec * fs)

    # Calculate stride (overlap_ratio=0.25 means 75% of window size is the stride)
    stride = int(window_samples * (1 - overlap_ratio))

    # Calculate number of possible windows with overlap
    if len(ppg) < window_samples:
        return 0, 0, window_counter

    num_possible_windows = (len(ppg) - window_samples) // stride + 1

    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    num_windows = 0
    num_rejected = 0

    for win_idx in range(num_possible_windows):
        start_idx = win_idx * stride  # Use stride instead of window_samples for overlap
        end_idx = start_idx + window_samples

        # Safety check
        if end_idx > len(ppg):
            break

        ppg_window = ppg[start_idx:end_idx]
        ecg_window = ecg[start_idx:end_idx]
        acc_window = acc[start_idx:end_idx]

        # Quality check
        if quality_filter:
            passes, ppg_sqi, ecg_sqi = check_window_quality(ppg_window, ecg_window, fs)

            if not passes:
                num_rejected += 1
                continue
        else:
            ppg_sqi = compute_sqi_ppg(ppg_window, fs)
            ecg_sqi = compute_sqi_ecg(ecg_window, fs)

        # Stack into [5, T] format: [ACC_X, ACC_Y, ACC_Z, PPG, ECG]
        signal = np.stack([
            acc_window[:, 0],  # ACC_X
            acc_window[:, 1],  # ACC_Y
            acc_window[:, 2],  # ACC_Z
            ppg_window,        # PPG
            ecg_window         # ECG
        ], axis=0).astype(np.float32)

        # Save as individual NPZ
        output_file = split_dir / f"window_{window_counter:06d}.npz"

        np.savez_compressed(
            output_file,
            # Signal data
            signal=signal,  # [5, 1024]

            # Metadata
            record_id=record_id,
            window_idx=win_idx,
            start_time=start_idx / fs,
            fs=fs,

            # Recording-level clinical labels
            quality=recording_labels['quality'],
            hr=recording_labels['hr'],
            motion=recording_labels['motion'],
            bp_systolic=recording_labels['bp_systolic'],
            bp_diastolic=recording_labels['bp_diastolic'],
            spo2=recording_labels['spo2'],
            glycaemia=recording_labels['glycaemia'],

            # Demographics
            age=recording_labels['age'],
            sex=recording_labels['sex_code'],
            bmi=recording_labels['bmi'],
            height=recording_labels['height'],
            weight=recording_labels['weight'],

            # Quality metrics
            ppg_quality=ppg_sqi,
            ecg_quality=ecg_sqi,

            # Normalization stats
            ppg_mean=ppg_mean,
            ppg_std=ppg_std,
            ecg_mean=ecg_mean,
            ecg_std=ecg_std,
            acc_mean=acc_mean,
            acc_std=acc_std,

            # Window overlap info
            overlap_ratio=overlap_ratio,
            stride_samples=stride
        )

        num_windows += 1
        window_counter += 1

    return num_windows, num_rejected, window_counter


def main():
    parser = argparse.ArgumentParser(description="Create BUT-PPG windows with embedded labels")
    parser.add_argument('--data-dir', type=str, required=True,
                       help='BUT-PPG dataset directory')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for windows')
    parser.add_argument('--splits-file', type=str, default='configs/splits/butppg_splits.json',
                       help='JSON file with recording splits')
    parser.add_argument('--window-sec', type=float, default=8.192,
                       help='Window duration in seconds')
    parser.add_argument('--fs', type=int, default=125,
                       help='Target sampling rate (Hz)')
    parser.add_argument('--max-recordings', type=int, default=None,
                       help='Maximum recordings to process (for testing)')
    parser.add_argument('--no-quality-filter', action='store_true',
                       help='Disable quality filtering')
    parser.add_argument('--overlap', type=float, default=0.25,
                       help='Overlap ratio between windows (0.25 = 25%% overlap)')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load splits
    with open(args.splits_file) as f:
        splits = json.load(f)

    # Load CSV annotations
    quality_hr_df, subject_info_df = load_csv_annotations(data_dir)

    # Calculate stride
    window_samples = int(args.window_sec * args.fs)
    stride_samples = int(window_samples * (1 - args.overlap))

    print("="*80)
    print("BUT-PPG WINDOW PROCESSOR WITH LABELS")
    print("="*80)
    print(f"Data dir: {data_dir}")
    print(f"Output: {output_dir}")
    print(f"Window: {args.window_sec}s @ {args.fs}Hz = {window_samples} samples")
    print(f"Overlap: {args.overlap*100:.0f}% ({int(window_samples * args.overlap)} samples)")
    print(f"Stride: {stride_samples} samples")
    print(f"Quality filter: {not args.no_quality_filter}")
    print(f"Annotations loaded:")
    print(f"  - Quality/HR: {len(quality_hr_df) if quality_hr_df is not None else 0} records")
    print(f"  - Subject info: {len(subject_info_df) if subject_info_df is not None else 0} subjects")
    print()

    # Process each split
    stats = {}

    for split_name in ['train', 'val', 'test']:
        if split_name not in splits:
            continue

        record_ids = [str(r) for r in splits[split_name]]
        if args.max_recordings:
            record_ids = record_ids[:args.max_recordings]

        print(f"\n{split_name.upper()}: Processing {len(record_ids)} recordings...")

        window_counter = 0
        total_windows = 0
        total_rejected = 0
        successful_recordings = 0

        for record_id in tqdm(record_ids, desc=f"{split_name}"):
            num_win, num_rej, window_counter = process_recording(
                record_id, data_dir, output_dir, split_name,
                args.window_sec, args.fs,
                quality_hr_df, subject_info_df,
                window_counter,
                quality_filter=not args.no_quality_filter,
                overlap_ratio=args.overlap
            )

            total_windows += num_win
            total_rejected += num_rej
            if num_win > 0:
                successful_recordings += 1

        stats[split_name] = {
            'num_recordings': len(record_ids),
            'successful_recordings': successful_recordings,
            'total_windows': total_windows,
            'rejected_windows': total_rejected,
            'accept_rate': total_windows / (total_windows + total_rejected) if total_windows + total_rejected > 0 else 0
        }

        print(f"  ✓ Created {total_windows} windows from {successful_recordings}/{len(record_ids)} recordings")
        print(f"  ✓ Rejected {total_rejected} windows (quality)")

    # Save metadata
    metadata = {
        'window_sec': args.window_sec,
        'fs': args.fs,
        'window_samples': int(args.window_sec * args.fs),
        'overlap_ratio': args.overlap,
        'stride_samples': stride_samples,
        'n_channels': 5,
        'channel_names': ['ACC_X', 'ACC_Y', 'ACC_Z', 'PPG', 'ECG'],
        'stats': stats
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    for split, st in stats.items():
        print(f"{split}: {st['total_windows']} windows, {st['accept_rate']:.1%} accept rate")


if __name__ == '__main__':
    main()
