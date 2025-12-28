#!/usr/bin/env python3
"""
VitalDB Window Processor with Embedded Labels

Creates one NPZ file per window with:
- Synchronized PPG + ECG signals (2 channels)
- Case-level clinical labels
- Quality metrics
- Normalization statistics

Output: One window = One NPZ file with all metadata

Usage:
    python scripts/create_vitaldb_windows_with_labels.py \
        --output-dir data/processed/vitaldb/windows_with_labels \
        --splits-file configs/splits/splits_full.json \
        --window-sec 8.192 \
        --fs 125 \
        --max-cases 100
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
from multiprocessing import Pool, cpu_count

# Import VitalDB loader
try:
    import vitaldb
    VITALDB_AVAILABLE = True
except ImportError:
    VITALDB_AVAILABLE = False
    print("⚠️  VitalDB not available, will use cached data")


def load_and_sync_signals(
    case_id: int,
    fs_target: int = 125,
    cache_dir: Optional[Path] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load PPG and ECG with proper temporal synchronization.

    CRITICAL: Ensures PPG and ECG come from EXACT same time points.

    Args:
        case_id: VitalDB case ID
        fs_target: Target sampling rate (Hz)
        cache_dir: Directory for caching downloaded signals

    Returns:
        ppg_sync: Synchronized PPG signal [N]
        ecg_sync: Synchronized ECG signal [N]
    """
    if not VITALDB_AVAILABLE:
        return None, None

    try:
        # Load PPG (PLETH track)
        ppg_signal = vitaldb.load_case(case_id, ['Solar/PLETH'])
        if ppg_signal is None or len(ppg_signal) == 0:
            return None, None

        ppg_fs = 100  # VitalDB PLETH is typically 100 Hz

        # Load ECG (ECG_II track)
        ecg_signal = vitaldb.load_case(case_id, ['Solar/ECG_II'])
        if ecg_signal is None or len(ecg_signal) == 0:
            return None, None

        ecg_fs = 500  # VitalDB ECG is typically 500 Hz

        # Resample to target frequency
        ppg_resampled = scipy_signal.resample_poly(
            ppg_signal, fs_target, ppg_fs
        )
        ecg_resampled = scipy_signal.resample_poly(
            ecg_signal, fs_target, ecg_fs
        )

        # CRITICAL: Synchronize to same length
        min_len = min(len(ppg_resampled), len(ecg_resampled))
        ppg_sync = ppg_resampled[:min_len]
        ecg_sync = ecg_resampled[:min_len]

        # Remove NaN values
        valid_mask = ~(np.isnan(ppg_sync) | np.isnan(ecg_sync))
        if not np.any(valid_mask):
            return None, None

        ppg_sync = ppg_sync[valid_mask]
        ecg_sync = ecg_sync[valid_mask]

        return ppg_sync, ecg_sync

    except Exception as e:
        print(f"  Error loading case {case_id}: {e}")
        return None, None


def compute_sqi_ppg(signal: np.ndarray, fs: int) -> float:
    """Compute Signal Quality Index for PPG.

    Args:
        signal: PPG signal [N]
        fs: Sampling rate

    Returns:
        SQI score (0-1, higher = better quality)
    """
    # Simple quality metrics
    # 1. Check for saturation
    signal_range = signal.max() - signal.min()
    if signal_range < 0.1:
        return 0.0  # Flat signal

    # 2. Check for excessive noise (std/range ratio)
    noise_ratio = signal.std() / signal_range
    if noise_ratio > 2.0:
        return 0.0  # Too noisy

    # 3. Check for reasonable frequency content (0.5-3 Hz for PPG)
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
    # Simple quality metrics
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

    # Relaxed thresholds for surgical data
    passes = ppg_sqi > 0.3 and ecg_sqi > 0.3

    return passes, ppg_sqi, ecg_sqi


def load_case_labels(case_id: int, metadata_df: Optional[pd.DataFrame] = None) -> Dict:
    """Load case-level labels from VitalDB metadata.

    Args:
        case_id: VitalDB case ID
        metadata_df: Preloaded metadata DataFrame

    Returns:
        Dictionary of case-level labels
    """
    labels = {
        'age': np.nan,
        'sex': 'U',  # Unknown
        'height': np.nan,
        'weight': np.nan,
        'bmi': np.nan,
        'asa': np.nan,
        'emergency': False,
        'death_inhosp': False,
        'icu_days': np.nan
    }

    if metadata_df is not None and case_id in metadata_df.index:
        row = metadata_df.loc[case_id]
        labels.update({
            'age': float(row.get('age', np.nan)),
            'sex': str(row.get('sex', 'U')),
            'height': float(row.get('height', np.nan)),
            'weight': float(row.get('weight', np.nan)),
            'bmi': float(row.get('bmi', np.nan)),
            'asa': int(row.get('asa', np.nan)) if not pd.isna(row.get('asa')) else np.nan,
            'emergency': bool(row.get('emop', False)),
            'death_inhosp': bool(row.get('death_inhosp', False)),
            'icu_days': float(row.get('icu_days', np.nan))
        })

    # Encode sex as integer
    sex_code = {'M': 0, 'F': 1, 'U': -1}.get(labels['sex'], -1)
    labels['sex_code'] = sex_code

    return labels


def process_case(
    case_id: int,
    output_dir: Path,
    split: str,
    window_sec: float,
    fs: int,
    metadata_df: Optional[pd.DataFrame],
    window_counter: int,
    overlap_ratio: float = 0.25
) -> Tuple[int, int, int]:
    """Process single case and save windows with overlap.

    Args:
        case_id: VitalDB case ID
        output_dir: Output directory
        split: 'train', 'val', or 'test'
        window_sec: Window duration in seconds
        fs: Sampling rate
        metadata_df: Metadata DataFrame
        window_counter: Starting window index
        overlap_ratio: Overlap ratio (0.25 = 25% overlap between consecutive windows)

    Returns:
        num_windows: Number of windows created
        num_rejected: Number of rejected windows
        next_counter: Next window counter value
    """
    # Load and synchronize signals
    ppg, ecg = load_and_sync_signals(case_id, fs_target=fs)

    if ppg is None or ecg is None:
        return 0, 0, window_counter

    # Normalize per-case (z-score)
    ppg_mean, ppg_std = ppg.mean(), ppg.std()
    ecg_mean, ecg_std = ecg.mean(), ecg.std()

    if ppg_std > 0:
        ppg = (ppg - ppg_mean) / ppg_std
    if ecg_std > 0:
        ecg = (ecg - ecg_mean) / ecg_std

    # Load case labels
    case_labels = load_case_labels(case_id, metadata_df)

    # Create overlapping windows within the same case
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

        # Quality check
        passes, ppg_sqi, ecg_sqi = check_window_quality(ppg_window, ecg_window, fs)

        if not passes:
            num_rejected += 1
            continue

        # Stack into [2, T] format
        signal = np.stack([ppg_window, ecg_window], axis=0).astype(np.float32)

        # Save as individual NPZ
        output_file = split_dir / f"window_{window_counter:06d}.npz"

        np.savez_compressed(
            output_file,
            # Signal data
            signal=signal,  # [2, 1024]

            # Metadata
            case_id=case_id,
            window_idx=win_idx,
            start_time=start_idx / fs,
            fs=fs,

            # Case-level labels
            age=case_labels['age'],
            sex=case_labels['sex_code'],
            height=case_labels['height'],
            weight=case_labels['weight'],
            bmi=case_labels['bmi'],
            asa=case_labels['asa'],
            emergency=case_labels['emergency'],
            death_inhosp=case_labels['death_inhosp'],
            icu_days=case_labels['icu_days'],

            # Quality metrics
            ppg_quality=ppg_sqi,
            ecg_quality=ecg_sqi,

            # Normalization stats
            ppg_mean=ppg_mean,
            ppg_std=ppg_std,
            ecg_mean=ecg_mean,
            ecg_std=ecg_std,

            # Window overlap info
            overlap_ratio=overlap_ratio,
            stride_samples=stride
        )

        num_windows += 1
        window_counter += 1

    return num_windows, num_rejected, window_counter


def main():
    parser = argparse.ArgumentParser(description="Create VitalDB windows with embedded labels")
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for windows')
    parser.add_argument('--splits-file', type=str, default='configs/splits/splits_full.json',
                       help='JSON file with case splits')
    parser.add_argument('--metadata-csv', type=str, default=None,
                       help='CSV file with case metadata')
    parser.add_argument('--window-sec', type=float, default=8.192,
                       help='Window duration in seconds')
    parser.add_argument('--fs', type=int, default=125,
                       help='Target sampling rate (Hz)')
    parser.add_argument('--max-cases', type=int, default=None,
                       help='Maximum cases to process (for testing)')
    parser.add_argument('--overlap', type=float, default=0.25,
                       help='Overlap ratio between windows (0.25 = 25%% overlap)')

    args = parser.parse_args()

    if not VITALDB_AVAILABLE:
        print("ERROR: VitalDB package not installed!")
        print("Install with: pip install vitaldb")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load splits
    with open(args.splits_file) as f:
        splits = json.load(f)

    # Load metadata if provided
    metadata_df = None
    if args.metadata_csv and Path(args.metadata_csv).exists():
        metadata_df = pd.read_csv(args.metadata_csv, index_col='caseid')

    # Calculate stride
    window_samples = int(args.window_sec * args.fs)
    stride_samples = int(window_samples * (1 - args.overlap))

    print("="*80)
    print("VITALDB WINDOW PROCESSOR WITH LABELS")
    print("="*80)
    print(f"Output: {output_dir}")
    print(f"Window: {args.window_sec}s @ {args.fs}Hz = {window_samples} samples")
    print(f"Overlap: {args.overlap*100:.0f}% ({int(window_samples * args.overlap)} samples)")
    print(f"Stride: {stride_samples} samples")
    print()

    # Process each split
    stats = {}

    for split_name in ['train', 'val', 'test']:
        if split_name not in splits:
            continue

        case_ids = [int(c) for c in splits[split_name]]
        if args.max_cases:
            case_ids = case_ids[:args.max_cases]

        print(f"\n{split_name.upper()}: Processing {len(case_ids)} cases...")

        window_counter = 0
        total_windows = 0
        total_rejected = 0
        successful_cases = 0

        for case_id in tqdm(case_ids, desc=f"{split_name}"):
            num_win, num_rej, window_counter = process_case(
                case_id, output_dir, split_name,
                args.window_sec, args.fs, metadata_df, window_counter,
                overlap_ratio=args.overlap
            )

            total_windows += num_win
            total_rejected += num_rej
            if num_win > 0:
                successful_cases += 1

        stats[split_name] = {
            'num_cases': len(case_ids),
            'successful_cases': successful_cases,
            'total_windows': total_windows,
            'rejected_windows': total_rejected,
            'accept_rate': total_windows / (total_windows + total_rejected) if total_windows + total_rejected > 0 else 0
        }

        print(f"  ✓ Created {total_windows} windows from {successful_cases}/{len(case_ids)} cases")
        print(f"  ✓ Rejected {total_rejected} windows (quality)")

    # Save metadata
    metadata = {
        'window_sec': args.window_sec,
        'fs': args.fs,
        'window_samples': int(args.window_sec * args.fs),
        'overlap_ratio': args.overlap,
        'stride_samples': stride_samples,
        'n_channels': 2,
        'channel_names': ['PPG', 'ECG'],
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
