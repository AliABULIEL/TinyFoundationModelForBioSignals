#!/usr/bin/env python3
"""
Check BUT-PPG recording lengths to understand windowing behavior.

This script examines a sample of BUT-PPG recordings to determine their durations
and explain why overlapping windows may not be created.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import wfdb
from collections import Counter

def check_recording_length(record_path: Path):
    """Check length of a BUT-PPG recording."""
    try:
        # Load PPG file
        ppg_record = wfdb.rdrecord(str(record_path))

        # Get duration
        fs = ppg_record.fs
        n_samples = len(ppg_record.p_signal)
        duration_sec = n_samples / fs

        return {
            'record_id': record_path.parent.name,
            'fs': fs,
            'n_samples': n_samples,
            'duration_sec': duration_sec
        }
    except Exception as e:
        return None


def main():
    # Find BUT-PPG data
    data_dir = Path('data/but_ppg/raw')

    if not data_dir.exists():
        # Try alternative path
        data_dir = Path('data/but_ppg/dataset')
        if not data_dir.exists():
            print(f"❌ BUT-PPG data not found at: data/but_ppg/raw or data/but_ppg/dataset")
            return

    print("="*80)
    print("BUT-PPG Recording Length Analysis")
    print("="*80)
    print()

    # Find all PPG records
    ppg_files = list(data_dir.rglob('*_PPG.dat'))
    print(f"Found {len(ppg_files)} PPG recordings")

    # Sample first 100 recordings
    sample_files = ppg_files[:100]
    print(f"Analyzing {len(sample_files)} recordings...")
    print()

    durations = []
    for ppg_file in sample_files:
        record_path = ppg_file.parent / ppg_file.stem
        info = check_recording_length(record_path)
        if info:
            durations.append(info['duration_sec'])

    if not durations:
        print("❌ No valid recordings found")
        return

    durations = np.array(durations)

    # Calculate statistics
    print("Duration Statistics:")
    print(f"  Mean: {durations.mean():.2f} seconds")
    print(f"  Median: {np.median(durations):.2f} seconds")
    print(f"  Min: {durations.min():.2f} seconds")
    print(f"  Max: {durations.max():.2f} seconds")
    print(f"  Std: {durations.std():.2f} seconds")
    print()

    # Calculate expected windows per recording
    window_sec = 8.192
    overlap_ratio = 0.25
    stride_sec = window_sec * (1 - overlap_ratio)  # 6.144 seconds

    print(f"Window Configuration:")
    print(f"  Window size: {window_sec} seconds")
    print(f"  Overlap: {overlap_ratio*100:.0f}%")
    print(f"  Stride: {stride_sec:.3f} seconds")
    print()

    # Calculate expected windows
    expected_windows = []
    for dur in durations:
        if dur < window_sec:
            n_windows = 0
        else:
            n_windows = int((dur - window_sec) / stride_sec) + 1
        expected_windows.append(n_windows)

    expected_windows = np.array(expected_windows)

    print("Expected Windows per Recording:")
    print(f"  Mean: {expected_windows.mean():.2f}")
    print(f"  Median: {int(np.median(expected_windows))}")
    print(f"  0 windows: {(expected_windows == 0).sum()}/{len(expected_windows)} recordings")
    print(f"  1 window: {(expected_windows == 1).sum()}/{len(expected_windows)} recordings")
    print(f"  2+ windows: {(expected_windows >= 2).sum()}/{len(expected_windows)} recordings")
    print()

    # Distribution
    dist = Counter(expected_windows)
    print("Distribution of expected windows:")
    for n_win in sorted(dist.keys())[:10]:
        count = dist[n_win]
        pct = count / len(expected_windows) * 100
        print(f"  {n_win} windows: {count} recordings ({pct:.1f}%)")

    print()
    print("="*80)

    # Diagnosis
    if np.median(expected_windows) == 1:
        print("⚠️  FINDING: Most recordings produce only 1 window!")
        print()
        print("This is expected if:")
        print("  1. BUT-PPG recordings are short (10-15 seconds)")
        print("  2. Window size (8.192s) is large relative to recording length")
        print()
        print("Solutions:")
        print("  - Accept 1 window per recording (no overlap testing needed)")
        print("  - Use smaller window size (e.g., 5 seconds)")
        print("  - Use larger stride (e.g., 50% overlap instead of 25%)")
    elif np.median(expected_windows) >= 2:
        print("✅ FINDING: Most recordings should produce 2+ windows")
        print()
        print("If actual pipeline only creates 1 window per recording,")
        print("there may be a bug in the windowing code.")

    print("="*80)


if __name__ == '__main__':
    main()
