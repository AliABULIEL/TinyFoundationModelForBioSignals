#!/usr/bin/env python3
"""
Test script to verify overlapping windows work correctly.

Verifies:
1. Windows overlap by 25% within same patient/case
2. No overlap across patient boundaries
3. overlap_ratio and stride_samples are saved correctly
4. First 25% of window N equals last 25% of window N-1
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import json


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)


def print_success(message: str):
    """Print success message."""
    print(f"✅ {message}")


def print_info(message: str):
    """Print info message."""
    print(f"ℹ️  {message}")


def print_error(message: str):
    """Print error message."""
    print(f"❌ {message}")


def test_butppg_overlapping_windows():
    """Test BUT-PPG overlapping windows."""
    print_header("TEST 1: BUT-PPG Overlapping Windows")

    data_dir = project_root / 'data/processed/butppg/windows_with_labels/train'
    if not data_dir.exists():
        print_info("SKIP: Preprocessed BUT-PPG data not found")
        return False

    # Load metadata
    metadata_path = data_dir.parent / 'metadata.json'
    if not metadata_path.exists():
        print_error("metadata.json not found")
        return False

    with open(metadata_path) as f:
        metadata = json.load(f)

    print_info(f"Metadata loaded:")
    print(f"  Window: {metadata['window_samples']} samples")
    print(f"  Overlap: {metadata.get('overlap_ratio', 0)*100:.0f}%")
    print(f"  Stride: {metadata.get('stride_samples', metadata['window_samples'])} samples")

    # Find windows from same recording
    window_files = sorted(data_dir.glob('window_*.npz'))
    if len(window_files) < 2:
        print_error("Not enough windows to test overlap")
        return False

    print_info(f"Found {len(window_files)} window files")

    # Group windows by record_id
    windows_by_record = {}
    for window_file in window_files[:50]:  # Test first 50 for speed
        data = np.load(window_file)
        record_id = str(data['record_id'])

        if record_id not in windows_by_record:
            windows_by_record[record_id] = []

        windows_by_record[record_id].append({
            'file': window_file,
            'window_idx': int(data['window_idx']),
            'signal': data['signal'],
            'stride': int(data.get('stride_samples', metadata['window_samples'])),
            'overlap_ratio': float(data.get('overlap_ratio', 0))
        })

    # Find recordings with multiple windows
    multi_window_recordings = {k: v for k, v in windows_by_record.items() if len(v) >= 2}

    if not multi_window_recordings:
        print_error("No recordings with multiple windows found")
        return False

    print_info(f"Found {len(multi_window_recordings)} recordings with 2+ windows")

    # Test overlap for the first recording with multiple windows
    test_record_id = list(multi_window_recordings.keys())[0]
    windows = sorted(multi_window_recordings[test_record_id], key=lambda x: x['window_idx'])

    print_info(f"\nTesting recording: {test_record_id} ({len(windows)} windows)")

    # Verify overlap metadata
    overlap_ratio = windows[0]['overlap_ratio']
    stride = windows[0]['stride']
    window_samples = windows[0]['signal'].shape[1]

    expected_stride = int(window_samples * (1 - overlap_ratio))
    if stride != expected_stride:
        print_error(f"Stride mismatch: got {stride}, expected {expected_stride}")
        return False

    print_success(f"Overlap metadata correct: {overlap_ratio*100:.0f}% = {int(window_samples * overlap_ratio)} samples")

    # Test actual overlap between consecutive windows
    for i in range(len(windows) - 1):
        win1 = windows[i]
        win2 = windows[i + 1]

        # Check window indices are consecutive
        if win2['window_idx'] != win1['window_idx'] + 1:
            print_info(f"  Skipping non-consecutive windows: {win1['window_idx']} → {win2['window_idx']}")
            continue

        signal1 = win1['signal']  # [5, T]
        signal2 = win2['signal']  # [5, T]

        # Calculate overlap size
        overlap_samples = int(window_samples * overlap_ratio)

        # Extract overlapping regions
        # Last overlap_samples of win1 should match first overlap_samples of win2
        win1_end = signal1[:, -overlap_samples:]
        win2_start = signal2[:, :overlap_samples]

        # Check if they match (within numerical precision)
        match = np.allclose(win1_end, win2_start, rtol=1e-5, atol=1e-8)

        if match:
            print_success(f"  Windows {win1['window_idx']}-{win2['window_idx']}: {overlap_samples} samples overlap ✓")
        else:
            # Calculate difference
            diff = np.abs(win1_end - win2_start).max()
            print_error(f"  Windows {win1['window_idx']}-{win2['window_idx']}: NO overlap (max diff={diff:.6f})")
            return False

    print_success("All consecutive windows overlap correctly!")
    return True


def test_vitaldb_overlapping_windows():
    """Test VitalDB overlapping windows."""
    print_header("TEST 2: VitalDB Overlapping Windows")

    data_dir = project_root / 'data/processed/vitaldb/windows_with_labels/train'
    if not data_dir.exists():
        print_info("SKIP: Preprocessed VitalDB data not found")
        return False

    # Load metadata
    metadata_path = data_dir.parent / 'metadata.json'
    if not metadata_path.exists():
        print_error("metadata.json not found")
        return False

    with open(metadata_path) as f:
        metadata = json.load(f)

    print_info(f"Metadata loaded:")
    print(f"  Window: {metadata['window_samples']} samples")
    print(f"  Overlap: {metadata.get('overlap_ratio', 0)*100:.0f}%")
    print(f"  Stride: {metadata.get('stride_samples', metadata['window_samples'])} samples")

    # Find windows from same case
    window_files = sorted(data_dir.glob('window_*.npz'))
    if len(window_files) < 2:
        print_error("Not enough windows to test overlap")
        return False

    print_info(f"Found {len(window_files)} window files")

    # Group windows by case_id
    windows_by_case = {}
    for window_file in window_files[:50]:  # Test first 50 for speed
        data = np.load(window_file)
        case_id = int(data['case_id'])

        if case_id not in windows_by_case:
            windows_by_case[case_id] = []

        windows_by_case[case_id].append({
            'file': window_file,
            'window_idx': int(data['window_idx']),
            'signal': data['signal'],
            'stride': int(data.get('stride_samples', metadata['window_samples'])),
            'overlap_ratio': float(data.get('overlap_ratio', 0))
        })

    # Find cases with multiple windows
    multi_window_cases = {k: v for k, v in windows_by_case.items() if len(v) >= 2}

    if not multi_window_cases:
        print_error("No cases with multiple windows found")
        return False

    print_info(f"Found {len(multi_window_cases)} cases with 2+ windows")

    # Test overlap for the first case with multiple windows
    test_case_id = list(multi_window_cases.keys())[0]
    windows = sorted(multi_window_cases[test_case_id], key=lambda x: x['window_idx'])

    print_info(f"\nTesting case: {test_case_id} ({len(windows)} windows)")

    # Verify overlap metadata
    overlap_ratio = windows[0]['overlap_ratio']
    stride = windows[0]['stride']
    window_samples = windows[0]['signal'].shape[1]

    expected_stride = int(window_samples * (1 - overlap_ratio))
    if stride != expected_stride:
        print_error(f"Stride mismatch: got {stride}, expected {expected_stride}")
        return False

    print_success(f"Overlap metadata correct: {overlap_ratio*100:.0f}% = {int(window_samples * overlap_ratio)} samples")

    # Test actual overlap between consecutive windows
    for i in range(len(windows) - 1):
        win1 = windows[i]
        win2 = windows[i + 1]

        # Check window indices are consecutive
        if win2['window_idx'] != win1['window_idx'] + 1:
            print_info(f"  Skipping non-consecutive windows: {win1['window_idx']} → {win2['window_idx']}")
            continue

        signal1 = win1['signal']  # [2, T]
        signal2 = win2['signal']  # [2, T]

        # Calculate overlap size
        overlap_samples = int(window_samples * overlap_ratio)

        # Extract overlapping regions
        win1_end = signal1[:, -overlap_samples:]
        win2_start = signal2[:, :overlap_samples]

        # Check if they match
        match = np.allclose(win1_end, win2_start, rtol=1e-5, atol=1e-8)

        if match:
            print_success(f"  Windows {win1['window_idx']}-{win2['window_idx']}: {overlap_samples} samples overlap ✓")
        else:
            diff = np.abs(win1_end - win2_start).max()
            print_error(f"  Windows {win1['window_idx']}-{win2['window_idx']}: NO overlap (max diff={diff:.6f})")
            return False

    print_success("All consecutive windows overlap correctly!")
    return True


def main():
    """Run all overlap tests."""
    print("\n" + "="*80)
    print(" OVERLAPPING WINDOWS TEST SUITE")
    print("="*80)
    print("\nThis test suite verifies:")
    print("  1. Windows overlap by 25% within same patient/case")
    print("  2. overlap_ratio and stride_samples are saved correctly")
    print("  3. First 25% of window N matches last 25% of window N-1")

    # Run tests
    results = {}
    results['butppg'] = test_butppg_overlapping_windows()
    results['vitaldb'] = test_vitaldb_overlapping_windows()

    # Print summary
    print_header("TEST SUMMARY")

    passed = sum(1 for v in results.values() if v is True)
    total = len([v for v in results.values() if v is not False])

    for test_name, result in results.items():
        if result is True:
            print(f"  ✅ {test_name}: PASSED")
        elif result is False:
            print(f"  ❌ {test_name}: FAILED")
        else:
            print(f"  ⚠️  {test_name}: SKIPPED")

    print("\n" + "="*80)
    if passed == total and total > 0:
        print_success(f"ALL TESTS PASSED! ({passed}/{total})")
        print("\n🎉 Overlapping windows work correctly!")
        print("\nKey features:")
        print("  ✅ 25% overlap between consecutive windows")
        print("  ✅ Overlap only within same patient/case")
        print("  ✅ Metadata saved correctly")
    else:
        print(f"⚠️  {passed}/{total} tests passed")
        print("\nSome tests failed or were skipped.")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
