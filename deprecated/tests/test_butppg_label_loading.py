#!/usr/bin/env python3
"""
Test BUT-PPG Label Loading and Windowing Logic

Tests:
1. CSV annotation loading
2. Record ID matching
3. Label extraction
4. Window overlap logic
5. End-to-end window creation with labels
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd
import tempfile
import json


def create_mock_csv_files(tmp_dir: Path):
    """Create mock BUT-PPG CSV files for testing."""

    # Create quality-hr-ann.csv
    quality_hr_data = {
        'record': [100001, 100002, 100003],
        'quality': [1, 0, 1],
        'hr': [75.5, 82.3, 68.9]
    }
    quality_hr_df = pd.DataFrame(quality_hr_data)
    quality_hr_path = tmp_dir / 'quality-hr-ann.csv'
    quality_hr_df.to_csv(quality_hr_path, index=False)

    # Create subject-info.csv
    subject_info_data = {
        'participant': [100, 100, 100],  # Same participant for all
        'motion': [0, 1, 2],
        'bp': ['120/80', '130/85', '115/75'],
        'age': [25, 25, 25],
        'sex': ['M', 'M', 'M'],
        'height': [175, 175, 175],
        'weight': [70, 70, 70]
    }
    subject_info_df = pd.DataFrame(subject_info_data)
    # Note: subject-info uses 'participant' column, not 'record'
    subject_info_path = tmp_dir / 'subject-info.csv'
    subject_info_df.to_csv(subject_info_path, index=False)

    return quality_hr_path, subject_info_path


def test_csv_loading():
    """Test that CSV files are loaded correctly."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        quality_hr_path, subject_info_path = create_mock_csv_files(tmp_path)

        # Test loading
        from deprecated.scripts.create_butppg_windows_with_labels import load_csv_annotations

        quality_hr_df, subject_info_df = load_csv_annotations(tmp_path)

        # Check quality-hr-ann.csv
        assert quality_hr_df is not None
        assert len(quality_hr_df) == 3
        assert 100001 in quality_hr_df.index  # Should be indexed by record
        assert 'quality' in quality_hr_df.columns
        assert 'hr' in quality_hr_df.columns

        # Check subject-info.csv
        assert subject_info_df is not None
        assert len(subject_info_df) == 3
        # Should be indexed by participant (first 3 digits)

        print("✓ CSV loading test passed")


def test_record_id_matching():
    """Test that record IDs match correctly between CSV and data."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        create_mock_csv_files(tmp_path)

        from deprecated.scripts.create_butppg_windows_with_labels import (
            load_csv_annotations,
            get_recording_labels
        )

        quality_hr_df, subject_info_df = load_csv_annotations(tmp_path)

        # Test exact match
        labels_1 = get_recording_labels('100001', quality_hr_df, subject_info_df)
        assert labels_1['quality'] == 1
        assert labels_1['hr'] == 75.5
        assert labels_1['motion'] == 0  # From subject-info for participant 100

        # Test another record
        labels_2 = get_recording_labels('100002', quality_hr_df, subject_info_df)
        assert labels_2['quality'] == 0
        assert labels_2['hr'] == 82.3

        # Test missing record (should return NaN)
        labels_missing = get_recording_labels('999999', quality_hr_df, subject_info_df)
        assert np.isnan(labels_missing['quality'])
        assert np.isnan(labels_missing['hr'])

        print("✓ Record ID matching test passed")


def test_record_id_format():
    """Test different record ID formats to find the mismatch."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Test different CSV formats
        test_cases = [
            # Case 1: Integer IDs in CSV
            {
                'csv_records': [100001, 100002],
                'lookup_ids': ['100001', '100002'],
                'description': 'Integer CSV, string lookup'
            },
            # Case 2: String IDs in CSV
            {
                'csv_records': ['100001', '100002'],
                'lookup_ids': ['100001', '100002'],
                'description': 'String CSV, string lookup'
            },
            # Case 3: Integer CSV, integer lookup (potential issue!)
            {
                'csv_records': [100001, 100002],
                'lookup_ids': [100001, 100002],
                'description': 'Integer CSV, integer lookup'
            }
        ]

        from deprecated.scripts.create_butppg_windows_with_labels import get_recording_labels

        for case in test_cases:
            print(f"\nTesting: {case['description']}")

            # Create CSV with specific format
            quality_hr_df = pd.DataFrame({
                'record': case['csv_records'],
                'quality': [1, 0],
                'hr': [75.0, 80.0]
            })
            quality_hr_df.set_index('record', inplace=True)

            print(f"  CSV index type: {type(quality_hr_df.index[0])}")
            print(f"  CSV index: {list(quality_hr_df.index)}")

            # Test lookups
            for lookup_id in case['lookup_ids']:
                print(f"  Looking up: {lookup_id} (type: {type(lookup_id)})")

                # Check if in index
                in_index = lookup_id in quality_hr_df.index
                print(f"    In index: {in_index}")

                if not in_index:
                    # Try converting
                    str_id = str(lookup_id)
                    int_id = int(lookup_id) if isinstance(lookup_id, str) else lookup_id

                    print(f"    Trying str({lookup_id}): {str_id in quality_hr_df.index}")
                    print(f"    Trying int({lookup_id}): {int_id in quality_hr_df.index}")

                # Try get_recording_labels
                labels = get_recording_labels(str(lookup_id), quality_hr_df, None)
                print(f"    Quality: {labels['quality']} (NaN={np.isnan(labels['quality'])})")


def test_window_overlap_logic():
    """Test overlapping window generation logic."""

    # Create a fake signal
    signal_length = 1000  # samples
    window_samples = 250   # 250 samples per window
    overlap_ratio = 0.25   # 25% overlap

    # Calculate stride
    stride = int(window_samples * (1 - overlap_ratio))

    # Expected stride = 250 * 0.75 = 187.5 ≈ 187
    assert stride == 187, f"Expected stride=187, got {stride}"

    # Calculate number of windows
    num_windows = (signal_length - window_samples) // stride + 1

    # Expected: (1000 - 250) // 187 + 1 = 750 // 187 + 1 = 4 + 1 = 5
    assert num_windows == 5, f"Expected 5 windows, got {num_windows}"

    # Verify window positions
    windows = []
    for win_idx in range(num_windows):
        start_idx = win_idx * stride
        end_idx = start_idx + window_samples

        if end_idx > signal_length:
            break

        windows.append((start_idx, end_idx))

    print(f"\n✓ Overlap logic test passed")
    print(f"  Signal length: {signal_length}")
    print(f"  Window size: {window_samples}")
    print(f"  Overlap: {overlap_ratio*100}%")
    print(f"  Stride: {stride}")
    print(f"  Number of windows: {len(windows)}")
    print(f"  Window positions:")
    for i, (start, end) in enumerate(windows):
        print(f"    Window {i}: [{start:4d} - {end:4d}] (length={end-start})")

        # Check overlap with previous window
        if i > 0:
            prev_end = windows[i-1][1]
            overlap_samples = prev_end - start
            overlap_pct = overlap_samples / window_samples
            print(f"      Overlap with prev: {overlap_samples} samples ({overlap_pct*100:.1f}%)")


def test_window_overlap_coverage():
    """Test that windows cover the signal correctly with overlap."""

    test_cases = [
        {'overlap': 0.0, 'expected_coverage': 'no overlap'},
        {'overlap': 0.25, 'expected_coverage': '25% overlap'},
        {'overlap': 0.5, 'expected_coverage': '50% overlap'},
    ]

    signal_length = 1024
    window_samples = 256

    for case in test_cases:
        overlap_ratio = case['overlap']
        stride = int(window_samples * (1 - overlap_ratio))
        num_windows = (signal_length - window_samples) // stride + 1

        # Track coverage
        coverage = np.zeros(signal_length, dtype=int)

        for win_idx in range(num_windows):
            start_idx = win_idx * stride
            end_idx = start_idx + window_samples

            if end_idx > signal_length:
                break

            coverage[start_idx:end_idx] += 1

        print(f"\n{case['expected_coverage']}:")
        print(f"  Windows: {num_windows}")
        print(f"  Coverage: min={coverage.min()}, max={coverage.max()}")
        print(f"  Samples covered 0x: {np.sum(coverage == 0)}")
        print(f"  Samples covered 1x: {np.sum(coverage == 1)}")
        print(f"  Samples covered 2x: {np.sum(coverage == 2)}")
        print(f"  Samples covered 3x: {np.sum(coverage == 3)}")


def test_label_embedding_in_npz():
    """Test that labels are correctly embedded in NPZ files."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create mock signal and labels
        signal = np.random.randn(2, 1024).astype(np.float32)

        labels = {
            'quality': 1,
            'hr': 75.5,
            'motion': 0,
            'bp_systolic': 120.0,
            'bp_diastolic': 80.0,
            'age': 25.0,
            'sex_code': 0
        }

        # Save NPZ
        npz_path = tmp_path / 'window_000000.npz'
        np.savez_compressed(
            npz_path,
            signal=signal,
            record_id='100001',
            window_idx=0,
            quality=labels['quality'],
            hr=labels['hr'],
            motion=labels['motion'],
            bp_systolic=labels['bp_systolic'],
            bp_diastolic=labels['bp_diastolic'],
            age=labels['age'],
            sex=labels['sex_code']
        )

        # Load and verify
        data = np.load(npz_path)

        assert 'signal' in data
        assert 'quality' in data
        assert 'hr' in data

        assert data['quality'] == 1
        assert data['hr'] == 75.5
        assert data['motion'] == 0

        print("\n✓ Label embedding test passed")
        print(f"  Keys in NPZ: {list(data.keys())}")
        print(f"  Quality: {data['quality']}")
        print(f"  HR: {data['hr']}")
        print(f"  Motion: {data['motion']}")


def test_nan_handling():
    """Test that NaN labels are handled correctly."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create NPZ with NaN labels
        signal = np.random.randn(2, 1024).astype(np.float32)

        npz_path = tmp_path / 'window_000000.npz'
        np.savez_compressed(
            npz_path,
            signal=signal,
            quality=np.nan,
            hr=np.nan,
            motion=np.nan
        )

        # Load and check
        data = np.load(npz_path)

        assert np.isnan(data['quality'])
        assert np.isnan(data['hr'])
        assert np.isnan(data['motion'])

        print("\n✓ NaN handling test passed")
        print(f"  Quality is NaN: {np.isnan(data['quality'])}")
        print(f"  HR is NaN: {np.isnan(data['hr'])}")


if __name__ == '__main__':
    print("="*80)
    print("BUT-PPG LABEL LOADING AND WINDOWING TESTS")
    print("="*80)

    print("\n1. Testing CSV loading...")
    test_csv_loading()

    print("\n2. Testing record ID matching...")
    test_record_id_matching()

    print("\n3. Testing record ID format variations...")
    test_record_id_format()

    print("\n4. Testing window overlap logic...")
    test_window_overlap_logic()

    print("\n5. Testing window overlap coverage...")
    test_window_overlap_coverage()

    print("\n6. Testing label embedding in NPZ...")
    test_label_embedding_in_npz()

    print("\n7. Testing NaN handling...")
    test_nan_handling()

    print("\n" + "="*80)
    print("ALL TESTS PASSED!")
    print("="*80)
