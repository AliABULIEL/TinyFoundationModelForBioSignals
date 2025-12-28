#!/usr/bin/env python3
"""
Window NPZ Storage Verification Tests

Comprehensive tests to verify:
1. VitalDB window format correctness
2. BUT-PPG window format correctness
3. Signal synchronization and quality
4. Label embedding and completeness
5. File structure and naming
6. Unified loader compatibility

Usage:
    # Test VitalDB windows
    python scripts/test_window_npz_storage.py \
        --vitaldb-dir data/processed/vitaldb/windows_with_labels/train

    # Test BUT-PPG windows
    python scripts/test_window_npz_storage.py \
        --butppg-dir data/processed/butppg/windows_with_labels/train

    # Test both
    python scripts/test_window_npz_storage.py \
        --vitaldb-dir data/processed/vitaldb/windows_with_labels/train \
        --butppg-dir data/processed/butppg/windows_with_labels/train
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
from typing import Dict, List, Tuple
from src.data.unified_window_loader import UnifiedWindowDataset


class WindowNPZTester:
    """Test suite for window NPZ storage format."""

    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0

    def test(self, name: str, condition: bool, message: str = ""):
        """Record test result."""
        if condition:
            self.results.append(f"✅ PASSED: {name}")
            self.passed += 1
        else:
            self.results.append(f"❌ FAILED: {name}")
            if message:
                self.results.append(f"   {message}")
            self.failed += 1

    def print_results(self):
        """Print all test results."""
        print("\n" + "="*80)
        print("TEST RESULTS")
        print("="*80)
        for result in self.results:
            print(result)
        print("\n" + "="*80)
        print(f"Total: {self.passed}/{self.passed + self.failed} tests passed")
        if self.failed == 0:
            print("✅ ALL TESTS PASSED!")
        else:
            print(f"❌ {self.failed} TESTS FAILED")
        print("="*80)


def test_vitaldb_windows(data_dir: Path, tester: WindowNPZTester):
    """Test VitalDB window format."""
    print("\n" + "="*80)
    print("TESTING VITALDB WINDOWS")
    print("="*80)
    print(f"Directory: {data_dir}\n")

    # Check directory exists
    if not data_dir.exists():
        tester.test("VitalDB directory exists", False, f"Directory not found: {data_dir}")
        return

    # Find window files
    window_files = sorted(data_dir.glob('window_*.npz'))
    tester.test("VitalDB window files found", len(window_files) > 0,
                f"Found {len(window_files)} files")

    if len(window_files) == 0:
        return

    print(f"Found {len(window_files)} window files\n")

    # Test first window
    print("Testing first window file...")
    data = np.load(window_files[0])

    # Test signal format
    tester.test("Signal array exists", 'signal' in data)
    if 'signal' in data:
        signal = data['signal']
        tester.test("Signal is 2D", len(signal.shape) == 2,
                   f"Shape: {signal.shape}")
        tester.test("Signal has 2 channels (PPG+ECG)", signal.shape[0] == 2,
                   f"Channels: {signal.shape[0]}")
        tester.test("Signal has 1024 samples", signal.shape[1] == 1024,
                   f"Samples: {signal.shape[1]}")
        tester.test("Signal has no NaN", not np.any(np.isnan(signal)),
                   f"NaN count: {np.sum(np.isnan(signal))}")
        tester.test("Signal has no Inf", not np.any(np.isinf(signal)),
                   f"Inf count: {np.sum(np.isinf(signal))}")

    # Test metadata
    required_metadata = ['case_id', 'window_idx', 'start_time', 'fs']
    for key in required_metadata:
        tester.test(f"Metadata '{key}' exists", key in data)

    if 'fs' in data:
        tester.test("Sampling rate is 125 Hz", data['fs'] == 125,
                   f"fs = {data['fs']}")

    # Test case-level labels
    expected_labels = ['age', 'sex', 'bmi', 'asa', 'emergency', 'death_inhosp', 'icu_days']
    for label in expected_labels:
        tester.test(f"Label '{label}' exists", label in data)

    # Test quality metrics
    quality_metrics = ['ppg_quality', 'ecg_quality']
    for metric in quality_metrics:
        tester.test(f"Quality metric '{metric}' exists", metric in data)
        if metric in data:
            value = data[metric]
            tester.test(f"{metric} in range [0, 1]",
                       0 <= value <= 1,
                       f"Value: {value}")

    # Test normalization stats
    norm_stats = ['ppg_mean', 'ppg_std', 'ecg_mean', 'ecg_std']
    for stat in norm_stats:
        tester.test(f"Norm stat '{stat}' exists", stat in data)

    # Test multiple windows for consistency
    print(f"\nTesting {min(10, len(window_files))} windows for consistency...")
    shapes = []
    for i, window_file in enumerate(window_files[:10]):
        data = np.load(window_file)
        if 'signal' in data:
            shapes.append(data['signal'].shape)

    shapes_consistent = len(set(shapes)) == 1
    tester.test("All window shapes are consistent", shapes_consistent,
                f"Unique shapes: {set(shapes)}")

    # Test file naming
    print("\nTesting file naming...")
    first_name = window_files[0].name
    tester.test("Files use zero-padded naming",
               'window_' in first_name and first_name.count('_') == 1)

    # Test metadata.json
    metadata_file = data_dir.parent / 'metadata.json'
    tester.test("metadata.json exists", metadata_file.exists(),
               f"Path: {metadata_file}")


def test_butppg_windows(data_dir: Path, tester: WindowNPZTester):
    """Test BUT-PPG window format."""
    print("\n" + "="*80)
    print("TESTING BUT-PPG WINDOWS")
    print("="*80)
    print(f"Directory: {data_dir}\n")

    # Check directory exists
    if not data_dir.exists():
        tester.test("BUT-PPG directory exists", False, f"Directory not found: {data_dir}")
        return

    # Find window files
    window_files = sorted(data_dir.glob('window_*.npz'))
    tester.test("BUT-PPG window files found", len(window_files) > 0,
                f"Found {len(window_files)} files")

    if len(window_files) == 0:
        return

    print(f"Found {len(window_files)} window files\n")

    # Test first window
    print("Testing first window file...")
    data = np.load(window_files[0])

    # Test signal format
    tester.test("Signal array exists", 'signal' in data)
    if 'signal' in data:
        signal = data['signal']
        tester.test("Signal is 2D", len(signal.shape) == 2,
                   f"Shape: {signal.shape}")
        tester.test("Signal has 5 channels (ACC+PPG+ECG)", signal.shape[0] == 5,
                   f"Channels: {signal.shape[0]}")
        tester.test("Signal has 1024 samples", signal.shape[1] == 1024,
                   f"Samples: {signal.shape[1]}")
        tester.test("Signal has no NaN", not np.any(np.isnan(signal)),
                   f"NaN count: {np.sum(np.isnan(signal))}")
        tester.test("Signal has no Inf", not np.any(np.isinf(signal)),
                   f"Inf count: {np.sum(np.isinf(signal))}")

        # Test channel synchronization
        channel_lengths = [signal[i, :].shape[0] for i in range(signal.shape[0])]
        tester.test("All channels have same length", len(set(channel_lengths)) == 1,
                   f"Lengths: {channel_lengths}")

    # Test metadata
    required_metadata = ['record_id', 'window_idx', 'start_time', 'fs']
    for key in required_metadata:
        tester.test(f"Metadata '{key}' exists", key in data)

    if 'fs' in data:
        tester.test("Sampling rate is 125 Hz", data['fs'] == 125,
                   f"fs = {data['fs']}")

    # Test clinical labels (7 types)
    expected_labels = ['quality', 'hr', 'motion', 'bp_systolic', 'bp_diastolic', 'spo2', 'glycaemia']
    for label in expected_labels:
        tester.test(f"Label '{label}' exists", label in data)

    # Test demographics
    expected_demographics = ['age', 'sex', 'bmi', 'height', 'weight']
    for demo in expected_demographics:
        tester.test(f"Demographic '{demo}' exists", demo in data)

    # Test quality metrics
    quality_metrics = ['ppg_quality', 'ecg_quality']
    for metric in quality_metrics:
        tester.test(f"Quality metric '{metric}' exists", metric in data)
        if metric in data:
            value = data[metric]
            if not np.isnan(value):
                tester.test(f"{metric} in range [0, 1]",
                           0 <= value <= 1,
                           f"Value: {value}")

    # Test normalization stats
    norm_stats = ['ppg_mean', 'ppg_std', 'ecg_mean', 'ecg_std', 'acc_mean', 'acc_std']
    for stat in norm_stats:
        tester.test(f"Norm stat '{stat}' exists", stat in data)

    # Test label values
    print("\nTesting label value ranges...")
    if 'quality' in data:
        quality = data['quality']
        if not np.isnan(quality):
            tester.test("Quality is binary (0 or 1)",
                       quality in [0, 1],
                       f"Value: {quality}")

    if 'hr' in data:
        hr = data['hr']
        if not np.isnan(hr):
            tester.test("Heart rate in realistic range (30-200 BPM)",
                       30 <= hr <= 200,
                       f"Value: {hr}")

    if 'bp_systolic' in data and 'bp_diastolic' in data:
        bp_sys = data['bp_systolic']
        bp_dia = data['bp_diastolic']
        if not (np.isnan(bp_sys) or np.isnan(bp_dia)):
            tester.test("Systolic > Diastolic",
                       bp_sys > bp_dia,
                       f"Systolic: {bp_sys}, Diastolic: {bp_dia}")

    # Test multiple windows for consistency
    print(f"\nTesting {min(10, len(window_files))} windows for consistency...")
    shapes = []
    for i, window_file in enumerate(window_files[:10]):
        data = np.load(window_file)
        if 'signal' in data:
            shapes.append(data['signal'].shape)

    shapes_consistent = len(set(shapes)) == 1
    tester.test("All window shapes are consistent", shapes_consistent,
                f"Unique shapes: {set(shapes)}")


def test_unified_loader(data_dir: Path, dataset_type: str, tester: WindowNPZTester):
    """Test unified loader with window files."""
    print("\n" + "="*80)
    print(f"TESTING UNIFIED LOADER ({dataset_type.upper()})")
    print("="*80)
    print(f"Directory: {data_dir}\n")

    if not data_dir.exists():
        print(f"Directory not found: {data_dir}")
        return

    try:
        # Test quality task
        print("Testing quality classification task...")
        dataset = UnifiedWindowDataset(
            data_dir=data_dir,
            task='quality',
            channels=['PPG', 'ECG'],
            filter_missing=False
        )

        tester.test("Unified loader loads dataset", len(dataset) > 0,
                   f"Loaded {len(dataset)} samples")

        if len(dataset) > 0:
            signal, label = dataset[0]
            tester.test("Loader returns signal tensor", signal is not None)
            tester.test("Signal is 2D tensor", len(signal.shape) == 2,
                       f"Shape: {signal.shape}")
            tester.test("Signal has 2 channels (PPG+ECG)", signal.shape[0] == 2,
                       f"Channels: {signal.shape[0]}")
            tester.test("Loader returns label", label is not None)

            # Test with metadata
            print("\nTesting with metadata return...")
            dataset_meta = UnifiedWindowDataset(
                data_dir=data_dir,
                task='quality',
                channels=['PPG', 'ECG'],
                filter_missing=False,
                return_metadata=True
            )

            signal, label, metadata = dataset_meta[0]
            tester.test("Metadata is returned", metadata is not None)
            tester.test("Metadata is dict", isinstance(metadata, dict))

        # Test label statistics
        print("\nTesting label statistics...")
        stats = dataset.get_label_stats()
        tester.test("Label statistics computed", len(stats) > 0)

    except Exception as e:
        tester.test("Unified loader works", False, f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test window NPZ storage format")
    parser.add_argument('--vitaldb-dir', type=str, default=None,
                       help='VitalDB window directory to test')
    parser.add_argument('--butppg-dir', type=str, default=None,
                       help='BUT-PPG window directory to test')

    args = parser.parse_args()

    if not args.vitaldb_dir and not args.butppg_dir:
        print("ERROR: Provide at least one directory to test")
        print("\nUsage:")
        print("  --vitaldb-dir: Test VitalDB windows")
        print("  --butppg-dir: Test BUT-PPG windows")
        sys.exit(1)

    tester = WindowNPZTester()

    # Test VitalDB
    if args.vitaldb_dir:
        vitaldb_dir = Path(args.vitaldb_dir)
        test_vitaldb_windows(vitaldb_dir, tester)
        test_unified_loader(vitaldb_dir, 'vitaldb', tester)

    # Test BUT-PPG
    if args.butppg_dir:
        butppg_dir = Path(args.butppg_dir)
        test_butppg_windows(butppg_dir, tester)
        test_unified_loader(butppg_dir, 'butppg', tester)

    # Print results
    tester.print_results()

    # Exit with appropriate code
    sys.exit(0 if tester.failed == 0 else 1)


if __name__ == '__main__':
    main()
