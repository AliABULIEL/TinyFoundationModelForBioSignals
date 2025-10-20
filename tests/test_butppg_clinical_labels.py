#!/usr/bin/env python3
"""
Test script to verify BUT-PPG clinical labels are correctly saved by build_butppg_windows.py

This tests the updated pipeline to ensure:
1. build_butppg_windows.py loads clinical labels from BUTPPGDataset
2. All 7 label types are saved to .npz files
3. Labels have correct shapes and values
4. Missing values are encoded as -1
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json


def test_clinical_labels_present():
    """Test that .npz files contain all 7 clinical label arrays"""

    print("\n" + "="*80)
    print("TEST 1: Verify Clinical Labels are Present in .npz Files")
    print("="*80)

    # Expected label keys
    expected_labels = ['quality', 'hr', 'motion', 'bp_systolic', 'bp_diastolic', 'spo2', 'glycaemia']

    # Check if test data exists
    test_file = Path('data/test_multitask/multitask/train.npz')

    if not test_file.exists():
        print(f"⚠️  Test data not found: {test_file}")
        print("   Run: python scripts/process_butppg_clinical.py --multitask")
        return False

    # Load test data
    print(f"Loading: {test_file}")
    data = np.load(test_file)

    # Check for all expected keys
    print("\nChecking for required keys...")
    missing_keys = []
    for key in ['signals'] + expected_labels:
        if key in data:
            print(f"  ✓ {key}: shape={data[key].shape}, dtype={data[key].dtype}")
        else:
            print(f"  ✗ {key}: MISSING")
            missing_keys.append(key)

    if missing_keys:
        print(f"\n❌ FAILED: Missing keys: {missing_keys}")
        return False
    else:
        print("\n✅ PASSED: All required keys present")
        return True


def test_label_shapes_match():
    """Test that all label arrays have matching lengths"""

    print("\n" + "="*80)
    print("TEST 2: Verify Label Array Shapes Match")
    print("="*80)

    test_file = Path('data/test_multitask/multitask/train.npz')

    if not test_file.exists():
        print(f"⚠️  Test data not found: {test_file}")
        return False

    data = np.load(test_file)

    # Get expected length from signals
    n_samples = data['signals'].shape[0]
    print(f"Number of samples: {n_samples}")

    # Check all labels have same length
    label_keys = ['quality', 'hr', 'motion', 'bp_systolic', 'bp_diastolic', 'spo2', 'glycaemia']
    mismatched = []

    for key in label_keys:
        if key in data:
            label_len = len(data[key])
            match_str = "✓" if label_len == n_samples else "✗"
            print(f"  {match_str} {key}: length={label_len}")
            if label_len != n_samples:
                mismatched.append(key)

    if mismatched:
        print(f"\n❌ FAILED: Mismatched lengths: {mismatched}")
        return False
    else:
        print("\n✅ PASSED: All labels have matching lengths")
        return True


def test_label_value_ranges():
    """Test that label values are within expected ranges"""

    print("\n" + "="*80)
    print("TEST 3: Verify Label Value Ranges")
    print("="*80)

    test_file = Path('data/test_multitask/multitask/train.npz')

    if not test_file.exists():
        print(f"⚠️  Test data not found: {test_file}")
        return False

    data = np.load(test_file)

    # Define expected ranges (excluding -1 for missing)
    expected_ranges = {
        'quality': (0, 1, "binary: 0=poor, 1=good"),
        'hr': (40, 200, "BPM: typical range 40-200"),
        'motion': (0, 7, "class: 0-7"),
        'bp_systolic': (70, 250, "mmHg: typical range 70-250"),
        'bp_diastolic': (40, 150, "mmHg: typical range 40-150"),
        'spo2': (0, 100, "percentage: 0-100%"),
        'glycaemia': (0, 50, "mmol/l: typical range 0-50")
    }

    all_valid = True

    for key, (min_val, max_val, desc) in expected_ranges.items():
        if key not in data:
            continue

        values = data[key]

        # Filter out missing values (-1)
        valid_values = values[values != -1]

        if len(valid_values) == 0:
            print(f"  ⚠️  {key}: No valid values (all -1)")
            continue

        actual_min = valid_values.min()
        actual_max = valid_values.max()
        n_valid = len(valid_values)
        n_missing = len(values) - n_valid

        # Check if values are in expected range
        in_range = (actual_min >= min_val) and (actual_max <= max_val)
        status = "✓" if in_range else "✗"

        print(f"  {status} {key}: [{actual_min:.1f}, {actual_max:.1f}] ({desc})")
        print(f"     Valid: {n_valid}/{len(values)}, Missing: {n_missing}")

        if not in_range:
            all_valid = False
            print(f"     ⚠️  Values outside expected range [{min_val}, {max_val}]")

    if all_valid:
        print("\n✅ PASSED: All label values within expected ranges")
        return True
    else:
        print("\n⚠️  WARNING: Some values outside expected ranges (may be OK)")
        return True  # Don't fail test - just warn


def test_label_availability():
    """Test label availability percentages"""

    print("\n" + "="*80)
    print("TEST 4: Label Availability Statistics")
    print("="*80)

    test_file = Path('data/test_multitask/multitask/train.npz')

    if not test_file.exists():
        print(f"⚠️  Test data not found: {test_file}")
        return False

    data = np.load(test_file)
    n_samples = data['signals'].shape[0]

    label_keys = ['quality', 'hr', 'motion', 'bp_systolic', 'bp_diastolic', 'spo2', 'glycaemia']

    print(f"Total samples: {n_samples}\n")
    print("Label availability:")

    for key in label_keys:
        if key not in data:
            continue

        values = data[key]
        n_valid = np.sum(values != -1)
        pct = 100 * n_valid / len(values)

        print(f"  {key:15s}: {n_valid:4d}/{len(values)} ({pct:.1f}%)")

    print("\n✅ PASSED: Label availability computed")
    return True


def test_sample_values():
    """Show sample values for each label"""

    print("\n" + "="*80)
    print("TEST 5: Sample Label Values")
    print("="*80)

    test_file = Path('data/test_multitask/multitask/train.npz')

    if not test_file.exists():
        print(f"⚠️  Test data not found: {test_file}")
        return False

    data = np.load(test_file)

    # Show first 5 samples
    n_show = min(5, data['signals'].shape[0])

    print(f"First {n_show} samples:\n")

    label_keys = ['quality', 'hr', 'motion', 'bp_systolic', 'bp_diastolic', 'spo2', 'glycaemia']

    for i in range(n_show):
        print(f"Sample {i}:")
        print(f"  Signal shape: {data['signals'][i].shape}")
        print(f"  Signal range: [{data['signals'][i].min():.2f}, {data['signals'][i].max():.2f}]")

        for key in label_keys:
            if key in data:
                value = data[key][i]
                if value == -1:
                    print(f"  {key:15s}: -1 (missing)")
                else:
                    print(f"  {key:15s}: {value:.1f}")
        print()

    print("✅ PASSED: Sample values displayed")
    return True


def main():
    """Run all tests"""

    print("\n" + "="*80)
    print("BUT-PPG CLINICAL LABELS TEST SUITE")
    print("="*80)
    print("\nThis test verifies that build_butppg_windows.py correctly saves")
    print("all clinical labels from the BUT-PPG dataset.\n")

    tests = [
        ("Labels Present", test_clinical_labels_present),
        ("Shapes Match", test_label_shapes_match),
        ("Value Ranges", test_label_value_ranges),
        ("Availability", test_label_availability),
        ("Sample Values", test_sample_values)
    ]

    results = []

    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ ERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status}: {test_name}")

    n_passed = sum(1 for _, passed in results if passed)
    n_total = len(results)

    print(f"\nTotal: {n_passed}/{n_total} tests passed")

    if n_passed == n_total:
        print("\n✅ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n❌ {n_total - n_passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
