"""Data integrity and split verification for biosignal evaluation.

This module provides functions to verify:
1. Subject-level split integrity (no data leakage)
2. Data quality (shapes, NaNs, Infs, normalization)
3. Preprocessing consistency

Critical for medical ML validation where data leakage can invalidate results.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def verify_subject_split(
    train_subjects: Union[List[str], List[int], Set],
    val_subjects: Union[List[str], List[int], Set],
    test_subjects: Union[List[str], List[int], Set],
    verbose: bool = True
) -> bool:
    """Verify no subject overlap between train/val/test splits.

    Critical for preventing data leakage in medical ML. Subject-level splits
    ensure that all windows from a given patient appear in only one split.

    Args:
        train_subjects: List/set of training subject IDs
        val_subjects: List/set of validation subject IDs
        test_subjects: List/set of test subject IDs
        verbose: Whether to print detailed report

    Returns:
        True if no leakage detected, False otherwise

    Example:
        >>> train = ['100', '101', '102']
        >>> val = ['110', '111']
        >>> test = ['120', '121']
        >>> verify_subject_split(train, val, test)
        === Subject-Level Split Verification ===
        Train subjects: 3
        Val subjects: 2
        Test subjects: 2

        Overlaps:
          Train-Val: 0 ✓
          Train-Test: 0 ✓
          Val-Test: 0 ✓

        ✓ PASS: No subject leakage detected!
        True
    """
    # Convert to sets for efficient intersection
    train_set = set(str(s) for s in train_subjects)
    val_set = set(str(s) for s in val_subjects)
    test_set = set(str(s) for s in test_subjects)

    # Check overlaps
    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set

    has_leakage = (
        len(train_val_overlap) > 0 or
        len(train_test_overlap) > 0 or
        len(val_test_overlap) > 0
    )

    if verbose:
        print("=" * 70)
        print("Subject-Level Split Verification")
        print("=" * 70)
        print(f"Train subjects: {len(train_set)}")
        print(f"Val subjects: {len(val_set)}")
        print(f"Test subjects: {len(test_set)}")
        print(f"Total unique subjects: {len(train_set | val_set | test_set)}")
        print(f"\nOverlaps:")
        print(f"  Train-Val: {len(train_val_overlap)} {'✓' if len(train_val_overlap)==0 else '✗ LEAKAGE!'}")
        print(f"  Train-Test: {len(train_test_overlap)} {'✓' if len(train_test_overlap)==0 else '✗ LEAKAGE!'}")
        print(f"  Val-Test: {len(val_test_overlap)} {'✓' if len(val_test_overlap)==0 else '✗ LEAKAGE!'}")

        if train_val_overlap:
            print(f"\n  ⚠️  Train-Val overlap subjects: {sorted(list(train_val_overlap))[:10]}")
        if train_test_overlap:
            print(f"  ⚠️  Train-Test overlap subjects: {sorted(list(train_test_overlap))[:10]}")
        if val_test_overlap:
            print(f"  ⚠️  Val-Test overlap subjects: {sorted(list(val_test_overlap))[:10]}")

        print()
        if has_leakage:
            print("✗ FAIL: Subject leakage detected!")
            print("   This will artificially inflate evaluation metrics.")
            print("   Fix: Re-split data at subject level, not segment level.")
        else:
            print("✓ PASS: No subject leakage detected!")
            print("   Safe to proceed with evaluation.")
        print("=" * 70)

    return not has_leakage


def verify_data_quality(
    signals: np.ndarray,
    expected_shape: Optional[Tuple[int, ...]] = None,
    expected_fs: int = 125,
    expected_duration_s: float = 10.0,
    check_normalization: bool = True,
    verbose: bool = True
) -> Dict[str, bool]:
    """Verify biosignal data quality and consistency.

    Checks for:
    - Correct shape [N, C, T] where T = fs * duration
    - No NaN or Inf values
    - Proper normalization (mean~0, std~1 per window)
    - Reasonable signal ranges

    Args:
        signals: Signal array, expected shape [N, C, T] or [N, T]
        expected_shape: Expected shape (None to skip check)
        expected_fs: Expected sampling rate in Hz
        expected_duration_s: Expected window duration in seconds
        check_normalization: Whether to check z-score normalization
        verbose: Whether to print detailed report

    Returns:
        Dictionary of check results:
            - 'shape_ok': Shape matches expected
            - 'no_nans': No NaN values present
            - 'no_infs': No Inf values present
            - 'normalized': Data appears normalized (if check_normalization=True)
            - 'all_passed': All checks passed

    Example:
        >>> data = np.random.randn(100, 2, 1250)  # 100 windows, 2 channels, 10s @ 125Hz
        >>> results = verify_data_quality(data, expected_fs=125, expected_duration_s=10.0)
        === Data Quality Verification ===
        Shape: (100, 2, 1250)
          N=100 windows, C=2 channels, T=1250 samples
        ✓ Window length: 1250 (10.0s @ 125Hz)
        ✓ NaNs: 0
        ✓ Infs: 0
        ✓ Data appears normalized (mean≈0, std≈1)

        ✓ ALL CHECKS PASSED
        >>> results['all_passed']
        True
    """
    results = {}

    if verbose:
        print("=" * 70)
        print("Data Quality Verification")
        print("=" * 70)
        print(f"Shape: {signals.shape}")

    # Handle 2D vs 3D arrays
    if signals.ndim == 2:
        N, T = signals.shape
        C = 1
        signals_3d = signals[:, np.newaxis, :]
        if verbose:
            print(f"  N={N} windows, T={T} samples (single channel)")
    elif signals.ndim == 3:
        N, C, T = signals.shape
        signals_3d = signals
        if verbose:
            print(f"  N={N} windows, C={C} channels, T={T} samples")
    else:
        if verbose:
            print(f"✗ ERROR: Expected 2D or 3D array, got {signals.ndim}D")
        results['shape_ok'] = False
        results['all_passed'] = False
        return results

    # Check expected length
    expected_T = int(expected_fs * expected_duration_s)
    shape_ok = True

    if expected_shape is not None:
        shape_ok = signals.shape == expected_shape
        if verbose:
            if shape_ok:
                print(f"✓ Shape matches expected: {expected_shape}")
            else:
                print(f"✗ Shape mismatch: expected {expected_shape}, got {signals.shape}")

    if T == expected_T:
        if verbose:
            print(f"✓ Window length: {T} ({expected_duration_s}s @ {expected_fs}Hz)")
    else:
        if verbose:
            print(f"✗ WARNING: Expected T={expected_T}, got T={T}")
            print(f"  (Expected {expected_duration_s}s @ {expected_fs}Hz)")
        shape_ok = False

    results['shape_ok'] = shape_ok

    # Check NaNs/Infs
    n_nans = np.isnan(signals_3d).sum()
    n_infs = np.isinf(signals_3d).sum()

    results['no_nans'] = (n_nans == 0)
    results['no_infs'] = (n_infs == 0)

    if verbose:
        print(f"{'✓' if n_nans==0 else '✗'} NaNs: {n_nans}")
        print(f"{'✓' if n_infs==0 else '✗'} Infs: {n_infs}")

    # Check normalization (per window)
    if check_normalization:
        # Compute per-window statistics
        means = signals_3d.mean(axis=2)  # [N, C]
        stds = signals_3d.std(axis=2)    # [N, C]

        # Check if roughly N(0,1)
        mean_of_means = np.abs(means.mean())
        mean_of_stds = stds.mean()

        normalized = (mean_of_means < 0.1) and (0.8 < mean_of_stds < 1.2)
        results['normalized'] = normalized

        if verbose:
            print(f"\nPer-window normalization check:")
            print(f"  Mean of means: {mean_of_means:.4f} (expect ≈0)")
            print(f"  Mean of stds: {mean_of_stds:.4f} (expect ≈1)")
            print(f"  Mean range: [{means.min():.3f}, {means.max():.3f}]")
            print(f"  Std range: [{stds.min():.3f}, {stds.max():.3f}]")

            if normalized:
                print(f"✓ Data appears normalized")
            else:
                print(f"✗ WARNING: Data may not be normalized")
                print(f"  Tip: Apply z-score normalization per channel per window")
    else:
        results['normalized'] = True  # Skip check

    # Signal range check
    signal_min = signals_3d.min()
    signal_max = signals_3d.max()
    signal_range_ok = (-10 < signal_min < 10) and (-10 < signal_max < 10)

    if verbose:
        print(f"\nSignal range:")
        print(f"  Min: {signal_min:.3f}, Max: {signal_max:.3f}")
        if signal_range_ok:
            print(f"✓ Signal range reasonable for normalized data")
        else:
            print(f"✗ WARNING: Extreme values detected (may indicate unnormalized data)")

    results['range_ok'] = signal_range_ok

    # Overall pass/fail
    all_passed = (
        results['shape_ok'] and
        results['no_nans'] and
        results['no_infs'] and
        results['normalized'] and
        results['range_ok']
    )
    results['all_passed'] = all_passed

    if verbose:
        print()
        if all_passed:
            print("✓ ALL CHECKS PASSED")
        else:
            print("✗ SOME CHECKS FAILED")
            failed = [k for k, v in results.items() if k != 'all_passed' and not v]
            print(f"  Failed checks: {failed}")
        print("=" * 70)

    return results


def verify_split_from_json(
    split_file: Union[str, Path],
    verbose: bool = True
) -> bool:
    """Verify subject-level splits from JSON file.

    Loads splits from JSON file (e.g., configs/splits/splits_full.json)
    and verifies no subject leakage.

    Args:
        split_file: Path to JSON file with 'train', 'val', 'test' keys
        verbose: Whether to print detailed report

    Returns:
        True if no leakage detected, False otherwise

    Example:
        >>> verify_split_from_json('configs/splits/splits_full.json')
        Loaded splits from: configs/splits/splits_full.json
        ✓ PASS: No subject leakage detected!
        True
    """
    import json

    split_path = Path(split_file)
    if not split_path.exists():
        logger.error(f"Split file not found: {split_path}")
        return False

    with open(split_path) as f:
        splits = json.load(f)

    if verbose:
        print(f"Loaded splits from: {split_path}")

    # Extract subject lists
    train_subjects = splits.get('train', [])
    val_subjects = splits.get('val', [])
    test_subjects = splits.get('test', [])

    return verify_subject_split(train_subjects, val_subjects, test_subjects, verbose=verbose)


def verify_preprocessing_consistency(
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
    tolerance: float = 0.5,
    verbose: bool = True
) -> bool:
    """Verify preprocessing consistency across splits.

    Checks that train/val/test have similar statistical properties,
    indicating consistent preprocessing. Large differences suggest
    preprocessing bugs or non-stationary data distributions.

    Args:
        train_data: Training data [N, C, T]
        val_data: Validation data [N, C, T]
        test_data: Test data [N, C, T]
        tolerance: Maximum allowed difference in mean/std ratios
        verbose: Whether to print detailed report

    Returns:
        True if preprocessing appears consistent, False otherwise

    Example:
        >>> train = np.random.randn(100, 2, 1250)
        >>> val = np.random.randn(20, 2, 1250)
        >>> test = np.random.randn(20, 2, 1250)
        >>> verify_preprocessing_consistency(train, val, test)
        ✓ Preprocessing appears consistent across splits
        True
    """
    if verbose:
        print("=" * 70)
        print("Preprocessing Consistency Check")
        print("=" * 70)

    # Compute global statistics for each split
    train_mean = np.mean(train_data)
    train_std = np.std(train_data)

    val_mean = np.mean(val_data)
    val_std = np.std(val_data)

    test_mean = np.mean(test_data)
    test_std = np.std(test_data)

    if verbose:
        print(f"Global statistics:")
        print(f"  Train: mean={train_mean:.4f}, std={train_std:.4f}")
        print(f"  Val:   mean={val_mean:.4f}, std={val_std:.4f}")
        print(f"  Test:  mean={test_mean:.4f}, std={test_std:.4f}")

    # Check if means are close to 0 and stds close to 1
    means_ok = (
        abs(train_mean) < 0.1 and
        abs(val_mean) < 0.1 and
        abs(test_mean) < 0.1
    )

    stds_ok = (
        0.8 < train_std < 1.2 and
        0.8 < val_std < 1.2 and
        0.8 < test_std < 1.2
    )

    # Check consistency between splits
    mean_diff = max(
        abs(train_mean - val_mean),
        abs(train_mean - test_mean),
        abs(val_mean - test_mean)
    )

    std_ratio = max(
        abs(train_std / val_std - 1),
        abs(train_std / test_std - 1),
        abs(val_std / test_std - 1)
    )

    consistent = (mean_diff < 0.1) and (std_ratio < tolerance)

    if verbose:
        print(f"\nConsistency metrics:")
        print(f"  Max mean difference: {mean_diff:.4f} (threshold: 0.1)")
        print(f"  Max std ratio difference: {std_ratio:.4f} (threshold: {tolerance})")
        print()

        if means_ok:
            print("✓ All splits have mean ≈ 0")
        else:
            print("✗ WARNING: Some splits have mean far from 0")

        if stds_ok:
            print("✓ All splits have std ≈ 1")
        else:
            print("✗ WARNING: Some splits have std far from 1")

        if consistent:
            print("✓ Preprocessing appears consistent across splits")
        else:
            print("✗ WARNING: Preprocessing may be inconsistent")
            print("  This could indicate preprocessing bugs or distribution shift")

        print("=" * 70)

    return means_ok and stds_ok and consistent


def generate_integrity_report(
    split_file: Union[str, Path],
    data_dir: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None
) -> Dict:
    """Generate comprehensive data integrity report.

    Runs all verification checks and generates a report.

    Args:
        split_file: Path to subject split JSON
        data_dir: Path to processed data directory
        output_file: Optional path to save report (JSON or txt)

    Returns:
        Dictionary with all verification results

    Example:
        >>> report = generate_integrity_report(
        ...     'configs/splits/splits_full.json',
        ...     'data/processed/vitaldb/windows'
        ... )
        >>> report['split_integrity']
        True
    """
    import json
    from datetime import datetime

    report = {
        'timestamp': datetime.now().isoformat(),
        'split_file': str(split_file),
        'data_dir': str(data_dir),
        'checks': {}
    }

    print("\n" + "=" * 70)
    print("DATA INTEGRITY REPORT")
    print("=" * 70)
    print(f"Split file: {split_file}")
    print(f"Data directory: {data_dir}")
    print(f"Timestamp: {report['timestamp']}")
    print("=" * 70 + "\n")

    # Check 1: Split integrity
    print("CHECK 1: Subject-Level Split Integrity")
    split_ok = verify_split_from_json(split_file, verbose=True)
    report['checks']['split_integrity'] = split_ok
    print()

    # Check 2: Data quality (if data files exist)
    data_path = Path(data_dir)
    if data_path.exists():
        print("CHECK 2: Data Quality")

        # Try to load sample data from each split
        for split in ['train', 'val', 'test']:
            split_dir = data_path / split
            if split_dir.exists():
                # Find first .npz file
                npz_files = list(split_dir.glob('*.npz'))
                if npz_files:
                    sample_file = npz_files[0]
                    print(f"\nChecking {split} split (sample: {sample_file.name})...")

                    try:
                        data = np.load(sample_file)
                        if 'data' in data:
                            signals = data['data']
                        elif 'windows' in data:
                            signals = data['windows']
                        elif 'signals' in data:
                            signals = data['signals']
                        else:
                            keys = [k for k in data.keys() if not k.startswith('_')]
                            signals = data[keys[0]] if keys else None

                        if signals is not None:
                            quality_results = verify_data_quality(
                                signals,
                                expected_fs=125,
                                expected_duration_s=10.0,
                                verbose=True
                            )
                            report['checks'][f'{split}_quality'] = quality_results
                        else:
                            print(f"✗ Could not find signal data in {sample_file.name}")
                            report['checks'][f'{split}_quality'] = {'error': 'no_signals'}
                    except Exception as e:
                        print(f"✗ Error loading {sample_file}: {e}")
                        report['checks'][f'{split}_quality'] = {'error': str(e)}
    else:
        print("CHECK 2: Data Quality - SKIPPED (data directory not found)")
        report['checks']['data_quality'] = 'skipped'

    # Overall status
    print("\n" + "=" * 70)
    all_passed = all(
        v.get('all_passed', v) if isinstance(v, dict) else v
        for v in report['checks'].values()
        if v not in ['skipped', None]
    )

    if all_passed:
        print("✓ OVERALL: ALL CHECKS PASSED")
        report['overall_status'] = 'PASS'
    else:
        print("✗ OVERALL: SOME CHECKS FAILED")
        report['overall_status'] = 'FAIL'
    print("=" * 70 + "\n")

    # Save report if output file specified
    if output_file:
        output_path = Path(output_file)

        if output_path.suffix == '.json':
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to: {output_path}")
        else:
            with open(output_path, 'w') as f:
                f.write(f"Data Integrity Report\n")
                f.write(f"Generated: {report['timestamp']}\n")
                f.write(f"Split file: {report['split_file']}\n")
                f.write(f"Data directory: {report['data_dir']}\n")
                f.write(f"\nOverall Status: {report['overall_status']}\n")
                f.write(f"\nDetailed Results:\n")
                for check, result in report['checks'].items():
                    f.write(f"  {check}: {result}\n")
            print(f"Report saved to: {output_path}")

    return report
