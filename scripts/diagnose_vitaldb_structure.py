#!/usr/bin/env python3
"""Diagnostic script to understand VitalDB data structure and channel mismatches.

This script analyzes the processed VitalDB windows to identify why PPG and ECG
have different numbers of windows (19,262 vs 8,119).
"""

import os
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def analyze_npz_file(filepath):
    """Analyze a single .npz file and return its statistics.

    Args:
        filepath: Path to the .npz file

    Returns:
        dict: Statistics about the file
    """
    try:
        data = np.load(filepath)

        # Get the main array (usually 'arr_0' or a named key)
        keys = list(data.keys())
        if len(keys) == 0:
            return None

        # Try common keys
        array_key = None
        for key in ['windows', 'data', 'arr_0']:
            if key in keys:
                array_key = key
                break

        if array_key is None:
            array_key = keys[0]

        array = data[array_key]

        stats = {
            'filename': filepath.name,
            'keys': keys,
            'array_key': array_key,
            'shape': array.shape,
            'dtype': array.dtype,
            'num_windows': array.shape[0] if len(array.shape) > 0 else 0,
            'min': float(np.nanmin(array)) if array.size > 0 else None,
            'max': float(np.nanmax(array)) if array.size > 0 else None,
            'has_nan': bool(np.any(np.isnan(array))),
            'has_inf': bool(np.any(np.isinf(array))),
        }

        data.close()
        return stats

    except Exception as e:
        return {
            'filename': filepath.name,
            'error': str(e)
        }


def scan_directory(base_path, modality):
    """Scan a modality directory and analyze all .npz files.

    Args:
        base_path: Base path to the windows directory
        modality: 'ppg' or 'ecg'

    Returns:
        dict: Mapping of case_id to file statistics
    """
    modality_path = base_path / modality

    if not modality_path.exists():
        print(f"⚠️  Directory not found: {modality_path}")
        return {}

    files = sorted(modality_path.glob("*.npz"))
    print(f"\n{'='*80}")
    print(f"📁 Scanning {modality.upper()} directory: {modality_path}")
    print(f"   Found {len(files)} .npz files")
    print(f"{'='*80}")

    results = {}
    total_windows = 0

    for filepath in files:
        # Extract case_id from filename (e.g., "case_123.npz" -> "123")
        case_id = filepath.stem.replace('case_', '')

        stats = analyze_npz_file(filepath)
        results[case_id] = stats

        if stats and 'error' not in stats:
            total_windows += stats['num_windows']

    print(f"\n📊 {modality.upper()} Summary:")
    print(f"   Total files: {len(results)}")
    print(f"   Total windows: {total_windows:,}")

    return results


def compare_modalities(ppg_data, ecg_data):
    """Compare PPG and ECG data to find mismatches.

    Args:
        ppg_data: Dictionary of PPG file statistics
        ecg_data: Dictionary of ECG file statistics
    """
    print(f"\n{'='*80}")
    print("🔍 COMPARISON ANALYSIS")
    print(f"{'='*80}")

    all_cases = sorted(set(ppg_data.keys()) | set(ecg_data.keys()))

    ppg_only = set(ppg_data.keys()) - set(ecg_data.keys())
    ecg_only = set(ecg_data.keys()) - set(ppg_data.keys())
    common = set(ppg_data.keys()) & set(ecg_data.keys())

    print(f"\n📈 Case Coverage:")
    print(f"   PPG files: {len(ppg_data)}")
    print(f"   ECG files: {len(ecg_data)}")
    print(f"   Common cases: {len(common)}")
    print(f"   PPG-only cases: {len(ppg_only)}")
    print(f"   ECG-only cases: {len(ecg_only)}")

    if ppg_only:
        print(f"\n⚠️  Cases with PPG but no ECG: {sorted(ppg_only)[:10]}")
        if len(ppg_only) > 10:
            print(f"      ... and {len(ppg_only) - 10} more")

    if ecg_only:
        print(f"\n⚠️  Cases with ECG but no PPG: {sorted(ecg_only)[:10]}")
        if len(ecg_only) > 10:
            print(f"      ... and {len(ecg_only) - 10} more")

    # Detailed comparison table
    print(f"\n{'='*80}")
    print("📋 DETAILED FILE COMPARISON")
    print(f"{'='*80}")
    print(f"{'Case ID':<12} {'PPG Windows':>15} {'ECG Windows':>15} {'Difference':>15} {'Status':<20}")
    print(f"{'-'*80}")

    mismatch_cases = []
    total_ppg_windows = 0
    total_ecg_windows = 0

    for case_id in sorted(all_cases, key=lambda x: int(x) if x.isdigit() else 0):
        ppg_stats = ppg_data.get(case_id)
        ecg_stats = ecg_data.get(case_id)

        ppg_windows = ppg_stats['num_windows'] if ppg_stats and 'num_windows' in ppg_stats else 0
        ecg_windows = ecg_stats['num_windows'] if ecg_stats and 'num_windows' in ecg_stats else 0

        total_ppg_windows += ppg_windows
        total_ecg_windows += ecg_windows

        diff = ppg_windows - ecg_windows

        # Determine status
        if case_id in ppg_only:
            status = "PPG ONLY"
        elif case_id in ecg_only:
            status = "ECG ONLY"
        elif diff != 0:
            status = f"MISMATCH ({diff:+d})"
            mismatch_cases.append((case_id, ppg_windows, ecg_windows, diff))
        else:
            status = "✓ MATCHED"

        # Print row
        print(f"{case_id:<12} {ppg_windows:>15,} {ecg_windows:>15,} {diff:>+15,} {status:<20}")

    print(f"{'-'*80}")
    print(f"{'TOTAL':<12} {total_ppg_windows:>15,} {total_ecg_windows:>15,} {total_ppg_windows - total_ecg_windows:>+15,}")

    # Summary of mismatches
    print(f"\n{'='*80}")
    print("📊 MISMATCH SUMMARY")
    print(f"{'='*80}")
    print(f"Total PPG windows: {total_ppg_windows:,}")
    print(f"Total ECG windows: {total_ecg_windows:,}")
    print(f"Difference: {total_ppg_windows - total_ecg_windows:,} windows")
    print(f"\nCases with window count mismatch: {len(mismatch_cases)}")

    if mismatch_cases:
        print(f"\n🔴 Top 10 cases contributing to mismatch:")
        sorted_mismatches = sorted(mismatch_cases, key=lambda x: abs(x[3]), reverse=True)
        for case_id, ppg_w, ecg_w, diff in sorted_mismatches[:10]:
            print(f"   Case {case_id}: PPG={ppg_w:,}, ECG={ecg_w:,}, Diff={diff:+,}")

    # Sample file details
    print(f"\n{'='*80}")
    print("🔬 SAMPLE FILE DETAILS (First 5 cases)")
    print(f"{'='*80}")

    for case_id in sorted(all_cases, key=lambda x: int(x) if x.isdigit() else 0)[:5]:
        print(f"\nCase {case_id}:")

        if case_id in ppg_data:
            ppg_stats = ppg_data[case_id]
            if 'error' in ppg_stats:
                print(f"  PPG: ERROR - {ppg_stats['error']}")
            else:
                print(f"  PPG: shape={ppg_stats['shape']}, "
                      f"windows={ppg_stats['num_windows']:,}, "
                      f"range=[{ppg_stats['min']:.2f}, {ppg_stats['max']:.2f}], "
                      f"NaN={ppg_stats['has_nan']}, Inf={ppg_stats['has_inf']}")
        else:
            print(f"  PPG: MISSING")

        if case_id in ecg_data:
            ecg_stats = ecg_data[case_id]
            if 'error' in ecg_stats:
                print(f"  ECG: ERROR - {ecg_stats['error']}")
            else:
                print(f"  ECG: shape={ecg_stats['shape']}, "
                      f"windows={ecg_stats['num_windows']:,}, "
                      f"range=[{ecg_stats['min']:.2f}, {ecg_stats['max']:.2f}], "
                      f"NaN={ecg_stats['has_nan']}, Inf={ecg_stats['has_inf']}")
        else:
            print(f"  ECG: MISSING")


def main():
    """Main diagnostic function."""
    print("="*80)
    print("🔬 VitalDB Data Structure Diagnostic Tool")
    print("="*80)

    # Define paths
    base_path = Path("data/processed/vitaldb/windows/train")

    if not base_path.exists():
        print(f"\n❌ ERROR: Base path not found: {base_path}")
        print(f"   Current working directory: {os.getcwd()}")
        return

    print(f"\n📂 Base path: {base_path.absolute()}")

    # Check directory structure
    print(f"\n📁 Directory structure:")
    for item in sorted(base_path.iterdir()):
        if item.is_dir():
            num_files = len(list(item.glob("*.npz")))
            print(f"   {item.name}/ ({num_files} .npz files)")
        else:
            print(f"   {item.name}")

    # Scan both modalities
    ppg_data = scan_directory(base_path, 'ppg')
    ecg_data = scan_directory(base_path, 'ecg')

    # Compare
    if ppg_data and ecg_data:
        compare_modalities(ppg_data, ecg_data)
    elif not ppg_data:
        print("\n❌ No PPG data found!")
    elif not ecg_data:
        print("\n❌ No ECG data found!")

    print(f"\n{'='*80}")
    print("✅ Diagnostic complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
