#!/usr/bin/env python3
"""Estimate the size of paired VitalDB dataset before rebuilding.

This script samples VitalDB cases to estimate how many paired PPG+ECG windows
we can expect after proper alignment and quality filtering.
"""

import vitaldb
import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def estimate_windows_for_case(case_id, window_size=1024, sampling_rate=125):
    """Estimate number of windows for a single case.

    Args:
        case_id: VitalDB case ID
        window_size: Window size in samples (default: 1024)
        sampling_rate: Sampling rate in Hz (default: 125)

    Returns:
        tuple: (num_windows, ppg_length, ecg_length, duration_minutes)
    """
    try:
        # Load both signals with 125 Hz sampling
        ppg = vitaldb.load(case_id, 'PLETH', interval=1/sampling_rate)
        ecg = vitaldb.load(case_id, 'ECG_II', interval=1/sampling_rate)

        if ppg is None or ecg is None:
            return None

        # Check for valid data
        if len(ppg) == 0 or len(ecg) == 0:
            return None

        # Get the minimum length (paired data constraint)
        min_len = min(len(ppg), len(ecg))

        # Calculate number of windows
        num_windows = min_len // window_size

        # Calculate duration in minutes
        duration_minutes = min_len / sampling_rate / 60

        return num_windows, len(ppg), len(ecg), duration_minutes

    except Exception as e:
        print(f"  ⚠️  Error loading case {case_id}: {e}", file=sys.stderr)
        return None


def main():
    """Main estimation function."""
    print("="*80)
    print("📊 VitalDB Paired Dataset Size Estimation")
    print("="*80)

    window_size = 1024
    sampling_rate = 125
    quality_pass_rate = 0.70  # Assume 70% of windows pass quality checks

    # Step 1: Find all cases with both signals
    print("\n🔍 Step 1: Finding cases with both PLETH and ECG_II...")

    try:
        ppg_cases = set(vitaldb.find_cases('PLETH'))
        print(f"   Found {len(ppg_cases)} cases with PLETH")
    except Exception as e:
        print(f"❌ Error finding PLETH cases: {e}")
        return

    try:
        ecg_cases = set(vitaldb.find_cases('ECG_II'))
        print(f"   Found {len(ecg_cases)} cases with ECG_II")
    except Exception as e:
        print(f"❌ Error finding ECG_II cases: {e}")
        return

    common_cases = sorted(list(ppg_cases.intersection(ecg_cases)))
    print(f"   ✓ {len(common_cases)} cases have BOTH signals")

    if len(common_cases) == 0:
        print("❌ No common cases found!")
        return

    # Step 2: Sample cases for estimation
    sample_size = min(10, len(common_cases))
    print(f"\n📈 Step 2: Sampling {sample_size} cases for estimation...")

    sampled_cases = common_cases[:sample_size]
    successful_samples = []
    total_windows = 0
    total_duration = 0
    ppg_lengths = []
    ecg_lengths = []

    for case_id in tqdm(sampled_cases, desc="Sampling cases"):
        result = estimate_windows_for_case(case_id, window_size, sampling_rate)

        if result is not None:
            num_windows, ppg_len, ecg_len, duration = result
            successful_samples.append({
                'case_id': case_id,
                'num_windows': num_windows,
                'ppg_length': ppg_len,
                'ecg_length': ecg_len,
                'duration_minutes': duration
            })
            total_windows += num_windows
            total_duration += duration
            ppg_lengths.append(ppg_len)
            ecg_lengths.append(ecg_len)

    if len(successful_samples) == 0:
        print("❌ No successful samples! Cannot estimate dataset size.")
        return

    print(f"   ✓ Successfully sampled {len(successful_samples)} cases")

    # Step 3: Calculate statistics
    print(f"\n{'='*80}")
    print("📊 SAMPLE STATISTICS")
    print(f"{'='*80}")

    avg_windows_per_case = total_windows / len(successful_samples)
    avg_duration = total_duration / len(successful_samples)
    avg_ppg_length = np.mean(ppg_lengths)
    avg_ecg_length = np.mean(ecg_lengths)

    print(f"\nPer-case averages (n={len(successful_samples)}):")
    print(f"  Windows per case: {avg_windows_per_case:.1f}")
    print(f"  Duration: {avg_duration:.1f} minutes ({avg_duration/60:.1f} hours)")
    print(f"  PPG length: {avg_ppg_length:,.0f} samples")
    print(f"  ECG length: {avg_ecg_length:,.0f} samples")

    # Print sample details
    print(f"\n📋 Sample case details:")
    print(f"{'Case ID':<10} {'Windows':>10} {'Duration (min)':>15} {'PPG Samples':>15} {'ECG Samples':>15}")
    print(f"{'-'*70}")
    for sample in successful_samples[:5]:  # Show first 5
        print(f"{sample['case_id']:<10} {sample['num_windows']:>10,} "
              f"{sample['duration_minutes']:>15.1f} "
              f"{sample['ppg_length']:>15,} {sample['ecg_length']:>15,}")
    if len(successful_samples) > 5:
        print(f"  ... and {len(successful_samples) - 5} more cases")

    # Step 4: Project to full dataset
    print(f"\n{'='*80}")
    print("🎯 DATASET SIZE PROJECTION")
    print(f"{'='*80}")

    estimated_total = avg_windows_per_case * len(common_cases)
    estimated_after_qc = estimated_total * quality_pass_rate

    print(f"\nTotal available cases: {len(common_cases)}")
    print(f"Average windows per case: {avg_windows_per_case:.0f}")
    print(f"\nEstimated raw windows: {estimated_total:,.0f}")
    print(f"After quality filtering ({quality_pass_rate*100:.0f}%): {estimated_after_qc:,.0f}")

    # Step 5: Train/Val/Test split projection
    print(f"\n{'='*80}")
    print("📂 PROJECTED TRAIN/VAL/TEST SPLIT (70/15/15)")
    print(f"{'='*80}")

    train_windows = estimated_after_qc * 0.70
    val_windows = estimated_after_qc * 0.15
    test_windows = estimated_after_qc * 0.15

    print(f"  Train: ~{train_windows:,.0f} windows")
    print(f"  Val:   ~{val_windows:,.0f} windows")
    print(f"  Test:  ~{test_windows:,.0f} windows")

    # Estimated training time
    estimated_hours = (avg_duration * len(common_cases)) / 60
    print(f"\nTotal data duration: ~{estimated_hours:,.0f} hours of recordings")

    # Step 6: Compare to current misaligned data
    print(f"\n{'='*80}")
    print("⚖️  COMPARISON TO CURRENT DATA")
    print(f"{'='*80}")

    current_ppg = 19262
    current_ecg = 8119
    current_total = current_ppg + current_ecg

    print(f"\nCurrent (misaligned):")
    print(f"  PPG: {current_ppg:,} windows")
    print(f"  ECG: {current_ecg:,} windows")
    print(f"  Total: {current_total:,} windows (but NOT paired!)")

    print(f"\nProjected (properly paired):")
    print(f"  Paired windows: ~{estimated_after_qc:,.0f}")
    print(f"  Each window has BOTH PPG and ECG")

    # Calculate the difference
    difference = estimated_after_qc - current_ppg
    percent_change = (difference / current_ppg) * 100

    print(f"\n{'='*80}")
    if difference > 0:
        print(f"✅ Expected INCREASE: +{difference:,.0f} windows ({percent_change:+.1f}%)")
        print(f"   This is because we'll have properly paired data instead of misaligned.")
    else:
        print(f"⚠️  Expected DECREASE: {difference:,.0f} windows ({percent_change:+.1f}%)")
        print(f"   This is the cost of proper alignment, but data will be CORRECT.")

    # Step 7: Recommendations
    print(f"\n{'='*80}")
    print("💡 RECOMMENDATIONS")
    print(f"{'='*80}")

    if estimated_after_qc > 20000:
        print("✅ Dataset size looks sufficient for SSL pre-training")
        print(f"   With ~{train_windows:,.0f} training windows, you should have enough data")
        print("   for effective self-supervised learning.")
    else:
        print("⚠️  Dataset might be on the smaller side for SSL")
        print("   Consider using data augmentation or focusing on smaller model.")

    print(f"\n🚀 Next step: Run the paired data preparation script")
    print(f"   This will take ~{len(common_cases) * 2} minutes for all {len(common_cases)} cases")

    print(f"\n{'='*80}")
    print("✅ Estimation complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
