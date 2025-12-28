#!/usr/bin/env python3
"""Verify that PPG and ECG are properly paired in the rebuilt dataset.

This script checks:
1. Correct data shape [N, 2, 1024]
2. PPG and ECG are different signals
3. Temporal correlation between PPG and ECG
4. Visual alignment of signals
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
import sys


def verify_case(filepath):
    """Verify one case file.

    Args:
        filepath: Path to .npz file

    Returns:
        dict: Verification results
    """
    print(f"\n{'='*80}")
    print(f"📁 Verifying: {filepath.name}")
    print('='*80)

    # Load data
    try:
        data = np.load(filepath)
        windows = data['data']  # [N, 2, 1024]
        case_id = data['case_id'] if 'case_id' in data else 'unknown'
        fs = data['fs'] if 'fs' in data else 125
        window_size = data['window_size'] if 'window_size' in data else 1024
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None

    N, C, T = windows.shape
    print(f"Shape: {windows.shape}")
    print(f"  Case ID: {case_id}")
    print(f"  Sampling rate: {fs} Hz")
    print(f"  Window size: {window_size} samples ({window_size/fs:.3f} seconds)")

    # Verify shape
    try:
        assert C == 2, f"Expected 2 channels, got {C}"
        assert T == 1024, f"Expected 1024 samples, got {T}"
        print(f"  ✅ Shape correct: [{N}, 2, 1024]")
    except AssertionError as e:
        print(f"  ❌ Shape error: {e}")
        return None

    # Extract channels
    ppg_all = windows[:, 0, :]  # [N, 1024]
    ecg_all = windows[:, 1, :]  # [N, 1024]

    # Verify they're different
    try:
        assert not np.allclose(ppg_all, ecg_all, rtol=1e-3), "PPG and ECG are identical!"
        print(f"  ✅ PPG and ECG are different signals")
    except AssertionError as e:
        print(f"  ❌ Signal error: {e}")
        return None

    # Check for NaN/Inf
    ppg_valid = not (np.any(np.isnan(ppg_all)) or np.any(np.isinf(ppg_all)))
    ecg_valid = not (np.any(np.isnan(ecg_all)) or np.any(np.isinf(ecg_all)))

    if ppg_valid and ecg_valid:
        print(f"  ✅ No NaN/Inf values in either signal")
    else:
        print(f"  ❌ Invalid values detected: PPG valid={ppg_valid}, ECG valid={ecg_valid}")
        return None

    # Compute statistics
    ppg_mean = np.mean(ppg_all)
    ppg_std = np.std(ppg_all)
    ecg_mean = np.mean(ecg_all)
    ecg_std = np.std(ecg_all)

    print(f"\n  📊 Signal statistics:")
    print(f"     PPG: mean={ppg_mean:.3f}, std={ppg_std:.3f}")
    print(f"     ECG: mean={ecg_mean:.3f}, std={ecg_std:.3f}")

    # Compute cross-correlations
    print(f"\n  🔗 Computing PPG-ECG correlations...")
    correlations = []
    num_samples = min(10, N)  # Sample up to 10 windows

    for i in range(num_samples):
        try:
            corr, pval = pearsonr(ppg_all[i], ecg_all[i])
            correlations.append(corr)
        except:
            correlations.append(np.nan)

    correlations = [c for c in correlations if not np.isnan(c)]

    if len(correlations) > 0:
        avg_corr = np.mean(correlations)
        min_corr = np.min(correlations)
        max_corr = np.max(correlations)

        print(f"     Average correlation: {avg_corr:.3f}")
        print(f"     Range: [{min_corr:.3f}, {max_corr:.3f}]")

        if avg_corr > 0.3:
            print(f"  ✅ Good correlation (signals are properly aligned)")
        elif avg_corr > 0.2:
            print(f"  ⚠️  Acceptable correlation")
        else:
            print(f"  ❌ Low correlation (might indicate misalignment)")
    else:
        avg_corr = 0.0
        print(f"  ⚠️  Could not compute correlations")

    # Create visualization for first window
    print(f"\n  🎨 Creating visualization...")
    ppg_sample = ppg_all[0]
    ecg_sample = ecg_all[0]

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Time axis
    time = np.arange(T) / fs

    # 1. PPG signal
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time, ppg_sample, 'b-', linewidth=1.0, label='PPG (Channel 0)')
    ax1.set_ylabel('PPG Amplitude', fontsize=11)
    ax1.set_title(f'{filepath.stem} - Window 0 | PPG-ECG Correlation: {correlations[0]:.3f}',
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

    # 2. ECG signal
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(time, ecg_sample, 'r-', linewidth=1.0, label='ECG (Channel 1)')
    ax2.set_ylabel('ECG Amplitude', fontsize=11)
    ax2.set_xlabel('Time (seconds)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    # 3. Overlay (zoomed)
    ax3 = fig.add_subplot(gs[2, 0])
    zoom_start = int(T * 0.2)
    zoom_end = int(T * 0.4)
    time_zoom = time[zoom_start:zoom_end]

    ax3.plot(time_zoom, ppg_sample[zoom_start:zoom_end], 'b-', linewidth=1.5,
             label='PPG', alpha=0.8)
    ax3_twin = ax3.twinx()
    ax3_twin.plot(time_zoom, ecg_sample[zoom_start:zoom_end], 'r-', linewidth=1.5,
                  label='ECG', alpha=0.8)
    ax3.set_xlabel('Time (seconds)', fontsize=10)
    ax3.set_ylabel('PPG', color='b', fontsize=10)
    ax3_twin.set_ylabel('ECG', color='r', fontsize=10)
    ax3.set_title('Zoomed View (20-40%)', fontsize=11)
    ax3.grid(True, alpha=0.3)

    # 4. Scatter plot
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.scatter(ppg_sample[::10], ecg_sample[::10], alpha=0.5, s=10)
    ax4.set_xlabel('PPG Amplitude', fontsize=10)
    ax4.set_ylabel('ECG Amplitude', fontsize=10)
    ax4.set_title(f'PPG vs ECG (Correlation: {correlations[0]:.3f})', fontsize=11)
    ax4.grid(True, alpha=0.3)

    # Add text box with summary
    summary_text = f"""
    Windows: {N}
    Duration: {T/fs:.2f}s per window
    Sampling Rate: {fs} Hz
    Avg Correlation: {avg_corr:.3f}
    """
    fig.text(0.02, 0.02, summary_text, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle(f'Paired PPG-ECG Verification', fontsize=14, fontweight='bold', y=0.98)

    output_file = f'verify_{filepath.stem}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✅ Visualization saved: {output_file}")

    return {
        'case': filepath.name,
        'num_windows': N,
        'avg_correlation': avg_corr,
        'shape': windows.shape,
        'ppg_stats': {'mean': ppg_mean, 'std': ppg_std},
        'ecg_stats': {'mean': ecg_mean, 'std': ecg_std}
    }


def main():
    """Main verification function."""
    data_dir = Path('data/processed/vitaldb/paired_1024/train')

    if not data_dir.exists():
        print(f"❌ Directory not found: {data_dir}")
        print("   Run rebuild_vitaldb_paired.py first!")
        return

    files = sorted(list(data_dir.glob('case_*.npz')))

    if len(files) == 0:
        print(f"❌ No case files found in: {data_dir}")
        return

    print("="*80)
    print("🔬 PAIRED DATA VERIFICATION")
    print("="*80)
    print(f"Data directory: {data_dir.absolute()}")
    print(f"Found {len(files)} case files")

    # Verify first 5 cases
    num_to_check = min(5, len(files))
    print(f"\nChecking first {num_to_check} cases...")

    results = []
    for f in files[:num_to_check]:
        try:
            result = verify_case(f)
            if result is not None:
                results.append(result)
        except Exception as e:
            print(f"\n❌ Error verifying {f.name}: {e}")
            import traceback
            traceback.print_exc()

    # Overall summary
    if len(results) > 0:
        avg_corr_all = np.mean([r['avg_correlation'] for r in results])
        total_windows = sum([r['num_windows'] for r in results])

        print("\n" + "="*80)
        print("📊 VERIFICATION SUMMARY")
        print("="*80)
        print(f"Cases verified: {len(results)}/{num_to_check}")
        print(f"Total windows: {total_windows}")
        print(f"Average PPG-ECG correlation: {avg_corr_all:.3f}")

        print(f"\n📋 Per-case breakdown:")
        for r in results:
            print(f"  {r['case']:30s} - {r['num_windows']:3d} windows, "
                  f"correlation: {r['avg_correlation']:.3f}")

        print(f"\n{'='*80}")
        if avg_corr_all > 0.3:
            print("✅ VERIFICATION PASSED!")
            print("   PPG and ECG show good correlation")
            print("   Signals are from same cardiac events")
            print("   Data is properly aligned")
        elif avg_corr_all > 0.2:
            print("⚠️  VERIFICATION MARGINAL")
            print("   Correlation is acceptable but could be better")
            print("   Signals are likely aligned but review visualizations")
        else:
            print("❌ VERIFICATION FAILED!")
            print("   Very low correlation suggests misalignment")
            print("   Review visualizations to diagnose issue")

        print(f"\n📸 Visualizations saved: verify_case_*.png")
        print(f"   Check these images to visually confirm alignment")
        print("="*80)
    else:
        print("\n❌ No cases could be verified!")
        print("   Check error messages above for details")


if __name__ == '__main__':
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
