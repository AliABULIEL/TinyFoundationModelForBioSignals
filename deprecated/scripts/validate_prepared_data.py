#!/usr/bin/env python3
"""
Validate synchronized PPG-ECG data quality after prepare_all_data.py fix.

This script verifies:
1. Data structure and format correctness
2. Temporal synchronization between PPG and ECG
3. Signal quality metrics (SNR, frequency content)
4. Statistical summaries and pass/fail criteria

Usage:
    python scripts/validate_prepared_data.py --data-dir data/processed/vitaldb/paired
    python scripts/validate_prepared_data.py --data-dir data/processed/vitaldb/paired --split train
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.signal import correlate, find_peaks, welch
from tqdm import tqdm


def load_window(file_path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load and return PPG, ECG from .npz file.

    Args:
        file_path: Path to .npz file

    Returns:
        Tuple of (ppg, ecg) arrays, or (None, None) if invalid
    """
    try:
        data = np.load(file_path)

        if 'data' not in data:
            return None, None

        windows = data['data']  # Expected: [N, 2, 1024]

        if windows.ndim != 3 or windows.shape[1] != 2:
            return None, None

        # Extract channels: 0=PPG, 1=ECG
        ppg = windows[:, 0, :]  # [N, 1024]
        ecg = windows[:, 1, :]  # [N, 1024]

        return ppg, ecg

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None


def compute_correlation(ppg: np.ndarray, ecg: np.ndarray) -> float:
    """Compute Pearson correlation between PPG and ECG signals.

    Args:
        ppg: PPG signal [samples]
        ecg: ECG signal [samples]

    Returns:
        Pearson correlation coefficient
    """
    try:
        corr, _ = stats.pearsonr(ppg, ecg)
        return corr if not np.isnan(corr) else 0.0
    except:
        return 0.0


def compute_cross_correlation(ppg: np.ndarray, ecg: np.ndarray,
                              fs: int = 125) -> Tuple[float, int]:
    """Compute cross-correlation and find peak lag.

    Args:
        ppg: PPG signal [samples]
        ecg: ECG signal [samples]
        fs: Sampling rate in Hz

    Returns:
        Tuple of (max_correlation, peak_lag_samples)
    """
    try:
        # Normalize signals
        ppg_norm = (ppg - np.mean(ppg)) / (np.std(ppg) + 1e-8)
        ecg_norm = (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-8)

        # Compute cross-correlation
        xcorr = correlate(ppg_norm, ecg_norm, mode='same')
        xcorr = xcorr / len(ppg)

        # Find peak
        center = len(xcorr) // 2
        max_idx = np.argmax(np.abs(xcorr))
        peak_lag = max_idx - center
        max_corr = xcorr[max_idx]

        return float(max_corr), int(peak_lag)

    except Exception as e:
        return 0.0, 0


def compute_snr(signal_data: np.ndarray, fs: int = 125) -> float:
    """Estimate SNR using signal power vs noise floor.

    Args:
        signal_data: Input signal [samples]
        fs: Sampling rate in Hz

    Returns:
        SNR in dB
    """
    try:
        # Compute power spectral density
        freqs, psd = welch(signal_data, fs=fs, nperseg=min(256, len(signal_data)))

        # Signal band (1-3 Hz for cardiac signals)
        signal_mask = (freqs >= 1.0) & (freqs <= 3.0)

        # Noise band (high frequencies, 20-50 Hz)
        noise_mask = (freqs >= 20.0) & (freqs <= 50.0)

        if np.sum(signal_mask) == 0 or np.sum(noise_mask) == 0:
            return 0.0

        signal_power = np.mean(psd[signal_mask])
        noise_power = np.mean(psd[noise_mask])

        if noise_power <= 0:
            return 100.0  # Very high SNR

        snr_db = 10 * np.log10(signal_power / noise_power)
        return float(snr_db)

    except:
        return 0.0


def compute_dominant_frequency(signal_data: np.ndarray, fs: int = 125) -> float:
    """Compute dominant frequency in signal.

    Args:
        signal_data: Input signal [samples]
        fs: Sampling rate in Hz

    Returns:
        Dominant frequency in Hz
    """
    try:
        freqs, psd = welch(signal_data, fs=fs, nperseg=min(256, len(signal_data)))

        # Focus on cardiac range (0.5-3 Hz)
        cardiac_mask = (freqs >= 0.5) & (freqs <= 3.0)

        if np.sum(cardiac_mask) == 0:
            return 0.0

        dominant_idx = np.argmax(psd[cardiac_mask])
        dominant_freq = freqs[cardiac_mask][dominant_idx]

        return float(dominant_freq)

    except:
        return 0.0


def check_signal_quality(signal_data: np.ndarray) -> Dict:
    """Check various signal quality metrics.

    Args:
        signal_data: Input signal [samples]

    Returns:
        Dictionary with quality metrics
    """
    has_nan = np.any(np.isnan(signal_data))
    has_inf = np.any(np.isinf(signal_data))
    variance = np.var(signal_data)
    is_zero_var = variance < 1e-6

    # Check for saturation (clipping)
    signal_min = np.min(signal_data)
    signal_max = np.max(signal_data)
    signal_range = signal_max - signal_min

    # Count samples at extremes (potential clipping)
    threshold = 0.01 * signal_range
    clipped_low = np.sum(signal_data <= (signal_min + threshold))
    clipped_high = np.sum(signal_data >= (signal_max - threshold))
    saturation_ratio = (clipped_low + clipped_high) / len(signal_data)

    return {
        'has_nan': bool(has_nan),
        'has_inf': bool(has_inf),
        'is_zero_var': bool(is_zero_var),
        'variance': float(variance),
        'saturation_ratio': float(saturation_ratio),
        'is_valid': not (has_nan or has_inf or is_zero_var or saturation_ratio > 0.3)
    }


def validate_split(split_path: Path, split_name: str,
                  max_windows: Optional[int] = None) -> Dict:
    """Validate all windows in a split (train/val/test).

    Args:
        split_path: Path to split directory
        split_name: Name of split ('train', 'val', or 'test')
        max_windows: Maximum windows to validate (None = all)

    Returns:
        Dictionary with validation results
    """
    results = {
        'split_name': split_name,
        'total_windows': 0,
        'total_cases': 0,
        'valid_windows': 0,
        'correlations': [],
        'peak_lags': [],
        'snr_ppg': [],
        'snr_ecg': [],
        'dominant_freq_ppg': [],
        'dominant_freq_ecg': [],
        'invalid_windows': [],
        'quality_issues': []
    }

    if not split_path.exists():
        print(f"⚠️  Split path does not exist: {split_path}")
        return results

    # Find all case files
    case_files = sorted(list(split_path.glob('case_*.npz')))

    if len(case_files) == 0:
        print(f"⚠️  No case files found in {split_path}")
        return results

    results['total_cases'] = len(case_files)

    print(f"\n📊 Validating {split_name} split ({len(case_files)} cases)...")

    window_count = 0

    for case_file in tqdm(case_files, desc=f"  {split_name}"):
        ppg_windows, ecg_windows = load_window(case_file)

        if ppg_windows is None or ecg_windows is None:
            results['invalid_windows'].append({
                'case': case_file.name,
                'reason': 'Failed to load data'
            })
            continue

        # Process each window
        for i in range(len(ppg_windows)):
            if max_windows and window_count >= max_windows:
                break

            ppg = ppg_windows[i]
            ecg = ecg_windows[i]

            window_count += 1
            results['total_windows'] += 1

            # Check signal quality
            ppg_quality = check_signal_quality(ppg)
            ecg_quality = check_signal_quality(ecg)

            if not ppg_quality['is_valid'] or not ecg_quality['is_valid']:
                results['invalid_windows'].append({
                    'case': case_file.name,
                    'window': i,
                    'ppg_quality': ppg_quality,
                    'ecg_quality': ecg_quality
                })
                continue

            results['valid_windows'] += 1

            # Compute correlation
            corr = compute_correlation(ppg, ecg)
            results['correlations'].append(corr)

            # Compute cross-correlation
            max_corr, peak_lag = compute_cross_correlation(ppg, ecg)
            results['peak_lags'].append(peak_lag)

            # Compute SNR
            snr_ppg = compute_snr(ppg)
            snr_ecg = compute_snr(ecg)
            results['snr_ppg'].append(snr_ppg)
            results['snr_ecg'].append(snr_ecg)

            # Compute dominant frequency
            dom_freq_ppg = compute_dominant_frequency(ppg)
            dom_freq_ecg = compute_dominant_frequency(ecg)
            results['dominant_freq_ppg'].append(dom_freq_ppg)
            results['dominant_freq_ecg'].append(dom_freq_ecg)

            # Flag poor synchronization
            if corr < 0.3:
                results['quality_issues'].append({
                    'case': case_file.name,
                    'window': i,
                    'issue': 'Low correlation',
                    'correlation': corr
                })

        if max_windows and window_count >= max_windows:
            break

    return results


def generate_plots(all_results: Dict, output_dir: Path, data_dir: Path):
    """Generate validation plots.

    Args:
        all_results: Combined results from all splits
        output_dir: Directory to save plots
        data_dir: Directory containing the data (for loading sample windows)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate data from all splits
    all_corrs = []
    all_lags = []
    all_snr_ppg = []
    all_snr_ecg = []
    all_freq_ppg = []
    all_freq_ecg = []

    for split_results in all_results.values():
        all_corrs.extend(split_results.get('correlations', []))
        all_lags.extend(split_results.get('peak_lags', []))
        all_snr_ppg.extend(split_results.get('snr_ppg', []))
        all_snr_ecg.extend(split_results.get('snr_ecg', []))
        all_freq_ppg.extend(split_results.get('dominant_freq_ppg', []))
        all_freq_ecg.extend(split_results.get('dominant_freq_ecg', []))

    if len(all_corrs) == 0:
        print("⚠️  No data to plot")
        return

    # Create figure with 2x2 subplots
    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Correlation histogram
    ax1 = plt.subplot(2, 2, 1)
    ax1.hist(all_corrs, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(np.mean(all_corrs), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(all_corrs):.3f}')
    ax1.axvline(0.4, color='green', linestyle='--',
                linewidth=2, label='Target: 0.40')
    ax1.set_xlabel('Correlation Coefficient', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('PPG-ECG Correlation Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cross-correlation lag distribution
    ax2 = plt.subplot(2, 2, 2)
    ax2.hist(all_lags, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--',
                linewidth=2, label='Perfect Sync (lag=0)')
    ax2.set_xlabel('Peak Lag (samples)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Cross-Correlation Peak Lag', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: SNR comparison
    ax3 = plt.subplot(2, 2, 3)
    ax3.scatter(all_snr_ppg, all_snr_ecg, alpha=0.3, s=10)
    ax3.axhline(10, color='red', linestyle='--', linewidth=1, label='Min SNR: 10 dB')
    ax3.axvline(10, color='red', linestyle='--', linewidth=1)
    ax3.set_xlabel('PPG SNR (dB)', fontsize=12)
    ax3.set_ylabel('ECG SNR (dB)', fontsize=12)
    ax3.set_title('Signal-to-Noise Ratio Comparison', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Dominant frequency comparison
    ax4 = plt.subplot(2, 2, 4)
    ax4.scatter(all_freq_ppg, all_freq_ecg, alpha=0.3, s=10)
    ax4.plot([0.5, 3], [0.5, 3], 'r--', linewidth=2, label='Perfect Agreement')
    ax4.set_xlabel('PPG Dominant Freq (Hz)', fontsize=12)
    ax4.set_ylabel('ECG Dominant Freq (Hz)', fontsize=12)
    ax4.set_title('Dominant Frequency Comparison', fontsize=14, fontweight='bold')
    ax4.set_xlim([0.5, 3.0])
    ax4.set_ylim([0.5, 3.0])
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = output_dir / 'validation_summary.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved summary plot: {plot_file}")

    # Plot 5: Sample window pairs
    plot_sample_windows(data_dir, output_dir)


def plot_sample_windows(data_dir: Path, output_dir: Path, num_samples: int = 5):
    """Plot sample PPG-ECG window pairs.

    Args:
        data_dir: Directory containing the data
        output_dir: Directory to save plots
        num_samples: Number of sample windows to plot
    """
    # Find train split
    train_dir = data_dir / 'train'

    if not train_dir.exists():
        print(f"⚠️  Train directory not found: {train_dir}")
        return

    case_files = list(train_dir.glob('case_*.npz'))

    if len(case_files) == 0:
        print("⚠️  No case files found for sample plots")
        return

    # Load random samples
    np.random.seed(42)
    sample_file = np.random.choice(case_files)

    ppg_windows, ecg_windows = load_window(sample_file)

    if ppg_windows is None or ecg_windows is None:
        print("⚠️  Failed to load sample data")
        return

    # Plot first N windows
    fig, axes = plt.subplots(num_samples, 1, figsize=(14, 2.5*num_samples))

    if num_samples == 1:
        axes = [axes]

    fs = 125  # Sampling rate
    time = np.arange(1024) / fs

    for i in range(min(num_samples, len(ppg_windows))):
        ppg = ppg_windows[i]
        ecg = ecg_windows[i]

        corr = compute_correlation(ppg, ecg)

        ax = axes[i]
        ax.plot(time, ppg, 'b-', linewidth=1, alpha=0.8, label='PPG')
        ax_twin = ax.twinx()
        ax_twin.plot(time, ecg, 'r-', linewidth=1, alpha=0.8, label='ECG')

        ax.set_ylabel('PPG', color='b', fontsize=10)
        ax_twin.set_ylabel('ECG', color='r', fontsize=10)
        ax.set_xlabel('Time (seconds)', fontsize=10)
        ax.set_title(f'Window {i+1} - Correlation: {corr:.3f}', fontsize=11)
        ax.grid(True, alpha=0.3)

        # Add legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    plot_file = output_dir / 'sample_windows.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved sample windows: {plot_file}")


def print_validation_report(all_results: Dict, output_dir: Path):
    """Print comprehensive validation report.

    Args:
        all_results: Combined results from all splits
        output_dir: Directory where plots are saved
    """
    print("\n" + "="*80)
    print("VALIDATION REPORT: VitalDB Synchronized Data")
    print("="*80)

    # Dataset statistics
    print("\nDATASET STATISTICS:")

    total_windows = 0
    total_cases = 0

    for split_name in ['train', 'val', 'test']:
        if split_name in all_results:
            split_results = all_results[split_name]
            n_windows = split_results['total_windows']
            n_cases = split_results['total_cases']
            total_windows += n_windows
            total_cases += n_cases
            print(f"  {split_name.capitalize():6s}: {n_windows:6,} windows from {n_cases:3d} cases")

    print(f"  {'Total':6s}: {total_windows:6,} windows from {total_cases:3d} cases")

    # Compute split percentages
    if total_windows > 0:
        train_pct = (all_results.get('train', {}).get('total_windows', 0) / total_windows) * 100
        val_pct = (all_results.get('val', {}).get('total_windows', 0) / total_windows) * 100
        test_pct = (all_results.get('test', {}).get('total_windows', 0) / total_windows) * 100

        split_ok = (65 <= train_pct <= 75) and (10 <= val_pct <= 20) and (10 <= test_pct <= 20)
        status = "✅" if split_ok else "⚠️"

        print(f"  Split:  {train_pct:.1f}% / {val_pct:.1f}% / {test_pct:.1f}% {status}")

    # Aggregate all metrics
    all_corrs = []
    all_lags = []
    all_snr_ppg = []
    all_snr_ecg = []
    all_freq_ppg = []
    all_freq_ecg = []

    for split_results in all_results.values():
        all_corrs.extend(split_results.get('correlations', []))
        all_lags.extend(split_results.get('peak_lags', []))
        all_snr_ppg.extend(split_results.get('snr_ppg', []))
        all_snr_ecg.extend(split_results.get('snr_ecg', []))
        all_freq_ppg.extend(split_results.get('dominant_freq_ppg', []))
        all_freq_ecg.extend(split_results.get('dominant_freq_ecg', []))

    if len(all_corrs) == 0:
        print("\n❌ No valid windows found!")
        return False

    # Synchronization quality
    print("\nSYNCHRONIZATION QUALITY:")

    mean_corr = np.mean(all_corrs)
    std_corr = np.std(all_corrs)
    median_corr = np.median(all_corrs)
    min_corr = np.min(all_corrs)
    max_corr = np.max(all_corrs)

    pct_good = (np.sum(np.array(all_corrs) > 0.4) / len(all_corrs)) * 100
    pct_acceptable = (np.sum(np.array(all_corrs) > 0.3) / len(all_corrs)) * 100
    pct_zero_lag = (np.sum(np.array(all_lags) == 0) / len(all_lags)) * 100

    corr_ok = mean_corr >= 0.40
    pct_ok = pct_acceptable >= 80.0
    lag_ok = pct_zero_lag >= 90.0

    print(f"  Mean correlation:        {mean_corr:.2f} ± {std_corr:.2f} {'✅' if corr_ok else '❌'}")
    print(f"  Median correlation:      {median_corr:.2f}")
    print(f"  Min/Max correlation:     {min_corr:.2f} / {max_corr:.2f}")
    print(f"  Windows with corr>0.4:   {pct_good:.1f}%")
    print(f"  Windows with corr>0.3:   {pct_acceptable:.1f}% {'✅' if pct_ok else '❌'}")
    print(f"  Peak lag = 0:            {pct_zero_lag:.1f}% {'✅' if lag_ok else '❌'}")

    # Signal quality
    print("\nSIGNAL QUALITY:")

    mean_snr_ppg = np.mean(all_snr_ppg)
    mean_snr_ecg = np.mean(all_snr_ecg)

    snr_ok = mean_snr_ppg > 10 and mean_snr_ecg > 10

    total_invalid = sum(len(r.get('invalid_windows', [])) for r in all_results.values())

    print(f"  Mean SNR (PPG):          {mean_snr_ppg:.1f} dB {'✅' if mean_snr_ppg > 10 else '❌'}")
    print(f"  Mean SNR (ECG):          {mean_snr_ecg:.1f} dB {'✅' if mean_snr_ecg > 10 else '❌'}")
    print(f"  Invalid windows:         {total_invalid} {'✅' if total_invalid == 0 else '⚠️'}")

    # Frequency content
    print("\nFREQUENCY CONTENT:")

    mean_freq_ppg = np.mean(all_freq_ppg)
    std_freq_ppg = np.std(all_freq_ppg)
    mean_freq_ecg = np.mean(all_freq_ecg)
    std_freq_ecg = np.std(all_freq_ecg)

    freq_ok = (1.0 <= mean_freq_ppg <= 2.5) and (1.0 <= mean_freq_ecg <= 2.5)

    print(f"  PPG dominant freq:       {mean_freq_ppg:.2f} ± {std_freq_ppg:.2f} Hz {'✅' if freq_ok else '⚠️'}")
    print(f"  ECG dominant freq:       {mean_freq_ecg:.2f} ± {std_freq_ecg:.2f} Hz {'✅' if freq_ok else '⚠️'}")

    # Final verdict
    print("\n" + "="*80)

    all_pass = corr_ok and pct_ok and lag_ok and snr_ok and freq_ok

    if all_pass:
        print("VALIDATION RESULT: ✅ PASS")
        print("\nAll criteria met. Data is properly synchronized and ready for SSL pretraining.")
    else:
        print("VALIDATION RESULT: ❌ FAIL")
        print("\nSome criteria not met. Review the metrics above.")

        if not corr_ok:
            print("  - Mean correlation < 0.40 (poor synchronization)")
        if not pct_ok:
            print("  - <80% windows with acceptable correlation")
        if not lag_ok:
            print("  - <90% windows have peak at lag=0")
        if not snr_ok:
            print("  - SNR below 10 dB threshold")
        if not freq_ok:
            print("  - Dominant frequency outside expected range")

    print(f"\nPlots saved to: {output_dir}")
    print("="*80 + "\n")

    return all_pass


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(
        description="Validate synchronized PPG-ECG data quality",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Path to processed data directory (e.g., data/processed/vitaldb/paired)'
    )

    parser.add_argument(
        '--plots-dir',
        type=str,
        default=None,
        help='Directory to save validation plots (default: <data-dir>/validation_plots)'
    )

    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'val', 'test', 'all'],
        default='all',
        help='Which split(s) to validate'
    )

    parser.add_argument(
        '--max-windows',
        type=int,
        default=None,
        help='Maximum windows to validate per split (for quick checks)'
    )

    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Save detailed results to JSON file'
    )

    args = parser.parse_args()

    # Setup paths
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("   Please run prepare_all_data.py first")
        return 1

    if args.plots_dir:
        plots_dir = Path(args.plots_dir)
    else:
        plots_dir = data_dir / 'validation_plots'

    plots_dir.mkdir(parents=True, exist_ok=True)

    # Determine which splits to validate
    if args.split == 'all':
        splits_to_validate = ['train', 'val', 'test']
    else:
        splits_to_validate = [args.split]

    # Validate each split
    all_results = {}

    for split_name in splits_to_validate:
        split_path = data_dir / split_name

        if not split_path.exists():
            print(f"⚠️  Split not found: {split_path}")
            continue

        results = validate_split(split_path, split_name, args.max_windows)
        all_results[split_name] = results

    if len(all_results) == 0:
        print("❌ No splits validated!")
        return 1

    # Generate plots
    print("\n📊 Generating validation plots...")
    generate_plots(all_results, plots_dir, data_dir)

    # Print report
    passed = print_validation_report(all_results, plots_dir)

    # Save detailed results to JSON
    if args.output_json:
        output_file = Path(args.output_json)
    else:
        output_file = data_dir / 'validation_results.json'

    # Convert numpy types to Python types for JSON
    json_results = {}
    for split_name, results in all_results.items():
        json_results[split_name] = {
            'split_name': results['split_name'],
            'total_windows': results['total_windows'],
            'total_cases': results['total_cases'],
            'valid_windows': results['valid_windows'],
            'mean_correlation': float(np.mean(results['correlations'])) if results['correlations'] else 0.0,
            'std_correlation': float(np.std(results['correlations'])) if results['correlations'] else 0.0,
            'mean_snr_ppg': float(np.mean(results['snr_ppg'])) if results['snr_ppg'] else 0.0,
            'mean_snr_ecg': float(np.mean(results['snr_ecg'])) if results['snr_ecg'] else 0.0,
            'num_invalid': len(results['invalid_windows']),
            'num_quality_issues': len(results['quality_issues'])
        }

    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"Detailed results saved to: {output_file}")

    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
