#!/usr/bin/env python3
"""
Inspect raw VitalDB data quality to diagnose synchronization issues.

This script checks:
1. Available signal tracks for each case
2. Signal quality (SNR, duration, sampling rate)
3. Which PPG/ECG sources have best quality
4. Raw correlation between signals before processing

Usage:
    python scripts/inspect_vitaldb_raw.py --cases 1,2,3,4,5
    python scripts/inspect_vitaldb_raw.py --num-cases 20
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.signal import welch
import ssl
import os

# Configure SSL for VitalDB
try:
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        import certifi
        os.environ['SSL_CERT_FILE'] = certifi.where()
        os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
    except ImportError:
        pass
except:
    pass

try:
    import vitaldb
    print("✓ VitalDB loaded successfully")
except ImportError:
    print("❌ VitalDB not available. Install with: pip install vitaldb")
    sys.exit(1)


def compute_snr_simple(signal_data, fs=125):
    """Compute SNR using power spectrum.

    Args:
        signal_data: Input signal
        fs: Sampling rate

    Returns:
        SNR in dB
    """
    try:
        # Remove NaN
        signal_clean = signal_data[~np.isnan(signal_data)]

        if len(signal_clean) < 100:
            return -999

        # Compute PSD
        freqs, psd = welch(signal_clean, fs=fs, nperseg=min(256, len(signal_clean)//4))

        # Signal power (0.5-3 Hz for cardiac)
        signal_mask = (freqs >= 0.5) & (freqs <= 3.0)

        # Noise power (high freq, 20-50 Hz)
        noise_mask = (freqs >= 20.0) & (freqs <= min(50.0, fs/2 - 1))

        if np.sum(signal_mask) == 0 or np.sum(noise_mask) == 0:
            return -999

        signal_power = np.mean(psd[signal_mask])
        noise_power = np.mean(psd[noise_mask])

        if noise_power <= 0:
            return 100

        snr_db = 10 * np.log10(signal_power / noise_power)
        return snr_db

    except:
        return -999


def load_and_inspect_signal(case_id, track_name):
    """Load a signal and compute quality metrics.

    Args:
        case_id: VitalDB case ID
        track_name: Track name to load

    Returns:
        Dictionary with quality metrics, or None if failed
    """
    try:
        # Determine expected sampling rate based on track type
        # CRITICAL: Use interval parameter to load at correct sampling rate
        if 'ECG' in track_name:
            # ECG tracks are typically 500 Hz
            expected_fs = 500.0
        elif 'PLETH' in track_name or 'ART' in track_name:
            # PPG/Pleth and arterial pressure are typically 100 Hz
            expected_fs = 100.0
        else:
            # Default for other tracks
            expected_fs = 100.0

        # Load with correct interval (interval = 1 / sampling_rate)
        interval = 1.0 / expected_fs
        data = vitaldb.load_case(case_id, [track_name], interval=interval)

        if data is None or len(data) == 0:
            return None

        # Extract signal
        if isinstance(data, pd.DataFrame):
            if track_name in data.columns:
                signal_data = data[track_name].values
            else:
                signal_data = data.iloc[:, 0].values
        else:
            signal_data = np.array(data)

        if len(signal_data) == 0:
            return None

        # Compute metrics
        signal_clean = signal_data[~np.isnan(signal_data)]

        if len(signal_clean) == 0:
            return None

        # Calculate duration based on actual sampling rate used
        duration_sec = len(signal_data) / expected_fs
        duration_min = duration_sec / 60.0
        est_fs = expected_fs

        metrics = {
            'track': track_name,
            'available': True,
            'length': len(signal_data),
            'duration_min': duration_min,
            'est_fs': est_fs,
            'nan_ratio': np.mean(np.isnan(signal_data)),
            'mean': float(np.mean(signal_clean)),
            'std': float(np.std(signal_clean)),
            'min': float(np.min(signal_clean)),
            'max': float(np.max(signal_clean)),
            'range': float(np.max(signal_clean) - np.min(signal_clean)),
            'snr_db': compute_snr_simple(signal_clean, est_fs)
        }

        return metrics

    except Exception as e:
        return None


def inspect_case(case_id):
    """Inspect all relevant tracks for a single case.

    Args:
        case_id: VitalDB case ID

    Returns:
        Dictionary with inspection results
    """
    print(f"\n{'='*80}")
    print(f"🔬 Inspecting Case {case_id}")
    print(f"{'='*80}")

    # Define tracks to check
    ppg_tracks = [
        'SNUADC/PLETH',      # Pulse oximetry plethysmograph
        'Solar8000/PLETH',   # Solar monitor pleth
        'Intellivue/PLETH',  # Philips Intellivue pleth
        'SNUADC/ART',        # Arterial pressure (NOT pleth, but cardiac)
        'Solar8000/ART'      # Solar arterial pressure
    ]

    ecg_tracks = [
        'SNUADC/ECG_II',     # Lead II ECG
        'Solar8000/ECG_II',  # Solar ECG
        'SNUADC/ECG_V5',     # Lead V5 ECG
        'SNUADC/ECG'         # Generic ECG
    ]

    results = {
        'case_id': case_id,
        'ppg_tracks': {},
        'ecg_tracks': {},
        'best_ppg': None,
        'best_ecg': None
    }

    # Check PPG tracks
    print("\n📊 PPG/Pleth Tracks:")
    best_ppg_snr = -9999

    for track in ppg_tracks:
        metrics = load_and_inspect_signal(case_id, track)

        if metrics is not None:
            results['ppg_tracks'][track] = metrics

            print(f"  ✓ {track:25s}: "
                  f"duration={metrics['duration_min']:6.1f}min, "
                  f"SNR={metrics['snr_db']:6.1f}dB, "
                  f"range=[{metrics['min']:7.1f}, {metrics['max']:7.1f}], "
                  f"NaN={metrics['nan_ratio']*100:4.1f}%")

            if metrics['snr_db'] > best_ppg_snr:
                best_ppg_snr = metrics['snr_db']
                results['best_ppg'] = track
        else:
            print(f"  ✗ {track:25s}: Not available")

    # Check ECG tracks
    print("\n📊 ECG Tracks:")
    best_ecg_snr = -9999

    for track in ecg_tracks:
        metrics = load_and_inspect_signal(case_id, track)

        if metrics is not None:
            results['ecg_tracks'][track] = metrics

            print(f"  ✓ {track:25s}: "
                  f"duration={metrics['duration_min']:6.1f}min, "
                  f"SNR={metrics['snr_db']:6.1f}dB, "
                  f"range=[{metrics['min']:7.1f}, {metrics['max']:7.1f}], "
                  f"NaN={metrics['nan_ratio']*100:4.1f}%")

            if metrics['snr_db'] > best_ecg_snr:
                best_ecg_snr = metrics['snr_db']
                results['best_ecg'] = track
        else:
            print(f"  ✗ {track:25s}: Not available")

    # Print recommendations
    print(f"\n💡 Recommendations:")

    if results['best_ppg']:
        ppg_metrics = results['ppg_tracks'][results['best_ppg']]
        print(f"  Best PPG: {results['best_ppg']} (SNR: {ppg_metrics['snr_db']:.1f} dB)")
    else:
        print(f"  ⚠️  No suitable PPG track found")

    if results['best_ecg']:
        ecg_metrics = results['ecg_tracks'][results['best_ecg']]
        print(f"  Best ECG: {results['best_ecg']} (SNR: {ecg_metrics['snr_db']:.1f} dB)")
    else:
        print(f"  ⚠️  No suitable ECG track found")

    # Check if both available
    if results['best_ppg'] and results['best_ecg']:
        ppg_metrics = results['ppg_tracks'][results['best_ppg']]
        ecg_metrics = results['ecg_tracks'][results['best_ecg']]

        # Check duration overlap
        min_duration = min(ppg_metrics['duration_min'], ecg_metrics['duration_min'])

        print(f"\n  Paired data available: {min_duration:.1f} minutes")

        if ppg_metrics['snr_db'] > 5 and ecg_metrics['snr_db'] > 5:
            print(f"  ✅ Case {case_id} is suitable for SSL pretraining")
        else:
            print(f"  ⚠️  Case {case_id} has low SNR - may need filtering")
    else:
        print(f"  ❌ Case {case_id} does not have both PPG and ECG")

    return results


def compare_signal_sources(all_results):
    """Compare different signal sources across all cases.

    Args:
        all_results: List of inspection results from all cases
    """
    print(f"\n{'='*80}")
    print("📊 SIGNAL SOURCE COMPARISON")
    print(f"{'='*80}")

    # Count availability
    ppg_availability = {}
    ecg_availability = {}
    ppg_snr_avg = {}
    ecg_snr_avg = {}

    for result in all_results:
        for track, metrics in result['ppg_tracks'].items():
            ppg_availability[track] = ppg_availability.get(track, 0) + 1

            if metrics['snr_db'] > -900:
                if track not in ppg_snr_avg:
                    ppg_snr_avg[track] = []
                ppg_snr_avg[track].append(metrics['snr_db'])

        for track, metrics in result['ecg_tracks'].items():
            ecg_availability[track] = ecg_availability.get(track, 0) + 1

            if metrics['snr_db'] > -900:
                if track not in ecg_snr_avg:
                    ecg_snr_avg[track] = []
                ecg_snr_avg[track].append(metrics['snr_db'])

    # Print PPG summary
    print(f"\nPPG/Pleth Sources:")
    print(f"{'Track':<30s} {'Availability':>15s} {'Avg SNR (dB)':>15s}")
    print(f"{'-'*60}")

    for track in sorted(ppg_availability.keys(), key=lambda x: ppg_availability[x], reverse=True):
        avail = f"{ppg_availability[track]}/{len(all_results)}"

        if track in ppg_snr_avg and len(ppg_snr_avg[track]) > 0:
            avg_snr = f"{np.mean(ppg_snr_avg[track]):.1f}"
        else:
            avg_snr = "N/A"

        print(f"{track:<30s} {avail:>15s} {avg_snr:>15s}")

    # Print ECG summary
    print(f"\nECG Sources:")
    print(f"{'Track':<30s} {'Availability':>15s} {'Avg SNR (dB)':>15s}")
    print(f"{'-'*60}")

    for track in sorted(ecg_availability.keys(), key=lambda x: ecg_availability[x], reverse=True):
        avail = f"{ecg_availability[track]}/{len(all_results)}"

        if track in ecg_snr_avg and len(ecg_snr_avg[track]) > 0:
            avg_snr = f"{np.mean(ecg_snr_avg[track]):.1f}"
        else:
            avg_snr = "N/A"

        print(f"{track:<30s} {avail:>15s} {avg_snr:>15s}")

    # Recommendations
    print(f"\n{'='*80}")
    print("💡 RECOMMENDATIONS")
    print(f"{'='*80}")

    # Find most common good quality sources
    if ppg_snr_avg:
        best_ppg = max(ppg_snr_avg.keys(), key=lambda x: np.mean(ppg_snr_avg[x]))
        print(f"\nBest PPG source: {best_ppg}")
        print(f"  Average SNR: {np.mean(ppg_snr_avg[best_ppg]):.1f} dB")
        print(f"  Available in: {ppg_availability[best_ppg]}/{len(all_results)} cases")

    if ecg_snr_avg:
        best_ecg = max(ecg_snr_avg.keys(), key=lambda x: np.mean(ecg_snr_avg[x]))
        print(f"\nBest ECG source: {best_ecg}")
        print(f"  Average SNR: {np.mean(ecg_snr_avg[best_ecg]):.1f} dB")
        print(f"  Available in: {ecg_availability[best_ecg]}/{len(all_results)} cases")

    # Count usable cases
    usable_cases = sum(1 for r in all_results if r['best_ppg'] and r['best_ecg'])

    print(f"\n{'='*80}")
    print(f"Usable cases (have both PPG and ECG): {usable_cases}/{len(all_results)} ({usable_cases/len(all_results)*100:.1f}%)")
    print(f"{'='*80}")


def main():
    """Main inspection function."""
    parser = argparse.ArgumentParser(
        description="Inspect raw VitalDB data quality",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--cases',
        type=str,
        default=None,
        help='Comma-separated list of case IDs to inspect (e.g., "1,2,3,4,5")'
    )

    parser.add_argument(
        '--num-cases',
        type=int,
        default=10,
        help='Number of cases to inspect (default: 10)'
    )

    parser.add_argument(
        '--start-case',
        type=int,
        default=1,
        help='Starting case ID (default: 1)'
    )

    args = parser.parse_args()

    # Determine which cases to inspect
    if args.cases:
        case_ids = [int(c.strip()) for c in args.cases.split(',')]
    else:
        case_ids = list(range(args.start_case, args.start_case + args.num_cases))

    print(f"{'='*80}")
    print(f"🔬 VitalDB Raw Data Quality Inspection")
    print(f"{'='*80}")
    print(f"Inspecting {len(case_ids)} cases: {case_ids}")

    # Inspect each case
    all_results = []

    for case_id in case_ids:
        try:
            result = inspect_case(case_id)
            all_results.append(result)
        except Exception as e:
            print(f"\n❌ Error inspecting case {case_id}: {e}")
            import traceback
            traceback.print_exc()

    # Compare sources
    if len(all_results) > 0:
        compare_signal_sources(all_results)
    else:
        print("\n❌ No cases successfully inspected!")

    print(f"\n{'='*80}")
    print("✅ Inspection complete!")
    print(f"{'='*80}\n")


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
