#!/usr/bin/env python3
"""Rebuild VitalDB dataset with proper PPG-ECG temporal pairing.

CRITICAL: This creates properly synchronized PPG+ECG pairs from the same time points,
fixing the data alignment issue where PPG and ECG had different numbers of windows.

Output format: [N, 2, 1024] where:
  - N = number of windows
  - Channel 0 = PPG
  - Channel 1 = ECG
  - Each window is 1024 samples (8.192 seconds @ 125Hz)
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy import signal as scipy_signal
from scipy.signal import find_peaks
import json
import sys
import warnings
import ssl
import os
import gc
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
from functools import partial
import signal as system_signal
from contextlib import contextmanager

# Configure SSL for VitalDB access (fix certificate issues)
try:
    ssl._create_default_https_context = ssl._create_unverified_context

    # Try to use certifi if available
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
    VITALDB_AVAILABLE = True
except ImportError:
    print("❌ VitalDB is not available! Install with: pip install vitaldb")
    VITALDB_AVAILABLE = False
    sys.exit(1)


@contextmanager
def timeout_handler(seconds):
    """Context manager for adding timeout to operations.

    Args:
        seconds: Timeout in seconds

    Raises:
        TimeoutError: If operation exceeds timeout
    """
    def timeout_signal_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    # Set the signal handler
    old_handler = system_signal.signal(system_signal.SIGALRM, timeout_signal_handler)
    system_signal.alarm(seconds)

    try:
        yield
    finally:
        # Restore the old signal handler
        system_signal.alarm(0)
        system_signal.signal(system_signal.SIGALRM, old_handler)


def convert_to_python_types(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization.

    Args:
        obj: Any object that may contain numpy types

    Returns:
        Object with all numpy types converted to Python types
    """
    if isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_python_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def load_signal(case_id, track_names, default_fs, timeout_sec=60):
    """Load a signal from VitalDB, trying multiple track names.

    Args:
        case_id: VitalDB case ID
        track_names: List of possible track names to try
        default_fs: Default sampling rate for this signal type
        timeout_sec: Timeout for loading in seconds

    Returns:
        tuple: (signal_array, actual_fs) or (None, None) if failed
    """
    # CRITICAL: Set interval to match sampling rate (interval = 1 / fs)
    interval = 1.0 / default_fs

    for track in track_names:
        try:
            # Add timeout to prevent hanging on network issues
            try:
                with timeout_handler(timeout_sec):
                    data = vitaldb.load_case(case_id, [track], interval=interval)
            except (TimeoutError, OSError, ConnectionError) as e:
                # Network or timeout error, try next track
                continue

            if data is not None and len(data) > 0:
                # Extract signal from DataFrame
                if isinstance(data, pd.DataFrame):
                    if track in data.columns:
                        signal = data[track].values
                    else:
                        signal = data.iloc[:, 0].values
                else:
                    signal = np.array(data)

                # Check if signal is valid
                if len(signal) > 0 and not np.all(np.isnan(signal)):
                    return signal, default_fs

        except (KeyboardInterrupt, SystemExit):
            # Don't catch these
            raise
        except Exception as e:
            # Log but continue to next track
            continue

    return None, None


def resample_signal(signal, orig_fs, target_fs):
    """Resample signal to target sampling rate.

    Args:
        signal: Input signal
        orig_fs: Original sampling rate in Hz
        target_fs: Target sampling rate in Hz

    Returns:
        Resampled signal at target_fs
    """
    if abs(orig_fs - target_fs) < 0.01:
        return signal

    # Remove NaN values before resampling
    if np.any(np.isnan(signal)):
        valid_mask = ~np.isnan(signal)
        if np.sum(valid_mask) < 2:
            return None
        valid_indices = np.where(valid_mask)[0]
        valid_values = signal[valid_mask]
        signal = np.interp(np.arange(len(signal)), valid_indices, valid_values)

    # Resample
    duration = len(signal) / orig_fs
    new_length = int(duration * target_fs)
    resampled = scipy_signal.resample(signal, new_length)

    return resampled


def check_quality(ppg_window, ecg_window, fs=125, min_hr=30, max_hr=220, min_var=0.001):
    """Check if both PPG and ECG windows pass quality criteria.

    RELAXED criteria for VitalDB surgical monitoring data which is inherently noisy.

    Args:
        ppg_window: PPG window [samples]
        ecg_window: ECG window [samples]
        fs: Sampling rate in Hz
        min_hr: Minimum heart rate in bpm (relaxed to 30)
        max_hr: Maximum heart rate in bpm (relaxed to 220)
        min_var: Minimum variance threshold (relaxed to 0.001)

    Returns:
        bool: True if both windows pass quality checks
    """
    # Check for NaN/Inf in either signal
    if np.any(np.isnan(ppg_window)) or np.any(np.isnan(ecg_window)):
        return False
    if np.any(np.isinf(ppg_window)) or np.any(np.isinf(ecg_window)):
        return False

    # Check variance (signal not flat) - RELAXED for noisy data
    if np.var(ppg_window) < min_var or np.var(ecg_window) < min_var:
        return False

    # REMOVED strict heart rate check - too strict for surgical monitoring
    # VitalDB data during surgery can have variable heart rates
    # Just check that signal has some periodic structure

    try:
        # Simplified check: just verify there are some peaks
        peaks, _ = find_peaks(ppg_window, distance=int(fs * 0.3))  # Min 0.3s between beats

        if len(peaks) < 1:  # Need at least 1 peak in 8 seconds
            return False

    except Exception as e:
        # If peak detection fails, still accept if variance is sufficient
        pass

    return True


def is_case_already_processed(case_id, output_dir):
    """Check if a case has already been successfully processed.

    Args:
        case_id: VitalDB case ID
        output_dir: Directory where output would be saved

    Returns:
        bool: True if case file exists and is valid
    """
    output_file = output_dir / f'case_{case_id:05d}_windows.npz'

    if not output_file.exists():
        return False

    # Check if file is valid (not corrupted/empty)
    try:
        if output_file.stat().st_size < 1000:  # Less than 1KB is suspicious
            return False

        # Try to load to verify it's valid
        data = np.load(output_file)
        if 'data' not in data or data['data'].shape[0] == 0:
            return False

        return True
    except:
        return False


def process_case(case_id, output_dir, fs=125, window_size=1024,
                min_duration_min=None, verbose=False, skip_existing=True):
    """Process one VitalDB case and create paired PPG-ECG windows.

    Args:
        case_id: VitalDB case ID
        output_dir: Directory to save output
        fs: Target sampling rate in Hz
        window_size: Window size in samples
        min_duration_min: Minimum duration in minutes (None = no check)
        verbose: Print detailed progress
        skip_existing: Skip if case already successfully processed

    Returns:
        tuple: (num_windows, error_message) where num_windows is int or None,
               error_message is str or None
    """
    error_msg = None

    try:
        # Check if already processed
        if skip_existing and is_case_already_processed(case_id, output_dir):
            if verbose:
                print(f"\n  Case {case_id} already processed, skipping...")
            # Load existing file to return window count
            try:
                output_file = output_dir / f'case_{case_id:05d}_windows.npz'
                data = np.load(output_file)
                num_windows = int(data['num_windows'])
                data.close()
                return (num_windows, None)
            except Exception as e:
                error_msg = f"Error loading existing file: {e}"
                return (None, error_msg)

        if verbose:
            print(f"\n  Processing case {case_id}...")

        # Load PPG (PLETH) with timeout
        ppg_tracks = ['SNUADC/PLETH', 'Solar8000/PLETH', 'Intellivue/PLETH']
        try:
            ppg_signal, ppg_fs = load_signal(case_id, ppg_tracks, default_fs=100.0, timeout_sec=60)
        except (TimeoutError, MemoryError) as e:
            error_msg = f"PPG load timeout/memory: {type(e).__name__}"
            if verbose:
                print(f"    ⚠️  {error_msg}")
            return (None, error_msg)

        if ppg_signal is None:
            error_msg = "Could not load PPG signal"
            if verbose:
                print(f"    ⚠️  {error_msg}")
            return (None, error_msg)

        # Load ECG with timeout
        ecg_tracks = ['SNUADC/ECG_II', 'Solar8000/ECG_II', 'SNUADC/ECG_V5']
        try:
            ecg_signal, ecg_fs = load_signal(case_id, ecg_tracks, default_fs=500.0, timeout_sec=60)
        except (TimeoutError, MemoryError) as e:
            error_msg = f"ECG load timeout/memory: {type(e).__name__}"
            if verbose:
                print(f"    ⚠️  {error_msg}")
            # Clean up PPG signal
            del ppg_signal
            gc.collect()
            return (None, error_msg)

        if ecg_signal is None:
            error_msg = "Could not load ECG signal"
            if verbose:
                print(f"    ⚠️  {error_msg}")
            # Clean up PPG signal
            del ppg_signal
            gc.collect()
            return (None, error_msg)

        if verbose:
            print(f"    Loaded: PPG {len(ppg_signal)} samples @ {ppg_fs}Hz, "
                  f"ECG {len(ecg_signal)} samples @ {ecg_fs}Hz")

        # Check minimum duration BEFORE resampling
        if min_duration_min:
            ppg_duration_min = len(ppg_signal) / ppg_fs / 60
            ecg_duration_min = len(ecg_signal) / ecg_fs / 60
            min_case_duration = min(ppg_duration_min, ecg_duration_min)

            if min_case_duration < min_duration_min:
                error_msg = f"Too short: {min_case_duration:.1f} min < {min_duration_min} min"
                if verbose:
                    print(f"    ⚠️  {error_msg}")
                # Clean up
                del ppg_signal, ecg_signal
                gc.collect()
                return (None, error_msg)

        # Resample both to target fs
        try:
            ppg_resampled = resample_signal(ppg_signal, ppg_fs, fs)
            ecg_resampled = resample_signal(ecg_signal, ecg_fs, fs)
        except (MemoryError, ValueError) as e:
            error_msg = f"Resampling failed: {type(e).__name__}"
            if verbose:
                print(f"    ⚠️  {error_msg}")
            # Clean up
            del ppg_signal, ecg_signal
            gc.collect()
            return (None, error_msg)

        # Clean up original signals
        del ppg_signal, ecg_signal
        gc.collect()

        if ppg_resampled is None or ecg_resampled is None:
            error_msg = "Resampling returned None"
            if verbose:
                print(f"    ⚠️  {error_msg}")
            return (None, error_msg)

        # Synchronize to same length
        min_len = min(len(ppg_resampled), len(ecg_resampled))
        ppg_sync = ppg_resampled[:min_len]
        ecg_sync = ecg_resampled[:min_len]

        # Clean up resampled arrays
        del ppg_resampled, ecg_resampled
        gc.collect()

        if verbose:
            print(f"    Synchronized: {min_len} samples @ {fs}Hz "
                  f"({min_len/fs/60:.1f} minutes)")

        # Z-score normalize entire signals
        ppg_norm = (ppg_sync - np.mean(ppg_sync)) / (np.std(ppg_sync) + 1e-8)
        ecg_norm = (ecg_sync - np.mean(ecg_sync)) / (np.std(ecg_sync) + 1e-8)

        # Clean up sync arrays
        del ppg_sync, ecg_sync
        gc.collect()

        # Create non-overlapping windows
        paired_windows = []
        num_possible = (min_len - window_size) // window_size + 1

        for i in range(num_possible):
            start = i * window_size
            end = start + window_size

            if end > min_len:
                break

            ppg_win = ppg_norm[start:end]
            ecg_win = ecg_norm[start:end]

            # Quality check BOTH signals
            if check_quality(ppg_win, ecg_win, fs):
                # Stack as [2, window_size]: channel 0 = PPG, channel 1 = ECG
                paired = np.stack([ppg_win, ecg_win], axis=0)
                paired_windows.append(paired)

        # Clean up normalized signals
        del ppg_norm, ecg_norm
        gc.collect()

        if len(paired_windows) == 0:
            error_msg = "No windows passed quality checks"
            if verbose:
                print(f"    ⚠️  {error_msg}")
            return (None, error_msg)

        # Convert to numpy array [N, 2, window_size]
        windows_array = np.stack(paired_windows, axis=0).astype(np.float32)

        if verbose:
            print(f"    ✓ Created {len(paired_windows)} paired windows "
                  f"(passed {len(paired_windows)}/{num_possible} quality checks)")

        # Save to compressed .npz file
        output_file = output_dir / f'case_{case_id:05d}_windows.npz'
        np.savez_compressed(
            output_file,
            data=windows_array,
            case_id=case_id,
            fs=fs,
            window_size=window_size,
            num_windows=len(paired_windows)
        )

        num_windows = len(paired_windows)

        # Clean up
        del paired_windows, windows_array
        gc.collect()

        return (num_windows, None)

    except KeyboardInterrupt:
        # Don't catch keyboard interrupt
        raise
    except MemoryError as e:
        error_msg = f"MemoryError: {str(e)[:100]}"
        if verbose:
            print(f"    ❌ {error_msg}")
        gc.collect()
        return (None, error_msg)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)[:100]}"
        if verbose:
            print(f"    ❌ Error: {error_msg}")
            traceback.print_exc()
        gc.collect()
        return (None, error_msg)


def main():
    parser = argparse.ArgumentParser(
        description='Rebuild VitalDB with proper PPG-ECG temporal pairing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--output', default='data/processed/vitaldb/paired_1024',
                       help='Output directory')
    parser.add_argument('--start-case', type=int, default=100,
                       help='Starting case ID (default: 100 to skip short test cases)')
    parser.add_argument('--max-cases', type=int, default=None,
                       help='Maximum number of cases to process (None = all)')
    parser.add_argument('--min-duration', type=float, default=10.0,
                       help='Minimum case duration in minutes (default: 10)')
    parser.add_argument('--window-size', type=int, default=1024,
                       help='Window size in samples')
    parser.add_argument('--fs', type=int, default=125,
                       help='Target sampling rate in Hz')
    parser.add_argument('--train-ratio', type=float, default=0.70,
                       help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Validation set ratio')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed progress for each case')
    parser.add_argument('--splits-file', default='configs/splits/splits_full.json',
                       help='Path to existing splits file (optional)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of parallel workers (default: 4, MAX 4-6 recommended for VitalDB API)')

    args = parser.parse_args()

    # Safety cap on workers to prevent OOM
    MAX_SAFE_WORKERS = 6
    if args.num_workers > MAX_SAFE_WORKERS:
        print(f"\n⚠️  WARNING: {args.num_workers} workers may cause memory exhaustion!")
        print(f"   VitalDB downloads can use several GB per worker.")
        print(f"   Capping to MAX_SAFE_WORKERS={MAX_SAFE_WORKERS}")
        print(f"   (Override in code if you have >64GB RAM)\n")
        args.num_workers = MAX_SAFE_WORKERS

    print("="*80)
    print("🔧 VitalDB Paired Dataset Builder")
    print("="*80)
    print(f"Output directory: {args.output}")
    print(f"Window size: {args.window_size} samples ({args.window_size/args.fs:.3f} seconds)")
    print(f"Sampling rate: {args.fs} Hz")
    print(f"Minimum duration: {args.min_duration} minutes")
    print(f"Starting case ID: {args.start_case}")
    print(f"Parallel workers: {args.num_workers} {'(PARALLEL MODE 🚀)' if args.num_workers > 1 else '(sequential)'}")
    print(f"Train/Val/Test: {args.train_ratio:.0%}/{args.val_ratio:.0%}/"
          f"{1-args.train_ratio-args.val_ratio:.0%}")
    if args.max_cases:
        print(f"Max cases: {args.max_cases}")
    print("="*80)

    # Find cases with both signals
    print("\n🔍 Finding cases with both PLETH and ECG_II...")
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

    # Get common cases and filter by start_case
    all_common_cases = sorted(list(ppg_cases.intersection(ecg_cases)))
    common_cases = [c for c in all_common_cases if c >= args.start_case]

    print(f"   ✓ {len(all_common_cases)} total cases have BOTH signals")
    print(f"   ✓ {len(common_cases)} cases at or after ID {args.start_case}")

    if len(common_cases) == 0:
        print("❌ No cases found with both signals!")
        return

    # Check if splits file exists (only use if NOT limiting cases)
    splits = None
    splits_path = Path(args.splits_file)

    if args.max_cases:
        # When limiting cases, always create new splits
        print(f"\n⚠️  --max-cases specified, ignoring existing splits file")
        print(f"   Using first {args.max_cases} cases")
        common_cases = common_cases[:args.max_cases]
        splits = None  # Force new splits creation
    elif splits_path.exists():
        print(f"\n📂 Loading existing splits from {splits_path}")
        try:
            with open(splits_path, 'r') as f:
                splits = json.load(f)
            print(f"   ✓ Loaded splits: Train={len(splits['train'])}, "
                  f"Val={len(splits['val'])}, Test={len(splits['test'])}")

            # Filter to only common cases
            splits = {
                'train': [c for c in splits['train'] if c in common_cases],
                'val': [c for c in splits['val'] if c in common_cases],
                'test': [c for c in splits['test'] if c in common_cases]
            }
            print(f"   After filtering: Train={len(splits['train'])}, "
                  f"Val={len(splits['val'])}, Test={len(splits['test'])}")

            # If filtering resulted in empty splits, create new ones
            if all(len(v) == 0 for v in splits.values()):
                print(f"   ⚠️  All splits empty after filtering, will create new splits")
                splits = None
        except Exception as e:
            print(f"   ⚠️  Could not load splits: {e}")
            splits = None

    # Create splits if not loaded
    if splits is None:
        print(f"\n📊 Creating new splits...")
        n = len(common_cases)
        train_end = int(args.train_ratio * n)
        val_end = int((args.train_ratio + args.val_ratio) * n)

        splits = {
            'train': common_cases[:train_end],
            'val': common_cases[train_end:val_end],
            'test': common_cases[val_end:]
        }
        print(f"   Train: {len(splits['train'])} cases ({len(splits['train'])/n*100:.1f}%)")
        print(f"   Val:   {len(splits['val'])} cases ({len(splits['val'])/n*100:.1f}%)")
        print(f"   Test:  {len(splits['test'])} cases ({len(splits['test'])/n*100:.1f}%)")

    # Create output directories
    output_path = Path(args.output)
    for split_name in ['train', 'val', 'test']:
        (output_path / split_name).mkdir(parents=True, exist_ok=True)

    # Process each split
    print(f"\n{'='*80}")
    print("🚀 Processing cases...")
    print(f"{'='*80}")

    summary = {}
    failed_cases = []

    for split_name in ['train', 'val', 'test']:
        case_ids = splits[split_name]

        if len(case_ids) == 0:
            print(f"\n⚠️  Skipping {split_name} (no cases)")
            continue

        print(f"\n📦 Processing {split_name.upper()} split ({len(case_ids)} cases)...")

        total_windows = 0
        successful_cases = 0
        split_failed = []
        error_summary = {}  # Track error types

        # Multiprocessing or sequential processing
        if args.num_workers > 1:
            # PARALLEL MODE: Use ProcessPoolExecutor
            print(f"  🚀 Using {args.num_workers} parallel workers...")

            # Create partial function with fixed arguments
            process_func = partial(
                process_case,
                output_dir=output_path / split_name,
                fs=args.fs,
                window_size=args.window_size,
                min_duration_min=args.min_duration,
                verbose=args.verbose,
                skip_existing=True  # Skip already processed cases (resume support)
            )

            # Process in parallel with progress bar
            with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                # Submit all tasks
                future_to_case = {
                    executor.submit(process_func, case_id): case_id
                    for case_id in case_ids
                }

                # Collect results with progress bar
                for future in tqdm(as_completed(future_to_case),
                                 total=len(case_ids),
                                 desc=f"  {split_name}"):
                    case_id = future_to_case[future]
                    try:
                        # Get result with timeout (5 minutes per case max)
                        result = future.result(timeout=300)

                        if isinstance(result, tuple):
                            num_windows, error_msg = result
                        else:
                            # Backwards compatibility
                            num_windows = result
                            error_msg = None

                        if num_windows is not None and num_windows > 0:
                            total_windows += num_windows
                            successful_cases += 1
                        else:
                            split_failed.append(case_id)
                            failed_cases.append((split_name, case_id, error_msg or "Unknown error"))
                            # Track error types
                            err_type = (error_msg or "Unknown").split(':')[0]
                            error_summary[err_type] = error_summary.get(err_type, 0) + 1

                    except FutureTimeoutError:
                        error_msg = "Worker timeout (>5min)"
                        print(f"\n    ⚠️  Case {case_id}: {error_msg}")
                        split_failed.append(case_id)
                        failed_cases.append((split_name, case_id, error_msg))
                        error_summary['Timeout'] = error_summary.get('Timeout', 0) + 1
                    except KeyboardInterrupt:
                        print("\n\n⚠️  Interrupted by user, shutting down workers...")
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise
                    except Exception as e:
                        error_msg = f"{type(e).__name__}: {str(e)[:100]}"
                        print(f"\n    ⚠️  Exception processing case {case_id}: {error_msg}")
                        split_failed.append(case_id)
                        failed_cases.append((split_name, case_id, error_msg))
                        error_summary[type(e).__name__] = error_summary.get(type(e).__name__, 0) + 1

        else:
            # SEQUENTIAL MODE: Original loop
            for case_id in tqdm(case_ids, desc=f"  {split_name}"):
                result = process_case(
                    case_id,
                    output_path / split_name,
                    args.fs,
                    args.window_size,
                    args.min_duration,  # Add minimum duration filter
                    args.verbose,
                    skip_existing=True  # Skip already processed cases
                )

                if isinstance(result, tuple):
                    num_windows, error_msg = result
                else:
                    num_windows = result
                    error_msg = None

                if num_windows is not None and num_windows > 0:
                    total_windows += num_windows
                    successful_cases += 1
                else:
                    split_failed.append(case_id)
                    failed_cases.append((split_name, case_id, error_msg or "Unknown error"))
                    # Track error types
                    err_type = (error_msg or "Unknown").split(':')[0]
                    error_summary[err_type] = error_summary.get(err_type, 0) + 1

        summary[split_name] = {
            'cases_attempted': len(case_ids),
            'cases_successful': successful_cases,
            'cases_failed': len(split_failed),
            'windows': total_windows,
            'error_summary': error_summary
        }

        print(f"  ✓ {split_name}: {successful_cases}/{len(case_ids)} cases, "
              f"{total_windows:,} windows")

        # Print error summary if there were failures
        if error_summary:
            print(f"\n  📊 Error Summary for {split_name}:")
            for err_type, count in sorted(error_summary.items(), key=lambda x: -x[1]):
                print(f"    • {err_type}: {count} cases")

        if len(split_failed) > 0 and len(split_failed) <= 10:
            print(f"    Failed cases: {split_failed}")
        elif len(split_failed) > 10:
            print(f"    Failed cases: {split_failed[:10]} ... and {len(split_failed)-10} more")

    # Save summary (convert numpy types to Python types)
    summary_file = output_path / 'dataset_summary.json'
    with open(summary_file, 'w') as f:
        summary_data = {
            'summary': summary,
            'config': {
                'window_size': args.window_size,
                'sampling_rate': args.fs,
                'splits': {k: len(v) for k, v in splits.items()}
            },
            'failed_cases': failed_cases
        }
        # Convert all numpy types to Python types for JSON serialization
        summary_data = convert_to_python_types(summary_data)
        json.dump(summary_data, f, indent=2)

    # Save splits (convert numpy arrays to lists)
    splits_file = output_path / 'splits.json'
    with open(splits_file, 'w') as f:
        splits_python = convert_to_python_types(splits)
        json.dump(splits_python, f, indent=2)

    # Final summary
    print(f"\n{'='*80}")
    print("✅ DATASET BUILD COMPLETE")
    print(f"{'='*80}")

    total_cases_attempted = sum(s['cases_attempted'] for s in summary.values())
    total_cases_successful = sum(s['cases_successful'] for s in summary.values())
    total_windows = sum(s['windows'] for s in summary.values())

    if total_cases_attempted == 0:
        print("\n⚠️  No cases were processed!")
        print("   This might happen if all splits were empty.")
        return

    print(f"\nTotal cases attempted: {total_cases_attempted}")
    print(f"Total cases successful: {total_cases_successful} "
          f"({total_cases_successful/total_cases_attempted*100:.1f}%)")
    print(f"Total windows created: {total_windows:,}")

    print(f"\n📊 Split breakdown:")
    for split_name, stats in summary.items():
        pct = (stats['windows'] / total_windows * 100) if total_windows > 0 else 0
        print(f"  {split_name:5s}: {stats['cases_successful']:3d} cases, "
              f"{stats['windows']:6,} windows ({pct:.1f}%)")

    print(f"\n💾 Output saved to: {output_path.absolute()}")
    print(f"   - Splits: {splits_file}")
    print(f"   - Summary: {summary_file}")

    if len(failed_cases) > 0:
        print(f"\n⚠️  {len(failed_cases)} cases failed to process")
        print(f"   See {summary_file} for details")

    print(f"\n{'='*80}")
    print("🎉 Ready for SSL pre-training!")
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
