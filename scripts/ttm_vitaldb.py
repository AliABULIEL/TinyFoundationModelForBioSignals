#!/usr/bin/env python3
"""
TTM × VitalDB Pipeline - Complete Version with All Fixes and Multiprocessing
============================================================================
Unified pipeline with:
- Fixed trainer initialization
- Multiprocessing support for window building
- All original functionality preserved
"""

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial
import traceback

import numpy as np
import torch
import yaml
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# FIXED IMPORTS - Match actual codebase
from src.data.splits import make_patient_level_splits, load_splits, save_splits
from src.data.vitaldb_loader import list_cases, load_channel, get_available_case_sets
from src.data.windows import (
    make_windows,
    compute_normalization_stats,
    normalize_windows,
    validate_cardiac_cycles,
    NormalizationStats
)
from src.data.filters import apply_bandpass_filter
from src.data.detect import find_ppg_peaks, find_ecg_rpeaks
from src.data.quality import compute_sqi
from src.eval.calibration import (
    TemperatureScaling,
    IsotonicCalibration,
    expected_calibration_error
)
from src.eval.metrics import compute_classification_metrics, compute_regression_metrics
from src.models.datasets import RawWindowDataset
from src.models.ttm_adapter import create_ttm_model
from src.models.trainers import TrainerClf, TrainerReg, create_optimizer
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)

# Suppress NeuroKit2 warnings in worker processes
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', message='NeuroKit2 detection failed')


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_splits_command(args):
    """Prepare train/val/test splits from VitalDB."""
    logger.info("Preparing patient splits...")

    # Get VitalDB case sets
    case_sets = get_available_case_sets()

    # Select case set
    if args.case_set and args.case_set in case_sets:
        available_cases = list(case_sets[args.case_set])
    else:
        # Default to BIS cases
        available_cases = list(case_sets.get('bis', []))

    logger.info(f"Available cases: {len(available_cases)}")

    # Filter based on mode
    if args.mode == "fasttrack":
        # Use first 70 cases for FastTrack mode
        case_ids = available_cases[:70]
        train_ratio = 50 / 70  # 50 train
        val_ratio = 0 / 70  # 0 val (skip for FastTrack)
    else:
        # Full mode uses all cases
        case_ids = available_cases
        train_ratio = 0.7
        val_ratio = 0.15

    logger.info(f"Using {len(case_ids)} cases for {args.mode} mode")

    # Create case dictionaries
    cases = [
        {'case_id': str(cid), 'subject_id': str(cid)}
        for cid in case_ids
    ]

    # Create splits
    if val_ratio > 0:
        splits = make_patient_level_splits(
            cases=cases,
            ratios=(train_ratio, val_ratio, 1.0 - train_ratio - val_ratio),
            seed=args.seed
        )
    else:
        splits = make_patient_level_splits(
            cases=cases,
            ratios=(train_ratio, 1.0 - train_ratio),
            seed=args.seed
        )

    # Convert to simple format (just case IDs)
    simple_splits = {}
    for split_name, split_cases in splits.items():
        simple_splits[split_name] = [c['case_id'] for c in split_cases]

    # Save splits
    output_file = Path(args.output) / f'splits_{args.mode}.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(simple_splits, f, indent=2)

    logger.info(f"Splits saved to {output_file}")
    for split_name, split_ids in simple_splits.items():
        logger.info(f"  {split_name}: {len(split_ids)} cases")


def process_single_case(args_tuple):
    """
    Process a single case - designed for multiprocessing.
    Returns windows and metadata or None if failed.
    """
    case_id, config_dict = args_tuple
    
    # Unpack configuration
    ch_config = config_dict['ch_config']
    vitaldb_track = config_dict['vitaldb_track']
    duration_sec = config_dict['duration_sec']
    window_s = config_dict['window_s']
    stride_s = config_dict['stride_s']
    min_cycles = config_dict['min_cycles']
    normalize_method = config_dict['normalize_method']
    min_sqi = config_dict['min_sqi']
    channel_name = config_dict['channel_name']
    train_stats = config_dict.get('train_stats')
    
    try:
        # Suppress warnings in worker
        import warnings
        warnings.filterwarnings('ignore')
        
        # Load signal with resampling to target fs (125 Hz)
        # IMPORTANT: use_cache=False to ensure fresh data after bug fixes
        signal, fs = load_channel(
            case_id=case_id,
            channel=vitaldb_track,
            duration_sec=duration_sec,
            auto_fix_alternating=True,
            target_fs=125.0,  # Always resample to 125 Hz
            use_cache=False  # Disable cache to use latest bug fixes
        )
        
        if signal is None or len(signal) < fs * 2:  # At least 2 seconds
            return None, f"No valid signal"
        
        # Apply filter
        if 'filter' in ch_config:
            filt = ch_config['filter']
            
            # Handle filter type mapping
            filter_type_map = {
                'butterworth': 'butter',
                'chebyshev2': 'cheby2',
                'cheby2': 'cheby2',
                'butter': 'butter'
            }
            filter_type = filter_type_map.get(filt.get('type', 'cheby2'), 'cheby2')
            
            # Handle both naming conventions
            lowcut = filt.get('lowcut', filt.get('low_freq', 0.5))
            highcut = filt.get('highcut', filt.get('high_freq', 10))
            
            signal = apply_bandpass_filter(
                signal, fs,
                lowcut=lowcut,
                highcut=highcut,
                filter_type=filter_type,
                order=filt.get('order', 4)
            )
        
        # Detect peaks based on signal type
        signal_type = channel_name.lower()
        peaks = None
        
        if signal_type in ['ppg', 'pleth']:
            peaks = find_ppg_peaks(signal, fs)
        elif signal_type in ['ecg']:
            peaks, _ = find_ecg_rpeaks(signal, fs)
        
        # Quality check if peaks available
        if peaks is not None and len(peaks) > 0:
            sqi = compute_sqi(signal, fs, peaks=peaks, signal_type=signal_type)
            if sqi < min_sqi:
                return None, f"Low SQI: {sqi:.3f}"
        
        # Create windows
        signal_tc = signal.reshape(-1, 1)
        peaks_tc = {0: peaks} if peaks is not None and len(peaks) > 0 else None
        
        # make_windows returns (windows, valid_mask)
        case_windows, valid_mask = make_windows(
            X=signal_tc,
            fs=fs,
            win_s=window_s,
            stride_s=stride_s,
            min_cycles=min_cycles if peaks_tc else 0,
            signal_type=signal_type
        )
        
        if case_windows is None or len(case_windows) == 0:
            return None, "No valid windows"
        
        # Normalize if train_stats provided
        if train_stats is not None:
            normalized = normalize_windows(
                W_ntc=case_windows,
                stats=train_stats,
                baseline_correction=False,
                per_channel=False
            )
        else:
            # Return raw windows for train stats computation
            normalized = case_windows
        
        return normalized, "Success"
        
    except Exception as e:
        return None, f"Error: {str(e)}"


def build_windows_command(args):
    """Build preprocessed windows from VitalDB data - with optional multiprocessing."""
    
    if args.multiprocess:
        logger.info("Building windows from VitalDB with MULTIPROCESSING...")
        return build_windows_multiprocess(args)
    else:
        logger.info("Building windows from VitalDB (single process)...")
        return build_windows_singleprocess(args)


def build_windows_multiprocess(args):
    """Build windows using multiprocessing for faster processing."""
    
    # Load configurations
    channels_config = load_config(args.channels_yaml)
    windows_config = load_config(args.windows_yaml)
    
    # Load splits
    splits_file = Path(args.split_file)
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    
    # Select split to process
    if args.split not in splits:
        raise ValueError(f"Split '{args.split}' not found in {splits_file}")
    
    case_ids = splits[args.split]
    logger.info(f"Processing {args.split} split: {len(case_ids)} cases")
    
    # Handle nested config structure - check for 'pretrain' or 'channels' keys
    if 'pretrain' in channels_config:
        # New format: pretrain -> PPG/ECG
        channels_dict = channels_config['pretrain']
    elif 'channels' in channels_config:
        # Old format: channels -> list
        channels_dict = channels_config['channels']
    else:
        # Fallback: use root level
        channels_dict = channels_config
    
    # Get channel configuration
    channel_name = args.channel or 'PPG'
    
    # Try exact match first
    if channel_name in channels_dict:
        ch_config = channels_dict[channel_name]
        logger.info(f"Found channel config for '{channel_name}'")
    else:
        # Try case-insensitive match
        channel_name_lower = channel_name.lower()
        found = False
        for key in channels_dict.keys():
            if key.lower() == channel_name_lower:
                channel_name = key
                ch_config = channels_dict[key]
                logger.info(f"Found channel config for '{channel_name}' (case-insensitive match)")
                found = True
                break
        
        if not found:
            logger.error(f"Channel '{channel_name}' not found in config!")
            logger.error(f"Available channels: {list(channels_dict.keys())}")
            raise ValueError(f"Channel '{channel_name}' not found in channels config")
    
    # Window parameters - Fixed to read correct config structure
    if 'window' in windows_config:
        window_s = windows_config['window'].get('size_seconds', 4.096)  # TTM default: 512 samples at 125Hz
        stride_s = windows_config['window'].get('step_seconds', 4.096)
    else:
        # Fallback for old config format
        window_s = windows_config.get('window_length_sec', 4.096)
        stride_s = windows_config.get('stride_sec', 4.096)
    
    min_cycles = windows_config.get('quality', {}).get('min_cycles', 3)
    normalize_method = windows_config.get('normalization', {}).get('method', 'zscore')
    
    # Get VitalDB track name
    track_mapping = {
        'PPG': 'PLETH',
        'ECG': 'ECG_II',
        'ABP': 'ABP',
        'EEG': 'EEG1'
    }
    vitaldb_track = ch_config.get('vitaldb_track', track_mapping.get(channel_name.upper(), 'PLETH'))
    
    # Prepare configuration dict for workers
    config_dict = {
        'ch_config': ch_config,
        'vitaldb_track': vitaldb_track,
        'duration_sec': args.duration_sec,
        'window_s': window_s,
        'stride_s': stride_s,
        'min_cycles': min_cycles,
        'normalize_method': normalize_method,
        'min_sqi': args.min_sqi,
        'channel_name': channel_name,
        'train_stats': None
    }
    
    # Load train stats if processing val/test
    train_stats = None
    if args.split != 'train':
        stats_file = Path(args.outdir).parent / 'train' / 'train_stats.npz'
        if not stats_file.exists():
            stats_file = Path(args.outdir).parent / 'train_stats.npz'
        
        if stats_file.exists():
            stats_data = np.load(stats_file)
            train_stats = NormalizationStats(
                mean=stats_data['mean'],
                std=stats_data['std'],
                method=str(stats_data['method']) if 'method' in stats_data else normalize_method
            )
            config_dict['train_stats'] = train_stats
            logger.info(f"Loaded train stats from {stats_file}")
    
    # Determine number of workers
    num_workers = args.num_workers if hasattr(args, 'num_workers') and args.num_workers else min(cpu_count() - 1, 8)
    logger.info(f"Using {num_workers} workers for parallel processing")
    
    # Process first case to compute train stats if needed
    all_windows = []
    all_labels = []
    first_batch = []
    
    if args.split == 'train' and train_stats is None:
        logger.info("Computing normalization statistics from first batch...")
        # Process first few cases to get stats
        first_batch = case_ids[:min(10, len(case_ids))]
        first_windows = []
        
        for case_id in first_batch:
            windows, status = process_single_case((case_id, config_dict))
            if windows is not None and len(windows) > 0:
                # Validate and fix window shapes before adding
                for w in windows:
                    # Ensure window is 2D [T, C]
                    if w.ndim == 1:
                        w = w.reshape(-1, 1)
                    elif w.ndim == 3:
                        w = w.squeeze(0)
                    
                    # Validate shape matches expected dimensions
                    expected_samples = int(window_s * 125)  # Assuming 125Hz
                    if w.shape[0] == expected_samples:
                        first_windows.append(w)
                    else:
                        logger.debug(f"Skipping window with shape {w.shape}, expected {expected_samples} samples")
        
        if first_windows:
            # Compute stats - windows should now have consistent shapes
            try:
                first_array = np.array(first_windows)
            except ValueError as e:
                logger.error(f"Shape inconsistency in windows: {e}")
                logger.error(f"Window shapes: {[w.shape for w in first_windows[:5]]}")  # Show first 5
                # Fallback: stack only windows that match the first window's shape
                ref_shape = first_windows[0].shape
                first_windows = [w for w in first_windows if w.shape == ref_shape]
                first_array = np.array(first_windows)
                logger.info(f"Filtered to {len(first_windows)} windows with consistent shape {ref_shape}")
            train_stats = compute_normalization_stats(
                X=first_array,
                method=normalize_method,
                axis=(0, 1)
            )
            config_dict['train_stats'] = train_stats
            
            # Save train stats
            stats_file = Path(args.outdir) / 'train_stats.npz'
            stats_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                stats_file,
                mean=train_stats.mean,
                std=train_stats.std,
                method=normalize_method
            )
            logger.info(f"Train stats saved to {stats_file}")
            
            # Normalize and store first batch
            for w in first_windows:
                normalized = normalize_windows(
                    W_ntc=w.reshape(1, *w.shape),
                    stats=train_stats,
                    baseline_correction=False,
                    per_channel=False
                )[0]
                all_windows.append(normalized)
                all_labels.append(0)
            
            # Remove processed cases from list
            case_ids = case_ids[len(first_batch):]
    
    # Prepare arguments for multiprocessing
    process_args = [(case_id, config_dict) for case_id in case_ids]
    
    # Process cases in parallel
    successful_cases = 0
    failed_cases = []
    
    with Pool(num_workers) as pool:
        # Use imap for progress bar
        results = pool.imap(process_single_case, process_args)
        
        # Process results with progress bar
        for case_id, (windows, status) in zip(case_ids, tqdm(results, total=len(case_ids), 
                                                              desc=f"Processing {args.split}")):
            if windows is not None and len(windows) > 0:
                # Validate window shapes before adding
                for w in windows:
                    # Ensure window is 2D [T, C]
                    if w.ndim == 1:
                        w = w.reshape(-1, 1)
                    elif w.ndim == 3:
                        w = w.squeeze(0)
                    
                    # Only add if shape is valid
                    expected_samples = int(window_s * 125)  # Assuming 125Hz
                    if w.shape[0] == expected_samples:
                        all_windows.append(w)
                        all_labels.append(0)  # Placeholder label
                successful_cases += 1
            else:
                failed_cases.append((case_id, status))
    
    # Summary
    total_cases = len(case_ids) + len(first_batch)
    logger.info(f"\nProcessing complete:")
    logger.info(f"  Successful cases: {successful_cases + len(first_batch)}/{total_cases}")
    logger.info(f"  Total windows: {len(all_windows)}")
    
    if len(failed_cases) > 0 and len(failed_cases) <= 20:
        logger.info(f"  Failed cases: {len(failed_cases)}")
        for case_id, reason in failed_cases[:10]:
            logger.debug(f"    {case_id}: {reason}")
    
    # Save windows
    if all_windows:
        # Final shape validation before creating array
        try:
            windows_array = np.array(all_windows)
            labels_array = np.array(all_labels)
        except ValueError as e:
            logger.error(f"Shape inconsistency when creating final array: {e}")
            logger.error(f"Sample shapes: {[w.shape for w in all_windows[:10]]}")
            # Filter to consistent shape
            ref_shape = all_windows[0].shape
            logger.info(f"Filtering windows to shape {ref_shape}")
            filtered_windows = []
            filtered_labels = []
            for w, l in zip(all_windows, all_labels):
                if w.shape == ref_shape:
                    filtered_windows.append(w)
                    filtered_labels.append(l)
            logger.info(f"Kept {len(filtered_windows)}/{len(all_windows)} windows with consistent shape")
            windows_array = np.array(filtered_windows)
            labels_array = np.array(filtered_labels)
        
        output_file = Path(args.outdir) / f'{args.split}_windows.npz'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez_compressed(
            output_file,
            data=windows_array,
            labels=labels_array
        )
        
        logger.info(f"\n✓ Saved {len(all_windows)} windows to {output_file}")
        logger.info(f"  Shape: {windows_array.shape}")
        logger.info(f"  Size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    else:
        logger.error(f"\n✗ No valid windows created for {args.split} split")
        sys.exit(1)


def build_windows_singleprocess(args):
    """Original single-process window building (fallback)."""
    logger.info("Building windows from VitalDB...")

    # Load configurations
    channels_config = load_config(args.channels_yaml)
    windows_config = load_config(args.windows_yaml)

    # Load splits
    splits_file = Path(args.split_file)
    with open(splits_file, 'r') as f:
        splits = json.load(f)

    # Select split to process
    if args.split not in splits:
        raise ValueError(f"Split '{args.split}' not found in {splits_file}")

    case_ids = splits[args.split]
    logger.info(f"Processing {args.split} split: {len(case_ids)} cases")

    # Handle nested config structure - check for 'pretrain' or 'channels' keys
    if 'pretrain' in channels_config:
        # New format: pretrain -> PPG/ECG
        channels_dict = channels_config['pretrain']
    elif 'channels' in channels_config:
        # Old format: channels -> list
        channels_dict = channels_config['channels']
    else:
        # Fallback: use root level
        channels_dict = channels_config
    
    # Get channel configuration
    channel_name = args.channel or 'PPG'
    
    # Try exact match first
    if channel_name in channels_dict:
        ch_config = channels_dict[channel_name]
        logger.info(f"Found channel config for '{channel_name}'")
    else:
        # Try case-insensitive match
        channel_name_lower = channel_name.lower()
        found = False
        for key in channels_dict.keys():
            if key.lower() == channel_name_lower:
                channel_name = key
                ch_config = channels_dict[key]
                logger.info(f"Found channel config for '{channel_name}' (case-insensitive match)")
                found = True
                break
        
        if not found:
            logger.error(f"Channel '{channel_name}' not found in config!")
            logger.error(f"Available channels: {list(channels_dict.keys())}")
            raise ValueError(f"Channel '{channel_name}' not found in channels config")

    # Window parameters - Fixed to read correct config structure
    if 'window' in windows_config:
        window_s = windows_config['window'].get('size_seconds', 4.096)  # TTM default: 512 samples at 125Hz
        stride_s = windows_config['window'].get('step_seconds', 4.096)
    else:
        # Fallback for old config format
        window_s = windows_config.get('window_length_sec', 4.096)
        stride_s = windows_config.get('stride_sec', 4.096)
    
    min_cycles = windows_config.get('quality', {}).get('min_cycles', 3)
    normalize_method = windows_config.get('normalization', {}).get('method', 'zscore')

    # Get VitalDB track name
    track_mapping = {
        'PPG': 'PLETH',
        'ECG': 'ECG_II',
        'ABP': 'ABP',
        'EEG': 'EEG1'
    }
    vitaldb_track = ch_config.get('vitaldb_track', track_mapping.get(channel_name.upper(), 'PLETH'))

    # Storage
    all_windows = []
    all_labels = []
    train_stats = None
    successful_cases = 0
    failed_cases = []

    # Process cases
    logger.info(f"Loading channel '{vitaldb_track}' from {len(case_ids)} cases...")
    for case_id in tqdm(case_ids, desc=f"Processing {args.split}"):
        try:
            # Load signal with resampling to target fs (125 Hz)
            signal, fs = load_channel(
                case_id=case_id,
                channel=vitaldb_track,
                duration_sec=args.duration_sec,
                auto_fix_alternating=True,
                target_fs=125.0  # Always resample to 125 Hz
            )

            if signal is None or len(signal) < fs * 2:
                logger.debug(f"Case {case_id}: No valid signal")
                failed_cases.append((case_id, "No signal"))
                continue

            # Apply filter
            if 'filter' in ch_config:
                filt = ch_config['filter']
                filter_type_map = {
                    'butterworth': 'butter',
                    'chebyshev2': 'cheby2',
                    'cheby2': 'cheby2',
                    'butter': 'butter'
                }
                filter_type = filter_type_map.get(filt.get('type', 'cheby2'), 'cheby2')
                lowcut = filt.get('lowcut', filt.get('low_freq', 0.5))
                highcut = filt.get('highcut', filt.get('high_freq', 10))

                signal = apply_bandpass_filter(
                    signal, fs,
                    lowcut=lowcut,
                    highcut=highcut,
                    filter_type=filter_type,
                    order=filt.get('order', 4)
                )

            # Detect peaks
            signal_type = channel_name.lower()
            peaks = None

            if signal_type in ['ppg', 'pleth']:
                peaks = find_ppg_peaks(signal, fs)
            elif signal_type in ['ecg']:
                peaks, _ = find_ecg_rpeaks(signal, fs)

            # Quality check
            if peaks is not None and len(peaks) > 0:
                sqi = compute_sqi(signal, fs, peaks=peaks, signal_type=signal_type)
                if sqi < args.min_sqi:
                    logger.debug(f"Case {case_id}: Low quality (SQI={sqi:.3f})")
                    failed_cases.append((case_id, f"Low SQI: {sqi:.3f}"))
                    continue

            # Create windows  
            signal_tc = signal.reshape(-1, 1)
            peaks_tc = {0: peaks} if peaks is not None and len(peaks) > 0 else None

            # make_windows returns (windows, valid_mask)
            case_windows, valid_mask = make_windows(
                X=signal_tc,
                fs=fs,
                win_s=window_s,
                stride_s=stride_s,
                min_cycles=min_cycles if peaks_tc else 0,
                signal_type=signal_type
            )

            if case_windows is None or len(case_windows) == 0:
                logger.debug(f"Case {case_id}: No valid windows")
                failed_cases.append((case_id, "No valid windows"))
                continue

            # Compute normalization stats on first successful train case
            if args.split == 'train' and train_stats is None:
                train_stats = compute_normalization_stats(
                    X=case_windows,
                    method=normalize_method,
                    axis=(0, 1)
                )
                stats_file = Path(args.outdir) / 'train_stats.npz'
                stats_file.parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    stats_file,
                    mean=train_stats.mean,
                    std=train_stats.std,
                    method=normalize_method
                )
                logger.info(f"Train stats computed and saved to {stats_file}")

            # Load train stats if processing val/test
            if args.split != 'train' and train_stats is None:
                stats_file = Path(args.outdir).parent / 'train' / 'train_stats.npz'
                if not stats_file.exists():
                    stats_file = Path(args.outdir).parent / 'train_stats.npz'

                if stats_file.exists():
                    stats_data = np.load(stats_file)
                    train_stats = NormalizationStats(
                        mean=stats_data['mean'],
                        std=stats_data['std'],
                        method=str(stats_data['method']) if 'method' in stats_data else normalize_method
                    )
                    logger.info(f"Loaded train stats from {stats_file}")

            # Normalize windows
            if train_stats is not None:
                normalized = normalize_windows(
                    W_ntc=case_windows,
                    stats=train_stats,
                    baseline_correction=False,
                    per_channel=False
                )
            else:
                normalized = case_windows
                mean = np.mean(normalized)
                std = np.std(normalized)
                if std > 1e-8:
                    normalized = (normalized - mean) / std

            # Store windows
            for w in normalized:
                all_windows.append(w)
                all_labels.append(0)

            successful_cases += 1

        except Exception as e:
            logger.warning(f"Case {case_id}: Error - {str(e)}")
            failed_cases.append((case_id, str(e)))
            continue

    # Summary
    logger.info(f"\nProcessing complete:")
    logger.info(f"  Successful cases: {successful_cases}/{len(case_ids)}")
    logger.info(f"  Total windows: {len(all_windows)}")

    if len(failed_cases) > 0:
        logger.info(f"  Failed cases: {len(failed_cases)}")
        if len(failed_cases) <= 10:
            for case_id, reason in failed_cases[:10]:
                logger.debug(f"    {case_id}: {reason}")

    # Save windows
    if all_windows:
        windows_array = np.array(all_windows)
        labels_array = np.array(all_labels)

        output_file = Path(args.outdir) / f'{args.split}_windows.npz'
        output_file.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            output_file,
            data=windows_array,
            labels=labels_array
        )

        logger.info(f"\n✓ Saved {len(all_windows)} windows to {output_file}")
        logger.info(f"  Shape: {windows_array.shape}")
        logger.info(f"  Size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    else:
        logger.error(f"\n✗ No valid windows created for {args.split} split")
        logger.error(f"  Check case availability and signal quality")
        sys.exit(1)


def train_command(args):
    """Train TTM model - FIXED VERSION with proper trainer initialization."""
    logger.info("Training TTM model...")

    # Load configurations
    model_config = load_config(args.model_yaml)
    run_config = load_config(args.run_yaml)

    # Set seed
    set_seed(run_config.get('seed', 42))

    # Load splits
    with open(args.split_file, 'r') as f:
        splits = json.load(f)

    # Find training data
    train_file = None
    possible_locations = [
        Path(args.outdir) / 'train_windows.npz',
        Path(args.outdir) / 'train' / 'train_windows.npz',
        Path('artifacts/raw_windows/train/train_windows.npz'),
        Path('artifacts/raw_windows/train_windows.npz'),
    ]
    
    for loc in possible_locations:
        if loc.exists():
            train_file = loc
            break
    
    if train_file is None:
        raise FileNotFoundError(f"Training data not found. Searched: {possible_locations}")

    train_data = np.load(train_file)
    logger.info(f"Loaded training data: {train_data['data'].shape} from {train_file}")

    # Find validation data
    val_file = None
    val_locations = [
        Path(args.outdir) / 'val_windows.npz',
        Path(args.outdir) / 'val' / 'val_windows.npz',
        Path(args.outdir) / 'test_windows.npz',
        Path(args.outdir) / 'test' / 'test_windows.npz',
        Path('artifacts/raw_windows/test/test_windows.npz'),
    ]
    
    for loc in val_locations:
        if loc.exists():
            val_file = loc
            break

    val_data = None
    if val_file and val_file.exists():
        val_data = np.load(val_file)
        logger.info(f"Loaded validation data: {val_data['data'].shape} from {val_file}")

    # Create datasets
    from torch.utils.data import TensorDataset, DataLoader

    X_train = torch.from_numpy(train_data['data']).float()
    
    # Handle labels
    if 'labels' in train_data:
        y_train = torch.from_numpy(train_data['labels']).long()
    else:
        logger.warning("No labels found in data, creating random binary labels for demonstration")
        y_train = torch.randint(0, 2, (len(X_train),))
    
    train_dataset = TensorDataset(X_train, y_train)

    val_dataset = None
    val_loader = None
    if val_data is not None:
        X_val = torch.from_numpy(val_data['data']).float()
        if 'labels' in val_data:
            y_val = torch.from_numpy(val_data['labels']).long()
        else:
            y_val = torch.randint(0, 2, (len(X_val),))
        val_dataset = TensorDataset(X_val, y_val)

    # Configure model
    model_config['input_channels'] = train_data['data'].shape[2]
    model_config['context_length'] = train_data['data'].shape[1]
    
    # Task configuration
    task = args.task if hasattr(args, 'task') else model_config.get('task', {}).get('type', 'classification')
    
    if task in ['classification', 'clf']:
        model_config['task'] = 'classification'
        model_config['num_classes'] = model_config.get('num_classes', 2)
    else:
        model_config['task'] = 'regression'
        model_config['out_features'] = model_config.get('out_features', 1)

    # Handle FastTrack mode
    if args.fasttrack:
        model_config['freeze_encoder'] = True
        model_config['head_type'] = 'linear'
        run_config['num_epochs'] = min(run_config.get('num_epochs', 10), 10)
        run_config['batch_size'] = min(run_config.get('batch_size', 32), 32)
        logger.info("FastTrack mode: Frozen encoder, linear head, max 10 epochs")

    # Create model
    model = create_ttm_model(model_config)
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")

    # Create data loaders
    batch_size = run_config.get('batch_size', 32)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

    # Create optimizer
    optimizer = create_optimizer(
        model,
        optimizer_type=run_config.get('optimizer', 'adamw'),
        lr=run_config.get('learning_rate', 1e-3),
        weight_decay=run_config.get('weight_decay', 1e-4)
    )

    # Get device
    device = run_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Select trainer based on task - FIXED INITIALIZATION
    if task in ['classification', 'clf']:
        trainer = TrainerClf(
            model=model,
            train_loader=train_loader,  # Required positional argument
            val_loader=val_loader,      # Optional
            optimizer=optimizer,         # Optional
            num_classes=model_config.get('num_classes', 2),
            checkpoint_dir=str(output_dir),
            device=device,
            use_amp=run_config.get('use_amp', False),
            gradient_clip=run_config.get('gradient_clip', 1.0),
            log_interval=run_config.get('log_interval', 10)
        )
    else:
        trainer = TrainerReg(
            model=model,
            train_loader=train_loader,  # Required positional argument
            val_loader=val_loader,      # Optional
            optimizer=optimizer,         # Optional
            loss_type=run_config.get('loss_type', 'mse'),
            checkpoint_dir=str(output_dir),
            device=device,
            use_amp=run_config.get('use_amp', False),
            gradient_clip=run_config.get('gradient_clip', 1.0),
            log_interval=run_config.get('log_interval', 10)
        )

    # Train
    num_epochs = run_config.get('num_epochs', 10)
    history = trainer.fit(
        num_epochs=num_epochs,
        save_best=True,
        early_stopping_patience=run_config.get('early_stopping_patience', 5),
        monitor_metric='accuracy' if task in ['classification', 'clf'] else 'mse',
        monitor_mode='max' if task in ['classification', 'clf'] else 'min'
    )

    # Save final results
    results_file = output_dir / 'training_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'best_epoch': history['best_epoch'],
            'best_metric': float(history['best_metric']),
            'train_history': history['train_history'],
            'val_history': history['val_history']
        }, f, indent=2)

    logger.info(f"\n✓ Training complete!")
    logger.info(f"  Best model: {output_dir / 'best_model.pt'}")
    logger.info(f"  Results: {results_file}")
    logger.info(f"  Best metric: {history['best_metric']:.4f} at epoch {history['best_epoch']}")


def test_command(args):
    """Test model on test set."""
    logger.info("Testing model...")

    # Load configurations
    model_config = load_config(args.model_yaml)
    run_config = load_config(args.run_yaml)

    # Load test data
    test_file = Path(args.outdir) / 'test_windows.npz'
    if not test_file.exists():
        test_file = Path(args.outdir) / 'test' / 'test_windows.npz'

    if not test_file.exists():
        raise FileNotFoundError(f"Test data not found: {test_file}")

    test_data = np.load(test_file)
    logger.info(f"Loaded test data: {test_data['data'].shape}")

    # Create dataset
    from torch.utils.data import TensorDataset, DataLoader
    X_test = torch.from_numpy(test_data['data']).float()
    
    if 'labels' in test_data:
        y_test = torch.from_numpy(test_data['labels']).long()
    else:
        y_test = torch.randint(0, 2, (len(X_test),))
        
    test_dataset = TensorDataset(X_test, y_test)

    # Create data loader
    batch_size = run_config.get('batch_size', 32)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load model
    checkpoint = torch.load(args.ckpt, map_location='cpu', weights_only=False)

    # Recreate model architecture
    model_config['input_channels'] = test_data['data'].shape[2]
    model_config['context_length'] = test_data['data'].shape[1]
    model = create_ttm_model(model_config)

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    logger.info(f"Loaded model from {args.ckpt}")

    # Move to device
    device = torch.device(run_config.get('device', 'cpu'))
    model = model.to(device)
    model.eval()

    # Evaluate
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)

            # Get predictions
            task = model_config.get('task', 'classification')
            if task == 'classification':
                probs = torch.softmax(output, dim=1)
                preds = torch.argmax(output, dim=1)
                all_probs.extend(probs.cpu().numpy())
            else:
                preds = output.squeeze()
                all_probs = None

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    if model_config.get('task', 'classification') == 'classification':
        metrics = compute_classification_metrics(all_labels, all_preds)
        if all_probs:
            all_probs = np.array(all_probs)
            ece = expected_calibration_error(all_labels, all_probs[:, 1] if all_probs.shape[1] == 2 else all_probs)
            metrics['ece'] = ece
    else:
        metrics = compute_regression_metrics(all_labels, all_preds)

    # Save results
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'metrics': metrics,
        'predictions': all_preds.tolist(),
        'labels': all_labels.tolist(),
        'model_path': str(args.ckpt)
    }

    results_file = output_dir / 'test_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nTest Results:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")

    logger.info(f"\nResults saved to {results_file}")


def inspect_command(args):
    """Inspect data or model."""
    if args.data:
        data = np.load(args.data)
        print(f"\nData file: {args.data}")
        print(f"Arrays: {list(data.keys())}")
        for key in data.keys():
            arr = data[key]
            print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
            if key == 'data':
                print(f"    Min: {arr.min():.3f}, Max: {arr.max():.3f}, Mean: {arr.mean():.3f}")

    if args.model:
        checkpoint = torch.load(args.model, map_location='cpu')
        print(f"\nModel file: {args.model}")
        print(f"Keys: {list(checkpoint.keys())}")
        
        if 'model_state_dict' in checkpoint:
            print(f"\nModel architecture:")
            for key in list(checkpoint['model_state_dict'].keys())[:10]:
                print(f"  {key}: {checkpoint['model_state_dict'][key].shape}")


def main():
    """Main entry point with all commands."""
    parser = argparse.ArgumentParser(
        description="TTM × VitalDB Pipeline - Complete Version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare splits
  python ttm_vitaldb.py prepare-splits --mode fasttrack --output data/splits
  
  # Build windows with multiprocessing
  python ttm_vitaldb.py build-windows --multiprocess --num-workers 8 \\
    --channels-yaml configs/channels.yaml --windows-yaml configs/windows.yaml \\
    --split-file data/splits/splits_fasttrack.json --split train \\
    --outdir artifacts/raw_windows/train
  
  # Train model
  python ttm_vitaldb.py train --fasttrack \\
    --model-yaml configs/model.yaml --run-yaml configs/run.yaml \\
    --split-file data/splits/splits_fasttrack.json \\
    --outdir artifacts/raw_windows --out artifacts/model_output
        """
    )

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s:%(message)s'
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # prepare-splits command
    splits_parser = subparsers.add_parser('prepare-splits', help='Prepare train/val/test splits')
    splits_parser.add_argument('--mode', choices=['fasttrack', 'full'], default='fasttrack')
    splits_parser.add_argument('--case-set', type=str, help='VitalDB case set')
    splits_parser.add_argument('--output', type=str, default='data', help='Output directory')
    splits_parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # build-windows command with multiprocessing support
    windows_parser = subparsers.add_parser('build-windows', help='Build preprocessed windows')
    windows_parser.add_argument('--channels-yaml', type=str, required=True, help='Channels config')
    windows_parser.add_argument('--windows-yaml', type=str, required=True, help='Windows config')
    windows_parser.add_argument('--split-file', type=str, required=True, help='Splits JSON file')
    windows_parser.add_argument('--split', type=str, required=True, help='Split to process')
    windows_parser.add_argument('--channel', type=str, help='Channel to process')
    windows_parser.add_argument('--duration-sec', type=float, default=60, help='Duration per case')
    windows_parser.add_argument('--min-sqi', type=float, default=0.5, help='Minimum SQI')
    windows_parser.add_argument('--outdir', type=str, required=True, help='Output directory')
    windows_parser.add_argument('--multiprocess', action='store_true', help='Use multiprocessing')
    windows_parser.add_argument('--num-workers', type=int, help='Number of parallel workers')

    # train command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--model-yaml', type=str, required=True, help='Model config')
    train_parser.add_argument('--run-yaml', type=str, required=True, help='Run config')
    train_parser.add_argument('--split-file', type=str, required=True, help='Splits file')
    train_parser.add_argument('--split', type=str, default='train', help='Split name')
    train_parser.add_argument('--task', type=str, default='clf', choices=['clf', 'reg'], help='Task type')
    train_parser.add_argument('--outdir', type=str, default='artifacts/raw_windows', help='Data directory')
    train_parser.add_argument('--out', type=str, required=True, help='Output directory')
    train_parser.add_argument('--fasttrack', action='store_true', help='FastTrack mode')
    train_parser.add_argument('--ecg-mode', type=str, choices=['diagnosis', 'analysis'], 
                            default='analysis', help='ECG processing mode')

    # test command
    test_parser = subparsers.add_parser('test', help='Test model')
    test_parser.add_argument('--ckpt', type=str, required=True, help='Model checkpoint')
    test_parser.add_argument('--model-yaml', type=str, required=True, help='Model config')
    test_parser.add_argument('--run-yaml', type=str, required=True, help='Run config')
    test_parser.add_argument('--split-file', type=str, required=True, help='Splits file')
    test_parser.add_argument('--outdir', type=str, required=True, help='Data directory')
    test_parser.add_argument('--out', type=str, required=True, help='Output directory')

    # inspect command
    inspect_parser = subparsers.add_parser('inspect', help='Inspect data or model')
    inspect_parser.add_argument('--data', type=str, help='Data file to inspect')
    inspect_parser.add_argument('--model', type=str, help='Model file to inspect')

    args = parser.parse_args()

    if args.command == 'prepare-splits':
        prepare_splits_command(args)
    elif args.command == 'build-windows':
        build_windows_command(args)
    elif args.command == 'train':
        train_command(args)
    elif args.command == 'test':
        test_command(args)
    elif args.command == 'inspect':
        inspect_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
