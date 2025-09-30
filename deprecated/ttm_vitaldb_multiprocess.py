#!/usr/bin/env python3
"""
TTM × VitalDB pipeline with MULTIPROCESSING support for faster window building.
Processes multiple cases in parallel to reduce preprocessing time.
"""

import argparse
import json
import logging
import os
import sys
import warnings
import importlib
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
from src.models.trainers import TrainerClf, TrainerReg
from src.utils.seed import set_seed

# Configure logging for multiprocessing
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(process)d:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

# Suppress NeuroKit2 warnings in worker processes
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', message='NeuroKit2 detection failed')


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
        
        # Load signal
        signal, fs = load_channel(
            case_id=case_id,
            channel=vitaldb_track,
            duration_sec=duration_sec,
            auto_fix_alternating=True
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
        
        case_windows = make_windows(
            X_tc=signal_tc,
            fs=fs,
            win_s=window_s,
            stride_s=stride_s,
            min_cycles=min_cycles if peaks_tc else 0,
            peaks_tc=peaks_tc
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


def build_windows_multiprocess(args):
    """
    Build preprocessed windows from VitalDB data using multiprocessing.
    """
    logger.info("Building windows from VitalDB with multiprocessing...")
    
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
    
    # Handle nested 'channels' structure
    if 'channels' in channels_config:
        channels_dict = channels_config['channels']
    else:
        channels_dict = channels_config
    
    # Get channel configuration
    channel_name = args.channel or 'PPG'
    if channel_name not in channels_dict:
        channel_name_lower = channel_name.lower()
        for key in channels_dict.keys():
            if key.lower() == channel_name_lower:
                channel_name = key
                break
        else:
            logger.warning(f"Channel '{channel_name}' not found, using PPG")
            channel_name = 'PPG'
    
    ch_config = channels_dict.get(channel_name, {})
    
    # Window parameters
    window_s = windows_config.get('window_length_sec', 10.0)
    stride_s = windows_config.get('stride_sec', 10.0)
    min_cycles = windows_config.get('min_cycles', 3)
    normalize_method = windows_config.get('normalize_method', 'zscore')
    
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
    num_workers = args.num_workers if args.num_workers else min(cpu_count() - 1, 8)
    logger.info(f"Using {num_workers} workers for parallel processing")
    
    # Process first case to compute train stats if needed
    all_windows = []
    all_labels = []
    
    if args.split == 'train' and train_stats is None:
        logger.info("Computing normalization statistics from first batch...")
        # Process first few cases to get stats
        first_batch = case_ids[:min(10, len(case_ids))]
        first_windows = []
        
        for case_id in first_batch:
            windows, status = process_single_case((case_id, config_dict))
            if windows is not None:
                first_windows.extend(windows)
        
        if first_windows:
            # Compute stats
            first_array = np.array(first_windows)
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
            if windows is not None:
                for w in windows:
                    all_windows.append(w)
                    all_labels.append(0)  # Placeholder label
                successful_cases += 1
            else:
                failed_cases.append((case_id, status))
    
    # Summary
    logger.info(f"\nProcessing complete:")
    logger.info(f"  Successful cases: {successful_cases}/{len(case_ids) + len(first_batch if args.split == 'train' else [])}")
    logger.info(f"  Total windows: {len(all_windows)}")
    
    if len(failed_cases) > 0 and len(failed_cases) <= 20:
        logger.info(f"  Failed cases: {len(failed_cases)}")
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
        sys.exit(1)


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# Keep all other command functions from ttm_vitaldb_fixed.py
# (prepare_splits_command, train_command, test_command, inspect_command)
# ... [Copy these from ttm_vitaldb_fixed.py] ...


def main():
    """Main entry point with multiprocessing option."""
    parser = argparse.ArgumentParser(description="TTM × VitalDB Pipeline with Multiprocessing")
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(process)d:%(thread)d:%(name)s:%(funcName)s:%(message)s'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # build-windows command with multiprocessing
    windows_parser = subparsers.add_parser('build-windows', help='Build preprocessed windows')
    windows_parser.add_argument('--channels-yaml', type=str, required=True, help='Channels config')
    windows_parser.add_argument('--windows-yaml', type=str, required=True, help='Windows config')
    windows_parser.add_argument('--split-file', type=str, required=True, help='Splits JSON file')
    windows_parser.add_argument('--split', type=str, required=True, help='Split to process')
    windows_parser.add_argument('--channel', type=str, help='Channel to process')
    windows_parser.add_argument('--duration-sec', type=float, default=60, help='Duration per case')
    windows_parser.add_argument('--min-sqi', type=float, default=0.5, help='Minimum SQI')
    windows_parser.add_argument('--outdir', type=str, required=True, help='Output directory')
    windows_parser.add_argument('--num-workers', type=int, help='Number of parallel workers (default: CPU count - 1)')
    
    # Add other subparsers here...
    # (prepare-splits, train, test, inspect)
    
    args = parser.parse_args()
    
    if args.command == 'build-windows':
        build_windows_multiprocess(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
