#!/usr/bin/env python3
"""
Fixed TTM × VitalDB pipeline with proper configuration handling.
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
    validate_cardiac_cycles
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

logger = logging.getLogger(__name__)


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
        train_ratio = 50/70  # 50 train
        val_ratio = 0/70     # 0 val (skip for FastTrack)
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


def build_windows_command(args):
    """Build preprocessed windows from VitalDB data - FIXED VERSION."""
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
    
    # FIX 1: Handle nested 'channels' structure
    if 'channels' in channels_config:
        channels_dict = channels_config['channels']
    else:
        channels_dict = channels_config
    
    # Get channel configuration
    channel_name = args.channel or 'PPG'  # Default to PPG
    if channel_name not in channels_dict:
        # Try case-insensitive match
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
    
    # Get VitalDB track name with fallback
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
            # Load signal
            signal, fs = load_channel(
                case_id=case_id,
                channel=vitaldb_track,
                duration_sec=args.duration_sec,
                auto_fix_alternating=True
            )
            
            if signal is None or len(signal) < fs * 2:  # At least 2 seconds
                logger.debug(f"Case {case_id}: No valid signal")
                failed_cases.append((case_id, "No signal"))
                continue
            
            # Apply filter
            if 'filter' in ch_config:
                filt = ch_config['filter']
                
                # FIX 2: Handle different filter parameter names
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
                if sqi < args.min_sqi:
                    logger.debug(f"Case {case_id}: Low quality (SQI={sqi:.3f})")
                    failed_cases.append((case_id, f"Low SQI: {sqi:.3f}"))
                    continue
            
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
                # Save train stats immediately
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
                    # Try alternate location
                    stats_file = Path(args.outdir).parent / 'train_stats.npz'
                
                if stats_file.exists():
                    stats_data = np.load(stats_file)
                    from src.data.windows import NormalizationStats
                    train_stats = NormalizationStats(
                        mean=stats_data['mean'],
                        std=stats_data['std'],
                        method=str(stats_data['method']) if 'method' in stats_data else normalize_method
                    )
                    logger.info(f"Loaded train stats from {stats_file}")
                else:
                    logger.warning("Train stats not found, using local normalization")
            
            # Normalize windows
            if train_stats is not None:
                normalized = normalize_windows(
                    W_ntc=case_windows,
                    stats=train_stats,
                    baseline_correction=False,
                    per_channel=False
                )
            else:
                # Fallback to per-batch normalization
                normalized = case_windows
                mean = np.mean(normalized)
                std = np.std(normalized)
                if std > 1e-8:
                    normalized = (normalized - mean) / std
            
            # Store windows
            for w in normalized:
                all_windows.append(w)
                all_labels.append(0)  # Placeholder label
            
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
    """Train TTM model."""
    logger.info("Training TTM model...")
    
    # Load configurations
    model_config = load_config(args.model_yaml)
    run_config = load_config(args.run_yaml)
    
    # Set seed
    set_seed(run_config.get('seed', 42))
    
    # Load splits
    with open(args.split_file, 'r') as f:
        splits = json.load(f)
    
    # Load training data
    train_file = Path(args.outdir) / 'train_windows.npz'
    if not train_file.exists():
        # Try parent directory structure
        train_file = Path(args.outdir) / 'train' / 'train_windows.npz'
    if not train_file.exists():
        raise FileNotFoundError(f"Training data not found: {train_file}")
    
    train_data = np.load(train_file)
    
    # Load validation data if exists
    val_file = Path(args.outdir) / 'val_windows.npz'
    if not val_file.exists():
        val_file = Path(args.outdir) / 'val' / 'val_windows.npz'
    
    val_data = None
    if val_file.exists():
        val_data = np.load(val_file)
        logger.info(f"Loaded validation data: {val_data['data'].shape}")
    
    logger.info(f"Loaded training data: {train_data['data'].shape}")
    
    # Create datasets
    from torch.utils.data import TensorDataset, DataLoader
    
    X_train = torch.from_numpy(train_data['data']).float()
    y_train = torch.from_numpy(train_data['labels']).long()
    train_dataset = TensorDataset(X_train, y_train)
    
    val_dataset = None
    if val_data is not None:
        X_val = torch.from_numpy(val_data['data']).float()
        y_val = torch.from_numpy(val_data['labels']).long()
        val_dataset = TensorDataset(X_val, y_val)
    
    # Configure model
    model_config['input_channels'] = train_data['data'].shape[2]
    model_config['context_length'] = train_data['data'].shape[1]
    
    # Handle FastTrack mode
    if args.fasttrack:
        model_config['freeze_encoder'] = True
        model_config['head_type'] = 'linear'
        run_config['num_epochs'] = run_config.get('num_epochs', 10)
        logger.info("FastTrack mode: Frozen encoder, linear head, 10 epochs")
    
    # Create model
    model = create_ttm_model(model_config)
    
    # Create data loaders
    batch_size = run_config.get('batch_size', 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
    
    # Select trainer based on task
    task = model_config.get('task', 'classification')
    if task == 'classification':
        trainer = TrainerClf(
            model=model,
            lr=run_config.get('learning_rate', 1e-3),
            weight_decay=run_config.get('weight_decay', 1e-4),
            use_amp=run_config.get('use_amp', False),
            grad_clip=run_config.get('grad_clip', 1.0),
            patience=run_config.get('patience', 10),
            save_dir=Path(args.out),
            device=run_config.get('device', 'cpu')
        )
    else:
        trainer = TrainerReg(
            model=model,
            lr=run_config.get('learning_rate', 1e-3),
            weight_decay=run_config.get('weight_decay', 1e-4),
            use_amp=run_config.get('use_amp', False),
            grad_clip=run_config.get('grad_clip', 1.0),
            patience=run_config.get('patience', 10),
            save_dir=Path(args.out),
            device=run_config.get('device', 'cpu')
        )
    
    # Train
    num_epochs = run_config.get('num_epochs', 10)
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        save_best=True,
        log_interval=10
    )
    
    logger.info(f"Training complete! Model saved to {args.out}")


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
    y_test = torch.from_numpy(test_data['labels']).long()
    test_dataset = TensorDataset(X_test, y_test)
    
    # Create data loader
    batch_size = run_config.get('batch_size', 32)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Load model
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    
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
            if model_config.get('task', 'classification') == 'classification':
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
        from src.eval.metrics import compute_classification_metrics
        metrics = compute_classification_metrics(all_labels, all_preds)
        
        # Add calibration metrics
        if all_probs:
            all_probs = np.array(all_probs)
            ece = expected_calibration_error(all_labels, all_probs[:, 1] if all_probs.shape[1] == 2 else all_probs)
            metrics['ece'] = ece
    else:
        from src.eval.metrics import compute_regression_metrics
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
    
    if args.model:
        checkpoint = torch.load(args.model, map_location='cpu')
        print(f"\nModel file: {args.model}")
        print(f"Keys: {list(checkpoint.keys())}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="TTM × VitalDB Pipeline")
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(process)d:%(thread)d:%(name)s:%(funcName)s:%(message)s'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # prepare-splits command
    splits_parser = subparsers.add_parser('prepare-splits', help='Prepare train/val/test splits')
    splits_parser.add_argument('--mode', choices=['fasttrack', 'full'], default='fasttrack')
    splits_parser.add_argument('--case-set', type=str, help='VitalDB case set')
    splits_parser.add_argument('--output', type=str, default='data', help='Output directory')
    splits_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # build-windows command
    windows_parser = subparsers.add_parser('build-windows', help='Build preprocessed windows')
    windows_parser.add_argument('--channels-yaml', type=str, required=True, help='Channels config')
    windows_parser.add_argument('--windows-yaml', type=str, required=True, help='Windows config')
    windows_parser.add_argument('--split-file', type=str, required=True, help='Splits JSON file')
    windows_parser.add_argument('--split', type=str, required=True, help='Split to process')
    windows_parser.add_argument('--channel', type=str, help='Channel to process')
    windows_parser.add_argument('--duration-sec', type=float, default=60, help='Duration per case')
    windows_parser.add_argument('--min-sqi', type=float, default=0.5, help='Minimum SQI')
    windows_parser.add_argument('--outdir', type=str, required=True, help='Output directory')
    
    # train command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--model-yaml', type=str, required=True, help='Model config')
    train_parser.add_argument('--run-yaml', type=str, required=True, help='Run config')
    train_parser.add_argument('--split-file', type=str, required=True, help='Splits file')
    train_parser.add_argument('--outdir', type=str, required=True, help='Data directory')
    train_parser.add_argument('--out', type=str, required=True, help='Output directory')
    train_parser.add_argument('--fasttrack', action='store_true', help='FastTrack mode')
    
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
