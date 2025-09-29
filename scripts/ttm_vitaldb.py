#!/usr/bin/env python3
"""
Main CLI entry point for TTM × VitalDB pipeline.

Commands:
    prepare-splits: Create train/val/test splits
    build-windows: Build preprocessed windows from VitalDB
    train: Train TTM model (FM or FT mode)
    test: Test model with overlap/context and calibration
    inspect: Inspect data and model (optional)
"""

import argparse
import json
import logging
import os
import sys
import warnings
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
    """Build preprocessed windows from VitalDB data."""
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
    
    # Get configuration
    channel_name = args.channel or list(channels_config.keys())[0]
    if channel_name not in channels_config:
        raise ValueError(f"Channel '{channel_name}' not in config")
    
    ch_config = channels_config[channel_name]
    
    # Window parameters
    window_s = windows_config.get('window_length_sec', 10.0)
    stride_s = windows_config.get('stride_sec', 10.0)
    min_cycles = windows_config.get('min_cycles', 3)
    normalize_method = windows_config.get('normalize_method', 'zscore')
    
    # Storage
    all_windows = []
    all_labels = []
    train_stats = None
    
    # Process cases
    logger.info(f"Processing {len(case_ids)} cases...")
    for case_id in tqdm(case_ids):
        try:
            # Load signal
            signal, fs = load_channel(
                case_id=case_id,
                channel=ch_config['vitaldb_track'],
                duration_sec=args.duration_sec,
                auto_fix_alternating=True
            )
            
            if signal is None or len(signal) < 100:
                logger.warning(f"Case {case_id}: Failed to load signal")
                continue
            
            # Apply filter
            if 'filter' in ch_config:
                filt = ch_config['filter']
                signal = apply_bandpass_filter(
                    signal, fs,
                    lowcut=filt['lowcut'],
                    highcut=filt['highcut'],
                    filter_type=filt.get('type', 'cheby2'),
                    order=filt.get('order', 4)
                )
            
            # Detect peaks
            signal_type = ch_config.get('type', 'ppg')
            if signal_type.lower() == 'ppg':
                peaks = find_ppg_peaks(signal, fs)
            elif signal_type.lower() == 'ecg':
                peaks, _ = find_ecg_rpeaks(signal, fs)
            else:
                peaks = np.array([])
            
            # Quality check
            if len(peaks) > 0:
                sqi = compute_sqi(signal, fs, peaks=peaks, signal_type=signal_type)
                if sqi < args.min_sqi:
                    logger.debug(f"Case {case_id}: Low quality (SQI={sqi:.3f})")
                    continue
            
            # Create windows
            signal_tc = signal.reshape(-1, 1)
            peaks_tc = {0: peaks} if len(peaks) > 0 else None
            
            case_windows = make_windows(
                X_tc=signal_tc,
                fs=fs,
                win_s=window_s,
                stride_s=stride_s,
                min_cycles=min_cycles,
                peaks_tc=peaks_tc
            )
            
            if case_windows is None or len(case_windows) == 0:
                logger.debug(f"Case {case_id}: No valid windows")
                continue
            
            # Compute normalization stats on first train case
            if args.split == 'train' and train_stats is None:
                train_stats = compute_normalization_stats(
                    X=case_windows,
                    method=normalize_method,
                    axis=(0, 1)
                )
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
            
            # Load train stats if processing val/test
            if args.split != 'train' and train_stats is None:
                stats_file = Path(args.outdir) / 'train_stats.npz'
                if stats_file.exists():
                    stats_data = np.load(stats_file)
                    from src.data.windows import NormalizationStats
                    train_stats = NormalizationStats(
                        mean=stats_data['mean'],
                        std=stats_data['std'],
                        method=str(stats_data['method'])
                    )
                    logger.info(f"Loaded train stats from {stats_file}")
            
            # Normalize
            if train_stats is not None:
                normalized = normalize_windows(
                    W_ntc=case_windows,
                    stats=train_stats,
                    baseline_correction=False,
                    per_channel=False
                )
            else:
                normalized = case_windows
            
            # Store windows
            for w in normalized:
                all_windows.append(w)
                all_labels.append(0)  # Placeholder label
            
        except Exception as e:
            logger.warning(f"Case {case_id}: Error - {e}")
            continue
    
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
        
        logger.info(f"Saved {len(all_windows)} windows to {output_file}")
        logger.info(f"  Shape: {windows_array.shape}")
    else:
        logger.error(f"No valid windows created for {args.split} split")


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
        raise FileNotFoundError(f"Training data not found: {train_file}")
    
    train_data = np.load(train_file)
    
    # Load validation data if exists
    val_file = Path(args.outdir) / 'val_windows.npz'
    val_data = None
    if val_file.exists():
        val_data = np.load(val_file)
        logger.info(f"Loaded validation data: {val_data['data'].shape}")
    
    logger.info(f"Loaded training data: {train_data['data'].shape}")
    
    # Create datasets
    from torch.utils.data import TensorDataset
    
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
    
    # Adjust for mode
    if args.fasttrack:
        model_config['freeze_encoder'] = True
        logger.info("FastTrack mode: Encoder frozen")
    
    # Create model
    model = create_ttm_model(model_config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Setup training
    train_config = {
        'num_epochs': run_config.get('num_epochs', 10),
        'batch_size': run_config.get('batch_size', 32),
        'learning_rate': run_config.get('learning_rate', 1e-3),
        'weight_decay': run_config.get('weight_decay', 1e-4),
        'use_amp': run_config.get('use_amp', False),
        'grad_clip': run_config.get('grad_clip', 1.0),
        'patience': run_config.get('patience', 10),
        'num_workers': run_config.get('num_workers', 0),
        'device': run_config.get('device', 'cpu')
    }
    
    # Create trainer
    task = model_config.get('task', 'classification')
    
    if task == 'classification':
        trainer = TrainerClf(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=train_config,
            save_dir=str(args.out)
        )
    else:
        trainer = TrainerReg(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=train_config,
            save_dir=str(args.out)
        )
    
    # Train
    logger.info("Starting training...")
    metrics = trainer.train()
    
    logger.info("Training completed!")
    logger.info(f"Final train loss: {metrics['train_loss'][-1]:.4f}")
    
    # Save metrics
    metrics_file = Path(args.out) / 'train_metrics.json'
    with open(metrics_file, 'w') as f:
        # Convert to serializable format
        serializable = {
            k: [float(x) for x in v] if isinstance(v, list) else float(v)
            for k, v in metrics.items()
        }
        json.dump(serializable, f, indent=2)
    
    logger.info(f"Metrics saved to {metrics_file}")


def test_command(args):
    """Test model."""
    logger.info("Testing model...")
    
    # Load configurations
    model_config = load_config(args.model_yaml)
    run_config = load_config(args.run_yaml)
    
    # Load test data
    test_file = Path(args.outdir) / 'test_windows.npz'
    if not test_file.exists():
        raise FileNotFoundError(f"Test data not found: {test_file}")
    
    test_data = np.load(test_file)
    logger.info(f"Loaded test data: {test_data['data'].shape}")
    
    # Create dataset
    from torch.utils.data import TensorDataset
    X_test = torch.from_numpy(test_data['data']).float()
    y_test = torch.from_numpy(test_data['labels']).long()
    test_dataset = TensorDataset(X_test, y_test)
    
    # Configure model
    model_config['input_channels'] = test_data['data'].shape[2]
    model_config['context_length'] = test_data['data'].shape[1]
    
    # Create model
    device = torch.device(run_config.get('device', 'cpu'))
    model = create_ttm_model(model_config)
    
    # Load checkpoint
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")
    
    # Create dataloader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=run_config.get('batch_size', 32),
        shuffle=False,
        num_workers=0
    )
    
    # Run inference
    all_preds = []
    all_labels = []
    all_probs = []
    
    logger.info("Running inference...")
    with torch.no_grad():
        for X, y in tqdm(test_loader):
            X, y = X.to(device), y.to(device)
            
            outputs = model(X)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Compute metrics
    task = model_config.get('task', 'classification')
    
    if task == 'classification':
        metrics = compute_classification_metrics(
            y_true=all_labels,
            y_pred=all_preds,
            y_prob=all_probs[:, 1] if all_probs.shape[1] > 1 else all_probs[:, 0]
        )
        
        logger.info("Test Results:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
        logger.info(f"  AUC-ROC:   {metrics['auroc']:.4f}")
    else:
        metrics = compute_regression_metrics(
            y_true=all_labels,
            y_pred=all_preds
        )
        
        logger.info("Test Results:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")
    
    # Save results
    results_file = Path(args.out) / 'test_results.json'
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        serializable = {k: float(v) for k, v in metrics.items()}
        json.dump(serializable, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")


def inspect_command(args):
    """Inspect data and model."""
    if args.data:
        logger.info(f"Inspecting data: {args.data}")
        data = np.load(args.data)
        
        logger.info(f"Keys: {list(data.keys())}")
        for key in data.keys():
            if isinstance(data[key], np.ndarray):
                logger.info(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}")
                if data[key].size > 0:
                    logger.info(f"    min={data[key].min():.3f}, max={data[key].max():.3f}, mean={data[key].mean():.3f}")
    
    if args.model:
        logger.info(f"Inspecting model: {args.model}")
        checkpoint = torch.load(args.model, map_location='cpu')
        
        logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            total_params = sum(p.numel() for p in state_dict.values())
            logger.info(f"Total parameters: {total_params:,}")
        
        if 'epoch' in checkpoint:
            logger.info(f"Epoch: {checkpoint['epoch']}")
        
        if 'best_val_loss' in checkpoint:
            logger.info(f"Best val loss: {checkpoint['best_val_loss']:.4f}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="TTM × VitalDB: Foundation Model for Biosignals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # prepare-splits command
    splits_parser = subparsers.add_parser(
        "prepare-splits", 
        help="Create train/val/test splits"
    )
    splits_parser.add_argument(
        "--mode", 
        type=str, 
        choices=["fasttrack", "full"], 
        default="full",
        help="Mode: fasttrack (70 cases) or full (all cases)"
    )
    splits_parser.add_argument(
        "--case-set",
        type=str,
        choices=['bis', 'desflurane', 'sevoflurane', 'remifentanil', 'propofol', 'tiva'],
        default='bis',
        help="VitalDB case set to use"
    )
    splits_parser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Output directory"
    )
    splits_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    # build-windows command
    windows_parser = subparsers.add_parser(
        "build-windows",
        help="Build preprocessed windows"
    )
    windows_parser.add_argument(
        "--channels-yaml",
        type=str,
        required=True,
        help="Path to channels config"
    )
    windows_parser.add_argument(
        "--windows-yaml",
        type=str,
        required=True,
        help="Path to windows config"
    )
    windows_parser.add_argument(
        "--split-file",
        type=str,
        required=True,
        help="Path to splits JSON"
    )
    windows_parser.add_argument(
        "--split",
        type=str,
        choices=['train', 'val', 'test'],
        required=True,
        help="Which split to process"
    )
    windows_parser.add_argument(
        "--channel",
        type=str,
        help="Channel to process (default: first in config)"
    )
    windows_parser.add_argument(
        "--duration-sec",
        type=int,
        default=60,
        help="Duration to load per case (seconds)"
    )
    windows_parser.add_argument(
        "--min-sqi",
        type=float,
        default=0.5,
        help="Minimum signal quality index"
    )
    windows_parser.add_argument(
        "--outdir",
        type=str,
        default="data",
        help="Output directory"
    )
    
    # train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train TTM model"
    )
    train_parser.add_argument(
        "--model-yaml",
        type=str,
        required=True,
        help="Path to model config"
    )
    train_parser.add_argument(
        "--run-yaml",
        type=str,
        required=True,
        help="Path to run config"
    )
    train_parser.add_argument(
        "--split-file",
        type=str,
        required=True,
        help="Path to splits JSON"
    )
    train_parser.add_argument(
        "--outdir",
        type=str,
        default="data",
        help="Data directory"
    )
    train_parser.add_argument(
        "--out",
        type=str,
        default="checkpoints",
        help="Output directory for model"
    )
    train_parser.add_argument(
        "--fasttrack",
        action='store_true',
        help="Use FastTrack mode (frozen encoder)"
    )
    
    # test command
    test_parser = subparsers.add_parser(
        "test",
        help="Test model"
    )
    test_parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Model checkpoint path"
    )
    test_parser.add_argument(
        "--model-yaml",
        type=str,
        required=True,
        help="Path to model config"
    )
    test_parser.add_argument(
        "--run-yaml",
        type=str,
        required=True,
        help="Path to run config"
    )
    test_parser.add_argument(
        "--split-file",
        type=str,
        required=True,
        help="Path to splits JSON"
    )
    test_parser.add_argument(
        "--outdir",
        type=str,
        default="data",
        help="Data directory"
    )
    test_parser.add_argument(
        "--out",
        type=str,
        default="results",
        help="Output directory for results"
    )
    
    # inspect command
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect data and model"
    )
    inspect_parser.add_argument(
        "--data",
        type=str,
        help="Data file to inspect"
    )
    inspect_parser.add_argument(
        "--model",
        type=str,
        help="Model checkpoint to inspect"
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Suppress warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Execute command
    if args.command == "prepare-splits":
        prepare_splits_command(args)
    elif args.command == "build-windows":
        build_windows_command(args)
    elif args.command == "train":
        train_command(args)
    elif args.command == "test":
        test_command(args)
    elif args.command == "inspect":
        inspect_command(args)


if __name__ == "__main__":
    main()
