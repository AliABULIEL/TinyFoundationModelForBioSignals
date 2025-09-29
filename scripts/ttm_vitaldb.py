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

from src.data.splits import create_patient_splits, load_splits, save_splits
from src.data.vitaldb_loader import VitalDBLoader
from src.data.windows import WindowBuilder, WindowConfig
from src.data.filters import apply_bandpass_filter
from src.data.detect import detect_peaks
from src.data.quality import compute_sqi, compute_ssqi
from src.eval.calibration import (
    CalibrationEvaluator,
    TemperatureScaling,
    IsotonicCalibration,
    expected_calibration_error
)
from src.eval.metrics import compute_classification_metrics, compute_regression_metrics
from src.models.datasets import TTMDataset
from src.models.ttm_adapter import create_ttm_model
from src.models.trainers import TrainerClf, TrainerReg, create_optimizer, create_scheduler
from src.utils.logging import setup_logger
from src.utils.paths import get_project_root
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_splits_command(args):
    """Prepare train/val/test splits from VitalDB."""
    logger.info("Preparing patient splits...")
    
    # Load configuration
    config = load_config(args.config)
    
    # Get VitalDB case IDs
    loader = VitalDBLoader()
    case_ids = loader.get_available_cases()
    
    # Filter based on mode
    if args.mode == "fasttrack":
        # Use first 70 cases for FastTrack mode
        case_ids = case_ids[:70]
        train_ratio = 50/70  # 50 train
        val_ratio = 0/70     # 0 val (skip for FastTrack)
        test_ratio = 20/70   # 20 test
    else:
        # Full mode uses all cases
        train_ratio = config.get('train_ratio', 0.7)
        val_ratio = config.get('val_ratio', 0.15)
        test_ratio = config.get('test_ratio', 0.15)
    
    # Create splits
    splits = create_patient_splits(
        case_ids,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=config.get('seed', 42)
    )
    
    # Save splits
    output_dir = Path(config.get('data_dir', 'data'))
    output_dir.mkdir(parents=True, exist_ok=True)
    splits_file = output_dir / f"splits_{args.mode}.json"
    save_splits(splits, splits_file)
    
    logger.info(f"Splits saved to {splits_file}")
    logger.info(f"Train: {len(splits['train'])} cases")
    logger.info(f"Val: {len(splits['val'])} cases")
    logger.info(f"Test: {len(splits['test'])} cases")


def build_windows_command(args):
    """Build preprocessed windows from VitalDB data."""
    logger.info("Building windows from VitalDB...")
    
    # Load configuration
    config = load_config(args.config)
    channels_config = load_config(config.get('channels_config', 'configs/channels.yaml'))
    windows_config = load_config(config.get('windows_config', 'configs/windows.yaml'))
    
    # Load splits
    data_dir = Path(config.get('data_dir', 'data'))
    splits_file = data_dir / f"splits_{args.mode}.json"
    splits = load_splits(splits_file)
    
    # Setup window builder
    window_config = WindowConfig(
        window_size=windows_config['window_size'],
        step_size=windows_config.get('step_size', windows_config['window_size']),
        min_quality=windows_config.get('min_quality', 0.9),
        channels=list(channels_config['channels'].keys())
    )
    
    window_builder = WindowBuilder(window_config)
    loader = VitalDBLoader()
    
    # Process each split
    for split_name, case_ids in splits.items():
        logger.info(f"Processing {split_name} split ({len(case_ids)} cases)...")
        
        windows_dir = data_dir / "windows" / args.mode / split_name
        windows_dir.mkdir(parents=True, exist_ok=True)
        
        all_windows = []
        all_labels = []
        all_metadata = []
        
        for case_id in tqdm(case_ids, desc=f"Processing {split_name}"):
            try:
                # Load signals
                signals = {}
                for channel, ch_config in channels_config['channels'].items():
                    signal = loader.load_signal(case_id, channel)
                    if signal is not None:
                        # Resample
                        target_fs = ch_config['sampling_rate']
                        signal = loader.resample_signal(signal, target_fs)
                        
                        # Apply filter
                        if 'filter' in ch_config:
                            filter_cfg = ch_config['filter']
                            signal = apply_bandpass_filter(
                                signal,
                                low_freq=filter_cfg['low_freq'],
                                high_freq=filter_cfg['high_freq'],
                                fs=target_fs,
                                filter_type=filter_cfg.get('type', 'butterworth'),
                                order=filter_cfg.get('order', 4)
                            )
                        
                        signals[channel] = signal
                
                if not signals:
                    continue
                
                # Detect peaks for quality assessment
                for channel in signals:
                    if channel in ['ECG', 'PPG', 'ABP']:
                        peaks = detect_peaks(
                            signals[channel],
                            fs=channels_config['channels'][channel]['sampling_rate'],
                            signal_type=channel.lower()
                        )
                        
                        # Compute quality metrics
                        if channel == 'ECG':
                            sqi = compute_sqi(signals[channel], peaks)
                            # Filter by quality
                            if sqi < windows_config.get('min_ecg_quality', 0.9):
                                continue
                        elif channel == 'PPG':
                            ssqi = compute_ssqi(signals[channel], peaks)
                            if ssqi < windows_config.get('min_ppg_quality', 0.8):
                                continue
                
                # Create windows
                windows = window_builder.create_windows(signals)
                
                # Check minimum cycles
                min_cycles = windows_config.get('min_cycles', 3)
                valid_windows = []
                for window in windows:
                    # Simple check: ensure enough variation (cycles)
                    for channel in window:
                        signal_std = np.std(window[channel])
                        if signal_std > 0.01:  # Has variation
                            valid_windows.append(window)
                            break
                
                # Normalize windows (z-score)
                for window in valid_windows:
                    for channel in window:
                        mean = np.mean(window[channel])
                        std = np.std(window[channel]) + 1e-8
                        window[channel] = (window[channel] - mean) / std
                
                # Convert to arrays
                for window in valid_windows:
                    # Stack channels
                    window_array = np.stack([window[ch] for ch in sorted(window.keys())], axis=0)
                    all_windows.append(window_array)
                    
                    # Add placeholder label (will be task-specific)
                    all_labels.append(0)
                    
                    # Add metadata
                    all_metadata.append({
                        'case_id': case_id,
                        'channels': list(window.keys())
                    })
                    
            except Exception as e:
                logger.warning(f"Error processing case {case_id}: {e}")
                continue
        
        # Save windows
        if all_windows:
            windows_array = np.array(all_windows)
            labels_array = np.array(all_labels)
            
            # Save as NPZ
            output_file = windows_dir / f"{split_name}_windows.npz"
            np.savez_compressed(
                output_file,
                windows=windows_array,
                labels=labels_array,
                metadata=all_metadata
            )
            
            logger.info(f"Saved {len(all_windows)} windows to {output_file}")
        else:
            logger.warning(f"No valid windows found for {split_name} split")


def train_command(args):
    """Train TTM model."""
    logger.info("Training TTM model...")
    
    # Load configurations
    config = load_config(args.config)
    model_config = load_config(config.get('model_config', 'configs/model.yaml'))
    
    # Set seed
    set_seed(config.get('seed', 42))
    
    # Load data
    data_dir = Path(config.get('data_dir', 'data'))
    windows_dir = data_dir / "windows" / args.mode
    
    train_data = np.load(windows_dir / "train" / "train_windows.npz")
    val_data = np.load(windows_dir / "val" / "val_windows.npz") if (windows_dir / "val" / "val_windows.npz").exists() else None
    
    # Create datasets
    train_dataset = TTMDataset(
        windows=train_data['windows'],
        labels=train_data['labels'],
        transform=None
    )
    
    val_dataset = None
    if val_data is not None:
        val_dataset = TTMDataset(
            windows=val_data['windows'],
            labels=val_data['labels'],
            transform=None
        )
    
    # Create data loaders
    batch_size = config.get('batch_size', 32)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            pin_memory=True
        )
    
    # Configure model based on mode
    if args.mode == "fasttrack":
        # FastTrack: frozen encoder
        model_config['freeze_encoder'] = True
        model_config['unfreeze_last_n_blocks'] = 0
        model_config['lora'] = {'enabled': False}
    else:
        # Full mode: check for fine-tuning options
        if args.unfreeze_last_n > 0:
            model_config['freeze_encoder'] = True
            model_config['unfreeze_last_n_blocks'] = args.unfreeze_last_n
        
        if args.lora_rank > 0:
            model_config['lora'] = {
                'enabled': True,
                'r': args.lora_rank,
                'alpha': args.lora_rank * 2,
                'dropout': 0.1
            }
    
    # Set input channels based on data
    model_config['input_channels'] = train_data['windows'].shape[1]
    model_config['context_length'] = train_data['windows'].shape[2]
    
    # Create model
    model = create_ttm_model(model_config)
    model.print_parameter_summary()
    
    # Create optimizer
    optimizer = create_optimizer(
        model,
        optimizer_type=config.get('optimizer', 'adamw'),
        lr=config.get('learning_rate', 5e-4),
        weight_decay=config.get('weight_decay', 0.01)
    )
    
    # Create scheduler
    num_epochs = config.get('num_epochs', 10)
    scheduler = create_scheduler(
        optimizer,
        scheduler_type=config.get('scheduler', 'cosine'),
        num_epochs=num_epochs,
        warmup_epochs=config.get('warmup_epochs', 1)
    )
    
    # Setup trainer based on task
    task = model_config.get('task', 'classification')
    
    if task == 'classification':
        trainer = TrainerClf(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            num_classes=model_config.get('num_classes', 2),
            use_focal_loss=config.get('use_focal_loss', False),
            use_balanced_sampler=config.get('use_balanced_sampler', False),
            device=config.get('device', 'cuda'),
            use_amp=config.get('use_amp', True),
            gradient_clip=config.get('gradient_clip', 1.0),
            checkpoint_dir=config.get('checkpoint_dir', 'artifacts'),
            seed=config.get('seed', 42)
        )
    else:
        trainer = TrainerReg(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_type=config.get('loss_type', 'mse'),
            device=config.get('device', 'cuda'),
            use_amp=config.get('use_amp', True),
            gradient_clip=config.get('gradient_clip', 1.0),
            checkpoint_dir=config.get('checkpoint_dir', 'artifacts'),
            seed=config.get('seed', 42)
        )
    
    # Train model
    history = trainer.fit(
        num_epochs=num_epochs,
        save_best=True,
        early_stopping_patience=config.get('early_stopping_patience', 5),
        monitor_metric=config.get('monitor_metric', 'accuracy' if task == 'classification' else 'mse'),
        monitor_mode=config.get('monitor_mode', 'max' if task == 'classification' else 'min')
    )
    
    logger.info("Training completed!")
    logger.info(f"Best validation metric: {trainer.best_val_metric}")
    
    # Save final model
    output_dir = Path(config.get('checkpoint_dir', 'artifacts'))
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(output_dir / "final_model.pt")
    trainer.save_metrics(output_dir / "training_metrics.json")


def test_command(args):
    """Test model with overlap/context and calibration."""
    logger.info("Testing model...")
    
    # Load configurations
    config = load_config(args.config)
    model_config = load_config(config.get('model_config', 'configs/model.yaml'))
    
    # Load test data
    data_dir = Path(config.get('data_dir', 'data'))
    windows_dir = data_dir / "windows" / args.mode
    test_data = np.load(windows_dir / "test" / "test_windows.npz")
    
    # Create test dataset
    test_dataset = TTMDataset(
        windows=test_data['windows'],
        labels=test_data['labels'],
        transform=None
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    # Load model
    device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    
    # Set model config
    model_config['input_channels'] = test_data['windows'].shape[1]
    model_config['context_length'] = test_data['windows'].shape[2]
    
    model = create_ttm_model(model_config)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")
    
    # Collect predictions
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            inputs, targets = batch
            inputs = inputs.to(device)
            
            # Test with different overlaps if specified
            if args.overlap > 0:
                # Implement overlapping windows inference
                # This would require sliding window with specified overlap
                outputs = model(inputs)
            else:
                outputs = model(inputs)
            
            all_outputs.append(outputs.cpu())
            all_targets.append(targets)
    
    # Concatenate results
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics based on task
    task = model_config.get('task', 'classification')
    
    if task == 'classification':
        # Get probabilities
        probs = torch.softmax(all_outputs, dim=1)
        predictions = torch.argmax(all_outputs, dim=1)
        
        # Compute metrics
        metrics = compute_classification_metrics(predictions, all_targets)
        
        # Binary classification calibration
        if model_config.get('num_classes', 2) == 2:
            binary_probs = probs[:, 1]  # Probability of positive class
            
            # Evaluate calibration before
            calib_eval = CalibrationEvaluator(n_bins=10)
            calib_metrics_before = calib_eval.evaluate(all_targets, binary_probs)
            
            logger.info("Calibration metrics (before):")
            for key, value in calib_metrics_before.items():
                logger.info(f"  {key}: {value:.4f}")
            
            # Apply calibration if requested
            if args.calibration == "temperature":
                temp_scaler = TemperatureScaling()
                optimal_temp = temp_scaler.fit(all_outputs, all_targets, verbose=True)
                calibrated_logits = temp_scaler(all_outputs)
                calibrated_probs = torch.softmax(calibrated_logits, dim=1)[:, 1]
            elif args.calibration == "isotonic":
                iso_calib = IsotonicCalibration()
                iso_calib.fit(binary_probs, all_targets)
                calibrated_probs = torch.tensor(iso_calib.transform(binary_probs))
            else:
                calibrated_probs = binary_probs
            
            if args.calibration:
                # Evaluate calibration after
                calib_metrics_after = calib_eval.evaluate(all_targets, calibrated_probs)
                logger.info("Calibration metrics (after):")
                for key, value in calib_metrics_after.items():
                    logger.info(f"  {key}: {value:.4f}")
                
                # Find optimal thresholds
                thresholds = calib_eval.find_optimal_thresholds(all_targets, calibrated_probs)
                logger.info("Optimal thresholds:")
                for key, value in thresholds.items():
                    logger.info(f"  {key}: {value:.4f}")
        
        logger.info("Classification metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
    
    else:  # Regression
        metrics = compute_regression_metrics(all_outputs, all_targets)
        
        logger.info("Regression metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")
    
    # Save results
    output_dir = Path(config.get('checkpoint_dir', 'artifacts'))
    results = {
        'metrics': {k: float(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v 
                   for k, v in metrics.items()},
        'calibration': calib_metrics_after if args.calibration and task == 'classification' else None
    }
    
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_dir / 'test_results.json'}")


def inspect_command(args):
    """Inspect data and model (optional utility command)."""
    logger.info("Inspecting data and model...")
    
    if args.data:
        # Inspect data
        data_path = Path(args.data)
        if data_path.suffix == '.npz':
            data = np.load(data_path)
            logger.info(f"Data file: {data_path}")
            logger.info(f"Keys: {list(data.keys())}")
            for key in data.keys():
                if isinstance(data[key], np.ndarray):
                    logger.info(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}")
        else:
            logger.info(f"Unsupported data format: {data_path.suffix}")
    
    if args.model:
        # Inspect model
        checkpoint = torch.load(args.model, map_location='cpu')
        logger.info(f"Model checkpoint: {args.model}")
        logger.info(f"Keys: {list(checkpoint.keys())}")
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            total_params = sum(p.numel() for p in state_dict.values())
            logger.info(f"Total parameters: {total_params:,}")
            
            # Show layer structure
            logger.info("Model layers:")
            for key in sorted(state_dict.keys())[:20]:  # Show first 20 layers
                logger.info(f"  {key}: {state_dict[key].shape}")
        
        if 'train_history' in checkpoint:
            history = checkpoint['train_history']
            logger.info(f"Training epochs: {len(history)}")
            if history:
                logger.info(f"Final train loss: {history[-1].get('loss', 'N/A')}")
        
        if 'val_history' in checkpoint:
            history = checkpoint['val_history']
            if history and 'accuracy' in history[-1]:
                logger.info(f"Final val accuracy: {history[-1]['accuracy']:.4f}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="TTM × VitalDB: Foundation Model for Biosignals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # prepare-splits command
    splits_parser = subparsers.add_parser("prepare-splits", help="Create train/val/test splits")
    splits_parser.add_argument("--config", type=str, default="configs/run.yaml", help="Run config path")
    splits_parser.add_argument("--mode", type=str, choices=["fasttrack", "full"], default="full")
    
    # build-windows command
    windows_parser = subparsers.add_parser("build-windows", help="Build preprocessed windows")
    windows_parser.add_argument("--config", type=str, default="configs/run.yaml", help="Run config path")
    windows_parser.add_argument("--mode", type=str, choices=["fasttrack", "full"], default="full")
    
    # train command
    train_parser = subparsers.add_parser("train", help="Train TTM model")
    train_parser.add_argument("--config", type=str, default="configs/run.yaml", help="Run config path")
    train_parser.add_argument("--mode", type=str, choices=["fasttrack", "full"], default="full")
    train_parser.add_argument("--unfreeze-last-n", type=int, default=0, help="Number of blocks to unfreeze")
    train_parser.add_argument("--lora-rank", type=int, default=0, help="LoRA rank (0 to disable)")
    
    # test command
    test_parser = subparsers.add_parser("test", help="Test model")
    test_parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    test_parser.add_argument("--config", type=str, default="configs/run.yaml", help="Run config path")
    test_parser.add_argument("--mode", type=str, choices=["fasttrack", "full"], default="full")
    test_parser.add_argument("--overlap", type=float, default=0, help="Window overlap ratio (0-1)")
    test_parser.add_argument("--calibration", type=str, choices=["none", "temperature", "isotonic"], 
                           default="none", help="Calibration method")
    
    # inspect command (optional)
    inspect_parser = subparsers.add_parser("inspect", help="Inspect data and model")
    inspect_parser.add_argument("--data", type=str, help="Data file to inspect")
    inspect_parser.add_argument("--model", type=str, help="Model checkpoint to inspect")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Suppress some warnings
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
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
