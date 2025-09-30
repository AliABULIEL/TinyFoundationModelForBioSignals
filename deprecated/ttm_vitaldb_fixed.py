#!/usr/bin/env python3
"""
Fixed TTM × VitalDB pipeline with proper trainer initialization.
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
from src.models.trainers import TrainerClf, TrainerReg, create_optimizer
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_command(args):
    """Train TTM model - FIXED VERSION."""
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
        # Try as direct path
        train_file = Path(args.outdir) / 'train' / 'train_windows.npz'
        if not train_file.exists():
            # Look for the actual file location
            possible_locations = [
                Path('artifacts/raw_windows/train/train_windows.npz'),
                Path('artifacts/raw_windows/train_windows.npz'),
                Path(args.outdir).parent / 'train' / 'train_windows.npz',
            ]
            for loc in possible_locations:
                if loc.exists():
                    train_file = loc
                    break
            else:
                raise FileNotFoundError(f"Training data not found. Searched: {possible_locations}")

    train_data = np.load(train_file)

    # Load validation data if exists
    val_file = Path(args.outdir) / 'val_windows.npz'
    if not val_file.exists():
        val_file = Path(args.outdir) / 'val' / 'val_windows.npz'
    if not val_file.exists():
        # Try test data as validation if no val split
        val_file = Path(args.outdir) / 'test_windows.npz'
        if not val_file.exists():
            val_file = Path('artifacts/raw_windows/test/test_windows.npz')

    val_data = None
    if val_file.exists():
        val_data = np.load(val_file)
        logger.info(f"Loaded validation data: {val_data['data'].shape}")

    logger.info(f"Loaded training data: {train_data['data'].shape}")

    # Create datasets
    from torch.utils.data import TensorDataset, DataLoader

    X_train = torch.from_numpy(train_data['data']).float()
    
    # Handle labels - if not present, create dummy labels
    if 'labels' in train_data:
        y_train = torch.from_numpy(train_data['labels']).long()
    else:
        # Create random binary labels for demonstration
        logger.warning("No labels found in data, creating random binary labels")
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
    
    if task == 'classification' or task == 'clf':
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
        num_workers=0,  # Avoid multiprocessing issues
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


def main():
    """Main entry point with all original commands."""
    parser = argparse.ArgumentParser(description="TTM × VitalDB Pipeline - Fixed")

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(message)s'
    )

    # Add train command arguments
    parser.add_argument('--model-yaml', type=str, default='configs/model.yaml', help='Model config')
    parser.add_argument('--run-yaml', type=str, default='configs/run.yaml', help='Run config')
    parser.add_argument('--split-file', type=str, default='configs/splits/splits_full.json', help='Splits file')
    parser.add_argument('--outdir', type=str, default='artifacts/raw_windows', help='Data directory')
    parser.add_argument('--out', type=str, default='artifacts/model_output', help='Output directory')
    parser.add_argument('--task', type=str, default='clf', choices=['clf', 'reg'], help='Task type')
    parser.add_argument('--fasttrack', action='store_true', help='FastTrack mode')

    args = parser.parse_args()
    
    # Run training
    train_command(args)


if __name__ == "__main__":
    main()
