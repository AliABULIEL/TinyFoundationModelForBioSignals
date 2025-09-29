#!/usr/bin/env python3
"""
Fixed TTM × VitalDB pipeline CLI.
Works with the actual codebase structure.
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
from scipy import signal as scipy_signal

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fixed imports that match your actual codebase
from src.data.splits import create_patient_splits, load_splits, save_splits
from src.data.vitaldb_loader import load_channel, list_cases, get_available_case_sets
from src.data.filters import apply_bandpass_filter
from src.data.detect import find_ecg_rpeaks, find_ppg_peaks
from src.data.quality import compute_sqi, compute_ssqi
from src.eval.calibration import (
    CalibrationEvaluator,
    TemperatureScaling,
    IsotonicCalibration
)
from src.eval.metrics import compute_classification_metrics, compute_regression_metrics
from src.models.datasets import VitalDBDataset  # Use your actual dataset
from src.models.ttm_adapter import TTMAdapter, create_ttm_model
from src.models.trainers import TrainerClf, TrainerReg
from src.utils.logging import setup_logger
from src.utils.paths import get_project_root
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
    
    # Get VitalDB case IDs using the fixed loader
    case_sets = get_available_case_sets()
    
    # Use BIS cases by default (high quality)
    if 'bis' in case_sets:
        all_case_ids = list(case_sets['bis'])
    else:
        # Fallback to any available cases
        all_case_ids = []
        for cases in case_sets.values():
            all_case_ids.extend(list(cases))
    
    # Filter based on mode
    if args.mode == "fasttrack":
        # Use first 70 cases for FastTrack mode
        case_ids = all_case_ids[:70]
        train_ratio = 50/70  # 50 train
        val_ratio = 0/70     # 0 val (skip for FastTrack)
        test_ratio = 20/70   # 20 test
    else:
        # Full mode uses more cases
        case_ids = all_case_ids[:500]  # Limit to 500 for manageable size
        train_ratio = args.train_ratio or 0.7
        val_ratio = args.val_ratio or 0.15
        test_ratio = 1 - train_ratio - val_ratio
    
    # Create splits
    splits = create_patient_splits(
        case_ids,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=args.seed or 42
    )
    
    # Save splits
    output_file = Path(args.out)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_splits(splits, output_file)
    
    logger.info(f"Splits saved to {output_file}")
    logger.info(f"Train: {len(splits['train'])} cases")
    logger.info(f"Val: {len(splits['val'])} cases")
    logger.info(f"Test: {len(splits['test'])} cases")


def build_windows_command(args):
    """Build preprocessed windows from VitalDB data."""
    logger.info("Building windows from VitalDB...")
    
    # Load configurations
    channels_config = load_config(args.channels_yaml)
    windows_config = load_config(args.windows_yaml)
    
    # Load splits
    splits = load_splits(args.split_file)
    
    # Get the requested split
    if args.split not in splits:
        raise ValueError(f"Split '{args.split}' not found in {args.split_file}")
    
    case_ids = splits[args.split]
    logger.info(f"Processing {len(case_ids)} cases from {args.split} split")
    
    # Output directory
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each case
    all_windows = []
    all_metadata = []
    window_size_sec = windows_config.get('window_size_sec', 10)
    hop_size_sec = windows_config.get('hop_size_sec', 5)
    
    for case_id in tqdm(case_ids, desc=f"Processing {args.split}"):
        try:
            # Load PPG/PLETH signal (primary modality)
            signal, fs = load_channel(
                case_id, 
                'PLETH',
                use_cache=True,
                duration_sec=None,  # Load all available
                auto_fix_alternating=True  # Use the fix we developed
            )
            
            if signal is None or len(signal) < fs * window_size_sec:
                continue
            
            # Resample to target rate if needed
            target_fs = channels_config.get('ppg', {}).get('sampling_rate', 125)
            if fs != target_fs:
                num_samples = int(len(signal) * target_fs / fs)
                signal = scipy_signal.resample(signal, num_samples)
                fs = target_fs
            
            # Apply bandpass filter
            if args.ecg_mode == 'analysis':  # Actually for PPG
                # Use 0.5-10 Hz for PPG
                filtered = apply_bandpass_filter(
                    signal,
                    low_freq=0.5,
                    high_freq=10,
                    fs=fs,
                    filter_type='butterworth',
                    order=4
                )
            else:
                filtered = signal
            
            # Normalize (z-score)
            mean = np.mean(filtered)
            std = np.std(filtered) + 1e-8
            normalized = (filtered - mean) / std
            
            # Create windows
            window_size = int(window_size_sec * fs)
            hop_size = int(hop_size_sec * fs)
            
            num_windows = (len(normalized) - window_size) // hop_size + 1
            
            for i in range(num_windows):
                start = i * hop_size
                end = start + window_size
                window = normalized[start:end]
                
                # Quality check (optional)
                if np.std(window) > 0.01:  # Has variation
                    all_windows.append(window)
                    all_metadata.append({
                        'case_id': str(case_id),
                        'window_idx': i,
                        'fs': fs
                    })
            
        except Exception as e:
            logger.warning(f"Error processing case {case_id}: {e}")
            continue
    
    # Save windows
    if all_windows:
        # Convert to array
        windows_array = np.array(all_windows)
        
        # Add channel dimension
        windows_array = np.expand_dims(windows_array, axis=1)  # (N, 1, T)
        
        # Save
        output_file = output_dir / f"{args.split}_windows.npz"
        np.savez_compressed(
            output_file,
            windows=windows_array,
            metadata=all_metadata,
            fs=target_fs,
            window_size_sec=window_size_sec
        )
        
        logger.info(f"Saved {len(all_windows)} windows to {output_file}")
        logger.info(f"Shape: {windows_array.shape}")
    else:
        logger.warning(f"No valid windows found for {args.split} split")


def train_command(args):
    """Train TTM model."""
    logger.info("Training TTM model...")
    
    # Load configurations
    model_config = load_config(args.model_yaml)
    run_config = load_config(args.run_yaml)
    
    # Set seed
    set_seed(run_config.get('seed', 42))
    
    # Load preprocessed windows
    train_file = Path(args.out) / "train_windows.npz"
    if not train_file.exists():
        raise FileNotFoundError(f"Training data not found: {train_file}")
    
    train_data = np.load(train_file)
    train_windows = train_data['windows']
    
    # Create simple dataset
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, windows, labels=None):
            self.windows = torch.FloatTensor(windows)
            # Create dummy labels if not provided
            self.labels = torch.zeros(len(windows)) if labels is None else torch.FloatTensor(labels)
        
        def __len__(self):
            return len(self.windows)
        
        def __getitem__(self, idx):
            return self.windows[idx], self.labels[idx]
    
    train_dataset = SimpleDataset(train_windows)
    
    # Create data loader
    batch_size = run_config.get('batch_size', 32)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=run_config.get('num_workers', 4),
        pin_memory=True
    )
    
    # Configure model
    if args.fasttrack:
        # FastTrack: frozen encoder
        model_config['freeze_encoder'] = True
        model_config['unfreeze_last_n_blocks'] = 0
    
    # Set dimensions based on data
    model_config['input_channels'] = train_windows.shape[1]  # Should be 1 for PPG
    model_config['context_length'] = train_windows.shape[2]  # Window size
    
    # Create model
    model = create_ttm_model(model_config)
    
    # Simple training loop for demonstration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=run_config.get('learning_rate', 1e-4),
        weight_decay=run_config.get('weight_decay', 0.01)
    )
    
    # Training loop
    num_epochs = run_config.get('num_epochs', 10)
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Simple loss (MSE for now)
            if outputs.dim() > 2:
                outputs = outputs.mean(dim=-1)  # Average over time if needed
            
            loss = torch.nn.functional.mse_loss(outputs.squeeze(), targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
    
    # Save model
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': num_epochs,
        'model_config': model_config
    }
    
    model_path = output_dir / "model.pt"
    torch.save(checkpoint, model_path)
    logger.info(f"Model saved to {model_path}")


def test_command(args):
    """Test model."""
    logger.info("Testing model...")
    
    # Load test data
    test_file = Path(args.out) / "test_windows.npz"
    if not test_file.exists():
        raise FileNotFoundError(f"Test data not found: {test_file}")
    
    test_data = np.load(test_file)
    test_windows = test_data['windows']
    
    # Load model
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    model_config = checkpoint.get('model_config', {})
    
    # Create model
    model = create_ttm_model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Simple inference
    test_tensor = torch.FloatTensor(test_windows).to(device)
    
    with torch.no_grad():
        outputs = model(test_tensor)
    
    logger.info(f"Test completed. Output shape: {outputs.shape}")
    
    # Save results
    results = {
        'num_samples': len(test_windows),
        'output_shape': list(outputs.shape),
        'output_mean': float(outputs.mean().cpu()),
        'output_std': float(outputs.std().cpu())
    }
    
    results_file = Path(args.out) / "test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="TTM × VitalDB Pipeline (Fixed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # prepare-splits command
    splits_parser = subparsers.add_parser("prepare-splits", help="Create train/val/test splits")
    splits_parser.add_argument("--train-ratio", type=float, default=0.7)
    splits_parser.add_argument("--val-ratio", type=float, default=0.15)
    splits_parser.add_argument("--test-ratio", type=float, default=0.15)
    splits_parser.add_argument("--seed", type=int, default=42)
    splits_parser.add_argument("--out", type=str, default="configs/splits/train_test.json")
    splits_parser.add_argument("--fasttrack", action="store_true", help="Use FastTrack mode")
    splits_parser.add_argument("--mode", type=str, choices=["fasttrack", "full"], default="full")
    
    # build-windows command
    windows_parser = subparsers.add_parser("build-windows", help="Build preprocessed windows")
    windows_parser.add_argument("--channels-yaml", type=str, default="configs/channels.yaml")
    windows_parser.add_argument("--windows-yaml", type=str, default="configs/windows.yaml")
    windows_parser.add_argument("--split-file", type=str, default="configs/splits/train_test.json")
    windows_parser.add_argument("--split", type=str, choices=["train", "val", "test"], required=True)
    windows_parser.add_argument("--outdir", type=str, required=True)
    windows_parser.add_argument("--ecg-mode", type=str, default="analysis")
    windows_parser.add_argument("--fasttrack", action="store_true")
    
    # train command
    train_parser = subparsers.add_parser("train", help="Train TTM model")
    train_parser.add_argument("--model-yaml", type=str, default="configs/model.yaml")
    train_parser.add_argument("--run-yaml", type=str, default="configs/run.yaml")
    train_parser.add_argument("--split-file", type=str, default="configs/splits/train_test.json")
    train_parser.add_argument("--task", type=str, default="clf", choices=["clf", "reg"])
    train_parser.add_argument("--out", type=str, required=True)
    train_parser.add_argument("--fasttrack", action="store_true")
    
    # test command
    test_parser = subparsers.add_parser("test", help="Test model")
    test_parser.add_argument("--model-yaml", type=str, default="configs/model.yaml")
    test_parser.add_argument("--run-yaml", type=str, default="configs/run.yaml")
    test_parser.add_argument("--split-file", type=str, default="configs/splits/train_test.json")
    test_parser.add_argument("--split", type=str, default="test")
    test_parser.add_argument("--task", type=str, default="clf")
    test_parser.add_argument("--ckpt", type=str, required=True)
    test_parser.add_argument("--out", type=str, required=True)
    test_parser.add_argument("--calibration", type=str, default="temperature")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    # Execute command
    if args.command == "prepare-splits":
        prepare_splits_command(args)
    elif args.command == "build-windows":
        build_windows_command(args)
    elif args.command == "train":
        train_command(args)
    elif args.command == "test":
        test_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
