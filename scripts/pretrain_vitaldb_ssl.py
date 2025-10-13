#!/usr/bin/env python3
"""SSL Pretraining Script for VitalDB.

This script implements self-supervised learning (SSL) pretraining using masked
signal modeling on VitalDB biosignals (PPG + ECG).

Strategy:
1. Load VitalDB data with 2 channels (PPG, ECG)
2. Apply random/block masking (40% mask ratio)
3. Train encoder to reconstruct masked regions
4. Use MSM + Multi-Resolution STFT losses
5. Save pretrained encoder for downstream tasks

Usage:
    python scripts/pretrain_vitaldb_ssl.py \\
        --config configs/ssl_pretrain.yaml \\
        --data-dir data/vitaldb_windows \\
        --output-dir artifacts/foundation_model \\
        --epochs 100 \\
        --batch-size 128

For quick testing:
    python scripts/pretrain_vitaldb_ssl.py \\
        --config configs/ssl_pretrain.yaml \\
        --data-dir data/vitaldb_windows \\
        --output-dir artifacts/ssl_test \\
        --epochs 5 \\
        --batch-size 32 \\
        --fast
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import yaml
import json
from typing import Dict, List, Tuple
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from src.models.ttm_adapter import create_ttm_model
from src.models.decoders import ReconstructionHead1D
from src.ssl.objectives import MaskedSignalModeling, MultiResolutionSTFT
from src.ssl.masking import random_masking, block_masking
from src.ssl.pretrainer import SSLTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SSL Pretraining on VitalDB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config and paths
    parser.add_argument(
        '--config',
        type=str,
        default='configs/ssl_pretrain.yaml',
        help='Path to SSL config YAML file'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing VitalDB window files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='artifacts/foundation_model',
        help='Output directory for checkpoints'
    )
    
    # Data configuration
    parser.add_argument(
        '--channels',
        nargs='+',
        default=['PPG', 'ECG'],
        help='Channels to use for pretraining'
    )
    parser.add_argument(
        '--train-file',
        type=str,
        default='train_windows.npz',
        help='Training data filename'
    )
    parser.add_argument(
        '--val-file',
        type=str,
        default='val_windows.npz',
        help='Validation data filename'
    )
    
    # Training hyperparameters
    parser.add_argument(
        '--mask-ratio',
        type=float,
        default=0.4,
        help='Masking ratio for SSL'
    )
    parser.add_argument(
        '--mask-type',
        type=str,
        default='random',
        choices=['random', 'block'],
        help='Masking strategy'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size for training'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=5e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.01,
        help='Weight decay'
    )
    parser.add_argument(
        '--warmup-epochs',
        type=int,
        default=10,
        help='Number of warmup epochs'
    )
    
    # Model configuration
    parser.add_argument(
        '--context-length',
        type=int,
        default=1250,
        help='Context length (10s at 125Hz)'
    )
    parser.add_argument(
        '--patch-size',
        type=int,
        default=125,
        help='Patch size in samples (1s at 125Hz)'
    )
    
    # Training options
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--no-amp',
        action='store_true',
        help='Disable automatic mixed precision'
    )
    parser.add_argument(
        '--gradient-clip',
        type=float,
        default=1.0,
        help='Gradient clipping value'
    )
    parser.add_argument(
        '--stft-weight',
        type=float,
        default=0.3,
        help='Weight for STFT loss'
    )
    
    # Logging and checkpointing
    parser.add_argument(
        '--log-interval',
        type=int,
        default=50,
        help='Log every N batches'
    )
    parser.add_argument(
        '--save-interval',
        type=int,
        default=10,
        help='Save checkpoint every N epochs'
    )
    
    # Fast mode for testing
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Fast mode: use small subset of data for testing'
    )
    
    # Random seed
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        config: Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        warnings.warn(f"Config file not found: {config_path}. Using defaults.")
        return {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def create_vitaldb_dataloaders(
    data_dir: str,
    train_file: str,
    val_file: str,
    channels: List[str],
    batch_size: int,
    num_workers: int = 4,
    fast_mode: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """Create VitalDB dataloaders for SSL pretraining.
    
    Args:
        data_dir: Directory containing window files
        train_file: Training data filename (e.g., 'train_windows.npz')
        val_file: Validation data filename
        channels: List of channel names
        batch_size: Batch size
        num_workers: Number of workers for data loading
        fast_mode: If True, use small subset for testing
    
    Returns:
        train_loader, val_loader: DataLoader objects
    """
    data_dir = Path(data_dir)
    
    # Load training data
    train_path = data_dir / train_file
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    
    train_data = np.load(train_path)
    
    # Load validation data
    val_path = data_dir / val_file
    if not val_path.exists():
        warnings.warn(f"Validation data not found: {val_path}. Using 10% of training data.")
        # Use 10% of training data for validation
        n_train = int(0.9 * len(train_data['data']))
        val_windows = train_data['data'][n_train:]
        train_windows = train_data['data'][:n_train]
    else:
        val_data = np.load(val_path)
        train_windows = train_data['data']
        val_windows = val_data['data']
    
    # Fast mode: use small subset
    if fast_mode:
        max_train = min(1000, len(train_windows))
        max_val = min(200, len(val_windows))
        train_windows = train_windows[:max_train]
        val_windows = val_windows[:max_val]
        print(f"Fast mode: Using {len(train_windows)} train, {len(val_windows)} val samples")
    
    # Convert to tensors
    # Expected shape: [N, C, T]
    train_tensor = torch.from_numpy(train_windows).float()
    val_tensor = torch.from_numpy(val_windows).float()
    
    print(f"Training data shape: {train_tensor.shape}")
    print(f"Validation data shape: {val_tensor.shape}")
    
    # Create datasets
    train_dataset = TensorDataset(train_tensor)
    val_dataset = TensorDataset(val_tensor)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader


def build_model(
    n_channels: int,
    context_length: int,
    patch_size: int,
    device: str
) -> Tuple[nn.Module, nn.Module]:
    """Build encoder and decoder for SSL pretraining.
    
    Args:
        n_channels: Number of input channels
        context_length: Input sequence length
        patch_size: Patch size for masking
        device: Device to use
    
    Returns:
        encoder, decoder: Encoder and decoder modules
    """
    # Build encoder (TTM)
    encoder_config = {
        'variant': 'ibm-granite/granite-timeseries-ttm-r1',
        'task': 'ssl',  # SSL task - no task head
        'input_channels': n_channels,
        'context_length': context_length,
        'patch_size': patch_size,
        'freeze_encoder': False  # Training from scratch or fine-tuning
    }
    
    print("\nBuilding encoder...")
    encoder = create_ttm_model(encoder_config)
    
    # Build decoder (Reconstruction Head)
    print("Building decoder...")
    decoder = ReconstructionHead1D(
        d_model=192,  # TTM hidden dimension
        patch_size=patch_size,
        n_channels=n_channels
    )
    
    # Print model summary
    encoder_params = sum(p.numel() for p in encoder.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())
    
    print(f"\nModel Summary:")
    print(f"  Encoder parameters: {encoder_params:,}")
    print(f"  Decoder parameters: {decoder_params:,}")
    print(f"  Total parameters: {encoder_params + decoder_params:,}")
    
    return encoder, decoder


def create_optimizer_scheduler(
    encoder: nn.Module,
    decoder: nn.Module,
    lr: float,
    weight_decay: float,
    warmup_epochs: int,
    total_epochs: int,
    steps_per_epoch: int
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Create optimizer and learning rate scheduler.
    
    Args:
        encoder: Encoder model
        decoder: Decoder model
        lr: Learning rate
        weight_decay: Weight decay
        warmup_epochs: Number of warmup epochs
        total_epochs: Total number of epochs
        steps_per_epoch: Steps per epoch
    
    Returns:
        optimizer, scheduler: Optimizer and LR scheduler
    """
    # Optimizer
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Cosine scheduler with warmup
    def lr_lambda(current_step: int):
        warmup_steps = warmup_epochs * steps_per_epoch
        total_steps = total_epochs * steps_per_epoch
        
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Print configuration
    print("=" * 70)
    print("SSL PRETRAINING ON VITALDB")
    print("=" * 70)
    print(f"Config file: {args.config}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Channels: {args.channels}")
    print(f"Mask ratio: {args.mask_ratio}")
    print(f"Mask type: {args.mask_type}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    print(f"Fast mode: {args.fast}")
    print("=" * 70)
    
    # Load config (can override CLI args)
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_save = vars(args)
    with open(output_dir / 'training_config.json', 'w') as f:
        json.dump(config_save, f, indent=2)
    
    # Step 1: Create dataloaders
    print("\n1. Loading VitalDB data...")
    train_loader, val_loader = create_vitaldb_dataloaders(
        data_dir=args.data_dir,
        train_file=args.train_file,
        val_file=args.val_file,
        channels=args.channels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fast_mode=args.fast
    )
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # Step 2: Build model
    print("\n2. Building model...")
    encoder, decoder = build_model(
        n_channels=len(args.channels),
        context_length=args.context_length,
        patch_size=args.patch_size,
        device=args.device
    )
    
    # Step 3: Create losses
    print("\n3. Setting up loss functions...")
    msm_criterion = MaskedSignalModeling(patch_size=args.patch_size)
    
    stft_criterion = MultiResolutionSTFT(
        n_ffts=[512, 1024, 2048],
        hop_lengths=[128, 256, 512],
        weight=1.0,
        use_spectral_convergence=False
    )
    
    print("   ✓ MSM loss (Masked Signal Modeling)")
    print(f"   ✓ Multi-Resolution STFT (weight: {args.stft_weight})")
    
    # Step 4: Create optimizer and scheduler
    print("\n4. Setting up optimizer and scheduler...")
    optimizer, scheduler = create_optimizer_scheduler(
        encoder=encoder,
        decoder=decoder,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        steps_per_epoch=len(train_loader)
    )
    
    print(f"   ✓ AdamW optimizer (lr={args.lr}, wd={args.weight_decay})")
    print(f"   ✓ Cosine scheduler with {args.warmup_epochs} warmup epochs")
    
    # Step 5: Select masking function
    mask_fn = random_masking if args.mask_type == 'random' else block_masking
    print(f"\n5. Masking strategy: {args.mask_type}")
    
    # Step 6: Create trainer
    print("\n6. Initializing SSL trainer...")
    trainer = SSLTrainer(
        encoder=encoder,
        decoder=decoder,
        optimizer=optimizer,
        msm_criterion=msm_criterion,
        stft_criterion=stft_criterion,
        mask_fn=mask_fn,
        device=args.device,
        use_amp=not args.no_amp,
        gradient_clip=args.gradient_clip,
        stft_weight=args.stft_weight
    )
    
    # Step 7: Train
    print("\n7. Starting training...")
    print("=" * 70)
    
    try:
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            save_dir=args.output_dir,
            mask_ratio=args.mask_ratio,
            log_interval=args.log_interval,
            save_interval=args.save_interval
        )
        
        # Print summary
        print("\n" + "=" * 70)
        print("TRAINING SUMMARY")
        print("=" * 70)
        print(f"Best validation loss: {trainer.best_val_loss:.4f}")
        print(f"Best checkpoint: {output_dir / 'best_model.pt'}")
        print(f"Final checkpoint: {output_dir / 'last_model.pt'}")
        print(f"Training history: {output_dir / 'training_history.json'}")
        print("=" * 70)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print(f"Checkpoints saved to: {output_dir}")
        return 1
    
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
