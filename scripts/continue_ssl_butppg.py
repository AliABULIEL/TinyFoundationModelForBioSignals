#!/usr/bin/env python3
"""
Continue SSL Pre-training on BUT-PPG Dataset
=============================================

This script continues Self-Supervised Learning (SSL) from a VitalDB checkpoint
on BUT-PPG data for domain adaptation. NO supervised labels are used - this is
pure unsupervised learning through masking + reconstruction.

Purpose: Adapt the VitalDB foundation model to BUT-PPG domain characteristics

Training Strategy:
- Load VitalDB SSL checkpoint (artifacts/foundation_model/best_model.pt)
- Continue SSL training on BUT-PPG signals (PPG + ECG, 2 channels)
- Same approach: 40% masking + MSM + Multi-Resolution STFT loss
- NO quality labels needed - this is unsupervised domain adaptation
- Output: Adapted checkpoint for BUT-PPG domain

Usage:
    # Basic usage
    python scripts/continue_ssl_butppg.py \
        --checkpoint artifacts/foundation_model/best_model.pt \
        --data-dir data/processed/butppg/windows_with_labels

    # Custom epochs and batch size
    python scripts/continue_ssl_butppg.py \
        --checkpoint artifacts/foundation_model/best_model.pt \
        --data-dir data/processed/butppg/windows_with_labels \
        --epochs 20 \
        --batch-size 64

    # Lower learning rate for fine adaptation
    python scripts/continue_ssl_butppg.py \
        --checkpoint artifacts/foundation_model/best_model.pt \
        --data-dir data/processed/butppg/windows_with_labels \
        --epochs 30 \
        --lr 1e-5

Author: Senior ML & SW Engineer
Date: October 2025
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import SSL components
from src.ssl.masking import random_masking
from src.ssl.objectives import MaskedSignalModeling, MultiResolutionSTFT
from src.ssl.pretrainer import SSLTrainer

# Import model components
from src.models.ttm_adapter import TTMAdapter
from src.models.decoders import ReconstructionHead1D

# Import utilities
from src.utils.seed import set_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BUTPPGSSLDataset(Dataset):
    """
    Dataset for SSL continuation on BUT-PPG.

    Loads BUT-PPG signals WITHOUT using quality labels.
    Only the raw signals are used for masking + reconstruction.

    Args:
        data_dir: Path to directory containing window_*.npz files
        target_length: Resize signals to this length to match SSL model
        normalize: Whether to apply z-score normalization
    """

    def __init__(
        self,
        data_dir: Path,
        target_length: int = 1024,
        normalize: bool = True
    ):
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # Find all window files
        window_files = sorted(data_dir.glob('window_*.npz'))

        if len(window_files) == 0:
            raise FileNotFoundError(
                f"No window files found in {data_dir}\n"
                f"Expected files matching pattern: window_*.npz"
            )

        # Load all signals (ignore labels for SSL)
        signals_list = []

        for window_file in window_files:
            data = np.load(window_file)

            if 'signal' not in data:
                raise KeyError(f"Expected 'signal' key in {window_file}")

            signal = data['signal']  # [2, T]
            signals_list.append(signal)

        # Stack into tensors
        self.signals = torch.from_numpy(np.stack(signals_list, axis=0)).float()  # [N, 2, T]

        N, C, T = self.signals.shape
        assert C == 2, f"Expected 2 channels (PPG + ECG), got {C}"

        # Resize to target length if needed
        if T != target_length:
            logger.info(f"Resizing signals from {T} to {target_length} samples")
            self.signals = torch.nn.functional.interpolate(
                self.signals,
                size=target_length,
                mode='linear',
                align_corners=False
            )

        # Normalize per channel
        if normalize:
            for c in range(C):
                channel_data = self.signals[:, c, :]  # [N, T]
                mean = channel_data.mean()
                std = channel_data.std()
                if std > 0:
                    self.signals[:, c, :] = (channel_data - mean) / std

        logger.info(f"Loaded {N} samples from {len(window_files)} windows")
        logger.info(f"Signal shape: {self.signals.shape}")

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx]


def load_vitaldb_checkpoint(
    checkpoint_path: Path,
    device: torch.device
) -> tuple:
    """
    Load VitalDB SSL checkpoint and extract model components.

    Returns:
        encoder: TTMAdapter encoder
        decoder: ReconstructionHead1D
        optimizer_state: Saved optimizer state (optional)
        config: Model configuration
    """
    logger.info(f"\nLoading VitalDB SSL checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract architecture info
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'encoder_state_dict' in checkpoint:
        state_dict = checkpoint['encoder_state_dict']
    else:
        raise KeyError(f"Cannot find model weights in checkpoint. Keys: {checkpoint.keys()}")

    # Detect architecture from checkpoint
    logger.info("Detecting architecture from checkpoint...")

    # Get d_model and context_length from patcher weights
    patcher_weight = None
    for key in state_dict.keys():
        if 'patcher' in key and 'weight' in key:
            patcher_weight = state_dict[key]
            break

    if patcher_weight is None:
        raise ValueError("Cannot find patcher weights in checkpoint")

    d_model = patcher_weight.shape[0]
    patch_size = patcher_weight.shape[-1]

    # Get context length from positional encoding
    for key in state_dict.keys():
        if 'pos_embed' in key or 'positional' in key:
            pos_embed = state_dict[key]
            # Assuming pos_embed shape is [1, num_patches, d_model]
            if len(pos_embed.shape) == 3:
                num_patches = pos_embed.shape[1]
                context_length = num_patches * patch_size
                break
    else:
        # Default for VitalDB SSL
        context_length = 1024

    logger.info(f"  Detected: d_model={d_model}, patch_size={patch_size}, context_length={context_length}")

    # Determine if IBM pretrained was used
    use_ibm_pretrained = (d_model == 192 and context_length == 1024)

    if use_ibm_pretrained:
        logger.info("  ✓ Checkpoint used IBM pretrained TTM-Enhanced")
        model_patch_size = 128  # Load IBM pretrained with correct config
    else:
        logger.info("  ✓ Checkpoint used fresh TTM")
        model_patch_size = patch_size

    # Recreate encoder with same configuration
    logger.info("\nRecreating encoder architecture...")
    encoder = TTMAdapter(
        context_length=context_length,
        patch_length=model_patch_size,
        num_input_channels=2,
        num_output_channels=2,
        d_model=d_model,
        use_ibm_pretrained=use_ibm_pretrained,
        for_pretraining=True
    ).to(device)

    # Load encoder weights
    encoder_state = {k.replace('encoder.', ''): v for k, v in state_dict.items() if 'encoder' in k}
    if len(encoder_state) == 0:
        # Try without prefix
        encoder_state = state_dict

    encoder.load_state_dict(encoder_state, strict=False)
    logger.info(f"  ✓ Encoder loaded: {sum(p.numel() for p in encoder.parameters())} parameters")

    # Recreate decoder
    # Get actual patch_size from encoder after initialization
    dummy_input = torch.randn(1, 2, context_length).to(device)
    with torch.no_grad():
        encoder_output = encoder.get_encoder_output(dummy_input)
        actual_d_model = encoder_output.shape[-1]
        num_patches = encoder_output.shape[1]
        actual_patch_size = context_length // num_patches

    logger.info(f"  Actual encoder output: d_model={actual_d_model}, num_patches={num_patches}, patch_size={actual_patch_size}")

    decoder = ReconstructionHead1D(
        d_model=actual_d_model,
        patch_size=actual_patch_size,
        n_channels=2
    ).to(device)

    # Try to load decoder weights if available
    if 'decoder_state_dict' in checkpoint:
        decoder_state = checkpoint['decoder_state_dict']
        try:
            decoder.load_state_dict(decoder_state)
            logger.info("  ✓ Decoder loaded from checkpoint")
        except RuntimeError as e:
            logger.warning(f"  ⚠️  Could not load decoder weights (will train from scratch): {e}")
    else:
        logger.info("  ℹ️  No decoder weights in checkpoint (will train from scratch)")

    # Get optimizer state if available
    optimizer_state = checkpoint.get('optimizer_state_dict', None)

    # Get config
    config = {
        'd_model': actual_d_model,
        'patch_size': actual_patch_size,
        'context_length': context_length,
        'use_ibm_pretrained': use_ibm_pretrained
    }

    return encoder, decoder, optimizer_state, config


def train_ssl_butppg(
    encoder: nn.Module,
    decoder: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict,
    args: argparse.Namespace,
    device: torch.device
):
    """
    Continue SSL training on BUT-PPG data.

    Uses same SSL approach as VitalDB pre-training:
    - Masking (40% of patches)
    - MSM loss (reconstruction on masked patches)
    - Multi-Resolution STFT loss
    """
    logger.info("\n" + "="*80)
    logger.info("CONTINUING SSL TRAINING ON BUT-PPG")
    logger.info("="*80)
    logger.info(f"Strategy: Unsupervised domain adaptation (NO labels used)")
    logger.info(f"Approach: Masking + Reconstruction")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Mask ratio: {args.mask_ratio}")

    # Create SSL losses
    msm_loss_fn = MaskedSignalModeling()
    stft_loss_fn = MultiResolutionSTFT(
        n_ffts=args.stft_n_ffts,
        hop_lengths=args.stft_hop_lengths,
        loss_weight=args.stft_weight
    )

    # Create optimizer
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )

    # Create SSL trainer
    trainer = SSLTrainer(
        encoder=encoder,
        decoder=decoder,
        optimizer=optimizer,
        scheduler=scheduler,
        msm_loss_fn=msm_loss_fn,
        stft_loss_fn=stft_loss_fn,
        mask_ratio=args.mask_ratio,
        patch_size=config['patch_size'],
        device=device,
        output_dir=args.output_dir,
        save_every=args.save_every,
        masking_strategy=args.masking_strategy
    )

    # Train
    logger.info("\nStarting SSL continuation training...")

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        logger.info(f"\n{'='*80}")
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        logger.info(f"{'='*80}")

        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)

        # Validate
        val_metrics = trainer.validate(val_loader, epoch)

        # Log metrics
        logger.info(f"\nEpoch {epoch + 1} Summary:")
        logger.info(f"  Train Loss: {train_metrics['total_loss']:.4f} "
                   f"(MSM: {train_metrics['msm_loss']:.4f}, STFT: {train_metrics['stft_loss']:.4f})")
        logger.info(f"  Val Loss:   {val_metrics['total_loss']:.4f} "
                   f"(MSM: {val_metrics['msm_loss']:.4f}, STFT: {val_metrics['stft_loss']:.4f})")
        logger.info(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            patience_counter = 0

            save_path = args.output_dir / 'best_model_butppg.pt'
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': config,
                'args': vars(args)
            }, save_path)
            logger.info(f"  ✓ Saved best model: {save_path}")
        else:
            patience_counter += 1

        # Early stopping
        if args.early_stopping and patience_counter >= args.patience:
            logger.info(f"\nEarly stopping triggered (patience={args.patience})")
            break

        # Step scheduler
        scheduler.step()

    logger.info("\n" + "="*80)
    logger.info("SSL CONTINUATION TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Saved to: {args.output_dir / 'best_model_butppg.pt'}")
    logger.info("\nNext steps:")
    logger.info("  1. Use this adapted checkpoint for BUT-PPG downstream tasks")
    logger.info("  2. Evaluate on BUT-PPG quality classification")
    logger.info("  3. Compare with VitalDB checkpoint to measure domain adaptation gains")


def main():
    parser = argparse.ArgumentParser(
        description="Continue SSL pre-training on BUT-PPG dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Data arguments
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to VitalDB SSL checkpoint (optional - will start from IBM pretrained if not provided)')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to BUT-PPG processed data directory')
    parser.add_argument('--output-dir', type=str, default='artifacts/butppg_ssl',
                       help='Output directory for checkpoints')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate (lower than pre-training for fine adaptation)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay')

    # SSL arguments
    parser.add_argument('--mask-ratio', type=float, default=0.4,
                       help='Masking ratio (40% default)')
    parser.add_argument('--masking-strategy', type=str, default='random',
                       choices=['random', 'block'],
                       help='Masking strategy')
    parser.add_argument('--stft-n-ffts', nargs='+', type=int, default=[512, 1024, 2048],
                       help='FFT sizes for multi-resolution STFT')
    parser.add_argument('--stft-hop-lengths', nargs='+', type=int, default=[128, 256, 512],
                       help='Hop lengths for multi-resolution STFT')
    parser.add_argument('--stft-weight', type=float, default=0.3,
                       help='Weight for STFT loss')

    # Other arguments
    parser.add_argument('--save-every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--early-stopping', action='store_true',
                       help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers')

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create output directory
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load or create model
    if args.checkpoint:
        # Option B: Load from VitalDB checkpoint
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info("\n" + "="*80)
        logger.info("MODE: Continue from VitalDB SSL checkpoint")
        logger.info("="*80)

        encoder, decoder, optimizer_state, config = load_vitaldb_checkpoint(
            checkpoint_path, device
        )
    else:
        # Option A: Start fresh from IBM pretrained
        logger.info("\n" + "="*80)
        logger.info("MODE: Start SSL on BUT-PPG from IBM pretrained")
        logger.info("="*80)
        logger.info("No VitalDB checkpoint provided - starting fresh from IBM TTM")

        from src.models.ttm_adapter import TTMAdapter
        from src.models.decoders import ReconstructionHead1D

        context_length = 1024
        patch_length = 128  # IBM TTM config
        d_model = 192  # TTM-Enhanced default

        logger.info(f"\nCreating encoder from IBM pretrained...")
        logger.info(f"  context_length: {context_length}")
        logger.info(f"  patch_length: {patch_length}")
        logger.info(f"  d_model: {d_model}")

        encoder = TTMAdapter(
            context_length=context_length,
            patch_length=patch_length,
            num_input_channels=2,
            num_output_channels=2,
            d_model=d_model,
            use_ibm_pretrained=True,
            for_pretraining=True
        ).to(device)

        logger.info(f"✓ Encoder created: {sum(p.numel() for p in encoder.parameters())} parameters")

        # Get actual dimensions from forward pass
        dummy_input = torch.randn(1, 2, context_length).to(device)
        with torch.no_grad():
            encoder_output = encoder.get_encoder_output(dummy_input)
            actual_d_model = encoder_output.shape[-1]
            num_patches = encoder_output.shape[1]
            actual_patch_size = context_length // num_patches

        logger.info(f"✓ Actual encoder output: d_model={actual_d_model}, patch_size={actual_patch_size}")

        # Create decoder
        decoder = ReconstructionHead1D(
            d_model=actual_d_model,
            patch_size=actual_patch_size,
            n_channels=2
        ).to(device)

        logger.info(f"✓ Decoder created: {sum(p.numel() for p in decoder.parameters())} parameters")

        config = {
            'd_model': actual_d_model,
            'patch_size': actual_patch_size,
            'context_length': context_length,
            'use_ibm_pretrained': True
        }
        optimizer_state = None

    # Load BUT-PPG data
    logger.info("\n" + "="*80)
    logger.info("LOADING BUT-PPG DATA")
    logger.info("="*80)

    data_dir = Path(args.data_dir)

    train_dataset = BUTPPGSSLDataset(
        data_dir=data_dir / 'train',
        target_length=config['context_length'],
        normalize=True
    )

    val_dataset = BUTPPGSSLDataset(
        data_dir=data_dir / 'val',
        target_length=config['context_length'],
        normalize=True
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    logger.info(f"\nDataset sizes:")
    logger.info(f"  Train: {len(train_dataset)} samples")
    logger.info(f"  Val:   {len(val_dataset)} samples")
    logger.info(f"  Batches per epoch: {len(train_loader)}")

    # Train
    train_ssl_butppg(
        encoder=encoder,
        decoder=decoder,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        args=args,
        device=device
    )


if __name__ == '__main__':
    main()
