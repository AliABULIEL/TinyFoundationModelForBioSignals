#!/usr/bin/env python3
"""Self-Supervised Pre-training on BUT-PPG Dataset.

This script performs SSL pre-training (MSM + STFT loss) on BUT-PPG data
WITHOUT using any labels (fully unsupervised domain adaptation).

Usage:
    # Quick test (5 epochs)
    python scripts/pretrain_butppg_ssl.py \
        --data-dir data/processed/butppg/windows_with_labels \
        --output-dir artifacts/butppg_ssl \
        --epochs 5 \
        --batch-size 64

    # Full training (50 epochs)
    python scripts/pretrain_butppg_ssl.py \
        --data-dir data/processed/butppg/windows_with_labels \
        --output-dir artifacts/butppg_ssl \
        --epochs 50 \
        --batch-size 64 \
        --lr 1e-4
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Import SSL components
from src.ssl.masking import create_masked_reconstruction_task
from src.ssl.objectives import STFTLoss, MSMLoss
from src.models.ttm_adapter import TTMAdapter


class BUTPPGSSLDataset(Dataset):
    """BUT-PPG dataset for SSL (loads signals only, ignores labels)."""

    def __init__(self, data_dir: Path, split: str):
        """
        Args:
            data_dir: Root data directory with train/val/test subdirs
            split: 'train', 'val', or 'test'
        """
        self.split_dir = data_dir / split

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        # Find all window files
        self.window_files = sorted(self.split_dir.glob('window_*.npz'))

        if len(self.window_files) == 0:
            raise ValueError(f"No window files found in {self.split_dir}")

        print(f"  {split}: {len(self.window_files)} windows")

    def __len__(self):
        return len(self.window_files)

    def __getitem__(self, idx):
        data = np.load(self.window_files[idx])

        # Load signal only (shape: [2, 1024])
        signal = torch.from_numpy(data['signal']).float()

        return signal


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    msm_criterion: MSMLoss,
    stft_criterion: STFTLoss,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: str,
    patch_size: int,
    mask_ratio: float = 0.4,
    stft_weight: float = 0.1,
    use_amp: bool = True
):
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_msm_loss = 0.0
    total_stft_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)

    for batch_idx, signals in enumerate(pbar):
        signals = signals.to(device)  # [B, 2, 1024]

        # Create masked task
        masked_input, mask, targets = create_masked_reconstruction_task(
            signals,
            patch_size=patch_size,
            mask_ratio=mask_ratio
        )

        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            # Forward pass
            reconstructed = model(masked_input)  # [B, 2, T]

            # Compute losses
            msm_loss = msm_criterion(reconstructed, targets, mask)
            stft_loss = stft_criterion(reconstructed, signals)

            # Combined loss
            loss = msm_loss + stft_weight * stft_loss

        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Track metrics
        total_loss += loss.item()
        total_msm_loss += msm_loss.item()
        total_stft_loss += stft_loss.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'msm': f'{msm_loss.item():.4f}',
            'stft': f'{stft_loss.item():.4f}'
        })

    return {
        'loss': total_loss / num_batches,
        'msm_loss': total_msm_loss / num_batches,
        'stft_loss': total_stft_loss / num_batches
    }


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    msm_criterion: MSMLoss,
    stft_criterion: STFTLoss,
    device: str,
    patch_size: int,
    mask_ratio: float = 0.4,
    stft_weight: float = 0.1
):
    """Validate for one epoch."""
    model.eval()

    total_loss = 0.0
    total_msm_loss = 0.0
    total_stft_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Validation", leave=False)

    for signals in pbar:
        signals = signals.to(device)

        # Create masked task
        masked_input, mask, targets = create_masked_reconstruction_task(
            signals,
            patch_size=patch_size,
            mask_ratio=mask_ratio
        )

        # Forward pass
        reconstructed = model(masked_input)

        # Compute losses
        msm_loss = msm_criterion(reconstructed, targets, mask)
        stft_loss = stft_criterion(reconstructed, signals)
        loss = msm_loss + stft_weight * stft_loss

        total_loss += loss.item()
        total_msm_loss += msm_loss.item()
        total_stft_loss += stft_loss.item()
        num_batches += 1

        pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})

    return {
        'val_loss': total_loss / num_batches,
        'val_msm_loss': total_msm_loss / num_batches,
        'val_stft_loss': total_stft_loss / num_batches
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="SSL pre-training on BUT-PPG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed/butppg/windows_with_labels',
        help='Directory containing BUT-PPG windowed data'
    )

    # Model
    parser.add_argument(
        '--context-length',
        type=int,
        default=1024,
        help='Context length (must match window size)'
    )
    parser.add_argument(
        '--num-channels',
        type=int,
        default=2,
        help='Number of input channels (PPG + ECG)'
    )

    # SSL
    parser.add_argument(
        '--mask-ratio',
        type=float,
        default=0.4,
        help='Masking ratio for MSM'
    )
    parser.add_argument(
        '--stft-weight',
        type=float,
        default=0.1,
        help='Weight for STFT loss'
    )

    # Training
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.01,
        help='Weight decay'
    )

    # Device
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

    # Output
    parser.add_argument(
        '--output-dir',
        type=str,
        default='artifacts/butppg_ssl',
        help='Output directory'
    )

    # Misc
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Print config
    print("\n" + "=" * 80)
    print("BUT-PPG SSL PRE-TRAINING")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Context length: {args.context_length}")
    print(f"Channels: {args.num_channels}")
    print(f"Mask ratio: {args.mask_ratio}")
    print(f"STFT weight: {args.stft_weight}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    print(f"AMP: {not args.no_amp}")
    print()

    # Load datasets
    print("Loading datasets...")
    data_dir = Path(args.data_dir)

    train_dataset = BUTPPGSSLDataset(data_dir, 'train')
    val_dataset = BUTPPGSSLDataset(data_dir, 'val')

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

    print(f"✓ Loaded {len(train_dataset)} train, {len(val_dataset)} val samples")
    print()

    # Create model
    print("Creating model...")
    model = TTMAdapter(
        num_channels=args.num_channels,
        context_length=args.context_length,
        patch_length=16,  # TTM will auto-adjust
        num_layers=4,
        d_model=64,
        use_positional_encoding=True,
        dropout=0.1
    ).to(args.device)

    # Get actual patch size after TTM adaptation
    if hasattr(model.backbone, 'config'):
        patch_size = model.backbone.config.patch_length
    else:
        patch_size = 16  # fallback

    print(f"✓ Model created")
    print(f"  Patch size: {patch_size}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Create loss functions
    msm_criterion = MSMLoss()
    stft_criterion = STFTLoss(
        n_ffts=[512, 1024, 2048],
        hop_lengths=[128, 256, 512]
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Create scaler for AMP
    scaler = GradScaler(enabled=not args.no_amp)

    # Training loop
    print("=" * 80)
    print("TRAINING")
    print("=" * 80)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_msm_loss': [],
        'val_msm_loss': [],
        'train_stft_loss': [],
        'val_stft_loss': []
    }

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 80)

        start_time = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, msm_criterion, stft_criterion,
            optimizer, scaler, args.device, patch_size,
            args.mask_ratio, args.stft_weight, not args.no_amp
        )

        # Validate
        val_metrics = validate_epoch(
            model, val_loader, msm_criterion, stft_criterion,
            args.device, patch_size, args.mask_ratio, args.stft_weight
        )

        epoch_time = time.time() - start_time

        # Log metrics
        print(f"Train Loss: {train_metrics['loss']:.4f} "
              f"(MSM: {train_metrics['msm_loss']:.4f}, "
              f"STFT: {train_metrics['stft_loss']:.4f})")
        print(f"Val Loss:   {val_metrics['val_loss']:.4f} "
              f"(MSM: {val_metrics['val_msm_loss']:.4f}, "
              f"STFT: {val_metrics['val_stft_loss']:.4f})")
        print(f"Time: {epoch_time:.1f}s")

        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['val_loss'])
        history['train_msm_loss'].append(train_metrics['msm_loss'])
        history['val_msm_loss'].append(val_metrics['val_msm_loss'])
        history['train_stft_loss'].append(train_metrics['stft_loss'])
        history['val_stft_loss'].append(val_metrics['val_stft_loss'])

        # Save best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['val_loss'],
                'config': vars(args)
            }

            torch.save(checkpoint, output_dir / 'best_model_butppg.pt')
            print(f"✓ Saved best model (val_loss: {best_val_loss:.4f})")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['val_loss'],
                'config': vars(args)
            }
            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch}.pt')

    # Save final checkpoint
    checkpoint = {
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_metrics['val_loss'],
        'config': vars(args)
    }
    torch.save(checkpoint, output_dir / 'last_checkpoint.pt')

    # Save training history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    print(f"  - best_model_butppg.pt")
    print(f"  - last_checkpoint.pt")
    print(f"  - history.json")
    print("\nNext steps:")
    print("  1. Use best_model_butppg.pt for downstream tasks")
    print("  2. Or fine-tune on specific BUT-PPG tasks:")
    print(f"     python scripts/finetune_butppg.py \\")
    print(f"       --pretrained {output_dir}/best_model_butppg.pt \\")
    print(f"       --data-dir {args.data_dir}")
    print()


if __name__ == '__main__':
    main()
