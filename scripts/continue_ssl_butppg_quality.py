#!/usr/bin/env python3
"""Continue SSL on BUT-PPG with Quality-Aware Contrastive Learning.

This script implements Stage 2 of the hybrid SSL pipeline:
- Loads VitalDB SSL checkpoint (Stage 1) as initialization
- Performs quality-aware contrastive learning on BUT-PPG
- Combines contrastive + lightweight reconstruction losses
- Learns quality-relevant features to bridge domain gap

Target: This should significantly improve downstream AUROC (≥0.85) compared to
pure reconstruction SSL (0.597).

Usage:
    python scripts/continue_ssl_butppg_quality.py \
        --vitaldb-checkpoint artifacts/foundation_model/best_model.pt \
        --data-dir data/processed/butppg/windows_with_labels \
        --output-dir artifacts/butppg_quality_ssl \
        --epochs 50 \
        --batch-size 128 \
        --lr 5e-5 \
        --contrastive-weight 1.0 \
        --reconstruction-weight 0.3

Quick test (5 minutes):
    python scripts/continue_ssl_butppg_quality.py \
        --vitaldb-checkpoint artifacts/foundation_model/best_model.pt \
        --data-dir data/processed/butppg/windows_with_labels \
        --output-dir artifacts/butppg_quality_ssl_test \
        --epochs 2 \
        --batch-size 32 \
        --max-samples 500
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

import argparse
import json
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import core components
from src.models.ttm_adapter import TTMAdapter
from src.models.decoders import ReconstructionHead1D
from src.ssl.masking import random_masking
from src.ssl.objectives import MaskedSignalModeling
from src.ssl.quality_contrastive import QualityContrastiveLoss, HybridSSLLoss
from src.data.butppg_quality_dataset import QualityStratifiedBUTPPGDataset, BalancedQualitySampler
from src.utils.seed import set_seed


def load_vitaldb_checkpoint(
    checkpoint_path: str,
    device: str = 'cuda'
) -> Tuple[TTMAdapter, dict]:
    """Load VitalDB SSL checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        encoder: TTMAdapter encoder
        checkpoint_info: Dict with checkpoint metadata
    """
    print(f"\nLoading VitalDB SSL checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract encoder
    if 'encoder_state_dict' in checkpoint:
        encoder_state = checkpoint['encoder_state_dict']
    elif 'model_state_dict' in checkpoint:
        # Extract encoder part from full model
        encoder_state = {
            k.replace('encoder.', ''): v
            for k, v in checkpoint['model_state_dict'].items()
            if k.startswith('encoder.')
        }
    else:
        # Assume checkpoint is the encoder state dict
        encoder_state = checkpoint

    # Get model config
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Default config matching VitalDB
        config = {
            'context_length': 1024,
            'num_channels': 2,
            'd_model': 64,
            'patch_length': 128
        }

    # Extract config values
    context_length = config.get('context_length', 1024)
    num_channels = config.get('num_channels', 2)
    d_model = config.get('d_model', 64)
    patch_length = config.get('patch_length', 128)

    # CRITICAL FIX: Check if patch_length is divisible by context_length
    # TTM's adaptive patching may have changed the actual patch_size at runtime
    if context_length % patch_length != 0:
        # Calculate what TTM would actually use
        # For context=1024, TTM typically uses patch_size=64 (16 patches)
        if context_length == 1024:
            actual_patch_length = 64
        elif context_length == 512:
            actual_patch_length = 64
        elif context_length == 1536:
            actual_patch_length = 128
        else:
            # Find largest divisor
            for p in [128, 64, 32, 16]:
                if context_length % p == 0:
                    actual_patch_length = p
                    break
            else:
                actual_patch_length = patch_length  # Fallback

        print(f"  Detected patch_length mismatch: config={patch_length}, using={actual_patch_length}")
        patch_length = actual_patch_length

    # Create encoder
    encoder = TTMAdapter(
        context_length=context_length,
        input_channels=num_channels,
        d_model=d_model,
        patch_size=patch_length,  # FIXED: Use patch_size (not patch_length)
        use_real_ttm=True,
        task='ssl'
    ).to(device)

    # Load weights
    try:
        encoder.load_state_dict(encoder_state, strict=False)
        print("✓ Encoder weights loaded successfully")
    except Exception as e:
        print(f"⚠️  Partial weight loading: {e}")
        encoder.load_state_dict(encoder_state, strict=False)

    # Update config with corrected patch_length
    config['patch_length'] = patch_length
    config['context_length'] = context_length
    config['num_channels'] = num_channels
    config['d_model'] = d_model

    checkpoint_info = {
        'epoch': checkpoint.get('epoch', 0),
        'best_loss': checkpoint.get('best_loss', float('inf')),
        'config': config
    }

    return encoder, checkpoint_info


def init_ibm_pretrained(
    variant: str = 'ibm-granite/granite-timeseries-ttm-r1',
    context_length: int = 1024,
    patch_size: int = 128,
    num_channels: int = 2,
    device: str = 'cuda'
) -> Tuple[TTMAdapter, dict]:
    """Initialize encoder from IBM pretrained TTM.

    Args:
        variant: IBM TTM variant to use
        context_length: Context length (512, 1024, or 1536)
        patch_size: Patch size (64 or 128)
        num_channels: Number of input channels
        device: Device to load model on

    Returns:
        encoder: TTMAdapter encoder initialized from IBM pretrained
        checkpoint_info: Dict with config metadata
    """
    print(f"\nInitializing from IBM Pretrained TTM:")
    print(f"  Variant: {variant}")
    print(f"  Context: {context_length}, Patch: {patch_size}")
    print(f"  Channels: {num_channels}")

    # Determine expected d_model based on pretrained variant
    if context_length == 512 and patch_size == 64:
        d_model = 192  # TTM-Base
        variant_name = "TTM-Base"
    elif context_length == 1024 and patch_size == 128:
        d_model = 192  # TTM-Enhanced
        variant_name = "TTM-Enhanced"
    elif context_length == 1536 and patch_size == 128:
        d_model = 192  # TTM-Advanced
        variant_name = "TTM-Advanced"
    else:
        d_model = 256  # Custom - may not load pretrained weights
        variant_name = "Custom"

    print(f"  Pretrained: {variant_name} (d_model={d_model})")

    # Create encoder with IBM pretrained weights
    encoder = TTMAdapter(
        variant=variant,
        task='ssl',  # We'll use for SSL pretraining
        input_channels=num_channels,
        context_length=context_length,
        patch_size=patch_size,
        d_model=d_model,
        use_real_ttm=True,  # Use real IBM TTM
        freeze_encoder=False,  # Allow fine-tuning
        output_type='features'  # Return patch features
    ).to(device)

    print("✓ IBM pretrained encoder initialized")

    # Create config dict
    config = {
        'context_length': context_length,
        'num_channels': num_channels,
        'd_model': encoder.encoder_dim,  # Use actual encoder dim
        'patch_length': patch_size,
        'variant': variant,
        'variant_name': variant_name
    }

    checkpoint_info = {
        'epoch': 0,  # Starting fresh
        'best_loss': float('inf'),
        'config': config
    }

    return encoder, checkpoint_info


def train_epoch(
    encoder: nn.Module,
    decoder: nn.Module,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    mask_ratio: float = 0.4,
    patch_size: int = 125
) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        encoder: TTM encoder
        decoder: Reconstruction head
        train_loader: Training data loader
        loss_fn: HybridSSLLoss instance
        optimizer: Optimizer
        device: Device
        mask_ratio: Masking ratio for reconstruction
        patch_size: Patch size for masking (must match encoder)

    Returns:
        metrics: Dict with epoch metrics
    """
    encoder.train()
    decoder.train()

    total_loss = 0.0
    total_contrastive = 0.0
    total_reconstruction = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training")

    for batch in pbar:
        # Unpack batch
        signals = batch['signal'].to(device)  # [B, C, T]
        quality_scores = batch['quality_score'].to(device)  # [B]

        batch_size = signals.shape[0]

        # Forward pass through encoder
        # CRITICAL: Use get_encoder_output() for SSL tasks to get patch-level features [B, P, D]
        # The default forward() returns pooled features [B, D] for classification tasks
        features = encoder.get_encoder_output(signals)  # [B, P, D]

        # Pool features for contrastive learning
        # Use mean pooling across patches
        features_pooled = features.mean(dim=1)  # [B, D]

        # Optional: Masked reconstruction
        masked_signals, mask = random_masking(signals, mask_ratio=mask_ratio, patch_size=patch_size)
        masked_features = encoder.get_encoder_output(masked_signals)  # [B, P, D]
        reconstructed = decoder(masked_features)  # [B, C, T]

        # Compute hybrid loss
        loss, components = loss_fn(
            features=features_pooled,
            quality_scores=quality_scores,
            pred_signals=reconstructed,
            target_signals=signals,
            mask=mask,
            return_components=True
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()),
            max_norm=1.0
        )
        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        total_contrastive += components.get('contrastive', 0.0)
        total_reconstruction += components.get('reconstruction', 0.0)
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'contrast': f"{components.get('contrastive', 0.0):.4f}",
            'recon': f"{components.get('reconstruction', 0.0):.4f}"
        })

    metrics = {
        'loss': total_loss / num_batches,
        'contrastive': total_contrastive / num_batches,
        'reconstruction': total_reconstruction / num_batches
    }

    return metrics


@torch.no_grad()
def validate_epoch(
    encoder: nn.Module,
    decoder: nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: str,
    mask_ratio: float = 0.4,
    patch_size: int = 125
) -> Dict[str, float]:
    """Validate for one epoch.

    Args:
        encoder: TTM encoder
        decoder: Reconstruction head
        val_loader: Validation data loader
        loss_fn: HybridSSLLoss instance
        device: Device
        mask_ratio: Masking ratio
        patch_size: Patch size for masking (must match encoder)

    Returns:
        metrics: Dict with validation metrics
    """
    encoder.eval()
    decoder.eval()

    total_loss = 0.0
    total_contrastive = 0.0
    total_reconstruction = 0.0
    num_batches = 0

    for batch in val_loader:
        signals = batch['signal'].to(device)
        quality_scores = batch['quality_score'].to(device)

        # Forward pass
        # CRITICAL: Use get_encoder_output() to get patch-level features [B, P, D]
        features = encoder.get_encoder_output(signals)  # [B, P, D]
        features_pooled = features.mean(dim=1)  # [B, D]

        # Reconstruction
        masked_signals, mask = random_masking(signals, mask_ratio=mask_ratio, patch_size=patch_size)
        masked_features = encoder.get_encoder_output(masked_signals)  # [B, P, D]
        reconstructed = decoder(masked_features)  # [B, C, T]

        # Loss
        loss, components = loss_fn(
            features=features_pooled,
            quality_scores=quality_scores,
            pred_signals=reconstructed,
            target_signals=signals,
            mask=mask,
            return_components=True
        )

        total_loss += loss.item()
        total_contrastive += components.get('contrastive', 0.0)
        total_reconstruction += components.get('reconstruction', 0.0)
        num_batches += 1

    metrics = {
        'loss': total_loss / num_batches,
        'contrastive': total_contrastive / num_batches,
        'reconstruction': total_reconstruction / num_batches
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Quality-Aware SSL on BUT-PPG (Stage 2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Paths
    parser.add_argument('--vitaldb-checkpoint', type=str, default=None,
                       help='Path to VitalDB SSL checkpoint (Stage 1) - if not provided, uses IBM pretrained')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to BUT-PPG preprocessed data')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for checkpoints')

    # IBM Pretrained TTM options
    parser.add_argument('--use-ibm-pretrained', action='store_true',
                       help='Use IBM pretrained TTM instead of VitalDB checkpoint')
    parser.add_argument('--ibm-variant', type=str, default='ibm-granite/granite-timeseries-ttm-r1',
                       help='IBM TTM variant to use')
    parser.add_argument('--ibm-context-length', type=int, default=1024,
                       help='Context length for IBM TTM (512, 1024, or 1536)')
    parser.add_argument('--ibm-patch-size', type=int, default=128,
                       help='Patch size for IBM TTM (64 or 128)')

    # Training
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='Learning rate (lower than initial SSL)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay')

    # SSL parameters
    parser.add_argument('--mask-ratio', type=float, default=0.4,
                       help='Masking ratio for reconstruction')
    parser.add_argument('--contrastive-weight', type=float, default=1.0,
                       help='Weight for contrastive loss')
    parser.add_argument('--reconstruction-weight', type=float, default=0.3,
                       help='Weight for reconstruction loss')
    parser.add_argument('--temperature', type=float, default=0.07,
                       help='Temperature for contrastive loss')

    # Quality stratification
    parser.add_argument('--quality-bins', type=int, default=3,
                       help='Number of quality bins (3 = low/medium/high)')
    parser.add_argument('--balanced-sampling', action='store_true',
                       help='Use balanced sampling across quality bins')

    # Other
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples for quick testing')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='DataLoader workers')

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("QUALITY-AWARE SSL ON BUT-PPG (STAGE 2)")
    print("="*80)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Contrastive weight: {args.contrastive_weight}")
    print(f"Reconstruction weight: {args.reconstruction_weight}")
    print(f"Temperature: {args.temperature}")
    print("="*80)

    # Initialize encoder - either from VitalDB checkpoint OR IBM pretrained
    if args.use_ibm_pretrained or args.vitaldb_checkpoint is None:
        # Use IBM pretrained TTM
        print("\n[Initialization] Using IBM Pretrained TTM")
        encoder, checkpoint_info = init_ibm_pretrained(
            variant=args.ibm_variant,
            context_length=args.ibm_context_length,
            patch_size=args.ibm_patch_size,
            num_channels=2,  # PPG + ECG from BUT-PPG
            device=str(device)
        )
    else:
        # Load VitalDB SSL checkpoint
        print("\n[Initialization] Using VitalDB SSL Checkpoint")
        print(f"  Checkpoint: {args.vitaldb_checkpoint}")
        encoder, checkpoint_info = load_vitaldb_checkpoint(args.vitaldb_checkpoint, device=str(device))

    # Create decoder
    context_length = checkpoint_info['config'].get('context_length', 1024)
    num_channels = checkpoint_info['config'].get('num_channels', 2)

    # CRITICAL: Get runtime parameters from encoder (not from checkpoint config)
    # TTM auto-adapts these values, so we must query the actual runtime values

    # 1. Get patch_size from encoder
    if hasattr(encoder, 'patch_size'):
        patch_length = encoder.patch_size
        print(f"  ✓ Using encoder runtime patch_size: {patch_length}")
    elif hasattr(encoder, 'backbone') and hasattr(encoder.backbone, 'config'):
        patch_length = encoder.backbone.config.patch_length
        print(f"  ✓ Using encoder config patch_length: {patch_length}")
    else:
        patch_length = checkpoint_info['config'].get('patch_length', 128)
        print(f"  ⚠️  Using checkpoint config patch_length: {patch_length}")

    # 2. Get d_model from encoder
    if hasattr(encoder, 'encoder_dim'):
        d_model = encoder.encoder_dim
        print(f"  ✓ Using encoder runtime d_model: {d_model}")
    elif hasattr(encoder, 'backbone') and hasattr(encoder.backbone, 'config'):
        d_model = encoder.backbone.config.d_model
        print(f"  ✓ Using encoder config d_model: {d_model}")
    else:
        d_model = checkpoint_info['config'].get('d_model', 192)
        print(f"  ⚠️  Using checkpoint config d_model: {d_model}")

    decoder = ReconstructionHead1D(
        d_model=d_model,
        patch_size=patch_length,  # FIXED: Use patch_size (not output_length)
        n_channels=num_channels    # FIXED: Use n_channels (not num_channels)
    ).to(device)

    print(f"\n✓ Model initialized:")
    print(f"  Context length: {context_length}")
    print(f"  Num channels: {num_channels}")
    print(f"  d_model: {d_model}")
    print(f"  patch_size: {patch_length}")

    # Create datasets
    print(f"\nLoading BUT-PPG data...")
    train_dataset = QualityStratifiedBUTPPGDataset(
        data_dir=args.data_dir,
        split='train',
        modality='all',  # PPG + ECG + ACC
        mode='preprocessed',
        quality_bins=args.quality_bins,
        precompute_quality=True
    )

    val_dataset = QualityStratifiedBUTPPGDataset(
        data_dir=args.data_dir,
        split='val',
        modality='all',
        mode='preprocessed',
        quality_bins=args.quality_bins,
        precompute_quality=True
    )

    # Limit samples for testing
    if args.max_samples:
        # Limit train dataset
        train_limit = min(args.max_samples, len(train_dataset))
        train_dataset.base_dataset.window_files = train_dataset.base_dataset.window_files[:train_limit]
        train_dataset.base_dataset.valid_indices = train_dataset.base_dataset.valid_indices[:train_limit]
        # Also update quality scores to match
        if train_dataset.quality_scores is not None:
            train_dataset.quality_scores = train_dataset.quality_scores[:train_limit]
            train_dataset.quality_bin_indices = train_dataset.quality_bin_indices[:train_limit]

        # Limit val dataset
        val_limit = min(args.max_samples // 4, len(val_dataset))
        val_dataset.base_dataset.window_files = val_dataset.base_dataset.window_files[:val_limit]
        val_dataset.base_dataset.valid_indices = val_dataset.base_dataset.valid_indices[:val_limit]
        # Also update quality scores to match
        if val_dataset.quality_scores is not None:
            val_dataset.quality_scores = val_dataset.quality_scores[:val_limit]
            val_dataset.quality_bin_indices = val_dataset.quality_bin_indices[:val_limit]

        print(f"\n⚠️  Limited to {train_limit} train / {val_limit} val samples for testing")

    # Create data loaders
    if args.balanced_sampling and train_dataset.quality_scores is not None:
        train_sampler = BalancedQualitySampler(train_dataset, samples_per_bin=None)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )
    else:
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

    print(f"✓ Data loaded:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")

    # Create loss function
    loss_fn = HybridSSLLoss(
        contrastive_weight=args.contrastive_weight,
        reconstruction_weight=args.reconstruction_weight,
        temperature=args.temperature
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Training loop
    print(f"\n{'='*80}")
    print("TRAINING")
    print(f"{'='*80}\n")

    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_contrastive': [],
        'val_contrastive': [],
        'train_reconstruction': [],
        'val_reconstruction': []
    }

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 80)

        # Train
        train_metrics = train_epoch(
            encoder, decoder, train_loader, loss_fn, optimizer, str(device), args.mask_ratio, patch_length
        )

        # Validate
        val_metrics = validate_epoch(
            encoder, decoder, val_loader, loss_fn, str(device), args.mask_ratio, patch_length
        )

        # Log metrics
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f} | "
              f"Contrastive: {train_metrics['contrastive']:.4f} | "
              f"Reconstruction: {train_metrics['reconstruction']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f} | "
              f"Contrastive: {val_metrics['contrastive']:.4f} | "
              f"Reconstruction: {val_metrics['reconstruction']:.4f}")

        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_contrastive'].append(train_metrics['contrastive'])
        history['val_contrastive'].append(val_metrics['contrastive'])
        history['train_reconstruction'].append(train_metrics['reconstruction'])
        history['val_reconstruction'].append(val_metrics['reconstruction'])

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']

            checkpoint = {
                'epoch': epoch + 1,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'config': checkpoint_info['config'],
                'args': vars(args)
            }

            torch.save(checkpoint, output_dir / 'best_model.pt')
            print(f"✓ Best model saved! (val_loss: {val_metrics['loss']:.4f})")

        # Save last checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_metrics['loss'],
            'config': checkpoint_info['config'],
            'args': vars(args)
        }
        torch.save(checkpoint, output_dir / 'last_checkpoint.pt')

    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    print(f"\nNext step: Fine-tune on BUT-PPG with labels (Stage 3)")
    print("="*80)


if __name__ == '__main__':
    main()
