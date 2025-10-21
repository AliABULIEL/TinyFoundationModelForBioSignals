#!/usr/bin/env python3
"""Self-Supervised Domain Adaptation on BUT-PPG Dataset.

This script performs SSL domain adaptation (MSM + STFT loss) on BUT-PPG data
WITHOUT using any labels (fully unsupervised transfer learning).

RECOMMENDED TRANSFER LEARNING HIERARCHY:
1. IBM TTM pre-trained weights (general time series)
2. VitalDB SSL pre-training (large biosignal dataset, ~50k samples)
3. BUT-PPG SSL domain adaptation (target domain, this script)
4. BUT-PPG supervised fine-tuning (with labels)
5. BUT-PPG downstream evaluation (test set)

Usage:
    # Transfer from VitalDB SSL checkpoint (RECOMMENDED)
    python scripts/pretrain_butppg_ssl.py \
        --pretrained artifacts/foundation_model/best_model.pt \
        --data-dir data/processed/butppg/windows_with_labels \
        --output-dir artifacts/butppg_ssl \
        --epochs 20 \
        --batch-size 64 \
        --lr 5e-5

    # Train from scratch (not recommended, small dataset)
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
from typing import Tuple
from tqdm import tqdm

# Import SSL components (use the EXISTING infrastructure!)
from src.ssl.masking import random_masking
from src.ssl.objectives import MaskedSignalModeling, MultiResolutionSTFT
from src.ssl.pretrainer import SSLTrainer
from src.models.ttm_adapter import TTMAdapter
from src.models.decoders import ReconstructionHead1D
from src.utils.seed import set_seed


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="SSL domain adaptation on BUT-PPG",
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
        '--pretrained',
        type=str,
        default=None,
        help='Path to VitalDB SSL pre-trained checkpoint (for transfer learning)'
    )
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
        default=0.3,
        help='Weight for STFT loss'
    )

    # Training
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
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
        default=5e-5,
        help='Learning rate (use 5e-5 for transfer, 1e-4 for scratch)'
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


def load_pretrained_encoder(checkpoint_path: str, device: str) -> Tuple[nn.Module, int]:
    """Load pre-trained encoder from VitalDB SSL checkpoint.

    Returns:
        encoder: Loaded encoder
        patch_size: Detected patch size
    """
    print(f"\nLoading pre-trained checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Extract encoder state dict
    if 'encoder_state_dict' in checkpoint:
        encoder_state = checkpoint['encoder_state_dict']
        print("  ✓ Found 'encoder_state_dict'")
    elif 'model_state_dict' in checkpoint:
        # Extract encoder weights (filter out decoder weights)
        full_state = checkpoint['model_state_dict']
        encoder_state = {k: v for k, v in full_state.items() if 'decoder' not in k}
        print("  ✓ Extracted encoder from 'model_state_dict'")
    else:
        # Try to use checkpoint as-is
        encoder_state = checkpoint
        print("  ⚠️  Using checkpoint as encoder state")

    # Detect architecture from weights
    # Find patcher weight to determine patch_size and d_model
    patcher_keys = [k for k in encoder_state.keys() if 'patcher.weight' in k or 'input_projection' in k]

    if patcher_keys:
        patcher_weight = encoder_state[patcher_keys[0]]
        d_model = patcher_weight.shape[0]
        patch_size_from_weights = patcher_weight.shape[1]

        print(f"  Detected from '{patcher_keys[0]}':")
        print(f"    d_model: {d_model}")
        print(f"    patch_size (per channel): {patch_size_from_weights}")

        # VitalDB uses patch_size=64 (with IBM pretrained adaptation)
        # BUT-PPG uses same architecture
        patch_size = patch_size_from_weights
    else:
        # Fallback
        d_model = 192
        patch_size = 64
        print(f"  ⚠️  Could not detect architecture, using defaults:")
        print(f"    d_model: {d_model}")
        print(f"    patch_size: {patch_size}")

    # Create encoder with matching architecture
    # IMPORTANT: Use patch_length that divides BUT-PPG context_length evenly
    # BUT-PPG: 1024 samples, so use patch_length=64 (16 patches) or 128 (8 patches)
    # VitalDB used patch_size=64 after adaptation, so we use the same
    encoder = TTMAdapter(
        num_channels=2,
        context_length=1024,  # BUT-PPG window size (NOT VitalDB's 1250!)
        patch_length=64,  # Match VitalDB's adapted patch_size
        d_model=d_model,
        num_layers=4,
        dropout=0.1,
        use_positional_encoding=True,
        task='ssl'  # No head, encoder only
    )

    # Load weights with flexible matching
    try:
        encoder.load_state_dict(encoder_state, strict=True)
        print("  ✓ Loaded encoder weights (strict=True)")
    except Exception as e:
        print(f"  ⚠️  Strict loading failed, trying flexible matching...")

        # Match keys flexibly
        model_dict = encoder.state_dict()
        matched_dict = {}

        for model_key in model_dict.keys():
            # Try exact match
            if model_key in encoder_state:
                matched_dict[model_key] = encoder_state[model_key]
            else:
                # Try suffix matching (last 3 parts)
                model_suffix = '.'.join(model_key.split('.')[-3:])
                for ckpt_key in encoder_state.keys():
                    ckpt_suffix = '.'.join(ckpt_key.split('.')[-3:])
                    if model_suffix == ckpt_suffix:
                        matched_dict[model_key] = encoder_state[ckpt_key]
                        break

        missing, unexpected = encoder.load_state_dict(matched_dict, strict=False)
        print(f"  ✓ Loaded {len(matched_dict)}/{len(model_dict)} weights")
        if missing:
            print(f"  ⚠️  Missing: {len(missing)} keys")
        if unexpected:
            print(f"  ⚠️  Unexpected: {len(unexpected)} keys")

    return encoder.to(device), patch_size


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Create output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Print config
    print("\n" + "=" * 80)
    print("BUT-PPG SSL DOMAIN ADAPTATION")
    print("=" * 80)
    print(f"Transfer Learning Hierarchy:")
    if args.pretrained:
        print(f"  1. ✓ IBM TTM pre-trained (general time series)")
        print(f"  2. ✓ VitalDB SSL pre-training (~50k samples)")
        print(f"  3. → BUT-PPG SSL adaptation (3.5k samples) ← THIS STAGE")
        print(f"  4. → BUT-PPG supervised fine-tuning (next)")
        print(f"  5. → Downstream evaluation (final)")
        print(f"\nPre-trained checkpoint: {args.pretrained}")
    else:
        print(f"  ⚠️  Training from scratch (not using VitalDB SSL weights)")
        print(f"  Note: Transfer learning from VitalDB is recommended for better performance")
    print(f"\nData directory: {args.data_dir}")
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

    # Create or load encoder
    print("Creating model...")

    if args.pretrained:
        encoder, patch_size = load_pretrained_encoder(args.pretrained, args.device)
        print(f"  ✓ Loaded pre-trained VitalDB SSL encoder")
    else:
        # Train from scratch
        # Use patch_length=64 for BUT-PPG (1024 samples = 16 patches of 64)
        encoder = TTMAdapter(
            num_channels=args.num_channels,
            context_length=args.context_length,  # 1024 for BUT-PPG
            patch_length=64,  # Matches BUT-PPG window size
            d_model=192,
            num_layers=4,
            dropout=0.1,
            use_positional_encoding=True,
            task='ssl'
        ).to(args.device)

        # Get actual patch size after TTM adaptation
        if hasattr(encoder, 'backbone') and hasattr(encoder.backbone, 'config'):
            patch_size = encoder.backbone.config.patch_length
        else:
            patch_size = 64

        print(f"  ✓ Created encoder from scratch (with IBM TTM base)")
        print(f"  Note: Using patch_length={patch_size} for BUT-PPG (1024/{patch_size}={1024//patch_size} patches)")

    # Create decoder for reconstruction
    decoder = ReconstructionHead1D(
        d_model=192,  # Must match encoder d_model
        patch_size=patch_size,
        n_channels=args.num_channels,
        context_length=args.context_length
    ).to(args.device)

    print(f"  Patch size: {patch_size}")
    print(f"  Encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"  Decoder parameters: {sum(p.numel() for p in decoder.parameters()):,}")
    print()

    # Create loss functions
    msm_criterion = MaskedSignalModeling(patch_size=patch_size)
    stft_criterion = MultiResolutionSTFT(
        n_ffts=[512, 1024, 2048],
        hop_lengths=[128, 256, 512],
        weight=1.0
    )

    # Create optimizer (optimize both encoder and decoder)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Create SSL trainer
    trainer = SSLTrainer(
        encoder=encoder,
        decoder=decoder,
        optimizer=optimizer,
        msm_criterion=msm_criterion,
        stft_criterion=stft_criterion,
        mask_fn=random_masking,
        device=args.device,
        use_amp=not args.no_amp,
        gradient_clip=1.0,
        stft_weight=args.stft_weight
    )

    # Train
    print("=" * 80)
    print("TRAINING")
    print("=" * 80)

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        save_dir=str(output_dir),
        mask_ratio=args.mask_ratio,
        log_interval=50,
        save_interval=10
    )

    # Save final checkpoint
    print("\n" + "=" * 80)
    print("SSL DOMAIN ADAPTATION COMPLETE!")
    print("=" * 80)
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    print(f"  - best_model.pt (encoder + decoder)")
    print(f"  - encoder_best.pt (encoder only, use for fine-tuning)")
    print()
    print("=" * 80)
    print("TRANSFER LEARNING PROGRESS")
    print("=" * 80)
    if args.pretrained:
        print("  ✅ Stage 1: IBM TTM pre-trained")
        print("  ✅ Stage 2: VitalDB SSL pre-training")
        print("  ✅ Stage 3: BUT-PPG SSL adaptation (DONE)")
        print("  ⏭️  Stage 4: BUT-PPG supervised fine-tuning (NEXT)")
        print("  ⏭️  Stage 5: Downstream evaluation")
    else:
        print("  ✅ BUT-PPG SSL training from scratch (DONE)")
        print("  ⏭️  BUT-PPG supervised fine-tuning (NEXT)")
        print("  ⏭️  Downstream evaluation")
    print()
    print("=" * 80)
    print("NEXT STEP: SUPERVISED FINE-TUNING")
    print("=" * 80)
    print("Run supervised fine-tuning on BUT-PPG quality classification:")
    print()
    print(f"  python scripts/finetune_butppg.py \\")
    print(f"    --pretrained {output_dir}/encoder_best.pt \\")
    print(f"    --data-dir {args.data_dir} \\")
    print(f"    --epochs 20 \\")
    print(f"    --batch-size 64 \\")
    print(f"    --lr 2e-5 \\")
    print(f"    --output-dir artifacts/butppg_finetuned")
    print()
    print("This will:")
    print("  • Add classification head for quality prediction")
    print("  • Fine-tune on BUT-PPG train set (with labels)")
    print("  • Validate on BUT-PPG val set")
    print("  • Save best model for evaluation on test set")
    print()
    print("=" * 80)
    print()


if __name__ == '__main__':
    main()
