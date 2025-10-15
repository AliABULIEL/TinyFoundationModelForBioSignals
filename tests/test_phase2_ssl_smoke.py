#!/usr/bin/env python3
"""
Phase 2: SSL Pretraining Smoke Test

Quick 5-minute test of SSL pretraining on real VitalDB data.
Tests the full training loop with a small subset of data.

Run:
    python tests/test_phase2_ssl_smoke.py
    
    # Or specify custom paths:
    python tests/test_phase2_ssl_smoke.py \\
        --data-dir data/vitaldb_windows \\
        --max-windows 64 \\
        --output-dir artifacts/smoke_ssl
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from tqdm import tqdm

from src.models.ttm_adapter import create_ttm_model
from src.models.decoders import ReconstructionHead1D
from src.ssl.objectives import MaskedSignalModeling, MultiResolutionSTFT
from src.ssl.masking import random_masking


def load_data(data_dir: Path, max_windows: int, seed: int):
    """Load small subset of VitalDB data."""
    train_file = data_dir / 'train_windows.npz'
    
    if not train_file.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_file}\n\n"
            f"Please run Phase 1 data preparation first:\n"
            f"  python tests/test_phase1_data_prep.py\n"
        )
    
    print(f"Loading: {train_file}")
    data = np.load(train_file)
    signals = torch.from_numpy(data['signals']).float()
    
    N, C, T = signals.shape
    print(f"  Full dataset: {N:,} windows")
    
    # Sample subset
    np.random.seed(seed)
    if N > max_windows:
        indices = np.random.choice(N, max_windows, replace=False)
        signals = signals[indices]
    
    print(f"  Using: {len(signals)} windows")
    print(f"  Shape: {signals.shape}")
    
    # Split train/val (80/20)
    n_train = int(0.8 * len(signals))
    train_data = signals[:n_train]
    val_data = signals[n_train:]
    
    print(f"  Train: {len(train_data)} windows")
    print(f"  Val:   {len(val_data)} windows")
    
    return TensorDataset(train_data), TensorDataset(val_data)


def train_one_epoch(encoder, decoder, loader, msm_loss, stft_loss, 
                     optimizer, device, mask_ratio, patch_size, stft_weight):
    """Train for one epoch."""
    encoder.train()
    decoder.train()
    
    total_loss = 0.0
    total_msm = 0.0
    total_stft = 0.0
    num_batches = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        if isinstance(batch, (tuple, list)):
            inputs = batch[0]
        else:
            inputs = batch
        
        inputs = inputs.to(device)
        
        # Mask
        masked_inputs, mask_bool = random_masking(inputs, mask_ratio, patch_size)
        
        # Forward
        latents = encoder(masked_inputs)
        if isinstance(latents, tuple):
            latents = latents[0]
        if latents.ndim == 2:
            latents = latents.unsqueeze(1)
        
        reconstructed = decoder(latents)
        
        # Losses
        loss_msm = msm_loss(reconstructed, inputs, mask_bool)
        loss_stft = stft_loss(reconstructed, inputs)
        
        loss = loss_msm + stft_weight * loss_stft
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()), 1.0
        )
        optimizer.step()
        
        # Stats
        total_loss += loss.item()
        total_msm += loss_msm.item()
        total_stft += loss_stft.item()
        num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'msm_loss': total_msm / num_batches,
        'stft_loss': total_stft / num_batches
    }


@torch.no_grad()
def validate(encoder, decoder, loader, msm_loss, stft_loss,
             device, mask_ratio, patch_size, stft_weight):
    """Validate."""
    encoder.eval()
    decoder.eval()
    
    total_loss = 0.0
    total_msm = 0.0
    total_stft = 0.0
    num_batches = 0
    
    for batch in loader:
        if isinstance(batch, (tuple, list)):
            inputs = batch[0]
        else:
            inputs = batch
        
        inputs = inputs.to(device)
        
        # Mask
        masked_inputs, mask_bool = random_masking(inputs, mask_ratio, patch_size)
        
        # Forward
        latents = encoder(masked_inputs)
        if isinstance(latents, tuple):
            latents = latents[0]
        if latents.ndim == 2:
            latents = latents.unsqueeze(1)
        
        reconstructed = decoder(latents)
        
        # Losses
        loss_msm = msm_loss(reconstructed, inputs, mask_bool)
        loss_stft = stft_loss(reconstructed, inputs)
        
        loss = loss_msm + stft_weight * loss_stft
        
        total_loss += loss.item()
        total_msm += loss_msm.item()
        total_stft += loss_stft.item()
        num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'msm_loss': total_msm / num_batches,
        'stft_loss': total_stft / num_batches
    }


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: SSL pretraining smoke test"
    )
    parser.add_argument('--data-dir', type=str, default='data/vitaldb_windows')
    parser.add_argument('--max-windows', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--mask-ratio', type=float, default=0.4)
    parser.add_argument('--output-dir', type=str, default='artifacts/smoke_ssl')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    print("="*70)
    print("PHASE 2: SSL PRETRAINING SMOKE TEST")
    print("="*70)
    print("\nQuick 5-minute test of SSL training loop on real data.\n")
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Device: {device}")
    print(f"Data dir: {args.data_dir}")
    print(f"Max windows: {args.max_windows}")
    print(f"Mask ratio: {args.mask_ratio}")
    
    # Load data
    print("\n" + "="*70)
    print("[1/5] LOADING DATA")
    print("="*70)
    train_dataset, val_dataset = load_data(
        Path(args.data_dir), args.max_windows, args.seed
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Build model
    print("\n" + "="*70)
    print("[2/5] BUILDING MODEL")
    print("="*70)
    print("Loading IBM TTM weights...")
    
    encoder = create_ttm_model({
        'variant': 'ibm-granite/granite-timeseries-ttm-r1',
        'task': 'ssl',
        'input_channels': 2,
        'context_length': 1024,
        'patch_size': 128,
        'freeze_encoder': False
    }).to(device)
    
    decoder = ReconstructionHead1D(
        d_model=192,
        patch_size=125,
        n_channels=2
    ).to(device)
    
    print("✓ Model loaded")
    
    # Setup training
    print("\n" + "="*70)
    print("[3/5] SETUP TRAINING")
    print("="*70)
    
    msm_criterion = MaskedSignalModeling(patch_size=125).to(device)
    stft_criterion = MultiResolutionSTFT(
        n_ffts=[512, 1024, 2048],
        hop_lengths=[128, 256, 512],
        weight=1.0
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr,
        weight_decay=0.01
    )
    
    print("✓ Optimizer ready")
    print("✓ Loss functions ready")
    
    # Shape check
    print("\n" + "="*70)
    print("[4/5] SHAPE VALIDATION")
    print("="*70)
    
    sample_batch = next(iter(train_loader))
    if isinstance(sample_batch, (tuple, list)):
        sample = sample_batch[0]
    else:
        sample = sample_batch
    
    sample = sample.to(device)
    print(f"Input: {sample.shape}")
    
    masked, mask = random_masking(sample, 0.4, 125)
    print(f"Masked: {masked.shape}")
    print(f"Mask: {mask.shape}")
    
    with torch.no_grad():
        latents = encoder(masked)
        if isinstance(latents, tuple):
            latents = latents[0]
        if latents.ndim == 2:
            latents = latents.unsqueeze(1)
        print(f"Latents: {latents.shape}")
        
        recon = decoder(latents)
        print(f"Reconstructed: {recon.shape}")
        
        assert recon.shape == sample.shape, "Shape mismatch!"
    
    print("✓ All shapes correct")
    
    # Train
    print("\n" + "="*70)
    print("[5/5] TRAINING 1 EPOCH")
    print("="*70)
    
    start_time = time.time()
    
    train_metrics = train_one_epoch(
        encoder, decoder, train_loader,
        msm_criterion, stft_criterion, optimizer,
        device, args.mask_ratio, 125, 0.3
    )
    
    val_metrics = validate(
        encoder, decoder, val_loader,
        msm_criterion, stft_criterion,
        device, args.mask_ratio, 125, 0.3
    )
    
    elapsed = time.time() - start_time
    
    print(f"\nEpoch completed in {elapsed:.1f}s")
    print(f"\nTrain:")
    print(f"  Loss:      {train_metrics['loss']:.4f}")
    print(f"  MSM Loss:  {train_metrics['msm_loss']:.4f}")
    print(f"  STFT Loss: {train_metrics['stft_loss']:.4f}")
    
    print(f"\nValidation:")
    print(f"  Loss:      {val_metrics['loss']:.4f}")
    print(f"  MSM Loss:  {val_metrics['msm_loss']:.4f}")
    print(f"  STFT Loss: {val_metrics['stft_loss']:.4f}")
    
    # Save checkpoint
    checkpoint_path = output_dir / 'checkpoint.pt'
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'config': {
            'input_channels': 2,
            'context_length': 1024,
            'patch_size': 128,
            'mask_ratio': args.mask_ratio
        }
    }, checkpoint_path)
    
    print(f"\n✓ Checkpoint saved: {checkpoint_path}")
    
    # Summary
    print("\n" + "="*70)
    print("PHASE 2 SUMMARY")
    print("="*70)
    
    checks = [
        ("Data loaded", True),
        ("IBM TTM loaded", True),
        ("Shapes correct", True),
        ("Training completed", True),
        ("Loss finite", not (np.isnan(train_metrics['loss']) or np.isinf(train_metrics['loss']))),
        ("Checkpoint saved", checkpoint_path.exists())
    ]
    
    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)
    
    for name, ok in checks:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"{status:8} | {name}")
    
    print(f"\nResult: {passed}/{total} checks passed")
    print(f"Runtime: {elapsed:.1f}s")
    
    if passed == total:
        print("\n🎉 SSL pretraining pipeline works!")
        print("\nNext step:")
        print("  → Run full SSL pretraining:")
        print("  → python scripts/pretrain_vitaldb_ssl.py \\")
        print("       --config configs/ssl_pretrain.yaml \\")
        print("       --data-dir data/vitaldb_windows \\")
        print("       --epochs 100")
        return 0
    else:
        print("\n⚠️  Some checks failed.")
        return 1


if __name__ == '__main__':
    exit(main())
