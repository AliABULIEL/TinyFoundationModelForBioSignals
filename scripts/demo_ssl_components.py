#!/usr/bin/env python3
"""Quick demo of SSL pretraining components without training.

This script verifies that all SSL pretraining components are correctly set up
and can work together, without actually training (which would take hours).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

print("=" * 70)
print("SSL PRETRAINING - COMPONENT VERIFICATION")
print("=" * 70)

# Test 1: Import all components
print("\n1. Testing imports...")
try:
    from src.models.ttm_adapter import create_ttm_model
    from src.models.decoders import ReconstructionHead1D
    from src.ssl.objectives import MaskedSignalModeling, MultiResolutionSTFT
    from src.ssl.masking import random_masking, block_masking
    from src.ssl.pretrainer import SSLTrainer
    print("   ✓ All imports successful")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Create encoder
print("\n2. Creating TTM encoder...")
try:
    encoder_config = {
        'variant': 'ibm-granite/granite-timeseries-ttm-r1',
        'task': 'ssl',
        'input_channels': 2,
        'context_length': 1250,
        'patch_size': 125,
        'freeze_encoder': False
    }
    encoder = create_ttm_model(encoder_config)
    encoder_params = sum(p.numel() for p in encoder.parameters())
    print(f"   ✓ Encoder created ({encoder_params:,} parameters)")
except Exception as e:
    print(f"   ✗ Encoder creation failed: {e}")
    sys.exit(1)

# Test 3: Create decoder
print("\n3. Creating reconstruction decoder...")
try:
    decoder = ReconstructionHead1D(
        d_model=192,
        patch_size=125,
        n_channels=2
    )
    decoder_params = sum(p.numel() for p in decoder.parameters())
    print(f"   ✓ Decoder created ({decoder_params:,} parameters)")
except Exception as e:
    print(f"   ✗ Decoder creation failed: {e}")
    sys.exit(1)

# Test 4: Create loss functions
print("\n4. Creating loss functions...")
try:
    msm_loss = MaskedSignalModeling(patch_size=125)
    stft_loss = MultiResolutionSTFT(
        n_ffts=[512, 1024, 2048],
        hop_lengths=[128, 256, 512],
        weight=1.0
    )
    print("   ✓ MSM loss created")
    print("   ✓ Multi-Resolution STFT loss created")
except Exception as e:
    print(f"   ✗ Loss creation failed: {e}")
    sys.exit(1)

# Test 5: Test masking
print("\n5. Testing masking functions...")
try:
    dummy_input = torch.randn(4, 2, 1250)  # [B, C, T]
    
    masked_random, mask_random = random_masking(dummy_input, mask_ratio=0.4, patch_size=125)
    print(f"   ✓ Random masking: {mask_random.sum().item()}/{mask_random.numel()} patches masked")
    
    masked_block, mask_block = block_masking(dummy_input, mask_ratio=0.4, span_length=3, patch_size=125)
    print(f"   ✓ Block masking: {mask_block.sum().item()}/{mask_block.numel()} patches masked")
except Exception as e:
    print(f"   ✗ Masking failed: {e}")
    sys.exit(1)

# Test 6: Test forward pass
print("\n6. Testing forward pass...")
try:
    # Create dummy batch
    batch = torch.randn(4, 2, 1250)  # [B, C, T]
    
    # Mask
    masked, mask_bool = random_masking(batch, mask_ratio=0.4, patch_size=125)
    print(f"   Input shape: {batch.shape}")
    print(f"   Masked shape: {masked.shape}")
    print(f"   Mask shape: {mask_bool.shape}")
    
    # Encode
    latents = encoder(masked)
    
    # Handle encoder output
    if isinstance(latents, tuple):
        latents = latents[0]
    if latents.ndim == 2:
        latents = latents.unsqueeze(1)
    
    print(f"   Latent shape: {latents.shape}")
    
    # Decode
    reconstructed = decoder(latents)
    print(f"   Reconstructed shape: {reconstructed.shape}")
    
    # Compute losses
    msm_loss_val = msm_loss(reconstructed, batch, mask_bool)
    stft_loss_val = stft_loss(reconstructed, batch)
    
    print(f"   ✓ Forward pass successful")
    print(f"   MSM loss: {msm_loss_val.item():.4f}")
    print(f"   STFT loss: {stft_loss_val.item():.4f}")
    
except Exception as e:
    print(f"   ✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Test optimizer and scheduler
print("\n7. Testing optimizer and scheduler...")
try:
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=5e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # Simple cosine scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=100,
        eta_min=1e-6
    )
    
    print(f"   ✓ Optimizer created")
    print(f"   ✓ Scheduler created")
    print(f"   Initial LR: {optimizer.param_groups[0]['lr']:.6f}")
    
except Exception as e:
    print(f"   ✗ Optimizer/scheduler creation failed: {e}")
    sys.exit(1)

# Test 8: Test SSLTrainer initialization
print("\n8. Testing SSLTrainer initialization...")
try:
    trainer = SSLTrainer(
        encoder=encoder,
        decoder=decoder,
        optimizer=optimizer,
        msm_criterion=msm_loss,
        stft_criterion=stft_loss,
        mask_fn=random_masking,
        device='cpu',
        use_amp=False,
        gradient_clip=1.0,
        stft_weight=0.3
    )
    print("   ✓ SSLTrainer initialized successfully")
except Exception as e:
    print(f"   ✗ SSLTrainer initialization failed: {e}")
    sys.exit(1)

# Test 9: Test single training step
print("\n9. Testing single training step...")
try:
    # Create small dummy dataset
    from torch.utils.data import TensorDataset, DataLoader
    
    dummy_data = torch.randn(32, 2, 1250)
    dataset = TensorDataset(dummy_data)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # One training step
    encoder.train()
    decoder.train()
    
    batch = next(iter(loader))[0]
    
    # Forward
    masked, mask_bool = random_masking(batch, mask_ratio=0.4, patch_size=125)
    latents = encoder(masked)
    
    if isinstance(latents, tuple):
        latents = latents[0]
    if latents.ndim == 2:
        latents = latents.unsqueeze(1)
    
    reconstructed = decoder(latents)
    
    # Loss
    loss = msm_loss(reconstructed, batch, mask_bool)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"   ✓ Training step successful")
    print(f"   Loss: {loss.item():.4f}")
    
except Exception as e:
    print(f"   ✗ Training step failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 70)
print("VERIFICATION COMPLETE - ALL COMPONENTS WORKING!")
print("=" * 70)
print("\nReady for SSL pretraining:")
print("  1. Prepare VitalDB windows:")
print("     python scripts/ttm_vitaldb.py build-windows ...")
print()
print("  2. Run SSL pretraining:")
print("     python scripts/pretrain_vitaldb_ssl.py \\")
print("         --data-dir data/vitaldb_windows \\")
print("         --output-dir artifacts/foundation_model \\")
print("         --epochs 100 --batch-size 128")
print()
print("  3. Or quick test:")
print("     bash scripts/test_ssl_pretrain.sh")
print("=" * 70)
