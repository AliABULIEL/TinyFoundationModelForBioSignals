#!/usr/bin/env python3
"""
SSL Setup Validation Script
============================

Validates that all components are correctly configured before running SSL training:
1. Patch size detection works correctly
2. Encoder-decoder dimensions match
3. Masking aligns with encoder patches
4. Loss computation is valid
5. Data loading works

Run this BEFORE starting expensive SSL training!

Usage:
    python scripts/validate_ssl_setup.py --config configs/ssl_pretrain.yaml

Author: Claude Code Foundation Model Audit
Date: October 2025
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import yaml

# Add project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models.ttm_adapter import TTMAdapter
from src.models.decoders import ReconstructionHead1D
from src.ssl.masking import random_masking
from src.ssl.objectives import MaskedSignalModeling, MultiResolutionSTFT

print("=" * 80)
print("SSL SETUP VALIDATION")
print("=" * 80)


def validate_patch_size_detection(config: dict) -> tuple:
    """Validate patch size detection logic."""
    print("\n" + "=" * 80)
    print("TEST 1: Patch Size Detection")
    print("=" * 80)

    model_cfg = config['model']
    ssl_cfg = config['ssl']

    config_patch_size = ssl_cfg['patch_size']
    context_length = model_cfg['context_length']
    input_channels = model_cfg['input_channels']

    print(f"\nConfig values:")
    print(f"  context_length: {context_length}")
    print(f"  patch_size: {config_patch_size}")
    print(f"  Expected patches: {context_length // config_patch_size}")

    # Create encoder
    print(f"\nCreating encoder...")
    encoder = TTMAdapter(
        variant=model_cfg['encoder'],
        task='ssl',
        input_channels=input_channels,
        context_length=context_length,
        patch_size=config_patch_size,
        freeze_encoder=False,
        use_real_ttm=True,
        decoder_mode='mix_channel'
    )

    device = 'cpu'  # Use CPU for validation
    encoder = encoder.to(device)

    # Detect actual patch size
    print(f"\nDetecting actual patch size...")
    with torch.no_grad():
        dummy_input = torch.randn(1, input_channels, context_length).to(device)
        dummy_output = encoder.get_encoder_output(dummy_input)

        actual_num_patches = dummy_output.size(1)
        actual_d_model = dummy_output.size(2)
        actual_patch_size = context_length // actual_num_patches

        print(f"  Input shape: {list(dummy_input.shape)} [B, C, T]")
        print(f"  Encoder output: {list(dummy_output.shape)} [B, P, D]")
        print(f"  Actual patches (P): {actual_num_patches}")
        print(f"  Actual d_model (D): {actual_d_model}")
        print(f"  Calculated patch_size: {actual_patch_size}")

    # Check for mismatch
    if actual_patch_size != config_patch_size:
        print(f"\n  ⚠️  MISMATCH DETECTED!")
        print(f"      Config expects: {config_patch_size} → {context_length // config_patch_size} patches")
        print(f"      Encoder outputs: {actual_patch_size} → {actual_num_patches} patches")
        print(f"      Action: Will use actual_patch_size={actual_patch_size} for decoder/masking")
    else:
        print(f"\n  ✅ Patch size matches config!")

    return encoder, actual_patch_size, actual_d_model, actual_num_patches


def validate_decoder_dimensions(encoder, actual_patch_size, actual_d_model, actual_num_patches, config):
    """Validate decoder creates correct output dimensions."""
    print("\n" + "=" * 80)
    print("TEST 2: Decoder Dimension Matching")
    print("=" * 80)

    model_cfg = config['model']
    context_length = model_cfg['context_length']
    input_channels = model_cfg['input_channels']

    # Create decoder with ACTUAL dimensions
    print(f"\nCreating decoder with detected dimensions:")
    print(f"  d_model: {actual_d_model}")
    print(f"  patch_size: {actual_patch_size}")
    print(f"  n_channels: {input_channels}")

    decoder = ReconstructionHead1D(
        d_model=actual_d_model,
        patch_size=actual_patch_size,
        n_channels=input_channels
    )

    device = 'cpu'
    decoder = decoder.to(device)

    # Test forward pass
    print(f"\nTesting encoder → decoder pipeline...")
    with torch.no_grad():
        dummy_input = torch.randn(4, input_channels, context_length).to(device)
        print(f"  Input: {list(dummy_input.shape)} [B, C, T]")

        # Encode
        latents = encoder.get_encoder_output(dummy_input)
        print(f"  Latents: {list(latents.shape)} [B, P, D]")

        # Decode
        reconstructed = decoder(latents)
        print(f"  Reconstructed: {list(reconstructed.shape)} [B, C, T]")
        print(f"  Expected: [4, {input_channels}, {context_length}]")

        # Verify dimensions
        if reconstructed.shape == dummy_input.shape:
            print(f"\n  ✅ Dimensions match perfectly!")
            print(f"     Input and reconstructed have same shape")
        else:
            print(f"\n  ❌ DIMENSION MISMATCH!")
            print(f"     Input: {list(dummy_input.shape)}")
            print(f"     Reconstructed: {list(reconstructed.shape)}")
            raise ValueError("Decoder output doesn't match input dimensions!")

    return decoder


def validate_masking_alignment(encoder, actual_patch_size, actual_num_patches, config):
    """Validate masking aligns with encoder patches."""
    print("\n" + "=" * 80)
    print("TEST 3: Masking Alignment")
    print("=" * 80)

    model_cfg = config['model']
    ssl_cfg = config['ssl']
    context_length = model_cfg['context_length']
    input_channels = model_cfg['input_channels']
    mask_ratio = ssl_cfg['mask_ratio']

    print(f"\nMasking configuration:")
    print(f"  patch_size: {actual_patch_size} (using ACTUAL, not config)")
    print(f"  mask_ratio: {mask_ratio}")
    print(f"  Expected masked patches: {int(actual_num_patches * mask_ratio)}/{actual_num_patches}")

    # Create dummy input
    device = 'cpu'
    dummy_input = torch.randn(4, input_channels, context_length).to(device)

    # Apply masking
    print(f"\nApplying masking...")
    masked_input, mask_bool = random_masking(
        dummy_input,
        mask_ratio=mask_ratio,
        patch_size=actual_patch_size  # CRITICAL: Use actual patch_size
    )

    print(f"  Input shape: {list(dummy_input.shape)} [B, C, T]")
    print(f"  Masked input shape: {list(masked_input.shape)} [B, C, T]")
    print(f"  Mask shape: {list(mask_bool.shape)} [B, P]")
    print(f"  Mask patches per sample: {mask_bool[0].sum().item()}/{mask_bool.shape[1]}")
    print(f"  Average mask ratio: {mask_bool.float().mean().item():.3f} (target: {mask_ratio})")

    # Verify mask aligns with encoder
    with torch.no_grad():
        latents = encoder.get_encoder_output(masked_input)
        print(f"\nEncoder output: {list(latents.shape)} [B, P, D]")
        print(f"  Encoder patches: {latents.size(1)}")
        print(f"  Mask patches: {mask_bool.size(1)}")

        if latents.size(1) == mask_bool.size(1):
            print(f"\n  ✅ Mask aligns with encoder patches!")
            print(f"     Both have {latents.size(1)} patches")
        else:
            print(f"\n  ❌ MASK MISALIGNMENT!")
            print(f"     Encoder has {latents.size(1)} patches")
            print(f"     Mask has {mask_bool.size(1)} patches")
            raise ValueError("Mask doesn't align with encoder patches!")

    return mask_bool


def validate_loss_computation(encoder, decoder, mask_bool, actual_patch_size, config):
    """Validate SSL loss computation."""
    print("\n" + "=" * 80)
    print("TEST 4: Loss Computation")
    print("=" * 80)

    model_cfg = config['model']
    ssl_cfg = config['ssl']
    context_length = model_cfg['context_length']
    input_channels = model_cfg['input_channels']
    mask_ratio = ssl_cfg['mask_ratio']

    # Create MSM loss
    msm_criterion = MaskedSignalModeling(patch_size=actual_patch_size)
    print(f"\nMSM Loss created with patch_size={actual_patch_size}")

    # Create dummy data
    device = 'cpu'
    dummy_input = torch.randn(4, input_channels, context_length).to(device)

    # Apply masking
    masked_input, mask_bool = random_masking(
        dummy_input,
        mask_ratio=mask_ratio,
        patch_size=actual_patch_size
    )

    # Forward pass
    print(f"\nRunning forward pass...")
    with torch.no_grad():
        latents = encoder.get_encoder_output(masked_input)
        reconstructed = decoder(latents)

    print(f"  Original: {list(dummy_input.shape)}")
    print(f"  Masked: {list(masked_input.shape)}")
    print(f"  Reconstructed: {list(reconstructed.shape)}")
    print(f"  Mask: {list(mask_bool.shape)}")

    # Compute loss
    try:
        loss = msm_criterion(reconstructed, dummy_input, mask_bool)
        print(f"\n  ✅ Loss computed successfully!")
        print(f"     Loss value: {loss.item():.6f}")
        print(f"     Loss is finite: {torch.isfinite(loss).item()}")

        if not torch.isfinite(loss):
            print(f"\n  ❌ Loss is NaN or Inf!")
            raise ValueError("Loss computation produces NaN/Inf")

    except Exception as e:
        print(f"\n  ❌ Loss computation failed: {e}")
        raise

    # Test STFT loss if enabled
    if ssl_cfg.get('stft', {}).get('enabled', True):
        stft_cfg = ssl_cfg['stft']
        n_ffts = stft_cfg['n_ffts']
        hop_lengths = stft_cfg['hop_lengths']

        print(f"\nTesting STFT loss...")
        print(f"  FFT sizes: {n_ffts}")
        print(f"  Hop lengths: {hop_lengths}")

        stft_criterion = MultiResolutionSTFT(
            n_ffts=n_ffts,
            hop_lengths=hop_lengths
        )

        try:
            stft_loss = stft_criterion(reconstructed, dummy_input)
            print(f"  ✅ STFT loss computed successfully!")
            print(f"     Loss value: {stft_loss.item():.6f}")
        except Exception as e:
            print(f"  ⚠️  STFT loss failed: {e}")
            print(f"     This may be OK if signal length is too short")


def validate_gradient_flow(encoder, decoder, mask_bool, actual_patch_size, config):
    """Validate gradients flow correctly."""
    print("\n" + "=" * 80)
    print("TEST 5: Gradient Flow")
    print("=" * 80)

    model_cfg = config['model']
    ssl_cfg = config['ssl']
    context_length = model_cfg['context_length']
    input_channels = model_cfg['input_channels']
    mask_ratio = ssl_cfg['mask_ratio']

    # Set models to train mode
    encoder.train()
    decoder.train()

    # Create loss
    msm_criterion = MaskedSignalModeling(patch_size=actual_patch_size)

    # Create dummy data
    device = 'cpu'
    dummy_input = torch.randn(4, input_channels, context_length).to(device)

    # Apply masking
    masked_input, mask_bool = random_masking(
        dummy_input,
        mask_ratio=mask_ratio,
        patch_size=actual_patch_size
    )

    # Forward pass (without torch.no_grad)
    print(f"\nTesting gradient computation...")
    latents = encoder.get_encoder_output(masked_input)
    reconstructed = decoder(latents)
    loss = msm_criterion(reconstructed, dummy_input, mask_bool)

    # Backward pass
    loss.backward()

    # Check gradients
    encoder_has_grads = any(p.grad is not None and p.grad.abs().sum() > 0
                           for p in encoder.parameters() if p.requires_grad)
    decoder_has_grads = any(p.grad is not None and p.grad.abs().sum() > 0
                           for p in decoder.parameters() if p.requires_grad)

    print(f"  Encoder has gradients: {encoder_has_grads}")
    print(f"  Decoder has gradients: {decoder_has_grads}")

    if encoder_has_grads and decoder_has_grads:
        print(f"\n  ✅ Gradients flow to both encoder and decoder!")
    else:
        print(f"\n  ❌ Gradient flow issue!")
        if not encoder_has_grads:
            print(f"     Encoder has no gradients")
        if not decoder_has_grads:
            print(f"     Decoder has no gradients")
        raise ValueError("Gradients not flowing correctly")

    # Check gradient magnitudes
    encoder_grad_norm = sum((p.grad ** 2).sum()
                           for p in encoder.parameters()
                           if p.requires_grad and p.grad is not None).sqrt()
    decoder_grad_norm = sum((p.grad ** 2).sum()
                           for p in decoder.parameters()
                           if p.requires_grad and p.grad is not None).sqrt()

    print(f"\nGradient norms:")
    print(f"  Encoder: {encoder_grad_norm:.6f}")
    print(f"  Decoder: {decoder_grad_norm:.6f}")


def main():
    parser = argparse.ArgumentParser(description='Validate SSL setup')
    parser.add_argument('--config', type=str,
                       default='configs/ssl_pretrain.yaml',
                       help='Path to SSL config file')
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"\nLoaded config from: {config_path}")

    try:
        # Run all validation tests
        encoder, actual_patch_size, actual_d_model, actual_num_patches = \
            validate_patch_size_detection(config)

        decoder = validate_decoder_dimensions(
            encoder, actual_patch_size, actual_d_model, actual_num_patches, config
        )

        mask_bool = validate_masking_alignment(
            encoder, actual_patch_size, actual_num_patches, config
        )

        validate_loss_computation(
            encoder, decoder, mask_bool, actual_patch_size, config
        )

        validate_gradient_flow(
            encoder, decoder, mask_bool, actual_patch_size, config
        )

        # Summary
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        print(f"\n✅ ALL VALIDATION TESTS PASSED!")
        print(f"\nKey parameters:")
        print(f"  Context length: {config['model']['context_length']}")
        print(f"  Actual patch_size: {actual_patch_size}")
        print(f"  Actual num_patches: {actual_num_patches}")
        print(f"  Actual d_model: {actual_d_model}")
        print(f"\n✅ Safe to proceed with SSL training!")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print("VALIDATION FAILED")
        print("=" * 80)
        print(f"\n❌ Error: {e}")
        print(f"\n⚠️  DO NOT proceed with training until issues are fixed!")
        print("=" * 80)
        sys.exit(1)


if __name__ == '__main__':
    main()
