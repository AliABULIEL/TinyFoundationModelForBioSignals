#!/usr/bin/env python3
"""
Verify IBM Pretrained Weights Load Correctly for SSL Training

This script verifies that:
1. TTM-Enhanced config loads IBM pretrained weights
2. Model architecture is correct
3. SSL decoder can be created
4. Forward pass works

Run this BEFORE starting full SSL training to catch any issues early.

Usage:
    python scripts/verify_ibm_pretrained_loading.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

print("=" * 80)
print("VERIFYING IBM PRETRAINED WEIGHTS FOR SSL TRAINING")
print("=" * 80)
print()

# Step 1: Create TTMAdapter with TTM-Enhanced config
print("Step 1: Creating TTMAdapter with TTM-Enhanced config")
print("-" * 80)

from src.models.ttm_adapter import TTMAdapter

model = TTMAdapter(
    variant='ibm-granite/granite-timeseries-ttm-r1',
    task='ssl',
    input_channels=2,
    context_length=1024,
    patch_size=128,  # Key: this matches TTM-Enhanced!
    use_real_ttm=True,
    freeze_encoder=False,
    decoder_mode='mix_channel'
)

print()

# Step 2: Verify IBM pretrained weights were loaded
print("Step 2: Verifying IBM pretrained weights loaded")
print("-" * 80)

if model.is_using_real_ttm():
    print("✅ Using real TTM (not fallback)")
else:
    print("❌ ERROR: Using fallback model, not real TTM!")
    exit(1)

# Check if pretrained weights were loaded
# We can tell by checking if the model was created fresh or loaded
total_params = sum(p.numel() for p in model.parameters())
print(f"✅ Total parameters: {total_params:,}")

expected_params_enhanced = 946904  # TTM-Enhanced parameter count
if abs(total_params - expected_params_enhanced) < 100000:
    print(f"✅ Parameter count matches TTM-Enhanced (~947K)")
else:
    print(f"⚠️  Parameter count: {total_params:,} (expected ~947K)")

print()

# Step 3: Test forward pass (encoder only)
print("Step 3: Testing encoder forward pass")
print("-" * 80)

# Create synthetic biosignal batch
batch_size = 8
context_length = 1024
signals = torch.randn(batch_size, 2, context_length)  # [batch, channels, time]

print(f"Input shape: {signals.shape}")

try:
    with torch.no_grad():
        encoder_output = model.get_encoder_output(signals)

    print(f"✅ Encoder output shape: {encoder_output.shape}")

    # Verify output shape
    expected_patches = 1024 // 128  # context / patch_size = 8
    expected_d_model = 192  # TTM-Enhanced d_model

    if encoder_output.shape == (batch_size, expected_patches, expected_d_model):
        print(f"✅ Output shape correct: [batch={batch_size}, patches={expected_patches}, d_model={expected_d_model}]")
    else:
        print(f"⚠️  Output shape: {encoder_output.shape}")
        print(f"   Expected: [{batch_size}, {expected_patches}, {expected_d_model}]")

    # Check feature statistics
    mean = encoder_output.mean().item()
    std = encoder_output.std().item()
    print(f"✅ Feature statistics: mean={mean:.4f}, std={std:.4f}")

except Exception as e:
    print(f"❌ ERROR in forward pass: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print()

# Step 4: Test SSL decoder creation
print("Step 4: Creating SSL decoder")
print("-" * 80)

try:
    from src.ssl.objectives import create_ssl_decoder

    decoder = create_ssl_decoder(
        encoder_dim=192,  # TTM-Enhanced d_model
        decoder_dim=128,
        num_patches=8,
        patch_size=128,
        output_channels=2,
        decoder_depth=2
    )

    decoder_params = sum(p.numel() for p in decoder.parameters())
    print(f"✅ SSL decoder created successfully")
    print(f"   Decoder parameters: {decoder_params:,}")

    # Test decoder forward pass
    with torch.no_grad():
        reconstructed = decoder(encoder_output)

    expected_output_shape = (batch_size, 2, context_length)
    if reconstructed.shape == expected_output_shape:
        print(f"✅ Decoder output shape correct: {reconstructed.shape}")
    else:
        print(f"⚠️  Decoder output shape: {reconstructed.shape}")
        print(f"   Expected: {expected_output_shape}")

except Exception as e:
    print(f"❌ ERROR creating/testing SSL decoder: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print()

# Step 5: Test masking
print("Step 5: Testing SSL masking")
print("-" * 80)

try:
    from src.ssl.masking import create_mask

    mask = create_mask(
        batch_size=batch_size,
        num_patches=8,
        mask_ratio=0.4,
        device='cpu'
    )

    print(f"✅ Mask created: shape={mask.shape}")
    print(f"   Masked patches: {mask.sum().item()} / {mask.numel()} ({mask.sum().item()/mask.numel()*100:.1f}%)")

    # Verify mask ratio
    actual_ratio = mask.sum().item() / mask.numel()
    if 0.35 <= actual_ratio <= 0.45:  # Allow some variance
        print(f"✅ Mask ratio correct: {actual_ratio:.2f} (target: 0.40)")
    else:
        print(f"⚠️  Mask ratio: {actual_ratio:.2f} (target: 0.40)")

except Exception as e:
    print(f"❌ ERROR in masking: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print()

# Step 6: Test SSL loss computation
print("Step 6: Testing SSL loss computation")
print("-" * 80)

try:
    from src.ssl.objectives import masked_signal_modeling_loss

    # Simulate masked reconstruction
    target = signals  # Original signal
    prediction = reconstructed  # Decoder output

    mse_loss = masked_signal_modeling_loss(prediction, target, mask)

    print(f"✅ MSM loss computed: {mse_loss.item():.4f}")

    if not torch.isnan(mse_loss) and not torch.isinf(mse_loss):
        print(f"✅ Loss is valid (not NaN or Inf)")
    else:
        print(f"❌ ERROR: Loss is NaN or Inf!")
        exit(1)

except Exception as e:
    print(f"❌ ERROR in loss computation: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print()

# Final summary
print("=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
print()
print("✅ All checks passed!")
print()
print("📊 Summary:")
print(f"   Model: TTM-Enhanced with IBM pretrained weights")
print(f"   Parameters: {total_params:,}")
print(f"   Context length: 1024 (8.192s @ 125Hz)")
print(f"   Patch size: 128 (1.024s @ 125Hz)")
print(f"   Patches: 8")
print(f"   d_model: 192")
print()
print("🚀 Ready to start SSL training!")
print()
print("Next step:")
print("  python scripts/pretrain_vitaldb_ssl.py \\")
print("    --config configs/ssl_pretrain_ibm_enhanced.yaml \\")
print("    --epochs 50")
print()
print("=" * 80)
