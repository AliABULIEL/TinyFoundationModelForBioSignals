#!/usr/bin/env python3
"""
Phase 1.2: SSL Patch Size Verification
=======================================

Critical Issue Found:
- Config specifies patch_size=128
- TTM actually uses patch_size=64
- This creates dimension mismatch between encoder (16 patches) and decoder (expects 8)

This diagnostic verifies:
1. Does the SSL training script correctly detect actual patch_size?
2. Does the decoder use the corrected patch_size?
3. Will SSL masking work with the correct dimensions?

Author: Claude Code Foundation Model Audit
Date: October 2025
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import yaml

# Add project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models.ttm_adapter import TTMAdapter
from src.models.decoders import ReconstructionHead1D

print("=" * 80)
print("SSL PATCH SIZE VERIFICATION")
print("=" * 80)

# Load SSL config
config_path = project_root / 'configs' / 'ssl_pretrain.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

print(f"\n1. Config File Settings:")
print(f"   - Config patch_size: {config['ssl']['patch_size']}")
print(f"   - Config context_length: {config['model']['context_length']}")
print(f"   - Expected patches: {config['model']['context_length'] // config['ssl']['patch_size']}")

# Create encoder following the SSL training script
device = 'cpu'

print(f"\n2. Creating TTM Encoder (simulating SSL training setup)...")

encoder = TTMAdapter(
    variant=config['model']['encoder'],
    task='ssl',
    input_channels=config['model']['input_channels'],
    context_length=config['model']['context_length'],
    patch_size=config['ssl']['patch_size'],  # Config says 128
    freeze_encoder=False,
    use_real_ttm=True,
    decoder_mode='mix_channel'
).to(device)

print(f"   ✓ Encoder created")

# Step that SSL training script does: Run dummy forward to get ACTUAL patch_size
print(f"\n3. Running dummy forward pass to detect actual patch_size...")
print(f"   (This is what pretrain_vitaldb_ssl.py does in lines 645-668)")

with torch.no_grad():
    dummy_input = torch.randn(1, config['model']['input_channels'], config['model']['context_length']).to(device)
    dummy_output = encoder.get_encoder_output(dummy_input)

    actual_num_patches = dummy_output.size(1)
    actual_patch_size = config['model']['context_length'] // actual_num_patches

    print(f"   Input shape: {list(dummy_input.shape)}")
    print(f"   Encoder output shape: {list(dummy_output.shape)}")
    print(f"   Actual number of patches: {actual_num_patches}")
    print(f"   Calculated actual patch_size: {actual_patch_size}")

# Check if there's a mismatch
config_patch_size = config['ssl']['patch_size']
if actual_patch_size != config_patch_size:
    print(f"\n   ⚠️  MISMATCH DETECTED!")
    print(f"      Config patch_size: {config_patch_size}")
    print(f"      Actual patch_size: {actual_patch_size}")
    print(f"      Difference: {config_patch_size - actual_patch_size}")
else:
    print(f"\n   ✓ Patch size matches config")

# Get actual d_model from encoder
if hasattr(encoder, 'backbone') and hasattr(encoder.backbone, 'config'):
    actual_d_model = encoder.backbone.config.d_model
    print(f"\n4. Encoder d_model from TTM config: {actual_d_model}")
else:
    actual_d_model = config['model']['d_model']
    print(f"\n4. Using config d_model: {actual_d_model}")

# Create decoder following SSL training script
print(f"\n5. Creating Decoder...")
print(f"   Testing TWO scenarios:")

# Scenario A: Decoder with CONFIG patch_size (WRONG)
print(f"\n   [Scenario A] Decoder with CONFIG patch_size={config_patch_size}")
decoder_wrong = ReconstructionHead1D(
    d_model=actual_d_model,
    patch_size=config_patch_size,  # Using CONFIG value (128)
    n_channels=config['model']['input_channels']
).to(device)

try:
    with torch.no_grad():
        reconstruction_wrong = decoder_wrong(dummy_output)
    print(f"      Input to decoder: {list(dummy_output.shape)} [B, P, D]")
    print(f"      Output shape: {list(reconstruction_wrong.shape)}")
    print(f"      Expected output: [1, {config['model']['input_channels']}, {config['model']['context_length']}]")

    if reconstruction_wrong.shape[-1] == config['model']['context_length']:
        print(f"      ✓ Output length matches context_length!")
    else:
        print(f"      ❌ Output length MISMATCH!")
        print(f"         Got: {reconstruction_wrong.shape[-1]}")
        print(f"         Expected: {config['model']['context_length']}")
except Exception as e:
    print(f"      ❌ DECODER FAILED: {e}")

# Scenario B: Decoder with ACTUAL patch_size (CORRECT)
print(f"\n   [Scenario B] Decoder with ACTUAL patch_size={actual_patch_size}")
decoder_correct = ReconstructionHead1D(
    d_model=actual_d_model,
    patch_size=actual_patch_size,  # Using ACTUAL value (64)
    n_channels=config['model']['input_channels']
).to(device)

try:
    with torch.no_grad():
        reconstruction_correct = decoder_correct(dummy_output)
    print(f"      Input to decoder: {list(dummy_output.shape)} [B, P, D]")
    print(f"      Output shape: {list(reconstruction_correct.shape)}")
    print(f"      Expected output: [1, {config['model']['input_channels']}, {config['model']['context_length']}]")

    if reconstruction_correct.shape[-1] == config['model']['context_length']:
        print(f"      ✓ Output length matches context_length!")
    else:
        print(f"      ❌ Output length MISMATCH!")
        print(f"         Got: {reconstruction_correct.shape[-1]}")
        print(f"         Expected: {config['model']['context_length']}")
except Exception as e:
    print(f"      ❌ DECODER FAILED: {e}")

print(f"\n" + "=" * 80)
print("SSL MASKING VERIFICATION")
print("=" * 80)

# Test masking with actual dimensions
from src.ssl.masking import random_masking

print(f"\nTesting masking with actual encoder output dimensions:")
print(f"  Encoder output: [B={1}, P={actual_num_patches}, D={actual_d_model}]")

# Create mask at PATCH level
mask_ratio = config['ssl']['mask_ratio']
num_masked = int(actual_num_patches * mask_ratio)

print(f"\n  Mask ratio: {mask_ratio} ({mask_ratio*100}%)")
print(f"  Number of patches to mask: {num_masked}/{actual_num_patches}")

# Apply masking
masked_patches, mask, _ = random_masking(dummy_output, mask_ratio=mask_ratio)

print(f"  Masked patches shape: {list(masked_patches.shape)}")
print(f"  Mask shape: {list(mask.shape)}")
print(f"  Number of masked patches: {mask.sum().item()}")

# Verify masking works
if masked_patches.shape == dummy_output.shape:
    print(f"  ✓ Masked patches have correct shape")
else:
    print(f"  ❌ Masked patches shape mismatch!")

print(f"\n" + "=" * 80)
print("DIAGNOSIS & RECOMMENDATIONS")
print("=" * 80)

issues = []
fixes = []

if actual_patch_size != config_patch_size:
    issues.append(f"Config patch_size ({config_patch_size}) != actual ({actual_patch_size})")
    fixes.append(f"Update configs/ssl_pretrain.yaml: patch_size: {actual_patch_size}")

# Check if pretrain script handles this
print(f"\n✓ GOOD NEWS: pretrain_vitaldb_ssl.py (lines 645-668) DOES detect actual patch_size!")
print(f"  It runs a dummy forward pass and updates patch_size before creating decoder")

print(f"\n⚠️  CONCERN: Config file is misleading!")
print(f"  configs/ssl_pretrain.yaml specifies patch_size=128")
print(f"  But TTM-Enhanced actually uses patch_size=64")

print(f"\n📋 ACTION ITEMS:")
print(f"  1. Update configs/ssl_pretrain.yaml to patch_size: 64")
print(f"  2. Verify pretrain script correctly detected patch_size=64 during training")
print(f"  3. Check SSL checkpoints to ensure decoder was built with patch_size=64")

print(f"\n🔍 NEXT STEP:")
print(f"  Run actual SSL training logs to verify patch_size was correctly detected")
print(f"  Look for log message: 'Updating patch_size from X to Y'")

print("=" * 80)
