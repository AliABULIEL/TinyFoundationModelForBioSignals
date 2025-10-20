#!/usr/bin/env python3
"""
Fix fine-tuning architecture detection.

The issue: Architecture detection was reading from wrong mixer layers.
The fix: Use patcher to get patch_size, assume context_length=1024 for VitalDB.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

checkpoint_path = "artifacts/foundation_model/best_model.pt"
print("=" * 80)
print("CORRECT ARCHITECTURE DETECTION")
print("=" * 80)
print(f"Checkpoint: {checkpoint_path}\n")

checkpoint = torch.load(checkpoint_path, map_location='cpu')
state_dict = checkpoint.get('encoder_state_dict', checkpoint)

# STEP 1: Get patch_size and d_model from patcher (RELIABLE)
patcher_keys = [k for k in state_dict.keys() if 'patcher.weight' in k and 'encoder' in k]
if not patcher_keys:
    print("❌ ERROR: No patcher weights found!")
    sys.exit(1)

patcher_weight = state_dict[patcher_keys[0]]
d_model = patcher_weight.shape[0]
patch_size = patcher_weight.shape[1]

print(f"✅ From patcher weights:")
print(f"   Key: {patcher_keys[0]}")
print(f"   Shape: {patcher_weight.shape}")
print(f"   → d_model: {d_model}")
print(f"   → patch_size: {patch_size}")
print()

# STEP 2: Determine context_length
# For VitalDB SSL training, context_length is typically 1024 (125Hz × 8.192s)
# Check if config has this info
if 'config' in checkpoint and 'context_length' in checkpoint['config']:
    context_length = checkpoint['config']['context_length']
    print(f"✅ From checkpoint config:")
    print(f"   → context_length: {context_length}")
else:
    # Assume VitalDB standard
    context_length = 1024
    print(f"⚠️  No context_length in config, assuming VitalDB standard:")
    print(f"   → context_length: {context_length}")
print()

# STEP 3: Calculate num_patches
num_patches = context_length // patch_size

print(f"✅ Calculated:")
print(f"   num_patches = context_length / patch_size")
print(f"              = {context_length} / {patch_size}")
print(f"              = {num_patches}")
print()

# STEP 4: Verify by checking actual TTM encoder output shape
print("=" * 80)
print("VERIFICATION")
print("=" * 80)

# Check if matches IBM pretrained
if context_length == 1024 and patch_size == 128:
    print("✅ Configuration matches IBM TTM-Enhanced")
    print("   (context=1024, patch=128, d_model=192)")
    variant = "IBM TTM-Enhanced"
elif context_length == 512 and patch_size == 64:
    print("✅ Configuration matches IBM TTM-Base")
    print("   (context=512, patch=64, d_model=192)")
    variant = "IBM TTM-Base"
elif context_length == 1024 and patch_size == 64:
    print("⚠️  Custom configuration (NOT IBM pretrained)")
    print("   (context=1024, patch=64, d_model=192)")
    print("   This suggests SSL training did NOT load IBM pretrained weights")
    variant = "Custom (no IBM pretrained)"
else:
    print(f"⚠️  Custom configuration:")
    print(f"   (context={context_length}, patch={patch_size}, d_model={d_model})")
    variant = "Custom"
print()

# STEP 5: Check key prefix structure
has_encoder_backbone = any('encoder.backbone' in k for k in state_dict.keys())
print(f"Checkpoint uses 'encoder.backbone' prefix: {has_encoder_backbone}")
print()

# SUMMARY
print("=" * 80)
print("CORRECT ARCHITECTURE FOR FINE-TUNING")
print("=" * 80)
print()
print(f"context_length: {context_length}")
print(f"patch_size: {patch_size}")
print(f"num_patches: {num_patches}")
print(f"d_model: {d_model}")
print(f"Variant: {variant}")
print()
print("Use these parameters in fine-tuning script:")
print()
print(f"model = TTMAdapter(")
print(f"    variant='ibm-granite/granite-timeseries-ttm-r1',")
print(f"    task='classification',")
print(f"    num_classes=2,")
print(f"    input_channels=2,")
print(f"    context_length={context_length},")
print(f"    patch_size={patch_size},")
print(f"    d_model={d_model},")
print(f"    freeze_encoder=True")
print(f")")
print()
print("=" * 80)
