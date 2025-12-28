#!/usr/bin/env python3
"""
Detect SSL checkpoint architecture from ENCODER backbone weights only.
Ignore SSL decoder which may have different patch configuration.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

checkpoint_path = "artifacts/foundation_model/best_model.pt"
print("=" * 80)
print(f"DETECTING SSL ENCODER ARCHITECTURE")
print("=" * 80)
print(f"Checkpoint: {checkpoint_path}\n")

checkpoint = torch.load(checkpoint_path, map_location='cpu')
state_dict = checkpoint.get('encoder_state_dict', checkpoint)

print(f"Total keys: {len(state_dict)}\n")

# STEP 1: Find encoder backbone patcher (this determines d_model and patch_size)
print("STEP 1: Detect d_model and patch_size from encoder patcher")
print("-" * 80)

patcher_keys = [k for k in state_dict.keys()
                if 'patcher.weight' in k and 'backbone.encoder' in k]

if not patcher_keys:
    print("❌ No patcher weights found with 'backbone.encoder' prefix")
    print("\nTrying alternative patterns...")
    patcher_keys = [k for k in state_dict.keys() if 'patcher.weight' in k]
    print(f"Found patcher keys: {patcher_keys[:3]}")

if patcher_keys:
    patcher_key = patcher_keys[0]
    patcher_weight = state_dict[patcher_key]
    d_model = patcher_weight.shape[0]
    patch_size = patcher_weight.shape[1]  # TTM patcher: per-channel, so input_dim = patch_size

    print(f"✅ Patcher key: {patcher_key}")
    print(f"   Shape: {patcher_weight.shape}")
    print(f"   → d_model: {d_model}")
    print(f"   → patch_size: {patch_size}")
    print()
else:
    print("❌ Could not find patcher weights!")
    d_model = None
    patch_size = None

# STEP 2: Find encoder backbone patch_mixer (this determines num_patches)
print("STEP 2: Detect num_patches from BACKBONE encoder patch_mixer")
print("-" * 80)

# CRITICAL: Use backbone.encoder.mixers (not encoder.decoder!)
backbone_mixer_keys = [k for k in state_dict.keys()
                      if 'backbone.encoder.mixers.0.patch_mixer.mlp.fc1.weight' in k]

if not backbone_mixer_keys:
    print("❌ No backbone encoder patch_mixer found with standard pattern")
    print("\nTrying alternative patterns...")

    # Try finding ANY patch_mixer in backbone encoder
    backbone_mixer_keys = [k for k in state_dict.keys()
                          if 'backbone.encoder' in k and 'patch_mixer' in k and 'mlp.fc1.weight' in k]

    if backbone_mixer_keys:
        print(f"Found alternative patterns: {backbone_mixer_keys[:2]}")

if backbone_mixer_keys:
    mixer_key = backbone_mixer_keys[0]
    mixer_weight = state_dict[mixer_key]
    num_patches = mixer_weight.shape[1]  # Input dimension = num_patches

    print(f"✅ Backbone encoder patch_mixer key: {mixer_key}")
    print(f"   Shape: {mixer_weight.shape}")
    print(f"   → num_patches: {num_patches}")
    print()
else:
    print("❌ Could not find backbone encoder patch_mixer!")
    num_patches = None

# STEP 3: Calculate context_length
print("STEP 3: Calculate context_length")
print("-" * 80)

if patch_size and num_patches:
    context_length = num_patches * patch_size
    print(f"✅ context_length = num_patches × patch_size")
    print(f"              = {num_patches} × {patch_size}")
    print(f"              = {context_length}")
    print()
else:
    context_length = None
    print("❌ Cannot calculate context_length (missing patch_size or num_patches)")
    print()

# STEP 4: Check if matches IBM pretrained variants
print("STEP 4: Check IBM pretrained variant match")
print("-" * 80)

if context_length and patch_size:
    if context_length == 512 and patch_size == 64:
        print("✅ Matches TTM-Base (512, 64, d_model=192)")
        variant = "TTM-Base"
    elif context_length == 1024 and patch_size == 128:
        print("✅ Matches TTM-Enhanced (1024, 128, d_model=192)")
        variant = "TTM-Enhanced"
    elif context_length == 1536 and patch_size == 128:
        print("✅ Matches TTM-Large (1536, 128, d_model=192)")
        variant = "TTM-Large"
    else:
        print(f"⚠️  Custom configuration: ({context_length}, {patch_size}, d_model={d_model})")
        print("   Does NOT match any IBM pretrained variant")
        variant = "Custom"
    print()
else:
    variant = "Unknown"

# STEP 5: Show key prefix structure
print("STEP 5: Analyze key prefix structure")
print("-" * 80)

has_backbone_prefix = any(k.startswith('backbone.') for k in state_dict.keys())
has_encoder_backbone = any('encoder.backbone' in k for k in state_dict.keys())

print(f"Keys start with 'backbone.': {has_backbone_prefix}")
print(f"Keys contain 'encoder.backbone': {has_encoder_backbone}")

if has_backbone_prefix and not has_encoder_backbone:
    print("\n✅ Checkpoint uses 'backbone.' prefix (standard for TTMAdapter)")
    print("   Fine-tuning model should also use 'backbone.' prefix")
elif has_encoder_backbone:
    print("\n⚠️  Checkpoint uses 'encoder.backbone' prefix")
    print("   May need prefix adjustment when loading")
print()

# SUMMARY
print("=" * 80)
print("ARCHITECTURE SUMMARY")
print("=" * 80)

if all([d_model, patch_size, num_patches, context_length]):
    print(f"✅ Successfully detected architecture:")
    print(f"   d_model: {d_model}")
    print(f"   patch_size: {patch_size}")
    print(f"   num_patches: {num_patches}")
    print(f"   context_length: {context_length}")
    print(f"   Variant: {variant}")
    print()
    print(f"Use these parameters for fine-tuning:")
    print(f"   model = TTMAdapter(")
    print(f"       variant='ibm-granite/granite-timeseries-ttm-r1',")
    print(f"       task='classification',")
    print(f"       num_classes=2,")
    print(f"       input_channels=2,")
    print(f"       context_length={context_length},")
    print(f"       patch_size={patch_size},")
    print(f"       d_model={d_model},")
    print(f"       freeze_encoder=True")
    print(f"   )")
else:
    print("❌ Could not fully detect architecture")
    print(f"   d_model: {d_model}")
    print(f"   patch_size: {patch_size}")
    print(f"   num_patches: {num_patches}")
    print(f"   context_length: {context_length}")
    print("\nRun the diagnostic script for more details:")
    print("  python3 scripts/fix_checkpoint_loading.py")

print("=" * 80)
