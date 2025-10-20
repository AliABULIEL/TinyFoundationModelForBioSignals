#!/usr/bin/env python3
"""
Fix checkpoint loading by inspecting and correcting key mismatches.
Run this on Colab to diagnose the issue.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

print("=" * 80)
print("DIAGNOSING CHECKPOINT LOADING ISSUE")
print("=" * 80)

checkpoint_path = "artifacts/foundation_model/best_model.pt"
print(f"\nLoading checkpoint: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location='cpu')
state_dict = checkpoint.get('encoder_state_dict', checkpoint)

print(f"Total keys: {len(state_dict)}\n")

# Analyze key structure
print("=" * 80)
print("KEY STRUCTURE ANALYSIS")
print("=" * 80)

# Count different key patterns
patterns = {
    'backbone.encoder': 0,
    'backbone.decoder': 0,
    'encoder.backbone': 0,
    'encoder.decoder': 0,
    'decoder': 0,
    'head': 0,
}

sample_keys = {pattern: [] for pattern in patterns}

for key in state_dict.keys():
    for pattern in patterns:
        if pattern in key:
            patterns[pattern] += 1
            if len(sample_keys[pattern]) < 3:
                sample_keys[pattern].append(key)
            break

print("\nKey pattern counts:")
for pattern, count in patterns.items():
    if count > 0:
        print(f"  {pattern}: {count} keys")
        for sample in sample_keys[pattern][:2]:
            print(f"    → {sample}")
print()

# Find critical parameters
print("=" * 80)
print("ARCHITECTURE PARAMETERS")
print("=" * 80)

# 1. Find patcher
patcher_keys = [k for k in state_dict.keys() if 'patcher.weight' in k and 'encoder' in k]
print(f"\n1. Patcher keys:")
for key in patcher_keys[:2]:
    shape = state_dict[key].shape
    print(f"   {key}: {shape}")
    if len(shape) == 2:
        d_model, patch_size = shape
        print(f"     → d_model={d_model}, patch_size={patch_size}")

# 2. Find patch_mixer in backbone encoder
mixer_keys = [k for k in state_dict.keys()
              if 'patch_mixer' in k and 'mlp.fc1.weight' in k and 'mixers.0' in k]
print(f"\n2. Patch mixer keys (mixers.0 layer):")
for key in mixer_keys[:3]:
    shape = state_dict[key].shape
    print(f"   {key}: {shape}")
    if len(shape) == 2:
        hidden_dim, num_patches = shape
        print(f"     → hidden_dim={hidden_dim}, num_patches={num_patches}")

# Calculate expected architecture
print("\n" + "=" * 80)
print("EXPECTED ARCHITECTURE")
print("=" * 80)

# Get from patcher
patcher_key = [k for k in state_dict.keys() if 'patcher.weight' in k and 'encoder' in k and 'backbone' in k]
if patcher_key:
    patcher_shape = state_dict[patcher_key[0]].shape
    d_model = patcher_shape[0]
    patch_size = patcher_shape[1]

    # Get num_patches from backbone encoder patch_mixer
    backbone_mixer_keys = [k for k in state_dict.keys()
                          if 'backbone.encoder.mixers.0.patch_mixer.mlp.fc1.weight' in k
                          or 'encoder.backbone.encoder.mixers.0.patch_mixer.mlp.fc1.weight' in k]

    if backbone_mixer_keys:
        mixer_shape = state_dict[backbone_mixer_keys[0]].shape
        num_patches = mixer_shape[1]
        context_length = num_patches * patch_size

        print(f"✓ Detected from checkpoint:")
        print(f"  d_model: {d_model}")
        print(f"  patch_size: {patch_size}")
        print(f"  num_patches: {num_patches}")
        print(f"  context_length: {context_length}")
        print()

        # Check if this is IBM pretrained config
        if context_length == 1024 and patch_size == 128:
            print("✓ This matches TTM-Enhanced (IBM pretrained)")
            print("  Should load IBM weights automatically when creating model")
        elif context_length == 1024 and patch_size == 64:
            print("⚠ This is context=1024, patch=64 (custom config)")
            print("  Won't load IBM pretrained (use patch=128 for IBM)")

        print()

# Check for key prefix issues
print("=" * 80)
print("KEY PREFIX DIAGNOSTIC")
print("=" * 80)

has_encoder_backbone = any('encoder.backbone' in k for k in state_dict.keys())
has_backbone_encoder = any(k.startswith('backbone.encoder') for k in state_dict.keys())

print(f"\nCheckpoint has 'encoder.backbone' prefix: {has_encoder_backbone}")
print(f"Checkpoint has 'backbone.encoder' prefix: {has_backbone_encoder}")

if has_backbone_encoder and not has_encoder_backbone:
    print("\n⚠ WARNING: Key prefix mismatch detected!")
    print("  Checkpoint keys start with 'backbone.'")
    print("  But fine-tuning expects 'encoder.backbone.'")
    print("\n  Solution: Add 'encoder.' prefix when loading")

    print("\n  Showing key transformation:")
    sample = [k for k in state_dict.keys() if k.startswith('backbone.')][0]
    print(f"    Old: {sample}")
    print(f"    New: encoder.{sample}")

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

if patcher_key and backbone_mixer_keys:
    print(f"\nYour SSL checkpoint has:")
    print(f"  Architecture: context={context_length}, patch={patch_size}, d_model={d_model}")
    print(f"  Num patches: {num_patches}")

    if has_backbone_encoder and not has_encoder_backbone:
        print(f"\n❌ KEY PREFIX ISSUE:")
        print(f"  Checkpoint uses 'backbone.' prefix")
        print(f"  Fine-tuning expects 'encoder.backbone.' prefix")
        print(f"\n  Fix: Update finetune_butppg.py to add 'encoder.' prefix")
    else:
        print(f"\n✓ Key prefixes look correct")

    print(f"\n  Use these parameters in fine-tuning:")
    print(f"    context_length={context_length}")
    print(f"    patch_size={patch_size}")
    print(f"    d_model={d_model}")

print("\n" + "=" * 80)
