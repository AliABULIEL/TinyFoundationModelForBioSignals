#!/usr/bin/env python3
"""
Inspect SSL checkpoint to understand exact architecture.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

print("=" * 80)
print("INSPECTING SSL CHECKPOINT")
print("=" * 80)

checkpoint_path = "artifacts/foundation_model/best_model.pt"
print(f"Loading: {checkpoint_path}\n")

checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("Checkpoint keys:", list(checkpoint.keys()))
print()

# Get encoder state dict
if 'encoder_state_dict' in checkpoint:
    state_dict = checkpoint['encoder_state_dict']
    print("Found 'encoder_state_dict'")
else:
    state_dict = checkpoint
    print("Using checkpoint directly as state_dict")

print(f"Total keys in state_dict: {len(state_dict)}")
print()

# Analyze key prefixes
prefixes = {}
for key in state_dict.keys():
    prefix = key.split('.')[0]
    if prefix not in prefixes:
        prefixes[prefix] = []
    prefixes[prefix].append(key)

print("Key prefixes:")
for prefix, keys in sorted(prefixes.items()):
    print(f"  {prefix}: {len(keys)} keys")
print()

# Find critical architecture parameters
print("=" * 80)
print("DETECTING ARCHITECTURE PARAMETERS")
print("=" * 80)

# 1. Check patcher (converts patches to embeddings)
patcher_keys = [k for k in state_dict.keys() if 'patcher.weight' in k and 'encoder' in k]
print(f"\n1. Patcher keys ({len(patcher_keys)}):")
for key in patcher_keys:
    shape = state_dict[key].shape
    print(f"   {key}")
    print(f"     Shape: {shape}")
    if len(shape) == 2:
        d_model, patch_input_dim = shape
        print(f"     d_model: {d_model}")
        print(f"     patch_input_dim: {patch_input_dim}")
        # TTM patcher operates per channel
        patch_size = patch_input_dim
        print(f"     → patch_size: {patch_size}")

# 2. Check decoder patch_mixer (processes patches)
decoder_patch_mixer_keys = [k for k in state_dict.keys()
                            if 'decoder' in k and 'patch_mixer' in k and 'mlp.fc1.weight' in k]
print(f"\n2. Decoder patch_mixer keys ({len(decoder_patch_mixer_keys)}):")
for key in decoder_patch_mixer_keys:
    shape = state_dict[key].shape
    print(f"   {key}")
    print(f"     Shape: {shape}")
    if len(shape) == 2:
        hidden_dim, input_dim = shape
        print(f"     hidden_dim: {hidden_dim}")
        print(f"     input_dim: {input_dim}")

        # Check if this is encoder or SSL decoder
        if 'encoder.decoder' in key:
            print(f"     → SSL decoder: num_patches = {input_dim}")
        elif 'encoder.backbone' in key or 'backbone.encoder' in key:
            print(f"     → TTM encoder decoder: num_patches = {input_dim}")

# 3. Check backbone encoder patch_mixer (should show actual num_patches)
encoder_patch_mixer_keys = [k for k in state_dict.keys()
                            if 'backbone.encoder' in k and 'patch_mixer' in k and 'mlp.fc1.weight' in k]
print(f"\n3. Backbone encoder patch_mixer keys ({len(encoder_patch_mixer_keys)}):")
for key in encoder_patch_mixer_keys:
    shape = state_dict[key].shape
    print(f"   {key}")
    print(f"     Shape: {shape}")
    if len(shape) == 2:
        hidden_dim, num_patches = shape
        print(f"     hidden_dim: {hidden_dim}")
        print(f"     → num_patches: {num_patches}")

# Summary
print("\n" + "=" * 80)
print("ARCHITECTURE SUMMARY")
print("=" * 80)

# Get d_model from patcher
patcher_key = [k for k in state_dict.keys() if 'patcher.weight' in k and 'backbone.encoder' in k]
if patcher_key:
    patcher_shape = state_dict[patcher_key[0]].shape
    d_model = patcher_shape[0]
    patch_size = patcher_shape[1]
    print(f"d_model: {d_model}")
    print(f"patch_size: {patch_size}")

    # Check context_length from config if available
    if 'config' in checkpoint:
        config = checkpoint['config']
        context_length = config.get('context_length', 'unknown')
        print(f"context_length: {context_length}")
        if context_length != 'unknown':
            num_patches = context_length // patch_size
            print(f"→ num_patches: {num_patches}")
    else:
        # Try to infer from patch_mixer
        mixer_keys = [k for k in state_dict.keys()
                      if 'backbone.encoder.mixers.0.patch_mixer.mlp.fc1.weight' in k]
        if mixer_keys:
            mixer_shape = state_dict[mixer_keys[0]].shape
            num_patches = mixer_shape[1]
            context_length = num_patches * patch_size
            print(f"→ num_patches: {num_patches} (from patch_mixer)")
            print(f"→ context_length: {context_length} (inferred)")

print("\n" + "=" * 80)
print("KEY SAMPLES")
print("=" * 80)

# Show first few keys from each category
categories = {
    'backbone.encoder': [],
    'encoder.backbone': [],
    'encoder.decoder': [],
    'decoder': [],
    'head': [],
}

for key in state_dict.keys():
    for category in categories:
        if category in key:
            categories[category].append(key)
            break

for category, keys in categories.items():
    if keys:
        print(f"\n{category} ({len(keys)} keys):")
        for key in keys[:3]:
            print(f"  {key}")
        if len(keys) > 3:
            print(f"  ... and {len(keys) - 3} more")

print("\n" + "=" * 80)
