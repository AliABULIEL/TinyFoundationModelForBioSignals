#!/usr/bin/env python3
"""Inspect SSL checkpoint to determine actual architecture."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse

def inspect_checkpoint(checkpoint_path):
    """Inspect checkpoint architecture from weight shapes."""
    print("\n" + "=" * 70)
    print("INSPECTING SSL CHECKPOINT")
    print("=" * 70)

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    print(f"\nCheckpoint keys: {list(checkpoint.keys())}")

    # Check config
    if 'config' in checkpoint:
        print("\n📋 Config stored in checkpoint:")
        config = checkpoint['config']
        for key, value in config.items():
            print(f"  {key}: {value}")

    # Inspect model state dict - handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("\n  Using 'model_state_dict'")
    elif 'encoder_state_dict' in checkpoint:
        state_dict = checkpoint['encoder_state_dict']
        print("\n  Using 'encoder_state_dict' (SSL checkpoint)")
    else:
        state_dict = checkpoint
        print("\n  Using raw checkpoint (assuming state dict)")

    print("\n🔍 Model architecture from weight shapes:")
    print(f"  Total keys in state dict: {len(state_dict)}")

    # Show first few keys to understand structure
    print("\n  Sample keys:")
    for i, key in enumerate(list(state_dict.keys())[:10]):
        print(f"    {key}")

    # Find patch_mixer weights to determine num_patches
    patch_mixer_keys = [k for k in state_dict.keys() if 'patch_mixer' in k and 'mlp.fc1.weight' in k]

    if patch_mixer_keys:
        # Get first patch_mixer MLP weight
        key = patch_mixer_keys[0]
        weight = state_dict[key]
        print(f"\n  ✓ Found patch_mixer weights!")
        print(f"\n  {key}")
        print(f"    Shape: {weight.shape}")

        # For TTM, the patch_mixer MLP input dimension = num_patches
        num_patches = weight.shape[1]
        print(f"\n  ✓ Detected num_patches: {num_patches}")
    else:
        print(f"\n  ❌ No patch_mixer keys found")
        print(f"\n  All keys in state_dict:")
        for key in sorted(state_dict.keys()):
            print(f"    {key}")
        return

    # Find embedding or input projection to determine d_model
    embed_keys = [k for k in state_dict.keys() if 'backbone.encoder' in k and 'fc1.weight' in k]
    if embed_keys:
        key = embed_keys[0]
        weight = state_dict[key]
        print(f"\n  {key}")
        print(f"    Shape: {weight.shape}")

    # Calculate implied context_length
    if 'config' in checkpoint and patch_mixer_keys:
        patch_size = checkpoint['config'].get('patch_size', None)
        if patch_size:
            implied_context_length = num_patches * patch_size
            print(f"\n  Implied context_length: {num_patches} patches × {patch_size} = {implied_context_length} samples")

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS:")
    print("=" * 70)

    if patch_mixer_keys:
        print(f"\nWhen loading this checkpoint for fine-tuning, use:")
        print(f"  - num_patches: {num_patches}")
        if 'config' in checkpoint:
            stored_context = checkpoint['config'].get('context_length', 'unknown')
            stored_patch = checkpoint['config'].get('patch_size', 'unknown')
            print(f"\nNote: Config says context_length={stored_context}, patch_size={stored_patch}")
            print(f"      But actual weights expect num_patches={num_patches}")

            if stored_patch and stored_patch != 'unknown':
                correct_context = num_patches * stored_patch
                print(f"      Correct context_length should be: {correct_context}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inspect SSL checkpoint architecture')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    args = parser.parse_args()

    inspect_checkpoint(args.checkpoint)
