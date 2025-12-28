#!/usr/bin/env python3
"""Diagnose checkpoint architecture by analyzing weight shapes."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse


def analyze_checkpoint(checkpoint_path: str):
    """Analyze checkpoint to determine actual architecture."""
    print("\n" + "="*80)
    print("CHECKPOINT ARCHITECTURE DIAGNOSIS")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}\n")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Get encoder state dict
    if 'encoder_state_dict' in checkpoint:
        state_dict = checkpoint['encoder_state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    print(f"✓ Found {len(state_dict)} parameters in checkpoint\n")

    # Analyze key weight shapes
    print("="*80)
    print("CRITICAL WEIGHT ANALYSIS")
    print("="*80)

    # 1. Analyze patcher (input projection)
    patcher_keys = [k for k in state_dict.keys() if 'patcher' in k and 'weight' in k]
    if patcher_keys:
        print("\n1. PATCHER (Input Projection):")
        for key in patcher_keys:
            shape = state_dict[key].shape
            print(f"   {key}: {shape}")
            if len(shape) == 2:
                d_model, patch_dim = shape
                print(f"      → d_model = {d_model}")
                print(f"      → patch_size * input_channels = {patch_dim}")

    # 2. Analyze TTM backbone encoder layers
    encoder_keys = [k for k in state_dict.keys() if 'backbone.encoder' in k]
    if encoder_keys:
        print("\n2. TTM ENCODER LAYERS:")

        # Find patch mixer MLP weights
        patch_mixer_keys = [k for k in encoder_keys if 'patch_mixer.mlp.fc1.weight' in k]
        if patch_mixer_keys:
            print("   Patch Mixer MLP (fc1):")
            for key in patch_mixer_keys[:2]:  # Show first 2 layers
                shape = state_dict[key].shape
                print(f"   {key}: {shape}")
                if len(shape) == 2:
                    out_features, in_features = shape
                    print(f"      → Input features (num_patches): {in_features}")
                    print(f"      → Output features (expansion): {out_features}")
                    expansion = out_features / in_features
                    print(f"      → Expansion factor: {expansion:.1f}x")

        # Find feature mixer MLP weights
        feature_mixer_keys = [k for k in encoder_keys if 'feature_mixer.mlp.fc1.weight' in k]
        if feature_mixer_keys:
            print("\n   Feature Mixer MLP (fc1):")
            for key in feature_mixer_keys[:2]:
                shape = state_dict[key].shape
                print(f"   {key}: {shape}")
                if len(shape) == 2:
                    out_features, in_features = shape
                    print(f"      → Input features (d_model): {in_features}")
                    print(f"      → Output features: {out_features}")

    # 3. Analyze decoder head
    head_keys = [k for k in state_dict.keys() if 'head' in k and 'weight' in k]
    if head_keys:
        print("\n3. DECODER HEAD:")
        for key in head_keys:
            shape = state_dict[key].shape
            print(f"   {key}: {shape}")
            if 'head.weight' in key and len(shape) == 2:
                out_dim, in_dim = shape
                print(f"      → Input dim: {in_dim}")
                print(f"      → Output dim: {out_dim}")

                # Try to infer architecture
                # Typical: in_dim = num_patches * d_model
                # Or: in_dim = num_patches * input_channels * patch_size
                print(f"\n      Possible configurations:")

                # If we know d_model from patcher
                if patcher_keys:
                    patcher_shape = state_dict[patcher_keys[0]].shape
                    if len(patcher_shape) == 2:
                        d_model = patcher_shape[0]

                        # Case 1: in_dim = num_patches * d_model
                        if in_dim % d_model == 0:
                            num_patches = in_dim // d_model
                            print(f"      → num_patches = {num_patches} (if head input is [num_patches, d_model] flattened)")

                        # Case 2: Try different arrangements
                        patch_dim = patcher_shape[1]
                        print(f"      → patch_size * input_channels = {patch_dim}")

                        # Assume 2 channels (PPG+ECG)
                        if patch_dim % 2 == 0:
                            patch_size = patch_dim // 2
                            print(f"      → If 2 channels: patch_size = {patch_size}")

    # 4. FINAL INFERENCE
    print("\n" + "="*80)
    print("ARCHITECTURE INFERENCE")
    print("="*80)

    # Get detected values
    d_model = None
    patch_size = None
    num_patches = None
    input_channels = 2  # Assume PPG+ECG

    # From patcher
    if patcher_keys:
        patcher_shape = state_dict[patcher_keys[0]].shape
        if len(patcher_shape) == 2:
            d_model = patcher_shape[0]
            patch_dim = patcher_shape[1]
            if patch_dim % input_channels == 0:
                patch_size = patch_dim // input_channels

    # From patch mixer
    patch_mixer_keys = [k for k in state_dict.keys() if 'backbone.encoder' in k and 'patch_mixer.mlp.fc1.weight' in k]
    if patch_mixer_keys:
        shape = state_dict[patch_mixer_keys[0]].shape
        if len(shape) == 2:
            num_patches = shape[1]  # Input features to patch mixer MLP

    if d_model and patch_size and num_patches:
        context_length = num_patches * patch_size

        print(f"\n✓ DETECTED ARCHITECTURE:")
        print(f"  d_model: {d_model}")
        print(f"  patch_size: {patch_size}")
        print(f"  num_patches: {num_patches}")
        print(f"  context_length: {context_length}")
        print(f"  input_channels: {input_channels}")

        print(f"\n✓ RECOMMENDED FIX:")
        print(f"  Update checkpoint_utils.py fallback detection to use:")
        print(f"  - d_model from patcher.weight[0]")
        print(f"  - num_patches from backbone.encoder.*.patch_mixer.mlp.fc1.weight[1]")
        print(f"  - patch_size = (patcher.weight[1] // input_channels)")
        print(f"  - context_length = num_patches * patch_size")
    else:
        print("\n❌ Could not fully detect architecture")
        print(f"  d_model: {d_model}")
        print(f"  patch_size: {patch_size}")
        print(f"  num_patches: {num_patches}")

    # 5. Check metadata
    print("\n" + "="*80)
    print("CHECKPOINT METADATA")
    print("="*80)

    if 'architecture' in checkpoint:
        print("✓ Checkpoint has NEW architecture metadata:")
        for key, value in checkpoint['architecture'].items():
            print(f"  {key}: {value}")
    else:
        print("⚠️  Checkpoint uses OLD format (no architecture metadata)")

    if 'metrics' in checkpoint:
        print("\n✓ Checkpoint metrics:")
        for key, value in checkpoint['metrics'].items():
            print(f"  {key}: {value}")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Diagnose checkpoint architecture")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    args = parser.parse_args()

    analyze_checkpoint(args.checkpoint)


if __name__ == '__main__':
    main()
