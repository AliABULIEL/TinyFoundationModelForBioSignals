#!/usr/bin/env python3
"""Check architecture of saved SSL checkpoints.

This script loads a checkpoint and analyzes its architecture to detect
potential issues like the 64 vs 16 patches mismatch.

Usage:
    python scripts/check_checkpoint_architecture.py <checkpoint_path>
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings('ignore')

import torch
import argparse


def analyze_checkpoint(checkpoint_path: str):
    """Analyze checkpoint architecture."""

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return False

    print("="*80)
    print("CHECKPOINT ARCHITECTURE ANALYSIS")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print("-" * 80)

    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print("✓ Checkpoint loaded successfully")

        # Check structure
        if 'encoder_state_dict' in checkpoint:
            state_dict = checkpoint['encoder_state_dict']
            print("✓ Found encoder_state_dict")
        elif 'model_state_dict' in checkpoint:
            # Extract encoder
            full_dict = checkpoint['model_state_dict']
            state_dict = {
                k.replace('encoder.', ''): v
                for k, v in full_dict.items()
                if k.startswith('encoder.')
            }
            print("✓ Extracted encoder from model_state_dict")
        else:
            state_dict = checkpoint
            print("⚠️  Using checkpoint as state_dict directly")

        print(f"  Total encoder parameters: {len(state_dict)}")

        # Analyze architecture from weight shapes
        print("\n" + "="*80)
        print("ARCHITECTURE DETECTION")
        print("="*80)

        # 1. Detect from patcher
        patcher_keys = [k for k in state_dict.keys() if 'patcher' in k and 'weight' in k]
        if patcher_keys:
            patcher_key = patcher_keys[0]
            patcher_weight = state_dict[patcher_key]
            d_model = patcher_weight.shape[0]
            patch_dim = patcher_weight.shape[1]

            # Assume 2 input channels
            input_channels = 2
            patch_size = patch_dim // input_channels

            print(f"From patcher ({patcher_key}):")
            print(f"  Shape: {patcher_weight.shape}")
            print(f"  d_model: {d_model}")
            print(f"  patch_dim: {patch_dim}")
            print(f"  patch_size: {patch_size} (assuming {input_channels} channels)")

        # 2. Detect from encoder patch mixer
        encoder_mixer_keys = [k for k in state_dict.keys()
                              if 'encoder' in k and 'patch_mixer.mlp.fc1.weight' in k and 'backbone' in k]

        if encoder_mixer_keys:
            first_mixer = sorted(encoder_mixer_keys)[0]
            mixer_weight = state_dict[first_mixer]
            hidden_dim = mixer_weight.shape[0]
            encoder_num_patches = mixer_weight.shape[1]

            print(f"\nFrom encoder patch_mixer ({first_mixer}):")
            print(f"  Shape: {mixer_weight.shape}")
            print(f"  hidden_dim: {hidden_dim}")
            print(f"  num_patches: {encoder_num_patches}")

        # 3. Detect from decoder patch mixer
        decoder_mixer_keys = [k for k in state_dict.keys()
                              if 'decoder' in k and 'patch_mixer.mlp.fc1.weight' in k]

        decoder_num_patches = None
        if decoder_mixer_keys:
            first_decoder_mixer = sorted(decoder_mixer_keys)[0]
            decoder_mixer_weight = state_dict[first_decoder_mixer]
            decoder_hidden_dim = decoder_mixer_weight.shape[0]
            decoder_num_patches = decoder_mixer_weight.shape[1]

            print(f"\nFrom decoder patch_mixer ({first_decoder_mixer}):")
            print(f"  Shape: {decoder_mixer_weight.shape}")
            print(f"  hidden_dim: {decoder_hidden_dim}")
            print(f"  num_patches: {decoder_num_patches}")

        # Check metadata
        print("\n" + "="*80)
        print("CHECKPOINT METADATA")
        print("="*80)

        if 'architecture' in checkpoint:
            arch = checkpoint['architecture']
            print("✓ Architecture metadata found:")
            for key, value in arch.items():
                print(f"  {key}: {value}")

            meta_num_patches = arch.get('num_patches')
        elif 'config' in checkpoint:
            config = checkpoint['config']
            print("⚠️  Old format: config found (no architecture metadata):")
            for key, value in config.items():
                print(f"  {key}: {value}")

            context_length = config.get('context_length', 1024)
            patch_length = config.get('patch_length', 64)
            meta_num_patches = context_length // patch_length
            print(f"\n  Calculated num_patches: {meta_num_patches} ({context_length} ÷ {patch_length})")
        else:
            print("⚠️  No metadata found")
            meta_num_patches = None

        # VALIDATION
        print("\n" + "="*80)
        print("VALIDATION")
        print("="*80)

        # Expected for TTM-Enhanced with context=1024, patch=64
        expected_num_patches = 16

        issues = []

        if encoder_num_patches != expected_num_patches:
            issues.append(f"Encoder has {encoder_num_patches} patches, expected {expected_num_patches}")

        if decoder_num_patches and decoder_num_patches != expected_num_patches:
            issues.append(f"Decoder has {decoder_num_patches} patches, expected {expected_num_patches}")

        if encoder_num_patches and decoder_num_patches and encoder_num_patches != decoder_num_patches:
            issues.append(f"Encoder/decoder mismatch: {encoder_num_patches} vs {decoder_num_patches} patches")

        if meta_num_patches and meta_num_patches != expected_num_patches:
            issues.append(f"Metadata says {meta_num_patches} patches, expected {expected_num_patches}")

        if issues:
            print("❌ ISSUES DETECTED:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")

            print("\n" + "="*80)
            print("RECOMMENDATION")
            print("="*80)
            print("This checkpoint has architecture issues and should NOT be used.")
            print("\nOptions:")
            print("  1. Delete/rename this checkpoint")
            print("  2. Re-train SSL from scratch with correct configuration")
            print("\nExpected configuration for TTM-Enhanced:")
            print("  context_length: 1024")
            print("  patch_size: 64")
            print("  num_patches: 16")
            print("  d_model: 192")
            print("="*80)

            return False

        else:
            print("✅ Architecture looks correct!")
            print(f"  Encoder patches: {encoder_num_patches}")
            if decoder_num_patches:
                print(f"  Decoder patches: {decoder_num_patches}")
            if meta_num_patches:
                print(f"  Metadata patches: {meta_num_patches}")
            print("\n  All match expected: 16 patches for TTM-Enhanced")

            print("\n" + "="*80)
            print("✅ CHECKPOINT IS VALID")
            print("="*80)

            return True

    except Exception as e:
        print(f"\n✗ Error analyzing checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Analyze SSL checkpoint architecture"
    )
    parser.add_argument('checkpoint', type=str,
                       help='Path to checkpoint file')

    args = parser.parse_args()

    success = analyze_checkpoint(args.checkpoint)

    if success:
        sys.exit(0)
    else:
        sys.exit(1)
