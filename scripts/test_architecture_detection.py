#!/usr/bin/env python3
"""
Quick test to verify architecture detection from SSL checkpoint.

This test loads the SSL checkpoint and detects:
- num_patches (should be 16)
- d_model (should be 192)
- patch_size (should be 64)
- context_length (should be 1024)

Usage:
    python scripts/test_architecture_detection.py \
        --checkpoint artifacts/foundation_model/best_model.pt
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch


def test_architecture_detection(checkpoint_path):
    """Test architecture detection from SSL checkpoint."""

    print("=" * 80)
    print("TESTING ARCHITECTURE DETECTION")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print()

    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Get state dict
    if 'encoder_state_dict' in checkpoint:
        state_dict = checkpoint['encoder_state_dict']
        print("✓ Found 'encoder_state_dict' (SSL checkpoint)")
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("✓ Found 'model_state_dict'")
    else:
        state_dict = checkpoint
        print("⚠ Using raw checkpoint")

    print()

    # =========================================================================
    # STEP 1: Detect num_patches from decoder patch_mixer
    # =========================================================================
    print("STEP 1: Detecting num_patches from decoder patch_mixer...")
    print("-" * 80)

    decoder_patch_mixer_keys = [k for k in state_dict.keys()
                                if 'encoder.decoder' in k and 'patch_mixer' in k and 'mlp.fc1.weight' in k]

    if not decoder_patch_mixer_keys:
        print("❌ ERROR: No decoder patch_mixer keys found!")
        print("\nSearching for any patch_mixer keys:")
        all_patch_mixer = [k for k in state_dict.keys() if 'patch_mixer' in k and 'weight' in k]
        for key in all_patch_mixer[:5]:
            print(f"  {key}")
        return False

    first_key = decoder_patch_mixer_keys[0]
    weight = state_dict[first_key]
    num_patches = weight.shape[1]

    print(f"✓ Found decoder patch_mixer:")
    print(f"  Key: {first_key}")
    print(f"  Shape: {weight.shape}")
    print(f"  num_patches = {num_patches}")
    print()

    # =========================================================================
    # STEP 2: Detect d_model and patch_size from patcher
    # =========================================================================
    print("STEP 2: Detecting d_model and patch_size from patcher...")
    print("-" * 80)

    patcher_keys = [k for k in state_dict.keys() if 'patcher.weight' in k and 'encoder' in k]

    if not patcher_keys:
        print("❌ ERROR: No patcher keys found!")
        return False

    patcher_weight = state_dict[patcher_keys[0]]
    d_model = patcher_weight.shape[0]
    patch_size = patcher_weight.shape[1]  # CRITICAL: Should NOT divide by 2!
    context_length = num_patches * patch_size

    print(f"✓ Found patcher:")
    print(f"  Key: {patcher_keys[0]}")
    print(f"  Shape: {patcher_weight.shape}")
    print(f"  d_model = {d_model}")
    print(f"  patch_size = {patch_size}")
    print(f"  context_length = num_patches × patch_size = {num_patches} × {patch_size} = {context_length}")
    print()

    # =========================================================================
    # STEP 3: Verify results
    # =========================================================================
    print("STEP 3: Verification")
    print("-" * 80)

    expected = {
        'num_patches': 16,
        'd_model': 192,
        'patch_size': 64,
        'context_length': 1024
    }

    actual = {
        'num_patches': num_patches,
        'd_model': d_model,
        'patch_size': patch_size,
        'context_length': context_length
    }

    all_correct = True
    for param, expected_value in expected.items():
        actual_value = actual[param]
        if actual_value == expected_value:
            print(f"✅ {param}: {actual_value} (correct)")
        else:
            print(f"❌ {param}: {actual_value} (expected {expected_value})")
            all_correct = False

    print()
    print("=" * 80)

    if all_correct:
        print("✅ ALL TESTS PASSED!")
        print()
        print("Architecture detection is working correctly.")
        print("You can now run fine-tuning with confidence.")
    else:
        print("❌ TESTS FAILED!")
        print()
        print("Architecture detection has errors.")
        print("Please check the fix was applied correctly.")
        print()
        print("Common issues:")
        print("  1. If patch_size=32 instead of 64:")
        print("     → The line still has 'patch_size = patch_input_dim // 2'")
        print("     → Should be 'patch_size = patcher_weight.shape[1]'")
        print("  2. If num_patches=64 instead of 16:")
        print("     → Using encoder patch_mixer instead of decoder patch_mixer")

    print("=" * 80)

    return all_correct


def main():
    parser = argparse.ArgumentParser(description='Test architecture detection from SSL checkpoint')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='artifacts/foundation_model/best_model.pt',
        help='Path to SSL checkpoint'
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print()
        print("Please provide a valid checkpoint path:")
        print("  python scripts/test_architecture_detection.py --checkpoint /path/to/checkpoint.pt")
        return 1

    success = test_architecture_detection(checkpoint_path)

    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
