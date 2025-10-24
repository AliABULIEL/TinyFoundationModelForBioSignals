#!/usr/bin/env python3
"""Test script to verify channel inflation mapping is correct.

Validates that the channel inflation logic correctly maps:
- Pretrained PPG (ch 0) → New ch 3
- Pretrained ECG (ch 1) → New ch 4
- New ACC channels (0, 1, 2) are initialized from mean

Usage:
    python test_channel_inflation.py
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.channel_utils import _inflate_channel_weights


def test_channel_inflation_dim0():
    """Test channel inflation for dimension 0 (e.g., out_channels in Conv)."""
    print("=" * 70)
    print("TEST 1: Channel dimension 0 (out_channels)")
    print("=" * 70)

    # Create fake pretrained weights: [2, 128] (2 channels, 128 features)
    pretrained = torch.randn(2, 128)
    pretrained[0] = 1.0  # PPG channel - all 1s for easy verification
    pretrained[1] = 2.0  # ECG channel - all 2s for easy verification

    # Create target shape: [5, 128] (5 channels, 128 features)
    new_param = torch.zeros(5, 128)

    # Inflate
    inflated = _inflate_channel_weights(
        pretrained_param=pretrained,
        new_param=new_param,
        pretrain_channels=2,
        finetune_channels=5,
        param_name="test.weight"
    )

    if inflated is None:
        print("❌ FAILED: Inflation returned None")
        return False

    # Verify mapping
    print("\nVerifying channel mapping...")

    # Check PPG (pretrained ch 0 → new ch 3)
    ppg_correct = torch.allclose(inflated[3], pretrained[0])
    print(f"  PPG (ch 0 → ch 3): {'✓' if ppg_correct else '✗'}")
    if not ppg_correct:
        print(f"    Expected: all 1.0, Got: {inflated[3][:5].tolist()}...")

    # Check ECG (pretrained ch 1 → new ch 4)
    ecg_correct = torch.allclose(inflated[4], pretrained[1])
    print(f"  ECG (ch 1 → ch 4): {'✓' if ecg_correct else '✗'}")
    if not ecg_correct:
        print(f"    Expected: all 2.0, Got: {inflated[4][:5].tolist()}...")

    # Check ACC channels are initialized (not zero, not exactly matching pretrained)
    acc_initialized = (
        not torch.allclose(inflated[0], torch.zeros_like(inflated[0])) and
        not torch.allclose(inflated[1], torch.zeros_like(inflated[1])) and
        not torch.allclose(inflated[2], torch.zeros_like(inflated[2]))
    )
    print(f"  ACC (ch 0,1,2) initialized: {'✓' if acc_initialized else '✗'}")
    if acc_initialized:
        print(f"    ACC_X mean: {inflated[0].mean().item():.4f}")
        print(f"    ACC_Y mean: {inflated[1].mean().item():.4f}")
        print(f"    ACC_Z mean: {inflated[2].mean().item():.4f}")
        print(f"    (Expected: close to {pretrained.mean().item():.4f})")

    # Check ACC channels are different from each other (due to noise)
    acc_unique = not (
        torch.allclose(inflated[0], inflated[1]) or
        torch.allclose(inflated[1], inflated[2]) or
        torch.allclose(inflated[0], inflated[2])
    )
    print(f"  ACC channels unique: {'✓' if acc_unique else '✗'}")

    success = ppg_correct and ecg_correct and acc_initialized and acc_unique
    print(f"\n{'✓ PASSED' if success else '✗ FAILED'}")
    print("=" * 70)

    return success


def test_channel_inflation_dim1():
    """Test channel inflation for dimension 1 (e.g., in_channels in Linear)."""
    print("\nTEST 2: Channel dimension 1 (in_channels)")
    print("=" * 70)

    # Create fake pretrained weights: [256, 2] (256 features, 2 channels)
    pretrained = torch.randn(256, 2)
    pretrained[:, 0] = 1.0  # PPG channel - all 1s
    pretrained[:, 1] = 2.0  # ECG channel - all 2s

    # Create target shape: [256, 5] (256 features, 5 channels)
    new_param = torch.zeros(256, 5)

    # Inflate
    inflated = _inflate_channel_weights(
        pretrained_param=pretrained,
        new_param=new_param,
        pretrain_channels=2,
        finetune_channels=5,
        param_name="test.weight"
    )

    if inflated is None:
        print("❌ FAILED: Inflation returned None")
        return False

    # Verify mapping
    print("\nVerifying channel mapping...")

    # Check PPG (pretrained ch 0 → new ch 3)
    ppg_correct = torch.allclose(inflated[:, 3], pretrained[:, 0])
    print(f"  PPG (ch 0 → ch 3): {'✓' if ppg_correct else '✗'}")
    if not ppg_correct:
        print(f"    Expected: all 1.0, Got: {inflated[:5, 3].tolist()}...")

    # Check ECG (pretrained ch 1 → new ch 4)
    ecg_correct = torch.allclose(inflated[:, 4], pretrained[:, 1])
    print(f"  ECG (ch 1 → ch 4): {'✓' if ecg_correct else '✗'}")
    if not ecg_correct:
        print(f"    Expected: all 2.0, Got: {inflated[:5, 4].tolist()}...")

    # Check ACC channels are initialized
    acc_initialized = (
        not torch.allclose(inflated[:, 0], torch.zeros_like(inflated[:, 0])) and
        not torch.allclose(inflated[:, 1], torch.zeros_like(inflated[:, 1])) and
        not torch.allclose(inflated[:, 2], torch.zeros_like(inflated[:, 2]))
    )
    print(f"  ACC (ch 0,1,2) initialized: {'✓' if acc_initialized else '✗'}")
    if acc_initialized:
        print(f"    ACC_X mean: {inflated[:, 0].mean().item():.4f}")
        print(f"    ACC_Y mean: {inflated[:, 1].mean().item():.4f}")
        print(f"    ACC_Z mean: {inflated[:, 2].mean().item():.4f}")
        print(f"    (Expected: close to {pretrained.mean().item():.4f})")

    # Check ACC channels are different from each other
    acc_unique = not (
        torch.allclose(inflated[:, 0], inflated[:, 1]) or
        torch.allclose(inflated[:, 1], inflated[:, 2]) or
        torch.allclose(inflated[:, 0], inflated[:, 2])
    )
    print(f"  ACC channels unique: {'✓' if acc_unique else '✗'}")

    success = ppg_correct and ecg_correct and acc_initialized and acc_unique
    print(f"\n{'✓ PASSED' if success else '✗ FAILED'}")
    print("=" * 70)

    return success


def main():
    print("\n" + "=" * 70)
    print("CHANNEL INFLATION VERIFICATION")
    print("=" * 70)
    print("\nTesting channel mapping: PPG/ECG (2-ch) → ACC_X/Y/Z + PPG + ECG (5-ch)")
    print("Expected mapping:")
    print("  - Pretrained ch 0 (PPG) → New ch 3 (PPG)")
    print("  - Pretrained ch 1 (ECG) → New ch 4 (ECG)")
    print("  - New ch 0,1,2 (ACC_X/Y/Z) → Initialize from mean + noise")
    print()

    # Run tests
    test1_pass = test_channel_inflation_dim0()
    test2_pass = test_channel_inflation_dim1()

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Test 1 (channel_dim=0): {'✓ PASSED' if test1_pass else '✗ FAILED'}")
    print(f"Test 2 (channel_dim=1): {'✓ PASSED' if test2_pass else '✗ FAILED'}")
    print()

    if test1_pass and test2_pass:
        print("✓ ALL TESTS PASSED")
        print("\nChannel inflation is working correctly!")
        print("You can now proceed with fine-tuning:")
        print("  python scripts/finetune_butppg.py --pretrained <path> --epochs 1")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease review the channel inflation logic in:")
        print("  src/models/channel_utils.py:_inflate_channel_weights()")
        return 1


if __name__ == "__main__":
    sys.exit(main())
