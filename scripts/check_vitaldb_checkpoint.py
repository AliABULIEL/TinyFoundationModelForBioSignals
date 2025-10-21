#!/usr/bin/env python3
"""
Diagnostic script to check VitalDB SSL checkpoint structure.

This script helps diagnose architecture mismatches between VitalDB and BUT-PPG.
"""

import torch
import sys
from pathlib import Path

def check_checkpoint(checkpoint_path: str):
    """
    Check VitalDB checkpoint structure and print diagnostics.

    Args:
        checkpoint_path: Path to VitalDB checkpoint (.pt file)
    """
    print(f"Checking checkpoint: {checkpoint_path}")
    print("=" * 80)

    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print("✓ Checkpoint loaded successfully\n")
    except FileNotFoundError:
        print(f"❌ ERROR: Checkpoint not found at {checkpoint_path}")
        print("\nExpected location:")
        print("  Colab: /content/drive/MyDrive/BioSignals/artifacts/foundation_model/best_model.pt")
        print("  Local: artifacts/foundation_model/best_model.pt")
        return
    except Exception as e:
        print(f"❌ ERROR loading checkpoint: {e}")
        return

    # Check checkpoint structure
    print("Checkpoint Keys:")
    for key in checkpoint.keys():
        print(f"  - {key}")
    print()

    # Extract encoder state dict
    if 'encoder_state_dict' in checkpoint:
        encoder_state = checkpoint['encoder_state_dict']
        print("✓ Found 'encoder_state_dict'")
    elif 'model_state_dict' in checkpoint:
        encoder_state = checkpoint['model_state_dict']
        print("✓ Found 'model_state_dict'")
    else:
        print("❌ No recognized state dict key found")
        return

    print(f"Total encoder parameters: {len(encoder_state)}")
    print()

    # Check for architecture patterns
    print("Architecture Analysis:")
    print("-" * 80)

    # Check for adaptive patching (mixer_layers)
    mixer_layers_keys = [k for k in encoder_state.keys() if 'mixer_layers' in k]
    if mixer_layers_keys:
        print(f"⚠️  ADAPTIVE PATCHING DETECTED ({len(mixer_layers_keys)} keys)")
        print("   This checkpoint uses multi-resolution adaptive patching")
        print("   Sample keys:")
        for key in mixer_layers_keys[:3]:
            print(f"     - {key}")
        print()

    # Check for flat mixers structure
    mixers_keys = [k for k in encoder_state.keys() if 'mixers.' in k and 'mixer_layers' not in k]
    if mixers_keys:
        print(f"✓ FLAT MIXER STRUCTURE ({len(mixers_keys)} keys)")
        print("   This checkpoint uses standard flat mixer structure")
        print("   Sample keys:")
        for key in mixers_keys[:3]:
            print(f"     - {key}")
        print()

    # Check patcher to detect patch_size and d_model
    patcher_keys = [k for k in encoder_state.keys() if 'patcher.weight' in k]
    if patcher_keys:
        patcher_weight = encoder_state[patcher_keys[0]]
        d_model = patcher_weight.shape[0]
        patch_size_per_channel = patcher_weight.shape[1]

        print(f"Patcher Configuration (from '{patcher_keys[0]}'):")
        print(f"  - Shape: {list(patcher_weight.shape)}")
        print(f"  - d_model: {d_model}")
        print(f"  - patch_size (per channel): {patch_size_per_channel}")
        print()

    # Check decoder
    if 'decoder_state_dict' in checkpoint:
        decoder_state = checkpoint['decoder_state_dict']
        print(f"✓ Found decoder with {len(decoder_state)} parameters")

        # Check adapter dimensions
        adapter_keys = [k for k in decoder_state.keys() if 'adapter' in k]
        if adapter_keys:
            for key in adapter_keys:
                print(f"  - {key}: {list(decoder_state[key].shape)}")
        print()

    # Check metadata
    if 'epoch' in checkpoint:
        print(f"Training Info:")
        print(f"  - Epoch: {checkpoint['epoch']}")
        if 'best_val_loss' in checkpoint:
            print(f"  - Best val loss: {checkpoint['best_val_loss']:.4f}")
        print()

    # Architecture compatibility check
    print("Compatibility Assessment:")
    print("-" * 80)

    if mixer_layers_keys and not mixers_keys:
        print("❌ INCOMPATIBLE: This checkpoint uses adaptive patching")
        print("   Current BUT-PPG script expects flat mixer structure")
        print()
        print("SOLUTION:")
        print("  1. Re-run VitalDB SSL training with current codebase:")
        print("     python3 scripts/pretrain_vitaldb_ssl.py \\")
        print("       --mode fasttrack \\")
        print("       --epochs 20 \\")
        print("       --data-dir /content/drive/MyDrive/BioSignals/data/processed/vitaldb/windows_with_labels/paired")
        print()
        print("  2. OR skip VitalDB and train BUT-PPG directly from IBM TTM:")
        print("     python3 scripts/pretrain_butppg_ssl.py \\")
        print("       --data-dir data/processed/butppg/windows_with_labels \\")
        print("       --output-dir artifacts/butppg_ssl \\")
        print("       --epochs 20 --batch-size 64 --lr 5e-5")

    elif mixers_keys and not mixer_layers_keys:
        print("✓ COMPATIBLE: This checkpoint uses flat mixer structure")
        print("   Should work with current BUT-PPG script")

    else:
        print("⚠️  UNKNOWN: Unable to determine architecture structure")
        print("   Please check the keys manually")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/check_vitaldb_checkpoint.py <checkpoint_path>")
        print("\nExamples:")
        print("  Colab: python scripts/check_vitaldb_checkpoint.py /content/drive/MyDrive/BioSignals/artifacts/foundation_model/best_model.pt")
        print("  Local: python scripts/check_vitaldb_checkpoint.py artifacts/foundation_model/best_model.pt")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    check_checkpoint(checkpoint_path)
