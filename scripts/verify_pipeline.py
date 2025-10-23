#!/usr/bin/env python3
"""Verify SSL Pipeline Architecture.

This script verifies that:
1. Data is in correct format [2, 1024]
2. SSL checkpoint has correct architecture metadata
3. Model can be loaded without errors
4. Forward pass works correctly
5. Fine-tuning can load the checkpoint

Usage:
    python scripts/verify_pipeline.py \\
        --checkpoint artifacts/butppg_ssl/best_model.pt \\
        --data-dir data/processed/butppg/windows_with_labels

Example output:
    ✅ ALL CHECKS PASSED - Pipeline is ready for training!
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.butppg_dataset import BUTPPGDataset
from src.utils.checkpoint_utils import load_ssl_checkpoint_safe


def check_data_format(data_dir: str) -> bool:
    """Verify data is in correct format."""
    print("\n" + "="*80)
    print("CHECK 1: DATA FORMAT")
    print("="*80)

    data_dir = Path(data_dir)

    # Check train data exists
    train_dir = data_dir / 'train'
    if not train_dir.exists():
        print(f"❌ Train directory not found: {train_dir}")
        return False

    # Load a sample file
    npz_files = list(train_dir.glob('*.npz'))
    if not npz_files:
        print(f"❌ No .npz files found in {train_dir}")
        return False

    print(f"✓ Found {len(npz_files)} training files")

    # Check first file
    sample_file = npz_files[0]
    data = np.load(sample_file)

    print(f"\nChecking sample file: {sample_file.name}")

    # Check signal shape
    if 'signal' not in data:
        print(f"❌ No 'signal' key in data file")
        return False

    signal = data['signal']
    print(f"  Signal shape: {signal.shape}")
    print(f"  Signal dtype: {signal.dtype}")

    # Verify shape is [2, 1024]
    if signal.shape != (2, 1024):
        print(f"❌ Incorrect signal shape! Expected [2, 1024], got {signal.shape}")
        print(f"   Data needs to be resampled to 1024 samples")
        return False

    print(f"  ✅ Signal shape is correct: [2, 1024]")

    # Check labels exist
    label_keys = ['quality', 'hr', 'motion', 'bp_systolic', 'bp_diastolic', 'spo2', 'glycaemia']
    found_labels = [k for k in label_keys if k in data]
    print(f"  ✓ Found {len(found_labels)}/{len(label_keys)} task labels")

    print(f"\n✅ CHECK 1 PASSED: Data format is correct")
    return True


def check_checkpoint_architecture(checkpoint_path: str) -> bool:
    """Verify checkpoint has proper architecture metadata."""
    print("\n" + "="*80)
    print("CHECK 2: CHECKPOINT ARCHITECTURE")
    print("="*80)

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False

    print(f"✓ Checkpoint exists: {checkpoint_path}")

    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"✓ Checkpoint loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return False

    # Check for architecture metadata
    if 'architecture' in checkpoint:
        print(f"✅ Checkpoint has NEW architecture metadata format")
        arch = checkpoint['architecture']
        print(f"\n  Architecture:")
        for key, value in arch.items():
            print(f"    {key}: {value}")

        # Verify critical fields
        required_fields = ['context_length', 'patch_size', 'd_model', 'input_channels']
        missing = [f for f in required_fields if f not in arch]
        if missing:
            print(f"❌ Missing required fields: {missing}")
            return False

        print(f"\n✅ All required architecture fields present")
    else:
        print(f"⚠️  Checkpoint uses OLD format (no architecture metadata)")
        print(f"   Will attempt fallback detection from weight shapes")

        # Check if we can detect from weights
        if 'encoder_state_dict' in checkpoint:
            encoder_state = checkpoint['encoder_state_dict']
            patcher_keys = [k for k in encoder_state.keys() if 'patcher.weight' in k]
            if patcher_keys:
                print(f"   ✓ Can detect architecture from patcher weights")
            else:
                print(f"   ❌ Cannot detect architecture - no patcher weights found")
                return False

    print(f"\n✅ CHECK 2 PASSED: Checkpoint architecture is valid")
    return True


def check_model_loading(checkpoint_path: str) -> bool:
    """Verify model can be loaded without errors."""
    print("\n" + "="*80)
    print("CHECK 3: MODEL LOADING")
    print("="*80)

    try:
        encoder, architecture_config, metrics = load_ssl_checkpoint_safe(
            checkpoint_path=checkpoint_path,
            device='cpu',
            verbose=True
        )

        print(f"\n✅ CHECK 3 PASSED: Model loaded successfully")
        return True

    except Exception as e:
        print(f"\n❌ CHECK 3 FAILED: Model loading error")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_forward_pass(checkpoint_path: str) -> bool:
    """Verify forward pass works correctly."""
    print("\n" + "="*80)
    print("CHECK 4: FORWARD PASS")
    print("="*80)

    try:
        # Load model
        encoder, architecture_config, metrics = load_ssl_checkpoint_safe(
            checkpoint_path=checkpoint_path,
            device='cpu',
            verbose=False
        )

        # Create test input
        batch_size = 4
        context_length = architecture_config['context_length']
        input_channels = architecture_config['input_channels']

        test_input = torch.randn(batch_size, input_channels, context_length)
        print(f"  Test input shape: {test_input.shape}")

        # Forward pass
        with torch.no_grad():
            features = encoder.get_encoder_output(test_input)

        print(f"  Output features shape: {features.shape}")

        # Verify output shape
        expected_patches = context_length // architecture_config['patch_size']
        expected_shape = (batch_size, expected_patches, architecture_config['d_model'])

        if features.shape != expected_shape:
            print(f"❌ Output shape mismatch!")
            print(f"   Expected: {expected_shape}")
            print(f"   Got: {features.shape}")
            return False

        print(f"  ✅ Output shape is correct")

        # Check for NaNs or Infs
        if torch.isnan(features).any():
            print(f"❌ Output contains NaN values!")
            return False

        if torch.isinf(features).any():
            print(f"❌ Output contains Inf values!")
            return False

        print(f"  ✅ No NaN or Inf values")

        print(f"\n✅ CHECK 4 PASSED: Forward pass works correctly")
        return True

    except Exception as e:
        print(f"\n❌ CHECK 4 FAILED: Forward pass error")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_finetuning_compatibility(checkpoint_path: str, data_dir: str) -> bool:
    """Verify checkpoint can be used for fine-tuning."""
    print("\n" + "="*80)
    print("CHECK 5: FINE-TUNING COMPATIBILITY")
    print("="*80)

    try:
        # Load model
        encoder, architecture_config, metrics = load_ssl_checkpoint_safe(
            checkpoint_path=checkpoint_path,
            device='cpu',
            verbose=False
        )

        print(f"  ✓ Model loaded")

        # Create dataset
        dataset = BUTPPGDataset(
            data_dir=str(Path(data_dir) / 'train'),
            modality=['ppg', 'ecg'],
            split='train',
            mode='preprocessed',
            task='quality',
            return_labels=True
        )

        print(f"  ✓ Dataset created ({len(dataset)} samples)")

        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        batch = next(iter(dataloader))

        if isinstance(batch, tuple):
            signals, labels = batch
        else:
            signals = batch

        print(f"  ✓ Loaded batch: {signals.shape}")

        # Check shape compatibility
        if signals.shape[1:] != (architecture_config['input_channels'], architecture_config['context_length']):
            print(f"❌ Data shape mismatch!")
            print(f"   Data shape: {signals.shape[1:]}")
            print(f"   Expected: ({architecture_config['input_channels']}, {architecture_config['context_length']})")
            return False

        print(f"  ✅ Data shape matches model input")

        # Test forward pass with real data
        with torch.no_grad():
            features = encoder.get_encoder_output(signals)

        print(f"  ✓ Forward pass with real data: {features.shape}")

        # Test freezing/unfreezing
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False

        frozen_params = sum(p.numel() for p in encoder.parameters() if not p.requires_grad)
        total_params = sum(p.numel() for p in encoder.parameters())

        print(f"  ✓ Encoder frozen: {frozen_params}/{total_params} params")

        print(f"\n✅ CHECK 5 PASSED: Ready for fine-tuning")
        return True

    except Exception as e:
        print(f"\n❌ CHECK 5 FAILED: Fine-tuning compatibility error")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify SSL pipeline")
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to SSL checkpoint')
    parser.add_argument('--data-dir', type=str, required=True,
                      help='Path to data directory')
    args = parser.parse_args()

    print("\n" + "="*80)
    print("SSL PIPELINE VERIFICATION")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data directory: {args.data_dir}")

    # Run all checks
    checks = [
        ("Data Format", lambda: check_data_format(args.data_dir)),
        ("Checkpoint Architecture", lambda: check_checkpoint_architecture(args.checkpoint)),
        ("Model Loading", lambda: check_model_loading(args.checkpoint)),
        ("Forward Pass", lambda: check_forward_pass(args.checkpoint)),
        ("Fine-tuning Compatibility", lambda: check_finetuning_compatibility(args.checkpoint, args.data_dir)),
    ]

    results = []
    for name, check_func in checks:
        try:
            passed = check_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ CHECK FAILED WITH EXCEPTION: {name}")
            print(f"   Error: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    all_passed = all(passed for _, passed in results)

    print("\n" + "="*80)
    if all_passed:
        print("✅✅✅ ALL CHECKS PASSED - PIPELINE IS READY! ✅✅✅")
        print("="*80)
        print("\nYou can now:")
        print("  1. Run SSL training: python scripts/continue_ssl_butppg_quality.py")
        print("  2. Run fine-tuning: python scripts/finetune_enhanced.py")
        print("  3. Run all tasks: python scripts/run_all_tasks.py")
        return 0
    else:
        print("❌❌❌ SOME CHECKS FAILED - PLEASE FIX ISSUES ❌❌❌")
        print("="*80)
        return 1


if __name__ == '__main__':
    sys.exit(main())
