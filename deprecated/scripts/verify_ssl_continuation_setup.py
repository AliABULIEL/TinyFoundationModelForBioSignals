#!/usr/bin/env python3
"""
Quick verification script to check SSL continuation setup.

This script verifies:
1. VitalDB checkpoint exists and is loadable
2. BUT-PPG data is accessible
3. Encoder/decoder can be created
4. Forward pass works correctly
5. Data dimensions match

Run this before starting full SSL continuation training.
"""

import sys
from pathlib import Path

import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def check_checkpoint(checkpoint_path: Path):
    """Verify VitalDB checkpoint exists and is loadable."""
    print("\n" + "="*80)
    print("1. CHECKING VITALDB CHECKPOINT")
    print("="*80)

    if not checkpoint_path.exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return False

    print(f"✓ Checkpoint exists: {checkpoint_path}")
    print(f"  Size: {checkpoint_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✓ Checkpoint loaded successfully")

        # Check keys
        print(f"\nCheckpoint keys:")
        for key in checkpoint.keys():
            if isinstance(checkpoint[key], dict):
                print(f"  - {key}: {len(checkpoint[key])} items")
            else:
                print(f"  - {key}: {type(checkpoint[key]).__name__}")

        return True

    except Exception as e:
        print(f"✗ Failed to load checkpoint: {e}")
        return False


def check_butppg_data(data_dir: Path):
    """Verify BUT-PPG data exists and is readable."""
    print("\n" + "="*80)
    print("2. CHECKING BUT-PPG DATA")
    print("="*80)

    if not data_dir.exists():
        print(f"✗ Data directory not found: {data_dir}")
        return False

    print(f"✓ Data directory exists: {data_dir}")

    # Check for splits
    splits = ['train', 'val', 'test']
    all_good = True

    for split in splits:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"  ✗ {split} directory not found")
            all_good = False
            continue

        window_files = sorted(split_dir.glob('window_*.npz'))
        print(f"  ✓ {split}: {len(window_files)} window files")

        if len(window_files) == 0:
            print(f"    ✗ No window files found!")
            all_good = False
            continue

        # Check first file
        try:
            data = np.load(window_files[0])
            signal = data['signal']
            print(f"    Signal shape: {signal.shape}")

            if 'quality' in data:
                print(f"    Quality label: {data['quality']} (ignored for SSL)")
            if 'ppg_quality' in data:
                print(f"    PPG SQI: {data['ppg_quality']:.3f}")
            if 'ecg_quality' in data:
                print(f"    ECG SQI: {data['ecg_quality']:.3f}")

        except Exception as e:
            print(f"    ✗ Failed to load sample file: {e}")
            all_good = False

    return all_good


def check_model_creation(checkpoint_path: Path):
    """Verify encoder/decoder can be created from checkpoint."""
    print("\n" + "="*80)
    print("3. CHECKING MODEL CREATION")
    print("="*80)

    try:
        from src.models.ttm_adapter import TTMAdapter
        from src.models.decoders import ReconstructionHead1D

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Get state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'encoder_state_dict' in checkpoint:
            state_dict = checkpoint['encoder_state_dict']
        else:
            print(f"✗ Cannot find model weights. Keys: {checkpoint.keys()}")
            return False

        # Detect architecture
        patcher_weight = None
        for key in state_dict.keys():
            if 'patcher' in key and 'weight' in key:
                patcher_weight = state_dict[key]
                break

        if patcher_weight is None:
            print("✗ Cannot find patcher weights")
            return False

        d_model = patcher_weight.shape[0]
        patch_size = patcher_weight.shape[-1]

        print(f"✓ Detected architecture:")
        print(f"  - d_model: {d_model}")
        print(f"  - patch_size: {patch_size}")

        # Determine if IBM pretrained
        context_length = 1024  # Default
        use_ibm_pretrained = (d_model == 192 and context_length == 1024)

        print(f"  - IBM pretrained: {use_ibm_pretrained}")

        if use_ibm_pretrained:
            model_patch_size = 128
        else:
            model_patch_size = patch_size

        # Create encoder
        print(f"\nCreating encoder...")
        encoder = TTMAdapter(
            context_length=context_length,
            patch_length=model_patch_size,
            num_input_channels=2,
            num_output_channels=2,
            d_model=d_model,
            use_ibm_pretrained=use_ibm_pretrained,
            for_pretraining=True
        )

        print(f"✓ Encoder created: {sum(p.numel() for p in encoder.parameters())} parameters")

        # Test forward pass to get actual dimensions
        dummy_input = torch.randn(1, 2, context_length)
        with torch.no_grad():
            encoder_output = encoder.get_encoder_output(dummy_input)
            actual_d_model = encoder_output.shape[-1]
            num_patches = encoder_output.shape[1]
            actual_patch_size = context_length // num_patches

        print(f"✓ Forward pass successful:")
        print(f"  - Input: [1, 2, {context_length}]")
        print(f"  - Output: {list(encoder_output.shape)}")
        print(f"  - Actual d_model: {actual_d_model}")
        print(f"  - Actual patch_size: {actual_patch_size}")

        # Create decoder
        print(f"\nCreating decoder...")
        decoder = ReconstructionHead1D(
            d_model=actual_d_model,
            patch_size=actual_patch_size,
            n_channels=2
        )

        print(f"✓ Decoder created: {sum(p.numel() for p in decoder.parameters())} parameters")

        # Test reconstruction
        with torch.no_grad():
            reconstructed = decoder(encoder_output)

        print(f"✓ Reconstruction successful:")
        print(f"  - Encoder output: {list(encoder_output.shape)}")
        print(f"  - Reconstructed: {list(reconstructed.shape)}")

        if reconstructed.shape[-1] != context_length:
            print(f"  ⚠️  Warning: Reconstructed length ({reconstructed.shape[-1]}) "
                  f"!= context_length ({context_length})")

        return True

    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_ssl_dataloader(data_dir: Path):
    """Verify SSL dataloader can be created."""
    print("\n" + "="*80)
    print("4. CHECKING SSL DATALOADER")
    print("="*80)

    try:
        from torch.utils.data import Dataset, DataLoader

        class BUTPPGSSLDataset(Dataset):
            def __init__(self, data_dir: Path, target_length: int = 1024):
                window_files = sorted(data_dir.glob('window_*.npz'))
                signals_list = []

                for window_file in window_files:
                    data = np.load(window_file)
                    signal = data['signal']
                    signals_list.append(signal)

                self.signals = torch.from_numpy(np.stack(signals_list, axis=0)).float()

                # Resize if needed
                N, C, T = self.signals.shape
                if T != target_length:
                    self.signals = torch.nn.functional.interpolate(
                        self.signals,
                        size=target_length,
                        mode='linear',
                        align_corners=False
                    )

                # Normalize
                for c in range(C):
                    channel_data = self.signals[:, c, :]
                    mean = channel_data.mean()
                    std = channel_data.std()
                    if std > 0:
                        self.signals[:, c, :] = (channel_data - mean) / std

            def __len__(self):
                return len(self.signals)

            def __getitem__(self, idx):
                return self.signals[idx]

        # Create dataset
        train_dir = data_dir / 'train'
        dataset = BUTPPGSSLDataset(train_dir, target_length=1024)

        print(f"✓ Dataset created:")
        print(f"  - Size: {len(dataset)} samples")
        print(f"  - Signal shape: {dataset.signals.shape}")

        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Test batch
        batch = next(iter(dataloader))
        print(f"✓ Dataloader created:")
        print(f"  - Batch shape: {batch.shape}")
        print(f"  - Expected: [32, 2, 1024]")

        if batch.shape[1] != 2 or batch.shape[2] != 1024:
            print(f"  ✗ Shape mismatch!")
            return False

        return True

    except Exception as e:
        print(f"✗ Dataloader creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*80)
    print("SSL CONTINUATION SETUP VERIFICATION")
    print("="*80)
    print("\nThis script checks if everything is ready for SSL continuation on BUT-PPG")

    # Default paths
    checkpoint_path = Path('artifacts/foundation_model/best_model.pt')
    data_dir = Path('data/processed/butppg/windows_with_labels')

    # Allow command line override
    if len(sys.argv) > 1:
        checkpoint_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        data_dir = Path(sys.argv[2])

    print(f"\nPaths:")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Data dir:   {data_dir}")

    # Run checks
    results = {
        'checkpoint': check_checkpoint(checkpoint_path),
        'data': check_butppg_data(data_dir),
        'model': check_model_creation(checkpoint_path),
        'dataloader': check_ssl_dataloader(data_dir)
    }

    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)

    all_passed = all(results.values())

    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {check.capitalize():15s}: {status}")

    print()

    if all_passed:
        print("✅ All checks passed! You're ready to run SSL continuation.")
        print("\nRun the following command to start training:")
        print(f"\n  python scripts/continue_ssl_butppg.py \\")
        print(f"      --checkpoint {checkpoint_path} \\")
        print(f"      --data-dir {data_dir} \\")
        print(f"      --epochs 20 \\")
        print(f"      --batch-size 64 \\")
        print(f"      --lr 1e-5")
        return 0
    else:
        print("❌ Some checks failed. Please fix the issues above before running SSL continuation.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
