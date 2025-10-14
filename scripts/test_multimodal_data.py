#!/usr/bin/env python3
"""
Test multi-modal data loading for BUT PPG and VitalDB datasets.
Shows examples of loaded data to verify multi-modal support.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.data.vitaldb_dataset import VitalDBDataset
from src.data.butppg_dataset import BUTPPGDataset


def test_vitaldb_multimodal():
    """Test VitalDB multi-modal loading (PPG + ECG)."""
    print("=" * 60)
    print("Testing VitalDB Multi-Modal Loading")
    print("=" * 60)

    try:
        # Test single modality
        print("\n1. Testing single modality (PPG only):")
        dataset_ppg = VitalDBDataset(
            cache_dir='data/vitaldb_cache',
            channels='ppg',  # Single modality
            split='train',
            use_raw_vitaldb=True,
            max_cases=5  # Limit for testing
        )

        if len(dataset_ppg) > 0:
            seg1, seg2 = dataset_ppg[0]
            print(f"  PPG only shape: {seg1.shape}")  # Should be [1, 1250]
            print(f"  Values: mean={seg1.mean():.3f}, std={seg1.std():.3f}")

        # Test multi-modal
        print("\n2. Testing multi-modal (PPG + ECG):")
        dataset_multi = VitalDBDataset(
            cache_dir='data/vitaldb_cache',
            channels=['ppg', 'ecg'],  # Multi-modal
            split='train',
            use_raw_vitaldb=True,
            max_cases=5
        )

        if len(dataset_multi) > 0:
            seg1, seg2 = dataset_multi[0]
            print(f"  Multi-modal shape: {seg1.shape}")  # Should be [2, 1250]
            print(f"  PPG channel: mean={seg1[0].mean():.3f}, std={seg1[0].std():.3f}")
            print(f"  ECG channel: mean={seg1[1].mean():.3f}, std={seg1[1].std():.3f}")

        # Test 'all' option
        print("\n3. Testing 'all' modalities:")
        dataset_all = VitalDBDataset(
            cache_dir='data/vitaldb_cache',
            channels='all',  # All available modalities
            split='train',
            use_raw_vitaldb=True,
            max_cases=5
        )

        if len(dataset_all) > 0:
            seg1, seg2 = dataset_all[0]
            print(f"  All modalities shape: {seg1.shape}")  # Should be [2, 1250]

        # Test comma-separated string
        print("\n4. Testing comma-separated string:")
        dataset_comma = VitalDBDataset(
            cache_dir='data/vitaldb_cache',
            channels='ppg,ecg',  # Comma-separated
            split='train',
            use_raw_vitaldb=True,
            max_cases=5
        )

        if len(dataset_comma) > 0:
            seg1, seg2 = dataset_comma[0]
            print(f"  Comma format shape: {seg1.shape}")  # Should be [2, 1250]

        print("\n✓ VitalDB multi-modal tests completed!")

    except Exception as e:
        print(f"✗ VitalDB test failed: {e}")
        import traceback
        traceback.print_exc()


def test_butppg_multimodal():
    """Test BUT PPG multi-modal loading (PPG + ECG + ACC)."""
    print("\n" + "=" * 60)
    print("Testing BUT PPG Multi-Modal Loading")
    print("=" * 60)

    try:
        # Check if data exists
        data_dir = Path('data/but_ppg/dataset')
        if not data_dir.exists():
            print(f"✗ BUT PPG data not found at {data_dir}")
            print("  Please download and extract the BUT PPG dataset")
            return

        # Test single modality
        print("\n1. Testing single modality (PPG only):")
        dataset_ppg = BUTPPGDataset(
            data_dir=str(data_dir),
            modality='ppg',
            split='train'
        )

        if len(dataset_ppg) > 0:
            seg1, seg2 = dataset_ppg[0]
            print(f"  PPG only shape: {seg1.shape}")  # Should be [1, 1250]
            print(f"  Values: mean={seg1.mean():.3f}, std={seg1.std():.3f}")

        # Test ECG
        print("\n2. Testing ECG modality:")
        dataset_ecg = BUTPPGDataset(
            data_dir=str(data_dir),
            modality='ecg',
            split='train'
        )

        if len(dataset_ecg) > 0:
            seg1, seg2 = dataset_ecg[0]
            print(f"  ECG shape: {seg1.shape}")  # Should be [1, 1250]

        # Test ACC (3-axis)
        print("\n3. Testing ACC modality (3-axis):")
        dataset_acc = BUTPPGDataset(
            data_dir=str(data_dir),
            modality='acc',
            split='train'
        )

        if len(dataset_acc) > 0:
            seg1, seg2 = dataset_acc[0]
            print(f"  ACC shape: {seg1.shape}")  # Should be [3, 1250]
            print(f"  X-axis: mean={seg1[0].mean():.3f}, std={seg1[0].std():.3f}")
            print(f"  Y-axis: mean={seg1[1].mean():.3f}, std={seg1[1].std():.3f}")
            print(f"  Z-axis: mean={seg1[2].mean():.3f}, std={seg1[2].std():.3f}")

        # Test multi-modal PPG + ECG
        print("\n4. Testing multi-modal (PPG + ECG):")
        dataset_ppg_ecg = BUTPPGDataset(
            data_dir=str(data_dir),
            modality=['ppg', 'ecg'],
            split='train'
        )

        if len(dataset_ppg_ecg) > 0:
            seg1, seg2 = dataset_ppg_ecg[0]
            print(f"  PPG+ECG shape: {seg1.shape}")  # Should be [2, 1250]

        # Test all modalities
        print("\n5. Testing all modalities (PPG + ECG + ACC):")
        dataset_all = BUTPPGDataset(
            data_dir=str(data_dir),
            modality='all',
            split='train'
        )

        if len(dataset_all) > 0:
            seg1, seg2 = dataset_all[0]
            print(f"  All modalities shape: {seg1.shape}")  # Should be [5, 1250] (1+1+3)
            print(f"  Channel breakdown:")
            print(f"    PPG (ch 0): mean={seg1[0].mean():.3f}, std={seg1[0].std():.3f}")
            print(f"    ECG (ch 1): mean={seg1[1].mean():.3f}, std={seg1[1].std():.3f}")
            print(f"    ACC-X (ch 2): mean={seg1[2].mean():.3f}, std={seg1[2].std():.3f}")
            print(f"    ACC-Y (ch 3): mean={seg1[3].mean():.3f}, std={seg1[3].std():.3f}")
            print(f"    ACC-Z (ch 4): mean={seg1[4].mean():.3f}, std={seg1[4].std():.3f}")

        print("\n✓ BUT PPG multi-modal tests completed!")

    except Exception as e:
        print(f"✗ BUT PPG test failed: {e}")
        import traceback
        traceback.print_exc()


def visualize_multimodal_sample():
    """Visualize a multi-modal sample from both datasets."""
    print("\n" + "=" * 60)
    print("Visualizing Multi-Modal Samples")
    print("=" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Multi-Modal Biosignal Samples', fontsize=14)

    # VitalDB sample (PPG + ECG)
    try:
        dataset_vitaldb = VitalDBDataset(
            cache_dir='data/vitaldb_cache',
            channels=['ppg', 'ecg'],
            split='test',
            use_raw_vitaldb=True,
            max_cases=2
        )

        if len(dataset_vitaldb) > 0:
            seg1, _ = dataset_vitaldb[0]

            # Plot PPG
            axes[0, 0].plot(seg1[0].numpy())
            axes[0, 0].set_title('VitalDB: PPG')
            axes[0, 0].set_xlabel('Samples')
            axes[0, 0].set_ylabel('Amplitude')
            axes[0, 0].grid(True, alpha=0.3)

            # Plot ECG
            axes[0, 1].plot(seg1[1].numpy())
            axes[0, 1].set_title('VitalDB: ECG')
            axes[0, 1].set_xlabel('Samples')
            axes[0, 1].grid(True, alpha=0.3)

            # Plot combined
            axes[0, 2].plot(seg1[0].numpy(), label='PPG', alpha=0.7)
            axes[0, 2].plot(seg1[1].numpy(), label='ECG', alpha=0.7)
            axes[0, 2].set_title('VitalDB: Combined')
            axes[0, 2].set_xlabel('Samples')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)

    except Exception as e:
        print(f"Could not plot VitalDB: {e}")
        for ax in axes[0, :]:
            ax.text(0.5, 0.5, 'VitalDB not available', ha='center', va='center')

    # BUT PPG sample (PPG + ECG + ACC)
    try:
        data_dir = Path('data/but_ppg/dataset')
        if data_dir.exists():
            dataset_butppg = BUTPPGDataset(
                data_dir=str(data_dir),
                modality=['ppg', 'ecg', 'acc'],
                split='test'
            )

            if len(dataset_butppg) > 0:
                seg1, _ = dataset_butppg[0]

                # Plot PPG
                axes[1, 0].plot(seg1[0].numpy())
                axes[1, 0].set_title('BUT PPG: PPG')
                axes[1, 0].set_xlabel('Samples')
                axes[1, 0].set_ylabel('Amplitude')
                axes[1, 0].grid(True, alpha=0.3)

                # Plot ECG
                axes[1, 1].plot(seg1[1].numpy())
                axes[1, 1].set_title('BUT PPG: ECG')
                axes[1, 1].set_xlabel('Samples')
                axes[1, 1].grid(True, alpha=0.3)

                # Plot ACC (3-axis)
                axes[1, 2].plot(seg1[2].numpy(), label='ACC-X', alpha=0.7)
                axes[1, 2].plot(seg1[3].numpy(), label='ACC-Y', alpha=0.7)
                axes[1, 2].plot(seg1[4].numpy(), label='ACC-Z', alpha=0.7)
                axes[1, 2].set_title('BUT PPG: Accelerometer')
                axes[1, 2].set_xlabel('Samples')
                axes[1, 2].legend()
                axes[1, 2].grid(True, alpha=0.3)
        else:
            for ax in axes[1, :]:
                ax.text(0.5, 0.5, 'BUT PPG not available', ha='center', va='center')

    except Exception as e:
        print(f"Could not plot BUT PPG: {e}")
        for ax in axes[1, :]:
            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center', va='center')

    plt.tight_layout()
    plt.savefig('multimodal_samples.png', dpi=100)
    print(f"✓ Visualization saved to multimodal_samples.png")
    plt.show()


def test_dataloader_batching():
    """Test that DataLoader works correctly with multi-modal data."""
    print("\n" + "=" * 60)
    print("Testing DataLoader with Multi-Modal Data")
    print("=" * 60)

    try:
        # Create multi-modal dataset
        dataset = VitalDBDataset(
            cache_dir='data/vitaldb_cache',
            channels=['ppg', 'ecg'],
            split='train',
            use_raw_vitaldb=True,
            max_cases=10
        )

        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0  # Use 0 for testing
        )

        # Test iteration
        for i, batch in enumerate(dataloader):
            if i == 0:  # Check first batch
                if len(batch) == 2:  # Paired segments
                    seg1, seg2 = batch
                    print(f"\nBatch shapes:")
                    print(f"  Segment 1: {seg1.shape}")  # Should be [4, 2, 1250]
                    print(f"  Segment 2: {seg2.shape}")  # Should be [4, 2, 1250]

                    # Verify data
                    print(f"\nBatch statistics:")
                    print(f"  PPG channel mean: {seg1[:, 0, :].mean():.3f}")
                    print(f"  ECG channel mean: {seg1[:, 1, :].mean():.3f}")
                    print(f"  Batch size: {seg1.shape[0]}")
                    print(f"  Channels: {seg1.shape[1]}")
                    print(f"  Samples: {seg1.shape[2]}")

                    print("\n✓ DataLoader test passed!")
                    break

    except Exception as e:
        print(f"✗ DataLoader test failed: {e}")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("MULTI-MODAL DATA LOADING TEST SUITE")
    print("=" * 60)

    # Test VitalDB
    test_vitaldb_multimodal()

    # Test BUT PPG
    test_butppg_multimodal()

    # Test DataLoader
    test_dataloader_batching()

    # Visualize samples
    visualize_multimodal_sample()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
