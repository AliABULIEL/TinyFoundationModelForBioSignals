#!/usr/bin/env python3
"""Test multi-modal data loading with mock data (no SSL required)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

# Set mock mode to avoid SSL issues
os.environ['VITALDB_MOCK'] = '1'


def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    try:
        from src.data.vitaldb_dataset import VitalDBDataset, create_vitaldb_dataloaders
        print("✓ VitalDB dataset imports OK")
    except ImportError as e:
        print(f"✗ VitalDB dataset import failed: {e}")
        return False
        
    try:
        from src.data.butppg_dataset import BUTPPGDataset, create_butppg_dataloaders
        print("✓ BUT PPG dataset imports OK")
    except ImportError as e:
        print(f"✗ BUT PPG dataset import failed: {e}")
        return False
        
    return True


def test_mock_vitaldb():
    """Test VitalDB with mock data."""
    print("\n" + "=" * 60)
    print("Testing VitalDB Dataset with Mock Data")
    print("=" * 60)
    
    from src.data.vitaldb_dataset import VitalDBDataset
    
    try:
        # Test single channel
        dataset = VitalDBDataset(
            channels='ppg',
            split='train',
            use_raw_vitaldb=True,
            cache_dir='data/mock_cache',
            max_cases=2,
            segments_per_case=3
        )
        
        print(f"✓ Created PPG dataset with {len(dataset)} samples")
        
        # Get a sample
        seg1, seg2 = dataset[0]
        print(f"  Shape: {seg1.shape}")
        print(f"  Range: [{seg1.min():.3f}, {seg1.max():.3f}]")
        
        # Test multi-channel
        dataset_multi = VitalDBDataset(
            channels=['ppg', 'ecg'],
            split='train',
            use_raw_vitaldb=True,
            cache_dir='data/mock_cache',
            max_cases=2,
            segments_per_case=3
        )
        
        print(f"✓ Created multi-modal dataset with {len(dataset_multi)} samples")
        
        seg1, seg2 = dataset_multi[0]
        print(f"  Shape: {seg1.shape}")
        print(f"  PPG range: [{seg1[0].min():.3f}, {seg1[0].max():.3f}]")
        print(f"  ECG range: [{seg1[1].min():.3f}, {seg1[1].max():.3f}]")
        
        # Visualize
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(seg1[0].numpy(), 'b-', alpha=0.7)
        plt.title('Mock PPG - Segment 1')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(seg2[0].numpy(), 'r-', alpha=0.7)
        plt.title('Mock PPG - Segment 2')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        if seg1.shape[0] > 1:
            plt.subplot(2, 2, 3)
            plt.plot(seg1[1].numpy(), 'g-', alpha=0.7)
            plt.title('Mock ECG - Segment 1')
            plt.xlabel('Samples')
            plt.ylabel('Amplitude')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 4)
            plt.plot(seg2[1].numpy(), 'orange', alpha=0.7)
            plt.title('Mock ECG - Segment 2')
            plt.xlabel('Samples')
            plt.ylabel('Amplitude')
            plt.grid(True, alpha=0.3)
            
        plt.suptitle('Mock VitalDB Data (No SSL Required)')
        plt.tight_layout()
        plt.savefig('mock_vitaldb_test.png')
        print("✓ Saved visualization to mock_vitaldb_test.png")
        
        return True
        
    except Exception as e:
        print(f"✗ Mock VitalDB test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_but_ppg():
    """Test BUT PPG dataset."""
    print("\n" + "=" * 60)
    print("Testing BUT PPG Dataset")
    print("=" * 60)
    
    # Check if data exists
    data_dir = Path('data/but_ppg/dataset')
    if not data_dir.exists():
        print("✗ BUT PPG data not found")
        print("  Run: python scripts/download_but_ppg.py")
        return False
        
    from src.data.butppg_dataset import BUTPPGDataset
    
    try:
        dataset = BUTPPGDataset(
            data_dir=str(data_dir),
            modality='all',  # PPG + ECG + ACC
            split='train'
        )
        
        print(f"✓ Created dataset with {len(dataset)} samples")
        
        seg1, seg2 = dataset[0]
        print(f"  Shape: {seg1.shape} (5 channels: PPG, ECG, ACC_X, ACC_Y, ACC_Z)")
        print(f"  PPG range: [{seg1[0].min():.3f}, {seg1[0].max():.3f}]")
        print(f"  ECG range: [{seg1[1].min():.3f}, {seg1[1].max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"✗ BUT PPG test failed: {e}")
        return False


def test_dataloaders():
    """Test dataloader creation."""
    print("\n" + "=" * 60)
    print("Testing DataLoader Creation")
    print("=" * 60)
    
    from src.data.vitaldb_dataset import create_vitaldb_dataloaders
    
    try:
        train_loader, val_loader, test_loader = create_vitaldb_dataloaders(
            channels=['ppg', 'ecg'],
            batch_size=4,
            num_workers=0,
            use_raw_vitaldb=True,
            max_cases=2,
            cache_dir='data/mock_cache'
        )
        
        print(f"✓ Created VitalDB dataloaders:")
        print(f"  Train: {len(train_loader)} batches")
        print(f"  Val: {len(val_loader)} batches")
        print(f"  Test: {len(test_loader)} batches")
        
        # Get one batch
        for batch in train_loader:
            if isinstance(batch, tuple):
                seg1, seg2 = batch
                print(f"  Batch shape: {seg1.shape}")
                print(f"  Batch dtype: {seg1.dtype}")
            else:
                print(f"  Single batch shape: {batch.shape}")
            break
            
        return True
        
    except Exception as e:
        print(f"✗ DataLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Multi-Modal Data Test (No SSL Required)")
    print("=" * 60)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Check error messages above.")
        sys.exit(1)
        
    # Test with mock data (no SSL needed)
    mock_ok = test_mock_vitaldb()
    
    # Test BUT PPG if available
    but_ok = test_but_ppg()
    
    # Test dataloaders
    loader_ok = test_dataloaders()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Mock VitalDB: {'✅ PASSED' if mock_ok else '❌ FAILED'}")
    print(f"  BUT PPG: {'✅ PASSED' if but_ok else '⚠️  Data not found' if not but_ok else '❌ FAILED'}")
    print(f"  DataLoaders: {'✅ PASSED' if loader_ok else '❌ FAILED'}")
    
    if mock_ok and loader_ok:
        print("\n✅ Core functionality working!")
        print("\nTo use real VitalDB data, fix SSL by running:")
        print("  python scripts/fix_ssl.py")
        print("\nThen test with real data:")
        print("  python scripts/test_vitaldb_quick.py")
    else:
        print("\n❌ Some tests failed. Check output above.")
