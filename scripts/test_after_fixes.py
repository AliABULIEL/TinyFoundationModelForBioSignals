#!/usr/bin/env python3
"""
Quick test to verify both datasets work after fixes.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch


def test_vitaldb_fixed():
    """Test VitalDB after fixing the array comparison bug."""
    print("=" * 60)
    print("Testing VitalDB (Fixed)")
    print("=" * 60)
    
    try:
        from src.data.vitaldb_dataset import VitalDBDataset
        
        # Test single modality
        dataset = VitalDBDataset(
            cache_dir='data/vitaldb_cache',
            channels='ppg',
            split='train',
            use_raw_vitaldb=True,
            max_cases=3
        )
        
        if len(dataset) > 0:
            seg1, seg2 = dataset[0]
            print(f"✓ PPG shape: {seg1.shape}")
            print(f"  Mean: {seg1.mean():.3f}, Std: {seg1.std():.3f}")
        
        # Test multi-modal
        dataset_multi = VitalDBDataset(
            cache_dir='data/vitaldb_cache',
            channels=['ppg', 'ecg'],
            split='train',
            use_raw_vitaldb=True,
            max_cases=3
        )
        
        if len(dataset_multi) > 0:
            seg1, seg2 = dataset_multi[0]
            print(f"\n✓ Multi-modal shape: {seg1.shape}")
            print(f"  PPG mean: {seg1[0].mean():.3f}")
            print(f"  ECG mean: {seg1[1].mean():.3f}")
            
        print("\n✅ VitalDB working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ VitalDB error: {e}")
        return False


def test_butppg_after_download():
    """Test BUT PPG after downloading."""
    print("\n" + "=" * 60)
    print("Testing BUT PPG")
    print("=" * 60)
    
    data_dir = Path('data/but_ppg/dataset')
    
    if not data_dir.exists():
        print("❌ BUT PPG not downloaded yet")
        print("  Run: python scripts/download_but_ppg.py")
        return False
    
    try:
        from src.data.butppg_dataset import BUTPPGDataset
        
        # Test all modalities
        dataset = BUTPPGDataset(
            data_dir=str(data_dir),
            modality='all',  # PPG + ECG + ACC
            split='train'
        )
        
        if len(dataset) > 0:
            seg1, seg2 = dataset[0]
            print(f"✓ All modalities shape: {seg1.shape}")  # Should be [5, 1250]
            print(f"  PPG mean: {seg1[0].mean():.3f}")
            print(f"  ECG mean: {seg1[1].mean():.3f}")
            print(f"  ACC-X mean: {seg1[2].mean():.3f}")
            
        print("\n✅ BUT PPG working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ BUT PPG error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloaders():
    """Test dataloader creation."""
    print("\n" + "=" * 60)
    print("Testing DataLoaders")
    print("=" * 60)
    
    try:
        from src.data import create_vitaldb_dataloaders, create_butppg_dataloaders
        
        # Test VitalDB dataloaders
        train_loader, val_loader, test_loader = create_vitaldb_dataloaders(
            channels=['ppg', 'ecg'],
            batch_size=4,
            num_workers=0,
            max_cases=5
        )
        
        print(f"✓ VitalDB dataloaders created")
        print(f"  Train: {len(train_loader)} batches")
        
        # Test one batch
        for batch in train_loader:
            seg1, seg2 = batch
            print(f"  Batch shape: {seg1.shape}")
            break
        
        # Test BUT PPG if available
        if Path('data/but_ppg/dataset').exists():
            train_loader, val_loader, test_loader = create_butppg_dataloaders(
                data_dir='data/but_ppg/dataset',
                modality='all',
                batch_size=4,
                num_workers=0
            )
            
            print(f"\n✓ BUT PPG dataloaders created")
            print(f"  Train: {len(train_loader)} batches")
            
        return True
        
    except Exception as e:
        print(f"❌ DataLoader error: {e}")
        return False


def main():
    print("\n" + "=" * 60)
    print("POST-FIX VERIFICATION TEST")
    print("=" * 60)
    
    results = []
    
    # Test VitalDB
    results.append(("VitalDB", test_vitaldb_fixed()))
    
    # Test BUT PPG
    results.append(("BUT PPG", test_butppg_after_download()))
    
    # Test DataLoaders
    results.append(("DataLoaders", test_dataloaders()))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name}: {status}")
    
    # Instructions if BUT PPG not downloaded
    if not results[1][1]:  # BUT PPG failed
        print("\n📝 To download BUT PPG dataset:")
        print("  python scripts/download_but_ppg.py")
        print("\nThis will download ~87MB from PhysioNet")


if __name__ == "__main__":
    main()
