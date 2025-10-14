#!/usr/bin/env python3
"""Quick test to verify dataloader creation functions work correctly."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_vitaldb_dataloaders():
    """Test VitalDB dataloader creation."""
    print("Testing VitalDB dataloader creation...")
    
    try:
        from src.data.vitaldb_dataset import create_vitaldb_dataloaders
        
        # Create dataloaders with multi-modal support
        train_loader, val_loader, test_loader = create_vitaldb_dataloaders(
            channels=['ppg', 'ecg'],  # Multi-modal
            batch_size=16,
            num_workers=0,  # Set to 0 for testing
            cache_dir='data/vitaldb_cache',
            use_raw_vitaldb=True,
            max_cases=5  # Limit for quick testing
        )
        
        print(f"✓ VitalDB dataloaders created:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        # Test one batch
        for batch in train_loader:
            if len(batch) == 2:
                seg1, seg2 = batch
                print(f"  Batch shape: {seg1.shape}")  # Should be [16, 2, 1250]
                break
        
        return True
        
    except Exception as e:
        print(f"✗ VitalDB test failed: {e}")
        return False


def test_butppg_dataloaders():
    """Test BUT PPG dataloader creation."""
    print("\nTesting BUT PPG dataloader creation...")
    
    try:
        from src.data.butppg_dataset import create_butppg_dataloaders
        
        data_dir = Path('data/but_ppg/dataset')
        if not data_dir.exists():
            print(f"⚠ BUT PPG data not found at {data_dir}")
            return False
        
        # Create dataloaders with all modalities
        train_loader, val_loader, test_loader = create_butppg_dataloaders(
            data_dir=str(data_dir),
            modality='all',  # PPG + ECG + ACC
            batch_size=16,
            num_workers=0,  # Set to 0 for testing
            quality_filter=False
        )
        
        print(f"✓ BUT PPG dataloaders created:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        # Test one batch
        for batch in train_loader:
            if len(batch) >= 2:
                seg1, seg2 = batch[0], batch[1]
                print(f"  Batch shape: {seg1.shape}")  # Should be [16, 5, 1250]
                break
        
        return True
        
    except Exception as e:
        print(f"✗ BUT PPG test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_import_from_init():
    """Test that imports from __init__.py work correctly."""
    print("\nTesting imports from __init__.py...")
    
    try:
        from src.data.vitaldb_dataset import create_vitaldb_dataloaders, create_butppg_dataloaders
        print("✓ Successfully imported dataloader creation functions from __init__.py")
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def main():
    print("=" * 60)
    print("DATALOADER CREATION TEST")
    print("=" * 60)
    
    results = []
    
    # Test VitalDB
    results.append(("VitalDB", test_vitaldb_dataloaders()))
    
    # Test BUT PPG
    results.append(("BUT PPG", test_butppg_dataloaders()))
    
    # Test imports
    results.append(("Imports", test_import_from_init()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n✅ ALL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
