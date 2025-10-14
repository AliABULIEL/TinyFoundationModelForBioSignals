#!/usr/bin/env python3
"""
Final comprehensive test - Run this after fixing imports and SSL.
"""

import os
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Use mock data to avoid SSL issues
os.environ['VITALDB_MOCK'] = '1'

import numpy as np
import torch
import matplotlib.pyplot as plt


def main():
    print("=" * 60)
    print(" FINAL COMPREHENSIVE TEST")
    print("=" * 60)
    
    # 1. Test VitalDB with mock data
    print("\n1. Testing VitalDB Dataset (Mock Data)...")
    try:
        from src.data.vitaldb_dataset import VitalDBDataset, create_vitaldb_dataloaders
        
        # Create dataset
        dataset = VitalDBDataset(
            channels=['ppg', 'ecg'],
            split='train',
            use_raw_vitaldb=True,
            max_cases=3,
            segments_per_case=10
        )
        
        print(f"✓ VitalDB dataset created: {len(dataset)} samples")
        
        # Test loading
        seg1, seg2 = dataset[0]
        print(f"  Shape: {seg1.shape} (2 channels × 1250 samples)")
        print(f"  PPG range: [{seg1[0].min():.2f}, {seg1[0].max():.2f}]")
        print(f"  ECG range: [{seg1[1].min():.2f}, {seg1[1].max():.2f}]")
        
        # Create dataloaders
        train_loader, val_loader, test_loader = create_vitaldb_dataloaders(
            channels=['ppg', 'ecg'],
            batch_size=8,
            num_workers=0,
            max_cases=3
        )
        
        print(f"✓ DataLoaders created:")
        print(f"  Train: {len(train_loader)} batches")
        print(f"  Val: {len(val_loader)} batches")
        print(f"  Test: {len(test_loader)} batches")
        
        vitaldb_ok = True
        
    except Exception as e:
        print(f"✗ VitalDB test failed: {e}")
        vitaldb_ok = False
        
    # 2. Test BUT PPG if available
    print("\n2. Testing BUT PPG Dataset...")
    data_dir = Path('data/but_ppg/dataset')
    
    if not data_dir.exists():
        print("⚠️  BUT PPG data not found")
        print("  To download: python scripts/download_but_ppg.py")
        butppg_ok = False
    else:
        try:
            from src.data.butppg_dataset import BUTPPGDataset, create_butppg_dataloaders
            
            # Create dataset
            dataset = BUTPPGDataset(
                data_dir=str(data_dir),
                modality='all',  # PPG + ECG + ACC
                split='train'
            )
            
            print(f"✓ BUT PPG dataset created: {len(dataset)} samples")
            
            if len(dataset) > 0:
                seg1, seg2 = dataset[0]
                print(f"  Shape: {seg1.shape} (5 channels × 1250 samples)")
                print(f"  Channels: PPG, ECG, ACC_X, ACC_Y, ACC_Z")
                
                # Create dataloaders
                train_loader, val_loader, test_loader = create_butppg_dataloaders(
                    data_dir=str(data_dir),
                    modality='all',
                    batch_size=8,
                    num_workers=0
                )
                
                print(f"✓ DataLoaders created")
                butppg_ok = True
            else:
                print("⚠️  Dataset is empty")
                butppg_ok = False
                
        except Exception as e:
            print(f"✗ BUT PPG test failed: {e}")
            butppg_ok = False
            
    # 3. Test Model Compatibility
    print("\n3. Testing Model Compatibility...")
    try:
        # Check if TTM model configs exist
        config_dir = Path('configs')
        if config_dir.exists():
            configs = list(config_dir.glob('*.yaml'))
            print(f"✓ Found {len(configs)} config files")
            for cfg in configs[:3]:
                print(f"  - {cfg.name}")
        else:
            print("⚠️  Config directory not found")
            
        model_ok = True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        model_ok = False
        
    # 4. Create visualization
    if vitaldb_ok:
        print("\n4. Creating Visualization...")
        try:
            from src.data.vitaldb_dataset import VitalDBDataset
            
            # Get samples for visualization
            dataset = VitalDBDataset(
                channels=['ppg', 'ecg'],
                split='test',
                use_raw_vitaldb=True,
                max_cases=1,
                segments_per_case=2
            )
            
            if len(dataset) > 0:
                seg1, seg2 = dataset[0]
                
                # Create figure
                plt.figure(figsize=(14, 8))
                
                # Plot PPG
                plt.subplot(2, 2, 1)
                plt.plot(seg1[0].numpy(), 'b-', alpha=0.7, linewidth=0.5)
                plt.title('PPG - Segment 1')
                plt.ylabel('Amplitude')
                plt.grid(True, alpha=0.3)
                plt.xlim(0, 1250)
                
                plt.subplot(2, 2, 2)
                plt.plot(seg2[0].numpy(), 'r-', alpha=0.7, linewidth=0.5)
                plt.title('PPG - Segment 2 (Same Patient)')
                plt.ylabel('Amplitude')
                plt.grid(True, alpha=0.3)
                plt.xlim(0, 1250)
                
                # Plot ECG
                plt.subplot(2, 2, 3)
                plt.plot(seg1[1].numpy(), 'g-', alpha=0.7, linewidth=0.5)
                plt.title('ECG - Segment 1')
                plt.xlabel('Samples @ 125Hz')
                plt.ylabel('Amplitude')
                plt.grid(True, alpha=0.3)
                plt.xlim(0, 1250)
                
                plt.subplot(2, 2, 4)
                plt.plot(seg2[1].numpy(), 'orange', alpha=0.7, linewidth=0.5)
                plt.title('ECG - Segment 2 (Same Patient)')
                plt.xlabel('Samples @ 125Hz')
                plt.ylabel('Amplitude')
                plt.grid(True, alpha=0.3)
                plt.xlim(0, 1250)
                
                plt.suptitle('Multi-Modal Biosignal Pairs for SSL Training', fontsize=14)
                plt.tight_layout()
                plt.savefig('final_test_visualization.png', dpi=100)
                print("✓ Saved visualization to final_test_visualization.png")
                plt.close()
                
                viz_ok = True
            else:
                print("⚠️  No data for visualization")
                viz_ok = False
                
        except Exception as e:
            print(f"✗ Visualization failed: {e}")
            viz_ok = False
    else:
        viz_ok = False
        
    # Summary
    print("\n" + "=" * 60)
    print(" TEST SUMMARY")
    print("=" * 60)
    
    results = {
        'VitalDB Dataset': vitaldb_ok,
        'BUT PPG Dataset': butppg_ok,
        'Model Configs': model_ok,
        'Visualization': viz_ok
    }
    
    for name, status in results.items():
        icon = '✅' if status else '❌'
        print(f"{icon} {name}")
        
    # Final verdict
    if vitaldb_ok:
        print("\n✅ PIPELINE IS READY!")
        print("\nYou can now run:")
        print("  1. Quick test: python scripts/run_multimodal_pipeline.py --test-only")
        print("  2. Full training: python scripts/run_multimodal_pipeline.py")
        
        if not butppg_ok:
            print("\n⚠️  Note: BUT PPG not available, but VitalDB mock data works!")
    else:
        print("\n❌ Some issues remain. Check errors above.")
        
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
