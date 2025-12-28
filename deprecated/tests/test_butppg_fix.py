#!/usr/bin/env python3
"""
Test script to verify BUT-PPG preprocessing fixes.
"""

import sys
import tempfile
import numpy as np
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.butppg_dataset import BUTPPGDataset


def test_butppg_preprocessing():
    """Test that BUT-PPG preprocessing works correctly."""
    
    print("\n" + "="*70)
    print("Testing BUT-PPG Preprocessing Fix")
    print("="*70)
    
    # Create temporary test data
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "but_ppg"
        data_dir.mkdir()
        
        # Create test PPG signals (60 seconds at 125 Hz)
        fs = 125.0
        duration = 60.0
        n_samples = int(fs * duration)
        
        for subject_id in ['subject_001', 'subject_002']:
            # Create realistic PPG signal
            t = np.arange(n_samples) / fs
            hr = 75  # 75 bpm
            ppg_signal = (
                np.sin(2 * np.pi * (hr/60) * t) +  # Main pulse
                0.3 * np.sin(4 * np.pi * (hr/60) * t) +  # Dicrotic notch
                0.1 * np.random.randn(n_samples)  # Noise
            )
            
            # Save as NPY
            np.save(data_dir / f"{subject_id}.npy", ppg_signal)
            
            # Save metadata
            with open(data_dir / f"{subject_id}.json", 'w') as f:
                json.dump({'fs': fs}, f)
        
        print(f"\n✓ Created test data: {data_dir}")
        print(f"  - 2 subjects")
        print(f"  - {duration}s at {fs} Hz")
        print(f"  - {n_samples} samples per subject")
        
        # Create dataset
        print("\nInitializing BUT-PPG Dataset...")
        dataset = BUTPPGDataset(
            data_dir=data_dir,
            split='train',
            modality='ppg',
            train_ratio=0.7,
            val_ratio=0.15,
            window_sec=10.0,
            hop_sec=5.0,
            enable_qc=True,
            segments_per_subject=5
        )
        
        print(f"\n✓ Dataset initialized:")
        print(f"  - {len(dataset.subjects)} subjects")
        print(f"  - {len(dataset)} total segments")
        
        # Test loading samples
        print("\nTesting sample loading...")
        means = []
        stds = []
        n_valid = 0
        
        for i in range(min(5, len(dataset))):
            try:
                seg1, seg2 = dataset[i]
                
                # Check shapes
                assert seg1.shape == (1, 1250), f"Wrong seg1 shape: {seg1.shape}"
                assert seg2.shape == (1, 1250), f"Wrong seg2 shape: {seg2.shape}"
                
                # Check if not all zeros
                if not torch.all(seg1 == 0):
                    means.append(float(seg1.mean()))
                    stds.append(float(seg1.std()))
                    n_valid += 1
                
                print(f"  Sample {i}: seg1 mean={seg1.mean():.3f}, std={seg1.std():.3f}")
                
            except Exception as e:
                print(f"  Sample {i}: ERROR - {e}")
        
        print(f"\n✓ Loaded {n_valid} valid samples")
        
        if n_valid > 0:
            avg_mean = np.mean(means)
            avg_std = np.mean(stds)
            
            print(f"\nNormalization Statistics:")
            print(f"  Average mean: {avg_mean:.4f}")
            print(f"  Average std:  {avg_std:.4f}")
            
            # Check normalization
            if abs(avg_mean) < 0.5 and 0.5 < avg_std < 1.5:
                print("\n✅ PASS: Preprocessing is working correctly!")
                print("  - Signals are properly normalized (mean≈0, std≈1)")
                return True
            else:
                print("\n❌ FAIL: Preprocessing not working correctly")
                print(f"  - Expected mean≈0, got {avg_mean:.4f}")
                print(f"  - Expected std≈1, got {avg_std:.4f}")
                return False
        else:
            print("\n❌ FAIL: No valid samples loaded")
            return False


if __name__ == '__main__':
    import torch
    success = test_butppg_preprocessing()
    sys.exit(0 if success else 1)
