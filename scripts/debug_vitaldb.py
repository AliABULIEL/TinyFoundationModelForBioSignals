#!/usr/bin/env python3
"""Debug VitalDB data loading to identify NaN/zero issues."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from src.data.vitaldb_loader import load_channel, find_cases_with_track
from src.data.vitaldb_dataset import VitalDBDataset


def test_direct_loading():
    """Test loading directly from VitalDB API."""
    print("=" * 60)
    print("Testing Direct VitalDB Loading")
    print("=" * 60)
    
    # Find cases with PPG
    ppg_cases = find_cases_with_track('PLETH', max_cases=3)
    print(f"\nFound {len(ppg_cases)} cases with PPG")
    
    if len(ppg_cases) > 0:
        case_id = ppg_cases[0]
        print(f"\nTesting case {case_id}:")
        
        # Load PPG
        try:
            signal, fs = load_channel(
                case_id=str(case_id),
                channel='PLETH',
                use_cache=True,
                cache_dir='data/vitaldb_cache_debug',
                auto_fix_alternating=True
            )
            
            print(f"  PPG loaded: {len(signal)} samples @ {fs}Hz")
            print(f"  Range: [{np.min(signal):.3f}, {np.max(signal):.3f}]")
            print(f"  Mean: {np.mean(signal):.3f}, Std: {np.std(signal):.3f}")
            print(f"  NaN ratio: {np.mean(np.isnan(signal)):.1%}")
            
            # Plot first 10 seconds
            if len(signal) > 0:
                plt.figure(figsize=(12, 4))
                samples_to_plot = min(int(fs * 10), len(signal))
                t = np.arange(samples_to_plot) / fs
                plt.plot(t, signal[:samples_to_plot], 'b-', alpha=0.7)
                plt.title(f'VitalDB Case {case_id} - PPG (First 10s)')
                plt.xlabel('Time (s)')
                plt.ylabel('Amplitude')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig('vitaldb_debug_ppg.png')
                print(f"  ✓ Saved plot to vitaldb_debug_ppg.png")
                
        except Exception as e:
            print(f"  ✗ Failed to load PPG: {e}")
            
    # Find cases with both PPG and ECG
    print("\n" + "=" * 60)
    print("Finding cases with both PPG and ECG...")
    ecg_cases = find_cases_with_track('ECG_II', max_cases=10)
    both_cases = list(set(ppg_cases) & set(ecg_cases))
    print(f"Found {len(both_cases)} cases with both PPG and ECG")
    
    if len(both_cases) > 0:
        case_id = both_cases[0]
        print(f"\nTesting multi-modal case {case_id}:")
        
        # Load ECG
        try:
            signal, fs = load_channel(
                case_id=str(case_id),
                channel='ECG_II',
                use_cache=True,
                cache_dir='data/vitaldb_cache_debug'
            )
            
            print(f"  ECG loaded: {len(signal)} samples @ {fs}Hz")
            print(f"  Range: [{np.min(signal):.3f}, {np.max(signal):.3f}]")
            print(f"  Mean: {np.mean(signal):.3f}, Std: {np.std(signal):.3f}")
            print(f"  NaN ratio: {np.mean(np.isnan(signal)):.1%}")
            
        except Exception as e:
            print(f"  ✗ Failed to load ECG: {e}")


def test_dataset_loading():
    """Test loading through VitalDBDataset."""
    print("\n" + "=" * 60)
    print("Testing VitalDBDataset")
    print("=" * 60)
    
    # Test single modality
    try:
        dataset = VitalDBDataset(
            channels='ppg',
            split='train',
            use_raw_vitaldb=True,
            cache_dir='data/vitaldb_cache_debug',
            max_cases=3,
            segments_per_case=5
        )
        
        # Get first sample
        seg1, seg2 = dataset[0]
        print(f"\nPPG-only dataset:")
        print(f"  Segment 1 shape: {seg1.shape}")
        print(f"  Segment 1 range: [{seg1.min():.3f}, {seg1.max():.3f}]")
        print(f"  Segment 1 mean: {seg1.mean():.3f}, std: {seg1.std():.3f}")
        print(f"  Zero ratio: {(seg1 == 0).float().mean():.1%}")
        
        # Visualize
        plt.figure(figsize=(14, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(seg1[0].numpy(), 'b-', alpha=0.7, label='PPG')
        plt.title('Segment 1')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(seg2[0].numpy(), 'r-', alpha=0.7, label='PPG')
        plt.title('Segment 2 (Paired from same patient)')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('vitaldb_dataset_ppg.png')
        print(f"  ✓ Saved plot to vitaldb_dataset_ppg.png")
        
    except Exception as e:
        print(f"  ✗ PPG dataset failed: {e}")
        import traceback
        traceback.print_exc()
        
    # Test multi-modal
    try:
        print("\nTesting multi-modal dataset (PPG + ECG)...")
        dataset = VitalDBDataset(
            channels=['ppg', 'ecg'],
            split='train',
            use_raw_vitaldb=True,
            cache_dir='data/vitaldb_cache_debug',
            max_cases=5,
            segments_per_case=3
        )
        
        # Get first sample
        seg1, seg2 = dataset[0]
        print(f"\nMulti-modal dataset:")
        print(f"  Segment shape: {seg1.shape}")
        print(f"  PPG range: [{seg1[0].min():.3f}, {seg1[0].max():.3f}]")
        print(f"  ECG range: [{seg1[1].min():.3f}, {seg1[1].max():.3f}]")
        
        # Visualize multi-modal
        plt.figure(figsize=(14, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(seg1[0].numpy(), 'b-', alpha=0.7)
        plt.title('Segment 1 - PPG')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(seg1[1].numpy(), 'g-', alpha=0.7)
        plt.title('Segment 1 - ECG')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.plot(seg2[0].numpy(), 'b-', alpha=0.7)
        plt.title('Segment 2 - PPG')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        plt.plot(seg2[1].numpy(), 'g-', alpha=0.7)
        plt.title('Segment 2 - ECG')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        plt.suptitle('Multi-Modal VitalDB Sample')
        plt.tight_layout()
        plt.savefig('vitaldb_dataset_multimodal.png')
        print(f"  ✓ Saved plot to vitaldb_dataset_multimodal.png")
        
    except Exception as e:
        print(f"  ✗ Multi-modal dataset failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("VitalDB Debug Tool")
    print("=" * 60)
    
    # Test direct loading first
    test_direct_loading()
    
    # Then test dataset
    test_dataset_loading()
    
    print("\n" + "=" * 60)
    print("Debug complete! Check the generated PNG files.")
    print("=" * 60)
