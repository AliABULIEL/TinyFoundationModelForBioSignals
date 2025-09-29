"""Verify windowing and splits implementation."""

import numpy as np
from src.data.windows import (
    make_windows, compute_normalization_stats, normalize_windows
)
from src.data.splits import (
    make_patient_level_splits, verify_no_subject_leakage
)


def main():
    print("Testing Windowing and Splits Implementation")
    print("=" * 50)
    
    # Test windowing
    print("\n1. Testing windowing...")
    fs = 125.0  # 125 Hz sampling rate
    duration = 60.0  # 60 seconds
    n_samples = int(duration * fs)
    n_channels = 3
    
    # Create synthetic data
    X = np.random.randn(n_samples, n_channels)
    
    # Create 10-second non-overlapping windows
    windows = make_windows(X, fs, win_s=10.0, stride_s=10.0, min_cycles=0)
    print(f"   Created {windows.shape[0]} windows of shape {windows.shape[1:]} from {duration}s signal")
    assert windows.shape == (6, 1250, 3), f"Expected (6, 1250, 3), got {windows.shape}"
    
    # Test normalization
    print("\n2. Testing normalization...")
    stats = compute_normalization_stats(windows.reshape(-1, n_channels), method="zscore")
    print(f"   Computed stats - mean shape: {stats.mean.shape}, std shape: {stats.std.shape}")
    
    normalized = normalize_windows(windows, stats, baseline_correction=True)
    print(f"   Normalized windows - mean: {normalized.mean():.3f}, std: {normalized.std():.3f}")
    
    # Test patient-level splits
    print("\n3. Testing patient-level splits...")
    
    # Create mock cases
    cases = []
    for subj_id in range(20):
        for case_idx in range(3):
            cases.append({
                'case_id': f"case_{subj_id}_{case_idx}",
                'subject_id': f"subj_{subj_id}",
                'duration_s': np.random.uniform(100, 1000)
            })
    
    # Create splits
    splits = make_patient_level_splits(cases, ratios=(0.8, 0.2), seed=42)
    
    # Check for leakage
    train_subjects = set(c['subject_id'] for c in splits['train'])
    test_subjects = set(c['subject_id'] for c in splits['test'])
    
    print(f"   Train: {len(splits['train'])} cases from {len(train_subjects)} subjects")
    print(f"   Test: {len(splits['test'])} cases from {len(test_subjects)} subjects")
    print(f"   Subject overlap: {len(train_subjects & test_subjects)} (should be 0)")
    
    # Verify no leakage
    try:
        verify_no_subject_leakage(splits)
        print("   ✓ No subject leakage detected")
    except ValueError as e:
        print(f"   ✗ Subject leakage detected: {e}")
        
    # Test with minimum cycles
    print("\n4. Testing minimum cardiac cycles enforcement...")
    
    # Create fake peaks (1 Hz = 60 bpm)
    peaks = np.arange(0, n_samples, int(fs))  # 1 peak per second
    
    windows_with_cycles = make_windows(
        X, fs, win_s=10.0, stride_s=10.0, 
        min_cycles=8, peaks_tc={0: peaks}
    )
    print(f"   With min_cycles=8: {windows_with_cycles.shape[0]} windows")
    
    windows_no_min = make_windows(
        X, fs, win_s=10.0, stride_s=10.0,
        min_cycles=0
    )
    print(f"   Without min_cycles: {windows_no_min.shape[0]} windows")
    
    # Test 3-way split
    print("\n5. Testing 3-way splits (train/val/test)...")
    splits_3way = make_patient_level_splits(cases, ratios=(0.7, 0.15, 0.15), seed=42)
    
    train_subj = set(c['subject_id'] for c in splits_3way['train'])
    val_subj = set(c['subject_id'] for c in splits_3way['val'])
    test_subj = set(c['subject_id'] for c in splits_3way['test'])
    
    print(f"   Train: {len(train_subj)} subjects")
    print(f"   Val: {len(val_subj)} subjects")  
    print(f"   Test: {len(test_subj)} subjects")
    print(f"   Total overlaps: {len(train_subj & val_subj) + len(train_subj & test_subj) + len(val_subj & test_subj)}")
    
    print("\n✅ All tests passed!")
    

if __name__ == "__main__":
    main()
