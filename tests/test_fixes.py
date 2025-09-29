"""Test fixes for windows and splits modules."""

import numpy as np
from src.data.windows import compute_normalization_stats
from src.data.splits import make_patient_level_splits


def test_fixes():
    print("Testing fixes...")
    
    # Test 1: Normalization stats with per-column axis
    X = np.array([[1, 2], [3, 4], [5, 6]])
    
    # Test zscore with axis=0 (per column)
    stats = compute_normalization_stats(X, method="zscore", axis=0)
    print(f"zscore stats: mean={stats.mean}, std={stats.std}")
    assert np.allclose(stats.mean, [3.0, 4.0]), f"Expected mean [3.0, 4.0], got {stats.mean}"
    assert stats.std.shape == (2,), f"Expected shape (2,), got {stats.std.shape}"
    
    # Test minmax with axis=0
    stats = compute_normalization_stats(X, method="minmax", axis=0)
    print(f"minmax stats: min={stats.min}, max={stats.max}")
    assert np.allclose(stats.min, [1, 2]), f"Expected min [1, 2], got {stats.min}"
    assert np.allclose(stats.max, [5, 6]), f"Expected max [5, 6], got {stats.max}"
    
    # Test robust normalization
    X_outlier = np.random.randn(1000, 2)
    X_outlier[0] = 100
    X_outlier[1] = -100
    
    stats_standard = compute_normalization_stats(X_outlier, method="zscore", axis=0, robust=False)
    stats_robust = compute_normalization_stats(X_outlier, method="zscore", axis=0, robust=True)
    
    print(f"Standard mean[0]: {stats_standard.mean[0]:.3f}")
    print(f"Robust mean[0]: {stats_robust.mean[0]:.3f}")
    assert abs(stats_robust.mean[0]) < abs(stats_standard.mean[0]), "Robust should be less affected by outliers"
    
    # Test 2: Single subject split
    cases = [
        {'case_id': f'case_0_{i}', 'subject_id': 'subj_0'} 
        for i in range(5)
    ]
    
    splits = make_patient_level_splits(cases, ratios=(0.8, 0.2), seed=42)
    print(f"Single subject - train: {len(splits['train'])}, test: {len(splits['test'])}")
    assert len(splits['train']) == 5, "Single subject should all go to train"
    assert len(splits['test']) == 0, "Single subject test should be empty"
    
    print("✅ All fixes working!")


if __name__ == "__main__":
    test_fixes()
