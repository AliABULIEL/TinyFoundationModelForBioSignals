"""
Tests for patient-level data splitting.
"""

import pytest
import numpy as np
from src.data.splits import (
    make_patient_level_splits,
    verify_no_subject_leakage,
    stratified_patient_split,
    create_cv_folds,
    save_splits,
    load_splits,
    get_split_statistics,
    create_temporal_splits,
    create_leave_one_subject_out_splits
)
import tempfile
import os


def create_mock_cases(n_subjects=10, cases_per_subject=5):
    """Create mock case data for testing."""
    cases = []
    for subj_id in range(n_subjects):
        for case_idx in range(cases_per_subject):
            case_id = f"case_{subj_id}_{case_idx}"
            cases.append({
                'case_id': case_id,
                'subject_id': f"subj_{subj_id}",
                'duration_s': np.random.uniform(100, 1000),
                'available_channels': ['ECG_II', 'PLETH', 'ART'],
                'label': subj_id % 2,  # Binary label for stratification
                'timestamp': subj_id * 100 + case_idx  # For temporal splits
            })
    return cases


def test_make_patient_level_splits_no_leakage():
    """Test that patient-level splits have no subject leakage."""
    cases = create_mock_cases(n_subjects=20, cases_per_subject=3)
    
    # Create 80/20 train/test split
    splits = make_patient_level_splits(cases, ratios=(0.8, 0.2), seed=42)
    
    assert 'train' in splits
    assert 'test' in splits
    assert len(splits['train']) > 0
    assert len(splits['test']) > 0
    
    # Extract subjects from each split
    train_subjects = set(c['subject_id'] for c in splits['train'])
    test_subjects = set(c['subject_id'] for c in splits['test'])
    
    # Check no overlap
    assert len(train_subjects & test_subjects) == 0
    
    # Check approximate split ratio
    total_subjects = len(train_subjects) + len(test_subjects)
    train_ratio = len(train_subjects) / total_subjects
    assert 0.75 <= train_ratio <= 0.85  # Allow some variance due to rounding


def test_make_patient_level_splits_three_way():
    """Test train/val/test splits."""
    cases = create_mock_cases(n_subjects=30, cases_per_subject=4)
    
    # Create 60/20/20 split
    splits = make_patient_level_splits(cases, ratios=(0.6, 0.2, 0.2), seed=42)
    
    assert 'train' in splits
    assert 'val' in splits
    assert 'test' in splits
    
    # Extract subjects
    train_subjects = set(c['subject_id'] for c in splits['train'])
    val_subjects = set(c['subject_id'] for c in splits['val'])
    test_subjects = set(c['subject_id'] for c in splits['test'])
    
    # Check no overlaps
    assert len(train_subjects & val_subjects) == 0
    assert len(train_subjects & test_subjects) == 0
    assert len(val_subjects & test_subjects) == 0
    
    # Check all subjects are accounted for
    all_subjects = set(c['subject_id'] for c in cases)
    split_subjects = train_subjects | val_subjects | test_subjects
    assert all_subjects == split_subjects


def test_verify_no_subject_leakage():
    """Test leakage verification function."""
    # Create splits with no leakage
    good_splits = {
        'train': [{'subject_id': 'subj_0'}, {'subject_id': 'subj_1'}],
        'test': [{'subject_id': 'subj_2'}, {'subject_id': 'subj_3'}]
    }
    
    assert verify_no_subject_leakage(good_splits) == True
    
    # Create splits with leakage
    bad_splits = {
        'train': [{'subject_id': 'subj_0'}, {'subject_id': 'subj_1'}],
        'test': [{'subject_id': 'subj_1'}, {'subject_id': 'subj_2'}]  # subj_1 in both
    }
    
    with pytest.raises(ValueError, match="Subject leakage detected"):
        verify_no_subject_leakage(bad_splits)


def test_stratified_patient_split():
    """Test stratified splitting."""
    cases = create_mock_cases(n_subjects=20, cases_per_subject=3)
    
    # Stratify by label
    splits = stratified_patient_split(cases, ratios=(0.8, 0.2), 
                                     stratify_key='label', seed=42)
    
    # Check label distribution is approximately maintained
    train_labels = [c['label'] for c in splits['train']]
    test_labels = [c['label'] for c in splits['test']]
    
    train_label_ratio = sum(train_labels) / len(train_labels)
    test_label_ratio = sum(test_labels) / len(test_labels)
    
    # Ratios should be similar (within 20%)
    assert abs(train_label_ratio - test_label_ratio) < 0.2
    
    # Verify no leakage
    verify_no_subject_leakage(splits)


def test_create_cv_folds():
    """Test cross-validation fold creation."""
    cases = create_mock_cases(n_subjects=25, cases_per_subject=2)
    
    folds = create_cv_folds(cases, n_folds=5, seed=42)
    
    assert len(folds) == 5
    
    # Check each fold
    all_val_subjects = set()
    for i, fold in enumerate(folds):
        assert 'train' in fold
        assert 'val' in fold
        
        train_subjects = set(c['subject_id'] for c in fold['train'])
        val_subjects = set(c['subject_id'] for c in fold['val'])
        
        # No overlap within fold
        assert len(train_subjects & val_subjects) == 0
        
        # Validation subjects should be unique across folds
        assert len(all_val_subjects & val_subjects) == 0
        all_val_subjects.update(val_subjects)
        
        # Check sizes (80/20 split approximately)
        total_in_fold = len(train_subjects) + len(val_subjects)
        assert 0.75 <= len(train_subjects) / total_in_fold <= 0.85


def test_save_load_splits():
    """Test saving and loading splits."""
    cases = create_mock_cases(n_subjects=10, cases_per_subject=2)
    splits = make_patient_level_splits(cases, ratios=(0.8, 0.2), seed=42)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "splits.json")
        
        # Save
        save_splits(splits, filepath)
        assert os.path.exists(filepath)
        
        # Load
        loaded_splits = load_splits(filepath)
        
        # Check equality
        assert set(loaded_splits.keys()) == set(splits.keys())
        assert len(loaded_splits['train']) == len(splits['train'])
        assert len(loaded_splits['test']) == len(splits['test'])


def test_get_split_statistics():
    """Test split statistics calculation."""
    cases = create_mock_cases(n_subjects=20, cases_per_subject=5)
    splits = make_patient_level_splits(cases, ratios=(0.7, 0.15, 0.15), seed=42)
    
    stats = get_split_statistics(splits)
    
    assert 'train' in stats
    assert 'val' in stats
    assert 'test' in stats
    
    for split_name in ['train', 'val', 'test']:
        assert 'n_cases' in stats[split_name]
        assert 'n_subjects' in stats[split_name]
        assert 'cases_per_subject' in stats[split_name]
        assert 'subject_ratio' in stats[split_name]
        assert 'case_ratio' in stats[split_name]
    
    # Check ratios sum to 1
    total_subject_ratio = sum(s['subject_ratio'] for s in stats.values())
    assert abs(total_subject_ratio - 1.0) < 0.01
    
    total_case_ratio = sum(s['case_ratio'] for s in stats.values())
    assert abs(total_case_ratio - 1.0) < 0.01


def test_temporal_splits():
    """Test temporal splitting (train on earlier, test on later)."""
    cases = create_mock_cases(n_subjects=20, cases_per_subject=3)
    
    splits = create_temporal_splits(cases, time_key='timestamp', ratios=(0.8, 0.2))
    
    # Get timestamps for each split
    train_times = [c['timestamp'] for c in splits['train']]
    test_times = [c['timestamp'] for c in splits['test']]
    
    # Generally, train should have earlier timestamps
    # (though some overlap is possible due to patient-level splitting)
    assert min(train_times) <= min(test_times)
    
    # Verify no subject leakage
    verify_no_subject_leakage(splits)


def test_leave_one_subject_out():
    """Test leave-one-subject-out cross-validation."""
    cases = create_mock_cases(n_subjects=5, cases_per_subject=3)
    
    loso_splits = create_leave_one_subject_out_splits(cases)
    
    # Should have one split per subject
    assert len(loso_splits) == 5
    
    all_subjects = set(c['subject_id'] for c in cases)
    
    for split in loso_splits:
        assert 'train' in split
        assert 'test' in split
        
        train_subjects = set(c['subject_id'] for c in split['train'])
        test_subjects = set(c['subject_id'] for c in split['test'])
        
        # Test should have exactly one subject
        assert len(test_subjects) == 1
        
        # Train should have all others
        assert len(train_subjects) == 4
        
        # No overlap
        assert len(train_subjects & test_subjects) == 0
        
        # Together they should be all subjects
        assert train_subjects | test_subjects == all_subjects


def test_deterministic_splits():
    """Test that splits are deterministic with same seed."""
    cases = create_mock_cases(n_subjects=30, cases_per_subject=4)
    
    # Create splits twice with same seed
    splits1 = make_patient_level_splits(cases, ratios=(0.8, 0.2), seed=42)
    splits2 = make_patient_level_splits(cases, ratios=(0.8, 0.2), seed=42)
    
    # Should be identical
    assert len(splits1['train']) == len(splits2['train'])
    assert len(splits1['test']) == len(splits2['test'])
    
    train1_ids = set(c['case_id'] for c in splits1['train'])
    train2_ids = set(c['case_id'] for c in splits2['train'])
    assert train1_ids == train2_ids
    
    # Different seed should give different splits
    splits3 = make_patient_level_splits(cases, ratios=(0.8, 0.2), seed=123)
    train3_ids = set(c['case_id'] for c in splits3['train'])
    assert train1_ids != train3_ids  # Very unlikely to be identical


def test_empty_cases():
    """Test handling of empty cases."""
    cases = []
    
    splits = make_patient_level_splits(cases, ratios=(0.8, 0.2), seed=42)
    
    assert 'train' in splits
    assert 'test' in splits
    assert len(splits['train']) == 0
    assert len(splits['test']) == 0


def test_single_subject():
    """Test handling of single subject (edge case)."""
    cases = create_mock_cases(n_subjects=1, cases_per_subject=5)
    
    splits = make_patient_level_splits(cases, ratios=(0.8, 0.2), seed=42)
    
    # With only one subject, it should go to train
    assert len(splits['train']) == 5
    assert len(splits['test']) == 0
    
    # No leakage (trivially true)
    verify_no_subject_leakage(splits)


def test_unequal_cases_per_subject():
    """Test with varying number of cases per subject."""
    cases = []
    # Create subjects with different numbers of cases
    for subj_id in range(10):
        n_cases = np.random.randint(1, 10)
        for case_idx in range(n_cases):
            cases.append({
                'case_id': f"case_{subj_id}_{case_idx}",
                'subject_id': f"subj_{subj_id}"
            })
    
    splits = make_patient_level_splits(cases, ratios=(0.7, 0.3), seed=42)
    
    # Verify no leakage
    verify_no_subject_leakage(splits)
    
    # Check that complete subjects are in one split or the other
    for subj_cases in [c for c in cases if c['subject_id'] == 'subj_0']:
        case_id = subj_cases['case_id']
        in_train = any(c['case_id'] == case_id for c in splits['train'])
        in_test = any(c['case_id'] == case_id for c in splits['test'])
        # Should be in exactly one split
        assert in_train != in_test  # XOR
