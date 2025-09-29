"""
Patient-level data splitting to prevent subject leakage.

Ensures no patient appears in both train and test sets.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import hashlib
import json


def make_patient_level_splits(
    cases: List[Dict],
    ratios: Tuple[float, ...] = (0.8, 0.2),
    seed: int = 42,
    stratify_by: Optional[str] = None
) -> Dict[str, List[Dict]]:
    """
    Create patient-level train/val/test splits.
    
    Args:
        cases: List of case dictionaries with 'case_id' and 'subject_id'
        ratios: Split ratios (must sum to 1.0)
        seed: Random seed for reproducibility
        stratify_by: Optional field to stratify splits
        
    Returns:
        Dictionary with 'train', 'val', 'test' splits
    """
    # Validate ratios
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios)}")
    
    # Group cases by subject
    subject_cases = defaultdict(list)
    for case in cases:
        subject_id = case.get('subject_id', case.get('case_id', ''))
        subject_cases[subject_id].append(case)
    
    # Get unique subjects
    subjects = list(subject_cases.keys())
    n_subjects = len(subjects)
    
    # Handle edge case of single subject
    if n_subjects == 1:
        # Put single subject in train
        splits = {'train': cases, 'test': []}
        if len(ratios) == 3:
            splits['val'] = []
        return splits
    
    # Set random seed
    rng = np.random.RandomState(seed)
    
    # Shuffle subjects
    rng.shuffle(subjects)
    
    # Calculate split points
    if len(ratios) == 2:
        # Train/test split
        n_train = int(n_subjects * ratios[0])
        train_subjects = subjects[:n_train]
        test_subjects = subjects[n_train:]
        
        splits = {
            'train': [],
            'test': []
        }
        
        for subj in train_subjects:
            splits['train'].extend(subject_cases[subj])
        for subj in test_subjects:
            splits['test'].extend(subject_cases[subj])
    
    elif len(ratios) == 3:
        # Train/val/test split
        n_train = int(n_subjects * ratios[0])
        n_val = int(n_subjects * ratios[1])
        
        train_subjects = subjects[:n_train]
        val_subjects = subjects[n_train:n_train + n_val]
        test_subjects = subjects[n_train + n_val:]
        
        splits = {
            'train': [],
            'val': [],
            'test': []
        }
        
        for subj in train_subjects:
            splits['train'].extend(subject_cases[subj])
        for subj in val_subjects:
            splits['val'].extend(subject_cases[subj])
        for subj in test_subjects:
            splits['test'].extend(subject_cases[subj])
    
    else:
        raise ValueError(f"Only 2 or 3 splits supported, got {len(ratios)}")
    
    # Verify no subject leakage
    verify_no_subject_leakage(splits)
    
    return splits


def verify_no_subject_leakage(splits: Dict[str, List[Dict]]) -> bool:
    """
    Verify no subject appears in multiple splits.
    
    Args:
        splits: Dictionary of split names to case lists
        
    Returns:
        True if no leakage detected
        
    Raises:
        ValueError if subject leakage detected
    """
    split_subjects = {}
    
    for split_name, cases in splits.items():
        subjects = set()
        for case in cases:
            subject_id = case.get('subject_id', case.get('case_id', ''))
            subjects.add(subject_id)
        split_subjects[split_name] = subjects
    
    # Check for overlaps
    split_names = list(splits.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            overlap = split_subjects[split_names[i]] & split_subjects[split_names[j]]
            if overlap:
                raise ValueError(
                    f"Subject leakage detected between {split_names[i]} and {split_names[j]}: "
                    f"{overlap}"
                )
    
    return True


def stratified_patient_split(
    cases: List[Dict],
    ratios: Tuple[float, ...],
    stratify_key: str,
    seed: int = 42
) -> Dict[str, List[Dict]]:
    """
    Create stratified patient-level splits.
    
    Args:
        cases: List of case dictionaries
        ratios: Split ratios
        stratify_key: Key to stratify by (e.g., 'label', 'severity')
        seed: Random seed
        
    Returns:
        Dictionary with stratified splits
    """
    # Group by stratification key
    strata = defaultdict(list)
    for case in cases:
        strata_value = case.get(stratify_key, 'unknown')
        strata[strata_value].append(case)
    
    # Split each stratum
    all_splits = defaultdict(list)
    
    for strata_value, strata_cases in strata.items():
        strata_splits = make_patient_level_splits(strata_cases, ratios, seed)
        for split_name, split_cases in strata_splits.items():
            all_splits[split_name].extend(split_cases)
    
    return dict(all_splits)


def create_cv_folds(
    cases: List[Dict],
    n_folds: int = 5,
    seed: int = 42
) -> List[Dict[str, List[Dict]]]:
    """
    Create cross-validation folds with patient-level splitting.
    
    Args:
        cases: List of case dictionaries
        n_folds: Number of CV folds
        seed: Random seed
        
    Returns:
        List of fold dictionaries with 'train' and 'val' keys
    """
    # Group cases by subject
    subject_cases = defaultdict(list)
    for case in cases:
        subject_id = case.get('subject_id', case.get('case_id', ''))
        subject_cases[subject_id].append(case)
    
    subjects = list(subject_cases.keys())
    n_subjects = len(subjects)
    
    # Shuffle subjects
    rng = np.random.RandomState(seed)
    rng.shuffle(subjects)
    
    # Create folds
    fold_size = n_subjects // n_folds
    folds = []
    
    for i in range(n_folds):
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < n_folds - 1 else n_subjects
        
        val_subjects = subjects[start_idx:end_idx]
        train_subjects = subjects[:start_idx] + subjects[end_idx:]
        
        fold = {
            'train': [],
            'val': []
        }
        
        for subj in train_subjects:
            fold['train'].extend(subject_cases[subj])
        for subj in val_subjects:
            fold['val'].extend(subject_cases[subj])
        
        folds.append(fold)
    
    return folds


def save_splits(splits: Dict[str, List[Dict]], output_path: str):
    """
    Save splits to JSON file.
    
    Args:
        splits: Dictionary of splits
        output_path: Path to save JSON file
    """
    import json
    from ..utils.paths import ensure_parent_dir
    
    ensure_parent_dir(output_path)
    
    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=2)


def load_splits(input_path: str) -> Dict[str, List[Dict]]:
    """
    Load splits from JSON file.
    
    Args:
        input_path: Path to JSON file
        
    Returns:
        Dictionary of splits
    """
    import json
    
    with open(input_path, 'r') as f:
        return json.load(f)


def get_split_statistics(splits: Dict[str, List[Dict]]) -> Dict:
    """
    Get statistics about splits.
    
    Args:
        splits: Dictionary of splits
        
    Returns:
        Statistics dictionary
    """
    stats = {}
    
    for split_name, cases in splits.items():
        # Count unique subjects
        subjects = set()
        for case in cases:
            subject_id = case.get('subject_id', case.get('case_id', ''))
            subjects.add(subject_id)
        
        stats[split_name] = {
            'n_cases': len(cases),
            'n_subjects': len(subjects),
            'cases_per_subject': len(cases) / len(subjects) if subjects else 0
        }
    
    # Calculate ratios
    total_subjects = sum(s['n_subjects'] for s in stats.values())
    total_cases = sum(s['n_cases'] for s in stats.values())
    
    for split_name in stats:
        stats[split_name]['subject_ratio'] = stats[split_name]['n_subjects'] / total_subjects if total_subjects > 0 else 0
        stats[split_name]['case_ratio'] = stats[split_name]['n_cases'] / total_cases if total_cases > 0 else 0
    
    return stats


def create_temporal_splits(
    cases: List[Dict],
    time_key: str = 'timestamp',
    ratios: Tuple[float, ...] = (0.8, 0.2)
) -> Dict[str, List[Dict]]:
    """
    Create temporal splits (train on earlier, test on later).
    
    Args:
        cases: List of case dictionaries with timestamps
        time_key: Key containing timestamp
        ratios: Split ratios
        
    Returns:
        Dictionary with temporal splits
    """
    # Sort cases by time
    sorted_cases = sorted(cases, key=lambda x: x.get(time_key, 0))
    
    # Group by subject to maintain patient-level splitting
    subject_cases = defaultdict(list)
    for case in sorted_cases:
        subject_id = case.get('subject_id', case.get('case_id', ''))
        subject_cases[subject_id].append(case)
    
    # Get earliest timestamp per subject
    subject_times = {}
    for subj, subj_cases in subject_cases.items():
        earliest = min(c.get(time_key, float('inf')) for c in subj_cases)
        subject_times[subj] = earliest
    
    # Sort subjects by earliest appearance
    sorted_subjects = sorted(subject_times.keys(), key=lambda x: subject_times[x])
    
    # Split subjects temporally
    n_subjects = len(sorted_subjects)
    n_train = int(n_subjects * ratios[0])
    
    train_subjects = sorted_subjects[:n_train]
    test_subjects = sorted_subjects[n_train:]
    
    # Create splits
    splits = {
        'train': [],
        'test': []
    }
    
    for subj in train_subjects:
        splits['train'].extend(subject_cases[subj])
    for subj in test_subjects:
        splits['test'].extend(subject_cases[subj])
    
    # Verify no leakage
    verify_no_subject_leakage(splits)
    
    return splits


def create_leave_one_subject_out_splits(cases: List[Dict]) -> List[Dict[str, List[Dict]]]:
    """
    Create leave-one-subject-out cross-validation splits.
    
    Args:
        cases: List of case dictionaries
        
    Returns:
        List of splits, each with one subject as test
    """
    # Group by subject
    subject_cases = defaultdict(list)
    for case in cases:
        subject_id = case.get('subject_id', case.get('case_id', ''))
        subject_cases[subject_id].append(case)
    
    subjects = list(subject_cases.keys())
    splits = []
    
    for test_subject in subjects:
        split = {
            'train': [],
            'test': subject_cases[test_subject]
        }
        
        for train_subject in subjects:
            if train_subject != test_subject:
                split['train'].extend(subject_cases[train_subject])
        
        splits.append(split)
    
    return splits
