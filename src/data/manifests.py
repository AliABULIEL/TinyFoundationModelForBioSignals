"""Manifest generation for subject-level train/val/test splits.

Creates CSV manifests that map preprocessed window files to splits while
ensuring no subject appears in multiple splits (prevents data leakage).
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import hashlib
import csv
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import numpy as np


def extract_subject_id(window_filename: str) -> str:
    """Extract subject/case ID from window filename.
    
    Expected format: case_0001_win_0000.npz
    
    Args:
        window_filename: Window file name
    
    Returns:
        subject_id: Subject identifier (e.g., '0001')
    
    Example:
        >>> extract_subject_id('case_0001_win_0000.npz')
        '0001'
        >>> extract_subject_id('case_0042_win_0123.npz')
        '0042'
    """
    # Remove extension
    name = window_filename.replace('.npz', '')
    
    # Split by underscore
    parts = name.split('_')
    
    # Expected format: ['case', subject_id, 'win', window_id]
    if len(parts) >= 2:
        return parts[1]  # Return subject ID
    else:
        # Fallback: use hash of filename
        return hashlib.md5(name.encode()).hexdigest()[:8]


def hash_subject_to_split(
    subject_id: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> str:
    """Deterministically assign subject to split using hash.
    
    This ensures:
    - Same subject always goes to same split
    - No subject appears in multiple splits
    - Deterministic and reproducible
    
    Args:
        subject_id: Subject identifier
        train_ratio: Fraction for training (default: 0.8)
        val_ratio: Fraction for validation (default: 0.1)
        test_ratio: Fraction for test (default: 0.1)
        seed: Random seed for reproducibility
    
    Returns:
        split_name: 'train', 'val', or 'test'
    
    Example:
        >>> hash_subject_to_split('0001', train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
        'train'
    """
    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {total}")
    
    # Hash subject ID with seed
    hash_input = f"{seed}_{subject_id}"
    hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
    
    # Map hash to [0, 1)
    normalized = (hash_value % 1000000) / 1000000.0
    
    # Assign to split
    if normalized < train_ratio:
        return 'train'
    elif normalized < train_ratio + val_ratio:
        return 'val'
    else:
        return 'test'


def build_manifest(
    root: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    output_dir: Optional[str] = None
) -> Dict[str, str]:
    """Build train/val/test manifests from preprocessed window files.
    
    Scans the root directory for .npz window files and assigns each to a split
    based on subject-level hashing (no subject leakage).
    
    Expected directory structure:
        root/
            case_0001_win_0000.npz
            case_0001_win_0001.npz
            case_0002_win_0000.npz
            ...
    
    Or:
        root/
            train/  # If already split
            val/
            test/
    
    Args:
        root: Root directory containing window files
        train_ratio: Training split ratio (default: 0.8)
        val_ratio: Validation split ratio (default: 0.1)
        test_ratio: Test split ratio (default: 0.1)
        seed: Random seed for reproducibility
        output_dir: Where to write manifests (default: root)
    
    Returns:
        manifest_paths: Dict with paths to {train,val,test}.csv
    
    Example:
        >>> manifest_paths = build_manifest(
        ...     root='data/vitaldb_windows',
        ...     train_ratio=0.8,
        ...     val_ratio=0.1,
        ...     test_ratio=0.1
        ... )
        >>> # Creates:
        >>> # data/vitaldb_windows/train.csv
        >>> # data/vitaldb_windows/val.csv
        >>> # data/vitaldb_windows/test.csv
    """
    root = Path(root)
    if output_dir is None:
        output_dir = root
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Building manifests from: {root}")
    print(f"Ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    
    # Find all window files
    window_files = list(root.glob('**/*.npz'))
    
    if len(window_files) == 0:
        raise ValueError(f"No .npz files found in {root}")
    
    print(f"Found {len(window_files)} window files")
    
    # Group windows by subject
    subject_windows = defaultdict(list)
    for window_file in window_files:
        subject_id = extract_subject_id(window_file.name)
        subject_windows[subject_id].append(window_file)
    
    print(f"Found {len(subject_windows)} unique subjects")
    
    # Assign subjects to splits
    split_files = defaultdict(list)
    split_subjects = defaultdict(set)
    
    for subject_id, windows in subject_windows.items():
        split = hash_subject_to_split(
            subject_id,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed
        )
        
        split_files[split].extend(windows)
        split_subjects[split].add(subject_id)
    
    # Verify no subject leakage
    all_splits = list(split_subjects.keys())
    for i in range(len(all_splits)):
        for j in range(i + 1, len(all_splits)):
            overlap = split_subjects[all_splits[i]] & split_subjects[all_splits[j]]
            if overlap:
                raise ValueError(
                    f"Subject leakage detected between {all_splits[i]} and {all_splits[j]}: "
                    f"{overlap}"
                )
    
    print("\nSubject-level split statistics:")
    for split in ['train', 'val', 'test']:
        n_subjects = len(split_subjects[split])
        n_windows = len(split_files[split])
        print(f"  {split:5s}: {n_subjects:4d} subjects, {n_windows:6d} windows")
    
    # Write CSV manifests
    manifest_paths = {}
    
    for split in ['train', 'val', 'test']:
        manifest_path = output_dir / f"{split}.csv"
        
        with open(manifest_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filepath', 'subject_id', 'split'])
            
            for window_file in sorted(split_files[split]):
                subject_id = extract_subject_id(window_file.name)
                # Use relative path if possible
                try:
                    rel_path = window_file.relative_to(root)
                except ValueError:
                    rel_path = window_file
                
                writer.writerow([str(rel_path), subject_id, split])
        
        manifest_paths[split] = str(manifest_path)
        print(f"  Wrote {manifest_path}")
    
    print("\n✓ Manifests created successfully!")
    return manifest_paths


def verify_manifest_integrity(manifest_paths: Dict[str, str]) -> bool:
    """Verify manifest integrity and no subject leakage.
    
    Args:
        manifest_paths: Dict with paths to manifests
    
    Returns:
        True if all checks pass
    
    Raises:
        ValueError if integrity issues detected
    """
    print("Verifying manifest integrity...")
    
    split_subjects = {}
    
    for split, manifest_path in manifest_paths.items():
        subjects = set()
        
        with open(manifest_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                subjects.add(row['subject_id'])
        
        split_subjects[split] = subjects
        print(f"  {split}: {len(subjects)} unique subjects")
    
    # Check for subject leakage
    splits = list(split_subjects.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            overlap = split_subjects[splits[i]] & split_subjects[splits[j]]
            if overlap:
                raise ValueError(
                    f"Subject leakage detected between {splits[i]} and {splits[j]}: "
                    f"{overlap}"
                )
    
    print("✓ No subject leakage detected")
    return True


if __name__ == "__main__":
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build VitalDB SSL manifests')
    parser.add_argument('--root', type=str, required=True,
                       help='Root directory with window files')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Training split ratio')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='Validation split ratio')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                       help='Test split ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for manifests')
    
    args = parser.parse_args()
    
    # Build manifests
    manifest_paths = build_manifest(
        root=args.root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        output_dir=args.output_dir
    )
    
    # Verify integrity
    verify_manifest_integrity(manifest_paths)
