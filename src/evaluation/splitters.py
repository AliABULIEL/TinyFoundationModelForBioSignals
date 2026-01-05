"""Subject-independent data splitting strategies for HAR evaluation.

This module ensures ZERO DATA LEAKAGE by maintaining strict subject independence
across train/val/test splits. No subject should appear in multiple splits.
"""

import logging
from typing import Dict, List, Tuple, Iterator, Optional
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class SubjectSplitter(ABC):
    """
    Abstract base class for subject-independent splitting strategies.

    All splitters must ensure zero data leakage - no subject appears
    in multiple splits.
    """

    @abstractmethod
    def split(
        self,
        subject_ids: np.ndarray,
        stratify_by: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Generate train/val/test splits.

        Args:
            subject_ids: Array of subject IDs (N,)
            stratify_by: Optional array for stratified splitting (N,)

        Yields:
            Tuples of (train_indices, val_indices, test_indices)

        Note:
            All implementations MUST verify zero leakage.
        """
        pass

    def _verify_no_leakage(
        self,
        train_subjects: np.ndarray,
        val_subjects: np.ndarray,
        test_subjects: np.ndarray,
    ) -> None:
        """
        Verify no subject appears in multiple splits.

        Args:
            train_subjects: Training subject IDs
            val_subjects: Validation subject IDs
            test_subjects: Test subject IDs

        Raises:
            AssertionError: If any subject appears in multiple splits
        """
        train_set = set(train_subjects)
        val_set = set(val_subjects)
        test_set = set(test_subjects)

        # Check for overlaps
        train_val_overlap = train_set & val_set
        train_test_overlap = train_set & test_set
        val_test_overlap = val_set & test_set

        if train_val_overlap:
            raise AssertionError(
                f"LEAKAGE DETECTED: {len(train_val_overlap)} subjects in both "
                f"train and val: {sorted(train_val_overlap)[:5]}"
            )

        if train_test_overlap:
            raise AssertionError(
                f"LEAKAGE DETECTED: {len(train_test_overlap)} subjects in both "
                f"train and test: {sorted(train_test_overlap)[:5]}"
            )

        if val_test_overlap:
            raise AssertionError(
                f"LEAKAGE DETECTED: {len(val_test_overlap)} subjects in both "
                f"val and test: {sorted(val_test_overlap)[:5]}"
            )

        logger.debug(
            f"No leakage: {len(train_subjects)} train, "
            f"{len(val_subjects)} val, {len(test_subjects)} test subjects"
        )


class HoldoutSubjectSplitter(SubjectSplitter):
    """
    Simple holdout split by subjects.

    Splits subjects into train/val/test with specified ratios.
    Ensures no subject appears in multiple splits.

    Args:
        train_ratio: Fraction of subjects for training (default: 0.7)
        val_ratio: Fraction of subjects for validation (default: 0.15)
        test_ratio: Fraction of subjects for testing (default: 0.15)
        shuffle: Whether to shuffle subjects before splitting
        random_state: Random seed for reproducibility

    Example:
        >>> splitter = HoldoutSubjectSplitter(
        ...     train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42
        ... )
        >>> subject_ids = np.array([1, 1, 1, 2, 2, 3, 3, 4, 4])  # 4 subjects
        >>> for train_idx, val_idx, test_idx in splitter.split(subject_ids):
        ...     print(f"Train subjects: {len(np.unique(subject_ids[train_idx]))}")
        Train subjects: 2
    """

    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        shuffle: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize holdout splitter."""
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(
                f"Ratios must sum to 1.0, got {total_ratio:.3f} "
                f"({train_ratio} + {val_ratio} + {test_ratio})"
            )

        if train_ratio <= 0 or val_ratio < 0 or test_ratio < 0:
            raise ValueError("All ratios must be non-negative, train_ratio must be > 0")

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.shuffle = shuffle
        self.random_state = random_state

        logger.info(
            f"Initialized HoldoutSubjectSplitter: "
            f"train={train_ratio:.2f}, val={val_ratio:.2f}, test={test_ratio:.2f}"
        )

    def split(
        self,
        subject_ids: np.ndarray,
        stratify_by: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Generate single train/val/test split.

        Args:
            subject_ids: Array of subject IDs (N,)
            stratify_by: Not used for holdout split

        Yields:
            Single tuple of (train_indices, val_indices, test_indices)
        """
        subject_ids = np.asarray(subject_ids)

        # Get unique subjects
        unique_subjects = np.unique(subject_ids)
        n_subjects = len(unique_subjects)

        # Shuffle subjects if requested
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(unique_subjects)

        # Compute split sizes
        n_train = int(n_subjects * self.train_ratio)
        n_val = int(n_subjects * self.val_ratio)

        # Ensure we have at least 1 subject per split if ratio > 0
        if self.val_ratio > 0 and n_val == 0:
            n_val = 1
        if self.test_ratio > 0 and (n_subjects - n_train - n_val) == 0:
            n_train -= 1  # Give one to test

        # Split subjects
        train_subjects = unique_subjects[:n_train]
        val_subjects = unique_subjects[n_train : n_train + n_val]
        test_subjects = unique_subjects[n_train + n_val :]

        # Verify no leakage
        self._verify_no_leakage(train_subjects, val_subjects, test_subjects)

        # Convert subject IDs to sample indices
        train_indices = np.where(np.isin(subject_ids, train_subjects))[0]
        val_indices = np.where(np.isin(subject_ids, val_subjects))[0]
        test_indices = np.where(np.isin(subject_ids, test_subjects))[0]

        logger.info(
            f"Split {n_subjects} subjects into "
            f"{len(train_subjects)} train ({len(train_indices)} samples), "
            f"{len(val_subjects)} val ({len(val_indices)} samples), "
            f"{len(test_subjects)} test ({len(test_indices)} samples)"
        )

        yield train_indices, val_indices, test_indices


class KFoldSubjectSplitter(SubjectSplitter):
    """
    K-fold cross-validation with subject-level splitting.

    Splits subjects into K folds, ensuring no subject appears in multiple folds.
    Each fold becomes test set once, with remaining folds split into train/val.

    Args:
        n_splits: Number of folds (default: 5)
        val_ratio: Fraction of non-test subjects for validation (default: 0.15)
        shuffle: Whether to shuffle subjects before folding
        random_state: Random seed for reproducibility

    Example:
        >>> splitter = KFoldSubjectSplitter(n_splits=5, random_state=42)
        >>> subject_ids = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])  # 5 subjects
        >>> splits = list(splitter.split(subject_ids))
        >>> len(splits)
        5
    """

    def __init__(
        self,
        n_splits: int = 5,
        val_ratio: float = 0.15,
        shuffle: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize K-fold splitter."""
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")

        if not 0 <= val_ratio < 1:
            raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")

        self.n_splits = n_splits
        self.val_ratio = val_ratio
        self.shuffle = shuffle
        self.random_state = random_state

        logger.info(
            f"Initialized KFoldSubjectSplitter: "
            f"n_splits={n_splits}, val_ratio={val_ratio}"
        )

    def split(
        self,
        subject_ids: np.ndarray,
        stratify_by: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Generate K train/val/test splits.

        Args:
            subject_ids: Array of subject IDs (N,)
            stratify_by: Not used for K-fold

        Yields:
            K tuples of (train_indices, val_indices, test_indices)
        """
        subject_ids = np.asarray(subject_ids)

        # Get unique subjects
        unique_subjects = np.unique(subject_ids)
        n_subjects = len(unique_subjects)

        if n_subjects < self.n_splits:
            raise ValueError(
                f"Cannot split {n_subjects} subjects into {self.n_splits} folds"
            )

        # Shuffle subjects if requested
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(unique_subjects)

        # Create folds
        fold_sizes = np.full(self.n_splits, n_subjects // self.n_splits, dtype=int)
        fold_sizes[: n_subjects % self.n_splits] += 1

        current = 0
        folds = []
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            folds.append(unique_subjects[start:stop])
            current = stop

        # Generate splits
        for i in range(self.n_splits):
            # Test fold
            test_subjects = folds[i]

            # Train + val folds
            trainval_subjects = np.concatenate([folds[j] for j in range(self.n_splits) if j != i])

            # Split train/val
            n_val = int(len(trainval_subjects) * self.val_ratio)
            if n_val > 0:
                val_subjects = trainval_subjects[:n_val]
                train_subjects = trainval_subjects[n_val:]
            else:
                val_subjects = np.array([])
                train_subjects = trainval_subjects

            # Verify no leakage
            self._verify_no_leakage(train_subjects, val_subjects, test_subjects)

            # Convert to indices
            train_indices = np.where(np.isin(subject_ids, train_subjects))[0]
            val_indices = np.where(np.isin(subject_ids, val_subjects))[0]
            test_indices = np.where(np.isin(subject_ids, test_subjects))[0]

            logger.debug(
                f"Fold {i + 1}/{self.n_splits}: "
                f"{len(train_subjects)} train, "
                f"{len(val_subjects)} val, "
                f"{len(test_subjects)} test subjects"
            )

            yield train_indices, val_indices, test_indices


class LeaveOneSubjectOut(SubjectSplitter):
    """
    Leave-One-Subject-Out cross-validation.

    Each subject becomes test set once, with remaining subjects split
    into train/val. This is the most rigorous subject-independent evaluation.

    Args:
        val_ratio: Fraction of non-test subjects for validation (default: 0.15)
        random_state: Random seed for train/val split

    Example:
        >>> splitter = LeaveOneSubjectOut(val_ratio=0.2, random_state=42)
        >>> subject_ids = np.array([1, 1, 2, 2, 3, 3])  # 3 subjects
        >>> splits = list(splitter.split(subject_ids))
        >>> len(splits)
        3  # One split per subject
    """

    def __init__(
        self,
        val_ratio: float = 0.15,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize LOSO splitter."""
        if not 0 <= val_ratio < 1:
            raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")

        self.val_ratio = val_ratio
        self.random_state = random_state

        logger.info(
            f"Initialized LeaveOneSubjectOut: val_ratio={val_ratio}"
        )

    def split(
        self,
        subject_ids: np.ndarray,
        stratify_by: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Generate N train/val/test splits (N = number of subjects).

        Args:
            subject_ids: Array of subject IDs (N,)
            stratify_by: Not used for LOSO

        Yields:
            N tuples of (train_indices, val_indices, test_indices)
        """
        subject_ids = np.asarray(subject_ids)

        # Get unique subjects
        unique_subjects = np.unique(subject_ids)
        n_subjects = len(unique_subjects)

        if n_subjects < 2:
            raise ValueError(
                f"Need at least 2 subjects for LOSO, got {n_subjects}"
            )

        rng = np.random.RandomState(self.random_state)

        # Leave each subject out once
        for test_subject in unique_subjects:
            # Test subject
            test_subjects = np.array([test_subject])

            # Remaining subjects
            trainval_subjects = unique_subjects[unique_subjects != test_subject]

            # Split train/val
            n_val = int(len(trainval_subjects) * self.val_ratio)
            if n_val > 0:
                # Shuffle and split
                shuffled = trainval_subjects.copy()
                rng.shuffle(shuffled)
                val_subjects = shuffled[:n_val]
                train_subjects = shuffled[n_val:]
            else:
                val_subjects = np.array([])
                train_subjects = trainval_subjects

            # Verify no leakage
            self._verify_no_leakage(train_subjects, val_subjects, test_subjects)

            # Convert to indices
            train_indices = np.where(np.isin(subject_ids, train_subjects))[0]
            val_indices = np.where(np.isin(subject_ids, val_subjects))[0]
            test_indices = np.where(np.isin(subject_ids, test_subjects))[0]

            logger.debug(
                f"Test subject {test_subject}: "
                f"{len(train_subjects)} train, "
                f"{len(val_subjects)} val subjects"
            )

            yield train_indices, val_indices, test_indices


def get_splitter(
    splitter_type: str,
    **kwargs,
) -> SubjectSplitter:
    """
    Factory function to create subject splitter.

    Args:
        splitter_type: Type of splitter ("holdout", "kfold", "loso")
        **kwargs: Arguments passed to splitter constructor

    Returns:
        SubjectSplitter instance

    Raises:
        ValueError: If splitter type is unknown

    Example:
        >>> splitter = get_splitter("kfold", n_splits=5, random_state=42)
    """
    splitter_type = splitter_type.lower()

    if splitter_type == "holdout":
        return HoldoutSubjectSplitter(**kwargs)

    elif splitter_type == "kfold":
        return KFoldSubjectSplitter(**kwargs)

    elif splitter_type == "loso":
        return LeaveOneSubjectOut(**kwargs)

    else:
        raise ValueError(
            f"Unknown splitter type: {splitter_type}\\n"
            f"  Supported: ['holdout', 'kfold', 'loso']"
        )
