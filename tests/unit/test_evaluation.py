"""Unit tests for evaluation module."""

import pytest
import numpy as np

from src.evaluation.metrics import (
    compute_metrics,
    accuracy,
    balanced_accuracy,
    macro_f1,
    weighted_f1,
    per_class_metrics,
    confusion_matrix,
)
from src.evaluation.splitters import (
    HoldoutSubjectSplitter,
    KFoldSubjectSplitter,
    LeaveOneSubjectOut,
)


@pytest.mark.unit
class TestMetrics:
    """Tests for evaluation metrics."""

    def test_accuracy(self):
        """Test accuracy computation."""
        y_true = np.array([0, 1, 2, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0])

        acc = accuracy(y_true, y_pred)
        assert acc == 0.8  # 4/5 correct

    def test_balanced_accuracy(self):
        """Test balanced accuracy with imbalanced classes."""
        # Imbalanced: 3 class 0, 2 class 1
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 0])  # Miss one class 1

        bal_acc = balanced_accuracy(y_true, y_pred)
        # Class 0: 3/3 = 1.0, Class 1: 1/2 = 0.5
        # Balanced: (1.0 + 0.5) / 2 = 0.75
        assert bal_acc == 0.75

    def test_macro_f1(self):
        """Test macro F1 score."""
        y_true = np.array([0, 1, 2, 1, 0, 2])
        y_pred = np.array([0, 1, 1, 1, 0, 2])

        f1 = macro_f1(y_true, y_pred)
        assert 0 <= f1 <= 1

    def test_weighted_f1(self):
        """Test weighted F1 score."""
        y_true = np.array([0, 1, 2, 1, 0, 2])
        y_pred = np.array([0, 1, 1, 1, 0, 2])

        f1 = weighted_f1(y_true, y_pred)
        assert 0 <= f1 <= 1

    def test_compute_metrics(self):
        """Test comprehensive metrics computation."""
        y_true = np.array([0, 1, 2, 1, 0, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 1, 0, 2, 0, 2, 2])

        metrics = compute_metrics(y_true, y_pred)

        # Check all expected keys
        expected_keys = [
            "accuracy",
            "balanced_accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_precision",
            "weighted_recall",
            "weighted_f1",
        ]

        for key in expected_keys:
            assert key in metrics
            assert 0 <= metrics[key] <= 1

    def test_per_class_metrics(self):
        """Test per-class metrics computation."""
        y_true = np.array([0, 1, 2, 1, 0, 2])
        y_pred = np.array([0, 1, 1, 1, 0, 2])

        metrics = per_class_metrics(
            y_true, y_pred,
            target_names=["Class A", "Class B", "Class C"]
        )

        assert len(metrics["precision"]) == 3
        assert len(metrics["recall"]) == 3
        assert len(metrics["f1"]) == 3
        assert len(metrics["support"]) == 3
        assert "names" in metrics

    def test_confusion_matrix(self):
        """Test confusion matrix computation."""
        y_true = np.array([0, 1, 2, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0])

        cm = confusion_matrix(y_true, y_pred)

        # Should be 3x3 for 3 classes
        assert cm.shape == (3, 3)
        # Diagonal should have correct counts
        assert cm[0, 0] == 2  # Class 0: 2 correct
        assert cm[1, 1] == 2  # Class 1: 2 correct
        assert cm[2, 2] == 0  # Class 2: 0 correct

    def test_confusion_matrix_normalized(self):
        """Test normalized confusion matrix."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 0])

        cm_norm = confusion_matrix(y_true, y_pred, normalize="true")

        # Each row should sum to 1
        row_sums = cm_norm.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(2))


@pytest.mark.unit
class TestSplitters:
    """Tests for subject splitters."""

    def test_holdout_splitter_basic(self):
        """Test basic holdout splitting."""
        # 10 subjects, 100 samples each
        subject_ids = np.repeat(np.arange(10), 100)

        splitter = HoldoutSubjectSplitter(
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42,
        )

        for train_idx, val_idx, test_idx in splitter.split(subject_ids):
            # Check no overlap
            assert len(set(train_idx) & set(val_idx)) == 0
            assert len(set(train_idx) & set(test_idx)) == 0
            assert len(set(val_idx) & set(test_idx)) == 0

            # Check subject counts
            train_subjects = set(subject_ids[train_idx])
            val_subjects = set(subject_ids[val_idx])
            test_subjects = set(subject_ids[test_idx])

            assert len(train_subjects & val_subjects) == 0
            assert len(train_subjects & test_subjects) == 0
            assert len(val_subjects & test_subjects) == 0

    def test_holdout_splitter_ratios(self):
        """Test that holdout splitter respects ratios."""
        subject_ids = np.repeat(np.arange(100), 10)  # 100 subjects

        splitter = HoldoutSubjectSplitter(
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            random_state=42,
        )

        for train_idx, val_idx, test_idx in splitter.split(subject_ids):
            train_subjects = len(np.unique(subject_ids[train_idx]))
            val_subjects = len(np.unique(subject_ids[val_idx]))
            test_subjects = len(np.unique(subject_ids[test_idx]))

            # Should be approximately 80/10/10
            assert 75 <= train_subjects <= 85
            assert 5 <= val_subjects <= 15
            assert 5 <= test_subjects <= 15

    def test_kfold_splitter_basic(self):
        """Test basic K-fold splitting."""
        subject_ids = np.repeat(np.arange(10), 100)

        splitter = KFoldSubjectSplitter(n_splits=5, random_state=42)

        splits = list(splitter.split(subject_ids))

        # Should have 5 splits
        assert len(splits) == 5

        # Each subject should appear in test set exactly once
        all_test_subjects = []
        for _, _, test_idx in splits:
            test_subjects = np.unique(subject_ids[test_idx])
            all_test_subjects.extend(test_subjects)

        # Each subject appears exactly once
        assert len(all_test_subjects) == 10
        assert len(set(all_test_subjects)) == 10

    def test_kfold_splitter_no_leakage(self):
        """Test that K-fold has no leakage in any fold."""
        subject_ids = np.repeat(np.arange(20), 50)

        splitter = KFoldSubjectSplitter(n_splits=5, random_state=42)

        for train_idx, val_idx, test_idx in splitter.split(subject_ids):
            train_subjects = set(subject_ids[train_idx])
            val_subjects = set(subject_ids[val_idx])
            test_subjects = set(subject_ids[test_idx])

            # No subject in multiple splits
            assert len(train_subjects & val_subjects) == 0
            assert len(train_subjects & test_subjects) == 0
            assert len(val_subjects & test_subjects) == 0

    def test_loso_splitter(self):
        """Test Leave-One-Subject-Out splitting."""
        n_subjects = 10
        subject_ids = np.repeat(np.arange(n_subjects), 100)

        splitter = LeaveOneSubjectOut(random_state=42)

        splits = list(splitter.split(subject_ids))

        # Should have n_subjects splits
        assert len(splits) == n_subjects

        # Check each split
        for i, (train_idx, val_idx, test_idx) in enumerate(splits):
            # Test set should have exactly 1 subject
            test_subjects = np.unique(subject_ids[test_idx])
            assert len(test_subjects) == 1

            # Train + val should have n-1 subjects
            train_val_subjects = set(np.unique(subject_ids[train_idx])) | set(np.unique(subject_ids[val_idx]))
            assert len(train_val_subjects) == n_subjects - 1

    def test_splitter_reproducibility(self):
        """Test that splitters are reproducible with same seed."""
        subject_ids = np.repeat(np.arange(20), 50)

        splitter1 = HoldoutSubjectSplitter(random_state=42)
        splitter2 = HoldoutSubjectSplitter(random_state=42)

        for (train1, val1, test1), (train2, val2, test2) in zip(
            splitter1.split(subject_ids),
            splitter2.split(subject_ids)
        ):
            np.testing.assert_array_equal(train1, train2)
            np.testing.assert_array_equal(val1, val2)
            np.testing.assert_array_equal(test1, test2)

    def test_splitter_invalid_ratios(self):
        """Test that invalid ratios raise error."""
        with pytest.raises(ValueError):
            # Ratios don't sum to 1
            HoldoutSubjectSplitter(train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)

    def test_kfold_invalid_n_splits(self):
        """Test that invalid n_splits raises error."""
        with pytest.raises(ValueError):
            KFoldSubjectSplitter(n_splits=1)  # Must be >= 2
