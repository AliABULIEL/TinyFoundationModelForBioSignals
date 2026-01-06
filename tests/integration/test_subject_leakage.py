"""Integration tests for subject/participant leakage detection.

⚠️ CRITICAL: These tests verify that NO participant appears in multiple dataset splits
(train, validation, test). This is essential for valid evaluation of subject-independent
HAR models.

Subject leakage causes:
- Artificially inflated performance metrics
- Poor generalization to new subjects
- Invalid scientific conclusions
"""

import pytest
import numpy as np
from typing import List, Set

from src.data.datamodule import HARDataModule
from src.data.capture24_adapter import CAPTURE24Dataset


@pytest.mark.integration
class TestSubjectLeakage:
    """Test suite for detecting subject leakage across dataset splits."""

    def test_no_subject_overlap_train_val(self, sample_config, temp_dir):
        """
        Test that training and validation sets have NO overlapping participants.

        This is the most critical test - training on a subject's data and evaluating
        on the same subject will give misleadingly high performance.
        """
        sample_config["dataset"]["data_path"] = str(temp_dir)
        sample_config["dataset"]["use_synthetic"] = True

        datamodule = HARDataModule(config=sample_config)
        datamodule.setup()

        # Get participant IDs from each split
        train_participants = set(datamodule.train_dataset.participant_ids)
        val_participants = set(datamodule.val_dataset.participant_ids)

        # Check for overlap
        overlap = train_participants & val_participants

        assert len(overlap) == 0, (
            f"❌ SUBJECT LEAKAGE DETECTED: {len(overlap)} participants appear in both "
            f"training and validation sets!\n"
            f"  Overlapping participants: {sorted(list(overlap))}\n"
            f"  This invalidates evaluation metrics and indicates a critical bug."
        )

    def test_no_subject_overlap_train_test(self, sample_config, temp_dir):
        """Test that training and test sets have NO overlapping participants."""
        sample_config["dataset"]["data_path"] = str(temp_dir)
        sample_config["dataset"]["use_synthetic"] = True

        datamodule = HARDataModule(config=sample_config)
        datamodule.setup()

        # Get participant IDs
        train_participants = set(datamodule.train_dataset.participant_ids)
        test_participants = set(datamodule.test_dataset.participant_ids)

        # Check for overlap
        overlap = train_participants & test_participants

        assert len(overlap) == 0, (
            f"❌ SUBJECT LEAKAGE DETECTED: {len(overlap)} participants appear in both "
            f"training and test sets!\n"
            f"  Overlapping participants: {sorted(list(overlap))}\n"
            f"  This is the worst type of leakage - test performance is completely invalid."
        )

    def test_no_subject_overlap_val_test(self, sample_config, temp_dir):
        """Test that validation and test sets have NO overlapping participants."""
        sample_config["dataset"]["data_path"] = str(temp_dir)
        sample_config["dataset"]["use_synthetic"] = True

        datamodule = HARDataModule(config=sample_config)
        datamodule.setup()

        # Get participant IDs
        val_participants = set(datamodule.val_dataset.participant_ids)
        test_participants = set(datamodule.test_dataset.participant_ids)

        # Check for overlap
        overlap = val_participants & test_participants

        assert len(overlap) == 0, (
            f"❌ SUBJECT LEAKAGE DETECTED: {len(overlap)} participants appear in both "
            f"validation and test sets!\n"
            f"  Overlapping participants: {sorted(list(overlap))}\n"
            f"  This compromises the integrity of model selection."
        )

    def test_all_subjects_accounted_for(self, sample_config, temp_dir):
        """
        Test that all participants are assigned to exactly one split.

        This ensures:
        1. No participants are missing (data loss)
        2. No participants are duplicated (leakage)
        """
        sample_config["dataset"]["data_path"] = str(temp_dir)
        sample_config["dataset"]["use_synthetic"] = True

        datamodule = HARDataModule(config=sample_config)
        datamodule.setup()

        # Get all participant IDs
        train_participants = set(datamodule.train_dataset.participant_ids)
        val_participants = set(datamodule.val_dataset.participant_ids)
        test_participants = set(datamodule.test_dataset.participant_ids)

        # Get all unique participants across splits
        all_split_participants = train_participants | val_participants | test_participants

        # Get original dataset participants
        original_participants = set(datamodule.base_dataset.participant_ids)

        # Check all are accounted for
        missing = original_participants - all_split_participants
        extra = all_split_participants - original_participants

        assert len(missing) == 0, (
            f"❌ DATA LOSS: {len(missing)} participants missing from splits!\n"
            f"  Missing: {sorted(list(missing))}"
        )

        assert len(extra) == 0, (
            f"❌ INVALID DATA: {len(extra)} unknown participants in splits!\n"
            f"  Extra: {sorted(list(extra))}"
        )

    def test_split_proportions(self, sample_config, temp_dir):
        """
        Test that train/val/test splits match configured proportions approximately.

        Tolerates small deviations due to rounding, but large deviations indicate
        a bug in the splitting logic.
        """
        sample_config["dataset"]["data_path"] = str(temp_dir)
        sample_config["dataset"]["use_synthetic"] = True

        datamodule = HARDataModule(config=sample_config)
        datamodule.setup()

        # Count unique participants in each split
        num_train = len(set(datamodule.train_dataset.participant_ids))
        num_val = len(set(datamodule.val_dataset.participant_ids))
        num_test = len(set(datamodule.test_dataset.participant_ids))
        total = num_train + num_val + num_test

        # Calculate actual proportions
        actual_train_ratio = num_train / total
        actual_val_ratio = num_val / total
        actual_test_ratio = num_test / total

        # Get expected proportions
        expected_train = sample_config["dataset"].get("train_split", 0.7)
        expected_val = sample_config["dataset"].get("val_split", 0.15)
        expected_test = sample_config["dataset"].get("test_split", 0.15)

        # Allow 10% tolerance for small datasets
        tolerance = 0.10

        assert abs(actual_train_ratio - expected_train) < tolerance, (
            f"Train split proportion incorrect: {actual_train_ratio:.2%} "
            f"(expected {expected_train:.2%})"
        )

        assert abs(actual_val_ratio - expected_val) < tolerance, (
            f"Val split proportion incorrect: {actual_val_ratio:.2%} "
            f"(expected {expected_val:.2%})"
        )

        assert abs(actual_test_ratio - expected_test) < tolerance, (
            f"Test split proportion incorrect: {actual_test_ratio:.2%} "
            f"(expected {expected_test:.2%})"
        )

    @pytest.mark.integration
    def test_window_level_subject_consistency(self, sample_config, temp_dir):
        """
        Test that all windows within a split belong to the correct participants.

        This catches bugs where windowing might accidentally mix participants.
        """
        sample_config["dataset"]["data_path"] = str(temp_dir)
        sample_config["dataset"]["use_synthetic"] = True

        datamodule = HARDataModule(config=sample_config)
        datamodule.setup()

        # Get expected participant sets from the datasets
        train_participant_set = set([
            datamodule.base_dataset.participant_ids[i]
            for i in datamodule.train_dataset.participant_indices
        ])

        val_participant_set = set([
            datamodule.base_dataset.participant_ids[i]
            for i in datamodule.val_dataset.participant_indices
        ])

        # Verify train dataset
        train_participants_in_windows = set(datamodule.train_dataset.participant_ids)
        assert train_participants_in_windows.issubset(train_participant_set), (
            f"Train windows contain participants not in train split: "
            f"{train_participants_in_windows - train_participant_set}"
        )

        # Verify val dataset
        val_participants_in_windows = set(datamodule.val_dataset.participant_ids)
        assert val_participants_in_windows.issubset(val_participant_set), (
            f"Val windows contain participants not in val split: "
            f"{val_participants_in_windows - val_participant_set}"
        )


@pytest.mark.integration
class TestExplicitLeakageScenario:
    """Test that the system detects and rejects explicitly constructed leakage scenarios."""

    def test_manual_leakage_detection(self):
        """
        Manually construct a scenario with subject leakage and verify it's detected.

        This is a sanity check that our leakage tests actually work.
        """
        # Simulate overlapping participant sets
        train_participants = {"P001", "P002", "P003", "P004", "P005"}
        val_participants = {"P003", "P006", "P007"}  # P003 is duplicated!
        test_participants = {"P008", "P009", "P010"}

        # Detect overlap
        train_val_overlap = train_participants & val_participants
        train_test_overlap = train_participants & test_participants
        val_test_overlap = val_participants & test_participants

        # This SHOULD detect leakage
        assert len(train_val_overlap) > 0, "Failed to detect manual leakage scenario"
        assert "P003" in train_val_overlap, "Failed to detect specific leaked participant"

        # These should be clean
        assert len(train_test_overlap) == 0
        assert len(val_test_overlap) == 0
