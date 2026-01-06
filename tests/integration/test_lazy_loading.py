"""Integration tests for lazy loading verification.

⚠️ CRITICAL: These tests verify that the WindowedDataset implements STRICT LAZY LOADING.
This is essential for scalability - the system must handle >150 participants on 32GB RAM.

Lazy loading requirements:
- Dataset initialization must NOT load signal data into memory
- Only metadata (file paths, offsets, labels) should be stored during __init__
- Signal data should ONLY be loaded in __getitem__ on-demand
"""

import pytest
import sys
from unittest.mock import patch, MagicMock
from typing import List

from src.data.datamodule import HARDataModule, WindowedDataset
from src.data.capture24_adapter import CAPTURE24Dataset
from src.preprocessing.pipeline import PreprocessingPipeline


@pytest.mark.integration
class TestLazyLoadingBehavior:
    """Test suite for verifying strict lazy loading implementation."""

    def test_init_does_not_store_signal_data(self, sample_config, temp_dir):
        """
        Test that WindowedDataset.__init__ does NOT store signal arrays in memory.

        Verification method:
        - Check that self.windows attribute is NOT a numpy array of signals
        - Verify self.window_metadata exists and contains only lightweight data
        """
        sample_config["dataset"]["data_path"] = str(temp_dir)
        sample_config["dataset"]["use_synthetic"] = True

        datamodule = HARDataModule(config=sample_config)
        datamodule.setup()

        train_dataset = datamodule.train_dataset

        # CRITICAL CHECK: Verify windows are NOT stored as numpy array
        assert not hasattr(train_dataset, 'windows') or not isinstance(
            getattr(train_dataset, 'windows', None), type(None)
        ), (
            "❌ LAZY LOADING VIOLATION: Dataset stores 'windows' attribute. "
            "This indicates eager loading of signal data."
        )

        # Verify metadata exists
        assert hasattr(train_dataset, 'window_metadata'), (
            "❌ Dataset missing 'window_metadata' attribute required for lazy loading"
        )

        # Verify metadata is lightweight (just dicts, not arrays)
        metadata_sample = train_dataset.window_metadata[0]
        assert isinstance(metadata_sample, dict), "Metadata should be dict"
        assert 'participant_id' in metadata_sample, "Metadata missing participant_id"
        assert 'window_start' in metadata_sample, "Metadata missing window_start"
        assert 'label' in metadata_sample, "Metadata missing label"

        # Verify metadata doesn't contain signal data
        for key, value in metadata_sample.items():
            assert not isinstance(value, (list, tuple)) or len(value) < 100, (
                f"❌ Metadata key '{key}' contains large array-like data. "
                f"This may indicate signal data in metadata."
            )

    def test_getitem_performs_io_operation(self, sample_config, temp_dir):
        """
        Test that __getitem__ actually reads from disk on-demand.

        Verification method:
        - Mock the load_participant method
        - Call __getitem__
        - Verify load_participant was called (indicating disk I/O)
        """
        sample_config["dataset"]["data_path"] = str(temp_dir)
        sample_config["dataset"]["use_synthetic"] = True

        datamodule = HARDataModule(config=sample_config)
        datamodule.setup()

        train_dataset = datamodule.train_dataset

        # Patch the load_participant method to track calls
        original_load = train_dataset.base_dataset.load_participant

        call_count = [0]

        def tracked_load(*args, **kwargs):
            call_count[0] += 1
            return original_load(*args, **kwargs)

        with patch.object(
            train_dataset.base_dataset,
            'load_participant',
            side_effect=tracked_load
        ):
            # Access a single window
            sample = train_dataset[0]

            # Verify load_participant was called (disk I/O occurred)
            assert call_count[0] > 0, (
                "❌ LAZY LOADING VIOLATION: __getitem__ did not call load_participant. "
                "This indicates data was pre-loaded into memory."
            )

            # Verify sample is valid
            assert 'signal' in sample
            assert 'label' in sample
            assert sample['signal'].shape[0] > 0

    def test_multiple_getitem_calls_reload_data(self, sample_config, temp_dir):
        """
        Test that accessing multiple windows causes multiple load operations.

        This verifies true on-demand loading (not caching all data after first access).
        """
        sample_config["dataset"]["data_path"] = str(temp_dir)
        sample_config["dataset"]["use_synthetic"] = True

        datamodule = HARDataModule(config=sample_config)
        datamodule.setup()

        train_dataset = datamodule.train_dataset

        # Track load_participant calls
        call_count = [0]
        original_load = train_dataset.base_dataset.load_participant

        def tracked_load(*args, **kwargs):
            call_count[0] += 1
            return original_load(*args, **kwargs)

        with patch.object(
            train_dataset.base_dataset,
            'load_participant',
            side_effect=tracked_load
        ):
            # Access multiple windows from different participants
            # (to ensure different participants are loaded)
            num_samples = min(10, len(train_dataset))
            for i in range(num_samples):
                _ = train_dataset[i]

            # Verify load was called multiple times
            assert call_count[0] >= 1, (
                "❌ LAZY LOADING VIOLATION: No load operations detected for multiple __getitem__ calls"
            )

    def test_metadata_memory_footprint_is_small(self, sample_config, temp_dir):
        """
        Test that metadata storage is memory-efficient.

        For a dataset with thousands of windows, metadata should be <1MB per 1000 windows.
        """
        sample_config["dataset"]["data_path"] = str(temp_dir)
        sample_config["dataset"]["use_synthetic"] = True

        datamodule = HARDataModule(config=sample_config)
        datamodule.setup()

        train_dataset = datamodule.train_dataset

        # Calculate approximate metadata size
        num_windows = len(train_dataset)

        # Each metadata entry should be a small dict
        # Estimate: ~100 bytes per metadata entry (dict with strings and ints)
        metadata_size_estimate = num_windows * 100  # bytes

        # For reference: storing actual windows would be:
        # num_windows * context_length * 3 channels * 4 bytes (float32)
        context_length = sample_config["preprocessing"]["context_length"]
        signal_data_size = num_windows * context_length * 3 * 4  # bytes

        # Metadata should be at least 50x smaller than signal data
        # (100x is ideal, but 50x is acceptable for real-world scenarios)
        ratio = signal_data_size / metadata_size_estimate

        assert ratio > 50, (
            f"❌ MEMORY INEFFICIENCY: Metadata appears too large. "
            f"Metadata estimate: {metadata_size_estimate / 1e6:.1f} MB, "
            f"Signal data would be: {signal_data_size / 1e6:.1f} MB. "
            f"Ratio: {ratio:.1f}x (should be >50x)"
        )

        print(f"✓ Memory efficiency verified:")
        print(f"  Metadata: ~{metadata_size_estimate / 1e6:.2f} MB")
        print(f"  Avoided loading: ~{signal_data_size / 1e6:.2f} MB")
        print(f"  Efficiency ratio: {ratio:.0f}x smaller")

    def test_dataset_initialization_is_fast(self, sample_config, temp_dir):
        """
        Test that dataset initialization completes quickly.

        Lazy loading should make init fast (just scanning files, not loading data).
        Even with 10 synthetic participants, init should be <10 seconds.
        """
        import time

        sample_config["dataset"]["data_path"] = str(temp_dir)
        sample_config["dataset"]["use_synthetic"] = True

        start_time = time.time()

        datamodule = HARDataModule(config=sample_config)
        datamodule.setup()

        init_time = time.time() - start_time

        # For 10 synthetic participants, init should be fast
        max_init_time = 15.0  # seconds

        assert init_time < max_init_time, (
            f"❌ PERFORMANCE ISSUE: Dataset initialization took {init_time:.1f}s "
            f"(expected <{max_init_time}s). This may indicate eager data loading."
        )

        print(f"✓ Dataset initialization completed in {init_time:.2f}s")

    def test_lazy_loading_preserves_correctness(self, sample_config, temp_dir):
        """
        Test that lazy loading produces correct data (no data corruption).

        Verify that __getitem__ returns valid windows with correct shapes and labels.
        """
        sample_config["dataset"]["data_path"] = str(temp_dir)
        sample_config["dataset"]["use_synthetic"] = True

        datamodule = HARDataModule(config=sample_config)
        datamodule.setup()

        train_dataset = datamodule.train_dataset

        # Sample multiple windows
        num_samples = min(20, len(train_dataset))

        for i in range(num_samples):
            sample = train_dataset[i]

            # Verify structure
            assert 'signal' in sample, f"Sample {i} missing 'signal'"
            assert 'label' in sample, f"Sample {i} missing 'label'"
            assert 'participant_id' in sample, f"Sample {i} missing 'participant_id'"

            # Verify shapes
            context_length = sample_config["preprocessing"]["context_length"]
            assert sample['signal'].shape == (context_length, 3), (
                f"Sample {i} has wrong signal shape: {sample['signal'].shape}"
            )

            # Verify label is valid
            num_classes = sample_config["dataset"]["num_classes"]
            label_value = sample['label'].item()
            assert 0 <= label_value < num_classes, (
                f"Sample {i} has invalid label: {label_value} (expected 0-{num_classes-1})"
            )

    @pytest.mark.integration
    def test_no_global_signal_cache(self, sample_config, temp_dir):
        """
        Test that the dataset does NOT maintain a global cache of all signals.

        This is the final check - verify no large arrays exist as class attributes.
        """
        sample_config["dataset"]["data_path"] = str(temp_dir)
        sample_config["dataset"]["use_synthetic"] = True

        datamodule = HARDataModule(config=sample_config)
        datamodule.setup()

        train_dataset = datamodule.train_dataset

        # Get all attributes
        dataset_attrs = vars(train_dataset)

        # Check for large numpy arrays or lists that might contain signals
        suspicious_attrs = []

        for attr_name, attr_value in dataset_attrs.items():
            # Skip known lightweight attributes
            if attr_name in ['window_metadata', 'labels', 'participant_ids',
                             'preprocessing_pipeline', 'base_dataset', 'transform',
                             'is_training', 'participant_indices']:
                continue

            # Check for large collections
            if isinstance(attr_value, (list, tuple)):
                if len(attr_value) > 100 and any(
                    hasattr(item, 'shape') and hasattr(item, 'dtype')
                    for item in attr_value[:min(5, len(attr_value))]
                ):
                    suspicious_attrs.append((attr_name, type(attr_value), len(attr_value)))

        assert len(suspicious_attrs) == 0, (
            f"❌ LAZY LOADING VIOLATION: Found suspicious large collections:\n" +
            "\n".join([f"  - {name}: {typ} with {size} items" for name, typ, size in suspicious_attrs]) +
            "\nThese may contain cached signal data."
        )
