"""
Comprehensive tests for label-window alignment module.

Tests cover:
- Window label creation and alignment
- Case-level to window-level propagation
- Temporal label extraction
- Batch alignment
- Array conversion for training
- Multi-task label organization
- Aggregation strategies

Author: Senior Data Engineering Team
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
import tempfile
import shutil
import numpy as np
import pandas as pd

from src.data.clinical_labels import (
    ClinicalLabelExtractor,
    LabelType
)
from src.data.label_alignment import (
    LabelWindowAligner,
    WindowLabel
)


class TestWindowLabel(unittest.TestCase):
    """Test WindowLabel dataclass."""
    
    def test_window_label_creation(self):
        """Test creating a window label."""
        window_label = WindowLabel(
            window_idx=0,
            case_id="1",
            start_time=0.0,
            end_time=10.0,
            labels={"mortality": 0, "age": 65}
        )
        
        self.assertEqual(window_label.window_idx, 0)
        self.assertEqual(window_label.case_id, "1")
        self.assertEqual(window_label.start_time, 0.0)
        self.assertEqual(window_label.end_time, 10.0)
        self.assertIn("mortality", window_label.labels)
    
    def test_window_label_to_dict(self):
        """Test converting window label to dictionary."""
        window_label = WindowLabel(
            window_idx=5,
            case_id="2",
            start_time=50.0,
            end_time=60.0,
            labels={"mortality": 1}
        )
        
        label_dict = window_label.to_dict()
        
        self.assertIsInstance(label_dict, dict)
        self.assertEqual(label_dict["window_idx"], 5)
        self.assertEqual(label_dict["case_id"], "2")


class TestLabelWindowAligner(unittest.TestCase):
    """Test LabelWindowAligner class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.metadata_path = Path(self.test_dir) / "test_metadata.csv"
        
        # Create sample metadata
        self.sample_data = pd.DataFrame({
            'caseid': ['1', '2', '3'],
            'death_inhosp': [0, 1, 0],
            'icu_days': [0, 3, 1],
            'age': [65, 72, 58],
            'sex': ['M', 'F', 'M'],
            'bmi': [25.0, 28.5, 22.0],
            'asa': [2, 3, 2]
        })
        
        self.sample_data.to_csv(self.metadata_path, index=False)
        
        # Create label extractor
        self.extractor = ClinicalLabelExtractor(
            metadata_path=self.metadata_path
        )
        
        # Create aligner
        self.aligner = LabelWindowAligner(
            label_extractor=self.extractor,
            window_duration=10.0
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test aligner initialization."""
        aligner = LabelWindowAligner(
            label_extractor=self.extractor,
            window_duration=10.0,
            propagation_strategy="replicate"
        )
        
        self.assertEqual(aligner.window_duration, 10.0)
        self.assertEqual(aligner.propagation_strategy, "replicate")
    
    def test_align_case_windows(self):
        """Test aligning labels for a case with multiple windows."""
        window_labels = self.aligner.align_case_windows(
            case_id="1",
            n_windows=5
        )
        
        self.assertEqual(len(window_labels), 5)
        self.assertIsInstance(window_labels[0], WindowLabel)
        
        # Check that all windows have the same case-level labels
        for window_label in window_labels:
            self.assertEqual(window_label.case_id, "1")
            self.assertIn("mortality", window_label.labels)
    
    def test_window_timing(self):
        """Test that window timing is correct."""
        window_labels = self.aligner.align_case_windows(
            case_id="1",
            n_windows=3,
            start_time=0.0
        )
        
        # Check window times
        self.assertEqual(window_labels[0].start_time, 0.0)
        self.assertEqual(window_labels[0].end_time, 10.0)
        
        self.assertEqual(window_labels[1].start_time, 10.0)
        self.assertEqual(window_labels[1].end_time, 20.0)
        
        self.assertEqual(window_labels[2].start_time, 20.0)
        self.assertEqual(window_labels[2].end_time, 30.0)
    
    def test_align_nonexistent_case(self):
        """Test aligning labels for a non-existent case."""
        window_labels = self.aligner.align_case_windows(
            case_id="999",
            n_windows=5
        )
        
        self.assertEqual(len(window_labels), 0)
    
    def test_specific_label_extraction(self):
        """Test extracting only specific labels."""
        window_labels = self.aligner.align_case_windows(
            case_id="2",
            n_windows=2,
            label_names=["mortality", "age"]
        )
        
        self.assertEqual(len(window_labels), 2)
        self.assertIn("mortality", window_labels[0].labels)
        self.assertIn("age", window_labels[0].labels)
    
    def test_temporal_label_extraction(self):
        """Test extracting temporal labels within windows."""
        # Create temporal labels (e.g., continuous blood pressure)
        temporal_data = pd.DataFrame({
            'timestamp': [0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
            'bp_systolic': [120, 125, 118, 122, 119, 121],
            'bp_diastolic': [80, 82, 78, 81, 79, 80]
        })
        
        window_labels = self.aligner.align_case_windows(
            case_id="1",
            n_windows=3,
            temporal_labels=temporal_data
        )
        
        # First window should have temporal values from 0-10s
        self.assertIsNotNone(window_labels[0].temporal_values)
        if window_labels[0].temporal_values:
            self.assertIn("bp_systolic_mean", window_labels[0].temporal_values)
    
    def test_propagation_replicate_strategy(self):
        """Test replication propagation strategy."""
        aligner = LabelWindowAligner(
            label_extractor=self.extractor,
            propagation_strategy="replicate"
        )
        
        window_labels = aligner.align_case_windows(
            case_id="1",
            n_windows=3
        )
        
        # All windows should have identical case-level labels
        first_mortality = window_labels[0].labels.get("mortality")
        for window_label in window_labels[1:]:
            self.assertEqual(window_label.labels.get("mortality"), first_mortality)
    
    def test_batch_alignment(self):
        """Test batch alignment for multiple cases."""
        case_window_mapping = {
            "1": 5,
            "2": 3,
            "3": 4
        }
        
        batch_results = self.aligner.align_batch_windows(case_window_mapping)
        
        self.assertEqual(len(batch_results), 3)
        self.assertEqual(len(batch_results["1"]), 5)
        self.assertEqual(len(batch_results["2"]), 3)
        self.assertEqual(len(batch_results["3"]), 4)
    
    def test_create_window_label_arrays(self):
        """Test converting window labels to arrays."""
        window_labels = self.aligner.align_case_windows(
            case_id="1",
            n_windows=5
        )
        
        label_names = ["mortality", "age"]
        label_array, mask_array = self.aligner.create_window_label_arrays(
            window_labels,
            label_names,
            return_masks=True
        )
        
        # Check shapes
        self.assertEqual(label_array.shape, (5, 2))
        self.assertEqual(mask_array.shape, (5, 2))
        
        # Check that mask is mostly True
        self.assertTrue(np.any(mask_array))
        
        # Check label values are reasonable
        self.assertTrue(np.all(label_array[mask_array] >= 0))
    
    def test_create_arrays_with_missing_labels(self):
        """Test array creation with some missing labels."""
        # Create window labels with intentionally missing labels
        window_labels = [
            WindowLabel(0, "1", 0.0, 10.0, {"mortality": 0, "age": 65}),
            WindowLabel(1, "1", 10.0, 20.0, {"mortality": 0}),  # Missing age
            WindowLabel(2, "1", 20.0, 30.0, {"age": 65})  # Missing mortality
        ]
        
        label_names = ["mortality", "age"]
        label_array, mask_array = self.aligner.create_window_label_arrays(
            window_labels,
            label_names,
            return_masks=True
        )
        
        # Check that mask correctly identifies missing values
        self.assertTrue(mask_array[0, 0])  # mortality present
        self.assertTrue(mask_array[0, 1])  # age present
        self.assertTrue(mask_array[1, 0])  # mortality present
        self.assertFalse(mask_array[1, 1])  # age missing
        self.assertFalse(mask_array[2, 0])  # mortality missing
        self.assertTrue(mask_array[2, 1])  # age present
    
    def test_aggregate_window_predictions_mean(self):
        """Test aggregating window predictions to case level using mean."""
        # Simulate predictions
        predictions = np.array([
            [0.1],  # case 1, window 0
            [0.2],  # case 1, window 1
            [0.8],  # case 2, window 0
            [0.9]   # case 2, window 1
        ])
        
        case_ids = ["1", "1", "2", "2"]
        
        case_preds = self.aligner.aggregate_window_predictions(
            predictions,
            case_ids,
            aggregation="mean"
        )
        
        self.assertIn("1", case_preds)
        self.assertIn("2", case_preds)
        
        # Check aggregated values
        self.assertAlmostEqual(case_preds["1"], 0.15, places=5)
        self.assertAlmostEqual(case_preds["2"], 0.85, places=5)
    
    def test_aggregate_window_predictions_max(self):
        """Test aggregating using max strategy."""
        predictions = np.array([[0.1], [0.8], [0.3], [0.2]])
        case_ids = ["1", "1", "2", "2"]
        
        case_preds = self.aligner.aggregate_window_predictions(
            predictions,
            case_ids,
            aggregation="max"
        )
        
        self.assertAlmostEqual(case_preds["1"], 0.8, places=5)
        self.assertAlmostEqual(case_preds["2"], 0.3, places=5)
    
    def test_aggregate_window_predictions_last(self):
        """Test aggregating using last window strategy."""
        predictions = np.array([[0.1], [0.8], [0.3], [0.9]])
        case_ids = ["1", "1", "2", "2"]
        
        case_preds = self.aligner.aggregate_window_predictions(
            predictions,
            case_ids,
            aggregation="last"
        )
        
        self.assertAlmostEqual(case_preds["1"], 0.8, places=5)
        self.assertAlmostEqual(case_preds["2"], 0.9, places=5)
    
    def test_create_multi_task_labels(self):
        """Test creating labels for multi-task learning."""
        window_labels = self.aligner.align_case_windows(
            case_id="1",
            n_windows=5
        )
        
        task_configs = {
            "mortality": ["mortality"],
            "demographics": ["age", "bmi"],
            "severity": ["asa", "icu_los"]
        }
        
        task_arrays = self.aligner.create_multi_task_labels(
            window_labels,
            task_configs
        )
        
        self.assertEqual(len(task_arrays), 3)
        self.assertIn("mortality", task_arrays)
        self.assertIn("demographics", task_arrays)
        self.assertIn("severity", task_arrays)
        
        # Check shapes
        self.assertEqual(task_arrays["mortality"].shape, (5, 1))
        self.assertEqual(task_arrays["demographics"].shape, (5, 2))
        self.assertEqual(task_arrays["severity"].shape, (5, 2))
    
    def test_save_and_load_npz(self):
        """Test saving and loading window labels in NPZ format."""
        window_labels = self.aligner.align_case_windows(
            case_id="1",
            n_windows=3
        )
        
        output_path = Path(self.test_dir) / "window_labels.npz"
        
        # Save
        self.aligner.save_aligned_labels(
            window_labels,
            output_path,
            format="npz"
        )
        
        self.assertTrue(output_path.exists())
        
        # Load
        loaded_labels = self.aligner.load_aligned_labels(
            output_path,
            format="npz"
        )
        
        self.assertEqual(len(loaded_labels), len(window_labels))
        self.assertEqual(loaded_labels[0].case_id, window_labels[0].case_id)
    
    def test_save_and_load_json(self):
        """Test saving and loading window labels in JSON format."""
        window_labels = self.aligner.align_case_windows(
            case_id="2",
            n_windows=2
        )
        
        output_path = Path(self.test_dir) / "window_labels.json"
        
        # Save
        self.aligner.save_aligned_labels(
            window_labels,
            output_path,
            format="json"
        )
        
        self.assertTrue(output_path.exists())
        
        # Load
        loaded_labels = self.aligner.load_aligned_labels(
            output_path,
            format="json"
        )
        
        self.assertEqual(len(loaded_labels), len(window_labels))


class TestLabelAlignmentIntegration(unittest.TestCase):
    """Integration tests for complete label alignment workflow."""
    
    def setUp(self):
        """Set up realistic test scenario."""
        self.test_dir = tempfile.mkdtemp()
        self.metadata_path = Path(self.test_dir) / "vitaldb_cases.csv"
        
        # Create realistic metadata
        self.vitaldb_data = pd.DataFrame({
            'caseid': list(range(1, 6)),
            'death_inhosp': [0, 1, 0, 0, 1],
            'icu_days': [0, 5, 0, 2, 10],
            'age': [65, 72, 58, 45, 75],
            'sex': ['M', 'F', 'M', 'F', 'M'],
            'bmi': [25.0, 28.5, 22.0, 24.0, 27.0],
            'asa': [2, 3, 2, 1, 3]
        })
        
        self.vitaldb_data.to_csv(self.metadata_path, index=False)
        
        self.extractor = ClinicalLabelExtractor(
            metadata_path=self.metadata_path
        )
        
        self.aligner = LabelWindowAligner(
            label_extractor=self.extractor,
            window_duration=10.0
        )
    
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.test_dir)
    
    def test_complete_training_workflow(self):
        """Test complete workflow from labels to training arrays."""
        # Simulate having 10 windows per case for training
        case_window_mapping = {str(i): 10 for i in range(1, 4)}
        
        # Align labels
        batch_results = self.aligner.align_batch_windows(
            case_window_mapping,
            label_names=["mortality", "age"]
        )
        
        # Flatten all windows
        all_windows = []
        for case_id, window_labels in batch_results.items():
            all_windows.extend(window_labels)
        
        # Convert to arrays
        label_array, mask_array = self.aligner.create_window_label_arrays(
            all_windows,
            ["mortality", "age"],
            return_masks=True
        )
        
        # Should have 30 windows total (3 cases * 10 windows)
        self.assertEqual(label_array.shape[0], 30)
        self.assertEqual(label_array.shape[1], 2)
        
        # All labels should be present
        self.assertTrue(np.all(mask_array))


def run_tests():
    """Run all tests with verbose output."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestWindowLabel))
    suite.addTests(loader.loadTestsFromTestCase(TestLabelWindowAligner))
    suite.addTests(loader.loadTestsFromTestCase(TestLabelAlignmentIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
