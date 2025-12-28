"""
Comprehensive tests for clinical label extraction module.

Tests cover:
- Label configuration and validation
- Case-level label extraction
- Batch processing
- Missing value handling strategies
- Label statistics and reporting
- Integration with VitalDB metadata

Author: Senior Data Engineering Team
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
import tempfile
import shutil
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd

from src.data.clinical_labels import (
    ClinicalLabelExtractor,
    LabelConfig,
    LabelType,
    ClinicalLabels
)


class TestLabelConfig(unittest.TestCase):
    """Test LabelConfig dataclass and validation."""
    
    def test_binary_label_config(self):
        """Test binary label configuration."""
        config = LabelConfig(
            name="mortality",
            column_name="death_inhosp",
            label_type=LabelType.BINARY,
            missing_value_strategy="fill",
            fill_value=0
        )
        
        self.assertTrue(config.validate())
        self.assertEqual(config.name, "mortality")
        self.assertEqual(config.label_type, LabelType.BINARY)
    
    def test_continuous_label_config(self):
        """Test continuous label with valid range."""
        config = LabelConfig(
            name="age",
            column_name="age",
            label_type=LabelType.CONTINUOUS,
            valid_range=(0, 120)
        )
        
        self.assertTrue(config.validate())
        self.assertEqual(config.valid_range, (0, 120))
    
    def test_categorical_label_config(self):
        """Test categorical label with categories."""
        config = LabelConfig(
            name="sex",
            column_name="sex",
            label_type=LabelType.CATEGORICAL,
            categories=['M', 'F']
        )
        
        self.assertTrue(config.validate())
        self.assertEqual(len(config.categories), 2)
    
    def test_invalid_categorical_config(self):
        """Test that categorical label without categories fails validation."""
        config = LabelConfig(
            name="invalid",
            column_name="invalid_col",
            label_type=LabelType.CATEGORICAL,
            categories=None
        )
        
        self.assertFalse(config.validate())


class TestClinicalLabelExtractor(unittest.TestCase):
    """Test ClinicalLabelExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.metadata_path = Path(self.test_dir) / "test_metadata.csv"
        
        # Create sample metadata
        self.sample_data = pd.DataFrame({
            'caseid': ['1', '2', '3', '4', '5'],
            'death_inhosp': [0, 1, 0, 0, np.nan],
            'icu_days': [0, 3, 0, 1, 0],
            'age': [65, 72, 58, np.nan, 45],
            'sex': ['M', 'F', 'M', 'F', 'M'],
            'bmi': [25.0, 28.5, np.nan, 22.0, 30.0],
            'asa': [2, 3, 2, 1, 2],
            'emop': [0, 1, 0, 0, 0]
        })
        
        self.sample_data.to_csv(self.metadata_path, index=False)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test extractor initialization."""
        extractor = ClinicalLabelExtractor(
            metadata_path=self.metadata_path
        )
        
        self.assertIsNotNone(extractor.metadata_df)
        self.assertEqual(len(extractor.metadata_df), 5)
        self.assertTrue(len(extractor.label_configs) > 0)
    
    def test_initialization_without_metadata(self):
        """Test initialization without metadata file."""
        extractor = ClinicalLabelExtractor()
        
        self.assertIsNone(extractor.metadata_df)
        self.assertTrue(len(extractor.label_configs) > 0)  # Standard labels loaded
    
    def test_add_label_config(self):
        """Test adding custom label configuration."""
        extractor = ClinicalLabelExtractor(
            metadata_path=self.metadata_path,
            auto_load_standard=False
        )
        
        initial_count = len(extractor.label_configs)
        
        extractor.add_label_config(
            name="custom_label",
            column_name="custom_col",
            label_type="continuous",
            valid_range=(0, 100)
        )
        
        self.assertEqual(len(extractor.label_configs), initial_count + 1)
        self.assertIn("custom_label", extractor.label_configs)
    
    def test_extract_case_labels_success(self):
        """Test successful label extraction for a case."""
        extractor = ClinicalLabelExtractor(
            metadata_path=self.metadata_path
        )
        
        labels = extractor.extract_case_labels(case_id="1")
        
        self.assertIsNotNone(labels)
        self.assertIsInstance(labels, ClinicalLabels)
        self.assertEqual(labels.case_id, "1")
        self.assertIn("mortality", labels.labels)
        self.assertEqual(labels.labels["mortality"], 0)
    
    def test_extract_case_labels_not_found(self):
        """Test label extraction for non-existent case."""
        extractor = ClinicalLabelExtractor(
            metadata_path=self.metadata_path
        )
        
        labels = extractor.extract_case_labels(case_id="999")
        
        self.assertIsNone(labels)
    
    def test_extract_specific_labels(self):
        """Test extracting only specific labels."""
        extractor = ClinicalLabelExtractor(
            metadata_path=self.metadata_path
        )
        
        labels = extractor.extract_case_labels(
            case_id="2",
            label_names=["mortality", "age"]
        )
        
        self.assertIsNotNone(labels)
        self.assertIn("mortality", labels.labels)
        self.assertIn("age", labels.labels)
        # Should not have other labels
        self.assertNotIn("bmi", labels.labels) or labels.labels.get("bmi") is None
    
    def test_missing_value_handling_fill(self):
        """Test missing value handling with fill strategy."""
        extractor = ClinicalLabelExtractor(
            metadata_path=self.metadata_path
        )
        
        # Case 5 has np.nan for death_inhosp
        labels = extractor.extract_case_labels(case_id="5")
        
        self.assertIsNotNone(labels)
        # mortality should be filled with 0
        self.assertIn("mortality", labels.labels)
        self.assertEqual(labels.labels["mortality"], 0)
    
    def test_missing_value_handling_drop(self):
        """Test missing value handling with drop strategy."""
        extractor = ClinicalLabelExtractor(
            metadata_path=self.metadata_path
        )
        
        # Case 4 has np.nan for age
        labels = extractor.extract_case_labels(case_id="4")
        
        self.assertIsNotNone(labels)
        # age should be dropped or filled with mean
        if "age" in labels.labels:
            # If filled with mean, it should be a reasonable value
            self.assertTrue(0 < labels.labels["age"] < 120)
    
    def test_valid_range_checking(self):
        """Test that values outside valid range are rejected."""
        # Add a case with invalid BMI
        invalid_data = pd.DataFrame({
            'caseid': ['6'],
            'death_inhosp': [0],
            'icu_days': [0],
            'age': [65],
            'sex': ['M'],
            'bmi': [200.0],  # Invalid BMI
            'asa': [2],
            'emop': [0]
        })
        
        invalid_path = Path(self.test_dir) / "invalid_metadata.csv"
        invalid_data.to_csv(invalid_path, index=False)
        
        extractor = ClinicalLabelExtractor(metadata_path=invalid_path)
        labels = extractor.extract_case_labels(case_id="6")
        
        # BMI should be None or not present due to range violation
        self.assertTrue(
            "bmi" not in labels.labels or 
            labels.labels["bmi"] is None
        )
    
    def test_batch_extraction(self):
        """Test batch label extraction."""
        extractor = ClinicalLabelExtractor(
            metadata_path=self.metadata_path
        )
        
        case_ids = ["1", "2", "3"]
        batch_labels = extractor.extract_batch_labels(case_ids)
        
        self.assertEqual(len(batch_labels), 3)
        for case_id in case_ids:
            self.assertIn(case_id, batch_labels)
            self.assertIsInstance(batch_labels[case_id], ClinicalLabels)
    
    def test_get_label_statistics(self):
        """Test label statistics computation."""
        extractor = ClinicalLabelExtractor(
            metadata_path=self.metadata_path
        )
        
        stats = extractor.get_label_statistics("age")
        
        self.assertIn("mean", stats)
        self.assertIn("std", stats)
        self.assertIn("n_missing", stats)
        self.assertTrue(stats["n_total"] == 5)
    
    def test_generate_label_report(self):
        """Test generating comprehensive label report."""
        extractor = ClinicalLabelExtractor(
            metadata_path=self.metadata_path
        )
        
        report = extractor.generate_label_report()
        
        self.assertIsInstance(report, pd.DataFrame)
        self.assertTrue(len(report) > 0)
        self.assertIn("name", report.columns)
        self.assertIn("type", report.columns)
        self.assertIn("n_missing", report.columns)
    
    def test_validate_labels(self):
        """Test label validation for multiple cases."""
        extractor = ClinicalLabelExtractor(
            metadata_path=self.metadata_path
        )
        
        case_ids = ["1", "2", "999"]  # 999 doesn't exist
        validation_results = extractor.validate_labels(case_ids)
        
        self.assertEqual(validation_results["n_cases"], 3)
        self.assertEqual(validation_results["n_valid"], 2)
        self.assertEqual(validation_results["n_invalid"], 1)
        self.assertIn("999", validation_results["missing_cases"])
    
    def test_categorical_label_validation(self):
        """Test categorical label with invalid category."""
        # Add invalid category
        invalid_data = self.sample_data.copy()
        invalid_data.loc[0, 'sex'] = 'X'  # Invalid sex
        
        invalid_path = Path(self.test_dir) / "invalid_cat.csv"
        invalid_data.to_csv(invalid_path, index=False)
        
        extractor = ClinicalLabelExtractor(metadata_path=invalid_path)
        labels = extractor.extract_case_labels(case_id="1")
        
        # Sex should be rejected
        self.assertTrue(
            "sex" not in labels.labels or 
            labels.labels["sex"] is None
        )


class TestClinicalLabelsIntegration(unittest.TestCase):
    """Integration tests with realistic VitalDB-like data."""
    
    def setUp(self):
        """Set up with realistic VitalDB structure."""
        self.test_dir = tempfile.mkdtemp()
        self.metadata_path = Path(self.test_dir) / "vitaldb_cases.csv"
        
        # Create realistic VitalDB metadata
        self.vitaldb_data = pd.DataFrame({
            'caseid': list(range(1, 11)),
            'death_inhosp': [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
            'icu_days': [0, 5, 0, 2, 0, 10, 1, 0, 0, 3],
            'age': [65, 72, 58, 45, 68, 75, 52, 60, 48, 70],
            'sex': ['M', 'F', 'M', 'F', 'M', 'M', 'F', 'M', 'F', 'M'],
            'height': [170, 160, 175, 165, 172, 168, 158, 180, 162, 175],
            'weight': [70, 65, 80, 60, 75, 72, 55, 85, 62, 78],
            'bmi': [24.2, 25.4, 26.1, 22.0, 25.3, 25.5, 22.0, 26.2, 23.6, 25.4],
            'asa': [2, 3, 2, 1, 2, 3, 2, 2, 1, 2],
            'emop': [0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
        })
        
        self.vitaldb_data.to_csv(self.metadata_path, index=False)
    
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.test_dir)
    
    def test_realistic_workflow(self):
        """Test realistic label extraction workflow."""
        # Initialize extractor
        extractor = ClinicalLabelExtractor(
            metadata_path=self.metadata_path
        )
        
        # Extract labels for training set
        train_cases = [1, 2, 3, 4, 5]
        train_labels = extractor.extract_batch_labels(train_cases)
        
        self.assertEqual(len(train_labels), 5)
        
        # Generate report
        report = extractor.generate_label_report()
        self.assertTrue(len(report) > 0)
        
        # Validate
        validation = extractor.validate_labels(train_cases)
        self.assertEqual(validation["n_valid"], 5)
    
    def test_multi_label_extraction(self):
        """Test extracting multiple labels for ML tasks."""
        extractor = ClinicalLabelExtractor(
            metadata_path=self.metadata_path
        )
        
        # Define multi-task setup
        labels_of_interest = ["mortality", "icu_los", "age", "sex"]
        
        case_labels = extractor.extract_case_labels(
            case_id=1,
            label_names=labels_of_interest
        )
        
        self.assertIsNotNone(case_labels)
        for label_name in ["mortality", "icu_los", "age"]:
            self.assertIn(label_name, case_labels.labels)


def run_tests():
    """Run all tests with verbose output."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestLabelConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestClinicalLabelExtractor))
    suite.addTests(loader.loadTestsFromTestCase(TestClinicalLabelsIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
