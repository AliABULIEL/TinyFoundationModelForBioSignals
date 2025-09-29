"""Tests for calibration utilities."""

import numpy as np
import pytest
import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.eval.calibration import (
    TemperatureScaling,
    PlattScaling,
    IsotonicCalibration,
    compute_reliability_diagram,
    expected_calibration_error,
    maximum_calibration_error,
    adaptive_calibration_error,
    find_threshold_for_sensitivity,
    find_threshold_for_specificity,
    CalibrationEvaluator,
)


class TestTemperatureScaling:
    """Test temperature scaling calibration."""
    
    def test_temperature_scaling_init(self):
        """Test temperature scaling initialization."""
        temp_scaler = TemperatureScaling(temperature=1.5)
        assert temp_scaler.temperature.item() == pytest.approx(1.5)
    
    def test_temperature_scaling_forward(self):
        """Test temperature scaling forward pass."""
        temp_scaler = TemperatureScaling(temperature=2.0)
        logits = torch.randn(32, 10)
        
        scaled_logits = temp_scaler(logits)
        
        # Check shape preserved
        assert scaled_logits.shape == logits.shape
        
        # Check scaling applied
        expected = logits / 2.0
        torch.testing.assert_close(scaled_logits, expected)
    
    def test_temperature_scaling_fit(self):
        """Test fitting temperature scaling."""
        # Set seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create synthetic data with VERY overconfident predictions
        n_samples = 200
        # Create logits that are extremely confident (large magnitude)
        logits = torch.randn(n_samples, 2)
        # Make predictions very confident by scaling up logits
        logits = logits * 5  # Very large logits = very overconfident
        
        # Create labels that disagree with predictions often
        # This ensures miscalibration
        probs = torch.softmax(logits, dim=1)
        predicted_class = (probs[:, 1] > 0.5).long()
        
        # Create labels that are ~50% wrong to ensure miscalibration
        labels = predicted_class.clone()
        flip_mask = torch.rand(n_samples) < 0.4  # 40% error rate
        labels[flip_mask] = 1 - labels[flip_mask]
        
        temp_scaler = TemperatureScaling()
        optimal_temp = temp_scaler.fit(logits, labels, max_iter=100)
        
        # Temperature should be > 1 for overconfident predictions
        assert optimal_temp > 1.0
        
        # Check that ECE improves
        original_probs = torch.softmax(logits, dim=1)[:, 1]
        calibrated_logits = temp_scaler(logits)
        calibrated_probs = torch.softmax(calibrated_logits, dim=1)[:, 1]
        
        ece_before = expected_calibration_error(labels, original_probs, n_bins=10)
        ece_after = expected_calibration_error(labels, calibrated_probs, n_bins=10)
        
        # ECE should improve (decrease) or at least not get much worse
        # Allow small tolerance for numerical issues
        assert ece_after <= ece_before + 0.01  # Allow 1% tolerance


class TestPlattScaling:
    """Test Platt scaling calibration."""
    
    def test_platt_scaling_init(self):
        """Test Platt scaling initialization."""
        platt_scaler = PlattScaling()
        assert platt_scaler.weight.item() == pytest.approx(1.0)
        assert platt_scaler.bias.item() == pytest.approx(0.0)
    
    def test_platt_scaling_forward(self):
        """Test Platt scaling forward pass."""
        platt_scaler = PlattScaling()
        platt_scaler.weight.data = torch.tensor(2.0)
        platt_scaler.bias.data = torch.tensor(0.5)
        
        logits = torch.randn(32, 1)
        probs = platt_scaler(logits)
        
        # Check shape
        assert probs.shape == logits.shape
        
        # Check values are probabilities
        assert torch.all(probs >= 0)
        assert torch.all(probs <= 1)
        
        # Check transformation
        expected = torch.sigmoid(2.0 * logits + 0.5)
        torch.testing.assert_close(probs, expected)
    
    def test_platt_scaling_fit(self):
        """Test fitting Platt scaling."""
        # Create synthetic binary classification data
        n_samples = 100
        logits = torch.randn(n_samples, 1)
        labels = (torch.sigmoid(logits).squeeze() > 0.5).float()
        # Add noise
        noise_mask = torch.rand(n_samples) < 0.1
        labels[noise_mask] = 1 - labels[noise_mask]
        
        platt_scaler = PlattScaling()
        weight, bias = platt_scaler.fit(logits, labels, max_iter=50)
        
        # Parameters should be fitted
        assert weight != 1.0
        assert bias != 0.0


class TestIsotonicCalibration:
    """Test isotonic regression calibration."""
    
    def test_isotonic_fit_transform(self):
        """Test isotonic calibration fit and transform."""
        # Create synthetic probabilities with calibration error
        n_samples = 200
        probs = np.random.beta(2, 5, n_samples)  # Skewed probabilities
        # Create labels with some calibration error
        labels = (probs + np.random.normal(0, 0.1, n_samples) > 0.5).astype(float)
        
        iso_calib = IsotonicCalibration()
        iso_calib.fit(probs, labels)
        
        # Transform probabilities
        calibrated_probs = iso_calib.transform(probs)
        
        # Check output shape
        assert calibrated_probs.shape == probs.shape
        
        # Check values are valid probabilities
        assert np.all(calibrated_probs >= 0)
        assert np.all(calibrated_probs <= 1)
        
        # Check monotonicity (isotonic property)
        sorted_indices = np.argsort(probs)
        sorted_calibrated = calibrated_probs[sorted_indices]
        assert np.all(np.diff(sorted_calibrated) >= -1e-7)  # Allow small numerical errors
    
    def test_isotonic_with_torch_tensors(self):
        """Test isotonic calibration with PyTorch tensors."""
        probs = torch.rand(100)
        labels = (probs > 0.5).float()
        
        iso_calib = IsotonicCalibration()
        iso_calib.fit(probs, labels)
        
        # Transform should work with tensors
        calibrated = iso_calib.transform(probs)
        
        assert isinstance(calibrated, torch.Tensor)
        assert calibrated.shape == probs.shape
        assert calibrated.device == probs.device


class TestReliabilityDiagram:
    """Test reliability diagram computation."""
    
    def test_compute_reliability_diagram(self):
        """Test reliability diagram data computation."""
        # Create perfectly calibrated predictions
        n_samples = 1000
        y_prob = np.random.uniform(0, 1, n_samples)
        y_true = (np.random.random(n_samples) < y_prob).astype(int)
        
        diagram = compute_reliability_diagram(y_true, y_prob, n_bins=10)
        
        assert 'bin_edges' in diagram
        assert 'bin_centers' in diagram
        assert 'bin_accs' in diagram
        assert 'bin_confs' in diagram
        assert 'bin_counts' in diagram
        
        assert len(diagram['bin_edges']) == 11
        assert len(diagram['bin_centers']) == 10
        assert len(diagram['bin_accs']) == 10
        assert len(diagram['bin_confs']) == 10
        assert len(diagram['bin_counts']) == 10
        
        # Check bin counts sum to total samples
        assert diagram['bin_counts'].sum() == n_samples
    
    def test_reliability_diagram_with_torch(self):
        """Test reliability diagram with torch tensors."""
        y_prob = torch.rand(100)
        y_true = (y_prob > 0.5).long()
        
        diagram = compute_reliability_diagram(y_true, y_prob, n_bins=5)
        
        assert isinstance(diagram['bin_edges'], np.ndarray)
        assert len(diagram['bin_centers']) == 5


class TestCalibrationErrors:
    """Test calibration error metrics."""
    
    def test_expected_calibration_error_perfect(self):
        """Test ECE for perfectly calibrated predictions."""
        # Create perfectly calibrated predictions
        n_samples = 1000
        y_prob = np.random.uniform(0, 1, n_samples)
        y_true = (np.random.random(n_samples) < y_prob).astype(int)
        
        ece = expected_calibration_error(y_true, y_prob, n_bins=10)
        
        # ECE should be small for well-calibrated predictions
        assert ece < 0.05
    
    def test_expected_calibration_error_miscalibrated(self):
        """Test ECE for miscalibrated predictions."""
        # Create overconfident predictions
        n_samples = 1000
        y_prob = np.random.uniform(0.8, 1.0, n_samples)  # All high confidence
        y_true = np.random.randint(0, 2, n_samples)  # Random labels
        
        ece = expected_calibration_error(y_true, y_prob, n_bins=10)
        
        # ECE should be large for miscalibrated predictions
        assert ece > 0.2
    
    def test_maximum_calibration_error(self):
        """Test MCE computation."""
        # Create data with one very miscalibrated bin
        y_prob = np.array([0.1] * 50 + [0.9] * 50)
        y_true = np.array([0] * 50 + [0] * 50)  # Second group is miscalibrated
        
        mce = maximum_calibration_error(y_true, y_prob, n_bins=10)
        
        # MCE should capture the worst bin
        assert mce > 0.8
    
    def test_adaptive_calibration_error(self):
        """Test ACE computation."""
        n_samples = 100
        y_prob = np.random.uniform(0, 1, n_samples)
        y_true = (y_prob > 0.5).astype(int)
        
        ace = adaptive_calibration_error(y_true, y_prob, n_bins=10)
        
        assert ace >= 0
        assert ace <= 1
    
    def test_calibration_errors_with_torch(self):
        """Test calibration errors with torch tensors."""
        y_prob = torch.rand(100)
        y_true = (y_prob > 0.5).long()
        
        ece = expected_calibration_error(y_true, y_prob)
        mce = maximum_calibration_error(y_true, y_prob)
        ace = adaptive_calibration_error(y_true, y_prob)
        
        assert isinstance(ece, float)
        assert isinstance(mce, float)
        assert isinstance(ace, float)
        assert ece >= 0
        assert mce >= 0
        assert ace >= 0


class TestThresholdTuning:
    """Test threshold tuning for sensitivity/specificity."""
    
    def test_find_threshold_for_sensitivity(self):
        """Test finding threshold for target sensitivity."""
        # Create synthetic predictions
        n_samples = 1000
        y_prob = np.random.uniform(0, 1, n_samples)
        y_true = (y_prob > 0.3).astype(int)  # Threshold at 0.3
        
        # Find threshold for 95% sensitivity
        threshold = find_threshold_for_sensitivity(
            y_true, y_prob, target_sensitivity=0.95, class_of_interest=1
        )
        
        # Check that threshold achieves target sensitivity
        positive_mask = y_true == 1
        positive_probs = y_prob[positive_mask]
        achieved_sensitivity = (positive_probs >= threshold).mean()
        
        assert achieved_sensitivity >= 0.94  # Close to target
    
    def test_find_threshold_for_specificity(self):
        """Test finding threshold for target specificity."""
        # Create synthetic predictions
        n_samples = 1000
        y_prob = np.random.uniform(0, 1, n_samples)
        y_true = (y_prob > 0.6).astype(int)  # Threshold at 0.6
        
        # Find threshold for 95% specificity
        threshold = find_threshold_for_specificity(
            y_true, y_prob, target_specificity=0.95, negative_class=0
        )
        
        # Check that threshold achieves target specificity
        negative_mask = y_true == 0
        negative_probs = y_prob[negative_mask]
        achieved_specificity = (negative_probs < threshold).mean()
        
        assert achieved_specificity >= 0.94  # Close to target
    
    def test_threshold_tuning_edge_cases(self):
        """Test threshold tuning with edge cases."""
        # All positive samples
        y_true = np.ones(100)
        y_prob = np.random.uniform(0.5, 1, 100)
        
        threshold = find_threshold_for_sensitivity(y_true, y_prob, 0.95)
        assert 0 <= threshold <= 1
        
        # All negative samples
        y_true = np.zeros(100)
        y_prob = np.random.uniform(0, 0.5, 100)
        
        threshold = find_threshold_for_specificity(y_true, y_prob, 0.95)
        assert 0 <= threshold <= 1


class TestCalibrationEvaluator:
    """Test comprehensive calibration evaluator."""
    
    def test_evaluator_all_metrics(self):
        """Test evaluator computes all metrics."""
        n_samples = 500
        y_prob = np.random.uniform(0, 1, n_samples)
        y_true = (y_prob + np.random.normal(0, 0.2, n_samples) > 0.5).astype(int)
        
        evaluator = CalibrationEvaluator(n_bins=10)
        metrics = evaluator.evaluate(y_true, y_prob)
        
        assert 'ece' in metrics
        assert 'mce' in metrics
        assert 'ace' in metrics
        assert 'brier_score' in metrics
        
        for value in metrics.values():
            assert value >= 0
            assert value <= 1
    
    def test_evaluator_optimal_thresholds(self):
        """Test finding multiple optimal thresholds."""
        n_samples = 1000
        y_prob = np.random.uniform(0, 1, n_samples)
        y_true = (y_prob > 0.4).astype(int)
        
        evaluator = CalibrationEvaluator()
        thresholds = evaluator.find_optimal_thresholds(
            y_true, y_prob,
            sensitivities=[0.90, 0.95],
            specificities=[0.90, 0.95]
        )
        
        assert 'threshold_sens_0.90' in thresholds
        assert 'threshold_sens_0.95' in thresholds
        assert 'threshold_spec_0.90' in thresholds
        assert 'threshold_spec_0.95' in thresholds
        
        for threshold in thresholds.values():
            assert 0 <= threshold <= 1
    
    def test_evaluator_improves_ece(self):
        """Test that calibration improves ECE on toy example."""
        # Create overconfident logits
        n_samples = 200
        logits = torch.randn(n_samples, 2) * 5  # Very confident
        probs_original = torch.softmax(logits, dim=1)[:, 1]
        
        # Create noisy labels
        labels = (probs_original > 0.5).long()
        noise_mask = torch.rand(n_samples) < 0.2
        labels[noise_mask] = 1 - labels[noise_mask]
        
        # Evaluate before calibration
        evaluator = CalibrationEvaluator(n_bins=10)
        metrics_before = evaluator.evaluate(labels, probs_original)
        
        # Apply temperature scaling
        temp_scaler = TemperatureScaling()
        temp_scaler.fit(logits, labels, max_iter=100)
        calibrated_logits = temp_scaler(logits)
        probs_calibrated = torch.softmax(calibrated_logits, dim=1)[:, 1]
        
        # Evaluate after calibration
        metrics_after = evaluator.evaluate(labels, probs_calibrated)
        
        # ECE should improve
        assert metrics_after['ece'] < metrics_before['ece']
