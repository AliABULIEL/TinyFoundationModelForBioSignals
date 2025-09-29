"""Tests for evaluation metrics."""

import numpy as np
import pytest
import torch

from src.eval.metrics import (
    # Classification metrics
    accuracy,
    auroc,
    auprc,
    f1,
    precision,
    recall,
    classification_metrics,
    # Regression metrics
    mae,
    rmse,
    mse,
    ccc,
    pearson_r,
    r2,
    regression_metrics,
    # Utilities
    MetricTracker
)


class TestClassificationMetrics:
    """Test classification metrics."""
    
    def test_accuracy(self):
        """Test accuracy computation."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 1, 0])
        
        acc = accuracy(y_true, y_pred)
        assert acc == 4/6  # 4 correct out of 6
        
        # Test with probabilities
        y_prob = np.array([0.2, 0.8, 0.3, 0.7, 0.6, 0.4])
        acc = accuracy(y_true, y_prob, threshold=0.5)
        assert acc == 4/6
    
    def test_accuracy_torch(self):
        """Test accuracy with torch tensors."""
        y_true = torch.tensor([0, 1, 0, 1, 0, 1])
        y_pred = torch.tensor([0, 1, 0, 1, 1, 0])
        
        acc = accuracy(y_true, y_pred)
        assert acc == 4/6
    
    def test_auroc(self):
        """Test AUROC computation."""
        # Perfect separation
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        
        auc = auroc(y_true, y_score)
        assert auc == 1.0
        
        # Random predictions
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_score = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        
        auc = auroc(y_true, y_score)
        assert 0.4 <= auc <= 0.6  # Should be around 0.5
    
    def test_auprc(self):
        """Test AUPRC computation."""
        # Perfect predictions
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
        
        auc = auprc(y_true, y_score)
        assert auc >= 0.9  # Should be close to 1
        
        # Test with imbalanced data
        y_true = np.array([0, 0, 0, 0, 0, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.9])
        
        auc = auprc(y_true, y_score)
        assert auc > 0  # Should be positive
    
    def test_f1(self):
        """Test F1 score computation."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])
        
        score = f1(y_true, y_pred)
        assert score == 1.0  # Perfect predictions
        
        # With some errors
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        score = f1(y_true, y_pred)
        assert 0 < score < 1
    
    def test_precision_recall(self):
        """Test precision and recall."""
        # TP=2, FP=1, FN=1
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        
        prec = precision(y_true, y_pred)
        assert prec == 2/3  # 2 TP out of 3 positive predictions
        
        rec = recall(y_true, y_pred)
        assert rec == 2/3  # 2 TP out of 3 actual positives
    
    def test_classification_metrics_all(self):
        """Test composite classification metrics."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_score = np.array([0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.1, 0.9])
        
        metrics = classification_metrics(y_true, y_score)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'auroc' in metrics
        assert 'auprc' in metrics
        
        # All metrics should be between 0 and 1
        for value in metrics.values():
            assert 0 <= value <= 1


class TestRegressionMetrics:
    """Test regression metrics."""
    
    def test_mae(self):
        """Test MAE computation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.2, 2.9, 3.8])
        
        error = mae(y_true, y_pred)
        expected = np.mean([0.1, 0.2, 0.1, 0.2])
        assert abs(error - expected) < 1e-6
        
        # Perfect predictions
        y_pred = y_true.copy()
        error = mae(y_true, y_pred)
        assert error == 0
    
    def test_rmse(self):
        """Test RMSE computation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        
        error = rmse(y_true, y_pred)
        assert error == 0  # Perfect predictions
        
        # With errors
        y_pred = np.array([1.1, 2.1, 3.1, 4.1])
        error = rmse(y_true, y_pred)
        assert abs(error - 0.1) < 1e-6
    
    def test_mse(self):
        """Test MSE computation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1])
        
        error = mse(y_true, y_pred)
        assert abs(error - 0.01) < 1e-6  # 0.1^2
    
    def test_ccc(self):
        """Test CCC computation."""
        # Perfect correlation
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()
        
        corr = ccc(y_true, y_pred)
        assert abs(corr - 1.0) < 1e-6
        
        # Shifted predictions
        y_pred = y_true + 1.0
        corr = ccc(y_true, y_pred)
        assert corr < 1.0  # Should be less than perfect
        
        # Scaled predictions
        y_pred = y_true * 2.0
        corr = ccc(y_true, y_pred)
        assert corr < 1.0
    
    def test_pearson_r(self):
        """Test Pearson correlation."""
        # Perfect positive correlation
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([2, 4, 6, 8, 10])
        
        r, p = pearson_r(y_true, y_pred)
        assert abs(r - 1.0) < 1e-6
        assert p < 0.05  # Significant
        
        # No correlation
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([3, 3, 3, 3, 3])
        
        r, p = pearson_r(y_true, y_pred)
        # Note: constant array gives NaN correlation
    
    def test_r2(self):
        """Test R-squared computation."""
        # Perfect predictions
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = y_true.copy()
        
        score = r2(y_true, y_pred)
        assert score == 1.0
        
        # Poor predictions
        y_pred = np.ones_like(y_true) * np.mean(y_true)
        score = r2(y_true, y_pred)
        assert abs(score) < 1e-6  # Should be ~0
    
    def test_regression_metrics_all(self):
        """Test composite regression metrics."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.9, 3.8, 5.1])
        
        metrics = regression_metrics(y_true, y_pred)
        
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'mse' in metrics
        assert 'ccc' in metrics
        assert 'r2' in metrics
        assert 'pearson_r' in metrics
        assert 'pearson_p' in metrics
        
        # Check reasonable values
        assert metrics['mae'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['mse'] >= 0
        assert -1 <= metrics['ccc'] <= 1
        assert -1 <= metrics['pearson_r'] <= 1


class TestMetricTracker:
    """Test MetricTracker utility."""
    
    def test_tracker_init(self):
        """Test tracker initialization."""
        tracker = MetricTracker(['loss', 'accuracy'])
        
        assert 'loss' in tracker.values
        assert 'accuracy' in tracker.values
        assert len(tracker.values['loss']) == 0
    
    def test_tracker_update(self):
        """Test updating tracker."""
        tracker = MetricTracker(['loss', 'accuracy'])
        
        # Update with values
        tracker.update({'loss': 0.5, 'accuracy': 0.8}, count=32)
        tracker.update({'loss': 0.4, 'accuracy': 0.85}, count=32)
        
        assert len(tracker.values['loss']) == 2
        assert len(tracker.values['accuracy']) == 2
    
    def test_tracker_average(self):
        """Test computing averages."""
        tracker = MetricTracker(['loss', 'accuracy'])
        
        # Add values with different counts
        tracker.update({'loss': 0.5, 'accuracy': 0.8}, count=30)
        tracker.update({'loss': 0.3, 'accuracy': 0.9}, count=10)
        
        averages = tracker.average()
        
        # Weighted average
        expected_loss = (0.5 * 30 + 0.3 * 10) / 40
        expected_acc = (0.8 * 30 + 0.9 * 10) / 40
        
        assert abs(averages['loss'] - expected_loss) < 1e-6
        assert abs(averages['accuracy'] - expected_acc) < 1e-6
    
    def test_tracker_last(self):
        """Test getting last values."""
        tracker = MetricTracker(['loss', 'accuracy'])
        
        tracker.update({'loss': 0.5, 'accuracy': 0.8})
        tracker.update({'loss': 0.3, 'accuracy': 0.9})
        
        last = tracker.last()
        
        assert last['loss'] == 0.3
        assert last['accuracy'] == 0.9
    
    def test_tracker_reset(self):
        """Test resetting tracker."""
        tracker = MetricTracker(['loss'])
        
        tracker.update({'loss': 0.5})
        assert len(tracker.values['loss']) == 1
        
        tracker.reset()
        assert len(tracker.values['loss']) == 0


def test_tensor_conversion():
    """Test automatic tensor to numpy conversion."""
    # PyTorch tensors
    y_true = torch.tensor([0, 1, 0, 1])
    y_pred = torch.tensor([0.1, 0.9, 0.2, 0.8])
    
    # Should work with tensors
    acc = accuracy(y_true, y_pred)
    assert isinstance(acc, float)
    
    auc = auroc(y_true, y_pred)
    assert isinstance(auc, float)
    
    # Mixed tensor and numpy
    y_true_np = np.array([0, 1, 0, 1])
    acc = accuracy(y_true_np, y_pred)
    assert isinstance(acc, float)


def test_shape_handling():
    """Test handling of different input shapes."""
    # 2D inputs (batch dimension)
    y_true = np.array([[0], [1], [0], [1]])
    y_pred = np.array([[0.1], [0.9], [0.2], [0.8]])
    
    acc = accuracy(y_true, y_pred)
    assert isinstance(acc, float)
    
    # Should squeeze to 1D
    auc = auroc(y_true, y_pred)
    assert isinstance(auc, float)


def test_edge_cases():
    """Test edge cases for metrics."""
    # Empty inputs
    y_true = np.array([])
    y_pred = np.array([])
    
    # MAE should handle empty
    with pytest.raises(Exception):
        mae(y_true, y_pred)
    
    # Single sample
    y_true = np.array([1])
    y_pred = np.array([1])
    
    error = mae(y_true, y_pred)
    assert error == 0
    
    # All same class
    y_true = np.array([1, 1, 1, 1])
    y_score = np.array([0.5, 0.6, 0.7, 0.8])
    
    # AUROC undefined for single class
    with pytest.raises(Exception):
        auroc(y_true, y_score)
