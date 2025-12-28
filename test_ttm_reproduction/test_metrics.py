"""
Metric computation correctness tests.
"""
import pytest
import numpy as np
import sys
import os

# Add repository root to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, repo_root)

from src.ttm_reproduction.evaluation import compute_mse_mae


class TestMetrics:

    def test_mse_computation(self):
        """Verify MSE is computed correctly."""
        predictions = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        targets = np.array([[[1.5, 2.5], [3.5, 4.5]]])

        metrics = compute_mse_mae(predictions, targets)

        # Expected MSE: mean of (0.5^2, 0.5^2, 0.5^2, 0.5^2) = 0.25
        expected_mse = 0.25
        assert np.isclose(metrics["mse"], expected_mse), \
            f"Expected MSE {expected_mse}, got {metrics['mse']}"

    def test_mae_computation(self):
        """Verify MAE is computed correctly."""
        predictions = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        targets = np.array([[[1.5, 2.5], [3.5, 4.5]]])

        metrics = compute_mse_mae(predictions, targets)

        # Expected MAE: mean of (0.5, 0.5, 0.5, 0.5) = 0.5
        expected_mae = 0.5
        assert np.isclose(metrics["mae"], expected_mae), \
            f"Expected MAE {expected_mae}, got {metrics['mae']}"

    def test_perfect_predictions(self):
        """Verify metrics are zero for perfect predictions."""
        predictions = np.random.randn(10, 192, 7)
        targets = predictions.copy()

        metrics = compute_mse_mae(predictions, targets)

        assert np.isclose(metrics["mse"], 0.0, atol=1e-10), \
            f"Expected MSE 0.0 for perfect predictions, got {metrics['mse']}"
        assert np.isclose(metrics["mae"], 0.0, atol=1e-10), \
            f"Expected MAE 0.0 for perfect predictions, got {metrics['mae']}"

    def test_shape_mismatch_raises_error(self):
        """Verify that shape mismatch raises an AssertionError."""
        predictions = np.random.randn(10, 192, 7)
        targets = np.random.randn(10, 96, 7)  # Different shape

        with pytest.raises(AssertionError):
            compute_mse_mae(predictions, targets)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
