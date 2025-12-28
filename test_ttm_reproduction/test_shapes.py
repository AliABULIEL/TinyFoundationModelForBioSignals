"""
Shape verification tests.
Every tensor must have the expected shape at every stage.
"""
import pytest
import torch
import numpy as np
import sys
import os

# Add repository root to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, repo_root)

from src.ttm_reproduction.config import CONFIG
from src.ttm_reproduction.data_loader import load_etth1_dataset
from src.ttm_reproduction.model_wrapper import load_ttm_model


class TestShapes:

    @pytest.fixture(scope="class")
    def test_dataset(self):
        _, _, test = load_etth1_dataset()
        return test

    @pytest.fixture(scope="class")
    def model(self):
        return load_ttm_model()

    def test_data_input_shape(self, test_dataset):
        """Verify input data has correct shape."""
        sample = test_dataset[0]
        past_values = sample["past_values"]

        assert past_values.shape == (CONFIG.CONTEXT_LENGTH, 7), \
            f"Expected ({CONFIG.CONTEXT_LENGTH}, 7), got {past_values.shape}"

    def test_data_target_shape(self, test_dataset):
        """Verify target data has correct shape."""
        sample = test_dataset[0]
        future_values = sample["future_values"]

        assert future_values.shape == (CONFIG.ROLLING_PREDICTION_LENGTH, 7), \
            f"Expected ({CONFIG.ROLLING_PREDICTION_LENGTH}, 7), got {future_values.shape}"

    def test_model_forward_shape(self, model, test_dataset):
        """Verify model output has correct shape."""
        sample = test_dataset[0]
        past_values = torch.tensor(sample["past_values"]).unsqueeze(0)  # Add batch dim

        with torch.no_grad():
            output = model(past_values=past_values)

        predictions = output.prediction_outputs
        expected_shape = (1, CONFIG.MODEL_PREDICTION_LENGTH, 7)

        assert predictions.shape == expected_shape, \
            f"Expected {expected_shape}, got {predictions.shape}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
