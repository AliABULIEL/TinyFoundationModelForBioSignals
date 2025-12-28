"""
Determinism tests.
Same seed must produce identical results across runs.
"""
import pytest
import torch
import numpy as np
from transformers import set_seed
import sys
import os

# Add repository root to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, repo_root)

from src.ttm_reproduction.config import CONFIG
from src.ttm_reproduction.data_loader import load_etth1_dataset
from src.ttm_reproduction.model_wrapper import load_ttm_model


class TestDeterminism:

    def test_data_loading_determinism(self):
        """Verify data loading is deterministic with same seed."""
        set_seed(CONFIG.SEED)
        _, _, test1 = load_etth1_dataset()
        sample1 = test1[0]["past_values"]

        set_seed(CONFIG.SEED)
        _, _, test2 = load_etth1_dataset()
        sample2 = test2[0]["past_values"]

        np.testing.assert_array_equal(sample1, sample2,
            err_msg="Data loading is not deterministic with same seed")

    def test_model_inference_determinism(self):
        """Verify model inference is deterministic with same seed."""
        set_seed(CONFIG.SEED)
        model = load_ttm_model()
        model.eval()

        # Create dummy input
        x = torch.randn(1, CONFIG.CONTEXT_LENGTH, 7)

        with torch.no_grad():
            torch.manual_seed(CONFIG.SEED)
            out1 = model(past_values=x).prediction_outputs

            torch.manual_seed(CONFIG.SEED)
            out2 = model(past_values=x).prediction_outputs

        torch.testing.assert_close(out1, out2,
            msg="Model inference is not deterministic with same seed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
