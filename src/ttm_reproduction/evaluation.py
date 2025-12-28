"""
Evaluation metrics matching IBM's protocol exactly.
"""
import numpy as np
import torch
from typing import Dict
from transformers import Trainer, TrainingArguments
import tempfile
import os

from .config import CONFIG

def run_zero_shot_evaluation(model, test_dataset) -> Dict[str, float]:
    """
    Run zero-shot evaluation using HuggingFace Trainer.
    This matches IBM's exact evaluation protocol.

    Args:
        model: TTM model (base or rolling)
        test_dataset: Test dataset from load_dataset

    Returns:
        Dictionary with eval_loss (MSE) and other metrics
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        eval_args = TrainingArguments(
            output_dir=temp_dir,
            per_device_eval_batch_size=CONFIG.BATCH_SIZE,
            report_to="none",
            seed=CONFIG.SEED,
            dataloader_num_workers=0,  # Colab compatibility
        )

        trainer = Trainer(
            model=model,
            args=eval_args,
        )

        results = trainer.evaluate(test_dataset)

    return results


def compute_mse_mae(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Compute MSE and MAE metrics.

    Args:
        predictions: Shape (N, horizon, channels)
        targets: Shape (N, horizon, channels)

    Returns:
        Dictionary with MSE and MAE values
    """
    assert predictions.shape == targets.shape, \
        f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"

    mse = float(np.mean((predictions - targets) ** 2))
    mae = float(np.mean(np.abs(predictions - targets)))

    return {"mse": mse, "mae": mae}
