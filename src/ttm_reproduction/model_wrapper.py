"""
Thin wrapper around IBM's TTM implementation.
NO reimplementation - direct usage of IBM code only.
"""
import torch
from transformers import set_seed

from .config import CONFIG

def load_ttm_model():
    """
    Load the official IBM TTM model.

    Returns:
        TinyTimeMixerForPrediction model

    Raises:
        AssertionError: If model config doesn't match expected values
    """
    set_seed(CONFIG.SEED)

    from tsfm_public import TinyTimeMixerForPrediction

    model = TinyTimeMixerForPrediction.from_pretrained(
        CONFIG.MODEL_PATH,
        revision=CONFIG.MODEL_REVISION,
    )

    # Verify model configuration
    assert model.config.context_length == CONFIG.CONTEXT_LENGTH, \
        f"Model context_length mismatch: expected {CONFIG.CONTEXT_LENGTH}, got {model.config.context_length}"
    assert model.config.prediction_length == CONFIG.MODEL_PREDICTION_LENGTH, \
        f"Model prediction_length mismatch: expected {CONFIG.MODEL_PREDICTION_LENGTH}, got {model.config.prediction_length}"

    print(f"[MODEL] Loaded TTM from {CONFIG.MODEL_PATH} (revision: {CONFIG.MODEL_REVISION})")
    print(f"[MODEL] Context length: {model.config.context_length}")
    print(f"[MODEL] Prediction length: {model.config.prediction_length}")
    print(f"[MODEL] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


def load_rolling_predictor(base_model):
    """
    Wrap the base model with RecursivePredictor for extended horizon forecasting.

    Args:
        base_model: TinyTimeMixerForPrediction instance

    Returns:
        RecursivePredictor wrapped model
    """
    from tsfm_public.toolkit import RecursivePredictor, RecursivePredictorConfig

    rec_config = RecursivePredictorConfig(
        model=base_model,
        requested_prediction_length=CONFIG.ROLLING_PREDICTION_LENGTH,
        model_prediction_length=base_model.config.prediction_length,
        loss=base_model.config.loss,
    )

    rolling_model = RecursivePredictor(rec_config)

    num_iterations = CONFIG.ROLLING_PREDICTION_LENGTH // CONFIG.MODEL_PREDICTION_LENGTH
    print(f"[ROLLING] Created RecursivePredictor")
    print(f"[ROLLING] Requested prediction length: {CONFIG.ROLLING_PREDICTION_LENGTH}")
    print(f"[ROLLING] Model prediction length: {CONFIG.MODEL_PREDICTION_LENGTH}")
    print(f"[ROLLING] Number of rolling iterations: {num_iterations}")

    return rolling_model
