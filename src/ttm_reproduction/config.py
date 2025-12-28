"""
TTM Reproduction Configuration
All constants must match IBM's ttm_rolling_prediction_getting_started.ipynb exactly.
"""
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class TTMReproductionConfig:
    # Reproducibility
    SEED: int = 42

    # Model configuration (TTM-B 512-96 variant)
    MODEL_PATH: str = "ibm-granite/granite-timeseries-ttm-r2"
    MODEL_REVISION: str = "main"  # 512-96 variant
    CONTEXT_LENGTH: int = 512
    MODEL_PREDICTION_LENGTH: int = 96

    # Evaluation configuration
    TARGET_DATASET: str = "etth1"
    DATASET_URL: str = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"

    # Rolling prediction
    ROLLING_PREDICTION_LENGTH: int = 192

    # Evaluation batch size
    BATCH_SIZE: int = 32

    # Verification thresholds
    # IBM reports ~0.39 MSE for zero-shot 512-192 on ETTh1
    EXPECTED_MSE_LOWER: float = 0.35
    EXPECTED_MSE_UPPER: float = 0.45
    MSE_TOLERANCE_PERCENT: float = 5.0  # Allow 5% deviation from expected range midpoint

    # Device
    DEVICE: Literal["cuda", "cpu"] = "cuda"

CONFIG = TTMReproductionConfig()
