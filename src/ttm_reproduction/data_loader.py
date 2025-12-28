"""
Data loading that exactly matches IBM's load_dataset behavior.
"""
import torch
import numpy as np
import pandas as pd
import ssl
import urllib.request
from typing import Tuple, Dict, Any
from transformers import set_seed

from .config import CONFIG

def load_etth1_dataset() -> Tuple[Any, Any, Any]:
    """
    Load ETTh1 dataset using IBM's exact protocol.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)

    Raises:
        AssertionError: If data shapes don't match expected dimensions
    """
    set_seed(CONFIG.SEED)

    # Temporarily disable SSL verification for dataset download
    # This is necessary on some systems (e.g., macOS) where SSL certificates may not be properly configured
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context

    try:
        # Use IBM's official data loader
        from tsfm_public import load_dataset

        dset_train, dset_val, dset_test = load_dataset(
            dataset_name=CONFIG.TARGET_DATASET,
            context_length=CONFIG.CONTEXT_LENGTH,
            forecast_length=CONFIG.ROLLING_PREDICTION_LENGTH,
            fewshot_fraction=1.0,
            dataset_path=CONFIG.DATASET_URL,
        )
    finally:
        # Restore default SSL context
        ssl._create_default_https_context = ssl.create_default_context

    # Shape verification
    # ETTh1 has 7 channels: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
    sample = dset_test[0]
    past_values = sample["past_values"]
    future_values = sample["future_values"]

    assert past_values.shape == (CONFIG.CONTEXT_LENGTH, 7), \
        f"past_values shape mismatch: expected ({CONFIG.CONTEXT_LENGTH}, 7), got {past_values.shape}"
    assert future_values.shape == (CONFIG.ROLLING_PREDICTION_LENGTH, 7), \
        f"future_values shape mismatch: expected ({CONFIG.ROLLING_PREDICTION_LENGTH}, 7), got {future_values.shape}"

    print(f"[DATA] Loaded ETTh1 dataset successfully")
    print(f"[DATA] Test set size: {len(dset_test)}")
    print(f"[DATA] Input shape: {past_values.shape}")
    print(f"[DATA] Target shape: {future_values.shape}")

    return dset_train, dset_val, dset_test


def verify_data_statistics(dataset) -> Dict[str, float]:
    """
    Compute and verify data statistics for debugging.
    """
    all_past = []
    all_future = []

    for i in range(min(100, len(dataset))):
        sample = dataset[i]
        all_past.append(sample["past_values"])
        all_future.append(sample["future_values"])

    past_tensor = np.stack(all_past)
    future_tensor = np.stack(all_future)

    stats = {
        "past_mean": float(np.mean(past_tensor)),
        "past_std": float(np.std(past_tensor)),
        "future_mean": float(np.mean(future_tensor)),
        "future_std": float(np.std(future_tensor)),
    }

    print(f"[DATA STATS] Past: mean={stats['past_mean']:.4f}, std={stats['past_std']:.4f}")
    print(f"[DATA STATS] Future: mean={stats['future_mean']:.4f}, std={stats['future_std']:.4f}")

    return stats
