"""Pytest configuration and shared fixtures for TTM-HAR tests."""

import pytest
import numpy as np
import torch
import tempfile
from pathlib import Path
from typing import Dict, Any

from src.utils.config import load_config


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Provide sample configuration for testing."""
    return {
        "experiment": {
            "name": "test_experiment",
            "seed": 42,
            "output_dir": "outputs",
        },
        "dataset": {
            "name": "capture24",
            "data_path": "data/capture24",  # Fixed: use data_path, not data_dir
            "num_classes": 5,
            "use_synthetic": True,  # Fixed: Enable synthetic data for tests
            "train_split": 0.7,
            "val_split": 0.15,
            "test_split": 0.15,
        },
        "preprocessing": {
            # Fixed: Use the correct structure matching PreprocessingPipeline
            "sampling_rate_original": 100,
            "sampling_rate_target": 30,
            "context_length": 512,  # Must be divisible by patch_length
            "patch_length": 16,
            "window_stride_train": 256,
            "window_stride_eval": 512,
            "resampling_method": "polyphase",
            "normalization": {
                "method": "zscore",
                "epsilon": 1e-8,
            },
            "gravity_removal": {
                "enabled": False,  # Disable for faster tests
                "method": "highpass",
                "cutoff_freq": 0.5,
            },
        },
        "model": {
            "backbone": "ttm",
            "checkpoint": "ibm-granite/granite-timeseries-ttm-r2",
            "num_channels": 3,
            "num_classes": 5,
            "context_length": 512,  # Must be divisible by patch_length (512 / 16 = 32)
            "patch_length": 16,
            "freeze_strategy": "all",
            "head": {
                "type": "linear",
                "pooling": "mean",
                "hidden_dims": None,
                "dropout": 0.1,
                "activation": "gelu",
            },
        },
        "training": {
            "strategy": "linear_probe",
            "epochs": 20,
            "batch_size": 8,  # Reduced for faster tests
            "lr_head": 1e-3,
            "lr_backbone": 1e-5,
            "weight_decay": 0.01,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "warmup_ratio": 0.1,
            "gradient_clip_norm": 1.0,
            "loss": {
                "type": "weighted_ce",
                "label_smoothing": 0.1,
            },
        },
        "hardware": {
            "device": None,  # Auto-select
            "num_workers": 0,  # Use 0 for tests to avoid multiprocessing issues
            "pin_memory": False,
            "mixed_precision": False,
        },
    }


@pytest.fixture
def minimal_config() -> Dict[str, Any]:
    """Minimal config for sanity tests."""
    return {
        "experiment": {"name": "test", "seed": 42},
        "dataset": {"num_classes": 5, "use_synthetic": True},
        "preprocessing": {
            "sampling_rate_original": 100,
            "sampling_rate_target": 30,
            "context_length": 512,
            "patch_length": 16,
            "window_stride_train": 256,
            "normalization": {"method": "zscore"},
        },
        "model": {
            "backbone": "ttm",
            "num_channels": 3,
            "num_classes": 5,
            "context_length": 512,
            "patch_length": 16,
            "freeze_strategy": "all",
            "head": {"type": "linear", "dropout": 0.0},
        },
        "training": {
            "strategy": "linear_probe",
            "batch_size": 4,
            "lr_head": 1e-3,
        },
        "hardware": {"num_workers": 0, "mixed_precision": False},
    }


@pytest.fixture
def sample_signal() -> np.ndarray:
    """Generate sample accelerometry signal."""
    np.random.seed(42)
    # 1 minute of 100Hz data, 3 channels
    duration_sec = 60
    sample_rate = 100
    n_samples = duration_sec * sample_rate
    signal = np.random.randn(n_samples, 3).astype(np.float32)
    return signal


@pytest.fixture
def sample_labels() -> np.ndarray:
    """Generate sample labels."""
    np.random.seed(42)
    duration_sec = 60
    sample_rate = 100
    n_samples = duration_sec * sample_rate
    # 5 classes
    labels = np.random.randint(0, 5, n_samples).astype(np.int64)
    return labels


@pytest.fixture
def sample_windows() -> np.ndarray:
    """Generate sample windowed data."""
    np.random.seed(42)
    # 100 windows, 512 timesteps, 3 channels
    windows = np.random.randn(100, 512, 3).astype(np.float32)
    return windows


@pytest.fixture
def sample_window_labels() -> np.ndarray:
    """Generate sample window labels."""
    np.random.seed(42)
    # 100 labels (one per window)
    labels = np.random.randint(0, 5, 100).astype(np.int64)
    return labels


@pytest.fixture
def sample_batch() -> Dict[str, torch.Tensor]:
    """Generate sample batch for training/testing."""
    torch.manual_seed(42)
    batch = {
        "signal": torch.randn(4, 512, 3),  # (B, L, C)
        "label": torch.randint(0, 5, (4,)),  # (B,)
        "participant_id": ["P001", "P002", "P003", "P004"],  # Added for completeness
    }
    return batch


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_config_file(temp_dir, sample_config):
    """Create temporary config file."""
    import yaml

    config_path = temp_dir / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.safe_dump(sample_config, f)

    return config_path


@pytest.fixture
def device() -> torch.device:
    """Get device for testing (CPU by default)."""
    return torch.device("cpu")


@pytest.fixture
def mock_checkpoint(temp_dir, sample_config):
    """Create mock checkpoint file."""
    checkpoint_path = temp_dir / "mock_checkpoint.pt"

    # Create minimal checkpoint
    checkpoint = {
        "model_state_dict": {},
        "optimizer_state_dict": {},
        "epoch": 10,
        "global_step": 1000,
        "config": sample_config,
    }

    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def label_map() -> Dict[int, str]:
    """Standard CAPTURE-24 5-class label map."""
    return {
        0: "Sleep",
        1: "Sedentary",
        2: "Light",
        3: "Moderate",
        4: "Vigorous",
    }
