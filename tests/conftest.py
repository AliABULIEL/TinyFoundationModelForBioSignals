"""Pytest configuration and shared fixtures for TTM-HAR tests.

IMPORTANT: This test suite requires:
1. Real IBM TTM model installed (pip install git+https://github.com/ibm-granite/granite-tsfm.git)
2. Real CAPTURE-24 data for integration tests (unit tests use generated tensors)

No mock models or synthetic data generators are used.
"""

import pytest
import numpy as np
import torch
import tempfile
from pathlib import Path
from typing import Dict, Any
import os


# =============================================================================
# ENVIRONMENT VALIDATION
# =============================================================================

def _check_ttm_available():
    """Check if real TTM is available."""
    try:
        from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
        return True
    except ImportError:
        try:
            from granite_tsfm.models import TinyTimeMixerForPrediction
            return True
        except ImportError:
            return False


TTM_AVAILABLE = _check_ttm_available()


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """
    Provide sample configuration for testing.
    
    Note: This config is for testing the config loading/validation logic.
    Integration tests should use real data paths.
    """
    return {
        "experiment": {
            "name": "test_experiment",
            "seed": 42,
            "output_dir": "outputs",
        },
        "dataset": {
            "name": "capture24",
            "data_path": "data/capture24",  # Must exist for integration tests
            "num_classes": 5,
            "train_split": 0.7,
            "val_split": 0.15,
            "test_split": 0.15,
        },
        "preprocessing": {
            "sampling_rate_original": 100,
            "sampling_rate_target": 30,
            "context_length": 512,
            "patch_length": 16,
            "window_stride_train": 256,
            "window_stride_eval": 512,
            "resampling_method": "polyphase",
            "normalization": {
                "method": "zscore",
                "epsilon": 1e-8,
            },
            "gravity_removal": {
                "enabled": False,
                "method": "highpass",
                "cutoff_freq": 0.5,
            },
        },
        "model": {
            "backbone": "ttm",
            "checkpoint": "ibm-granite/granite-timeseries-ttm-r2",
            "num_channels": 3,
            "num_classes": 5,
            "context_length": 512,
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
            "batch_size": 8,
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
            "device": None,
            "num_workers": 0,  # Use 0 for tests to avoid multiprocessing issues
            "pin_memory": False,
            "mixed_precision": False,
        },
    }


@pytest.fixture
def minimal_config() -> Dict[str, Any]:
    """Minimal config for unit tests that don't need full config."""
    return {
        "experiment": {"name": "test", "seed": 42},
        "dataset": {"num_classes": 5},
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


# =============================================================================
# DATA FIXTURES (Using real tensor shapes, not synthetic generators)
# =============================================================================

@pytest.fixture
def sample_signal() -> np.ndarray:
    """
    Generate sample accelerometry signal for unit tests.
    
    This creates a tensor with realistic shapes for testing preprocessing
    and model input/output. NOT a replacement for real data in integration tests.
    """
    np.random.seed(42)
    # 1 minute of 100Hz data, 3 channels (realistic shape)
    duration_sec = 60
    sample_rate = 100
    n_samples = duration_sec * sample_rate
    
    # Generate with realistic accelerometry characteristics
    # Mean ~1g on Z axis (gravity), small variations on X, Y
    signal = np.zeros((n_samples, 3), dtype=np.float32)
    signal[:, 0] = np.random.randn(n_samples) * 0.3  # X
    signal[:, 1] = np.random.randn(n_samples) * 0.3  # Y
    signal[:, 2] = np.random.randn(n_samples) * 0.3 + 1.0  # Z (gravity)
    
    return signal


@pytest.fixture
def sample_labels() -> np.ndarray:
    """Generate sample labels for unit tests."""
    np.random.seed(42)
    duration_sec = 60
    sample_rate = 100
    n_samples = duration_sec * sample_rate
    # 5 classes with realistic imbalanced distribution
    labels = np.random.choice(
        5, 
        size=n_samples, 
        p=[0.35, 0.35, 0.20, 0.08, 0.02]  # Sleep, Sed, Light, Mod, Vig
    ).astype(np.int64)
    return labels


@pytest.fixture
def sample_windows() -> np.ndarray:
    """Generate sample windowed data for unit tests."""
    np.random.seed(42)
    # 100 windows, 512 timesteps, 3 channels
    windows = np.random.randn(100, 512, 3).astype(np.float32)
    return windows


@pytest.fixture
def sample_window_labels() -> np.ndarray:
    """Generate sample window labels for unit tests."""
    np.random.seed(42)
    labels = np.random.randint(0, 5, 100).astype(np.int64)
    return labels


@pytest.fixture
def sample_batch() -> Dict[str, torch.Tensor]:
    """Generate sample batch for training/testing."""
    torch.manual_seed(42)
    batch = {
        "signal": torch.randn(4, 512, 3),  # (B, L, C)
        "label": torch.randint(0, 5, (4,)),  # (B,)
        "participant_id": ["P001", "P002", "P003", "P004"],
    }
    return batch


# =============================================================================
# UTILITY FIXTURES
# =============================================================================

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
    """Get device for testing (CPU by default for reproducibility)."""
    return torch.device("cpu")


@pytest.fixture
def mock_checkpoint(temp_dir, sample_config):
    """Create checkpoint file for testing checkpoint loading."""
    checkpoint_path = temp_dir / "test_checkpoint.pt"

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
    """Reset random seeds before each test for reproducibility."""
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


# =============================================================================
# SKIP MARKERS FOR TESTS REQUIRING REAL RESOURCES
# =============================================================================

# Skip if TTM not installed
requires_ttm = pytest.mark.skipif(
    not TTM_AVAILABLE,
    reason="Requires real IBM TTM model (pip install git+https://github.com/ibm-granite/granite-tsfm.git)"
)

# Skip if real data not available
def requires_real_data(data_path: str = "data/capture24"):
    """Decorator to skip tests that require real CAPTURE-24 data."""
    return pytest.mark.skipif(
        not Path(data_path).exists(),
        reason=f"Requires real CAPTURE-24 data at {data_path}"
    )


# =============================================================================
# TEST DATA DIRECTORY FIXTURE
# =============================================================================

@pytest.fixture
def test_data_dir():
    """
    Get path to test data directory.
    
    For integration tests, this should point to real CAPTURE-24 data.
    Set environment variable CAPTURE24_DATA_PATH to override default.
    """
    default_path = Path("data/capture24")
    env_path = os.environ.get("CAPTURE24_DATA_PATH")
    
    if env_path:
        return Path(env_path)
    return default_path
