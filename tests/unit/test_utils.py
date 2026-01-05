"""Unit tests for utility modules."""

import pytest
import numpy as np
import torch
from pathlib import Path

from src.utils.config import (
    load_config,
    merge_configs,
    merge_config_overrides,
    parse_override_value,
    set_nested_value,
    get_nested_value,
)
from src.utils.reproducibility import set_seed
from src.utils.device import get_device
from src.utils.checkpointing import save_checkpoint, load_checkpoint


@pytest.mark.unit
class TestConfig:
    """Tests for configuration utilities."""

    def test_load_config(self, temp_config_file):
        """Test loading configuration from file."""
        config = load_config(temp_config_file)

        assert isinstance(config, dict)
        assert "experiment" in config
        assert "model" in config

    def test_load_config_nonexistent(self):
        """Test loading non-existent config file raises error."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")

    def test_merge_configs(self):
        """Test merging multiple configs."""
        base = {
            "model": {"hidden_dim": 128},
            "training": {"lr": 0.001, "epochs": 10},
        }

        override = {
            "training": {"lr": 0.0001, "batch_size": 64},
            "new_key": "value",
        }

        merged = merge_configs(base, override)

        # Check merged values
        assert merged["model"]["hidden_dim"] == 128  # From base
        assert merged["training"]["lr"] == 0.0001  # Overridden
        assert merged["training"]["epochs"] == 10  # From base
        assert merged["training"]["batch_size"] == 64  # From override
        assert merged["new_key"] == "value"  # From override

    def test_parse_override_value(self):
        """Test parsing override values from strings."""
        # Boolean
        assert parse_override_value("true") is True
        assert parse_override_value("false") is False

        # Integer
        assert parse_override_value("42") == 42

        # Float
        assert parse_override_value("3.14") == 3.14

        # List
        assert parse_override_value("1,2,3") == [1, 2, 3]

        # String
        assert parse_override_value("hello") == "hello"

    def test_merge_config_overrides(self):
        """Test merging command-line overrides."""
        config = {
            "training": {"epochs": 10, "lr": 0.001},
            "model": {"num_classes": 5},
        }

        overrides = [
            "training.epochs=100",
            "training.lr=0.0001",
            "model.dropout=0.5",
        ]

        merged = merge_config_overrides(config, overrides)

        assert merged["training"]["epochs"] == 100
        assert merged["training"]["lr"] == 0.0001
        assert merged["model"]["dropout"] == 0.5

    def test_get_nested_value(self):
        """Test getting nested values."""
        config = {
            "model": {
                "backbone": {
                    "type": "ttm",
                    "hidden_dim": 128,
                },
            },
        }

        # Get nested value
        value = get_nested_value(config, "model.backbone.hidden_dim")
        assert value == 128

        # Get with default
        value = get_nested_value(config, "model.missing.key", default=42)
        assert value == 42

    def test_set_nested_value(self):
        """Test setting nested values."""
        config = {}

        # Set nested value (creates intermediate dicts)
        set_nested_value(config, "model.backbone.hidden_dim", 256)

        assert config["model"]["backbone"]["hidden_dim"] == 256


@pytest.mark.unit
class TestReproducibility:
    """Tests for reproducibility utilities."""

    def test_set_seed_numpy(self):
        """Test that setting seed makes numpy reproducible."""
        set_seed(42)
        val1 = np.random.rand()

        set_seed(42)
        val2 = np.random.rand()

        assert val1 == val2

    def test_set_seed_torch(self):
        """Test that setting seed makes torch reproducible."""
        set_seed(42)
        val1 = torch.rand(1).item()

        set_seed(42)
        val2 = torch.rand(1).item()

        assert val1 == val2


@pytest.mark.unit
class TestDevice:
    """Tests for device management."""

    def test_get_device(self):
        """Test getting device."""
        device = get_device()

        assert isinstance(device, torch.device)
        # Should be either cpu or cuda
        assert device.type in ["cpu", "cuda"]


@pytest.mark.unit
class TestCheckpointing:
    """Tests for checkpointing utilities."""

    def test_save_and_load_checkpoint(self, temp_dir):
        """Test saving and loading checkpoint."""
        # Create checkpoint with all required fields
        checkpoint = {
            "model_state_dict": {"weight": torch.randn(10, 5)},
            "optimizer_state_dict": {},
            "epoch": 10,
            "global_step": 1000,
            "config": {"model": "test"},
        }

        checkpoint_path = temp_dir / "test_checkpoint.pt"

        # Save
        save_checkpoint(checkpoint, str(checkpoint_path))

        assert checkpoint_path.exists()

        # Load
        loaded = load_checkpoint(str(checkpoint_path), device=torch.device("cpu"))

        # Check contents
        assert loaded["epoch"] == 10
        assert loaded["global_step"] == 1000
        assert "model_state_dict" in loaded
        assert "config" in loaded

    def test_save_checkpoint_with_best(self, temp_dir):
        """Test saving best checkpoint."""
        checkpoint = {
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "epoch": 10,
            "global_step": 100,
            "config": {"model": "test"},
        }

        checkpoint_path = temp_dir / "checkpoint.pt"
        best_path = temp_dir / "checkpoint_best.pt"  # Uses _best suffix

        # Save with is_best=True
        save_checkpoint(checkpoint, str(checkpoint_path), is_best=True)

        # Both files should exist
        assert checkpoint_path.exists()
        assert best_path.exists()

    def test_load_checkpoint_nonexistent(self):
        """Test loading non-existent checkpoint raises error."""
        with pytest.raises(FileNotFoundError):
            load_checkpoint("nonexistent.pt")
