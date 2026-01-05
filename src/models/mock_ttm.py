"""Mock TTM model for testing without TTM dependencies."""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MockTTMConfig:
    """Mock configuration for TTM model."""

    def __init__(self, num_input_channels: int = 1, hidden_dim: int = 768):
        self.num_input_channels = num_input_channels
        self.hidden_dim = hidden_dim


class MockTTMModel(nn.Module):
    """
    Mock TTM model for testing.

    Simulates TTM architecture without requiring the actual package.
    Useful for testing the wrapper and pipeline.
    """

    def __init__(self, config: MockTTMConfig):
        super().__init__()
        self.config = config

        # Simple linear projection to simulate TTM
        self.projection = nn.Linear(config.num_input_channels, config.hidden_dim)

    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass returning dict like real TTM."""
        # x shape: (B, L, C)
        batch_size, seq_len, num_channels = x.shape

        # Project channels
        hidden = self.projection(x)  # (B, L, hidden_dim)

        # Return in TTM format
        return {"last_hidden_state": hidden}

    @classmethod
    def from_pretrained(cls, checkpoint_id: str):
        """Mock from_pretrained method."""
        logger.warning(
            f"Using MockTTMModel instead of real TTM for testing.\n"
            f"  Requested checkpoint: {checkpoint_id}\n"
            f"  This is expected if TTM dependencies are not installed."
        )

        config = MockTTMConfig(num_input_channels=1, hidden_dim=768)
        return cls(config)
