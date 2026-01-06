"""Mock TTM model for testing without TTM dependencies.

╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  ⚠️  ⚠️  ⚠️              TEST ONLY - NOT FOR PRODUCTION            ⚠️  ⚠️  ⚠️   ║
║                                                                              ║
║  This is a MOCK model that simulates TTM behavior for testing purposes.     ║
║  It should NEVER be used in production or for real research.                ║
║                                                                              ║
║  To enable this mock (tests only), set environment variable:                ║
║      export TTM_HAR_ALLOW_MOCK=1                                            ║
║                                                                              ║
║  For production, install the real IBM TTM model:                            ║
║      pip install git+https://github.com/ibm-granite/granite-tsfm.git        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import logging
import os
import warnings

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
    Mock TTM model for testing ONLY.

    ⚠️  WARNING: This is a simplified mock that does NOT replicate real TTM behavior.
    It's only for testing the pipeline without requiring TTM installation.

    Simulates TTM architecture without requiring the actual package.
    Useful for testing the wrapper and pipeline structure.

    Raises:
        RuntimeError: If TTM_HAR_ALLOW_MOCK environment variable is not set to "1"
    """

    def __init__(self, config: MockTTMConfig):
        # ⚠️ CRITICAL: Check environment variable before allowing instantiation
        if os.environ.get("TTM_HAR_ALLOW_MOCK") != "1":
            raise RuntimeError(
                f"\n{'=' * 80}\n"
                f"❌ MOCK MODEL BLOCKED - PRODUCTION USE NOT ALLOWED\n"
                f"{'=' * 80}\n\n"
                f"MockTTMModel is ONLY for testing and should NOT be used in production.\n\n"
                f"You attempted to instantiate a mock model without explicit permission.\n"
                f"This is blocked to prevent accidental use of non-functional models.\n\n"
                f"FOR TESTING PURPOSES ONLY:\n"
                f"  Set environment variable before running tests:\n"
                f"    export TTM_HAR_ALLOW_MOCK=1\n"
                f"  Or in Python:\n"
                f"    import os\n"
                f"    os.environ['TTM_HAR_ALLOW_MOCK'] = '1'\n\n"
                f"FOR PRODUCTION USE:\n"
                f"  Install the REAL IBM TTM model:\n"
                f"    pip install git+https://github.com/ibm-granite/granite-tsfm.git\n\n"
                f"IMPORTANT:\n"
                f"  • Mock models do NOT provide real TTM functionality\n"
                f"  • Mock models do NOT use pre-trained weights\n"
                f"  • Results from mock models are MEANINGLESS for research\n"
                f"  • This guard prevents silent degradation to mock models\n"
                f"{'=' * 80}\n"
            )

        super().__init__()
        self.config = config

        # Emit a loud warning every time mock is used
        warnings.warn(
            "\n"
            "=" * 80 + "\n"
            "⚠️  WARNING: Using MockTTMModel instead of real IBM TTM!\n"
            "=" * 80 + "\n"
            "This is a MOCK model for TESTING ONLY.\n"
            "Results are NOT scientifically valid.\n"
            "For production, install real TTM:\n"
            "  pip install git+https://github.com/ibm-granite/granite-tsfm.git\n"
            "=" * 80,
            UserWarning,
            stacklevel=2
        )

        logger.warning(
            f"🚨 MockTTMModel instantiated - THIS IS FOR TESTING ONLY!\n"
            f"  Environment: TTM_HAR_ALLOW_MOCK={os.environ.get('TTM_HAR_ALLOW_MOCK')}\n"
            f"  Config: {config.num_input_channels} input channels, {config.hidden_dim} hidden dim"
        )

        # Simple linear projection to simulate TTM
        self.projection = nn.Linear(config.num_input_channels, config.hidden_dim)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass returning dict like real TTM.

        ⚠️  This is a MOCK forward pass that does NOT replicate TTM behavior!
        """
        # x shape: (B, L, C)
        batch_size, seq_len, num_channels = x.shape

        # Project channels (this is NOT what real TTM does!)
        hidden = self.projection(x)  # (B, L, hidden_dim)

        # Return in TTM format
        return {"last_hidden_state": hidden}

    @classmethod
    def from_pretrained(cls, checkpoint_id: str):
        """
        Mock from_pretrained method.

        ⚠️  WARNING: This does NOT load real pre-trained weights!
        It creates a randomly initialized model.

        Args:
            checkpoint_id: Ignored (mock doesn't use real checkpoints)

        Returns:
            MockTTMModel instance (randomly initialized)

        Raises:
            RuntimeError: If TTM_HAR_ALLOW_MOCK != "1"
        """
        logger.warning(
            f"\n{'=' * 80}\n"
            f"⚠️  MockTTMModel.from_pretrained() called - TESTING MODE\n"
            f"{'=' * 80}\n"
            f"  Requested checkpoint: {checkpoint_id}\n"
            f"  Actual behavior: Creating RANDOM weights (NOT pre-trained!)\n"
            f"  Environment: TTM_HAR_ALLOW_MOCK={os.environ.get('TTM_HAR_ALLOW_MOCK')}\n\n"
            f"This is expected ONLY for testing without TTM dependencies.\n"
            f"For production, install real TTM:\n"
            f"  pip install git+https://github.com/ibm-granite/granite-tsfm.git\n"
            f"{'=' * 80}"
        )

        config = MockTTMConfig(num_input_channels=1, hidden_dim=768)
        return cls(config)
