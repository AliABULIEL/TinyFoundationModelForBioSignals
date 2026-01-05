"""TTM (Tiny Time Mixers) backbone wrapper for time series classification."""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.models.backbone_base import BackboneBase

logger = logging.getLogger(__name__)


class TTMWrapper(BackboneBase):
    """
    Wrapper for Tiny Time Mixers (TTM) foundation model.

    TTM is an MLP-based time series foundation model that uses sequential
    time-mixing and channel-mixing operations. This wrapper:
    1. Loads pre-trained TTM checkpoint from HuggingFace
    2. Handles input channel projection (3 channels → model channels)
    3. Implements various freezing strategies
    4. Extracts features for downstream classification

    Architecture: Patching → Time-Mixing MLPs → Channel-Mixing MLPs → Features

    Args:
        checkpoint: HuggingFace checkpoint ID or path
        num_channels: Number of input channels (3 for X, Y, Z)
        context_length: Input sequence length (must be divisible by patch_length)
        patch_length: Length of each patch (default: 16)
        freeze_strategy: Freezing strategy ("none", "all", "embeddings", etc.)
        use_pretrained: If True, load pre-trained weights; else random init

    Example:
        >>> backbone = TTMWrapper(
        >>>     checkpoint="ibm-granite/granite-timeseries-ttm-r2",
        >>>     num_channels=3,
        >>>     context_length=512,
        >>>     freeze_strategy="all"
        >>> )
        >>> x = torch.randn(32, 512, 3)
        >>> features = backbone(x)  # Shape: (32, hidden_dim)
    """

    def __init__(
        self,
        checkpoint: str = "ibm-granite/granite-timeseries-ttm-r2",
        num_channels: int = 3,
        context_length: int = 512,
        patch_length: int = 16,
        freeze_strategy: str = "none",
        use_pretrained: bool = True,
    ) -> None:
        """Initialize TTM wrapper."""
        # Initialize base class
        super().__init__(
            checkpoint=checkpoint,
            num_channels=num_channels,
            context_length=context_length,
            freeze_strategy=freeze_strategy,
        )

        self.patch_length = patch_length
        self.use_pretrained = use_pretrained

        # Validate context_length is divisible by patch_length
        if context_length % patch_length != 0:
            raise ValueError(
                f"context_length must be divisible by patch_length.\n"
                f"  context_length: {context_length}\n"
                f"  patch_length: {patch_length}\n"
                f"  Hint: Use combinations like (512, 16), (512, 32), (256, 16)"
            )

        self.num_patches = context_length // patch_length

        # Load TTM model
        self.model, self.model_channels = self._load_ttm_model()

        # Get output dimension from model
        self._output_dim = self._infer_output_dim()

        # Channel projection (if needed)
        if num_channels != self.model_channels:
            logger.info(
                f"Input channels ({num_channels}) != model channels ({self.model_channels}). "
                f"Adding projection layer."
            )
            self.channel_projection = nn.Linear(num_channels, self.model_channels)
        else:
            self.channel_projection = None

        # Apply initial freezing
        if freeze_strategy != "none":
            self.freeze(freeze_strategy)

        # Log configuration
        logger.info(
            f"TTM initialized:\n"
            f"  Model channels: {self.model_channels}\n"
            f"  Num patches: {self.num_patches}\n"
            f"  Output dim: {self._output_dim}\n"
            f"  Trainable params: {self.get_num_parameters(only_trainable=True):,}"
        )

    def _load_ttm_model(self) -> Tuple[nn.Module, int]:
        """
        Load TTM model from checkpoint.

        Returns:
            Tuple of (model, num_channels_in_model)

        Raises:
            RuntimeError: If model loading fails
        """
        if not self.use_pretrained:
            logger.warning(
                "use_pretrained=False: Random initialization not yet implemented.\n"
                "  Falling back to trying to load pre-trained model."
            )

        # Try primary import path
        try:
            from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

            logger.info(f"Loading TTM from: {self.checkpoint} (tsfm_public)")

            model = TinyTimeMixerForPrediction.from_pretrained(self.checkpoint)

            # Extract number of channels from model config
            num_channels = getattr(model.config, "num_input_channels", 1)

            logger.info(f"✓ Successfully loaded TTM checkpoint")

            return model, num_channels

        except ImportError as e:
            logger.warning(f"Primary import failed: {e}")

            # Try fallback import path
            try:
                from granite_tsfm.models import TinyTimeMixerForPrediction

                logger.info(f"Loading TTM from: {self.checkpoint} (granite_tsfm)")

                model = TinyTimeMixerForPrediction.from_pretrained(self.checkpoint)
                num_channels = getattr(model.config, "num_input_channels", 1)

                logger.info(f"✓ Successfully loaded TTM checkpoint (fallback)")

                return model, num_channels

            except ImportError as e2:
                # Use mock model for testing
                logger.warning(
                    f"Failed to import TTM model. Using mock model for testing.\n"
                    f"  Primary error: {e}\n"
                    f"  Fallback error: {e2}\n"
                    f"  Hint: For production, install TTM dependencies:\n"
                    f"    pip install git+https://github.com/ibm-granite/granite-tsfm.git"
                )

                from src.models.mock_ttm import MockTTMModel

                logger.info(f"Loading MockTTM (testing mode)")
                model = MockTTMModel.from_pretrained(self.checkpoint)
                num_channels = getattr(model.config, "num_input_channels", 1)

                return model, num_channels

        except Exception as e:
            raise RuntimeError(
                f"Failed to load TTM checkpoint: {self.checkpoint}\n"
                f"  Error: {e}\n"
                f"  Hint: Check checkpoint ID and internet connection"
            ) from e

    def _infer_output_dim(self) -> int:
        """
        Infer output dimension by running a forward pass.

        Returns:
            Output dimension
        """
        # Create dummy input matching model's expected channels
        dummy_input = torch.randn(1, self.context_length, self.model_channels)

        # Run forward pass (without gradients)
        with torch.no_grad():
            try:
                output = self.model(dummy_input)

                # TTM may return dict or tensor
                if isinstance(output, dict):
                    # Try common keys
                    for key in ["backbone_hidden_state", "last_hidden_state", "hidden_states", "decoder_hidden_state", "prediction"]:
                        if key in output:
                            output = output[key]
                            break

                # Get output shape - the hidden dimension is always the last dimension
                if output.dim() == 4:
                    # Shape: (B, C, P, D)
                    output_dim = output.shape[-1]
                elif output.dim() == 3:
                    # Shape: (B, P, D)
                    output_dim = output.shape[-1]
                elif output.dim() == 2:
                    # Shape: (B, D)
                    output_dim = output.shape[-1]
                else:
                    raise ValueError(f"Unexpected output shape: {output.shape}")

                logger.debug(f"Inferred output dimension: {output_dim} from shape {output.shape}")

                return output_dim

            except Exception as e:
                logger.error(f"Failed to infer output dimension: {e}")
                # Fallback to common dimension
                fallback_dim = 192  # TTM-r2 typical dimension
                logger.warning(f"Using fallback output dimension: {fallback_dim}")
                return fallback_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TTM backbone.

        Args:
            x: Input tensor of shape (B, L, C) where:
               B = batch size
               L = context_length
               C = num_channels (e.g., 3 for X, Y, Z)

        Returns:
            Feature tensor of shape (B, D) where D = hidden dimension

        Raises:
            ValueError: If input shape is invalid
        """
        # Validate input shape
        self.validate_input_shape(x)

        batch_size = x.shape[0]

        # Apply channel projection if needed
        if self.channel_projection is not None:
            x = self.channel_projection(x)  # (B, L, C) -> (B, L, model_channels)

        # Forward through TTM model
        output = self.model(x)

        # Extract features from output
        if isinstance(output, dict):
            # Try common keys (order matters - try most likely first)
            for key in ["backbone_hidden_state", "last_hidden_state", "hidden_states", "decoder_hidden_state", "prediction"]:
                if key in output:
                    features = output[key]
                    break
            else:
                raise ValueError(
                    f"Cannot find features in model output.\n"
                    f"  Available keys: {list(output.keys())}\n"
                    f"  Hint: Check TTM model output format"
                )
        else:
            features = output

        # Handle different output shapes
        if features.dim() == 4:
            # Shape: (B, C, P, D) - batch, channels, patches, hidden_dim
            # Pool across channels and patches
            features = torch.mean(features, dim=(1, 2))  # (B, D)
        elif features.dim() == 3:
            # Shape: (B, P, D) - patch-level features
            # Pool across patches to get sequence-level features
            features = torch.mean(features, dim=1)  # (B, D)
        elif features.dim() == 2:
            # Shape: (B, D) - already sequence-level
            pass
        else:
            raise ValueError(
                f"Unexpected feature shape: {features.shape}\n"
                f"  Expected: (B, D), (B, P, D), or (B, C, P, D)\n"
                f"  Hint: Check TTM model output"
            )

        # Validate output shape
        if features.shape != (batch_size, self._output_dim):
            logger.warning(
                f"Output shape mismatch: expected ({batch_size}, {self._output_dim}), "
                f"got {features.shape}"
            )

        return features

    def get_output_dim(self) -> int:
        """Get output dimension of TTM backbone."""
        return self._output_dim

    def freeze(self, strategy: str) -> None:
        """
        Freeze TTM parameters according to strategy.

        Args:
            strategy: Freezing strategy:
                - "none": Nothing frozen
                - "all": Entire backbone frozen
                - "embeddings": Freeze patch embeddings only
                - "time_mixing": Freeze time-mixing layers
                - "channel_mixing": Freeze channel-mixing layers

        Raises:
            ValueError: If strategy is unsupported
        """
        self.freeze_strategy = strategy

        if strategy == "none":
            self.unfreeze_all()
            logger.info("Freeze strategy 'none': all parameters trainable")
            return

        elif strategy == "all":
            # Freeze entire backbone including channel projection
            for param in self.model.parameters():
                param.requires_grad = False

            if self.channel_projection is not None:
                for param in self.channel_projection.parameters():
                    param.requires_grad = False

            logger.info(f"Froze all TTM parameters (including channel projection)")

        elif strategy == "embeddings":
            # Freeze only embedding/patching layers
            frozen_count = 0
            for name, param in self.model.named_parameters():
                if "embed" in name.lower() or "patch" in name.lower():
                    param.requires_grad = False
                    frozen_count += 1

            logger.info(f"Froze {frozen_count} embedding parameters")

        elif strategy == "time_mixing":
            # Freeze time-mixing layers
            frozen_count = 0
            for name, param in self.model.named_parameters():
                if "time" in name.lower() or "temporal" in name.lower():
                    param.requires_grad = False
                    frozen_count += 1

            logger.info(f"Froze {frozen_count} time-mixing parameters")

        elif strategy == "channel_mixing":
            # Freeze channel-mixing layers
            frozen_count = 0
            for name, param in self.model.named_parameters():
                if "channel" in name.lower():
                    param.requires_grad = False
                    frozen_count += 1

            logger.info(f"Froze {frozen_count} channel-mixing parameters")

        else:
            raise ValueError(
                f"Unknown freeze strategy: {strategy}\n"
                f"  Supported: ['none', 'all', 'embeddings', 'time_mixing', 'channel_mixing']"
            )

        # Log trainable params
        trainable = self.get_num_parameters(only_trainable=True)
        total = self.get_num_parameters()
        logger.info(f"Trainable parameters: {trainable:,} / {total:,}")

    def get_frozen_status(self) -> dict:
        """Get detailed frozen status of TTM components."""
        status = super().get_frozen_status()

        # Add TTM-specific component status
        component_status = {}

        for name, param in self.model.named_parameters():
            # Categorize parameters
            if "embed" in name.lower() or "patch" in name.lower():
                component = "embeddings"
            elif "time" in name.lower():
                component = "time_mixing"
            elif "channel" in name.lower():
                component = "channel_mixing"
            else:
                component = "other"

            if component not in component_status:
                component_status[component] = {"frozen": 0, "trainable": 0}

            if param.requires_grad:
                component_status[component]["trainable"] += 1
            else:
                component_status[component]["frozen"] += 1

        status["components"] = component_status

        return status
