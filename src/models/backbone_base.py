"""Abstract base class for time series backbone models."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BackboneBase(ABC, nn.Module):
    """
    Abstract base class for time series backbone models.

    All backbone models (TTM, TimesFM, etc.) must inherit from this class
    and implement the abstract methods.

    This ensures:
    1. Consistent interface across different backbones
    2. Easy swapping of foundation models
    3. Uniform freezing and parameter management

    Args:
        checkpoint: Path or identifier for model checkpoint
        num_channels: Number of input channels (e.g., 3 for X, Y, Z)
        context_length: Length of input sequence
        freeze_strategy: Freezing strategy ("none", "all", "embeddings", etc.)

    Example:
        >>> backbone = TTMWrapper(
        >>>     checkpoint="ibm-granite/granite-timeseries-ttm-r2",
        >>>     num_channels=3,
        >>>     context_length=512,
        >>>     freeze_strategy="all"
        >>> )
        >>> output = backbone(x)  # Shape: (B, D) or (B, P, D)
    """

    def __init__(
        self,
        checkpoint: str,
        num_channels: int,
        context_length: int,
        freeze_strategy: str = "none",
    ) -> None:
        """Initialize backbone."""
        super().__init__()

        self.checkpoint = checkpoint
        self.num_channels = num_channels
        self.context_length = context_length
        self.freeze_strategy = freeze_strategy

        logger.info(
            f"Initializing {self.__class__.__name__}:\n"
            f"  Checkpoint: {checkpoint}\n"
            f"  Input: ({num_channels} channels, {context_length} length)\n"
            f"  Freeze strategy: {freeze_strategy}"
        )

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through backbone.

        Args:
            x: Input tensor of shape (B, L, C) where:
               B = batch size
               L = sequence length (context_length)
               C = num_channels

        Returns:
            Output tensor of shape (B, D) or (B, P, D) where:
               D = hidden dimension
               P = number of patches (for patch-based models)

        Raises:
            ValueError: If input shape is invalid
        """
        pass

    @abstractmethod
    def get_output_dim(self) -> int:
        """
        Get output dimension of the backbone.

        Returns:
            Hidden dimension D

        Example:
            >>> backbone = TTMWrapper(...)
            >>> dim = backbone.get_output_dim()
            >>> print(dim)  # e.g., 768
        """
        pass

    @abstractmethod
    def freeze(self, strategy: str) -> None:
        """
        Freeze backbone parameters according to strategy.

        Args:
            strategy: Freezing strategy:
                - "none": Nothing frozen (full fine-tuning)
                - "all": Entire backbone frozen (linear probe)
                - "embeddings": Only freeze embedding layers
                - "time_mixing": Freeze time-mixing layers (if applicable)
                - "channel_mixing": Freeze channel-mixing layers (if applicable)

        Raises:
            ValueError: If strategy is unsupported

        Example:
            >>> backbone.freeze("all")
            >>> # Now only classification head will train
        """
        pass

    def unfreeze_all(self) -> None:
        """
        Unfreeze all backbone parameters.

        Example:
            >>> backbone.freeze("all")
            >>> # ... linear probe training ...
            >>> backbone.unfreeze_all()
            >>> # ... full fine-tuning ...
        """
        for param in self.parameters():
            param.requires_grad = True

        logger.info("Unfroze all backbone parameters")

    def get_num_parameters(self, only_trainable: bool = False) -> int:
        """
        Get number of parameters in backbone.

        Args:
            only_trainable: If True, count only trainable parameters

        Returns:
            Number of parameters

        Example:
            >>> total_params = backbone.get_num_parameters()
            >>> trainable_params = backbone.get_num_parameters(only_trainable=True)
        """
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())

    def get_frozen_status(self) -> Dict[str, bool]:
        """
        Get frozen/unfrozen status of parameter groups.

        Returns:
            Dictionary mapping parameter group names to frozen status

        Example:
            >>> status = backbone.get_frozen_status()
            >>> status
            {'embeddings': True, 'time_mixing': False, ...}
        """
        # Default implementation: just check if any params are frozen
        total_params = self.get_num_parameters()
        trainable_params = self.get_num_parameters(only_trainable=True)

        return {
            "all_frozen": trainable_params == 0,
            "all_unfrozen": trainable_params == total_params,
            "partially_frozen": 0 < trainable_params < total_params,
            "trainable_params": trainable_params,
            "total_params": total_params,
        }

    def validate_input_shape(self, x: torch.Tensor) -> None:
        """
        Validate input tensor shape.

        Args:
            x: Input tensor

        Raises:
            ValueError: If shape is invalid
        """
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3D input (batch, length, channels) in {self.__class__.__name__}.\n"
                f"  Received: {x.dim()}D with shape {x.shape}\n"
                f"  Hint: Reshape to (batch_size, {self.context_length}, {self.num_channels})"
            )

        batch_size, seq_len, num_ch = x.shape

        if seq_len != self.context_length:
            raise ValueError(
                f"Sequence length mismatch in {self.__class__.__name__}.\n"
                f"  Expected: {self.context_length}\n"
                f"  Received: {seq_len}\n"
                f"  Hint: Check preprocessing configuration"
            )

        if num_ch != self.num_channels:
            raise ValueError(
                f"Channel count mismatch in {self.__class__.__name__}.\n"
                f"  Expected: {self.num_channels}\n"
                f"  Received: {num_ch}\n"
                f"  Hint: Check dataset configuration"
            )

    def __repr__(self) -> str:
        """String representation of backbone."""
        trainable = self.get_num_parameters(only_trainable=True)
        total = self.get_num_parameters()

        return (
            f"{self.__class__.__name__}(\n"
            f"  checkpoint={self.checkpoint},\n"
            f"  input_shape=({self.context_length}, {self.num_channels}),\n"
            f"  output_dim={self.get_output_dim()},\n"
            f"  freeze_strategy={self.freeze_strategy},\n"
            f"  trainable_params={trainable:,} / {total:,}\n"
            f")"
        )
