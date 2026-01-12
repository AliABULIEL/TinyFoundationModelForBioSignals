"""Model factory for creating complete classification models.

This module provides factory functions for creating HAR models.
Only real IBM TTM models are supported - no mocks allowed.
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.models.backbone_base import BackboneBase
from src.models.heads import get_classification_head
from src.models.ttm_wrapper import TTMWrapper

logger = logging.getLogger(__name__)


class HARModel(nn.Module):
    """
    Complete Human Activity Recognition model.

    Combines a backbone (feature extractor) with a classification head.

    Architecture: input → backbone → features → head → logits

    Args:
        backbone: Backbone model for feature extraction
        head: Classification head

    Example:
        >>> backbone = TTMWrapper(...)
        >>> head = LinearHead(input_dim=768, num_classes=5)
        >>> model = HARModel(backbone, head)
        >>> logits = model(x)
    """

    def __init__(
        self,
        backbone: BackboneBase,
        head: nn.Module,
    ) -> None:
        """Initialize HAR model."""
        super().__init__()

        self.backbone = backbone
        self.head = head

        # Validate dimensions match
        backbone_dim = backbone.get_output_dim()
        head_input_dim = head.input_dim

        if backbone_dim != head_input_dim:
            raise ValueError(
                f"Dimension mismatch between backbone and head.\n"
                f"  Backbone output: {backbone_dim}\n"
                f"  Head input: {head_input_dim}\n"
                f"  Hint: Check model configuration"
            )

        logger.info(
            f"Created HARModel:\n"
            f"  Backbone: {backbone.__class__.__name__} (output_dim={backbone_dim})\n"
            f"  Head: {head.__class__.__name__} (num_classes={head.num_classes})\n"
            f"  Total params: {self.get_num_parameters():,}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through model.

        Args:
            x: Input tensor of shape (B, L, C)

        Returns:
            Logits of shape (B, K)
        """
        # Extract features
        features = self.backbone(x)  # (B, D)

        # Classify
        logits = self.head(features)  # (B, K)

        return logits

    def freeze_backbone(self, strategy: str = "all") -> None:
        """
        Freeze backbone parameters.

        Args:
            strategy: Freezing strategy
        """
        self.backbone.freeze(strategy)

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters."""
        self.backbone.unfreeze_all()

    def get_num_parameters(self, only_trainable: bool = False) -> int:
        """Get number of parameters."""
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())

    def get_parameter_groups(self) -> Dict[str, list]:
        """
        Get parameter groups for optimizer.

        Returns:
            Dictionary with 'backbone' and 'head' parameter lists

        Example:
            >>> groups = model.get_parameter_groups()
            >>> optimizer = torch.optim.AdamW([
            >>>     {'params': groups['backbone'], 'lr': 1e-5},
            >>>     {'params': groups['head'], 'lr': 1e-3},
            >>> ])
        """
        return {
            "backbone": list(self.backbone.parameters()),
            "head": list(self.head.parameters()),
        }


def create_model(config: Dict[str, Any]) -> HARModel:
    """
    Create complete HAR model from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        HARModel instance

    Raises:
        ValueError: If configuration is invalid
        ImportError: If TTM is not installed

    Example:
        >>> config = load_config("configs/default.yaml")
        >>> model = create_model(config)
    """
    model_config = config.get("model", {})
    preprocessing_config = config.get("preprocessing", {})

    # Extract parameters
    backbone_type = model_config.get("backbone", "ttm").lower()
    checkpoint = model_config.get("checkpoint", "ibm-granite/granite-timeseries-ttm-r2")
    num_channels = model_config.get("num_channels", 3)
    context_length = model_config.get("context_length", 512)
    patch_length = model_config.get("patch_length", 16)
    num_classes = model_config.get("num_classes", 5)
    freeze_strategy = model_config.get("freeze_strategy", "all")

    # Head configuration
    head_config = model_config.get("head", {})
    head_type = head_config.get("type", "linear")
    pooling = head_config.get("pooling", "mean")
    hidden_dims = head_config.get("hidden_dims", None)
    dropout = head_config.get("dropout", 0.1)
    activation = head_config.get("activation", "gelu")

    logger.info("Creating model from configuration...")

    # Create backbone
    if backbone_type == "ttm":
        backbone = TTMWrapper(
            checkpoint=checkpoint,
            num_channels=num_channels,
            context_length=context_length,
            patch_length=patch_length,
            freeze_strategy=freeze_strategy,
            use_pretrained=True,
        )
    else:
        raise ValueError(
            f"Unknown backbone type: {backbone_type}\n"
            f"  Supported: ['ttm']\n"
            f"  Hint: Only real IBM TTM backbone is supported"
        )

    # Get backbone output dimension
    backbone_output_dim = backbone.get_output_dim()

    # Create classification head
    head = get_classification_head(
        head_type=head_type,
        input_dim=backbone_output_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        dropout=dropout,
        activation=activation,
    )

    # Create complete model
    model = HARModel(backbone=backbone, head=head)

    logger.info(f"✓ Model created successfully")
    logger.info(f"  Backbone params: {backbone.get_num_parameters():,}")
    logger.info(f"  Head params: {sum(p.numel() for p in head.parameters()):,}")
    logger.info(f"  Total params: {model.get_num_parameters():,}")
    logger.info(f"  Trainable params: {model.get_num_parameters(only_trainable=True):,}")

    return model


def load_model_from_checkpoint(
    checkpoint_path: str,
    config: Dict[str, Any],
    device: Optional[torch.device] = None,
) -> HARModel:
    """
    Load model from checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration dictionary
        device: Device to load model to

    Returns:
        HARModel with loaded weights

    Example:
        >>> model = load_model_from_checkpoint(
        >>>     "checkpoints/best.pt",
        >>>     config,
        >>>     device=torch.device("cuda")
        >>> )
    """
    from src.utils.checkpointing import load_checkpoint

    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path, device=device)

    # Create model from config
    model = create_model(config)

    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])

    logger.info(f"Loaded model from checkpoint: {checkpoint_path}")
    logger.info(f"  Epoch: {checkpoint['epoch']}")
    logger.info(f"  Best metric: {checkpoint.get('best_metric', 'N/A')}")

    return model
