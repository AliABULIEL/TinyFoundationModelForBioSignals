"""Loss functions for activity recognition."""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross-Entropy Loss for handling class imbalance.

    Applies different weights to each class based on their frequency.
    Higher weights for rare classes encourage the model to focus on them.

    Args:
        class_weights: Tensor of class weights of shape (num_classes,)
        reduction: Reduction method ("mean", "sum", "none")

    Example:
        >>> weights = torch.tensor([1.0, 1.0, 3.5, 8.0, 15.0])  # Sleep to Vigorous
        >>> loss_fn = WeightedCrossEntropyLoss(weights)
        >>> logits = torch.randn(32, 5)
        >>> labels = torch.randint(0, 5, (32,))
        >>> loss = loss_fn(logits, labels)
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        """Initialize weighted cross-entropy loss."""
        super().__init__()

        self.class_weights = class_weights
        self.reduction = reduction

        if class_weights is not None:
            logger.debug(f"Initialized WeightedCrossEntropyLoss with weights: {class_weights}")
        else:
            logger.debug("Initialized WeightedCrossEntropyLoss (no weights)")

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted cross-entropy loss.

        Args:
            logits: Predicted logits of shape (B, K)
            labels: Ground truth labels of shape (B,)

        Returns:
            Scalar loss value (if reduction="mean" or "sum")
            or per-sample losses of shape (B,) (if reduction="none")
        """
        # Move weights to same device as logits
        weights = self.class_weights
        if weights is not None:
            weights = weights.to(logits.device)

        # Use PyTorch's built-in weighted CE
        loss = F.cross_entropy(
            logits,
            labels,
            weight=weights,
            reduction=self.reduction,
        )

        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing extreme class imbalance.

    Focal Loss down-weights easy examples and focuses on hard negatives.
    This is particularly useful when one class is extremely rare.

    Reference: Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)

    Args:
        alpha: Weighting factor in [0, 1] for balancing classes, or
               tensor of per-class weights
        gamma: Focusing parameter (default: 2.0). Higher values focus
               more on hard examples
        reduction: Reduction method ("mean", "sum", "none")

    Example:
        >>> loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        >>> logits = torch.randn(32, 5)
        >>> labels = torch.randint(0, 5, (32,))
        >>> loss = loss_fn(logits, labels)
    """

    def __init__(
        self,
        alpha: Optional[float | torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        """Initialize focal loss."""
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        logger.debug(f"Initialized FocalLoss: alpha={alpha}, gamma={gamma}")

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Predicted logits of shape (B, K)
            labels: Ground truth labels of shape (B,)

        Returns:
            Scalar loss value
        """
        # Compute cross-entropy without reduction
        ce_loss = F.cross_entropy(logits, labels, reduction="none")

        # Get probabilities
        probs = torch.softmax(logits, dim=1)

        # Get probability of true class
        pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1)

        # Compute focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma

        # Apply focal weight
        focal_loss = focal_weight * ce_loss

        # Apply alpha weighting if provided
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                # Per-class alpha
                alpha = self.alpha.to(logits.device)
                alpha_t = alpha.gather(0, labels)
            else:
                # Single alpha value
                alpha_t = self.alpha

            focal_loss = alpha_t * focal_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropyLoss(nn.Module):
    """
    Cross-Entropy Loss with Label Smoothing.

    Label smoothing softens hard targets to improve generalization and
    handle annotation noise. Instead of [0, 0, 1, 0, 0], targets become
    [ε, ε, 1-4ε, ε, ε] where ε is the smoothing parameter.

    Reference: Szegedy et al. "Rethinking the Inception Architecture for
    Computer Vision" (CVPR 2016)

    Args:
        smoothing: Label smoothing factor in [0, 1] (default: 0.1)
                   0.0 = no smoothing, 0.1 = 10% smoothing
        reduction: Reduction method ("mean", "sum", "none")

    Example:
        >>> loss_fn = LabelSmoothingCrossEntropyLoss(smoothing=0.1)
        >>> logits = torch.randn(32, 5)
        >>> labels = torch.randint(0, 5, (32,))
        >>> loss = loss_fn(logits, labels)
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean") -> None:
        """Initialize label smoothing cross-entropy loss."""
        super().__init__()

        if not 0.0 <= smoothing <= 1.0:
            raise ValueError(
                f"Smoothing must be in [0, 1].\n"
                f"  Received: {smoothing}\n"
                f"  Hint: Typical values are 0.1 (10% smoothing)"
            )

        self.smoothing = smoothing
        self.reduction = reduction

        logger.debug(f"Initialized LabelSmoothingCrossEntropyLoss: smoothing={smoothing}")

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing cross-entropy loss.

        Args:
            logits: Predicted logits of shape (B, K)
            labels: Ground truth labels of shape (B,)

        Returns:
            Scalar loss value
        """
        batch_size, num_classes = logits.shape

        # Convert hard labels to smoothed distribution
        # True class gets (1 - smoothing), others get smoothing / (K - 1)
        with torch.no_grad():
            # Start with uniform distribution weighted by smoothing
            smooth_labels = torch.full_like(
                logits, self.smoothing / (num_classes - 1)
            )

            # Set true class to (1 - smoothing)
            smooth_labels.scatter_(1, labels.unsqueeze(1), 1.0 - self.smoothing)

        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=1)

        # Compute KL divergence between smooth_labels and predictions
        loss = -torch.sum(smooth_labels * log_probs, dim=1)

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def get_loss_function(
    loss_type: str,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
    focal_alpha: Optional[float] = None,
) -> nn.Module:
    """
    Factory function to create loss function.

    Args:
        loss_type: Type of loss ("weighted_ce", "focal", "label_smoothing_ce")
        class_weights: Class weights for weighted CE or focal loss
        label_smoothing: Smoothing factor for label smoothing CE
        focal_gamma: Gamma parameter for focal loss
        focal_alpha: Alpha parameter for focal loss

    Returns:
        Loss function module

    Raises:
        ValueError: If loss_type is unknown

    Example:
        >>> loss_fn = get_loss_function(
        >>>     loss_type="weighted_ce",
        >>>     class_weights=torch.tensor([1.0, 1.0, 3.5, 8.0, 15.0])
        >>> )
    """
    loss_type = loss_type.lower()

    if loss_type == "weighted_ce":
        return WeightedCrossEntropyLoss(class_weights=class_weights)

    elif loss_type == "focal":
        # Use class weights as alpha if provided
        alpha = focal_alpha if focal_alpha is not None else class_weights
        return FocalLoss(alpha=alpha, gamma=focal_gamma)

    elif loss_type == "label_smoothing_ce":
        return LabelSmoothingCrossEntropyLoss(smoothing=label_smoothing)

    elif loss_type == "ce":
        # Standard cross-entropy (no weighting)
        return nn.CrossEntropyLoss()

    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}\n"
            f"  Supported types: ['weighted_ce', 'focal', 'label_smoothing_ce', 'ce']"
        )


class CombinedLoss(nn.Module):
    """
    Combination of multiple loss functions.

    Useful for combining different objectives (e.g., weighted CE + focal).

    Args:
        losses: List of loss modules
        weights: List of weights for each loss (default: equal weights)

    Example:
        >>> loss_fn = CombinedLoss(
        >>>     losses=[
        >>>         WeightedCrossEntropyLoss(class_weights),
        >>>         FocalLoss(gamma=2.0),
        >>>     ],
        >>>     weights=[0.7, 0.3]
        >>> )
    """

    def __init__(
        self,
        losses: list[nn.Module],
        weights: Optional[list[float]] = None,
    ) -> None:
        """Initialize combined loss."""
        super().__init__()

        self.losses = nn.ModuleList(losses)

        if weights is None:
            self.weights = [1.0 / len(losses)] * len(losses)
        else:
            if len(weights) != len(losses):
                raise ValueError(
                    f"Number of weights ({len(weights)}) must match "
                    f"number of losses ({len(losses)})"
                )
            self.weights = weights

        logger.debug(f"Initialized CombinedLoss with {len(losses)} losses")

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute combined loss."""
        total_loss = 0.0

        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(logits, labels)

        return total_loss
