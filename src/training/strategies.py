"""Training strategies for TTM-HAR."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch.nn as nn

logger = logging.getLogger(__name__)


class TrainingStrategy(ABC):
    """
    Abstract base class for training strategies.

    Training strategies define how the model is trained (freezing, learning rates, etc.)
    across different phases of training.

    Args:
        model: Model to train
        config: Training configuration
    """

    def __init__(self, model: nn.Module, config: Dict) -> None:
        """Initialize training strategy."""
        self.model = model
        self.config = config
        self.current_epoch = 0

    @abstractmethod
    def on_epoch_start(self, epoch: int) -> None:
        """
        Hook called at the start of each epoch.

        Args:
            epoch: Current epoch number (0-indexed)
        """
        pass

    @abstractmethod
    def get_num_epochs(self) -> int:
        """
        Get total number of training epochs.

        Returns:
            Total epochs for this strategy
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """
        Get human-readable description of strategy.

        Returns:
            Strategy description
        """
        pass


class LinearProbeStrategy(TrainingStrategy):
    """
    Linear Probe training strategy.

    Freezes the backbone and trains only the classification head.
    This is the fastest strategy and good for quick experimentation.

    Architecture: [Frozen Backbone] → [Trainable Head]

    Args:
        model: Model to train
        config: Training configuration

    Example:
        >>> strategy = LinearProbeStrategy(model, config)
        >>> strategy.on_epoch_start(0)  # Freezes backbone
    """

    def __init__(self, model: nn.Module, config: Dict) -> None:
        """Initialize linear probe strategy."""
        super().__init__(model, config)

        self.epochs = config.get("epochs", 20)

        logger.info(
            f"Initialized LinearProbeStrategy:\n"
            f"  Epochs: {self.epochs}\n"
            f"  Backbone: FROZEN\n"
            f"  Head: TRAINABLE"
        )

    def on_epoch_start(self, epoch: int) -> None:
        """Freeze backbone at start of training."""
        if epoch == 0:
            # Freeze backbone on first epoch
            self.model.freeze_backbone(strategy="all")
            logger.info("Froze backbone for linear probe training")

        self.current_epoch = epoch

    def get_num_epochs(self) -> int:
        """Get total epochs."""
        return self.epochs

    def get_description(self) -> str:
        """Get strategy description."""
        return f"Linear Probe ({self.epochs} epochs, frozen backbone)"


class FullFinetuneStrategy(TrainingStrategy):
    """
    Full Fine-tuning strategy.

    Trains both backbone and head end-to-end.
    This is the most expensive but potentially most effective strategy.

    Architecture: [Trainable Backbone] → [Trainable Head]

    Args:
        model: Model to train
        config: Training configuration

    Example:
        >>> strategy = FullFinetuneStrategy(model, config)
        >>> strategy.on_epoch_start(0)  # Unfreezes everything
    """

    def __init__(self, model: nn.Module, config: Dict) -> None:
        """Initialize full fine-tune strategy."""
        super().__init__(model, config)

        self.epochs = config.get("epochs", 50)

        logger.info(
            f"Initialized FullFinetuneStrategy:\n"
            f"  Epochs: {self.epochs}\n"
            f"  Backbone: TRAINABLE\n"
            f"  Head: TRAINABLE"
        )

    def on_epoch_start(self, epoch: int) -> None:
        """Unfreeze everything at start of training."""
        if epoch == 0:
            # Unfreeze everything on first epoch
            self.model.unfreeze_backbone()
            logger.info("Unfroze backbone for full fine-tuning")

        self.current_epoch = epoch

    def get_num_epochs(self) -> int:
        """Get total epochs."""
        return self.epochs

    def get_description(self) -> str:
        """Get strategy description."""
        return f"Full Fine-tuning ({self.epochs} epochs, trainable backbone)"


class LPThenFTStrategy(TrainingStrategy):
    """
    Linear Probe then Fine-Tuning strategy.

    Two-phase training:
    1. Phase 1 (Linear Probe): Freeze backbone, train head only
    2. Phase 2 (Fine-tune): Unfreeze backbone, train everything

    This often gives the best results as it:
    - Quickly adapts the head to the task
    - Then fine-tunes the backbone for optimal performance

    Args:
        model: Model to train
        config: Training configuration
        reset_head_at_transition: If True, reinitialize head at phase 2 start

    Example:
        >>> strategy = LPThenFTStrategy(model, config)
        >>> strategy.on_epoch_start(0)   # Phase 1: frozen backbone
        >>> strategy.on_epoch_start(20)  # Phase 2: unfrozen backbone
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        reset_head_at_transition: bool = False,
    ) -> None:
        """Initialize LP-then-FT strategy."""
        super().__init__(model, config)

        self.epochs_lp = config.get("epochs_linear_probe", 20)
        self.epochs_ft = config.get("epochs_finetune", 30)
        self.reset_head_at_transition = reset_head_at_transition

        self.total_epochs = self.epochs_lp + self.epochs_ft
        self.transition_epoch = self.epochs_lp

        logger.info(
            f"Initialized LPThenFTStrategy:\n"
            f"  Phase 1 (Linear Probe): {self.epochs_lp} epochs\n"
            f"  Phase 2 (Fine-tune): {self.epochs_ft} epochs\n"
            f"  Total: {self.total_epochs} epochs\n"
            f"  Transition at epoch: {self.transition_epoch}\n"
            f"  Reset head at transition: {reset_head_at_transition}"
        )

    def on_epoch_start(self, epoch: int) -> None:
        """Handle phase transitions."""
        self.current_epoch = epoch

        if epoch == 0:
            # Phase 1 start: Freeze backbone
            self.model.freeze_backbone(strategy="all")
            logger.info("Phase 1: Froze backbone for linear probe")

        elif epoch == self.transition_epoch:
            # Phase 2 start: Unfreeze backbone
            logger.info(f"Transitioning to Phase 2 at epoch {epoch}")

            # Optionally reset head
            if self.reset_head_at_transition:
                logger.info("Resetting classification head")
                self._reset_head()

            # Unfreeze backbone
            self.model.unfreeze_backbone()
            logger.info("Phase 2: Unfroze backbone for fine-tuning")

    def _reset_head(self) -> None:
        """Reset classification head parameters."""
        for module in self.model.head.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def get_num_epochs(self) -> int:
        """Get total epochs."""
        return self.total_epochs

    def get_description(self) -> str:
        """Get strategy description."""
        phase = 1 if self.current_epoch < self.transition_epoch else 2
        return (
            f"LP-then-FT (Phase {phase}/2: "
            f"{self.epochs_lp}+{self.epochs_ft}={self.total_epochs} epochs)"
        )

    def get_current_phase(self) -> int:
        """
        Get current training phase.

        Returns:
            1 for linear probe phase, 2 for fine-tuning phase
        """
        return 1 if self.current_epoch < self.transition_epoch else 2


def get_strategy(
    strategy_name: str,
    model: nn.Module,
    config: Dict,
) -> TrainingStrategy:
    """
    Factory function to create training strategy.

    Args:
        strategy_name: Name of strategy ("linear_probe", "full_finetune", "lp_then_ft")
        model: Model to train
        config: Training configuration

    Returns:
        TrainingStrategy instance

    Raises:
        ValueError: If strategy name is unknown

    Example:
        >>> strategy = get_strategy("linear_probe", model, config)
    """
    strategy_name = strategy_name.lower()

    if strategy_name == "linear_probe":
        return LinearProbeStrategy(model, config)

    elif strategy_name == "full_finetune":
        return FullFinetuneStrategy(model, config)

    elif strategy_name == "lp_then_ft":
        reset_head = config.get("reset_head_at_transition", False)
        return LPThenFTStrategy(model, config, reset_head_at_transition=reset_head)

    else:
        raise ValueError(
            f"Unknown training strategy: {strategy_name}\n"
            f"  Supported strategies: ['linear_probe', 'full_finetune', 'lp_then_ft']"
        )
