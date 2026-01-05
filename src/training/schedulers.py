"""Learning rate schedulers for TTM-HAR."""

import logging
import math
from typing import Dict

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler

logger = logging.getLogger(__name__)


def create_scheduler(
    optimizer: optim.Optimizer,
    config: Dict,
    num_training_steps: int,
) -> _LRScheduler:
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer to schedule
        config: Training configuration
        num_training_steps: Total number of training steps

    Returns:
        Learning rate scheduler

    Example:
        >>> scheduler = create_scheduler(optimizer, config, num_training_steps=1000)
        >>> for epoch in range(num_epochs):
        >>>     for batch in dataloader:
        >>>         ...
        >>>         scheduler.step()
    """
    scheduler_type = config.get("scheduler", "cosine").lower()
    warmup_ratio = config.get("warmup_ratio", 0.1)

    num_warmup_steps = int(num_training_steps * warmup_ratio)

    logger.info(
        f"Creating {scheduler_type} scheduler:\n"
        f"  Total steps: {num_training_steps}\n"
        f"  Warmup steps: {num_warmup_steps} ({warmup_ratio*100:.1f}%)"
    )

    if scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    elif scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    elif scheduler_type == "constant":
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
        )

    elif scheduler_type == "cosine_with_restarts":
        num_cycles = config.get("num_cycles", 1)
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
        )

    else:
        raise ValueError(
            f"Unknown scheduler type: {scheduler_type}\n"
            f"  Supported: ['cosine', 'linear', 'constant', 'cosine_with_restarts']"
        )

    return scheduler


def get_cosine_schedule_with_warmup(
    optimizer: optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create cosine learning rate schedule with linear warmup.

    Learning rate increases linearly during warmup, then follows a cosine decay.

    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total training steps
        num_cycles: Number of cosine cycles (default: 0.5 for half cycle)
        last_epoch: Last epoch number

    Returns:
        LambdaLR scheduler
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_linear_schedule_with_warmup(
    optimizer: optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create linear learning rate schedule with linear warmup.

    Learning rate increases linearly during warmup, then decreases linearly.

    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total training steps
        last_epoch: Last epoch number

    Returns:
        LambdaLR scheduler
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))

        # Linear decay
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_constant_schedule_with_warmup(
    optimizer: optim.Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create constant learning rate schedule with linear warmup.

    Learning rate increases linearly during warmup, then stays constant.

    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        last_epoch: Last epoch number

    Returns:
        LambdaLR scheduler
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))

        # Constant
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer: optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int = 1,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create cosine learning rate schedule with hard restarts and warmup.

    Learning rate follows multiple cosine cycles with hard restarts.

    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total training steps
        num_cycles: Number of restart cycles
        last_epoch: Last epoch number

    Returns:
        LambdaLR scheduler
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine with hard restarts
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        if progress >= 1.0:
            return 0.0

        return max(
            0.0,
            0.5 * (1.0 + math.cos(math.pi * ((progress * num_cycles) % 1.0))),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class WarmupScheduler(_LRScheduler):
    """
    Wrapper scheduler that adds warmup to any base scheduler.

    Args:
        optimizer: Optimizer to schedule
        base_scheduler: Base scheduler to wrap
        num_warmup_steps: Number of warmup steps
        warmup_start_lr: Starting learning rate for warmup

    Example:
        >>> base = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        >>> scheduler = WarmupScheduler(optimizer, base, num_warmup_steps=10)
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        base_scheduler: _LRScheduler,
        num_warmup_steps: int,
        warmup_start_lr: float = 1e-7,
    ) -> None:
        """Initialize warmup scheduler."""
        self.base_scheduler = base_scheduler
        self.num_warmup_steps = num_warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.current_step = 0

        super().__init__(optimizer)

    def get_lr(self):
        """Get current learning rates."""
        if self.current_step < self.num_warmup_steps:
            # Linear warmup
            alpha = float(self.current_step) / float(self.num_warmup_steps)
            return [
                self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]
        else:
            # Use base scheduler
            return self.base_scheduler.get_last_lr()

    def step(self, epoch=None):
        """Step the scheduler."""
        self.current_step += 1

        if self.current_step > self.num_warmup_steps:
            self.base_scheduler.step(epoch)

        super().step(epoch)
