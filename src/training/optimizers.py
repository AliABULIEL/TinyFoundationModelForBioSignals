"""Optimizer configuration for TTM-HAR."""

import logging
from typing import Dict, List

import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


def create_optimizer(
    model: nn.Module,
    config: Dict,
    strategy_name: str,
) -> optim.Optimizer:
    """
    Create optimizer with proper parameter grouping and learning rates.

    Args:
        model: Model to optimize
        config: Training configuration
        strategy_name: Training strategy name (affects LR assignment)

    Returns:
        Configured optimizer

    Example:
        >>> optimizer = create_optimizer(model, config, strategy_name="linear_probe")
    """
    optimizer_name = config.get("optimizer", "adamw").lower()
    lr_head = config.get("lr_head", 1e-3)
    lr_backbone = config.get("lr_backbone", 1e-5)
    weight_decay = config.get("weight_decay", 0.01)

    # Get parameter groups from model
    param_groups = model.get_parameter_groups()

    # Determine learning rates based on strategy
    if strategy_name == "linear_probe":
        # Only head is trainable, backbone is frozen
        # Use lr_head for all trainable params
        groups = [
            {
                "params": param_groups["head"],
                "lr": lr_head,
                "weight_decay": weight_decay,
            }
        ]
        logger.info(f"Linear probe: using lr={lr_head} for head only")

    elif strategy_name == "full_finetune":
        # Both trainable: use different LRs
        groups = [
            {
                "params": param_groups["backbone"],
                "lr": lr_backbone,
                "weight_decay": weight_decay,
            },
            {
                "params": param_groups["head"],
                "lr": lr_head,
                "weight_decay": weight_decay,
            },
        ]
        logger.info(
            f"Full fine-tune: backbone lr={lr_backbone}, head lr={lr_head}"
        )

    elif strategy_name == "lp_then_ft":
        # Phase 1: Only head trainable
        # Phase 2: Both trainable (handled by strategy on_epoch_start)
        # We'll create groups for both, but frozen params won't update
        groups = [
            {
                "params": param_groups["backbone"],
                "lr": lr_backbone,
                "weight_decay": weight_decay,
            },
            {
                "params": param_groups["head"],
                "lr": lr_head,
                "weight_decay": weight_decay,
            },
        ]
        logger.info(
            f"LP-then-FT: Phase 1 head lr={lr_head}, "
            f"Phase 2 backbone lr={lr_backbone}, head lr={lr_head}"
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    # Apply weight decay exclusions
    groups = _apply_weight_decay_exclusions(groups)

    # Create optimizer
    if optimizer_name == "adamw":
        optimizer = optim.AdamW(groups)
    elif optimizer_name == "adam":
        optimizer = optim.Adam(groups)
    elif optimizer_name == "sgd":
        momentum = config.get("momentum", 0.9)
        optimizer = optim.SGD(groups, momentum=momentum)
    else:
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}\n"
            f"  Supported: ['adamw', 'adam', 'sgd']"
        )

    # Log optimizer info
    total_params = sum(p.numel() for group in groups for p in group["params"])
    trainable_params = sum(
        p.numel() for group in groups for p in group["params"] if p.requires_grad
    )

    logger.info(
        f"Created {optimizer_name.upper()} optimizer:\n"
        f"  Total params: {total_params:,}\n"
        f"  Trainable params: {trainable_params:,}\n"
        f"  Parameter groups: {len(groups)}"
    )

    return optimizer


def _apply_weight_decay_exclusions(param_groups: List[Dict]) -> List[Dict]:
    """
    Apply weight decay exclusions for bias and normalization parameters.

    Weight decay should not be applied to:
    - Bias terms
    - BatchNorm/LayerNorm parameters

    Args:
        param_groups: List of parameter group dictionaries

    Returns:
        Updated parameter groups with exclusions applied
    """
    new_groups = []

    for group in param_groups:
        # Separate parameters that should/shouldn't have weight decay
        decay_params = []
        no_decay_params = []

        for param in group["params"]:
            # Check parameter name to determine if it should have decay
            # This is a heuristic - ideally we'd check parameter names
            if param.dim() == 1:
                # 1D parameters are typically biases or norm parameters
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        # Create separate groups
        if decay_params:
            new_groups.append({
                **group,
                "params": decay_params,
            })

        if no_decay_params:
            new_groups.append({
                **group,
                "params": no_decay_params,
                "weight_decay": 0.0,  # No decay for biases/norms
            })

    return new_groups


def update_learning_rate(
    optimizer: optim.Optimizer,
    new_lr: float,
    group_idx: int = 0,
) -> None:
    """
    Update learning rate for a specific parameter group.

    Args:
        optimizer: Optimizer to update
        new_lr: New learning rate
        group_idx: Index of parameter group to update

    Example:
        >>> update_learning_rate(optimizer, new_lr=1e-4, group_idx=0)
    """
    optimizer.param_groups[group_idx]["lr"] = new_lr
    logger.debug(f"Updated LR for group {group_idx} to {new_lr}")


def get_current_learning_rates(optimizer: optim.Optimizer) -> List[float]:
    """
    Get current learning rates for all parameter groups.

    Args:
        optimizer: Optimizer to query

    Returns:
        List of learning rates

    Example:
        >>> lrs = get_current_learning_rates(optimizer)
        >>> print(f"Backbone LR: {lrs[0]}, Head LR: {lrs[1]}")
    """
    return [group["lr"] for group in optimizer.param_groups]
