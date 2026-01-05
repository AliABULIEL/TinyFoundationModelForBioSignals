"""Checkpointing utilities for TTM-HAR."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
    checkpoint: Dict[str, Any],
    save_path: Union[str, Path],
    is_best: bool = False,
    best_suffix: str = "_best",
) -> None:
    """
    Save training checkpoint to disk.

    Args:
        checkpoint: Dictionary containing checkpoint data. Should include:
            - model_state_dict: Model weights
            - optimizer_state_dict: Optimizer state
            - scheduler_state_dict: LR scheduler state
            - epoch: Current epoch
            - global_step: Total training steps
            - best_metric: Best validation metric
            - config: Complete configuration
            - seed: Random seed used
            - timestamp: ISO format timestamp
        save_path: Path where to save checkpoint
        is_best: If True, also save a copy with best_suffix
        best_suffix: Suffix to add for best checkpoint

    Example:
        >>> checkpoint = {
        >>>     "model_state_dict": model.state_dict(),
        >>>     "optimizer_state_dict": optimizer.state_dict(),
        >>>     "epoch": epoch,
        >>>     "best_metric": best_val_acc,
        >>>     "config": config,
        >>> }
        >>> save_checkpoint(checkpoint, "checkpoints/epoch_10.pt", is_best=True)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate checkpoint contents
    _validate_checkpoint(checkpoint, is_save=True)

    # Add timestamp if not present
    if "timestamp" not in checkpoint:
        checkpoint["timestamp"] = datetime.now().isoformat()

    try:
        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint to {save_path}")

        # Save best checkpoint copy if requested
        if is_best:
            best_path = save_path.parent / f"{save_path.stem}{best_suffix}{save_path.suffix}"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint to {best_path}")

    except Exception as e:
        raise IOError(
            f"Failed to save checkpoint to {save_path}\n" f"  Error: {e}"
        ) from e


def load_checkpoint(
    load_path: Union[str, Path],
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load checkpoint from disk.

    Args:
        load_path: Path to checkpoint file
        device: Device to load checkpoint to. If None, uses original device
        strict: If True, strictly validate checkpoint contents

    Returns:
        Dictionary containing checkpoint data

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If checkpoint is corrupted or invalid

    Example:
        >>> device = torch.device("cuda")
        >>> checkpoint = load_checkpoint("checkpoints/best.pt", device=device)
        >>> model.load_state_dict(checkpoint["model_state_dict"])
        >>> optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    """
    load_path = Path(load_path)

    if not load_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {load_path}\n"
            f"  Hint: Check that the path is correct and the file exists."
        )

    try:
        if device is None:
            checkpoint = torch.load(load_path)
        else:
            checkpoint = torch.load(load_path, map_location=device)

        logger.info(f"Loaded checkpoint from {load_path}")

    except Exception as e:
        raise RuntimeError(
            f"Failed to load checkpoint from {load_path}\n"
            f"  Error: {e}\n"
            f"  Hint: Checkpoint file may be corrupted"
        ) from e

    # Validate checkpoint contents
    if strict:
        _validate_checkpoint(checkpoint, is_save=False)

    return checkpoint


def _validate_checkpoint(checkpoint: Dict[str, Any], is_save: bool) -> None:
    """
    Validate checkpoint contents.

    Args:
        checkpoint: Checkpoint dictionary to validate
        is_save: True if validating for save, False if validating after load

    Raises:
        ValueError: If checkpoint is missing required keys or has invalid values
    """
    required_keys = [
        "model_state_dict",
        "optimizer_state_dict",
        "epoch",
        "global_step",
        "config",
    ]

    missing_keys = [key for key in required_keys if key not in checkpoint]

    if missing_keys:
        error_msg = (
            f"Checkpoint missing required keys: {missing_keys}\n"
            f"  Required keys: {required_keys}\n"
            f"  Present keys: {list(checkpoint.keys())}\n"
        )

        if is_save:
            error_msg += "  Hint: Ensure all required state is included when creating checkpoint"
        else:
            error_msg += "  Hint: Checkpoint may be from an incompatible version"

        raise ValueError(error_msg)

    # Validate epoch is non-negative
    if checkpoint["epoch"] < 0:
        raise ValueError(
            f"Invalid epoch value: {checkpoint['epoch']}\n"
            f"  Epoch must be non-negative"
        )

    # Validate global_step is non-negative
    if checkpoint["global_step"] < 0:
        raise ValueError(
            f"Invalid global_step value: {checkpoint['global_step']}\n"
            f"  global_step must be non-negative"
        )


def get_latest_checkpoint(checkpoint_dir: Union[str, Path]) -> Optional[Path]:
    """
    Get path to the most recent checkpoint in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Path to latest checkpoint, or None if no checkpoints found

    Example:
        >>> latest = get_latest_checkpoint("checkpoints/")
        >>> if latest:
        >>>     checkpoint = load_checkpoint(latest)
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        logger.warning(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return None

    # Find all .pt or .pth files
    checkpoints = list(checkpoint_dir.glob("*.pt")) + list(checkpoint_dir.glob("*.pth"))

    if not checkpoints:
        logger.warning(f"No checkpoints found in {checkpoint_dir}")
        return None

    # Sort by modification time, most recent first
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)

    logger.info(f"Latest checkpoint: {latest}")
    return latest


def cleanup_old_checkpoints(
    checkpoint_dir: Union[str, Path],
    keep_last_n: int = 3,
    keep_best: bool = True,
    best_suffix: str = "_best",
) -> None:
    """
    Remove old checkpoints, keeping only the most recent ones.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of most recent checkpoints to keep
        keep_best: If True, always keep best checkpoint regardless of age
        best_suffix: Suffix used for best checkpoint

    Example:
        >>> cleanup_old_checkpoints("checkpoints/", keep_last_n=3, keep_best=True)
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        logger.warning(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return

    # Find all checkpoints
    checkpoints = list(checkpoint_dir.glob("*.pt")) + list(checkpoint_dir.glob("*.pth"))

    if not checkpoints:
        logger.info(f"No checkpoints to clean up in {checkpoint_dir}")
        return

    # Separate best checkpoint if it exists
    best_checkpoints = [cp for cp in checkpoints if best_suffix in cp.stem]
    regular_checkpoints = [cp for cp in checkpoints if best_suffix not in cp.stem]

    # Sort regular checkpoints by modification time, newest first
    regular_checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    # Determine which checkpoints to remove
    checkpoints_to_remove = regular_checkpoints[keep_last_n:]

    # Remove old checkpoints
    for checkpoint_path in checkpoints_to_remove:
        checkpoint_path.unlink()
        logger.info(f"Removed old checkpoint: {checkpoint_path}")

    if checkpoints_to_remove:
        logger.info(
            f"Cleaned up {len(checkpoints_to_remove)} old checkpoints, "
            f"kept {len(regular_checkpoints) - len(checkpoints_to_remove)} recent"
        )
    else:
        logger.info(f"No checkpoints to remove (total: {len(checkpoints)})")
