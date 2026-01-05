"""Training callbacks for TTM-HAR."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from src.utils.checkpointing import save_checkpoint, cleanup_old_checkpoints

logger = logging.getLogger(__name__)


class Callback:
    """
    Base class for training callbacks.

    Callbacks allow custom code to be executed at specific points during training.
    """

    def on_train_begin(self, trainer: Any) -> None:
        """Called at the start of training."""
        pass

    def on_train_end(self, trainer: Any) -> None:
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, epoch: int, trainer: Any) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: Any) -> None:
        """Called at the end of each epoch."""
        pass

    def on_batch_begin(self, batch_idx: int, trainer: Any) -> None:
        """Called at the start of each batch."""
        pass

    def on_batch_end(self, batch_idx: int, loss: float, trainer: Any) -> None:
        """Called at the end of each batch."""
        pass

    def on_validation_end(self, metrics: Dict[str, float], trainer: Any) -> None:
        """Called after validation."""
        pass


class CallbackList:
    """
    Container for managing multiple callbacks.

    Args:
        callbacks: List of callback instances

    Example:
        >>> callbacks = CallbackList([
        >>>     CheckpointCallback(checkpoint_dir="checkpoints"),
        >>>     EarlyStoppingCallback(patience=10)
        >>> ])
    """

    def __init__(self, callbacks: List[Callback]) -> None:
        """Initialize callback list."""
        self.callbacks = callbacks

    def on_train_begin(self, trainer: Any) -> None:
        """Call on_train_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_begin(trainer)

    def on_train_end(self, trainer: Any) -> None:
        """Call on_train_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_end(trainer)

    def on_epoch_begin(self, epoch: int, trainer: Any) -> None:
        """Call on_epoch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, trainer)

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: Any) -> None:
        """Call on_epoch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, metrics, trainer)

    def on_batch_begin(self, batch_idx: int, trainer: Any) -> None:
        """Call on_batch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_begin(batch_idx, trainer)

    def on_batch_end(self, batch_idx: int, loss: float, trainer: Any) -> None:
        """Call on_batch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_end(batch_idx, loss, trainer)

    def on_validation_end(self, metrics: Dict[str, float], trainer: Any) -> None:
        """Call on_validation_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_validation_end(metrics, trainer)


class CheckpointCallback(Callback):
    """
    Callback for saving model checkpoints.

    Args:
        checkpoint_dir: Directory to save checkpoints
        monitor_metric: Metric to monitor for best checkpoint
        mode: "max" or "min" for metric optimization
        save_every_n_epochs: Save checkpoint every N epochs
        keep_last_n: Keep only last N checkpoints
        save_best: Whether to save best checkpoint separately

    Example:
        >>> callback = CheckpointCallback(
        >>>     checkpoint_dir="checkpoints",
        >>>     monitor_metric="balanced_accuracy",
        >>>     mode="max"
        >>> )
    """

    def __init__(
        self,
        checkpoint_dir: str,
        monitor_metric: str = "balanced_accuracy",
        mode: str = "max",
        save_every_n_epochs: int = 1,
        keep_last_n: int = 3,
        save_best: bool = True,
    ) -> None:
        """Initialize checkpoint callback."""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.monitor_metric = monitor_metric
        self.mode = mode
        self.save_every_n_epochs = save_every_n_epochs
        self.keep_last_n = keep_last_n
        self.save_best = save_best

        # Track best metric
        self.best_metric = float('-inf') if mode == "max" else float('inf')

        logger.info(
            f"Initialized CheckpointCallback:\n"
            f"  Directory: {self.checkpoint_dir}\n"
            f"  Monitor: {monitor_metric} ({mode})\n"
            f"  Save every: {save_every_n_epochs} epochs\n"
            f"  Keep last: {keep_last_n}\n"
            f"  Save best: {save_best}"
        )

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: Any) -> None:
        """Save checkpoint if needed."""
        # Check if we should save this epoch
        if (epoch + 1) % self.save_every_n_epochs != 0:
            return

        # Get current metric value
        current_metric = metrics.get(self.monitor_metric, None)

        if current_metric is None:
            logger.warning(
                f"Metric '{self.monitor_metric}' not found in metrics. "
                f"Available: {list(metrics.keys())}"
            )
            return

        # Check if this is the best model
        is_best = self._is_better(current_metric, self.best_metric)

        if is_best:
            self.best_metric = current_metric
            logger.info(
                f"New best {self.monitor_metric}: {current_metric:.4f} "
                f"(previous: {self.best_metric:.4f})"
            )

        # Create checkpoint
        checkpoint = {
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": trainer.scheduler.state_dict() if trainer.scheduler else None,
            "epoch": epoch,
            "global_step": trainer.global_step,
            "best_metric": self.best_metric,
            "config": trainer.config,
            "seed": trainer.config.get("experiment", {}).get("seed", 42),
        }

        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f"epoch_{epoch+1}.pt"
        save_checkpoint(checkpoint, checkpoint_path, is_best=is_best and self.save_best)

        # Cleanup old checkpoints
        if self.keep_last_n > 0:
            cleanup_old_checkpoints(
                self.checkpoint_dir,
                keep_last_n=self.keep_last_n,
                keep_best=self.save_best,
            )

    def _is_better(self, current: float, best: float) -> bool:
        """Check if current metric is better than best."""
        if self.mode == "max":
            return current > best
        else:
            return current < best


class EarlyStoppingCallback(Callback):
    """
    Callback for early stopping based on validation metric.

    Stops training if the monitored metric doesn't improve for a given number of epochs.

    Args:
        monitor_metric: Metric to monitor
        patience: Number of epochs to wait before stopping
        mode: "max" or "min" for metric optimization
        min_delta: Minimum change to qualify as improvement

    Example:
        >>> callback = EarlyStoppingCallback(
        >>>     monitor_metric="balanced_accuracy",
        >>>     patience=10,
        >>>     mode="max"
        >>> )
    """

    def __init__(
        self,
        monitor_metric: str = "balanced_accuracy",
        patience: int = 10,
        mode: str = "max",
        min_delta: float = 0.0,
    ) -> None:
        """Initialize early stopping callback."""
        self.monitor_metric = monitor_metric
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self.best_metric = float('-inf') if mode == "max" else float('inf')
        self.epochs_without_improvement = 0
        self.should_stop = False

        logger.info(
            f"Initialized EarlyStoppingCallback:\n"
            f"  Monitor: {monitor_metric} ({mode})\n"
            f"  Patience: {patience} epochs\n"
            f"  Min delta: {min_delta}"
        )

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: Any) -> None:
        """Check if we should stop training."""
        current_metric = metrics.get(self.monitor_metric, None)

        if current_metric is None:
            logger.warning(
                f"Metric '{self.monitor_metric}' not found for early stopping. "
                f"Available: {list(metrics.keys())}"
            )
            return

        # Check for improvement
        improved = self._is_better(current_metric, self.best_metric)

        if improved:
            self.best_metric = current_metric
            self.epochs_without_improvement = 0
            logger.debug(f"Metric improved to {current_metric:.4f}")
        else:
            self.epochs_without_improvement += 1
            logger.debug(
                f"No improvement for {self.epochs_without_improvement}/{self.patience} epochs"
            )

        # Check if we should stop
        if self.epochs_without_improvement >= self.patience:
            self.should_stop = True
            logger.info(
                f"Early stopping triggered after {epoch + 1} epochs. "
                f"Best {self.monitor_metric}: {self.best_metric:.4f}"
            )
            trainer.should_stop = True

    def _is_better(self, current: float, best: float) -> bool:
        """Check if current metric is better than best."""
        if self.mode == "max":
            return current > best + self.min_delta
        else:
            return current < best - self.min_delta


class TensorBoardCallback(Callback):
    """
    Callback for logging to TensorBoard.

    Args:
        log_dir: Directory for TensorBoard logs
        log_every_n_steps: Log every N training steps

    Example:
        >>> callback = TensorBoardCallback(log_dir="runs/experiment1")
    """

    def __init__(
        self,
        log_dir: str,
        log_every_n_steps: int = 10,
    ) -> None:
        """Initialize TensorBoard callback."""
        self.log_dir = Path(log_dir)
        self.log_every_n_steps = log_every_n_steps
        self.writer: Optional[SummaryWriter] = None

        logger.info(f"Initialized TensorBoardCallback: log_dir={log_dir}")

    def on_train_begin(self, trainer: Any) -> None:
        """Create TensorBoard writer."""
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        logger.info(f"TensorBoard logging to: {self.log_dir}")

    def on_train_end(self, trainer: Any) -> None:
        """Close TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()

    def on_batch_end(self, batch_idx: int, loss: float, trainer: Any) -> None:
        """Log training loss."""
        if self.writer is None:
            return

        if trainer.global_step % self.log_every_n_steps == 0:
            self.writer.add_scalar("train/loss", loss, trainer.global_step)

            # Log learning rates
            lrs = [group["lr"] for group in trainer.optimizer.param_groups]
            for i, lr in enumerate(lrs):
                self.writer.add_scalar(f"train/lr_group_{i}", lr, trainer.global_step)

    def on_validation_end(self, metrics: Dict[str, float], trainer: Any) -> None:
        """Log validation metrics."""
        if self.writer is None:
            return

        for metric_name, metric_value in metrics.items():
            self.writer.add_scalar(f"val/{metric_name}", metric_value, trainer.global_step)


class LearningRateLoggerCallback(Callback):
    """
    Callback for logging learning rates.

    Example:
        >>> callback = LearningRateLoggerCallback()
    """

    def on_epoch_begin(self, epoch: int, trainer: Any) -> None:
        """Log learning rates at epoch start."""
        lrs = [group["lr"] for group in trainer.optimizer.param_groups]

        if len(lrs) == 1:
            logger.info(f"Epoch {epoch + 1}: LR = {lrs[0]:.2e}")
        else:
            lr_str = ", ".join([f"group_{i}={lr:.2e}" for i, lr in enumerate(lrs)])
            logger.info(f"Epoch {epoch + 1}: {lr_str}")
