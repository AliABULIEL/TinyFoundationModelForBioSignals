"""Main trainer for TTM-HAR."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.losses import get_loss_function
from src.training.callbacks import CallbackList, Callback
from src.training.optimizers import create_optimizer
from src.training.schedulers import create_scheduler
from src.training.strategies import TrainingStrategy, get_strategy

logger = logging.getLogger(__name__)


class Trainer:
    """
    Main trainer for Human Activity Recognition.

    Handles the complete training loop including:
    - Strategy-based training (Linear Probe, Fine-tune, LP-then-FT)
    - Optimization and scheduling
    - Validation and metrics
    - Checkpointing and callbacks

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Complete configuration dictionary
        device: Device to train on
        callbacks: List of callbacks

    Example:
        >>> trainer = Trainer(
        >>>     model=model,
        >>>     train_loader=train_loader,
        >>>     val_loader=val_loader,
        >>>     config=config,
        >>>     device=device
        >>> )
        >>> trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: torch.device,
        callbacks: Optional[list[Callback]] = None,
    ) -> None:
        """Initialize trainer."""
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Move model to device
        self.model = self.model.to(device)

        # Training configuration
        training_config = config.get("training", {})
        self.batch_size = training_config.get("batch_size", 64)
        self.gradient_clip_norm = training_config.get("gradient_clip_norm", 1.0)

        # Get training strategy
        strategy_name = training_config.get("strategy", "linear_probe")
        self.strategy: TrainingStrategy = get_strategy(strategy_name, model, training_config)

        # Create optimizer
        self.optimizer = create_optimizer(model, training_config, strategy_name)

        # Create scheduler
        num_training_steps = len(train_loader) * self.strategy.get_num_epochs()
        self.scheduler = create_scheduler(self.optimizer, training_config, num_training_steps)

        # Create loss function
        loss_config = training_config.get("loss", {})
        loss_type = loss_config.get("type", "weighted_ce")

        # Get class weights from data module if available
        class_weights = config.get("class_weights", None)
        if class_weights is not None:
            class_weights = torch.tensor(class_weights).to(device)

        self.criterion = get_loss_function(
            loss_type=loss_type,
            class_weights=class_weights,
            label_smoothing=loss_config.get("label_smoothing", 0.1),
            focal_gamma=loss_config.get("focal_gamma", 2.0),
            focal_alpha=loss_config.get("focal_alpha", None),
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.should_stop = False

        # Callbacks
        self.callbacks = CallbackList(callbacks if callbacks else [])

        # Metrics tracking
        self.train_losses = []
        self.val_metrics = []

        logger.info(
            f"Initialized Trainer:\n"
            f"  Strategy: {self.strategy.get_description()}\n"
            f"  Optimizer: {self.optimizer.__class__.__name__}\n"
            f"  Scheduler: {self.scheduler.__class__.__name__}\n"
            f"  Loss: {self.criterion.__class__.__name__}\n"
            f"  Device: {device}\n"
            f"  Batch size: {self.batch_size}\n"
            f"  Training steps: {num_training_steps}"
        )

    def train(self) -> Dict[str, Any]:
        """
        Run complete training loop.

        Returns:
            Dictionary with training history

        Example:
            >>> history = trainer.train()
            >>> print(f"Best val acc: {history['best_val_metric']:.4f}")
        """
        logger.info("Starting training")
        self.callbacks.on_train_begin(self)

        start_time = time.time()
        num_epochs = self.strategy.get_num_epochs()

        for epoch in range(num_epochs):
            if self.should_stop:
                logger.info(f"Training stopped early at epoch {epoch + 1}")
                break

            self.current_epoch = epoch

            # Strategy hook (may freeze/unfreeze)
            self.strategy.on_epoch_start(epoch)

            # Callbacks
            self.callbacks.on_epoch_begin(epoch, self)

            # Train one epoch
            train_loss = self._train_epoch()

            # Validate
            val_metrics = self._validate()

            # Log epoch results
            self._log_epoch(epoch, train_loss, val_metrics)

            # Callbacks
            self.callbacks.on_epoch_end(epoch, val_metrics, self)

        # Training complete
        elapsed_time = time.time() - start_time
        logger.info(f"Training complete in {elapsed_time / 60:.1f} minutes")

        self.callbacks.on_train_end(self)

        return {
            "train_losses": self.train_losses,
            "val_metrics": self.val_metrics,
            "num_epochs": self.current_epoch + 1,
            "total_steps": self.global_step,
        }

    def _train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            Average training loss
        """
        self.model.train()

        epoch_losses = []
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}",
            leave=False,
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Callbacks
            self.callbacks.on_batch_begin(batch_idx, self)

            # Forward pass
            loss = self._train_step(batch)

            # Track loss
            epoch_losses.append(loss)

            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss:.4f}"})

            # Callbacks
            self.callbacks.on_batch_end(batch_idx, loss, self)

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        self.train_losses.append(avg_loss)

        return avg_loss

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Perform single training step.

        Args:
            batch: Batch dictionary with 'signal' and 'label'

        Returns:
            Loss value
        """
        # Move batch to device
        inputs = batch["signal"].to(self.device)
        labels = batch["label"].to(self.device)

        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if self.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip_norm
            )

        # Optimizer step
        self.optimizer.step()

        # Scheduler step (per-step scheduling)
        if self.scheduler is not None:
            self.scheduler.step()

        self.global_step += 1

        return loss.item()

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """
        Run validation.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        all_preds = []
        all_labels = []
        val_losses = []

        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            # Move to device
            inputs = batch["signal"].to(self.device)
            labels = batch["label"].to(self.device)

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Get predictions
            preds = torch.argmax(outputs, dim=1)

            # Collect results
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            val_losses.append(loss.item())

        # Concatenate all predictions and labels
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        # Compute metrics
        metrics = self._compute_metrics(all_preds, all_labels)
        metrics["loss"] = sum(val_losses) / len(val_losses)

        self.val_metrics.append(metrics)

        # Callbacks
        self.callbacks.on_validation_end(metrics, self)

        return metrics

    def _compute_metrics(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute validation metrics.

        Args:
            preds: Predicted labels
            labels: Ground truth labels

        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        preds_np = preds.numpy()
        labels_np = labels.numpy()

        metrics = {
            "accuracy": accuracy_score(labels_np, preds_np),
            "balanced_accuracy": balanced_accuracy_score(labels_np, preds_np),
            "macro_f1": f1_score(labels_np, preds_np, average="macro", zero_division=0),
            "weighted_f1": f1_score(labels_np, preds_np, average="weighted", zero_division=0),
        }

        # Per-class metrics (optional, can be verbose)
        # Uncomment if needed:
        # metrics["per_class_precision"] = precision_score(
        #     labels_np, preds_np, average=None, zero_division=0
        # ).tolist()
        # metrics["per_class_recall"] = recall_score(
        #     labels_np, preds_np, average=None, zero_division=0
        # ).tolist()

        return metrics

    def _log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_metrics: Dict[str, float],
    ) -> None:
        """Log epoch results."""
        logger.info(
            f"Epoch {epoch + 1}/{self.strategy.get_num_epochs()} - "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['balanced_accuracy']:.4f} | "
            f"Val F1: {val_metrics['macro_f1']:.4f}"
        )

    def save_checkpoint(self, path: str) -> None:
        """
        Save training checkpoint.

        Args:
            path: Path to save checkpoint
        """
        from src.utils.checkpointing import save_checkpoint

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "config": self.config,
        }

        save_checkpoint(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        """
        Load training checkpoint.

        Args:
            path: Path to checkpoint
        """
        from src.utils.checkpointing import load_checkpoint

        checkpoint = load_checkpoint(path, device=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]

        logger.info(f"Loaded checkpoint from {path} (epoch {self.current_epoch + 1})")
