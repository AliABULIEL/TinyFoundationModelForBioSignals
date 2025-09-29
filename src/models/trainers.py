"""Training utilities and trainers for TTM models with BEST MODEL SAVING."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class EarlyStopping:
    """Early stopping callback."""
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
        verbose: bool = True
    ):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_value = np.Inf if mode == "min" else -np.Inf
    
    def __call__(self, val_score: float, model: Optional[nn.Module] = None) -> bool:
        """Check if should stop.
        
        Args:
            val_score: Current validation score
            model: Model to save if best
            
        Returns:
            Whether to stop training
        """
        if self.mode == "min":
            is_improvement = val_score < self.best_value - self.min_delta
        else:
            is_improvement = val_score > self.best_value + self.min_delta
        
        if self.best_score is None or is_improvement:
            self.best_score = val_score
            self.best_value = val_score
            self.save_checkpoint(val_score, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def save_checkpoint(self, val_score: float, model: Optional[nn.Module] = None):
        """Save model when validation score improves."""
        if self.verbose:
            if self.mode == "min":
                logger.info(f"Validation score decreased ({self.best_value:.6f} --> {val_score:.6f})")
            else:
                logger.info(f"Validation score increased ({self.best_value:.6f} --> {val_score:.6f})")


class TrainerBase:
    """Base trainer class with common functionality."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        device: str = "cuda",
        use_amp: bool = True,
        gradient_clip: float = 1.0,
        log_interval: int = 10,
        checkpoint_dir: Optional[str] = None,
        seed: int = 42,
    ):
        """Initialize base trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer (will create AdamW if None)
            criterion: Loss criterion
            device: Device to use
            use_amp: Whether to use automatic mixed precision
            gradient_clip: Gradient clipping value
            log_interval: Logging interval
            checkpoint_dir: Directory for checkpoints
            seed: Random seed
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp and torch.cuda.is_available()
        self.gradient_clip = gradient_clip
        self.log_interval = log_interval
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Set seed for reproducibility
        self.set_seed(seed)
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer
        if optimizer is None:
            self.optimizer = AdamW(self.model.parameters(), lr=5e-4, weight_decay=0.01)
        else:
            self.optimizer = optimizer
        
        # Setup criterion
        self.criterion = criterion
        
        # Setup AMP
        self.scaler = GradScaler() if self.use_amp else None
        
        # Initialize metrics
        self.train_history = []
        self.val_history = []
        self.best_val_metric = None
        self.epoch = 0
    
    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    
    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Single training step.
        
        Args:
            batch: Batch of data (inputs, targets)
            
        Returns:
            Dictionary of metrics
        """
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        # Forward pass
        if self.use_amp:
            with autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
        else:
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        
        if self.use_amp:
            self.scaler.scale(loss).backward()
            if self.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()
        
        return {"loss": loss.item()}
    
    def val_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, Any]:
        """Single validation step.
        
        Args:
            batch: Batch of data (inputs, targets)
            
        Returns:
            Dictionary with loss, outputs, and targets
        """
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
        
        return {
            "loss": loss.item(),
            "outputs": outputs.cpu(),
            "targets": targets.cpu()
        }
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of metrics
        """
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for i, batch in enumerate(pbar):
            metrics = self.train_step(batch)
            total_loss += metrics["loss"]
            
            # Update progress bar
            avg_loss = total_loss / (i + 1)
            pbar.set_postfix(loss=avg_loss)
            
            # Log at intervals
            if (i + 1) % self.log_interval == 0:
                logger.info(f"Epoch {epoch} [{i+1}/{num_batches}] - Loss: {avg_loss:.4f}")
        
        return {"loss": total_loss / num_batches}
    
    def validate(self) -> Dict[str, float]:
        """Validate model.
        
        Returns:
            Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0
        all_outputs = []
        all_targets = []
        
        pbar = tqdm(self.val_loader, desc="Validation")
        for batch in pbar:
            metrics = self.val_step(batch)
            total_loss += metrics["loss"]
            all_outputs.append(metrics["outputs"])
            all_targets.append(metrics["targets"])
            
            pbar.set_postfix(loss=total_loss / (len(all_outputs)))
        
        # Concatenate all outputs and targets
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute additional metrics
        val_metrics = {
            "loss": total_loss / len(self.val_loader),
            "outputs": all_outputs,
            "targets": all_targets
        }
        
        return val_metrics
    
    def save_checkpoint(self, path: Optional[str] = None, **kwargs) -> None:
        """Save model checkpoint.
        
        Args:
            path: Checkpoint path (if None, auto-generate)
            **kwargs: Additional items to save
        """
        if path is None:
            path = self.checkpoint_dir / f"checkpoint_epoch_{self.epoch}.pt"
        else:
            path = Path(path)
        
        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_history": self.train_history,
            "val_history": self.val_history,
            "best_val_metric": self.best_val_metric,
        }
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        checkpoint.update(kwargs)
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load model checkpoint.
        
        Args:
            path: Checkpoint path
            
        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if "scaler_state_dict" in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        # Restore training state
        if "epoch" in checkpoint:
            self.epoch = checkpoint["epoch"]
        if "train_history" in checkpoint:
            self.train_history = checkpoint["train_history"]
        if "val_history" in checkpoint:
            self.val_history = checkpoint["val_history"]
        if "best_val_metric" in checkpoint:
            self.best_val_metric = checkpoint["best_val_metric"]
        
        logger.info(f"Loaded checkpoint from {path} (epoch {self.epoch})")
        
        return checkpoint
    
    def save_metrics(self, path: Optional[str] = None):
        """Save training metrics to JSON.
        
        Args:
            path: Path to save metrics (if None, auto-generate)
        """
        if path is None:
            path = self.checkpoint_dir / "metrics.json"
        else:
            path = Path(path)
        
        metrics = {
            "train_history": self.train_history,
            "val_history": self.val_history,
            "best_val_metric": self.best_val_metric,
            "final_epoch": self.epoch
        }
        
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved metrics to {path}")


class TrainerClf(TrainerBase):
    """Trainer for classification tasks with BEST MODEL SAVING."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        num_classes: int = 2,
        class_weights: Optional[torch.Tensor] = None,
        use_focal_loss: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        use_balanced_sampler: bool = False,
        **kwargs
    ):
        """Initialize classification trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            num_classes: Number of classes
            class_weights: Class weights for loss
            use_focal_loss: Whether to use focal loss
            focal_alpha: Focal loss alpha parameter
            focal_gamma: Focal loss gamma parameter
            use_balanced_sampler: Whether to use balanced sampling
            **kwargs: Additional arguments for base trainer
        """
        # Setup criterion
        if use_focal_loss:
            criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            **kwargs
        )
        
        self.num_classes = num_classes
        self.use_balanced_sampler = use_balanced_sampler
    
    def compute_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute classification metrics.
        
        Args:
            outputs: Model outputs (logits)
            targets: Ground truth labels
            
        Returns:
            Dictionary of metrics
        """
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == targets).float().mean().item()
        
        # Per-class accuracy
        per_class_acc = []
        for c in range(self.num_classes):
            mask = targets == c
            if mask.sum() > 0:
                class_acc = (predictions[mask] == c).float().mean().item()
                per_class_acc.append(class_acc)
        
        metrics = {
            "accuracy": accuracy,
            "mean_per_class_accuracy": np.mean(per_class_acc) if per_class_acc else 0.0
        }
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate model with classification metrics."""
        val_results = super().validate()
        
        if val_results:
            outputs = val_results.pop("outputs")
            targets = val_results.pop("targets")
            
            # Compute classification metrics
            clf_metrics = self.compute_metrics(outputs, targets)
            val_results.update(clf_metrics)
        
        return val_results
    
    def fit(
        self,
        num_epochs: int,
        save_best: bool = True,
        early_stopping_patience: int = 5,
        monitor_metric: str = "accuracy",
        monitor_mode: str = "max"
    ) -> Dict[str, List[float]]:
        """Full training loop for classification with BEST MODEL SAVING.
        
        Args:
            num_epochs: Number of epochs to train
            save_best: Whether to save best model
            early_stopping_patience: Patience for early stopping
            monitor_metric: Metric to monitor for best model
            monitor_mode: 'min' or 'max' for metric
            
        Returns:
            Training history
        """
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            mode=monitor_mode,
            verbose=True
        )
        
        best_metric = -np.inf if monitor_mode == "max" else np.inf
        best_epoch = 0
        
        logger.info("\n" + "="*70)
        logger.info("TRAINING STARTED")
        logger.info("="*70)
        logger.info(f"Monitor metric: {monitor_metric} ({monitor_mode})")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
        logger.info(f"Early stopping patience: {early_stopping_patience}")
        logger.info("="*70 + "\n")
        
        for epoch in range(1, num_epochs + 1):
            self.epoch = epoch
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            self.train_history.append(train_metrics)
            
            # Validate
            val_metrics = self.validate()
            self.val_history.append(val_metrics)
            
            # Log metrics
            elapsed = time.time() - start_time
            logger.info(f"Epoch {epoch}/{num_epochs} ({elapsed:.2f}s)")
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}")
            if val_metrics:
                logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                          f"Acc: {val_metrics['accuracy']:.4f}")
            
            # Check for best model
            if val_metrics and monitor_metric in val_metrics:
                current_metric = val_metrics[monitor_metric]
                is_best = (monitor_mode == "max" and current_metric > best_metric) or \
                         (monitor_mode == "min" and current_metric < best_metric)
                
                if is_best:
                    best_metric = current_metric
                    best_epoch = epoch
                    self.best_val_metric = current_metric
                    
                    # ALWAYS save best model (not just when save_best=True)
                    best_model_path = self.checkpoint_dir / "best_model.pt"
                    logger.info(f"✓ New best {monitor_metric}: {current_metric:.6f} - Saving to {best_model_path}")
                    self.save_checkpoint(best_model_path)
                    
                    # Also save with standard name for backward compatibility
                    if save_best:
                        self.save_checkpoint(self.checkpoint_dir / "model.pt")
                        self.save_metrics(self.checkpoint_dir / "metrics.json")
                
                # Early stopping
                if early_stopping(current_metric):
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
            
            # Save last checkpoint
            last_checkpoint_path = self.checkpoint_dir / "last_checkpoint.pt"
            self.save_checkpoint(last_checkpoint_path)
        
        # Training complete - log summary
        logger.info("\n" + "="*70)
        logger.info("TRAINING COMPLETE")
        logger.info("="*70)
        logger.info(f"Total epochs: {self.epoch}")
        logger.info(f"Best epoch: {best_epoch}")
        logger.info(f"Best {monitor_metric}: {best_metric:.6f}")
        logger.info(f"Best model saved to: {self.checkpoint_dir / 'best_model.pt'}")
        logger.info(f"Last checkpoint saved to: {self.checkpoint_dir / 'last_checkpoint.pt'}")
        logger.info("="*70 + "\n")
        
        # Save final metrics
        self.save_metrics(self.checkpoint_dir / "metrics.json")
        
        return {
            "train_history": self.train_history,
            "val_history": self.val_history,
            "best_epoch": best_epoch,
            "best_metric": best_metric
        }


class TrainerReg(TrainerBase):
    """Trainer for regression tasks with BEST MODEL SAVING."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        loss_type: str = "mse",
        **kwargs
    ):
        """Initialize regression trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            loss_type: Loss type ('mse', 'mae', 'huber')
            **kwargs: Additional arguments for base trainer
        """
        # Setup criterion
        if loss_type == "mse":
            criterion = nn.MSELoss()
        elif loss_type == "mae":
            criterion = nn.L1Loss()
        elif loss_type == "huber":
            criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            **kwargs
        )
    
    def compute_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute regression metrics.
        
        Args:
            outputs: Model outputs
            targets: Ground truth values
            
        Returns:
            Dictionary of metrics
        """
        mse = F.mse_loss(outputs, targets).item()
        mae = F.l1_loss(outputs, targets).item()
        rmse = np.sqrt(mse)
        
        # R-squared
        ss_res = ((targets - outputs) ** 2).sum()
        ss_tot = ((targets - targets.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        
        metrics = {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2.item()
        }
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate model with regression metrics."""
        val_results = super().validate()
        
        if val_results:
            outputs = val_results.pop("outputs")
            targets = val_results.pop("targets")
            
            # Compute regression metrics
            reg_metrics = self.compute_metrics(outputs, targets)
            val_results.update(reg_metrics)
        
        return val_results
    
    def fit(
        self,
        num_epochs: int,
        save_best: bool = True,
        early_stopping_patience: int = 5,
        monitor_metric: str = "mse",
        monitor_mode: str = "min"
    ) -> Dict[str, List[float]]:
        """Full training loop for regression with BEST MODEL SAVING.
        
        Args:
            num_epochs: Number of epochs to train
            save_best: Whether to save best model
            early_stopping_patience: Patience for early stopping
            monitor_metric: Metric to monitor for best model
            monitor_mode: 'min' or 'max' for metric
            
        Returns:
            Training history
        """
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            mode=monitor_mode,
            verbose=True
        )
        
        best_metric = -np.inf if monitor_mode == "max" else np.inf
        best_epoch = 0
        
        logger.info("\n" + "="*70)
        logger.info("TRAINING STARTED")
        logger.info("="*70)
        logger.info(f"Monitor metric: {monitor_metric} ({monitor_mode})")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
        logger.info(f"Early stopping patience: {early_stopping_patience}")
        logger.info("="*70 + "\n")
        
        for epoch in range(1, num_epochs + 1):
            self.epoch = epoch
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            self.train_history.append(train_metrics)
            
            # Validate
            val_metrics = self.validate()
            self.val_history.append(val_metrics)
            
            # Log metrics
            elapsed = time.time() - start_time
            logger.info(f"Epoch {epoch}/{num_epochs} ({elapsed:.2f}s)")
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}")
            if val_metrics:
                logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                          f"MSE: {val_metrics.get('mse', 0):.4f}, "
                          f"MAE: {val_metrics.get('mae', 0):.4f}")
            
            # Check for best model
            if val_metrics and monitor_metric in val_metrics:
                current_metric = val_metrics[monitor_metric]
                is_best = (monitor_mode == "max" and current_metric > best_metric) or \
                         (monitor_mode == "min" and current_metric < best_metric)
                
                if is_best:
                    best_metric = current_metric
                    best_epoch = epoch
                    self.best_val_metric = current_metric
                    
                    # ALWAYS save best model (not just when save_best=True)
                    best_model_path = self.checkpoint_dir / "best_model.pt"
                    logger.info(f"✓ New best {monitor_metric}: {current_metric:.6f} - Saving to {best_model_path}")
                    self.save_checkpoint(best_model_path)
                    
                    # Also save with standard name for backward compatibility
                    if save_best:
                        self.save_checkpoint(self.checkpoint_dir / "model.pt")
                        self.save_metrics(self.checkpoint_dir / "metrics.json")
                
                # Early stopping
                if early_stopping(current_metric):
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
            
            # Save last checkpoint
            last_checkpoint_path = self.checkpoint_dir / "last_checkpoint.pt"
            self.save_checkpoint(last_checkpoint_path)
        
        # Training complete - log summary
        logger.info("\n" + "="*70)
        logger.info("TRAINING COMPLETE")
        logger.info("="*70)
        logger.info(f"Total epochs: {self.epoch}")
        logger.info(f"Best epoch: {best_epoch}")
        logger.info(f"Best {monitor_metric}: {best_metric:.6f}")
        logger.info(f"Best model saved to: {self.checkpoint_dir / 'best_model.pt'}")
        logger.info(f"Last checkpoint saved to: {self.checkpoint_dir / 'last_checkpoint.pt'}")
        logger.info("="*70 + "\n")
        
        # Save final metrics
        self.save_metrics(self.checkpoint_dir / "metrics.json")
        
        return {
            "train_history": self.train_history,
            "val_history": self.val_history,
            "best_epoch": best_epoch,
            "best_metric": best_metric
        }


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = "adamw",
    lr: float = 5e-4,
    weight_decay: float = 0.01,
    **kwargs
) -> Optimizer:
    """Create optimizer.
    
    Args:
        model: Model to optimize
        optimizer_type: Type of optimizer
        lr: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer arguments
        
    Returns:
        Optimizer instance
    """
    if optimizer_type.lower() == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_type.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_type.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "cosine",
    num_epochs: int = 10,
    warmup_epochs: int = 1,
    **kwargs
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler with warmup.
    
    Args:
        optimizer: Optimizer
        scheduler_type: Type of scheduler
        num_epochs: Total number of epochs
        warmup_epochs: Number of warmup epochs
        **kwargs: Additional scheduler arguments
        
    Returns:
        Scheduler instance
    """
    total_steps = num_epochs
    warmup_steps = warmup_epochs
    
    # Create warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=warmup_steps
    )
    
    # Create main scheduler
    if scheduler_type.lower() == "cosine":
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            **kwargs
        )
    elif scheduler_type.lower() == "linear":
        main_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=total_steps - warmup_steps
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    # Combine with warmup
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_steps]
    )
    
    return scheduler
