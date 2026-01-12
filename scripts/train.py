#!/usr/bin/env python3
"""Training script for TTM-HAR.

This script provides a complete training pipeline with:
- Configuration management
- Data loading and preprocessing
- Model creation and training
- Checkpointing and callbacks
- Logging and monitoring

Usage:
    # Train with default config
    python scripts/train.py

    # Train with custom config
    python scripts/train.py --config configs/my_experiment.yaml

    # Override specific parameters
    python scripts/train.py --config configs/default.yaml --training.epochs 100 --training.batch_size 128

    # Resume from checkpoint
    python scripts/train.py --resume checkpoints/epoch_10.pt
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch

from src.utils.config import load_config, merge_config_overrides
from src.utils.logging import setup_logging
from src.utils.reproducibility import set_seed
from src.utils.device import get_device
from src.data.datamodule import HARDataModule
from src.models.model_factory import create_model
from src.training.trainer import Trainer
from src.training.callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    TensorBoardCallback,
    LearningRateLoggerCallback,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train TTM-HAR model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default configuration
  python scripts/train.py

  # Train with custom config
  python scripts/train.py --config configs/my_config.yaml

  # Override config parameters
  python scripts/train.py --training.epochs 100 --training.batch_size 64

  # Resume training from checkpoint
  python scripts/train.py --resume checkpoints/epoch_10.pt

  # Specify output directory
  python scripts/train.py --output_dir experiments/exp_001
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file (default: configs/default.yaml)",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for checkpoints and logs (default: outputs)",
    )

    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--no_tensorboard",
        action="store_true",
        help="Disable TensorBoard logging",
    )

    parser.add_argument(
        "--no_checkpointing",
        action="store_true",
        help="Disable checkpoint saving",
    )

    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides in format key=value (e.g., training.epochs=100)",
    )

    return parser.parse_args()


def setup_output_directories(output_dir: Path) -> dict:
    """
    Create output directory structure.

    Args:
        output_dir: Base output directory

    Returns:
        Dictionary with paths to subdirectories
    """
    output_dir = Path(output_dir)

    paths = {
        "output_dir": output_dir,
        "checkpoints": output_dir / "checkpoints",
        "logs": output_dir / "logs",
        "tensorboard": output_dir / "tensorboard",
    }

    # Create all directories
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Created output directory structure at {output_dir}")

    return paths


def create_callbacks(config: dict, paths: dict, args: argparse.Namespace) -> list:
    """
    Create training callbacks.

    Args:
        config: Configuration dictionary
        paths: Dictionary of output paths
        args: Command-line arguments

    Returns:
        List of callback instances
    """
    callbacks = []

    # Learning rate logger (always enabled)
    callbacks.append(LearningRateLoggerCallback())

    # Checkpointing callback
    if not args.no_checkpointing:
        checkpoint_callback = CheckpointCallback(
            checkpoint_dir=str(paths["checkpoints"]),
            monitor_metric="balanced_accuracy",
            mode="max",
            save_every_n_epochs=config.get("training", {}).get("save_every_n_epochs", 5),
            keep_last_n=config.get("training", {}).get("keep_last_n", 3),
            save_best=True,
        )
        callbacks.append(checkpoint_callback)

    # Early stopping callback
    early_stopping_config = config.get("training", {}).get("early_stopping", {})
    if early_stopping_config.get("enabled", False):
        early_stopping_callback = EarlyStoppingCallback(
            monitor_metric=early_stopping_config.get("monitor_metric", "balanced_accuracy"),
            patience=early_stopping_config.get("patience", 10),
            mode=early_stopping_config.get("mode", "max"),
            min_delta=early_stopping_config.get("min_delta", 0.0),
        )
        callbacks.append(early_stopping_callback)

    # TensorBoard callback
    if not args.no_tensorboard:
        tensorboard_callback = TensorBoardCallback(
            log_dir=str(paths["tensorboard"]),
            log_every_n_steps=config.get("training", {}).get("log_every_n_steps", 10),
        )
        callbacks.append(tensorboard_callback)

    logger.info(f"Created {len(callbacks)} callbacks")

    return callbacks


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    log_file = Path(args.output_dir) / "logs" / "train.log"
    setup_logging(
        level=args.log_level,
        log_file=str(log_file),
    )

    logger.info("=" * 80)
    logger.info("TTM-HAR Training Script")
    logger.info("=" * 80)

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Apply overrides
    if args.overrides:
        logger.info(f"Applying {len(args.overrides)} config overrides")
        config = merge_config_overrides(config, args.overrides)

    # Set random seed
    seed = config.get("experiment", {}).get("seed", 42)
    set_seed(seed)
    logger.info(f"Set random seed to {seed}")

    # Setup output directories
    paths = setup_output_directories(args.output_dir)

    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Create data module
    logger.info("Creating data module...")
    data_module = HARDataModule(config=config)
    data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    logger.info(
        f"Data loaded: {len(train_loader)} train batches, "
        f"{len(val_loader)} val batches"
    )

    # Add class weights to config if available
    if hasattr(data_module, 'class_weights') and data_module.class_weights is not None:
        config["class_weights"] = data_module.class_weights.tolist()
        logger.info(f"Using class weights: {config['class_weights']}")

    # Create model
    logger.info("Creating model...")
    model = create_model(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model created: {total_params:,} total parameters, "
        f"{trainable_params:,} trainable"
    )

    # Create callbacks
    callbacks = create_callbacks(config, paths, args)

    # Create trainer
    logger.info("Creating trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        callbacks=callbacks,
    )

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming training from {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    logger.info("Starting training...")
    logger.info("=" * 80)

    history = trainer.train()

    logger.info("=" * 80)
    logger.info("Training complete!")
    logger.info(f"Total epochs: {history['num_epochs']}")
    logger.info(f"Total steps: {history['total_steps']}")

    # Save training history to JSON
    import json
    history_path = paths["output_dir"] / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Saved training history to {history_path}")

    # Save final model
    if not args.no_checkpointing:
        final_checkpoint_path = paths["checkpoints"] / "final_model.pt"
        trainer.save_checkpoint(str(final_checkpoint_path))
        logger.info(f"Saved final model to {final_checkpoint_path}")

    # Print best metrics
    if history['val_metrics']:
        best_epoch = max(
            range(len(history['val_metrics'])),
            key=lambda i: history['val_metrics'][i]['balanced_accuracy']
        )
        best_metrics = history['val_metrics'][best_epoch]

        logger.info("=" * 80)
        logger.info(f"Best validation metrics (epoch {best_epoch + 1}):")
        logger.info(f"  Balanced Accuracy: {best_metrics['balanced_accuracy']:.4f}")
        logger.info(f"  Macro F1: {best_metrics['macro_f1']:.4f}")
        logger.info(f"  Weighted F1: {best_metrics['weighted_f1']:.4f}")
        logger.info(f"  Loss: {best_metrics['loss']:.4f}")
        logger.info("=" * 80)

        # Save results summary
        results_summary = {
            "best_epoch": best_epoch + 1,
            "best_metrics": best_metrics,
            "final_metrics": history['val_metrics'][-1] if history['val_metrics'] else None,
            "total_epochs": history['num_epochs'],
            "total_steps": history['total_steps'],
            "config": {
                "model": config.get("model", {}),
                "training": config.get("training", {}),
                "dataset": config.get("dataset", {}),
            },
            "checkpoint_path": str(paths["checkpoints"] / "best_model.pt"),
        }
        
        results_path = paths["output_dir"] / "results_summary.json"
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        logger.info(f"Saved results summary to {results_path}")

    logger.info(f"All outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
