#!/usr/bin/env python3
"""Evaluation script for TTM-HAR.

This script provides comprehensive model evaluation with:
- Checkpoint loading
- Test set evaluation
- Comprehensive metrics computation
- Confusion matrix and classification report
- Results visualization and export

Usage:
    # Evaluate best checkpoint
    python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pt

    # Evaluate with custom config
    python scripts/evaluate.py --checkpoint model.pt --config configs/default.yaml

    # Save predictions
    python scripts/evaluate.py --checkpoint model.pt --save_predictions

    # Generate visualizations
    python scripts/evaluate.py --checkpoint model.pt --plot
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.utils.device import get_device
from src.utils.checkpointing import load_checkpoint
from src.data.datamodule import HARDataModule
from src.models.model_factory import load_model_from_checkpoint
from src.evaluation import (
    Evaluator,
    ModelEvaluator,
    plot_confusion_matrix,
    plot_per_class_metrics,
    analyze_predictions,
    compute_subject_level_metrics,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate TTM-HAR model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate best checkpoint
  python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pt

  # Evaluate with visualizations
  python scripts/evaluate.py --checkpoint model.pt --plot

  # Save predictions and analysis
  python scripts/evaluate.py --checkpoint model.pt --save_predictions --save_analysis

  # Evaluate on validation set instead of test set
  python scripts/evaluate.py --checkpoint model.pt --split val
        """
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (if not in checkpoint)",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on (default: test)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Output directory for results (default: evaluation_results)",
    )

    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save model predictions to file",
    )

    parser.add_argument(
        "--save_analysis",
        action="store_true",
        help="Save detailed analysis results",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate and save visualization plots",
    )

    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


def setup_output_directory(output_dir: Path) -> dict:
    """
    Create output directory structure for evaluation results.

    Args:
        output_dir: Base output directory

    Returns:
        Dictionary with paths to subdirectories
    """
    output_dir = Path(output_dir)

    paths = {
        "output_dir": output_dir,
        "predictions": output_dir / "predictions",
        "plots": output_dir / "plots",
        "analysis": output_dir / "analysis",
    }

    # Create all directories
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Created output directory structure at {output_dir}")

    return paths


def save_metrics_json(metrics: dict, output_path: Path) -> None:
    """
    Save metrics to JSON file.

    Args:
        metrics: Dictionary of metrics
        output_path: Path to save JSON file
    """
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    serializable_metrics = convert_to_serializable(metrics)

    with open(output_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)

    logger.info(f"Saved metrics to {output_path}")


def print_metrics_summary(metrics: dict, label_names: list = None) -> None:
    """
    Print formatted metrics summary.

    Args:
        metrics: Dictionary of evaluation metrics
        label_names: List of class names
    """
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    # Overall metrics
    print("\nOverall Metrics:")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Macro F1:          {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:       {metrics['weighted_f1']:.4f}")
    print(f"  Macro Precision:   {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:      {metrics['macro_recall']:.4f}")

    # Per-class metrics if available
    if 'per_class' in metrics:
        print("\nPer-Class Metrics:")
        per_class = metrics['per_class']

        # Table header
        print(f"  {'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("  " + "-" * 61)

        for i in range(len(per_class['precision'])):
            class_name = label_names[i] if label_names else f"Class {i}"
            precision = per_class['precision'][i]
            recall = per_class['recall'][i]
            f1 = per_class['f1'][i]
            support = per_class['support'][i]

            print(f"  {class_name:<15} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<10}")

    print("=" * 80)


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    log_file = Path(args.output_dir) / "evaluate.log"
    setup_logging(
        level=args.log_level,
        log_file=str(log_file),
    )

    logger.info("=" * 80)
    logger.info("TTM-HAR Evaluation Script")
    logger.info("=" * 80)

    # Setup output directories
    paths = setup_output_directory(args.output_dir)

    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint, device=device)

    # Get config from checkpoint or load separately
    if 'config' in checkpoint:
        config = checkpoint['config']
        logger.info("Using config from checkpoint")
    elif args.config:
        logger.info(f"Loading config from {args.config}")
        config = load_config(args.config)
    else:
        raise ValueError("No config found in checkpoint and --config not provided")

    # Create data module
    logger.info(f"Creating data module (split: {args.split})...")
    data_module = HARDataModule(config=config)
    data_module.setup()

    # Get appropriate dataloader
    if args.split == "train":
        dataloader = data_module.train_dataloader()
    elif args.split == "val":
        dataloader = data_module.val_dataloader()
    else:  # test
        dataloader = data_module.test_dataloader()

    logger.info(f"Loaded {len(dataloader.dataset)} samples ({len(dataloader)} batches)")

    # Get label names
    label_names = list(data_module.get_label_map().values())
    logger.info(f"Classes: {label_names}")

    # Load model from checkpoint
    logger.info("Loading model from checkpoint...")
    model = load_model_from_checkpoint(args.checkpoint, config=config, device=device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {total_params:,} parameters")

    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        device=device,
        label_names=label_names,
    )

    # Run evaluation
    logger.info("Running evaluation...")
    logger.info("=" * 80)

    results = evaluator.evaluator.evaluate(
        dataloader,
        include_per_class=True,
        include_confusion_matrix=True,
        include_classification_report=True,
    )

    # Print summary
    print_metrics_summary(results, label_names)

    # Save metrics
    metrics_file = paths["output_dir"] / "metrics.json"
    save_metrics_json(results, metrics_file)

    # Save predictions if requested
    if args.save_predictions:
        logger.info("Saving predictions...")
        predictions_file = paths["predictions"] / f"{args.split}_predictions.npz"
        evaluator.evaluator.save_predictions(dataloader, str(predictions_file))

    # Save detailed analysis if requested
    if args.save_analysis:
        logger.info("Generating detailed analysis...")

        # Get predictions for analysis
        predictions, labels, _ = evaluator.evaluator.predict(dataloader)

        # Run analysis
        analysis = analyze_predictions(
            labels,
            predictions,
            class_names=label_names,
        )

        # Save analysis
        analysis_file = paths["analysis"] / "detailed_analysis.json"
        save_metrics_json(analysis, analysis_file)

        # Get subject IDs if available
        if hasattr(dataloader.dataset, 'subject_ids'):
            subject_ids = dataloader.dataset.subject_ids

            # Compute subject-level metrics
            subject_metrics = compute_subject_level_metrics(
                labels,
                predictions,
                subject_ids,
            )

            # Save subject-level metrics
            subject_file = paths["analysis"] / "subject_level_metrics.json"
            save_metrics_json(subject_metrics, subject_file)

            print("\nSubject-Level Metrics:")
            print(f"  Mean Accuracy: {subject_metrics['mean_accuracy']:.4f} ± {subject_metrics['std_accuracy']:.4f}")
            print(f"  Mean F1:       {subject_metrics['mean_f1']:.4f} ± {subject_metrics['std_f1']:.4f}")
            print(f"  N Subjects:    {subject_metrics['n_subjects']}")

    # Generate plots if requested
    if args.plot:
        logger.info("Generating visualization plots...")

        # Plot confusion matrix
        if 'confusion_matrix' in results:
            cm_file = paths["plots"] / "confusion_matrix.png"
            fig = plot_confusion_matrix(
                results['confusion_matrix'],
                class_names=label_names,
                normalize=True,
                title=f"Confusion Matrix ({args.split.capitalize()} Set)",
                save_path=str(cm_file),
            )
            plt.close(fig)

        # Plot per-class metrics
        if 'per_class' in results:
            per_class_file = paths["plots"] / "per_class_metrics.png"
            fig = plot_per_class_metrics(
                results['per_class'],
                class_names=label_names,
                save_path=str(per_class_file),
            )
            plt.close(fig)

        logger.info(f"Plots saved to {paths['plots']}")

    # Print classification report
    if 'classification_report' in results:
        print("\nClassification Report:")
        print("=" * 80)

        from src.evaluation.metrics import classification_report

        # Get predictions for report
        predictions, labels, _ = evaluator.evaluator.predict(dataloader)

        report_str = classification_report(
            labels,
            predictions,
            target_names=label_names,
            output_dict=False,
        )
        print(report_str)
        print("=" * 80)

    logger.info(f"\nAll results saved to {args.output_dir}")
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
