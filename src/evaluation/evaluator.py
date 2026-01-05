"""Complete evaluation pipeline for HAR models.

This module provides end-to-end evaluation with comprehensive metric reporting.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import (
    compute_metrics,
    compute_metrics_from_logits,
    per_class_metrics,
    confusion_matrix,
    classification_report,
)

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Base evaluator class for model evaluation.

    Args:
        model: Model to evaluate
        device: Device to run evaluation on
        label_names: Optional names for each class

    Example:
        >>> evaluator = Evaluator(model, device=torch.device('cuda'))
        >>> results = evaluator.evaluate(test_loader)
        >>> print(f"Test accuracy: {results['accuracy']:.4f}")
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        label_names: Optional[List[str]] = None,
    ) -> None:
        """Initialize evaluator."""
        self.model = model
        self.device = device
        self.label_names = label_names

        # Move model to device
        self.model = self.model.to(device)

        logger.info(f"Initialized Evaluator on device: {device}")

    @torch.no_grad()
    def predict(
        self,
        dataloader: DataLoader,
        return_logits: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Generate predictions for entire dataset.

        Args:
            dataloader: DataLoader with evaluation data
            return_logits: If True, also return raw logits

        Returns:
            Tuple of:
            - predictions: Predicted labels (N,)
            - labels: Ground truth labels (N,)
            - logits: Raw model outputs (N, K) if return_logits=True, else None

        Example:
            >>> preds, labels, logits = evaluator.predict(test_loader, return_logits=True)
            >>> print(f"Predicted {len(preds)} samples")
        """
        self.model.eval()

        all_preds = []
        all_labels = []
        all_logits = [] if return_logits else None

        for batch in tqdm(dataloader, desc="Predicting", leave=False):
            # Move to device
            inputs = batch["signal"].to(self.device)
            labels = batch["label"].to(self.device)

            # Forward pass
            logits = self.model(inputs)

            # Get predictions
            preds = torch.argmax(logits, dim=1)

            # Collect results
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            if return_logits:
                all_logits.append(logits.cpu().numpy())

        # Concatenate all batches
        predictions = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        logits = np.concatenate(all_logits) if return_logits else None

        logger.info(f"Generated predictions for {len(predictions)} samples")

        return predictions, labels, logits

    def evaluate(
        self,
        dataloader: DataLoader,
        include_per_class: bool = True,
        include_confusion_matrix: bool = True,
        include_classification_report: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate model on dataset.

        Args:
            dataloader: DataLoader with evaluation data
            include_per_class: Include per-class metrics
            include_confusion_matrix: Include confusion matrix
            include_classification_report: Include classification report

        Returns:
            Dictionary containing:
            - All aggregate metrics (accuracy, F1, etc.)
            - Optional per-class metrics
            - Optional confusion matrix
            - Optional classification report

        Example:
            >>> results = evaluator.evaluate(test_loader)
            >>> print(f"Accuracy: {results['accuracy']:.4f}")
            >>> print(f"Macro F1: {results['macro_f1']:.4f}")
        """
        # Generate predictions
        predictions, labels, logits = self.predict(dataloader, return_logits=True)

        # Get unique labels
        unique_labels = np.unique(np.concatenate([labels, predictions]))

        # Compute aggregate metrics
        metrics = compute_metrics(
            labels,
            predictions,
            labels=unique_labels.tolist(),
            target_names=self.label_names,
        )

        # Add per-class metrics if requested
        if include_per_class:
            per_class = per_class_metrics(
                labels,
                predictions,
                labels=unique_labels.tolist(),
                target_names=self.label_names,
            )
            metrics["per_class"] = per_class

        # Add confusion matrix if requested
        if include_confusion_matrix:
            cm = confusion_matrix(labels, predictions, labels=unique_labels.tolist())
            metrics["confusion_matrix"] = cm

            # Also add normalized version
            cm_normalized = confusion_matrix(
                labels, predictions, labels=unique_labels.tolist(), normalize="true"
            )
            metrics["confusion_matrix_normalized"] = cm_normalized

        # Add classification report if requested
        if include_classification_report:
            report = classification_report(
                labels,
                predictions,
                labels=unique_labels.tolist(),
                target_names=self.label_names,
                output_dict=True,
            )
            metrics["classification_report"] = report

        logger.info(
            f"Evaluation complete: "
            f"Accuracy={metrics['accuracy']:.4f}, "
            f"Balanced Acc={metrics['balanced_accuracy']:.4f}, "
            f"Macro F1={metrics['macro_f1']:.4f}"
        )

        return metrics

    def save_predictions(
        self,
        dataloader: DataLoader,
        output_path: str,
    ) -> None:
        """
        Save predictions to file.

        Args:
            dataloader: DataLoader with evaluation data
            output_path: Path to save predictions (.npz format)

        Example:
            >>> evaluator.save_predictions(test_loader, "predictions.npz")
        """
        predictions, labels, logits = self.predict(dataloader, return_logits=True)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            output_path,
            predictions=predictions,
            labels=labels,
            logits=logits,
        )

        logger.info(f"Saved predictions to {output_path}")


class ModelEvaluator:
    """
    High-level evaluator with subject-level aggregation and cross-validation support.

    Args:
        model: Model to evaluate
        device: Device to run evaluation on
        label_names: Optional names for each class

    Example:
        >>> evaluator = ModelEvaluator(model, device=torch.device('cuda'))
        >>> results = evaluator.evaluate_with_subjects(
        ...     test_loader, subject_ids
        ... )
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        label_names: Optional[List[str]] = None,
    ) -> None:
        """Initialize model evaluator."""
        self.evaluator = Evaluator(model, device, label_names)
        self.label_names = label_names

    def evaluate_with_subjects(
        self,
        dataloader: DataLoader,
        subject_ids: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Evaluate with subject-level aggregation.

        Computes both window-level and subject-level metrics.

        Args:
            dataloader: DataLoader with evaluation data
            subject_ids: Subject ID for each sample (N,)

        Returns:
            Dictionary with:
            - window_metrics: Window-level metrics
            - subject_metrics: Subject-level metrics (averaged per subject)

        Example:
            >>> results = evaluator.evaluate_with_subjects(test_loader, subject_ids)
            >>> print(f"Window-level accuracy: {results['window_metrics']['accuracy']:.4f}")
            >>> print(f"Subject-level accuracy: {results['subject_metrics']['accuracy']:.4f}")
        """
        # Get predictions
        predictions, labels, _ = self.evaluator.predict(dataloader)

        subject_ids = np.asarray(subject_ids)

        if len(subject_ids) != len(predictions):
            raise ValueError(
                f"subject_ids length ({len(subject_ids)}) must match "
                f"predictions length ({len(predictions)})"
            )

        # Window-level metrics
        window_metrics = compute_metrics(labels, predictions)

        # Subject-level metrics
        subject_metrics = self._compute_subject_metrics(
            labels, predictions, subject_ids
        )

        logger.info(
            f"Window-level: Acc={window_metrics['accuracy']:.4f}, "
            f"F1={window_metrics['macro_f1']:.4f}"
        )
        logger.info(
            f"Subject-level: Acc={subject_metrics['accuracy']:.4f}, "
            f"F1={subject_metrics['macro_f1']:.4f}"
        )

        return {
            "window_metrics": window_metrics,
            "subject_metrics": subject_metrics,
        }

    def _compute_subject_metrics(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        subject_ids: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute subject-level metrics.

        For each subject, compute per-subject accuracy, then average across subjects.

        Args:
            labels: Ground truth labels (N,)
            predictions: Predicted labels (N,)
            subject_ids: Subject ID for each sample (N,)

        Returns:
            Dictionary of subject-level metrics
        """
        unique_subjects = np.unique(subject_ids)

        subject_accuracies = []
        subject_f1s = []

        for subject in unique_subjects:
            # Get subject's samples
            mask = subject_ids == subject
            subj_labels = labels[mask]
            subj_preds = predictions[mask]

            # Compute metrics
            subj_metrics = compute_metrics(subj_labels, subj_preds)

            subject_accuracies.append(subj_metrics["accuracy"])
            subject_f1s.append(subj_metrics["macro_f1"])

        # Average across subjects
        return {
            "accuracy": np.mean(subject_accuracies),
            "macro_f1": np.mean(subject_f1s),
            "accuracy_std": np.std(subject_accuracies),
            "macro_f1_std": np.std(subject_f1s),
            "n_subjects": len(unique_subjects),
        }

    def cross_validate(
        self,
        splits: List[Tuple[DataLoader, DataLoader, DataLoader]],
        metric: str = "macro_f1",
    ) -> Dict[str, Any]:
        """
        Perform cross-validation evaluation.

        Args:
            splits: List of (train_loader, val_loader, test_loader) tuples
            metric: Metric to track ("accuracy", "macro_f1", etc.)

        Returns:
            Dictionary with:
            - fold_results: List of results per fold
            - mean_metrics: Mean metrics across folds
            - std_metrics: Std metrics across folds

        Example:
            >>> splits = [...]  # List of data splits
            >>> cv_results = evaluator.cross_validate(splits, metric="macro_f1")
            >>> print(f"CV F1: {cv_results['mean_metrics']['macro_f1']:.4f} ± {cv_results['std_metrics']['macro_f1']:.4f}")
        """
        fold_results = []

        for fold_idx, (_, _, test_loader) in enumerate(splits):
            logger.info(f"Evaluating fold {fold_idx + 1}/{len(splits)}")

            # Evaluate on test set
            results = self.evaluator.evaluate(test_loader)

            fold_results.append(results)

            logger.info(
                f"Fold {fold_idx + 1}: {metric}={results[metric]:.4f}"
            )

        # Aggregate results
        mean_metrics = {}
        std_metrics = {}

        # Get all metric keys from first fold
        metric_keys = [k for k, v in fold_results[0].items() if isinstance(v, (int, float))]

        for key in metric_keys:
            values = [fold[key] for fold in fold_results]
            mean_metrics[key] = np.mean(values)
            std_metrics[key] = np.std(values)

        logger.info(
            f"Cross-validation complete: "
            f"{metric}={mean_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}"
        )

        return {
            "fold_results": fold_results,
            "mean_metrics": mean_metrics,
            "std_metrics": std_metrics,
            "n_folds": len(splits),
        }


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    label_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Convenience function for quick model evaluation.

    Args:
        model: Model to evaluate
        dataloader: DataLoader with evaluation data
        device: Device to run on
        label_names: Optional class names

    Returns:
        Dictionary of evaluation metrics

    Example:
        >>> results = evaluate_model(model, test_loader, device)
        >>> print(f"Accuracy: {results['accuracy']:.4f}")
    """
    evaluator = Evaluator(model, device, label_names)
    results = evaluator.evaluate(dataloader)
    return results
