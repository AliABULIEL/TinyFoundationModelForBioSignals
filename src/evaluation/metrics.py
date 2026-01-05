"""Evaluation metrics for Human Activity Recognition.

This module provides all standard HAR metrics with sklearn-compatible implementations.
All metrics use the same interface and match sklearn reference implementations.
"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix as sklearn_confusion_matrix,
    classification_report as sklearn_classification_report,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
    target_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive set of classification metrics.

    Args:
        y_true: Ground truth labels (N,)
        y_pred: Predicted labels (N,)
        labels: List of label integers to include
        target_names: List of label names (for reporting)

    Returns:
        Dictionary containing all computed metrics

    Example:
        >>> y_true = np.array([0, 1, 2, 1, 0])
        >>> y_pred = np.array([0, 1, 1, 1, 0])
        >>> metrics = compute_metrics(y_true, y_pred)
        >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
        Accuracy: 0.8000
    """
    # Validate inputs
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics on empty arrays")

    # Determine labels if not provided
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))

    # Compute all metrics
    metrics = {
        # Overall accuracy
        "accuracy": accuracy(y_true, y_pred),

        # Balanced accuracy (important for imbalanced datasets)
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),

        # Macro-averaged metrics (equal weight per class)
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0, labels=labels
        ),
        "macro_recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0, labels=labels
        ),
        "macro_f1": macro_f1(y_true, y_pred, labels=labels),

        # Weighted metrics (weight by class support)
        "weighted_precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0, labels=labels
        ),
        "weighted_recall": recall_score(
            y_true, y_pred, average="weighted", zero_division=0, labels=labels
        ),
        "weighted_f1": weighted_f1(y_true, y_pred, labels=labels),
    }

    logger.debug(
        f"Computed metrics for {len(y_true)} samples, {len(labels)} classes"
    )

    return metrics


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy.

    Accuracy = (Number of correct predictions) / (Total predictions)

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Accuracy score in [0, 1]

    Example:
        >>> y_true = np.array([0, 1, 2, 1, 0])
        >>> y_pred = np.array([0, 1, 1, 1, 0])
        >>> accuracy(y_true, y_pred)
        0.8
    """
    return accuracy_score(y_true, y_pred)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute balanced accuracy.

    Balanced accuracy is the average of recall obtained on each class.
    This metric is important for imbalanced datasets where standard
    accuracy can be misleading.

    Formula: balanced_acc = mean(recall_per_class)

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Balanced accuracy in [0, 1]

    Example:
        >>> y_true = np.array([0, 0, 0, 1, 1])  # Imbalanced
        >>> y_pred = np.array([0, 0, 0, 1, 0])
        >>> balanced_accuracy(y_true, y_pred)
        0.75  # (1.0 + 0.5) / 2
    """
    return balanced_accuracy_score(y_true, y_pred)


def macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
) -> float:
    """
    Compute macro-averaged F1 score.

    Computes F1 for each class independently, then takes unweighted mean.
    Treats all classes equally regardless of support.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: List of labels to include

    Returns:
        Macro F1 score in [0, 1]

    Example:
        >>> y_true = np.array([0, 1, 2, 1, 0])
        >>> y_pred = np.array([0, 1, 1, 1, 0])
        >>> macro_f1(y_true, y_pred)
        0.8333...
    """
    return f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)


def weighted_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
) -> float:
    """
    Compute weighted F1 score.

    Computes F1 for each class, then takes weighted average by support.
    Gives more weight to classes with more samples.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: List of labels to include

    Returns:
        Weighted F1 score in [0, 1]

    Example:
        >>> y_true = np.array([0, 0, 0, 1, 1])
        >>> y_pred = np.array([0, 0, 0, 1, 0])
        >>> weighted_f1(y_true, y_pred)
        0.8666...
    """
    return f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels)


def per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
    target_names: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute per-class precision, recall, and F1 scores.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: List of label integers to include
        target_names: List of label names (must match labels order)

    Returns:
        Dictionary with keys:
        - 'precision': Per-class precision (K,)
        - 'recall': Per-class recall (K,)
        - 'f1': Per-class F1 scores (K,)
        - 'support': Number of samples per class (K,)
        - 'labels': Class labels (K,)
        - 'names': Class names if provided (K,)

    Example:
        >>> y_true = np.array([0, 1, 2, 1, 0, 2])
        >>> y_pred = np.array([0, 1, 1, 1, 0, 2])
        >>> metrics = per_class_metrics(y_true, y_pred)
        >>> metrics['precision']
        array([1.0, 0.667, 1.0])
    """
    # Validate inputs
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Determine labels
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels)

    # Compute metrics
    precision = precision_score(
        y_true, y_pred, average=None, zero_division=0, labels=labels
    )
    recall = recall_score(
        y_true, y_pred, average=None, zero_division=0, labels=labels
    )
    f1 = f1_score(
        y_true, y_pred, average=None, zero_division=0, labels=labels
    )

    # Compute support (number of true samples per class)
    support = np.array([np.sum(y_true == label) for label in labels])

    result = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "labels": labels,
    }

    # Add names if provided
    if target_names is not None:
        if len(target_names) != len(labels):
            raise ValueError(
                f"target_names length ({len(target_names)}) must match "
                f"labels length ({len(labels)})"
            )
        result["names"] = np.array(target_names)

    return result


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
    normalize: Optional[str] = None,
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: List of labels to include in matrix
        normalize: Normalization mode {'true', 'pred', 'all', None}
            - 'true': Normalize over true labels (rows sum to 1)
            - 'pred': Normalize over predictions (columns sum to 1)
            - 'all': Normalize over all samples (matrix sums to 1)
            - None: Return raw counts

    Returns:
        Confusion matrix (K, K) where K is number of classes
        Entry (i, j) is number of samples with true label i predicted as j

    Example:
        >>> y_true = np.array([0, 1, 2, 1, 0])
        >>> y_pred = np.array([0, 1, 1, 1, 0])
        >>> confusion_matrix(y_true, y_pred)
        array([[2, 0, 0],
               [0, 2, 0],
               [0, 1, 0]])
    """
    cm = sklearn_confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    return cm


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
    target_names: Optional[List[str]] = None,
    output_dict: bool = False,
) -> Union[str, Dict]:
    """
    Generate classification report with per-class and average metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: List of labels to include
        target_names: List of label names
        output_dict: If True, return dict instead of string

    Returns:
        Classification report (string or dict)

    Example:
        >>> y_true = np.array([0, 1, 2, 1, 0, 2])
        >>> y_pred = np.array([0, 1, 1, 1, 0, 2])
        >>> target_names = ['Sleep', 'Sedentary', 'Light']
        >>> print(classification_report(y_true, y_pred, target_names=target_names))
                      precision    recall  f1-score   support

            Sleep       1.00      1.00      1.00         2
        Sedentary       0.67      1.00      0.80         2
            Light       0.00      0.00      0.00         2

         accuracy                           0.67         6
        macro avg       0.56      0.67      0.60         6
     weighted avg       0.56      0.67      0.60         6
    """
    report = sklearn_classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
        output_dict=output_dict,
    )
    return report


def cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Cohen's Kappa coefficient.

    Cohen's Kappa measures inter-annotator agreement, accounting for
    chance agreement. Useful for evaluating HAR against human labels.

    Kappa = (p_o - p_e) / (1 - p_e)
    where:
    - p_o = observed agreement
    - p_e = expected agreement by chance

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Cohen's Kappa in [-1, 1]
        - 1: Perfect agreement
        - 0: Agreement by chance
        - <0: Less than chance agreement

    Example:
        >>> y_true = np.array([0, 1, 2, 1, 0])
        >>> y_pred = np.array([0, 1, 1, 1, 0])
        >>> cohen_kappa(y_true, y_pred)
        0.666...
    """
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(y_true, y_pred)


def matthews_corrcoef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Matthews Correlation Coefficient (MCC).

    MCC is a balanced metric even for imbalanced classes.
    Ranges from -1 (total disagreement) to +1 (perfect prediction).

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        MCC in [-1, 1]

    Example:
        >>> y_true = np.array([0, 1, 1, 0])
        >>> y_pred = np.array([0, 1, 0, 0])
        >>> matthews_corrcoef(y_true, y_pred)
        0.5
    """
    from sklearn.metrics import matthews_corrcoef as sklearn_mcc
    return sklearn_mcc(y_true, y_pred)


def top_k_accuracy(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int = 2,
) -> float:
    """
    Compute top-k accuracy.

    Fraction of samples where true label is in top k predictions.

    Args:
        y_true: Ground truth labels (N,)
        y_score: Predicted scores/probabilities (N, K)
        k: Number of top predictions to consider

    Returns:
        Top-k accuracy in [0, 1]

    Example:
        >>> y_true = np.array([0, 1, 2])
        >>> y_score = np.array([
        ...     [0.8, 0.1, 0.1],  # Top-1: 0 ✓
        ...     [0.3, 0.4, 0.3],  # Top-1: 1 ✓
        ...     [0.6, 0.3, 0.1],  # Top-1: 0, Top-2: [0,1] ✗
        ... ])
        >>> top_k_accuracy(y_true, y_score, k=2)
        0.666...
    """
    from sklearn.metrics import top_k_accuracy_score
    return top_k_accuracy_score(y_true, y_score, k=k)


def compute_metrics_from_logits(
    y_true: np.ndarray,
    logits: np.ndarray,
    labels: Optional[List[int]] = None,
    target_names: Optional[List[str]] = None,
    include_top_k: bool = False,
) -> Dict[str, float]:
    """
    Compute metrics from model logits.

    Converts logits to predictions and computes all standard metrics.
    Optionally includes top-k accuracy metrics.

    Args:
        y_true: Ground truth labels (N,)
        logits: Model logits/scores (N, K)
        labels: List of label integers
        target_names: List of label names
        include_top_k: Whether to compute top-2 and top-3 accuracy

    Returns:
        Dictionary of all computed metrics

    Example:
        >>> y_true = np.array([0, 1, 2])
        >>> logits = np.array([[2.1, 0.5, 0.1],
        ...                    [0.2, 3.0, 0.4],
        ...                    [0.3, 0.2, 1.5]])
        >>> metrics = compute_metrics_from_logits(y_true, logits)
        >>> metrics['accuracy']
        1.0
    """
    # Convert logits to predictions
    y_pred = np.argmax(logits, axis=1)

    # Compute standard metrics
    metrics = compute_metrics(y_true, y_pred, labels=labels, target_names=target_names)

    # Add top-k metrics if requested
    if include_top_k and logits.shape[1] >= 2:
        metrics["top_2_accuracy"] = top_k_accuracy(y_true, logits, k=2)

        if logits.shape[1] >= 3:
            metrics["top_3_accuracy"] = top_k_accuracy(y_true, logits, k=3)

    return metrics
