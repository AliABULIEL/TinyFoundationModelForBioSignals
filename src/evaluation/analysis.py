"""Analysis and visualization utilities for HAR evaluation.

This module provides tools for analyzing model predictions, visualizing results,
and generating diagnostic plots.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.evaluation.metrics import compute_metrics, per_class_metrics

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot confusion matrix as heatmap.

    Args:
        confusion_matrix: Confusion matrix (K, K)
        class_names: Names for each class
        normalize: If True, show percentages instead of counts
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object

    Example:
        >>> cm = np.array([[50, 5], [10, 35]])
        >>> fig = plot_confusion_matrix(cm, class_names=['Class A', 'Class B'])
        >>> plt.show()
    """
    n_classes = confusion_matrix.shape[0]

    # Generate default class names if not provided
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    # Normalize if requested
    if normalize:
        # Normalize by row (true labels)
        cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1, keepdims=True)
        cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
        data = cm_normalized
        fmt = '.2%'
        cbar_label = "Proportion"
    else:
        data = confusion_matrix
        fmt = 'd'
        cbar_label = "Count"

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        data,
        annot=True,
        fmt=fmt if not normalize else '.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': cbar_label},
        ax=ax,
    )

    # If normalized, convert annotations to percentages
    if normalize:
        for text in ax.texts:
            value = float(text.get_text())
            text.set_text(f'{value:.1%}')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)

    plt.tight_layout()

    # Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")

    return fig


def plot_per_class_metrics(
    metrics: Dict[str, np.ndarray],
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot per-class precision, recall, and F1 scores.

    Args:
        metrics: Dictionary from per_class_metrics() function
        class_names: Names for each class (overrides metrics['names'])
        figsize: Figure size (width, height)
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object

    Example:
        >>> metrics = per_class_metrics(y_true, y_pred)
        >>> fig = plot_per_class_metrics(metrics, class_names=['A', 'B', 'C'])
        >>> plt.show()
    """
    # Extract data
    precision = metrics['precision']
    recall = metrics['recall']
    f1 = metrics['f1']
    support = metrics['support']

    # Get class names
    if class_names is None:
        class_names = metrics.get('names', [f"Class {i}" for i in range(len(precision))])

    n_classes = len(precision)
    x = np.arange(n_classes)
    width = 0.25

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Metrics comparison
    ax1.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax1.bar(x, recall, width, label='Recall', alpha=0.8)
    ax1.bar(x + width, f1, width, label='F1-Score', alpha=0.8)

    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Per-Class Metrics', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim([0, 1.05])
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Support (sample count)
    ax2.bar(x, support, alpha=0.8, color='coral')
    ax2.set_xlabel('Class', fontsize=12)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title('Class Support', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved per-class metrics to {save_path}")

    return fig


def plot_training_curves(
    train_losses: List[float],
    val_metrics: List[Dict[str, float]],
    metrics_to_plot: List[str] = None,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training and validation curves.

    Args:
        train_losses: List of training losses per epoch
        val_metrics: List of validation metric dicts per epoch
        metrics_to_plot: List of metric names to plot (default: accuracy, macro_f1)
        figsize: Figure size (width, height)
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object

    Example:
        >>> train_losses = [0.5, 0.3, 0.2, 0.15]
        >>> val_metrics = [
        ...     {'accuracy': 0.7, 'macro_f1': 0.65},
        ...     {'accuracy': 0.8, 'macro_f1': 0.75},
        ...     {'accuracy': 0.85, 'macro_f1': 0.82},
        ...     {'accuracy': 0.87, 'macro_f1': 0.85},
        ... ]
        >>> fig = plot_training_curves(train_losses, val_metrics)
        >>> plt.show()
    """
    if metrics_to_plot is None:
        metrics_to_plot = ['accuracy', 'macro_f1']

    epochs = range(1, len(train_losses) + 1)

    # Create figure with subplots
    n_plots = 1 + len(metrics_to_plot)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    if n_plots == 1:
        axes = [axes]

    # Plot 1: Training loss
    ax = axes[0]
    ax.plot(epochs, train_losses, 'o-', linewidth=2, markersize=6, label='Train Loss')

    # Also plot validation loss if available
    if val_metrics and 'loss' in val_metrics[0]:
        val_losses = [m['loss'] for m in val_metrics]
        ax.plot(epochs, val_losses, 's-', linewidth=2, markersize=6, label='Val Loss')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot validation metrics
    for i, metric_name in enumerate(metrics_to_plot, start=1):
        ax = axes[i]

        # Extract metric values
        metric_values = [m.get(metric_name, 0) for m in val_metrics]

        ax.plot(epochs, metric_values, 'o-', linewidth=2, markersize=6, color='green')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'Validation {metric_name.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.grid(alpha=0.3)

    plt.tight_layout()

    # Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training curves to {save_path}")

    return fig


def analyze_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    top_k_errors: int = 5,
) -> Dict[str, Any]:
    """
    Analyze model predictions to identify patterns and errors.

    Args:
        y_true: Ground truth labels (N,)
        y_pred: Predicted labels (N,)
        class_names: Names for each class
        top_k_errors: Number of top error pairs to report

    Returns:
        Dictionary containing:
        - overall_metrics: Overall performance metrics
        - per_class_metrics: Per-class performance
        - error_analysis: Common error patterns
        - class_confusion: Most confused class pairs

    Example:
        >>> analysis = analyze_predictions(y_true, y_pred, class_names=['A', 'B', 'C'])
        >>> print(f"Most confused: {analysis['class_confusion'][0]}")
    """
    # Compute overall metrics
    overall_metrics = compute_metrics(y_true, y_pred)

    # Compute per-class metrics
    per_class = per_class_metrics(
        y_true, y_pred, target_names=class_names
    )

    # Find misclassifications
    errors = y_true != y_pred
    n_errors = np.sum(errors)
    error_rate = n_errors / len(y_true)

    # Analyze error patterns
    error_pairs = []
    unique_labels = np.unique(y_true)

    for true_label in unique_labels:
        for pred_label in unique_labels:
            if true_label != pred_label:
                # Count this type of error
                mask = (y_true == true_label) & (y_pred == pred_label)
                count = np.sum(mask)

                if count > 0:
                    # Calculate proportion of true_label samples misclassified as pred_label
                    total_true = np.sum(y_true == true_label)
                    proportion = count / total_true if total_true > 0 else 0

                    error_pairs.append({
                        'true_class': int(true_label),
                        'predicted_class': int(pred_label),
                        'count': int(count),
                        'proportion': float(proportion),
                    })

    # Sort by count (descending)
    error_pairs.sort(key=lambda x: x['count'], reverse=True)

    # Format top errors with class names if available
    top_errors = error_pairs[:top_k_errors]
    if class_names is not None:
        for error in top_errors:
            error['true_class_name'] = class_names[error['true_class']]
            error['predicted_class_name'] = class_names[error['predicted_class']]

    # Find best and worst performing classes
    f1_scores = per_class['f1']
    best_class_idx = int(np.argmax(f1_scores))
    worst_class_idx = int(np.argmin(f1_scores))

    best_class = {
        'index': best_class_idx,
        'f1_score': float(f1_scores[best_class_idx]),
    }
    worst_class = {
        'index': worst_class_idx,
        'f1_score': float(f1_scores[worst_class_idx]),
    }

    if class_names is not None:
        best_class['name'] = class_names[best_class_idx]
        worst_class['name'] = class_names[worst_class_idx]

    analysis = {
        'overall_metrics': overall_metrics,
        'per_class_metrics': {
            'precision': per_class['precision'].tolist(),
            'recall': per_class['recall'].tolist(),
            'f1': per_class['f1'].tolist(),
            'support': per_class['support'].tolist(),
        },
        'error_analysis': {
            'total_errors': int(n_errors),
            'error_rate': float(error_rate),
            'top_confusion_pairs': top_errors,
        },
        'best_class': best_class,
        'worst_class': worst_class,
    }

    logger.info(
        f"Analysis complete: {n_errors}/{len(y_true)} errors ({error_rate:.2%})"
    )

    return analysis


def compute_subject_level_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subject_ids: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute metrics aggregated at subject level.

    For each subject, compute accuracy and F1, then aggregate across subjects.

    Args:
        y_true: Ground truth labels (N,)
        y_pred: Predicted labels (N,)
        subject_ids: Subject ID for each sample (N,)

    Returns:
        Dictionary containing:
        - subject_accuracies: List of per-subject accuracies
        - subject_f1_scores: List of per-subject F1 scores
        - mean_accuracy: Mean accuracy across subjects
        - std_accuracy: Std of accuracy across subjects
        - mean_f1: Mean F1 across subjects
        - std_f1: Std of F1 across subjects

    Example:
        >>> metrics = compute_subject_level_metrics(y_true, y_pred, subject_ids)
        >>> print(f"Subject-level accuracy: {metrics['mean_accuracy']:.4f} ± {metrics['std_accuracy']:.4f}")
    """
    unique_subjects = np.unique(subject_ids)

    subject_accuracies = []
    subject_f1_scores = []
    subject_results = []

    for subject in unique_subjects:
        # Get subject's samples
        mask = subject_ids == subject
        subj_true = y_true[mask]
        subj_pred = y_pred[mask]

        # Compute metrics
        metrics = compute_metrics(subj_true, subj_pred)

        subject_accuracies.append(metrics['accuracy'])
        subject_f1_scores.append(metrics['macro_f1'])

        subject_results.append({
            'subject_id': str(subject),
            'n_samples': int(np.sum(mask)),
            'accuracy': float(metrics['accuracy']),
            'macro_f1': float(metrics['macro_f1']),
        })

    # Aggregate statistics
    result = {
        'subject_results': subject_results,
        'mean_accuracy': float(np.mean(subject_accuracies)),
        'std_accuracy': float(np.std(subject_accuracies)),
        'mean_f1': float(np.mean(subject_f1_scores)),
        'std_f1': float(np.std(subject_f1_scores)),
        'n_subjects': len(unique_subjects),
    }

    logger.info(
        f"Subject-level metrics: "
        f"Accuracy={result['mean_accuracy']:.4f}±{result['std_accuracy']:.4f}, "
        f"F1={result['mean_f1']:.4f}±{result['std_f1']:.4f} "
        f"({result['n_subjects']} subjects)"
    )

    return result


def plot_subject_variability(
    subject_metrics: Dict[str, Any],
    metric: str = 'accuracy',
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot variability of metrics across subjects.

    Args:
        subject_metrics: Dictionary from compute_subject_level_metrics()
        metric: Metric to plot ('accuracy' or 'macro_f1')
        figsize: Figure size (width, height)
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object

    Example:
        >>> subject_metrics = compute_subject_level_metrics(y_true, y_pred, subject_ids)
        >>> fig = plot_subject_variability(subject_metrics, metric='accuracy')
        >>> plt.show()
    """
    subject_results = subject_metrics['subject_results']

    # Extract data
    subject_ids = [r['subject_id'] for r in subject_results]
    values = [r[metric] for r in subject_results]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot bars
    x = np.arange(len(subject_ids))
    ax.bar(x, values, alpha=0.7)

    # Add mean line
    mean_value = subject_metrics[f'mean_{metric}']
    ax.axhline(mean_value, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_value:.3f}')

    # Add std shading
    std_value = subject_metrics[f'std_{metric}']
    ax.axhspan(mean_value - std_value, mean_value + std_value, alpha=0.2, color='red', label=f'±1 Std: {std_value:.3f}')

    ax.set_xlabel('Subject ID', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'Subject-Level {metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(subject_ids, rotation=45, ha='right')
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved subject variability plot to {save_path}")

    return fig
