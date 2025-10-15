"""Visualization utilities for evaluation results.

Provides plotting functions for ROC curves, PR curves, confusion matrices,
reliability diagrams, and benchmark comparisons.
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    confusion_matrix as sklearn_confusion_matrix,
    auc
)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "ROC Curve",
    show_plot: bool = True
) -> Tuple[plt.Figure, float]:
    """Plot ROC curve.
    
    Args:
        y_true: True labels
        y_score: Predicted scores
        save_path: Path to save figure
        title: Plot title
        show_plot: Whether to display plot
        
    Returns:
        Figure and AUROC value
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot ROC curve
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    
    # Plot diagonal
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random Classifier')
    
    # Formatting
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ ROC curve saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig, roc_auc


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Precision-Recall Curve",
    show_plot: bool = True
) -> Tuple[plt.Figure, float]:
    """Plot Precision-Recall curve.
    
    Args:
        y_true: True labels
        y_score: Predicted scores
        save_path: Path to save figure
        title: Plot title
        show_plot: Whether to display plot
        
    Returns:
        Figure and AUPRC value
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot PR curve
    ax.plot(recall, precision, color='darkorange', lw=2,
            label=f'PR curve (AUC = {pr_auc:.3f})')
    
    # Plot baseline (proportion of positives)
    baseline = y_true.sum() / len(y_true)
    ax.plot([0, 1], [baseline, baseline], color='navy', lw=2, 
            linestyle='--', label=f'Baseline ({baseline:.3f})')
    
    # Formatting
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ PR curve saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig, pr_auc


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    title: str = "Confusion Matrix",
    show_plot: bool = True,
    normalize: bool = False
) -> plt.Figure:
    """Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        save_path: Path to save figure
        title: Plot title
        show_plot: Whether to display plot
        normalize: Whether to normalize
        
    Returns:
        Figure
    """
    cm = sklearn_confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                square=True, cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
                ax=ax)
    
    # Formatting
    if class_names:
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
    
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_regression_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Regression: True vs Predicted",
    show_plot: bool = True,
    xlabel: str = "True Values",
    ylabel: str = "Predicted Values"
) -> plt.Figure:
    """Plot regression scatter plot with identity line.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        save_path: Path to save figure
        title: Plot title
        show_plot: Whether to display plot
        xlabel: X-axis label
        ylabel: Y-axis label
        
    Returns:
        Figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, color='steelblue')
    
    # Identity line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', lw=2, label='Perfect Prediction')
    
    # Calculate metrics
    from scipy.stats import pearsonr
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r, _ = pearsonr(y_true, y_pred)
    
    # Add text box with metrics
    textstr = f'MAE = {mae:.3f}\nRMSE = {rmse:.3f}\nPearson r = {r:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Formatting
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Regression scatter saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_bland_altman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Bland-Altman Plot",
    show_plot: bool = True,
    xlabel: str = "Mean of True and Predicted",
    ylabel: str = "Difference (True - Predicted)"
) -> plt.Figure:
    """Plot Bland-Altman plot for agreement analysis.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        save_path: Path to save figure
        title: Plot title
        show_plot: Whether to display plot
        xlabel: X-axis label
        ylabel: Y-axis label
        
    Returns:
        Figure
    """
    mean = (y_true + y_pred) / 2
    diff = y_true - y_pred
    
    md = np.mean(diff)
    sd = np.std(diff, ddof=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot
    ax.scatter(mean, diff, alpha=0.5, s=20, color='steelblue')
    
    # Mean difference line
    ax.axhline(md, color='red', linestyle='-', lw=2, label=f'Mean Diff = {md:.3f}')
    
    # Limits of agreement
    ax.axhline(md + 1.96*sd, color='red', linestyle='--', lw=2, 
               label=f'+1.96 SD = {md + 1.96*sd:.3f}')
    ax.axhline(md - 1.96*sd, color='red', linestyle='--', lw=2,
               label=f'-1.96 SD = {md - 1.96*sd:.3f}')
    
    # Zero line
    ax.axhline(0, color='black', linestyle='-', lw=1, alpha=0.3)
    
    # Formatting
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Bland-Altman plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_benchmark_comparison(
    results: Dict,
    save_path: Optional[Path] = None,
    show_plot: bool = True
) -> plt.Figure:
    """Plot benchmark comparison bar chart.
    
    Args:
        results: Results dictionary from DownstreamEvaluator
        save_path: Path to save figure
        show_plot: Whether to display plot
        
    Returns:
        Figure
    """
    benchmark_data = results.get('benchmarks', {})
    
    if not benchmark_data or 'target' not in benchmark_data:
        print("No benchmark data available for comparison")
        return None
    
    # Extract data
    categories = []
    values = []
    colors = []
    
    if 'baseline' in benchmark_data and benchmark_data['baseline']:
        categories.append('Baseline\n' + benchmark_data['baseline']['model'][:20])
        values.append(benchmark_data['baseline']['value'])
        colors.append('lightcoral')
    
    if 'target' in benchmark_data and benchmark_data['target']:
        categories.append('Target\n' + benchmark_data['target']['model'][:20])
        values.append(benchmark_data['target']['value'])
        colors.append('gold')
    
    if 'sota' in benchmark_data and benchmark_data['sota']:
        categories.append('SOTA\n' + benchmark_data['sota']['model'][:20])
        values.append(benchmark_data['sota']['value'])
        colors.append('lightgreen')
    
    # Add achieved value
    if 'target' in benchmark_data:
        categories.append('Achieved\n(This Model)')
        values.append(benchmark_data['target']['achieved'])
        colors.append('steelblue')
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Formatting
    metric_name = benchmark_data['target']['metric'] if 'target' in benchmark_data else 'Metric'
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f'Benchmark Comparison - {results["dataset"].upper()} - {results["task"].upper()}',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add performance category
    if 'performance_category' in benchmark_data:
        cat = benchmark_data['performance_category']
        ax.text(0.5, 0.98, f'Performance: {cat}', 
                transform=ax.transAxes, ha='center', va='top',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Benchmark comparison saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_per_subject_distribution(
    per_subject_metrics: Dict,
    metric_name: str = 'auroc',
    save_path: Optional[Path] = None,
    title: str = "Per-Subject Performance Distribution",
    show_plot: bool = True
) -> plt.Figure:
    """Plot distribution of per-subject metrics.
    
    Args:
        per_subject_metrics: Per-subject metrics dictionary
        metric_name: Metric to plot
        save_path: Path to save figure
        title: Plot title
        show_plot: Whether to display plot
        
    Returns:
        Figure
    """
    # Extract values
    values = []
    subjects = []
    
    for subject, metrics in per_subject_metrics.items():
        if subject == '_summary':
            continue
        if metric_name in metrics:
            values.append(metrics[metric_name])
            subjects.append(subject)
    
    if len(values) == 0:
        print(f"No per-subject data available for metric: {metric_name}")
        return None
    
    values = np.array(values)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(values, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(values.mean(), color='red', linestyle='--', lw=2, 
                label=f'Mean = {values.mean():.3f}')
    ax1.axvline(np.median(values), color='green', linestyle='--', lw=2,
                label=f'Median = {np.median(values):.3f}')
    ax1.set_xlabel(metric_name.upper(), fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # Box plot
    ax2.boxplot(values, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='black'),
                medianprops=dict(color='red', linewidth=2),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'))
    ax2.set_ylabel(metric_name.upper(), fontsize=12)
    ax2.set_title('Box Plot', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add statistics
    textstr = f'Mean: {values.mean():.3f}\nStd: {values.std():.3f}\nMin: {values.min():.3f}\nMax: {values.max():.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax2.text(0.5, 0.02, textstr, transform=ax2.transAxes, fontsize=9,
             verticalalignment='bottom', horizontalalignment='center', bbox=props)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Per-subject distribution saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def create_evaluation_report(
    results: Dict,
    save_dir: Path,
    show_plots: bool = False
):
    """Create comprehensive evaluation report with all plots.
    
    Args:
        results: Results dictionary from DownstreamEvaluator
        save_dir: Directory to save plots
        show_plots: Whether to display plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    task_type = results['task_type']
    dataset = results['dataset']
    task = results['task']
    
    print(f"\n{'='*80}")
    print(f"CREATING EVALUATION REPORT - {dataset.upper()} - {task.upper()}")
    print(f"{'='*80}\n")
    
    # Classification plots
    if task_type == "classification" and 'probabilities' in results and len(results['probabilities']) > 0:
        y_true = np.array(results['labels'])
        y_pred = np.array(results['predictions'])
        y_prob = np.array(results['probabilities'])
        
        # ROC curve
        plot_roc_curve(
            y_true, y_prob,
            save_path=save_dir / f"{dataset}_{task}_roc_curve.png",
            title=f"ROC Curve - {dataset.upper()} - {task.upper()}",
            show_plot=show_plots
        )
        
        # PR curve
        plot_precision_recall_curve(
            y_true, y_prob,
            save_path=save_dir / f"{dataset}_{task}_pr_curve.png",
            title=f"Precision-Recall Curve - {dataset.upper()} - {task.upper()}",
            show_plot=show_plots
        )
        
        # Confusion matrix
        plot_confusion_matrix(
            y_true, y_pred,
            save_path=save_dir / f"{dataset}_{task}_confusion_matrix.png",
            title=f"Confusion Matrix - {dataset.upper()} - {task.upper()}",
            show_plot=show_plots
        )
    
    # Regression plots
    elif task_type == "regression" and len(results['predictions']) > 0:
        y_true = np.array(results['labels'])
        y_pred = np.array(results['predictions'])
        
        # Scatter plot
        plot_regression_scatter(
            y_true, y_pred,
            save_path=save_dir / f"{dataset}_{task}_scatter.png",
            title=f"Regression - {dataset.upper()} - {task.upper()}",
            show_plot=show_plots
        )
        
        # Bland-Altman plot
        plot_bland_altman(
            y_true, y_pred,
            save_path=save_dir / f"{dataset}_{task}_bland_altman.png",
            title=f"Bland-Altman - {dataset.upper()} - {task.upper()}",
            show_plot=show_plots
        )
    
    # Benchmark comparison
    if 'benchmarks' in results and results['benchmarks']:
        plot_benchmark_comparison(
            results,
            save_path=save_dir / f"{dataset}_{task}_benchmark_comparison.png",
            show_plot=show_plots
        )
    
    # Per-subject distribution
    if 'per_subject_metrics' in results and len(results['per_subject_metrics']) > 1:
        primary_metric = 'auroc' if task_type == 'classification' else 'mae'
        plot_per_subject_distribution(
            results['per_subject_metrics'],
            metric_name=primary_metric,
            save_path=save_dir / f"{dataset}_{task}_per_subject.png",
            title=f"Per-Subject {primary_metric.upper()} - {dataset.upper()} - {task.upper()}",
            show_plot=show_plots
        )
    
    print(f"\n✅ All plots saved to: {save_dir}\n")
