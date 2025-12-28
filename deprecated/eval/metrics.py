"""Evaluation metrics for classification and regression tasks.

Provides standard metrics for biosignal analysis tasks.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    r2_score
)
from scipy.stats import pearsonr


def _to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert tensor to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def _check_shape(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Check and align shapes of predictions."""
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    
    # Flatten if needed
    if y_true.ndim > 1:
        y_true = y_true.squeeze()
    if y_pred.ndim > 1:
        y_pred = y_pred.squeeze()
    
    # Check same length
    if len(y_true) != len(y_pred):
        raise ValueError(f"Shape mismatch: {len(y_true)} vs {len(y_pred)}")
    
    return y_true, y_pred


# Classification Metrics

def accuracy(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5
) -> float:
    """Compute accuracy score.
    
    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted probabilities or labels
        threshold: Threshold for binary classification
        
    Returns:
        Accuracy score
    """
    y_true, y_pred = _check_shape(y_true, y_pred)
    
    # Convert probabilities to labels if needed
    if y_pred.dtype == np.float32 or y_pred.dtype == np.float64:
        if y_pred.min() >= 0 and y_pred.max() <= 1:
            # Probabilities - threshold
            y_pred = (y_pred >= threshold).astype(int)
    
    return accuracy_score(y_true, y_pred)


def auroc(
    y_true: Union[torch.Tensor, np.ndarray],
    y_score: Union[torch.Tensor, np.ndarray],
    multi_class: str = 'ovr'
) -> float:
    """Compute Area Under ROC Curve.
    
    Args:
        y_true: True labels
        y_score: Predicted scores/probabilities
        multi_class: How to handle multi-class ('ovr', 'ovo')
        
    Returns:
        AUROC score
    """
    y_true, y_score = _check_shape(y_true, y_score)
    
    # Check if binary or multi-class
    n_classes = len(np.unique(y_true))
    
    if n_classes == 2:
        return roc_auc_score(y_true, y_score)
    else:
        return roc_auc_score(y_true, y_score, multi_class=multi_class)


def auprc(
    y_true: Union[torch.Tensor, np.ndarray],
    y_score: Union[torch.Tensor, np.ndarray]
) -> float:
    """Compute Area Under Precision-Recall Curve.
    
    Args:
        y_true: True labels
        y_score: Predicted scores/probabilities
        
    Returns:
        AUPRC score
    """
    y_true, y_score = _check_shape(y_true, y_score)
    return average_precision_score(y_true, y_score)


def f1(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    average: str = 'binary'
) -> float:
    """Compute F1 score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels or probabilities
        threshold: Threshold for binary classification
        average: Averaging method for multi-class
        
    Returns:
        F1 score
    """
    y_true, y_pred = _check_shape(y_true, y_pred)
    
    # Convert probabilities to labels if needed
    if y_pred.dtype == np.float32 or y_pred.dtype == np.float64:
        if y_pred.min() >= 0 and y_pred.max() <= 1:
            y_pred = (y_pred >= threshold).astype(int)
    
    return f1_score(y_true, y_pred, average=average)


def precision(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    average: str = 'binary'
) -> float:
    """Compute precision score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels or probabilities
        threshold: Threshold for binary classification
        average: Averaging method for multi-class
        
    Returns:
        Precision score
    """
    y_true, y_pred = _check_shape(y_true, y_pred)
    
    # Convert probabilities to labels if needed
    if y_pred.dtype == np.float32 or y_pred.dtype == np.float64:
        if y_pred.min() >= 0 and y_pred.max() <= 1:
            y_pred = (y_pred >= threshold).astype(int)
    
    return precision_score(y_true, y_pred, average=average, zero_division=0)


def recall(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    average: str = 'binary'
) -> float:
    """Compute recall score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels or probabilities
        threshold: Threshold for binary classification
        average: Averaging method for multi-class
        
    Returns:
        Recall score
    """
    y_true, y_pred = _check_shape(y_true, y_pred)
    
    # Convert probabilities to labels if needed
    if y_pred.dtype == np.float32 or y_pred.dtype == np.float64:
        if y_pred.min() >= 0 and y_pred.max() <= 1:
            y_pred = (y_pred >= threshold).astype(int)
    
    return recall_score(y_true, y_pred, average=average, zero_division=0)


# Regression Metrics

def mae(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray]
) -> float:
    """Compute Mean Absolute Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAE score
        
    Raises:
        ValueError: If arrays are empty or mismatched.
    """
    y_true, y_pred = _check_shape(y_true, y_pred)
    
    if len(y_true) == 0:
        raise ValueError("Cannot compute MAE for empty arrays")
    
    return mean_absolute_error(y_true, y_pred)


def rmse(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray]
) -> float:
    """Compute Root Mean Squared Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        RMSE score
    """
    y_true, y_pred = _check_shape(y_true, y_pred)
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mse(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray]
) -> float:
    """Compute Mean Squared Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MSE score
    """
    y_true, y_pred = _check_shape(y_true, y_pred)
    return mean_squared_error(y_true, y_pred)


def ccc(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray]
) -> float:
    """Compute Concordance Correlation Coefficient.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        CCC score
    """
    y_true, y_pred = _check_shape(y_true, y_pred)
    
    # Calculate means
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    
    # Calculate variances
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    
    # Calculate covariance
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    
    # Calculate CCC
    numerator = 2 * cov
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    
    if denominator < 1e-10:
        return 0.0
    
    return numerator / denominator


def pearson_r(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray]
) -> Tuple[float, float]:
    """Compute Pearson correlation coefficient.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Tuple of (correlation, p-value)
    """
    y_true, y_pred = _check_shape(y_true, y_pred)
    return pearsonr(y_true, y_pred)


def r2(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray]
) -> float:
    """Compute R-squared (coefficient of determination).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        R2 score
    """
    y_true, y_pred = _check_shape(y_true, y_pred)
    return r2_score(y_true, y_pred)


# Composite Metrics

def classification_metrics(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    y_score: Optional[Union[torch.Tensor, np.ndarray]] = None,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Compute all classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels or probabilities
        y_score: Predicted scores (if different from y_pred)
        threshold: Threshold for binary classification
        
    Returns:
        Dictionary of metrics
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    
    if y_score is None:
        y_score = y_pred
    else:
        y_score = _to_numpy(y_score)
    
    metrics = {
        'accuracy': accuracy(y_true, y_pred, threshold),
        'precision': precision(y_true, y_pred, threshold),
        'recall': recall(y_true, y_pred, threshold),
        'f1': f1(y_true, y_pred, threshold)
    }
    
    # Add ROC/PRC metrics if scores available
    try:
        if y_score is not None:
            metrics['auroc'] = auroc(y_true, y_score)
            metrics['auprc'] = auprc(y_true, y_score)
    except Exception:
        # May fail for multi-class without proper scores
        pass
    
    return metrics


def compute_classification_metrics(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    y_prob: Optional[Union[torch.Tensor, np.ndarray]] = None,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Compute all classification metrics (alias for classification_metrics).
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        threshold: Threshold for binary classification
        
    Returns:
        Dictionary of metrics
    """
    return classification_metrics(y_true, y_pred, y_prob, threshold)


def regression_metrics(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray]
) -> Dict[str, float]:
    """Compute all regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    
    corr, p_val = pearson_r(y_true, y_pred)
    
    metrics = {
        'mae': mae(y_true, y_pred),
        'rmse': rmse(y_true, y_pred),
        'mse': mse(y_true, y_pred),
        'ccc': ccc(y_true, y_pred),
        'r2': r2(y_true, y_pred),
        'pearson_r': corr,
        'pearson_p': p_val
    }
    
    return metrics


def compute_regression_metrics(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray]
) -> Dict[str, float]:
    """Compute all regression metrics (alias for regression_metrics).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    return regression_metrics(y_true, y_pred)


class MetricTracker:
    """Track and aggregate metrics over batches/epochs."""
    
    def __init__(self, metrics: List[str]):
        """Initialize tracker.
        
        Args:
            metrics: List of metric names to track
        """
        self.metrics = metrics
        self.reset()
    
    def reset(self):
        """Reset all tracked values."""
        self.values = {m: [] for m in self.metrics}
        self.counts = {m: [] for m in self.metrics}
    
    def update(self, metric_dict: Dict[str, float], count: int = 1):
        """Update tracked metrics.
        
        Args:
            metric_dict: Dictionary of metric values
            count: Batch size for weighted averaging
        """
        for name, value in metric_dict.items():
            if name in self.values:
                self.values[name].append(value)
                self.counts[name].append(count)
    
    def average(self) -> Dict[str, float]:
        """Get weighted average of all metrics."""
        averages = {}
        
        for name in self.metrics:
            if len(self.values[name]) > 0:
                # Weighted average
                total = sum(v * c for v, c in zip(self.values[name], self.counts[name]))
                count = sum(self.counts[name])
                averages[name] = total / count if count > 0 else 0.0
            else:
                averages[name] = 0.0
        
        return averages
    
    def last(self) -> Dict[str, float]:
        """Get last values of all metrics."""
        last_values = {}
        
        for name in self.metrics:
            if len(self.values[name]) > 0:
                last_values[name] = self.values[name][-1]
            else:
                last_values[name] = 0.0
        
        return last_values
