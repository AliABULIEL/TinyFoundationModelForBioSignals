"""Base classes for VitalDB downstream tasks.

Provides abstract interfaces for task definition, label generation,
evaluation metrics, and benchmark comparison.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    mean_absolute_error, mean_squared_error,
    r2_score, accuracy_score, f1_score
)


@dataclass
class TaskConfig:
    """Configuration for a downstream task."""
    name: str
    task_type: str  # 'classification', 'regression', 'time_series'
    num_classes: Optional[int] = None
    target_dim: Optional[int] = None
    clinical_threshold: Optional[float] = None
    prediction_window_s: Optional[float] = None
    window_size_s: float = 10.0
    sampling_rate: float = 125.0
    
    # Channels required for this task
    required_channels: List[str] = None
    
    # Quality thresholds
    min_sqi: float = 0.8
    min_cycles: int = 3
    
    def __post_init__(self):
        if self.required_channels is None:
            self.required_channels = ['PLETH']


@dataclass
class Benchmark:
    """Published benchmark for comparison."""
    paper: str
    year: int
    dataset: str
    n_patients: int
    metrics: Dict[str, float]
    method: str
    notes: str = ""
    
    def __repr__(self) -> str:
        metric_str = ", ".join([f"{k}={v:.3f}" for k, v in self.metrics.items()])
        return f"{self.paper} ({self.year}): {metric_str} on {self.n_patients} patients"


class BaseTask(ABC):
    """Abstract base class for VitalDB downstream tasks."""
    
    def __init__(self, config: TaskConfig):
        self.config = config
        self.benchmarks: List[Benchmark] = []
        self._load_benchmarks()
    
    @abstractmethod
    def generate_labels(
        self,
        case_id: str,
        signals: Dict[str, np.ndarray],
        clinical_data: pd.Series,
        fs: float = 125.0
    ) -> Union[np.ndarray, float, int]:
        """Generate labels for this task.
        
        Args:
            case_id: VitalDB case identifier
            signals: Dictionary of signal arrays (e.g., {'ECG': array, 'PPG': array})
            clinical_data: Clinical parameters for this case
            fs: Sampling frequency
            
        Returns:
            Labels appropriate for the task (array for time-series, scalar for outcomes)
        """
        pass
    
    @abstractmethod
    def evaluate(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        return_detailed: bool = False
    ) -> Dict[str, float]:
        """Evaluate predictions against targets.
        
        Args:
            predictions: Model predictions
            targets: Ground truth labels
            return_detailed: Whether to return detailed metrics
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass
    
    @abstractmethod
    def _load_benchmarks(self):
        """Load published benchmarks for this task."""
        pass
    
    def compare_to_benchmarks(
        self,
        results: Dict[str, float]
    ) -> pd.DataFrame:
        """Compare results to published benchmarks.
        
        Args:
            results: Dictionary of evaluation metrics
            
        Returns:
            DataFrame comparing this work to benchmarks
        """
        comparison_data = []
        
        # Add our results
        comparison_data.append({
            'Paper': 'This Work',
            'Year': 2025,
            'Dataset': 'VitalDB',
            'N_Patients': results.get('n_patients', 'N/A'),
            **results
        })
        
        # Add benchmarks
        for benchmark in self.benchmarks:
            row = {
                'Paper': benchmark.paper,
                'Year': benchmark.year,
                'Dataset': benchmark.dataset,
                'N_Patients': benchmark.n_patients,
                **benchmark.metrics
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        return df
    
    def validate_signals(
        self,
        signals: Dict[str, np.ndarray]
    ) -> Tuple[bool, List[str]]:
        """Validate that required signals are present and of sufficient quality.
        
        Args:
            signals: Dictionary of signal arrays
            
        Returns:
            Tuple of (valid, reasons)
        """
        reasons = []
        
        # Check required channels
        for channel in self.config.required_channels:
            if channel not in signals:
                reasons.append(f"Missing required channel: {channel}")
        
        if reasons:
            return False, reasons
        
        # Check signal length
        expected_samples = int(self.config.window_size_s * self.config.sampling_rate)
        for channel, signal in signals.items():
            if len(signal) < expected_samples:
                reasons.append(
                    f"Signal {channel} too short: {len(signal)} < {expected_samples}"
                )
        
        # Check for NaN/inf
        for channel, signal in signals.items():
            if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
                reasons.append(f"Signal {channel} contains NaN or inf values")
        
        valid = len(reasons) == 0
        return valid, reasons
    
    def get_clinical_param(
        self,
        clinical_data: pd.Series,
        param: str,
        default: Any = None
    ) -> Any:
        """Safely get clinical parameter with default fallback.
        
        Args:
            clinical_data: Series of clinical parameters
            param: Parameter name
            default: Default value if missing
            
        Returns:
            Parameter value or default
        """
        if param in clinical_data.index:
            value = clinical_data[param]
            if pd.notna(value):
                return value
        return default


class ClassificationTask(BaseTask):
    """Base class for classification tasks."""
    
    def evaluate(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        return_detailed: bool = False
    ) -> Dict[str, float]:
        """Evaluate classification task.
        
        Args:
            predictions: Model predictions (probabilities or logits)
            targets: Ground truth labels
            return_detailed: Whether to return detailed metrics
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Handle multi-class vs binary
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            # Multi-class: use softmax probabilities
            probs = torch.softmax(torch.tensor(predictions), dim=1).numpy()
            pred_classes = np.argmax(probs, axis=1)
        else:
            # Binary: use sigmoid
            if predictions.ndim > 1:
                predictions = predictions[:, 0]
            probs = 1 / (1 + np.exp(-predictions))  # sigmoid
            pred_classes = (probs > 0.5).astype(int)
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(targets, pred_classes)
        
        # AUROC and AUPRC
        try:
            if len(np.unique(targets)) > 1:
                if predictions.ndim > 1 and predictions.shape[1] > 2:
                    # Multi-class
                    metrics['auroc'] = roc_auc_score(
                        targets, probs, multi_class='ovr', average='macro'
                    )
                else:
                    # Binary
                    metrics['auroc'] = roc_auc_score(targets, probs)
                    metrics['auprc'] = average_precision_score(targets, probs)
        except ValueError as e:
            print(f"Warning: Could not compute AUROC/AUPRC: {e}")
        
        # F1 score
        metrics['f1'] = f1_score(
            targets, pred_classes, average='binary' if len(np.unique(targets)) == 2 else 'macro'
        )
        
        if return_detailed:
            # Per-class metrics, confusion matrix, etc.
            from sklearn.metrics import classification_report, confusion_matrix
            metrics['classification_report'] = classification_report(
                targets, pred_classes, output_dict=True
            )
            metrics['confusion_matrix'] = confusion_matrix(targets, pred_classes)
        
        return metrics


class RegressionTask(BaseTask):
    """Base class for regression tasks."""
    
    def evaluate(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        return_detailed: bool = False
    ) -> Dict[str, float]:
        """Evaluate regression task.
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            return_detailed: Whether to return detailed metrics
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Flatten if needed
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        # Basic metrics
        metrics['mae'] = mean_absolute_error(targets, predictions)
        metrics['mse'] = mean_squared_error(targets, predictions)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(targets, predictions)
        
        # Correlation
        metrics['pearson_r'] = np.corrcoef(targets, predictions)[0, 1]
        
        # Mean error (bias)
        metrics['mean_error'] = np.mean(predictions - targets)
        metrics['std_error'] = np.std(predictions - targets)
        
        # Percentage error (for clinical CO, etc.)
        mean_target = np.mean(np.abs(targets))
        if mean_target > 0:
            metrics['percentage_error'] = 100 * np.mean(np.abs(predictions - targets)) / mean_target
        
        if return_detailed:
            # Per-sample errors
            errors = predictions - targets
            metrics['errors'] = errors
            metrics['abs_errors'] = np.abs(errors)
            
            # Bland-Altman statistics
            mean_diff = np.mean(errors)
            std_diff = np.std(errors)
            metrics['bland_altman_bias'] = mean_diff
            metrics['bland_altman_loa'] = (mean_diff - 1.96*std_diff, mean_diff + 1.96*std_diff)
        
        return metrics


class TimeSeriesTask(BaseTask):
    """Base class for time-series prediction tasks."""
    
    def evaluate(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        return_detailed: bool = False
    ) -> Dict[str, float]:
        """Evaluate time-series task.
        
        Args:
            predictions: Model predictions [samples, time, features]
            targets: Ground truth values [samples, time, features]
            return_detailed: Whether to return detailed metrics
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Flatten time dimension for overall metrics
        pred_flat = predictions.reshape(-1, predictions.shape[-1])
        target_flat = targets.reshape(-1, targets.shape[-1])
        
        # Overall metrics
        metrics['mae'] = mean_absolute_error(target_flat, pred_flat)
        metrics['mse'] = mean_squared_error(target_flat, pred_flat)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        if return_detailed:
            # Per-timestep metrics
            metrics['mae_per_timestep'] = np.mean(
                np.abs(predictions - targets), axis=(0, 2)
            )
            
            # Per-sample metrics
            metrics['mae_per_sample'] = np.mean(
                np.abs(predictions - targets), axis=(1, 2)
            )
        
        return metrics
