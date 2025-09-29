"""Model calibration utilities for improving prediction confidence."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class TemperatureScaling(nn.Module):
    """Temperature scaling for calibration.
    
    A simple method to calibrate neural network predictions by learning
    a single scalar temperature parameter.
    """
    
    def __init__(self, temperature: float = 1.0):
        """Initialize temperature scaling.
        
        Args:
            temperature: Initial temperature value
        """
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling.
        
        Args:
            logits: Model logits [B, C]
            
        Returns:
            Scaled logits
        """
        return logits / self.temperature
    
    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
        verbose: bool = False
    ) -> float:
        """Fit temperature using validation data.
        
        Args:
            logits: Validation logits [N, C]
            labels: Validation labels [N]
            lr: Learning rate
            max_iter: Maximum iterations
            verbose: Whether to print progress
            
        Returns:
            Optimal temperature value
        """
        device = logits.device if isinstance(logits, torch.Tensor) else torch.device("cpu")
        
        # Ensure tensors
        if not isinstance(logits, torch.Tensor):
            logits = torch.tensor(logits, dtype=torch.float32, device=device)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long, device=device)
        
        # Setup optimizer
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()
        
        def closure():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = criterion(scaled_logits, labels)
            loss.backward()
            return loss
        
        # Optimize
        self.train()
        optimizer.step(closure)
        self.eval()
        
        optimal_temp = self.temperature.item()
        
        if verbose:
            print(f"Optimal temperature: {optimal_temp:.3f}")
        
        return optimal_temp


class PlattScaling(nn.Module):
    """Platt scaling for binary calibration.
    
    Learns a sigmoid transformation of the logits to calibrate
    binary classification probabilities.
    """
    
    def __init__(self):
        """Initialize Platt scaling."""
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply Platt scaling.
        
        Args:
            logits: Model logits
            
        Returns:
            Calibrated probabilities
        """
        return torch.sigmoid(self.weight * logits + self.bias)
    
    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
        verbose: bool = False
    ) -> Tuple[float, float]:
        """Fit Platt scaling parameters.
        
        Args:
            logits: Validation logits
            labels: Validation labels
            lr: Learning rate
            max_iter: Maximum iterations
            verbose: Whether to print progress
            
        Returns:
            Tuple of (weight, bias)
        """
        device = logits.device if isinstance(logits, torch.Tensor) else torch.device("cpu")
        
        # Ensure tensors
        if not isinstance(logits, torch.Tensor):
            logits = torch.tensor(logits, dtype=torch.float32, device=device)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.float32, device=device)
        
        # Reshape if needed
        if logits.dim() == 1:
            logits = logits.unsqueeze(-1)
        
        # Setup optimizer
        optimizer = optim.LBFGS([self.weight, self.bias], lr=lr, max_iter=max_iter)
        criterion = nn.BCELoss()
        
        def closure():
            optimizer.zero_grad()
            probs = self.forward(logits)
            loss = criterion(probs.squeeze(), labels.float())
            loss.backward()
            return loss
        
        # Optimize
        self.train()
        optimizer.step(closure)
        self.eval()
        
        weight = self.weight.item()
        bias = self.bias.item()
        
        if verbose:
            print(f"Platt scaling: weight={weight:.3f}, bias={bias:.3f}")
        
        return weight, bias


class IsotonicCalibration:
    """Isotonic regression calibration.
    
    Non-parametric calibration method that learns a monotonic
    mapping from predicted probabilities to calibrated probabilities.
    """
    
    def __init__(self):
        """Initialize isotonic calibration."""
        self.calibrator = None
    
    def fit(
        self,
        probs: Union[np.ndarray, torch.Tensor],
        labels: Union[np.ndarray, torch.Tensor],
    ) -> None:
        """Fit isotonic regression.
        
        Args:
            probs: Validation probabilities
            labels: Validation labels
        """
        # Convert to numpy
        if isinstance(probs, torch.Tensor):
            probs = probs.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        # Flatten if needed
        if probs.ndim > 1:
            probs = probs.ravel()
        if labels.ndim > 1:
            labels = labels.ravel()
        
        # Fit isotonic regression
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(probs, labels)
    
    def transform(
        self,
        probs: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """Apply isotonic calibration.
        
        Args:
            probs: Input probabilities
            
        Returns:
            Calibrated probabilities
        """
        if self.calibrator is None:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        
        is_tensor = isinstance(probs, torch.Tensor)
        device = probs.device if is_tensor else None
        
        # Convert to numpy
        if is_tensor:
            probs_np = probs.cpu().numpy()
        else:
            probs_np = probs
        
        # Store original shape
        original_shape = probs_np.shape
        
        # Flatten for calibration
        if probs_np.ndim > 1:
            probs_np = probs_np.ravel()
        
        # Apply calibration
        calibrated = self.calibrator.predict(probs_np)
        
        # Reshape to original
        calibrated = calibrated.reshape(original_shape)
        
        # Convert back to tensor if needed
        if is_tensor:
            calibrated = torch.tensor(calibrated, dtype=probs.dtype, device=device)
        
        return calibrated


def compute_reliability_diagram(
    y_true: Union[np.ndarray, torch.Tensor],
    y_prob: Union[np.ndarray, torch.Tensor],
    n_bins: int = 10,
) -> Dict[str, np.ndarray]:
    """Compute reliability diagram data.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        
    Returns:
        Dictionary with bin data for plotting
    """
    # Convert to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()
    
    # Flatten
    y_true = y_true.ravel()
    y_prob = y_prob.ravel()
    
    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Compute bin statistics
    bin_accs = []
    bin_confs = []
    bin_counts = []
    
    for i in range(n_bins):
        bin_mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        
        if i == n_bins - 1:  # Include right edge in last bin
            bin_mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        
        if bin_mask.sum() > 0:
            bin_acc = y_true[bin_mask].mean()
            bin_conf = y_prob[bin_mask].mean()
            bin_count = bin_mask.sum()
        else:
            bin_acc = 0
            bin_conf = bin_centers[i]
            bin_count = 0
        
        bin_accs.append(bin_acc)
        bin_confs.append(bin_conf)
        bin_counts.append(bin_count)
    
    return {
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "bin_accs": np.array(bin_accs),
        "bin_confs": np.array(bin_confs),
        "bin_counts": np.array(bin_counts),
    }


def expected_calibration_error(
    y_true: Union[np.ndarray, torch.Tensor],
    y_prob: Union[np.ndarray, torch.Tensor],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).
    
    ECE measures the expected difference between confidence and accuracy.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        
    Returns:
        ECE value
    """
    diagram = compute_reliability_diagram(y_true, y_prob, n_bins)
    
    bin_accs = diagram["bin_accs"]
    bin_confs = diagram["bin_confs"]
    bin_counts = diagram["bin_counts"]
    
    n_samples = bin_counts.sum()
    
    if n_samples == 0:
        return 0.0
    
    # Weighted average of bin errors
    ece = 0
    for acc, conf, count in zip(bin_accs, bin_confs, bin_counts):
        if count > 0:
            ece += (count / n_samples) * abs(acc - conf)
    
    return ece


def maximum_calibration_error(
    y_true: Union[np.ndarray, torch.Tensor],
    y_prob: Union[np.ndarray, torch.Tensor],
    n_bins: int = 10,
) -> float:
    """Compute Maximum Calibration Error (MCE).
    
    MCE measures the maximum difference between confidence and accuracy
    across all bins.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        
    Returns:
        MCE value
    """
    diagram = compute_reliability_diagram(y_true, y_prob, n_bins)
    
    bin_accs = diagram["bin_accs"]
    bin_confs = diagram["bin_confs"]
    bin_counts = diagram["bin_counts"]
    
    # Maximum error across non-empty bins
    mce = 0
    for acc, conf, count in zip(bin_accs, bin_confs, bin_counts):
        if count > 0:
            mce = max(mce, abs(acc - conf))
    
    return mce


def adaptive_calibration_error(
    y_true: Union[np.ndarray, torch.Tensor],
    y_prob: Union[np.ndarray, torch.Tensor],
    n_bins: int = 10,
) -> float:
    """Compute Adaptive Calibration Error (ACE).
    
    ACE uses adaptive binning to have equal number of samples per bin.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        
    Returns:
        ACE value
    """
    # Convert to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()
    
    # Flatten
    y_true = y_true.ravel()
    y_prob = y_prob.ravel()
    
    n_samples = len(y_true)
    
    if n_samples == 0:
        return 0.0
    
    # Sort by predicted probability
    sorted_indices = np.argsort(y_prob)
    y_true_sorted = y_true[sorted_indices]
    y_prob_sorted = y_prob[sorted_indices]
    
    # Create adaptive bins with equal number of samples
    samples_per_bin = n_samples // n_bins
    
    ace = 0
    for i in range(n_bins):
        start_idx = i * samples_per_bin
        
        if i == n_bins - 1:
            # Last bin includes remaining samples
            end_idx = n_samples
        else:
            end_idx = (i + 1) * samples_per_bin
        
        bin_true = y_true_sorted[start_idx:end_idx]
        bin_prob = y_prob_sorted[start_idx:end_idx]
        
        if len(bin_true) > 0:
            bin_acc = bin_true.mean()
            bin_conf = bin_prob.mean()
            bin_weight = len(bin_true) / n_samples
            ace += bin_weight * abs(bin_acc - bin_conf)
    
    return ace


def find_threshold_for_sensitivity(
    y_true: Union[np.ndarray, torch.Tensor],
    y_prob: Union[np.ndarray, torch.Tensor],
    target_sensitivity: float = 0.95,
    class_of_interest: int = 1
) -> float:
    """Find threshold to achieve target sensitivity.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        target_sensitivity: Target sensitivity/recall
        class_of_interest: Class for which to optimize sensitivity
        
    Returns:
        Optimal threshold
    """
    # Convert to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()
    
    # Get samples of class of interest
    class_mask = y_true == class_of_interest
    class_probs = y_prob[class_mask]
    
    if len(class_probs) == 0:
        return 0.5
    
    # Sort probabilities
    sorted_probs = np.sort(class_probs)
    
    # Find threshold that achieves target sensitivity
    # Sensitivity = TP / (TP + FN)
    # We want to correctly classify target_sensitivity proportion of positive samples
    n_positives = len(class_probs)
    n_false_negatives_allowed = int(n_positives * (1 - target_sensitivity))
    
    if n_false_negatives_allowed >= len(sorted_probs):
        threshold = 0.0
    else:
        # Threshold is the (n_false_negatives_allowed + 1)-th smallest probability
        threshold = sorted_probs[n_false_negatives_allowed]
    
    return float(threshold)


def find_threshold_for_specificity(
    y_true: Union[np.ndarray, torch.Tensor],
    y_prob: Union[np.ndarray, torch.Tensor],
    target_specificity: float = 0.95,
    negative_class: int = 0
) -> float:
    """Find threshold to achieve target specificity.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        target_specificity: Target specificity
        negative_class: Negative class label
        
    Returns:
        Optimal threshold
    """
    # Convert to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()
    
    # Get samples of negative class
    neg_mask = y_true == negative_class
    neg_probs = y_prob[neg_mask]
    
    if len(neg_probs) == 0:
        return 0.5
    
    # Sort probabilities
    sorted_probs = np.sort(neg_probs)
    
    # Find threshold that achieves target specificity
    # Specificity = TN / (TN + FP)
    # We want to correctly classify target_specificity proportion of negative samples
    n_negatives = len(neg_probs)
    n_false_positives_allowed = int(n_negatives * (1 - target_specificity))
    
    if n_false_positives_allowed == 0:
        threshold = 1.0
    else:
        # Threshold is the (n_false_positives_allowed)-th largest probability
        threshold = sorted_probs[-n_false_positives_allowed]
    
    return float(threshold)


class CalibrationEvaluator:
    """Comprehensive calibration evaluation."""
    
    def __init__(self, n_bins: int = 10):
        """Initialize evaluator.
        
        Args:
            n_bins: Number of bins for calibration metrics
        """
        self.n_bins = n_bins
    
    def evaluate(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_prob: Union[np.ndarray, torch.Tensor],
    ) -> Dict[str, float]:
        """Evaluate all calibration metrics.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            
        Returns:
            Dictionary of calibration metrics
        """
        ece = expected_calibration_error(y_true, y_prob, self.n_bins)
        mce = maximum_calibration_error(y_true, y_prob, self.n_bins)
        ace = adaptive_calibration_error(y_true, y_prob, self.n_bins)
        
        # Compute Brier score
        if isinstance(y_true, torch.Tensor):
            y_true_np = y_true.cpu().numpy()
        else:
            y_true_np = y_true
        
        if isinstance(y_prob, torch.Tensor):
            y_prob_np = y_prob.cpu().numpy()
        else:
            y_prob_np = y_prob
        
        brier_score = np.mean((y_prob_np - y_true_np) ** 2)
        
        return {
            "ece": ece,
            "mce": mce,
            "ace": ace,
            "brier_score": brier_score,
        }
    
    def find_optimal_thresholds(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_prob: Union[np.ndarray, torch.Tensor],
        sensitivities: List[float] = [0.90, 0.95, 0.99],
        specificities: List[float] = [0.90, 0.95, 0.99],
    ) -> Dict[str, float]:
        """Find optimal thresholds for various operating points.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            sensitivities: Target sensitivity values
            specificities: Target specificity values
            
        Returns:
            Dictionary of thresholds
        """
        thresholds = {}
        
        for sens in sensitivities:
            thresh = find_threshold_for_sensitivity(y_true, y_prob, sens)
            thresholds[f"threshold_sens_{sens:.2f}"] = thresh
        
        for spec in specificities:
            thresh = find_threshold_for_specificity(y_true, y_prob, spec)
            thresholds[f"threshold_spec_{spec:.2f}"] = thresh
        
        return thresholds
