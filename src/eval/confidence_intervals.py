"""Confidence interval computation for medical AI evaluation.

Implements:
1. Bootstrap confidence intervals (percentile and BCa)
2. Exact binomial confidence intervals (Wilson score)
3. Subject-level stratified bootstrapping

Essential for publication-ready medical AI results. WITHOUT confidence intervals,
results are not statistically rigorous and cannot be published in medical journals.

References:
- Efron & Tibshirani (1993): An Introduction to the Bootstrap
- Wilson (1927): Probable Inference, the Law of Succession, and Statistical Inference
- Carpenter & Bithell (2000): Bootstrap confidence intervals
"""

from typing import Callable, Dict, Optional, Tuple, Union
import logging

import numpy as np
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)

logger = logging.getLogger(__name__)


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    method: str = 'percentile',
    random_state: Optional[int] = 42,
    subject_ids: Optional[np.ndarray] = None
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for any metric.

    Uses percentile bootstrap (default) or BCa (bias-corrected and accelerated).
    Supports subject-level stratified bootstrapping to maintain subject independence.

    Args:
        y_true: Ground truth labels/values [N]
        y_pred: Predicted labels/values [N] or probabilities [N, C]
        metric_fn: Function that takes (y_true, y_pred) and returns scalar metric
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        method: 'percentile' or 'bca' (bias-corrected accelerated)
        random_state: Random seed for reproducibility
        subject_ids: Optional subject IDs for stratified bootstrapping [N]

    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound)

    Example:
        >>> y_true = np.array([0, 0, 1, 1, 1])
        >>> y_pred = np.array([0.1, 0.3, 0.6, 0.8, 0.9])
        >>> auroc, ci_low, ci_high = bootstrap_ci(
        ...     y_true, y_pred,
        ...     metric_fn=lambda yt, yp: roc_auc_score(yt, yp),
        ...     n_bootstrap=1000
        ... )
        >>> print(f"AUROC: {auroc:.3f} [{ci_low:.3f}, {ci_high:.3f}]")
        AUROC: 0.933 [0.733, 1.000]
    """
    # Compute point estimate
    point_estimate = metric_fn(y_true, y_pred)

    # Set random seed
    if random_state is not None:
        np.random.seed(random_state)

    # Bootstrap samples
    n_samples = len(y_true)
    bootstrap_scores = []

    for i in range(n_bootstrap):
        # Subject-level or sample-level bootstrap
        if subject_ids is not None:
            # Stratified bootstrap: sample subjects, then take all their samples
            unique_subjects = np.unique(subject_ids)
            sampled_subjects = np.random.choice(
                unique_subjects,
                size=len(unique_subjects),
                replace=True
            )

            # Get indices for sampled subjects
            indices = np.concatenate([
                np.where(subject_ids == subj)[0]
                for subj in sampled_subjects
            ])
        else:
            # Regular bootstrap: sample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)

        # Handle empty bootstrap sample (rare but possible)
        if len(indices) == 0:
            continue

        # Bootstrap sample
        y_true_boot = y_true[indices]
        if y_pred.ndim > 1:
            y_pred_boot = y_pred[indices, :]
        else:
            y_pred_boot = y_pred[indices]

        # Skip if bootstrap sample has only one class (cannot compute AUROC, etc.)
        if len(np.unique(y_true_boot)) < 2:
            continue

        # Compute metric on bootstrap sample
        try:
            score = metric_fn(y_true_boot, y_pred_boot)
            if not np.isnan(score) and not np.isinf(score):
                bootstrap_scores.append(score)
        except Exception as e:
            # Skip this bootstrap sample if metric computation fails
            logger.debug(f"Bootstrap sample {i} failed: {e}")
            continue

    if len(bootstrap_scores) < n_bootstrap * 0.8:
        logger.warning(
            f"Only {len(bootstrap_scores)}/{n_bootstrap} bootstrap samples succeeded. "
            f"CI may be unreliable."
        )

    bootstrap_scores = np.array(bootstrap_scores)

    # Compute confidence interval
    alpha = 1 - confidence_level

    if method == 'percentile':
        # Percentile method (simple and robust)
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)

        ci_lower = np.percentile(bootstrap_scores, lower_percentile)
        ci_upper = np.percentile(bootstrap_scores, upper_percentile)

    elif method == 'bca':
        # BCa method (more accurate but complex)
        # Compute bias-correction factor
        z0 = stats.norm.ppf((bootstrap_scores < point_estimate).mean())

        # Compute acceleration factor (jackknife)
        jackknife_scores = []
        for i in range(n_samples):
            indices = np.concatenate([np.arange(i), np.arange(i + 1, n_samples)])
            y_true_jack = y_true[indices]
            if y_pred.ndim > 1:
                y_pred_jack = y_pred[indices, :]
            else:
                y_pred_jack = y_pred[indices]

            try:
                score = metric_fn(y_true_jack, y_pred_jack)
                if not np.isnan(score):
                    jackknife_scores.append(score)
            except:
                continue

        jackknife_scores = np.array(jackknife_scores)
        jackknife_mean = jackknife_scores.mean()

        # Acceleration factor
        numerator = ((jackknife_mean - jackknife_scores) ** 3).sum()
        denominator = 6 * ((jackknife_mean - jackknife_scores) ** 2).sum() ** 1.5
        a = numerator / (denominator + 1e-8)

        # Adjusted percentiles
        z_alpha_lower = stats.norm.ppf(alpha / 2)
        z_alpha_upper = stats.norm.ppf(1 - alpha / 2)

        p_lower = stats.norm.cdf(z0 + (z0 + z_alpha_lower) / (1 - a * (z0 + z_alpha_lower)))
        p_upper = stats.norm.cdf(z0 + (z0 + z_alpha_upper) / (1 - a * (z0 + z_alpha_upper)))

        ci_lower = np.percentile(bootstrap_scores, 100 * p_lower)
        ci_upper = np.percentile(bootstrap_scores, 100 * p_upper)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'percentile' or 'bca'.")

    return point_estimate, ci_lower, ci_upper


def exact_binomial_ci(
    n_successes: int,
    n_trials: int,
    confidence_level: float = 0.95,
    method: str = 'wilson'
) -> Tuple[float, float, float]:
    """Compute exact confidence interval for binomial proportion.

    Uses Wilson score interval (recommended) or Clopper-Pearson (exact but conservative).

    Appropriate for:
    - Sensitivity, Specificity, Accuracy (classification metrics)
    - Any metric that is a proportion/rate

    Args:
        n_successes: Number of successes
        n_trials: Total number of trials
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        method: 'wilson' (recommended) or 'clopper-pearson'

    Returns:
        Tuple of (proportion, lower_bound, upper_bound)

    Example:
        >>> # 90 true positives out of 100 positive cases
        >>> sensitivity, ci_low, ci_high = exact_binomial_ci(90, 100)
        >>> print(f"Sensitivity: {sensitivity:.3f} [{ci_low:.3f}, {ci_high:.3f}]")
        Sensitivity: 0.900 [0.829, 0.947]
    """
    if n_trials == 0:
        return 0.0, 0.0, 0.0

    # Point estimate
    p = n_successes / n_trials

    alpha = 1 - confidence_level

    if method == 'wilson':
        # Wilson score interval (recommended for most cases)
        # More accurate than normal approximation, especially for small n or extreme p

        z = stats.norm.ppf(1 - alpha / 2)
        z2 = z ** 2

        denominator = 1 + z2 / n_trials
        center = (p + z2 / (2 * n_trials)) / denominator
        margin = z * np.sqrt(p * (1 - p) / n_trials + z2 / (4 * n_trials ** 2)) / denominator

        ci_lower = max(0.0, center - margin)
        ci_upper = min(1.0, center + margin)

    elif method == 'clopper-pearson':
        # Clopper-Pearson (exact but conservative)
        # Based on Beta distribution
        if n_successes == 0:
            ci_lower = 0.0
        else:
            ci_lower = stats.beta.ppf(alpha / 2, n_successes, n_trials - n_successes + 1)

        if n_successes == n_trials:
            ci_upper = 1.0
        else:
            ci_upper = stats.beta.ppf(1 - alpha / 2, n_successes + 1, n_trials - n_successes)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'wilson' or 'clopper-pearson'.")

    return p, ci_lower, ci_upper


def compute_classification_metrics_with_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    subject_ids: Optional[np.ndarray] = None,
    random_state: Optional[int] = 42
) -> Dict[str, Dict[str, float]]:
    """Compute classification metrics with 95% confidence intervals.

    Includes:
    - AUROC (bootstrap CI)
    - AUPRC (bootstrap CI)
    - Accuracy (exact binomial CI)
    - Sensitivity/Recall (exact binomial CI)
    - Specificity (exact binomial CI)
    - Precision (exact binomial CI)
    - F1 score (bootstrap CI)

    Args:
        y_true: Ground truth binary labels [N]
        y_pred: Predicted binary labels [N]
        y_pred_proba: Predicted probabilities [N] (for AUROC/AUPRC)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default: 0.95)
        subject_ids: Optional subject IDs for stratified bootstrapping
        random_state: Random seed

    Returns:
        Dictionary with structure:
            {
                'auroc': {'value': float, 'ci_lower': float, 'ci_upper': float},
                'auprc': {...},
                'accuracy': {...},
                ...
            }

    Example:
        >>> y_true = np.array([0, 0, 1, 1, 1])
        >>> y_pred = np.array([0, 0, 1, 1, 0])
        >>> y_proba = np.array([0.1, 0.3, 0.8, 0.9, 0.6])
        >>> metrics = compute_classification_metrics_with_ci(y_true, y_pred, y_proba)
        >>> print(f"AUROC: {metrics['auroc']['value']:.3f} "
        ...       f"[{metrics['auroc']['ci_lower']:.3f}, {metrics['auroc']['ci_upper']:.3f}]")
    """
    results = {}

    # AUROC with bootstrap CI
    if y_pred_proba is not None and len(np.unique(y_true)) > 1:
        try:
            auroc, auroc_lower, auroc_upper = bootstrap_ci(
                y_true, y_pred_proba,
                metric_fn=roc_auc_score,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                subject_ids=subject_ids,
                random_state=random_state
            )
            results['auroc'] = {
                'value': auroc,
                'ci_lower': auroc_lower,
                'ci_upper': auroc_upper,
                'method': 'bootstrap'
            }
        except Exception as e:
            logger.warning(f"Could not compute AUROC: {e}")

    # AUPRC with bootstrap CI
    if y_pred_proba is not None and len(np.unique(y_true)) > 1:
        try:
            auprc, auprc_lower, auprc_upper = bootstrap_ci(
                y_true, y_pred_proba,
                metric_fn=average_precision_score,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                subject_ids=subject_ids,
                random_state=random_state
            )
            results['auprc'] = {
                'value': auprc,
                'ci_lower': auprc_lower,
                'ci_upper': auprc_upper,
                'method': 'bootstrap'
            }
        except Exception as e:
            logger.warning(f"Could not compute AUPRC: {e}")

    # Confusion matrix elements
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    n_samples = len(y_true)
    n_positive = np.sum(y_true == 1)
    n_negative = np.sum(y_true == 0)

    # Accuracy with exact binomial CI
    n_correct = tp + tn
    acc, acc_lower, acc_upper = exact_binomial_ci(n_correct, n_samples, confidence_level)
    results['accuracy'] = {
        'value': acc,
        'ci_lower': acc_lower,
        'ci_upper': acc_upper,
        'method': 'wilson_score'
    }

    # Sensitivity (Recall) with exact binomial CI
    if n_positive > 0:
        sens, sens_lower, sens_upper = exact_binomial_ci(tp, n_positive, confidence_level)
        results['sensitivity'] = {
            'value': sens,
            'ci_lower': sens_lower,
            'ci_upper': sens_upper,
            'method': 'wilson_score'
        }
        results['recall'] = results['sensitivity']  # Alias

    # Specificity with exact binomial CI
    if n_negative > 0:
        spec, spec_lower, spec_upper = exact_binomial_ci(tn, n_negative, confidence_level)
        results['specificity'] = {
            'value': spec,
            'ci_lower': spec_lower,
            'ci_upper': spec_upper,
            'method': 'wilson_score'
        }

    # Precision with exact binomial CI
    n_predicted_positive = tp + fp
    if n_predicted_positive > 0:
        prec, prec_lower, prec_upper = exact_binomial_ci(tp, n_predicted_positive, confidence_level)
        results['precision'] = {
            'value': prec,
            'ci_lower': prec_lower,
            'ci_upper': prec_upper,
            'method': 'wilson_score'
        }

    # F1 score with bootstrap CI
    try:
        f1, f1_lower, f1_upper = bootstrap_ci(
            y_true, y_pred,
            metric_fn=lambda yt, yp: f1_score(yt, yp),
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            subject_ids=subject_ids,
            random_state=random_state
        )
        results['f1'] = {
            'value': f1,
            'ci_lower': f1_lower,
            'ci_upper': f1_upper,
            'method': 'bootstrap'
        }
    except Exception as e:
        logger.warning(f"Could not compute F1: {e}")

    return results


def compute_regression_metrics_with_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    subject_ids: Optional[np.ndarray] = None,
    random_state: Optional[int] = 42
) -> Dict[str, Dict[str, float]]:
    """Compute regression metrics with 95% confidence intervals.

    Includes:
    - MAE (bootstrap CI)
    - RMSE (bootstrap CI)
    - MSE (bootstrap CI)
    - R² (bootstrap CI, use with caution)
    - Mean Error / Bias (bootstrap CI)
    - Std Error (bootstrap CI)

    Args:
        y_true: Ground truth values [N]
        y_pred: Predicted values [N]
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default: 0.95)
        subject_ids: Optional subject IDs for stratified bootstrapping
        random_state: Random seed

    Returns:
        Dictionary with structure:
            {
                'mae': {'value': float, 'ci_lower': float, 'ci_upper': float},
                'rmse': {...},
                ...
            }

    Example:
        >>> y_true = np.array([65, 70, 75, 80, 85])
        >>> y_pred = np.array([63, 72, 74, 81, 88])
        >>> metrics = compute_regression_metrics_with_ci(y_true, y_pred)
        >>> print(f"MAE: {metrics['mae']['value']:.2f} mmHg "
        ...       f"[{metrics['mae']['ci_lower']:.2f}, {metrics['mae']['ci_upper']:.2f}]")
    """
    results = {}

    # MAE with bootstrap CI
    mae, mae_lower, mae_upper = bootstrap_ci(
        y_true, y_pred,
        metric_fn=mean_absolute_error,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        subject_ids=subject_ids,
        random_state=random_state
    )
    results['mae'] = {
        'value': mae,
        'ci_lower': mae_lower,
        'ci_upper': mae_upper,
        'method': 'bootstrap'
    }

    # RMSE with bootstrap CI
    def rmse_fn(yt, yp):
        return np.sqrt(mean_squared_error(yt, yp))

    rmse, rmse_lower, rmse_upper = bootstrap_ci(
        y_true, y_pred,
        metric_fn=rmse_fn,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        subject_ids=subject_ids,
        random_state=random_state
    )
    results['rmse'] = {
        'value': rmse,
        'ci_lower': rmse_lower,
        'ci_upper': rmse_upper,
        'method': 'bootstrap'
    }

    # MSE with bootstrap CI
    mse, mse_lower, mse_upper = bootstrap_ci(
        y_true, y_pred,
        metric_fn=mean_squared_error,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        subject_ids=subject_ids,
        random_state=random_state
    )
    results['mse'] = {
        'value': mse,
        'ci_lower': mse_lower,
        'ci_upper': mse_upper,
        'method': 'bootstrap'
    }

    # Mean Error (bias) with bootstrap CI
    def mean_error_fn(yt, yp):
        return np.mean(yp - yt)

    me, me_lower, me_upper = bootstrap_ci(
        y_true, y_pred,
        metric_fn=mean_error_fn,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        subject_ids=subject_ids,
        random_state=random_state
    )
    results['mean_error'] = {
        'value': me,
        'ci_lower': me_lower,
        'ci_upper': me_upper,
        'method': 'bootstrap'
    }

    # Std Error with bootstrap CI
    def std_error_fn(yt, yp):
        return np.std(yp - yt)

    se, se_lower, se_upper = bootstrap_ci(
        y_true, y_pred,
        metric_fn=std_error_fn,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        subject_ids=subject_ids,
        random_state=random_state
    )
    results['std_error'] = {
        'value': se,
        'ci_lower': se_lower,
        'ci_upper': se_upper,
        'method': 'bootstrap'
    }

    # R² with bootstrap CI (use with caution - can be negative)
    try:
        r2, r2_lower, r2_upper = bootstrap_ci(
            y_true, y_pred,
            metric_fn=r2_score,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            subject_ids=subject_ids,
            random_state=random_state
        )
        results['r2'] = {
            'value': r2,
            'ci_lower': r2_lower,
            'ci_upper': r2_upper,
            'method': 'bootstrap',
            'warning': 'R² can be negative and misleading for small samples'
        }
    except Exception as e:
        logger.warning(f"Could not compute R²: {e}")

    return results


def format_metric_with_ci(
    metric_dict: Dict[str, float],
    precision: int = 3
) -> str:
    """Format metric with CI for display.

    Args:
        metric_dict: Dict with 'value', 'ci_lower', 'ci_upper' keys
        precision: Number of decimal places

    Returns:
        Formatted string like "0.934 [0.912, 0.956]"

    Example:
        >>> metric = {'value': 0.934, 'ci_lower': 0.912, 'ci_upper': 0.956}
        >>> print(format_metric_with_ci(metric))
        0.934 [0.912, 0.956]
    """
    value = metric_dict['value']
    ci_lower = metric_dict['ci_lower']
    ci_upper = metric_dict['ci_upper']

    return f"{value:.{precision}f} [{ci_lower:.{precision}f}, {ci_upper:.{precision}f}]"
