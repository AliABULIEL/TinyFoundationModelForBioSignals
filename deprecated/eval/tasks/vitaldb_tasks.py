"""
VitalDB Downstream Task Evaluators

Implements the 3 main VitalDB tasks from the article:
1. Hypotension Prediction (10-min ahead)
2. Blood Pressure (MAP) Estimation
3. Blood Pressure with AAMI Compliance
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve


@dataclass
class TaskBenchmark:
    """Benchmark targets from article"""
    task_name: str
    metric_name: str
    target: float
    sota: float
    sota_paper: str
    baseline: Optional[float] = None


# Benchmarks from article
VITALDB_BENCHMARKS = {
    'hypotension': TaskBenchmark(
        task_name='Hypotension Prediction (10-min)',
        metric_name='AUROC',
        target=0.91,
        sota=0.934,
        sota_paper='SAFDNet (2024)',
        baseline=0.88  # ABP-only baseline
    ),
    'bp_mae': TaskBenchmark(
        task_name='Blood Pressure (MAP) Estimation',
        metric_name='MAE',
        target=5.0,
        sota=3.8,
        sota_paper='AnesthNet (2025)',
        baseline=None
    ),
    'bp_aami': TaskBenchmark(
        task_name='Blood Pressure AAMI Compliance',
        metric_name='ME/SDE',
        target=5.0,  # ME
        sota=5.0,    # AAMI standard
        sota_paper='AAMI Standard',
        baseline=None
    )
}


class HypotensionPredictor:
    """
    Task 1: Hypotension Prediction (10-min ahead)

    Definition (from article):
    - Predict hypotension 10 minutes in advance
    - Hypotension: MAP < 65 mmHg or SBP < 90 mmHg
    - Binary classification task
    - Subject-level evaluation

    Metrics:
    - Primary: AUROC (target ≥0.91)
    - Secondary: AUPRC, Sensitivity, Specificity
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.benchmark = VITALDB_BENCHMARKS['hypotension']

    def predict(self, data_loader) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Generate predictions for hypotension

        Returns:
            predictions: [N] probability scores
            labels: [N] binary labels (0=normal, 1=hypotension)
            subject_ids: [N] subject IDs for subject-level evaluation (optional)
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_subjects = []

        with torch.no_grad():
            for batch in data_loader:
                # Handle different batch formats
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    signals, labels = batch
                    subject_ids = None
                elif isinstance(batch, (list, tuple)) and len(batch) == 3:
                    signals, labels, subject_ids = batch
                else:
                    signals = batch['signals']
                    labels = batch['labels']
                    subject_ids = batch.get('subject_ids', None)

                signals = signals.to(self.device)

                # Forward pass
                logits = self.model(signals)
                probs = torch.softmax(logits, dim=1)[:, 1]  # Prob of hypotension

                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy() if torch.is_tensor(labels) else labels)

                if subject_ids is not None:
                    all_subjects.extend(subject_ids.cpu().numpy() if torch.is_tensor(subject_ids) else subject_ids)

        subject_arr = np.array(all_subjects) if all_subjects else None
        return np.array(all_preds), np.array(all_labels), subject_arr

    def compute_bootstrap_ci(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        metric_fn,
        n_bootstrap: int = 1000,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Compute bootstrap confidence interval for a metric"""
        np.random.seed(42)
        bootstrap_scores = []

        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(len(predictions), size=len(predictions), replace=True)
            boot_preds = predictions[indices]
            boot_labels = labels[indices]

            # Compute metric
            try:
                score = metric_fn(boot_labels, boot_preds)
                bootstrap_scores.append(score)
            except:
                continue

        # Compute confidence interval
        alpha = (1 - confidence) / 2
        ci_lower = np.percentile(bootstrap_scores, alpha * 100)
        ci_upper = np.percentile(bootstrap_scores, (1 - alpha) * 100)

        return ci_lower, ci_upper

    def evaluate(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        subject_ids: Optional[np.ndarray] = None,
        compute_ci: bool = True
    ) -> Dict[str, float]:
        """
        Compute all metrics for hypotension prediction

        Args:
            predictions: Probability scores
            labels: Binary labels
            subject_ids: Subject IDs for subject-level evaluation
            compute_ci: Whether to compute confidence intervals

        Returns:
            metrics dict with AUROC, AUPRC, sensitivity, specificity, and CIs
        """
        # Subject-level aggregation if IDs provided
        if subject_ids is not None:
            # Aggregate per subject (mean probability, majority vote label)
            unique_subjects = np.unique(subject_ids)
            subject_preds = []
            subject_labels = []

            for subj in unique_subjects:
                mask = subject_ids == subj
                subject_preds.append(np.mean(predictions[mask]))
                subject_labels.append(int(np.round(np.mean(labels[mask]))))

            predictions = np.array(subject_preds)
            labels = np.array(subject_labels)

        # AUROC
        auroc = roc_auc_score(labels, predictions)

        # AUPRC
        precision, recall, _ = precision_recall_curve(labels, predictions)
        auprc = auc(recall, precision)

        # Sensitivity/Specificity at optimal threshold
        fpr, tpr, thresholds = roc_curve(labels, predictions)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        binary_preds = (predictions >= optimal_threshold).astype(int)
        tp = ((binary_preds == 1) & (labels == 1)).sum()
        tn = ((binary_preds == 0) & (labels == 0)).sum()
        fp = ((binary_preds == 1) & (labels == 0)).sum()
        fn = ((binary_preds == 0) & (labels == 1)).sum()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Compute confidence intervals
        results = {
            'auroc': auroc,
            'auprc': auprc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'optimal_threshold': optimal_threshold,
            'n_samples': len(predictions),
            'n_positive': int(labels.sum()),
            'n_negative': int((labels == 0).sum()),
        }

        if compute_ci:
            auroc_ci = self.compute_bootstrap_ci(predictions, labels, roc_auc_score)
            results['auroc_ci_lower'] = auroc_ci[0]
            results['auroc_ci_upper'] = auroc_ci[1]

        # Compare to benchmarks
        results['target'] = self.benchmark.target
        results['sota'] = self.benchmark.sota
        results['target_met'] = auroc >= self.benchmark.target
        results['sota_gap'] = self.benchmark.sota - auroc
        results['baseline'] = self.benchmark.baseline

        return results


class BPEstimator:
    """
    Task 2 & 3: Blood Pressure Estimation with AAMI Compliance

    Definition (from article):
    - Estimate Mean Arterial Pressure (MAP) from PPG/ECG
    - Regression task
    - Evaluate both MAE and AAMI compliance

    Metrics:
    - MAE (target ≤5.0 mmHg)
    - AAMI: ME (Mean Error) ≤5, SDE (Std Dev Error) ≤8
    - Per-subject evaluation
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.benchmark_mae = VITALDB_BENCHMARKS['bp_mae']
        self.benchmark_aami = VITALDB_BENCHMARKS['bp_aami']

    def predict(self, data_loader) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Generate BP predictions

        Returns:
            predictions: [N] predicted MAP values
            labels: [N] true MAP values
            subject_ids: [N] subject IDs (optional)
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_subjects = []

        with torch.no_grad():
            for batch in data_loader:
                # Handle different batch formats
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    signals, labels = batch
                    subject_ids = None
                elif isinstance(batch, (list, tuple)) and len(batch) == 3:
                    signals, labels, subject_ids = batch
                else:
                    signals = batch['signals']
                    labels = batch['labels']
                    subject_ids = batch.get('subject_ids', None)

                signals = signals.to(self.device)

                # Forward pass
                preds = self.model(signals).squeeze()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy() if torch.is_tensor(labels) else labels)

                if subject_ids is not None:
                    all_subjects.extend(subject_ids.cpu().numpy() if torch.is_tensor(subject_ids) else subject_ids)

        subject_arr = np.array(all_subjects) if all_subjects else None
        return np.array(all_preds), np.array(all_labels), subject_arr

    def compute_aami_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        per_subject: bool = True,
        subject_ids: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute AAMI compliance metrics

        AAMI Standard (ANSI/AAMI SP10):
        - Mean Error (ME) ≤ 5 mmHg
        - Standard Deviation of Error (SDE) ≤ 8 mmHg

        Article specifies: Must compute per-subject first, then aggregate
        """
        if per_subject and subject_ids is not None:
            # Compute per-subject errors first
            unique_subjects = np.unique(subject_ids)
            subject_mes = []

            for subj in unique_subjects:
                mask = subject_ids == subj
                subj_preds = predictions[mask]
                subj_labels = labels[mask]

                # Mean error for this subject
                me = np.mean(subj_preds - subj_labels)
                subject_mes.append(me)

            # Aggregate across subjects
            me = np.mean(subject_mes)
            sde = np.std(subject_mes, ddof=1)  # Use sample std
        else:
            # Simple pooled calculation (not recommended by article!)
            errors = predictions - labels
            me = np.mean(errors)
            sde = np.std(errors, ddof=1)

        # AAMI compliance
        me_compliant = np.abs(me) <= 5.0
        sde_compliant = sde <= 8.0
        aami_compliant = me_compliant and sde_compliant

        return {
            'me': me,
            'sde': sde,
            'me_compliant': me_compliant,
            'sde_compliant': sde_compliant,
            'aami_compliant': aami_compliant
        }

    def compute_bootstrap_ci(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        metric_fn,
        n_bootstrap: int = 1000,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Compute bootstrap confidence interval"""
        np.random.seed(42)
        bootstrap_scores = []

        for _ in range(n_bootstrap):
            indices = np.random.choice(len(predictions), size=len(predictions), replace=True)
            boot_preds = predictions[indices]
            boot_labels = labels[indices]

            try:
                score = metric_fn(boot_preds, boot_labels)
                bootstrap_scores.append(score)
            except:
                continue

        alpha = (1 - confidence) / 2
        ci_lower = np.percentile(bootstrap_scores, alpha * 100)
        ci_upper = np.percentile(bootstrap_scores, (1 - alpha) * 100)

        return ci_lower, ci_upper

    def evaluate(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        subject_ids: Optional[np.ndarray] = None,
        compute_ci: bool = True
    ) -> Dict[str, float]:
        """
        Compute all BP estimation metrics

        Returns:
        - MAE, RMSE, R²
        - AAMI: ME, SDE, compliance
        - Confidence intervals
        - Comparison to benchmarks
        """
        # Basic regression metrics
        mae = np.mean(np.abs(predictions - labels))
        rmse = np.sqrt(np.mean((predictions - labels) ** 2))

        # R² (with caution - article warns about this)
        ss_res = np.sum((labels - predictions) ** 2)
        ss_tot = np.sum((labels - np.mean(labels)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # AAMI metrics
        aami_metrics = self.compute_aami_metrics(
            predictions, labels,
            per_subject=True if subject_ids is not None else False,
            subject_ids=subject_ids
        )

        results = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'n_samples': len(predictions),
            **aami_metrics,
        }

        # Compute confidence intervals
        if compute_ci:
            mae_fn = lambda p, l: np.mean(np.abs(p - l))
            mae_ci = self.compute_bootstrap_ci(predictions, labels, mae_fn)
            results['mae_ci_lower'] = mae_ci[0]
            results['mae_ci_upper'] = mae_ci[1]

        # Compare to benchmarks
        results['mae_target'] = self.benchmark_mae.target
        results['mae_sota'] = self.benchmark_mae.sota
        results['mae_target_met'] = mae <= self.benchmark_mae.target
        results['mae_sota_gap'] = mae - self.benchmark_mae.sota

        return results


def run_all_vitaldb_tasks(
    model: nn.Module,
    hypotension_loader,
    bp_loader,
    device: str = 'cuda',
    compute_ci: bool = True
) -> Dict[str, Dict]:
    """
    Run all 3 VitalDB tasks and compile benchmark report

    Args:
        model: Fine-tuned model
        hypotension_loader: DataLoader for hypotension task
        bp_loader: DataLoader for BP estimation task
        device: Device to run on
        compute_ci: Whether to compute confidence intervals

    Returns:
        results: Dict with keys 'hypotension', 'bp_estimation'
    """
    results = {}

    # Task 1: Hypotension Prediction
    print("\n" + "="*80)
    print("TASK 1: Hypotension Prediction (10-min ahead)")
    print("="*80)

    hypo_predictor = HypotensionPredictor(model, device)
    hypo_preds, hypo_labels, hypo_subjects = hypo_predictor.predict(hypotension_loader)
    hypo_metrics = hypo_predictor.evaluate(hypo_preds, hypo_labels, hypo_subjects, compute_ci)

    results['hypotension'] = hypo_metrics

    print(f"AUROC: {hypo_metrics['auroc']:.3f} (Target: {hypo_metrics['target']:.3f})")
    if compute_ci:
        print(f"  95% CI: [{hypo_metrics['auroc_ci_lower']:.3f}, {hypo_metrics['auroc_ci_upper']:.3f}]")
    print(f"AUPRC: {hypo_metrics['auprc']:.3f}")
    print(f"Sensitivity: {hypo_metrics['sensitivity']:.3f}")
    print(f"Specificity: {hypo_metrics['specificity']:.3f}")
    print(f"Samples: {hypo_metrics['n_samples']} (Positive: {hypo_metrics['n_positive']}, Negative: {hypo_metrics['n_negative']})")
    print(f"Target Met: {'✅ YES' if hypo_metrics['target_met'] else '❌ NO'}")
    print(f"Gap to SOTA: {hypo_metrics['sota_gap']:.3f}")

    # Task 2 & 3: BP Estimation
    print("\n" + "="*80)
    print("TASK 2 & 3: Blood Pressure Estimation (MAE + AAMI)")
    print("="*80)

    bp_estimator = BPEstimator(model, device)
    bp_preds, bp_labels, bp_subjects = bp_estimator.predict(bp_loader)
    bp_metrics = bp_estimator.evaluate(bp_preds, bp_labels, bp_subjects, compute_ci)

    results['bp_estimation'] = bp_metrics

    print(f"MAE: {bp_metrics['mae']:.2f} mmHg (Target: {bp_metrics['mae_target']:.2f})")
    if compute_ci:
        print(f"  95% CI: [{bp_metrics['mae_ci_lower']:.2f}, {bp_metrics['mae_ci_upper']:.2f}]")
    print(f"RMSE: {bp_metrics['rmse']:.2f} mmHg")
    print(f"R²: {bp_metrics['r2']:.3f}")
    print(f"\nAAMI Compliance:")
    print(f"  ME: {bp_metrics['me']:.2f} mmHg (Limit: 5.0) {'✅' if bp_metrics['me_compliant'] else '❌'}")
    print(f"  SDE: {bp_metrics['sde']:.2f} mmHg (Limit: 8.0) {'✅' if bp_metrics['sde_compliant'] else '❌'}")
    print(f"  AAMI Compliant: {'✅ YES' if bp_metrics['aami_compliant'] else '❌ NO'}")
    print(f"Samples: {bp_metrics['n_samples']}")
    print(f"MAE Target Met: {'✅ YES' if bp_metrics['mae_target_met'] else '❌ NO'}")
    print(f"Gap to SOTA: {bp_metrics['mae_sota_gap']:.2f} mmHg")

    return results
