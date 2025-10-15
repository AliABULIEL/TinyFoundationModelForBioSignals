"""
BUT-PPG Downstream Task Evaluators

Implements the 3 main BUT-PPG tasks from the article:
1. Signal Quality Classification (binary)
2. Heart Rate Estimation (regression)
3. Motion Classification (8-class)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    confusion_matrix, classification_report
)


# Benchmarks from article
BUTPPG_BENCHMARKS = {
    'quality': {
        'task_name': 'Signal Quality Classification',
        'metric': 'AUROC',
        'target': 0.88,
        'baseline': 0.758,  # STD-width SQI (traditional)
        'dl_baseline': 0.85,  # Deep learning baseline
        'sota_paper': 'Article Target'
    },
    'hr': {
        'task_name': 'Heart Rate Estimation',
        'metric': 'MAE',
        'target': 2.0,  # bpm
        'human_expert': 1.5,  # Human expert performance
        'baseline': 3.0,  # Traditional methods
        'sota_paper': 'Human Expert (2021)'
    },
    'motion': {
        'task_name': 'Motion Classification',
        'metric': 'Accuracy',
        'target': 0.85,  # 8-class accuracy
        'baseline': 0.70,  # Simple features
        'sota_paper': 'Article Target'
    }
}


class QualityClassifier:
    """
    Task 1: Signal Quality Classification

    Definition (from article):
    - Binary classification: good vs poor quality PPG
    - Expert consensus labels from BUT-PPG
    - Subject-level evaluation

    Metrics:
    - Primary: AUROC (target ≥0.88)
    - Secondary: Accuracy, F1, Sensitivity, Specificity
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.benchmark = BUTPPG_BENCHMARKS['quality']

    def predict(self, data_loader) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Generate quality predictions

        Returns:
            predictions: [N] probability scores for good quality
            labels: [N] binary labels (0=poor, 1=good)
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
                logits = self.model(signals)
                probs = torch.softmax(logits, dim=1)[:, 1]  # Prob of good quality

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
        """Compute bootstrap confidence interval"""
        np.random.seed(42)
        bootstrap_scores = []

        for _ in range(n_bootstrap):
            indices = np.random.choice(len(predictions), size=len(predictions), replace=True)
            boot_preds = predictions[indices]
            boot_labels = labels[indices]

            try:
                score = metric_fn(boot_labels, boot_preds)
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
    ) -> Dict:
        """
        Compute all quality classification metrics

        Args:
            predictions: Probability scores
            labels: Binary labels
            subject_ids: Subject IDs for subject-level evaluation
            compute_ci: Whether to compute confidence intervals

        Returns:
            Dict with all metrics and benchmark comparisons
        """
        # Subject-level aggregation if IDs provided
        if subject_ids is not None:
            unique_subjects = np.unique(subject_ids)
            subject_preds = []
            subject_labels = []

            for subj in unique_subjects:
                mask = subject_ids == subj
                subject_preds.append(np.mean(predictions[mask]))
                subject_labels.append(int(np.round(np.mean(labels[mask]))))

            predictions = np.array(subject_preds)
            labels = np.array(subject_labels)

        # AUROC (primary metric)
        auroc = roc_auc_score(labels, predictions)

        # Binary metrics at threshold 0.5
        binary_preds = (predictions >= 0.5).astype(int)
        accuracy = accuracy_score(labels, binary_preds)
        f1 = f1_score(labels, binary_preds)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(labels, binary_preds).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        results = {
            'auroc': auroc,
            'accuracy': accuracy,
            'f1': f1,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'n_samples': len(predictions),
            'n_positive': int(labels.sum()),
            'n_negative': int((labels == 0).sum()),
        }

        # Compute confidence intervals
        if compute_ci:
            auroc_ci = self.compute_bootstrap_ci(predictions, labels, roc_auc_score)
            results['auroc_ci_lower'] = auroc_ci[0]
            results['auroc_ci_upper'] = auroc_ci[1]

        # Compare to benchmarks
        results['target'] = self.benchmark['target']
        results['baseline'] = self.benchmark['baseline']
        results['dl_baseline'] = self.benchmark['dl_baseline']
        results['target_met'] = auroc >= self.benchmark['target']
        results['improvement_over_baseline'] = auroc - self.benchmark['baseline']
        results['improvement_over_dl'] = auroc - self.benchmark['dl_baseline']

        return results


class HREstimator:
    """
    Task 2: Heart Rate Estimation

    Definition (from article):
    - Estimate HR in bpm from PPG signal
    - Regression task
    - Compare against human expert performance (1.5-2.0 bpm MAE)

    Metrics:
    - Primary: MAE (target ≤2.0 bpm)
    - Secondary: RMSE, MAPE, Within-5-bpm accuracy
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.benchmark = BUTPPG_BENCHMARKS['hr']

    def predict(self, data_loader) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Generate HR predictions

        Returns:
            predictions: [N] predicted HR values
            labels: [N] true HR values
            subject_ids: [N] subject IDs (optional)
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_subjects = []

        with torch.no_grad():
            for batch in data_loader:
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
                preds = self.model(signals).squeeze()

                all_preds.extend(preds.cpu().numpy())
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
    ) -> Dict:
        """
        Compute all HR estimation metrics

        Args:
            predictions: Predicted HR values
            labels: True HR values
            subject_ids: Subject IDs for subject-level evaluation
            compute_ci: Whether to compute confidence intervals

        Returns:
            Dict with all metrics and benchmark comparisons
        """
        # MAE (primary metric)
        mae = np.mean(np.abs(predictions - labels))

        # RMSE
        rmse = np.sqrt(np.mean((predictions - labels) ** 2))

        # MAPE (Mean Absolute Percentage Error)
        # Avoid division by zero
        valid_mask = labels > 0
        if valid_mask.sum() > 0:
            mape = np.mean(np.abs((labels[valid_mask] - predictions[valid_mask]) / labels[valid_mask])) * 100
        else:
            mape = float('inf')

        # Within 5 bpm accuracy (clinical relevance)
        within_5bpm = np.mean(np.abs(predictions - labels) <= 5) * 100

        results = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'within_5bpm': within_5bpm,
            'n_samples': len(predictions),
        }

        # Compute confidence intervals
        if compute_ci:
            mae_fn = lambda p, l: np.mean(np.abs(p - l))
            mae_ci = self.compute_bootstrap_ci(predictions, labels, mae_fn)
            results['mae_ci_lower'] = mae_ci[0]
            results['mae_ci_upper'] = mae_ci[1]

        # Compare to benchmarks
        results['target'] = self.benchmark['target']
        results['human_expert'] = self.benchmark['human_expert']
        results['baseline'] = self.benchmark['baseline']
        results['target_met'] = mae <= self.benchmark['target']
        results['vs_human_expert'] = mae - self.benchmark['human_expert']
        results['vs_baseline'] = self.benchmark['baseline'] - mae

        return results


class MotionClassifier:
    """
    Task 3: Motion Classification (8-class)

    Definition (from article):
    - Classify motion type from accelerometer data
    - 8 motion classes (sitting, standing, walking, running, etc.)
    - Multi-class classification

    Metrics:
    - Primary: Accuracy (target ≥0.85)
    - Secondary: F1 (macro), Per-class accuracy
    """

    def __init__(self, model: nn.Module, device: str = 'cuda', num_classes: int = 8):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.benchmark = BUTPPG_BENCHMARKS['motion']

    def predict(self, data_loader) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Generate motion predictions

        Returns:
            predictions: [N] predicted class labels
            labels: [N] true class labels
            subject_ids: [N] subject IDs (optional)
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_subjects = []

        with torch.no_grad():
            for batch in data_loader:
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
                logits = self.model(signals)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
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
        """Compute bootstrap confidence interval"""
        np.random.seed(42)
        bootstrap_scores = []

        for _ in range(n_bootstrap):
            indices = np.random.choice(len(predictions), size=len(predictions), replace=True)
            boot_preds = predictions[indices]
            boot_labels = labels[indices]

            try:
                score = metric_fn(boot_labels, boot_preds)
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
    ) -> Dict:
        """
        Compute all motion classification metrics

        Args:
            predictions: Predicted class labels
            labels: True class labels
            subject_ids: Subject IDs for subject-level evaluation
            compute_ci: Whether to compute confidence intervals

        Returns:
            Dict with all metrics and benchmark comparisons
        """
        # Accuracy (primary metric)
        accuracy = accuracy_score(labels, predictions)

        # F1 scores
        f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
        f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)

        # Per-class accuracy
        cm = confusion_matrix(labels, predictions, labels=range(self.num_classes))
        per_class_acc = []
        for i in range(self.num_classes):
            if cm[i].sum() > 0:
                per_class_acc.append(cm[i, i] / cm[i].sum())
            else:
                per_class_acc.append(0.0)

        results = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'per_class_accuracy': per_class_acc,
            'confusion_matrix': cm.tolist(),
            'n_samples': len(predictions),
        }

        # Compute confidence intervals
        if compute_ci:
            acc_ci = self.compute_bootstrap_ci(predictions, labels, accuracy_score)
            results['accuracy_ci_lower'] = acc_ci[0]
            results['accuracy_ci_upper'] = acc_ci[1]

        # Compare to benchmarks
        results['target'] = self.benchmark['target']
        results['baseline'] = self.benchmark['baseline']
        results['target_met'] = accuracy >= self.benchmark['target']
        results['improvement_over_baseline'] = accuracy - self.benchmark['baseline']

        return results


def run_all_butppg_tasks(
    quality_model: nn.Module,
    hr_model: Optional[nn.Module],
    motion_model: Optional[nn.Module],
    quality_loader,
    hr_loader=None,
    motion_loader=None,
    device: str = 'cuda',
    compute_ci: bool = True
) -> Dict[str, Dict]:
    """
    Run all 3 BUT-PPG tasks and compile benchmark report

    Args:
        quality_model: Model for quality classification
        hr_model: Model for HR estimation (optional, can be same as quality_model)
        motion_model: Model for motion classification (optional)
        quality_loader: DataLoader for quality task
        hr_loader: DataLoader for HR task (optional)
        motion_loader: DataLoader for motion task (optional)
        device: Device to run on
        compute_ci: Whether to compute confidence intervals

    Returns:
        results: Dict with task results
    """
    results = {}

    # Task 1: Quality Classification
    print("\n" + "="*80)
    print("TASK 1: Signal Quality Classification")
    print("="*80)

    quality_classifier = QualityClassifier(quality_model, device)
    qual_preds, qual_labels, qual_subjects = quality_classifier.predict(quality_loader)
    qual_metrics = quality_classifier.evaluate(qual_preds, qual_labels, qual_subjects, compute_ci)

    results['quality'] = qual_metrics

    print(f"AUROC: {qual_metrics['auroc']:.3f} (Target: {qual_metrics['target']:.3f})")
    if compute_ci:
        print(f"  95% CI: [{qual_metrics['auroc_ci_lower']:.3f}, {qual_metrics['auroc_ci_upper']:.3f}]")
    print(f"Accuracy: {qual_metrics['accuracy']:.3f}")
    print(f"F1: {qual_metrics['f1']:.3f}")
    print(f"Sensitivity: {qual_metrics['sensitivity']:.3f}")
    print(f"Specificity: {qual_metrics['specificity']:.3f}")
    print(f"Samples: {qual_metrics['n_samples']} (Positive: {qual_metrics['n_positive']}, Negative: {qual_metrics['n_negative']})")
    print(f"Target Met: {'✅ YES' if qual_metrics['target_met'] else '❌ NO'}")
    print(f"vs Traditional Baseline: +{qual_metrics['improvement_over_baseline']:.3f}")
    print(f"vs DL Baseline: +{qual_metrics['improvement_over_dl']:.3f}")

    # Task 2: HR Estimation (optional)
    if hr_loader is not None and hr_model is not None:
        print("\n" + "="*80)
        print("TASK 2: Heart Rate Estimation")
        print("="*80)

        hr_estimator = HREstimator(hr_model, device)
        hr_preds, hr_labels, hr_subjects = hr_estimator.predict(hr_loader)
        hr_metrics = hr_estimator.evaluate(hr_preds, hr_labels, hr_subjects, compute_ci)

        results['hr_estimation'] = hr_metrics

        print(f"MAE: {hr_metrics['mae']:.2f} bpm (Target: {hr_metrics['target']:.2f})")
        if compute_ci:
            print(f"  95% CI: [{hr_metrics['mae_ci_lower']:.2f}, {hr_metrics['mae_ci_upper']:.2f}]")
        print(f"RMSE: {hr_metrics['rmse']:.2f} bpm")
        print(f"MAPE: {hr_metrics['mape']:.1f}%")
        print(f"Within 5 bpm: {hr_metrics['within_5bpm']:.1f}%")
        print(f"Samples: {hr_metrics['n_samples']}")
        print(f"Target Met: {'✅ YES' if hr_metrics['target_met'] else '❌ NO'}")
        print(f"vs Human Expert: {hr_metrics['vs_human_expert']:+.2f} bpm")
        print(f"vs Traditional Baseline: +{hr_metrics['vs_baseline']:.2f} bpm")

    # Task 3: Motion Classification (optional)
    if motion_loader is not None and motion_model is not None:
        print("\n" + "="*80)
        print("TASK 3: Motion Classification (8-class)")
        print("="*80)

        motion_classifier = MotionClassifier(motion_model, device)
        motion_preds, motion_labels, motion_subjects = motion_classifier.predict(motion_loader)
        motion_metrics = motion_classifier.evaluate(motion_preds, motion_labels, motion_subjects, compute_ci)

        results['motion'] = motion_metrics

        print(f"Accuracy: {motion_metrics['accuracy']:.3f} (Target: {motion_metrics['target']:.3f})")
        if compute_ci:
            print(f"  95% CI: [{motion_metrics['accuracy_ci_lower']:.3f}, {motion_metrics['accuracy_ci_upper']:.3f}]")
        print(f"F1 (macro): {motion_metrics['f1_macro']:.3f}")
        print(f"F1 (weighted): {motion_metrics['f1_weighted']:.3f}")
        print(f"Samples: {motion_metrics['n_samples']}")
        print(f"Target Met: {'✅ YES' if motion_metrics['target_met'] else '❌ NO'}")
        print(f"vs Baseline: +{motion_metrics['improvement_over_baseline']:.3f}")

    return results
