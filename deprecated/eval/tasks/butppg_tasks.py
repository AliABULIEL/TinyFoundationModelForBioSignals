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


# Benchmarks from article and clinical standards
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
    },
    'bp_systolic': {
        'task_name': 'Systolic Blood Pressure Estimation',
        'metric': 'MAE',
        'target': 5.0,  # mmHg (AAMI standard)
        'aami_standard': 5.0,  # AAMI: ME ± 5 mmHg
        'aami_std': 8.0,  # AAMI: SDE ≤ 8 mmHg
        'baseline': 8.0,  # Traditional cuffless methods
        'sota_paper': 'AAMI/ISO Standard (2013)'
    },
    'bp_diastolic': {
        'task_name': 'Diastolic Blood Pressure Estimation',
        'metric': 'MAE',
        'target': 5.0,  # mmHg (AAMI standard)
        'aami_standard': 5.0,
        'aami_std': 8.0,
        'baseline': 6.0,
        'sota_paper': 'AAMI/ISO Standard (2013)'
    },
    'spo2': {
        'task_name': 'SpO2 Estimation',
        'metric': 'MAE',
        'target': 2.0,  # percentage (clinical standard)
        'clinical_standard': 2.0,  # ±2% accuracy
        'baseline': 3.0,  # Traditional pulse oximetry estimation
        'sota_paper': 'Clinical Standard'
    },
    'glycaemia': {
        'task_name': 'Glycaemia Estimation',
        'metric': 'MAE',
        'target': 1.0,  # mmol/l
        'baseline': 1.5,  # Traditional methods
        'sota_paper': 'Research Target'
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


class BPEstimator:
    """
    Task 4: Blood Pressure Estimation (Systolic + Diastolic)

    Definition:
    - Estimate systolic and diastolic BP from PPG/ECG
    - Regression task for cuffless BP monitoring
    - AAMI/ISO standard: MAE ≤ 5 mmHg, STD ≤ 8 mmHg

    Metrics:
    - Primary: MAE for systolic and diastolic separately
    - Secondary: RMSE, STD (for AAMI compliance), correlation
    """

    def __init__(self, model: nn.Module, device: str = 'cuda', bp_type: str = 'systolic'):
        self.model = model
        self.device = device
        self.bp_type = bp_type  # 'systolic' or 'diastolic'
        self.benchmark = BUTPPG_BENCHMARKS[f'bp_{bp_type}']

    def predict(self, data_loader) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Generate BP predictions

        Returns:
            predictions: [N] predicted BP values (mmHg)
            labels: [N] true BP values (mmHg)
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

                # Model may output 1 value (single BP) or 2 values (sys + dia)
                output = self.model(signals)

                if output.dim() == 1:
                    preds = output
                elif output.shape[1] == 2:
                    # Model outputs [systolic, diastolic]
                    preds = output[:, 0] if self.bp_type == 'systolic' else output[:, 1]
                else:
                    # Single output
                    preds = output.squeeze()

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
        Compute all BP estimation metrics with AAMI compliance

        Args:
            predictions: Predicted BP values (mmHg)
            labels: True BP values (mmHg)
            subject_ids: Subject IDs for subject-level evaluation
            compute_ci: Whether to compute confidence intervals

        Returns:
            Dict with all metrics and AAMI compliance
        """
        # MAE (primary metric)
        mae = np.mean(np.abs(predictions - labels))

        # RMSE
        rmse = np.sqrt(np.mean((predictions - labels) ** 2))

        # Mean Error (bias) and STD for AAMI compliance
        errors = predictions - labels
        me = np.mean(errors)  # Mean Error (bias)
        std = np.std(errors)  # Standard Deviation of Errors

        # Pearson correlation
        correlation = np.corrcoef(predictions, labels)[0, 1]

        # AAMI compliance check
        aami_mae_compliant = mae <= self.benchmark['aami_standard']
        aami_std_compliant = std <= self.benchmark['aami_std']
        aami_compliant = aami_mae_compliant and aami_std_compliant

        results = {
            'mae': mae,
            'rmse': rmse,
            'me': me,  # Mean Error (bias)
            'std': std,  # Standard deviation of errors
            'correlation': correlation,
            'n_samples': len(predictions),
            'bp_type': self.bp_type,
        }

        # AAMI compliance
        results['aami_compliant'] = aami_compliant
        results['aami_mae_compliant'] = aami_mae_compliant
        results['aami_std_compliant'] = aami_std_compliant
        results['aami_standard'] = self.benchmark['aami_standard']
        results['aami_std_standard'] = self.benchmark['aami_std']

        # Compute confidence intervals
        if compute_ci:
            mae_fn = lambda p, l: np.mean(np.abs(p - l))
            mae_ci = self.compute_bootstrap_ci(predictions, labels, mae_fn)
            results['mae_ci_lower'] = mae_ci[0]
            results['mae_ci_upper'] = mae_ci[1]

        # Compare to benchmarks
        results['target'] = self.benchmark['target']
        results['baseline'] = self.benchmark['baseline']
        results['target_met'] = mae <= self.benchmark['target']
        results['vs_baseline'] = self.benchmark['baseline'] - mae

        return results


class SpO2Estimator:
    """
    Task 5: SpO2 (Oxygen Saturation) Estimation

    Definition:
    - Estimate SpO2 percentage from PPG signal
    - Regression task
    - Clinical standard: ±2% accuracy

    Metrics:
    - Primary: MAE (target ≤ 2%)
    - Secondary: RMSE, correlation, within-2% accuracy
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.benchmark = BUTPPG_BENCHMARKS['spo2']

    def predict(self, data_loader) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Generate SpO2 predictions

        Returns:
            predictions: [N] predicted SpO2 values (%)
            labels: [N] true SpO2 values (%)
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
        Compute all SpO2 estimation metrics

        Args:
            predictions: Predicted SpO2 values (%)
            labels: True SpO2 values (%)
            subject_ids: Subject IDs for subject-level evaluation
            compute_ci: Whether to compute confidence intervals

        Returns:
            Dict with all metrics and clinical standard compliance
        """
        # MAE (primary metric)
        mae = np.mean(np.abs(predictions - labels))

        # RMSE
        rmse = np.sqrt(np.mean((predictions - labels) ** 2))

        # Correlation
        correlation = np.corrcoef(predictions, labels)[0, 1]

        # Within 2% accuracy (clinical relevance)
        within_2pct = np.mean(np.abs(predictions - labels) <= 2) * 100

        # Within 3% accuracy
        within_3pct = np.mean(np.abs(predictions - labels) <= 3) * 100

        results = {
            'mae': mae,
            'rmse': rmse,
            'correlation': correlation,
            'within_2pct': within_2pct,
            'within_3pct': within_3pct,
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
        results['clinical_standard'] = self.benchmark['clinical_standard']
        results['baseline'] = self.benchmark['baseline']
        results['target_met'] = mae <= self.benchmark['target']
        results['clinical_compliant'] = mae <= self.benchmark['clinical_standard']
        results['vs_baseline'] = self.benchmark['baseline'] - mae

        return results


class GlycaemiaEstimator:
    """
    Task 6: Glycaemia (Blood Glucose) Estimation

    Definition:
    - Estimate blood glucose from PPG signal
    - Regression task
    - Research target: MAE ≤ 1.0 mmol/l

    Metrics:
    - Primary: MAE (target ≤ 1.0 mmol/l)
    - Secondary: RMSE, correlation, MAPE
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.benchmark = BUTPPG_BENCHMARKS['glycaemia']

    def predict(self, data_loader) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Generate glycaemia predictions

        Returns:
            predictions: [N] predicted glucose values (mmol/l)
            labels: [N] true glucose values (mmol/l)
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
        Compute all glycaemia estimation metrics

        Args:
            predictions: Predicted glucose values (mmol/l)
            labels: True glucose values (mmol/l)
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
        valid_mask = labels > 0
        if valid_mask.sum() > 0:
            mape = np.mean(np.abs((labels[valid_mask] - predictions[valid_mask]) / labels[valid_mask])) * 100
        else:
            mape = float('inf')

        # Correlation
        correlation = np.corrcoef(predictions, labels)[0, 1]

        results = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'correlation': correlation,
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
        results['baseline'] = self.benchmark['baseline']
        results['target_met'] = mae <= self.benchmark['target']
        results['vs_baseline'] = self.benchmark['baseline'] - mae

        return results


def run_all_butppg_tasks(
    quality_model: nn.Module,
    hr_model: Optional[nn.Module],
    motion_model: Optional[nn.Module],
    quality_loader,
    hr_loader=None,
    motion_loader=None,
    bp_systolic_model: Optional[nn.Module] = None,
    bp_diastolic_model: Optional[nn.Module] = None,
    spo2_model: Optional[nn.Module] = None,
    glycaemia_model: Optional[nn.Module] = None,
    bp_systolic_loader=None,
    bp_diastolic_loader=None,
    spo2_loader=None,
    glycaemia_loader=None,
    device: str = 'cuda',
    compute_ci: bool = True
) -> Dict[str, Dict]:
    """
    Run all 6 BUT-PPG tasks and compile benchmark report

    Args:
        quality_model: Model for quality classification
        hr_model: Model for HR estimation (optional, can be same as quality_model)
        motion_model: Model for motion classification (optional)
        quality_loader: DataLoader for quality task
        hr_loader: DataLoader for HR task (optional)
        motion_loader: DataLoader for motion task (optional)
        bp_systolic_model: Model for systolic BP estimation (optional)
        bp_diastolic_model: Model for diastolic BP estimation (optional)
        spo2_model: Model for SpO2 estimation (optional)
        glycaemia_model: Model for glycaemia estimation (optional)
        bp_systolic_loader: DataLoader for systolic BP task (optional)
        bp_diastolic_loader: DataLoader for diastolic BP task (optional)
        spo2_loader: DataLoader for SpO2 task (optional)
        glycaemia_loader: DataLoader for glycaemia task (optional)
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

    # Task 4: Blood Pressure Estimation - Systolic (optional)
    if bp_systolic_loader is not None and bp_systolic_model is not None:
        print("\n" + "="*80)
        print("TASK 4: Systolic Blood Pressure Estimation")
        print("="*80)

        bp_sys_estimator = BPEstimator(bp_systolic_model, device, bp_type='systolic')
        bp_sys_preds, bp_sys_labels, bp_sys_subjects = bp_sys_estimator.predict(bp_systolic_loader)
        bp_sys_metrics = bp_sys_estimator.evaluate(bp_sys_preds, bp_sys_labels, bp_sys_subjects, compute_ci)

        results['bp_systolic'] = bp_sys_metrics

        print(f"MAE: {bp_sys_metrics['mae']:.2f} mmHg (Target: {bp_sys_metrics['target']:.2f})")
        if compute_ci:
            print(f"  95% CI: [{bp_sys_metrics['mae_ci_lower']:.2f}, {bp_sys_metrics['mae_ci_upper']:.2f}]")
        print(f"RMSE: {bp_sys_metrics['rmse']:.2f} mmHg")
        print(f"ME (bias): {bp_sys_metrics['me']:.2f} mmHg")
        print(f"STD: {bp_sys_metrics['std']:.2f} mmHg")
        print(f"Correlation: {bp_sys_metrics['correlation']:.3f}")
        print(f"Samples: {bp_sys_metrics['n_samples']}")
        print(f"AAMI Compliant: {'✅ YES' if bp_sys_metrics['aami_compliant'] else '❌ NO'} (MAE ≤ {bp_sys_metrics['aami_standard']} mmHg, STD ≤ {bp_sys_metrics['aami_std_standard']} mmHg)")
        print(f"Target Met: {'✅ YES' if bp_sys_metrics['target_met'] else '❌ NO'}")
        print(f"vs Baseline: +{bp_sys_metrics['vs_baseline']:.2f} mmHg")

    # Task 5: Blood Pressure Estimation - Diastolic (optional)
    if bp_diastolic_loader is not None and bp_diastolic_model is not None:
        print("\n" + "="*80)
        print("TASK 5: Diastolic Blood Pressure Estimation")
        print("="*80)

        bp_dia_estimator = BPEstimator(bp_diastolic_model, device, bp_type='diastolic')
        bp_dia_preds, bp_dia_labels, bp_dia_subjects = bp_dia_estimator.predict(bp_diastolic_loader)
        bp_dia_metrics = bp_dia_estimator.evaluate(bp_dia_preds, bp_dia_labels, bp_dia_subjects, compute_ci)

        results['bp_diastolic'] = bp_dia_metrics

        print(f"MAE: {bp_dia_metrics['mae']:.2f} mmHg (Target: {bp_dia_metrics['target']:.2f})")
        if compute_ci:
            print(f"  95% CI: [{bp_dia_metrics['mae_ci_lower']:.2f}, {bp_dia_metrics['mae_ci_upper']:.2f}]")
        print(f"RMSE: {bp_dia_metrics['rmse']:.2f} mmHg")
        print(f"ME (bias): {bp_dia_metrics['me']:.2f} mmHg")
        print(f"STD: {bp_dia_metrics['std']:.2f} mmHg")
        print(f"Correlation: {bp_dia_metrics['correlation']:.3f}")
        print(f"Samples: {bp_dia_metrics['n_samples']}")
        print(f"AAMI Compliant: {'✅ YES' if bp_dia_metrics['aami_compliant'] else '❌ NO'} (MAE ≤ {bp_dia_metrics['aami_standard']} mmHg, STD ≤ {bp_dia_metrics['aami_std_standard']} mmHg)")
        print(f"Target Met: {'✅ YES' if bp_dia_metrics['target_met'] else '❌ NO'}")
        print(f"vs Baseline: +{bp_dia_metrics['vs_baseline']:.2f} mmHg")

    # Task 6: SpO2 Estimation (optional)
    if spo2_loader is not None and spo2_model is not None:
        print("\n" + "="*80)
        print("TASK 6: SpO2 Estimation")
        print("="*80)

        spo2_estimator = SpO2Estimator(spo2_model, device)
        spo2_preds, spo2_labels, spo2_subjects = spo2_estimator.predict(spo2_loader)
        spo2_metrics = spo2_estimator.evaluate(spo2_preds, spo2_labels, spo2_subjects, compute_ci)

        results['spo2'] = spo2_metrics

        print(f"MAE: {spo2_metrics['mae']:.2f}% (Target: {spo2_metrics['target']:.2f}%)")
        if compute_ci:
            print(f"  95% CI: [{spo2_metrics['mae_ci_lower']:.2f}, {spo2_metrics['mae_ci_upper']:.2f}]")
        print(f"RMSE: {spo2_metrics['rmse']:.2f}%")
        print(f"Correlation: {spo2_metrics['correlation']:.3f}")
        print(f"Within 2%: {spo2_metrics['within_2pct']:.1f}%")
        print(f"Within 3%: {spo2_metrics['within_3pct']:.1f}%")
        print(f"Samples: {spo2_metrics['n_samples']}")
        print(f"Clinical Compliant: {'✅ YES' if spo2_metrics['clinical_compliant'] else '❌ NO'} (±{spo2_metrics['clinical_standard']}%)")
        print(f"Target Met: {'✅ YES' if spo2_metrics['target_met'] else '❌ NO'}")
        print(f"vs Baseline: +{spo2_metrics['vs_baseline']:.2f}%")

    # Task 7: Glycaemia Estimation (optional)
    if glycaemia_loader is not None and glycaemia_model is not None:
        print("\n" + "="*80)
        print("TASK 7: Glycaemia Estimation")
        print("="*80)

        glyc_estimator = GlycaemiaEstimator(glycaemia_model, device)
        glyc_preds, glyc_labels, glyc_subjects = glyc_estimator.predict(glycaemia_loader)
        glyc_metrics = glyc_estimator.evaluate(glyc_preds, glyc_labels, glyc_subjects, compute_ci)

        results['glycaemia'] = glyc_metrics

        print(f"MAE: {glyc_metrics['mae']:.2f} mmol/l (Target: {glyc_metrics['target']:.2f})")
        if compute_ci:
            print(f"  95% CI: [{glyc_metrics['mae_ci_lower']:.2f}, {glyc_metrics['mae_ci_upper']:.2f}]")
        print(f"RMSE: {glyc_metrics['rmse']:.2f} mmol/l")
        print(f"MAPE: {glyc_metrics['mape']:.1f}%")
        print(f"Correlation: {glyc_metrics['correlation']:.3f}")
        print(f"Samples: {glyc_metrics['n_samples']}")
        print(f"Target Met: {'✅ YES' if glyc_metrics['target_met'] else '❌ NO'}")
        print(f"vs Baseline: +{glyc_metrics['vs_baseline']:.2f} mmol/l")

    return results
