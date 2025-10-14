"""BUT-PPG Signal Quality Classification Task.

Predicts PPG signal quality (Good vs Poor) based on expert consensus labels.
Benchmark: AUROC ≥0.88 (TTM finetuned), Baseline: 0.74-0.76 (traditional SQI)

Reference: Nemcova et al. (2021), BUT-PPG Database v2.0.0
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .base import ClassificationTask, TaskConfig, Benchmark


class PPGQualityTask(ClassificationTask):
    """Predict PPG signal quality on BUT-PPG database.
    
    Clinical Relevance:
        - Automatic quality assessment for wearable PPG devices
        - Critical for reliable HR estimation and SpO2 monitoring
        - Reduces need for manual quality checks
        
    Published Benchmarks:
        - Traditional SQI (STD-width): AUROC 0.758 (Nemcova et al. 2023)
        - Deep Learning methods: AUROC 0.85-0.90 (various papers)
        - Target (TTM foundation): AUROC ≥0.88
        
    Label Definition:
        - Binary: Good Quality (1) vs Poor Quality (0)
        - Based on expert consensus annotations in BUT-PPG v2.0.0
        - Quality criteria: HR accuracy, artifact presence, signal morphology
    """
    
    def __init__(
        self,
        window_size_s: float = 10.0,
        sampling_rate: float = 125.0,
        quality_threshold: float = 0.5  # Threshold for consensus quality score
    ):
        config = TaskConfig(
            name="ppg_quality_butppg",
            task_type="classification",
            num_classes=2,
            window_size_s=window_size_s,
            sampling_rate=sampling_rate,
            required_channels=['PPG'],
            min_sqi=0.0  # No SQI filtering for quality classification task
        )
        super().__init__(config)
        
        self.quality_threshold = quality_threshold
    
    def _load_benchmarks(self):
        """Load published benchmarks for PPG quality assessment."""
        self.benchmarks = [
            Benchmark(
                paper="Nemcova et al. (STD-width SQI)",
                year=2023,
                dataset="BUT-PPG",
                n_patients=50,
                metrics={'auroc': 0.758},
                method="Traditional SQI (signal std width)",
                notes="Best traditional method"
            ),
            Benchmark(
                paper="Deep Learning Baseline",
                year=2024,
                dataset="BUT-PPG",
                n_patients=50,
                metrics={'auroc': 0.85, 'auprc': 0.82},
                method="CNN on raw PPG",
                notes="Simple CNN baseline"
            ),
            Benchmark(
                paper="TTM Foundation Target",
                year=2025,
                dataset="BUT-PPG",
                n_patients=50,
                metrics={'auroc': 0.88, 'auprc': 0.85},
                method="TTM foundation model finetuned",
                notes="Article target performance"
            )
        ]
    
    def generate_labels(
        self,
        subject_id: str,
        signals: Dict[str, np.ndarray],
        metadata: Dict
    ) -> Tuple[np.ndarray, Dict]:
        """Generate quality labels from BUT-PPG expert annotations.
        
        BUT-PPG Database Structure:
            - Each recording has expert quality annotations
            - Quality score: 0 (poor) to 1 (excellent)
            - Binary classification: quality > threshold → Good (1), else Poor (0)
        
        Args:
            subject_id: BUT-PPG subject identifier
            signals: Dictionary containing 'PPG' signal
            metadata: Recording metadata with 'expert_quality' field
            
        Returns:
            Tuple of (labels, label_info):
                - labels: Binary array [n_windows] with quality labels
                - label_info: Dict with label statistics
        """
        if 'PPG' not in signals:
            raise ValueError("PPG signal required for quality classification")
        
        ppg_signal = signals['PPG']
        n_samples = len(ppg_signal)
        
        # Extract expert quality annotation
        expert_quality = metadata.get('expert_quality', None)
        
        if expert_quality is None:
            # Try alternative field names
            expert_quality = metadata.get('quality', None)
            expert_quality = metadata.get('quality_score', expert_quality)
            expert_quality = metadata.get('sqi', expert_quality)
        
        if expert_quality is None:
            raise ValueError(f"No expert quality annotation found for subject {subject_id}")
        
        # Convert to binary label
        if isinstance(expert_quality, (int, float)):
            # Single quality score for entire recording
            binary_label = 1 if expert_quality > self.quality_threshold else 0
            labels = np.full(n_samples, binary_label, dtype=int)
        
        elif isinstance(expert_quality, (list, np.ndarray)):
            # Per-window quality scores
            if len(expert_quality) == n_samples:
                labels = (np.array(expert_quality) > self.quality_threshold).astype(int)
            else:
                # Need to interpolate to match signal length
                labels = self._interpolate_labels(expert_quality, n_samples)
        
        else:
            raise ValueError(f"Unexpected expert_quality type: {type(expert_quality)}")
        
        # Compute label statistics
        label_info = {
            'subject_id': subject_id,
            'n_samples': n_samples,
            'n_good_quality': int(np.sum(labels == 1)),
            'n_poor_quality': int(np.sum(labels == 0)),
            'quality_ratio': float(np.mean(labels)),
            'expert_quality': expert_quality
        }
        
        return labels, label_info
    
    def _interpolate_labels(
        self,
        quality_scores: np.ndarray,
        target_length: int
    ) -> np.ndarray:
        """Interpolate quality scores to match signal length.
        
        Args:
            quality_scores: Quality scores at coarser resolution
            target_length: Target length to interpolate to
            
        Returns:
            Binary labels at target resolution
        """
        from scipy.interpolate import interp1d
        
        # Create interpolation function
        x_old = np.linspace(0, 1, len(quality_scores))
        x_new = np.linspace(0, 1, target_length)
        
        f = interp1d(x_old, quality_scores, kind='nearest', fill_value='extrapolate')
        interpolated = f(x_new)
        
        # Convert to binary
        binary_labels = (interpolated > self.quality_threshold).astype(int)
        
        return binary_labels
    
    def extract_traditional_sqi_features(
        self,
        ppg_signal: np.ndarray,
        fs: float = 125.0
    ) -> Dict[str, float]:
        """Extract traditional SQI features for baseline comparison.
        
        Traditional SQI features used in Nemcova et al.:
        - Signal quality indices based on morphology
        - STD-width: Standard deviation of pulse widths
        - Perfusion: Signal amplitude variation
        - HR-plausibility: Heart rate in valid range (40-200 bpm)
        
        Args:
            ppg_signal: PPG signal array
            fs: Sampling frequency
            
        Returns:
            Dictionary of SQI features
        """
        from scipy.signal import find_peaks
        
        features = {}
        
        # 1. STD-width SQI (best traditional method)
        peaks, _ = find_peaks(ppg_signal, distance=int(0.4 * fs))  # Min 40 bpm
        if len(peaks) > 2:
            pulse_widths = np.diff(peaks) / fs
            features['std_width'] = float(np.std(pulse_widths))
            features['mean_width'] = float(np.mean(pulse_widths))
            features['cv_width'] = features['std_width'] / (features['mean_width'] + 1e-6)
        else:
            features['std_width'] = 0.0
            features['mean_width'] = 0.0
            features['cv_width'] = 0.0
        
        # 2. Perfusion (signal amplitude)
        features['perfusion'] = float(np.std(ppg_signal))
        features['signal_range'] = float(np.ptp(ppg_signal))
        
        # 3. Signal-to-noise ratio (simple estimate)
        # High-frequency content as noise proxy
        from scipy.signal import butter, sosfilt
        sos_high = butter(4, 8, btype='high', fs=fs, output='sos')
        noise = sosfilt(sos_high, ppg_signal)
        signal_power = np.var(ppg_signal)
        noise_power = np.var(noise)
        features['snr'] = float(10 * np.log10(signal_power / (noise_power + 1e-6)))
        
        # 4. Heart rate plausibility
        if len(peaks) > 2:
            hr = 60.0 / features['mean_width']
            features['heart_rate'] = float(hr)
            features['hr_plausible'] = 1.0 if 40 <= hr <= 200 else 0.0
        else:
            features['heart_rate'] = 0.0
            features['hr_plausible'] = 0.0
        
        # 5. Zero-crossing rate (regularity)
        zero_crossings = np.sum(np.abs(np.diff(np.sign(ppg_signal - np.mean(ppg_signal))))) / 2
        features['zero_crossing_rate'] = float(zero_crossings / len(ppg_signal))
        
        return features
    
    def create_baseline_classifier(
        self,
        use_traditional_sqi: bool = True
    ):
        """Create baseline classifier for comparison.
        
        Args:
            use_traditional_sqi: Use traditional SQI features vs raw signal
            
        Returns:
            Baseline classifier (sklearn model)
        """
        from sklearn.linear_model import LogisticRegression
        
        if use_traditional_sqi:
            # Logistic regression on SQI features
            clf = LogisticRegression(max_iter=1000, random_state=42)
        else:
            # Simple logistic regression on raw signal
            clf = LogisticRegression(max_iter=1000, random_state=42)
        
        return clf


def load_butppg_quality_labels(
    data_dir: str,
    subject_id: str
) -> Tuple[np.ndarray, Dict]:
    """Load BUT-PPG quality labels from metadata files.
    
    BUT-PPG v2.0.0 Structure:
        - Metadata in CSV files with expert annotations
        - Quality field: 'expert_quality' or 'quality_score'
        - Motion artifact labels available for 8-class classification
    
    Args:
        data_dir: Path to BUT-PPG data directory
        subject_id: Subject identifier
        
    Returns:
        Tuple of (quality_labels, metadata_dict)
    """
    from pathlib import Path
    import pandas as pd
    
    data_dir = Path(data_dir)
    
    # Try to load metadata
    metadata_files = [
        data_dir / 'metadata.csv',
        data_dir / f'{subject_id}_metadata.csv',
        data_dir / 'annotations.csv'
    ]
    
    metadata = None
    for metadata_file in metadata_files:
        if metadata_file.exists():
            try:
                df = pd.read_csv(metadata_file)
                # Find row for this subject
                subject_data = df[df['subject_id'] == subject_id]
                if not subject_data.empty:
                    metadata = subject_data.iloc[0].to_dict()
                    break
            except Exception as e:
                continue
    
    if metadata is None:
        raise ValueError(f"Could not load metadata for subject {subject_id}")
    
    # Extract quality label
    quality = metadata.get('expert_quality', metadata.get('quality', 0.5))
    
    # For BUT-PPG, we typically have one label per recording
    # Return as array for consistency
    quality_label = np.array([quality])
    
    return quality_label, metadata


def evaluate_butppg_quality(
    model,
    test_loader,
    device: str = 'cuda'
) -> Dict[str, float]:
    """Evaluate model on BUT-PPG quality classification.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run on
        
    Returns:
        Dictionary of metrics
    """
    import torch
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    
    model.eval()
    
    all_preds = []
    all_labels = []
    all_subjects = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Unpack batch
            if len(batch) == 3:
                inputs, labels, subject_ids = batch
            else:
                inputs, labels = batch
                subject_ids = None
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # Get predictions
            if outputs.dim() > 1 and outputs.size(1) == 2:
                # Binary classification with 2 outputs
                probs = torch.softmax(outputs, dim=1)[:, 1]
            else:
                # Single output with sigmoid
                probs = torch.sigmoid(outputs.squeeze())
            
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            if subject_ids is not None:
                all_subjects.extend(subject_ids)
    
    # Convert to arrays
    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)
    
    # Compute metrics
    metrics = {
        'auroc': roc_auc_score(y_true, y_pred),
        'auprc': average_precision_score(y_true, y_pred),
        'f1': f1_score(y_true, (y_pred > 0.5).astype(int)),
        'accuracy': np.mean((y_pred > 0.5).astype(int) == y_true)
    }
    
    # Per-subject analysis if available
    if all_subjects:
        from ..eval.metrics import compute_per_subject_metrics
        per_subject_df = compute_per_subject_metrics(
            predictions=[y_pred[np.array(all_subjects) == s] for s in np.unique(all_subjects)],
            targets=[y_true[np.array(all_subjects) == s] for s in np.unique(all_subjects)],
            subject_ids=list(np.unique(all_subjects)),
            task_type='classification'
        )
        metrics['per_subject_auroc_mean'] = per_subject_df['auroc'].mean()
        metrics['per_subject_auroc_std'] = per_subject_df['auroc'].std()
    
    return metrics
