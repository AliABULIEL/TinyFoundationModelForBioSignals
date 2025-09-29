"""Unit tests for VitalDB downstream tasks.

Tests task implementations, label generation, evaluation metrics,
and benchmark comparison.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tasks import (
    get_task, list_tasks,
    HypotensionPredictionTask,
    BloodPressureEstimationTask
)
from src.tasks.base import TaskConfig, Benchmark, ClassificationTask, RegressionTask


class TestTaskRegistry:
    """Test task registry and discovery."""
    
    def test_list_tasks(self):
        """Test listing all available tasks."""
        tasks = list_tasks()
        assert len(tasks) > 0
        assert 'hypotension_5min' in tasks
        assert 'blood_pressure_both' in tasks
    
    def test_get_task(self):
        """Test retrieving task by name."""
        task = get_task('hypotension_5min')
        assert isinstance(task, HypotensionPredictionTask)
        assert task.config.name == 'hypotension_prediction_5min'
    
    def test_get_invalid_task(self):
        """Test error handling for invalid task name."""
        with pytest.raises(ValueError, match="Task .* not found"):
            get_task('nonexistent_task')


class TestTaskConfig:
    """Test task configuration."""
    
    def test_task_config_creation(self):
        """Test creating task configuration."""
        config = TaskConfig(
            name='test_task',
            task_type='classification',
            num_classes=2,
            required_channels=['ECG', 'PPG']
        )
        
        assert config.name == 'test_task'
        assert config.task_type == 'classification'
        assert config.num_classes == 2
        assert 'ECG' in config.required_channels


class TestHypotensionTask:
    """Test hypotension prediction task."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.task = HypotensionPredictionTask(prediction_window_min=5)
    
    def test_task_initialization(self):
        """Test task initialization."""
        assert self.task.config.task_type == 'classification'
        assert self.task.config.num_classes == 2
        assert self.task.prediction_window_min == 5
        assert self.task.map_threshold == 65.0
    
    def test_benchmarks_loaded(self):
        """Test that benchmarks are loaded."""
        assert len(self.task.benchmarks) > 0
        
        # Check for known benchmarks
        papers = [b.paper for b in self.task.benchmarks]
        assert any('STEP-OP' in p for p in papers)
        assert any('Jo et al' in p for p in papers)
    
    def test_find_hypotension_episodes(self):
        """Test hypotension episode detection."""
        # Create synthetic MAP signal with hypotension
        fs = 0.5  # 2-second intervals
        duration_s = 600  # 10 minutes
        n_samples = int(duration_s * fs)
        
        # Normal MAP ~80 mmHg, with hypotension episode 60-62 mmHg
        map_signal = np.full(n_samples, 80.0)
        
        # Create hypotension episode: 120s at 60 mmHg
        hypotension_start = 150
        hypotension_end = 210
        map_signal[hypotension_start:hypotension_end] = 60.0
        
        # Find episodes
        hypotensive_mask = map_signal < 65.0
        min_duration = int(60 * fs)  # 60 seconds
        
        episodes = self.task._find_hypotension_episodes(
            hypotensive_mask, min_duration, fs
        )
        
        assert len(episodes) == 1
        assert episodes[0]['duration_s'] >= 60
        assert episodes[0]['start_idx'] == hypotension_start
    
    def test_create_prediction_labels(self):
        """Test prediction label creation."""
        fs = 0.5
        n_samples = 600  # 20 minutes
        
        map_signal = np.full(n_samples, 80.0)
        
        # Create episode at 10 minutes
        episode_start = 300
        episode_end = 360
        map_signal[episode_start:episode_end] = 60.0
        
        episodes = [{
            'start_idx': episode_start,
            'end_idx': episode_end,
            'duration_s': 120.0
        }]
        
        # 5-minute prediction window
        prediction_samples = int(5 * 60 * fs)
        
        labels, valid_mask = self.task._create_prediction_labels(
            map_signal, episodes, prediction_samples, fs
        )
        
        # Check that prediction window is labeled positive
        window_start = episode_start - prediction_samples
        window_end = episode_start
        assert np.all(labels[window_start:window_end] == 1)
        
        # Check that normal periods are labeled negative
        assert np.all(labels[:window_start] == 0)
    
    def test_extract_clinical_features(self):
        """Test clinical feature extraction."""
        # Create synthetic MAP signal
        fs = 0.5
        duration_s = 300  # 5 minutes
        n_samples = int(duration_s * fs)
        
        # Declining MAP trend
        map_signal = np.linspace(85, 70, n_samples)
        
        features = self.task.extract_clinical_features(map_signal, duration_s, fs)
        
        assert 'map_mean' in features
        assert 'map_std' in features
        assert 'map_trend' in features
        assert 'map_min' in features
        
        # Check that trend is negative (declining)
        assert features['map_trend'] < 0
    
    def test_episode_statistics(self):
        """Test episode statistics computation."""
        episodes = [
            {'duration_s': 90.0},
            {'duration_s': 120.0},
            {'duration_s': 60.0}
        ]
        
        stats = self.task.compute_episode_statistics(episodes)
        
        assert stats['n_episodes'] == 3
        assert stats['mean_duration_s'] == 90.0
        assert stats['max_duration_s'] == 120.0
        assert stats['min_duration_s'] == 60.0


class TestBloodPressureTask:
    """Test blood pressure estimation task."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.task = BloodPressureEstimationTask(target='both')
    
    def test_task_initialization(self):
        """Test task initialization."""
        assert self.task.config.task_type == 'regression'
        assert self.task.target == 'both'
        assert self.task.ppg_abp_corr_threshold == 0.9
    
    def test_benchmarks_loaded(self):
        """Test benchmarks are loaded."""
        assert len(self.task.benchmarks) > 0
        
        papers = [b.paper for b in self.task.benchmarks]
        assert any('Pan et al' in p for p in papers)
        assert any('AAMI' in p for p in papers)
    
    def test_detect_systolic_peaks(self):
        """Test systolic peak detection."""
        # Create synthetic ABP waveform
        fs = 125.0
        duration_s = 10.0
        t = np.arange(0, duration_s, 1/fs)
        
        # Simulate cardiac cycles at 60 bpm (1 Hz)
        heart_rate = 1.0  # Hz
        
        # ABP waveform: systolic peaks ~120 mmHg, diastolic ~80 mmHg
        abp = 100 + 20 * np.sin(2 * np.pi * heart_rate * t)
        abp += 10 * np.sin(4 * np.pi * heart_rate * t)  # Add harmonics
        
        peaks = self.task._detect_systolic_peaks(abp, fs)
        
        # Should detect ~10 peaks (60 bpm for 10 seconds)
        assert len(peaks) >= 8
        assert len(peaks) <= 12
    
    def test_detect_diastolic_troughs(self):
        """Test diastolic trough detection."""
        fs = 125.0
        duration_s = 10.0
        t = np.arange(0, duration_s, 1/fs)
        
        heart_rate = 1.0
        abp = 100 + 20 * np.sin(2 * np.pi * heart_rate * t)
        
        troughs = self.task._detect_diastolic_troughs(abp, fs)
        
        assert len(troughs) >= 8
        assert len(troughs) <= 12
    
    def test_ppg_abp_correlation(self):
        """Test PPG-ABP correlation computation."""
        # Create correlated signals
        fs = 125.0
        duration_s = 10.0
        t = np.arange(0, duration_s, 1/fs)
        
        heart_rate = 1.0
        ppg = np.sin(2 * np.pi * heart_rate * t)
        
        # ABP with slight delay (pulse transit time)
        delay_s = 0.1
        abp = np.sin(2 * np.pi * heart_rate * (t - delay_s))
        
        corr = self.task._compute_ppg_abp_correlation(ppg, abp, fs)
        
        # Should be highly correlated
        assert corr > 0.85
    
    def test_bhs_grade_computation(self):
        """Test BHS grade computation."""
        # Create errors that meet Grade A criteria
        errors_grade_a = np.concatenate([
            np.random.uniform(-5, 5, 600),   # 60% within 5 mmHg
            np.random.uniform(-10, 10, 250),  # 25% within 10 mmHg
            np.random.uniform(-15, 15, 100),  # 10% within 15 mmHg
            np.random.uniform(-20, 20, 50)   # 5% beyond 15 mmHg
        ])
        
        grade = self.task._compute_bhs_grade(errors_grade_a)
        assert grade in ['A', 'B']  # Should be A or B
        
        # Create errors that fail all grades
        errors_grade_d = np.random.uniform(-30, 30, 1000)
        grade_d = self.task._compute_bhs_grade(errors_grade_d)
        assert grade_d in ['C', 'D']
    
    def test_aami_compliance(self):
        """Test AAMI compliance checking."""
        # Create synthetic predictions and targets
        n_samples = 100
        
        # Meet AAMI: ME ≤ 5, SD ≤ 8
        targets = np.random.uniform(100, 140, (n_samples, 2))
        predictions = targets + np.random.normal(2, 5, (n_samples, 2))
        
        metrics = self.task.evaluate(predictions, targets)
        
        assert 'sbp_me' in metrics
        assert 'sbp_sd' in metrics
        assert 'sbp_aami_compliant' in metrics
        
        # Check individual compliance
        if abs(metrics['sbp_me']) <= 5 and metrics['sbp_sd'] <= 8:
            assert metrics['sbp_aami_compliant'] == True


class TestClassificationEvaluation:
    """Test classification evaluation metrics."""
    
    def test_binary_classification_metrics(self):
        """Test binary classification evaluation."""
        # Create dummy classification task
        config = TaskConfig(
            name='test_clf',
            task_type='classification',
            num_classes=2
        )
        
        class DummyTask(ClassificationTask):
            def generate_labels(self, *args, **kwargs):
                pass
            def _load_benchmarks(self):
                pass
        
        task = DummyTask(config)
        
        # Create predictions and targets
        n_samples = 100
        targets = np.random.randint(0, 2, n_samples)
        
        # Logits (pre-sigmoid)
        logits = np.random.randn(n_samples, 1)
        
        metrics = task.evaluate(logits, targets)
        
        assert 'accuracy' in metrics
        assert 'auroc' in metrics
        assert 'auprc' in metrics
        assert 'f1' in metrics
        
        # Metrics should be in valid range
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['auroc'] <= 1
        assert 0 <= metrics['f1'] <= 1


class TestRegressionEvaluation:
    """Test regression evaluation metrics."""
    
    def test_regression_metrics(self):
        """Test regression evaluation."""
        config = TaskConfig(
            name='test_reg',
            task_type='regression',
            target_dim=1
        )
        
        class DummyTask(RegressionTask):
            def generate_labels(self, *args, **kwargs):
                pass
            def _load_benchmarks(self):
                pass
        
        task = DummyTask(config)
        
        # Create predictions and targets
        n_samples = 100
        targets = np.random.uniform(60, 120, n_samples)
        predictions = targets + np.random.normal(0, 5, n_samples)
        
        metrics = task.evaluate(predictions, targets)
        
        assert 'mae' in metrics
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
        assert 'pearson_r' in metrics
        assert 'mean_error' in metrics
        assert 'std_error' in metrics
        
        # MAE should be reasonable
        assert metrics['mae'] >= 0
        assert metrics['mae'] < 20  # Should be within 20 units


class TestBenchmarkComparison:
    """Test benchmark comparison functionality."""
    
    def test_benchmark_creation(self):
        """Test benchmark object creation."""
        benchmark = Benchmark(
            paper="Test Paper",
            year=2024,
            dataset="VitalDB",
            n_patients=100,
            metrics={'auroc': 0.90, 'auprc': 0.85},
            method="Test method",
            notes="Test notes"
        )
        
        assert benchmark.paper == "Test Paper"
        assert benchmark.year == 2024
        assert benchmark.metrics['auroc'] == 0.90
    
    def test_compare_to_benchmarks(self):
        """Test comparing results to benchmarks."""
        task = get_task('hypotension_5min')
        
        results = {
            'auroc': 0.895,
            'auprc': 0.742,
            'n_patients': 100
        }
        
        comparison = task.compare_to_benchmarks(results)
        
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) > 0
        assert 'Paper' in comparison.columns
        assert 'auroc' in comparison.columns or 'AUROC' in comparison.columns


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v'])
