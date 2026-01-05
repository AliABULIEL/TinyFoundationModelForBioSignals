"""Unit tests for preprocessing module."""

import pytest
import numpy as np

from src.preprocessing.resampling import resample_signal
from src.preprocessing.windowing import create_windows
from src.preprocessing.normalization import normalize_window, RevIN
from src.preprocessing.gravity import remove_gravity
from src.preprocessing.pipeline import PreprocessingPipeline


@pytest.mark.unit
class TestResampling:
    """Tests for signal resampling."""

    def test_resample_signal_upsampling(self, sample_signal):
        """Test upsampling from 100Hz to 200Hz."""
        original_rate = 100
        target_rate = 200

        resampled = resample_signal(sample_signal, original_rate, target_rate)

        # Should have twice as many samples
        expected_length = len(sample_signal) * 2
        assert len(resampled) == expected_length
        assert resampled.shape[1] == 3  # Same number of channels

    def test_resample_signal_downsampling(self, sample_signal):
        """Test downsampling from 100Hz to 30Hz."""
        original_rate = 100
        target_rate = 30

        resampled = resample_signal(sample_signal, original_rate, target_rate)

        # Should have 30% of original samples
        expected_length = int(len(sample_signal) * target_rate / original_rate)
        assert abs(len(resampled) - expected_length) <= 1  # Allow small rounding
        assert resampled.shape[1] == 3

    def test_resample_signal_same_rate(self, sample_signal):
        """Test resampling with same rate (no change)."""
        rate = 100

        resampled = resample_signal(sample_signal, rate, rate)

        # Should be unchanged
        assert len(resampled) == len(sample_signal)
        np.testing.assert_array_almost_equal(resampled, sample_signal)

    def test_resample_preserves_energy(self, sample_signal):
        """Test that resampling preserves signal energy (Parseval's theorem)."""
        original_rate = 100
        target_rate = 30

        resampled = resample_signal(sample_signal, original_rate, target_rate)

        # Energy should be approximately preserved
        original_energy = np.sum(sample_signal ** 2)
        resampled_energy = np.sum(resampled ** 2)

        # Allow some tolerance due to resampling
        ratio = resampled_energy / original_energy
        assert 0.9 < ratio < 1.1


@pytest.mark.unit
class TestWindowing:
    """Tests for windowing."""

    def test_create_windows_basic(self):
        """Test basic windowing."""
        signal = np.arange(1000).reshape(-1, 1)
        labels = np.arange(1000)

        window_length = 100
        stride = 50

        windows, window_labels = create_windows(
            signal, labels, window_length, stride, label_strategy="center"
        )

        # Expected number of windows
        expected_n_windows = (len(signal) - window_length) // stride + 1
        assert len(windows) == expected_n_windows
        assert len(window_labels) == expected_n_windows
        assert windows.shape[1] == window_length
        assert windows.shape[2] == 1

    def test_create_windows_label_strategies(self, sample_signal, sample_labels):
        """Test different label strategies."""
        window_length = 300
        stride = 100

        strategies = ["center", "mode", "first", "last"]

        for strategy in strategies:
            windows, window_labels = create_windows(
                sample_signal, sample_labels, window_length, stride, label_strategy=strategy
            )

            assert len(windows) == len(window_labels)
            assert all(0 <= label < 5 for label in window_labels)

    def test_create_windows_no_overlap(self, sample_signal, sample_labels):
        """Test windowing with no overlap (stride = window_length)."""
        window_length = 300
        stride = 300

        windows, window_labels = create_windows(
            sample_signal, sample_labels, window_length, stride
        )

        # No overlap, so windows should be disjoint
        n_windows = len(sample_signal) // window_length
        assert len(windows) == n_windows


@pytest.mark.unit
class TestNormalization:
    """Tests for normalization."""

    def test_zscore_normalization(self):
        """Test z-score normalization."""
        data = np.random.randn(100, 3) * 10 + 5  # Mean ~5, std ~10

        normalized = normalize_window(data, method="zscore")

        # Should have mean ~0, std ~1
        assert np.abs(np.mean(normalized)) < 0.1
        assert np.abs(np.std(normalized) - 1.0) < 0.1

    def test_minmax_normalization(self):
        """Test min-max normalization."""
        data = np.random.randn(100, 3) * 10 + 5

        normalized = normalize_window(data, method="minmax")

        # Should be in [0, 1]
        assert np.min(normalized) >= 0
        assert np.max(normalized) <= 1

    def test_robust_normalization(self):
        """Test robust normalization."""
        data = np.random.randn(100, 3) * 10 + 5

        normalized = normalize_window(data, method="robust")

        # Should be roughly centered
        median = np.median(normalized)
        assert np.abs(median) < 1.0

    def test_revin_forward_backward(self):
        """Test RevIN forward and backward pass."""
        import torch

        revin = RevIN(num_features=3)
        data = torch.randn(4, 100, 3)

        # Forward pass
        normalized = revin(data, mode="norm")

        # Should have mean ~0, std ~1 per channel
        for c in range(3):
            channel_data = normalized[:, :, c]
            assert torch.abs(channel_data.mean()) < 0.1
            assert torch.abs(channel_data.std() - 1.0) < 0.2

        # Backward pass
        denormalized = revin(normalized, mode="denorm")

        # Should recover original
        torch.testing.assert_close(denormalized, data, rtol=1e-4, atol=1e-4)


@pytest.mark.unit
class TestGravityRemoval:
    """Tests for gravity removal."""

    def test_remove_gravity(self):
        """Test gravity removal via high-pass filter."""
        # Create signal with low-frequency component (gravity)
        t = np.linspace(0, 10, 1000)
        gravity = np.sin(0.5 * t)  # Low frequency
        body_motion = np.sin(10 * t)  # High frequency

        signal = (gravity + body_motion).reshape(-1, 1)

        # Remove gravity
        filtered = remove_gravity(signal, sample_rate=100, cutoff_freq=0.5)

        # Low frequency should be attenuated
        assert filtered.shape == signal.shape
        # Body motion should dominate
        assert np.std(filtered) > 0

    def test_remove_gravity_multichannel(self, sample_signal):
        """Test gravity removal on multi-channel signal."""
        filtered = remove_gravity(sample_signal, sample_rate=100)

        assert filtered.shape == sample_signal.shape
        # Filtered signal should have similar magnitude
        assert 0.5 < np.std(filtered) / np.std(sample_signal) < 1.5


@pytest.mark.unit
class TestPreprocessingPipeline:
    """Tests for complete preprocessing pipeline."""

    def test_pipeline_initialization(self):
        """Test pipeline initialization with different configs."""
        pipeline = PreprocessingPipeline(
            target_sample_rate=30,
            window_size_sec=10,
            stride_sec=5,
            remove_gravity=True,
            normalization="zscore",
        )

        config = pipeline.get_config()
        assert config["target_sample_rate"] == 30
        assert config["window_size_sec"] == 10
        assert config["normalization"] == "zscore"

    def test_pipeline_process_participant(self, sample_signal, sample_labels):
        """Test full participant processing."""
        pipeline = PreprocessingPipeline(
            target_sample_rate=30,
            window_size_sec=10,
            stride_sec=5,
            remove_gravity=False,
            normalization="zscore",
        )

        windows, window_labels = pipeline.process_participant(
            sample_signal, sample_labels, original_sample_rate=100
        )

        # Should have windows
        assert len(windows) > 0
        assert len(windows) == len(window_labels)

        # Windows should be normalized
        for window in windows:
            assert np.abs(np.mean(window)) < 1.0

    def test_pipeline_with_gravity_removal(self, sample_signal, sample_labels):
        """Test pipeline with gravity removal enabled."""
        pipeline = PreprocessingPipeline(
            target_sample_rate=30,
            window_size_sec=10,
            stride_sec=5,
            remove_gravity=True,
            normalization="zscore",
        )

        windows, window_labels = pipeline.process_participant(
            sample_signal, sample_labels, original_sample_rate=100
        )

        assert len(windows) > 0
        assert windows.shape[2] == 3  # 3 channels preserved
