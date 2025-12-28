"""Tests for signal synchronization and resampling."""

import numpy as np
import pytest
from scipy import signal as scipy_signal

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.sync import (
    align_streams,
    compute_stream_delays,
    resample_to_fs,
    synchronize_events
)


class TestResampleToFs:
    """Test signal resampling functionality."""

    def test_resample_identity(self):
        """Test that resampling to same frequency returns identical signal."""
        x = np.random.randn(1000).astype(np.float32)
        fs = 125.0

        y = resample_to_fs(x, fs, fs)

        assert y.shape == x.shape
        np.testing.assert_array_almost_equal(y, x)

    def test_resample_downsample(self):
        """Test downsampling preserves signal characteristics."""
        # Create a 10 Hz sine wave at 1000 Hz
        fs_in = 1000.0
        duration = 2.0
        t = np.arange(0, duration, 1 / fs_in)
        freq = 10.0
        x = np.sin(2 * np.pi * freq * t)

        # Downsample to 125 Hz (still > Nyquist for 10 Hz)
        fs_out = 125.0
        y = resample_to_fs(x, fs_in, fs_out)

        # Check output length
        expected_samples = int(len(x) * fs_out / fs_in)
        assert abs(len(y) - expected_samples) <= 1

        # Check that frequency is preserved using FFT
        fft_y = np.fft.rfft(y)
        freqs = np.fft.rfftfreq(len(y), 1 / fs_out)

        # Find peak frequency
        peak_idx = np.argmax(np.abs(fft_y[1:100])) + 1  # Skip DC
        peak_freq = freqs[peak_idx]

        # Should be close to 10 Hz
        assert abs(peak_freq - freq) < 1.0, f"Expected {freq} Hz, got {peak_freq} Hz"

    def test_resample_upsample(self):
        """Test upsampling maintains signal shape."""
        # Create signal at 50 Hz
        fs_in = 50.0
        duration = 1.0
        t = np.arange(0, duration, 1 / fs_in)
        x = np.sin(2 * np.pi * 5 * t)  # 5 Hz signal

        # Upsample to 200 Hz
        fs_out = 200.0
        y = resample_to_fs(x, fs_in, fs_out)

        # Check output length
        expected_samples = int(len(x) * fs_out / fs_in)
        assert abs(len(y) - expected_samples) <= 1

        # Signal should be smoother but maintain overall shape
        # Downsample result back and compare
        x_recovered = resample_to_fs(y, fs_out, fs_in)

        # Should be close to original (some loss due to filtering)
        correlation = np.corrcoef(x[:len(x_recovered)], x_recovered)[0, 1]
        assert correlation > 0.99

    def test_resample_antialiasing(self):
        """Test that anti-aliasing filter is applied during downsampling."""
        # Create signal with high frequency component
        fs_in = 1000.0
        duration = 1.0
        t = np.arange(0, duration, 1 / fs_in)

        # Mix of 10 Hz and 100 Hz (100 Hz will alias at 125 Hz)
        x = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 100 * t)

        # Downsample to 125 Hz (Nyquist = 62.5 Hz)
        fs_out = 125.0
        y = resample_to_fs(x, fs_in, fs_out)

        # Check that high frequency is attenuated
        fft_y = np.fft.rfft(y)
        freqs = np.fft.rfftfreq(len(y), 1 / fs_out)

        # Power at 10 Hz should be preserved
        idx_10hz = np.argmin(np.abs(freqs - 10))
        power_10hz = np.abs(fft_y[idx_10hz])

        # Power above Nyquist should be minimal
        idx_nyquist = np.argmin(np.abs(freqs - fs_out / 2))
        power_high = np.max(np.abs(fft_y[idx_nyquist:]))

        # High frequency should be much smaller than low frequency
        assert power_high < power_10hz * 0.1

    def test_resample_methods(self):
        """Test different resampling methods produce similar results."""
        x = np.random.randn(1000).astype(np.float32)
        fs_in = 500.0
        fs_out = 125.0

        y_poly = resample_to_fs(x, fs_in, fs_out, method='poly')
        y_fourier = resample_to_fs(x, fs_in, fs_out, method='fourier')

        # Methods should produce similar results
        correlation = np.corrcoef(y_poly, y_fourier[:len(y_poly)])[0, 1]
        assert correlation > 0.95


class TestAlignStreams:
    """Test multi-stream alignment functionality."""

    def test_align_same_fs(self):
        """Test alignment of streams with same sampling frequency."""
        n_samples = 1000
        fs = 125.0

        streams = {
            'ECG': (np.random.randn(n_samples), fs),
            'PPG': (np.random.randn(n_samples), fs),
            'ABP': (np.random.randn(n_samples), fs)
        }

        aligned = align_streams(streams, target_fs_hz=fs)

        # Check shape
        assert aligned.shape == (n_samples, 3)

        # Check channels are ordered alphabetically
        # ABP (0), ECG (1), PPG (2)
        assert aligned.shape[1] == len(streams)

    def test_align_different_fs(self):
        """Test alignment of streams with different sampling frequencies."""
        duration_s = 10.0

        # Create signals at different frequencies
        fs_ecg = 500.0
        fs_ppg = 125.0
        fs_abp = 100.0

        n_ecg = int(duration_s * fs_ecg)
        n_ppg = int(duration_s * fs_ppg)
        n_abp = int(duration_s * fs_abp)

        streams = {
            'ECG': (np.random.randn(n_ecg), fs_ecg),
            'PPG': (np.random.randn(n_ppg), fs_ppg),
            'ABP': (np.random.randn(n_abp), fs_abp)
        }

        target_fs = 125.0
        aligned = align_streams(streams, target_fs_hz=target_fs)

        # Check output
        expected_samples = int(duration_s * target_fs)
        assert abs(aligned.shape[0] - expected_samples) <= 1
        assert aligned.shape[1] == 3

    def test_align_with_duration_limit(self):
        """Test alignment with maximum duration limit."""
        fs = 125.0
        long_signal = np.random.randn(10000)  # 80 seconds at 125 Hz

        streams = {
            'signal1': (long_signal, fs),
            'signal2': (long_signal, fs)
        }

        max_duration_s = 10.0
        aligned = align_streams(streams, target_fs_hz=fs, max_duration_s=max_duration_s)

        expected_samples = int(max_duration_s * fs)
        assert aligned.shape[0] == expected_samples
        assert aligned.shape[1] == 2

    def test_align_with_offset(self):
        """Test alignment with start offset."""
        fs = 125.0
        n_samples = 2500  # 20 seconds

        streams = {
            'signal': (np.arange(n_samples), fs)
        }

        start_offset_s = 5.0
        aligned = align_streams(
            streams,
            target_fs_hz=fs,
            start_offset_s=start_offset_s
        )

        # Check that we got the right portion
        offset_samples = int(start_offset_s * fs)
        expected_samples = n_samples - offset_samples

        assert aligned.shape[0] == expected_samples
        # First value should be the offset value
        assert aligned[0, 0] == offset_samples

    def test_align_empty_streams(self):
        """Test that empty streams raise an error."""
        with pytest.raises(ValueError, match="No streams provided"):
            align_streams({})

    def test_align_channel_ordering(self):
        """Test that channels are consistently ordered alphabetically."""
        streams = {
            'ZZZ': (np.array([3, 3, 3]), 1.0),
            'AAA': (np.array([1, 1, 1]), 1.0),
            'MMM': (np.array([2, 2, 2]), 1.0)
        }

        aligned = align_streams(streams, target_fs_hz=1.0)

        # Channels should be ordered: AAA, MMM, ZZZ
        np.testing.assert_array_equal(aligned[:, 0], [1, 1, 1])  # AAA
        np.testing.assert_array_equal(aligned[:, 1], [2, 2, 2])  # MMM
        np.testing.assert_array_equal(aligned[:, 2], [3, 3, 3])  # ZZZ


class TestSynchronizeEvents:
    """Test event synchronization."""

    def test_synchronize_events_basic(self):
        """Test basic event synchronization."""
        fs = 100.0

        # Create events with slight offsets
        events = {
            'ECG': np.array([100, 200, 300, 400]),
            'PPG': np.array([102, 198, 303, 399])  # Slight delays
        }

        synchronized = synchronize_events(events, fs, window_s=0.05)

        # Should have synchronized events
        assert 'ECG' in synchronized
        assert 'PPG' in synchronized

        # PPG events should be aligned to ECG (reference)
        assert len(synchronized['PPG']) > 0
        assert len(synchronized['PPG']) <= len(events['PPG'])

    def test_synchronize_events_no_match(self):
        """Test synchronization when events don't match."""
        fs = 100.0

        events = {
            'ECG': np.array([100, 200, 300]),
            'PPG': np.array([500, 600, 700])  # No overlap
        }

        synchronized = synchronize_events(events, fs, window_s=0.05)

        # ECG should be preserved, PPG should have no matches
        assert 'ECG' in synchronized
        assert 'PPG' not in synchronized or len(synchronized['PPG']) == 0


class TestComputeStreamDelays:
    """Test delay computation between streams."""

    def test_compute_delays_no_delay(self):
        """Test delay computation with aligned signals."""
        fs = 100.0
        signal = np.random.randn(1000)

        streams = {
            'signal1': (signal, fs),
            'signal2': (signal, fs)
        }

        delays = compute_stream_delays(streams)

        # Both should have zero delay
        assert delays['signal1'] == 0.0
        assert abs(delays['signal2']) < 0.01  # Small tolerance

    def test_compute_delays_with_shift(self):
        """Test delay computation with shifted signal."""
        fs = 100.0
        signal = np.random.randn(1000)
        
        # Shift signal2 by 10 samples (0.1 seconds)
        # np.roll with positive shift moves elements forward (earlier in time)
        shift_samples = 10
        signal_shifted = np.roll(signal, shift_samples)
        
        streams = {
            'signal1': (signal, fs),
            'signal2': (signal_shifted, fs)
        }
        
        delays = compute_stream_delays(streams, max_lag_s=1.0)
        
        # signal2 should have negative delay (leading) because roll shifts forward
        expected_delay = -shift_samples / fs
        assert abs(delays['signal2'] - expected_delay) < 0.02  # 20ms tolerance

    def test_compute_delays_different_fs(self):
        """Test delay computation with different sampling rates."""
        # Create correlated signals at different rates
        duration = 5.0
        fs1 = 100.0
        fs2 = 125.0

        t1 = np.arange(0, duration, 1 / fs1)
        t2 = np.arange(0, duration, 1 / fs2)

        # Same underlying signal
        signal1 = np.sin(2 * np.pi * 2 * t1)
        signal2 = np.sin(2 * np.pi * 2 * t2)

        streams = {
            'signal1': (signal1, fs1),
            'signal2': (signal2, fs2)
        }

        delays = compute_stream_delays(streams)

        # Should be minimal delay
        assert abs(delays['signal2']) < 0.05  # 50ms tolerance


def test_integration_full_pipeline():
    """Test full pipeline of loading, resampling, and aligning."""
    # Create synthetic multi-channel data
    duration_s = 5.0

    # ECG at 500 Hz
    fs_ecg = 500.0
    t_ecg = np.arange(0, duration_s, 1 / fs_ecg)
    ecg = np.sin(2 * np.pi * 1.2 * t_ecg)  # ~72 bpm

    # PPG at 125 Hz
    fs_ppg = 125.0
    t_ppg = np.arange(0, duration_s, 1 / fs_ppg)
    ppg = np.sin(2 * np.pi * 1.2 * t_ppg) ** 2

    # ABP at 100 Hz
    fs_abp = 100.0
    t_abp = np.arange(0, duration_s, 1 / fs_abp)
    abp = 100 + 20 * np.sin(2 * np.pi * 1.2 * t_abp)

    # Create streams
    streams = {
        'ECG_II': (ecg, fs_ecg),
        'PLETH': (ppg, fs_ppg),
        'ART': (abp, fs_abp)
    }

    # Align to 125 Hz
    target_fs = 125.0
    aligned = align_streams(streams, target_fs_hz=target_fs)

    # Verify output
    expected_samples = int(duration_s * target_fs)
    assert aligned.shape[0] == expected_samples
    assert aligned.shape[1] == 3

    # Verify no NaN or Inf values
    assert np.all(np.isfinite(aligned))

    # Verify reasonable value ranges
    assert aligned.min() > -1000 and aligned.max() < 1000
