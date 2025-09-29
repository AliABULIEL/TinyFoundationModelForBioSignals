"""Tests for peak detection algorithms."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.data.detect import (
    find_ecg_rpeaks,
    find_ppg_peaks,
    compute_peak_statistics,
    validate_peaks,
    align_peaks
)


class TestECGDetection:
    """Test ECG R-peak detection."""
    
    def test_ecg_synthetic_regular(self):
        """Test detection on synthetic regular ECG."""
        fs = 500.0  # High sampling rate for ECG
        duration = 10.0
        hr = 72  # bpm
        
        # Create synthetic ECG with R-peaks
        t = np.arange(0, duration, 1/fs)
        ecg = np.zeros_like(t)
        
        # Add R-peaks at regular intervals
        peak_interval = 60.0 / hr  # seconds
        peak_samples = int(peak_interval * fs)
        expected_peaks = np.arange(0, len(ecg), peak_samples)
        
        # Create QRS-like spikes
        for peak_idx in expected_peaks:
            if peak_idx < len(ecg):
                # Simple triangular QRS
                ecg[peak_idx] = 1.0
                if peak_idx > 0:
                    ecg[peak_idx-1] = 0.5
                if peak_idx < len(ecg) - 1:
                    ecg[peak_idx+1] = 0.5
        
        # Add some noise
        ecg += 0.05 * np.random.randn(len(ecg))
        
        # Detect peaks
        peaks, hr_series = find_ecg_rpeaks(ecg, fs, mode='analysis')
        
        # Check detection accuracy
        assert len(peaks) > 0, "No peaks detected"
        
        # Should detect approximately the right number of peaks
        expected_count = int(duration * hr / 60)
        assert abs(len(peaks) - expected_count) <= 2, f"Expected ~{expected_count} peaks, got {len(peaks)}"
        
        # Check heart rate series
        assert len(hr_series) == len(ecg), "HR series length mismatch"
        mean_hr = np.mean(hr_series[hr_series > 0])
        assert abs(mean_hr - hr) < 10, f"Mean HR {mean_hr:.1f} far from expected {hr}"
    
    def test_ecg_variable_rate(self):
        """Test detection with variable heart rate."""
        fs = 500.0
        duration = 10.0
        
        t = np.arange(0, duration, 1/fs)
        ecg = np.zeros_like(t)
        
        # Variable heart rate (60-80 bpm)
        current_time = 0
        expected_peaks = []
        
        while current_time < duration:
            # Variable HR
            hr = 60 + 20 * np.sin(2 * np.pi * 0.1 * current_time)
            interval = 60.0 / hr
            
            peak_sample = int(current_time * fs)
            if peak_sample < len(ecg):
                ecg[peak_sample] = 1.0
                expected_peaks.append(peak_sample)
            
            current_time += interval
        
        # Detect peaks
        peaks, hr_series = find_ecg_rpeaks(ecg, fs, mode='analysis')
        
        # Should detect most peaks
        assert len(peaks) > 0
        assert len(peaks) >= len(expected_peaks) - 2
    
    def test_ecg_rpeak_mode(self):
        """Test R-peak specific detection mode."""
        fs = 500.0
        duration = 5.0
        
        # Create ECG with high-frequency noise
        t = np.arange(0, duration, 1/fs)
        ecg = np.sin(2 * np.pi * 1.2 * t)  # ~72 bpm base
        ecg += 0.3 * np.sin(2 * np.pi * 50 * t)  # High-frequency noise
        
        # Add clear R-peaks
        peak_interval = int(0.83 * fs)  # ~72 bpm
        for i in range(0, len(ecg), peak_interval):
            if i < len(ecg):
                ecg[i] += 2.0  # Strong R-peak
        
        # Detect in R-peak mode (should filter out HF noise)
        peaks_rpeak, _ = find_ecg_rpeaks(ecg, fs, mode='rpeak')
        
        # Detect in analysis mode
        peaks_analysis, _ = find_ecg_rpeaks(ecg, fs, mode='analysis')
        
        # Both should detect peaks
        assert len(peaks_rpeak) > 0
        assert len(peaks_analysis) > 0
        
        # Counts should be similar
        assert abs(len(peaks_rpeak) - len(peaks_analysis)) <= 2


class TestPPGDetection:
    """Test PPG peak detection."""
    
    def test_ppg_synthetic_regular(self):
        """Test detection on synthetic regular PPG."""
        fs = 125.0
        duration = 10.0
        hr = 72  # bpm
        
        # Create synthetic PPG
        t = np.arange(0, duration, 1/fs)
        
        # PPG-like waveform (sine squared for pulse-like shape)
        f_pulse = hr / 60.0  # Hz
        ppg = np.sin(2 * np.pi * f_pulse * t) ** 2
        
        # Add small amount of noise
        ppg += 0.02 * np.random.randn(len(ppg))
        
        # Detect peaks
        peaks = find_ppg_peaks(ppg, fs)
        
        # Check detection
        assert len(peaks) > 0, "No peaks detected"
        
        # Should detect approximately the right number
        expected_count = int(duration * hr / 60)
        assert abs(len(peaks) - expected_count) <= 2, f"Expected ~{expected_count} peaks, got {len(peaks)}"
    
    def test_ppg_with_artifacts(self):
        """Test PPG detection with artifacts."""
        fs = 125.0
        duration = 10.0
        
        # Create PPG with artifacts
        t = np.arange(0, duration, 1/fs)
        ppg = np.sin(2 * np.pi * 1.2 * t) ** 2
        
        # Add motion artifact (low frequency)
        ppg += 0.5 * np.sin(2 * np.pi * 0.2 * t)
        
        # Add spike artifacts
        spike_indices = np.random.choice(len(ppg), 10, replace=False)
        ppg[spike_indices] += 2.0
        
        # Detect peaks
        peaks = find_ppg_peaks(ppg, fs)
        
        # Should still detect some peaks
        assert len(peaks) > 5, "Too few peaks detected with artifacts"
    
    def test_ppg_low_quality(self):
        """Test PPG detection on low quality signal."""
        fs = 125.0
        duration = 5.0
        
        # Very noisy PPG
        t = np.arange(0, duration, 1/fs)
        ppg = 0.2 * np.sin(2 * np.pi * 1.2 * t) ** 2  # Weak signal
        ppg += 0.5 * np.random.randn(len(ppg))  # Strong noise
        
        # Detect peaks
        peaks = find_ppg_peaks(ppg, fs)
        
        # May detect fewer peaks but shouldn't crash
        assert isinstance(peaks, np.ndarray)
        # May have no peaks in very noisy signal
        assert len(peaks) >= 0


class TestPeakStatistics:
    """Test peak statistics computation."""
    
    def test_compute_statistics(self):
        """Test peak statistics calculation."""
        fs = 125.0
        
        # Regular peaks at 60 bpm
        peak_interval = int(fs * 1.0)  # 1 second = 60 bpm
        peaks = np.arange(0, 10) * peak_interval
        
        stats = compute_peak_statistics(peaks, fs, signal_length=10*int(fs))
        
        # Check statistics
        assert stats['count'] == 10
        assert abs(stats['mean_rate'] - 60.0) < 1.0
        assert stats['std_rate'] < 1.0  # Should be very consistent
        assert abs(stats['mean_interval'] - 1.0) < 0.01
        assert stats['coverage'] > 0.7  # Most of signal covered
    
    def test_statistics_few_peaks(self):
        """Test statistics with insufficient peaks."""
        fs = 125.0
        
        # Only one peak
        peaks = np.array([100])
        
        stats = compute_peak_statistics(peaks, fs, signal_length=1000)
        
        assert stats['count'] == 1
        assert np.isnan(stats['mean_rate'])
        assert np.isnan(stats['std_rate'])
        assert stats['coverage'] == 0.0


class TestPeakValidation:
    """Test peak validation functionality."""
    
    def test_validate_physiological(self):
        """Test validation of physiological peaks."""
        fs = 125.0
        
        # Mix of valid and invalid peaks
        peaks = np.array([0, 50, 100, 102, 200, 300, 301])  # Some too close
        
        valid_peaks, removed = validate_peaks(peaks, fs, signal_length=400)
        
        # Should remove peaks that are too close (>200 bpm)
        assert len(valid_peaks) < len(peaks)
        assert len(removed) > 0
        
        # Check remaining peaks are physiologically valid
        if len(valid_peaks) > 1:
            intervals = np.diff(valid_peaks) / fs
            rates = 60.0 / intervals
            assert np.all(rates <= 200)
            assert np.all(rates >= 30)
    
    def test_validate_empty(self):
        """Test validation with no peaks."""
        peaks = np.array([])
        valid_peaks, removed = validate_peaks(peaks, 125.0, 1000)
        
        assert len(valid_peaks) == 0
        assert len(removed) == 0


class TestPeakAlignment:
    """Test ECG-PPG peak alignment."""
    
    def test_align_with_delay(self):
        """Test alignment with pulse transit time."""
        fs = 125.0
        
        # ECG peaks
        ecg_peaks = np.array([0, 125, 250, 375, 500])
        
        # PPG peaks with 200ms delay
        delay_samples = int(0.2 * fs)  # 200ms
        ppg_peaks = ecg_peaks + delay_samples
        
        # Add some jitter
        ppg_peaks += np.random.randint(-2, 3, size=len(ppg_peaks))
        
        # Align peaks
        aligned_ecg, aligned_ppg, mean_delay = align_peaks(
            ecg_peaks, ppg_peaks, max_delay_samples=int(0.5*fs), fs=fs
        )
        
        # Should align most peaks
        assert len(aligned_ecg) >= len(ecg_peaks) - 1
        assert len(aligned_ecg) == len(aligned_ppg)
        
        # Mean delay should be close to true delay
        assert abs(mean_delay - delay_samples) < 5
    
    def test_align_no_overlap(self):
        """Test alignment with no overlapping peaks."""
        ecg_peaks = np.array([0, 100, 200])
        ppg_peaks = np.array([500, 600, 700])  # No overlap
        
        aligned_ecg, aligned_ppg, mean_delay = align_peaks(
            ecg_peaks, ppg_peaks, max_delay_samples=50
        )
        
        # Should find no alignments
        assert len(aligned_ecg) == 0
        assert len(aligned_ppg) == 0
        assert mean_delay == 0.0


def test_integration_ecg_ppg():
    """Integration test for ECG and PPG detection."""
    fs = 125.0
    duration = 10.0
    t = np.arange(0, duration, 1/fs)
    
    # Create correlated ECG and PPG
    hr = 72
    f = hr / 60.0
    
    # Simple ECG with spikes
    ecg = np.zeros_like(t)
    peak_interval = int(fs / f)
    for i in range(0, len(ecg), peak_interval):
        if i < len(ecg):
            ecg[i] = 1.0
    
    # PPG with delay
    delay = 0.15  # 150ms PTT
    ppg = np.sin(2 * np.pi * f * (t - delay)) ** 2
    
    # Add noise
    ecg += 0.05 * np.random.randn(len(ecg))
    ppg += 0.02 * np.random.randn(len(ppg))
    
    # Detect peaks
    ecg_peaks, hr_series = find_ecg_rpeaks(ecg, fs)
    ppg_peaks = find_ppg_peaks(ppg, fs)
    
    # Both should detect peaks
    assert len(ecg_peaks) > 5
    assert len(ppg_peaks) > 5
    
    # Align peaks
    aligned_ecg, aligned_ppg, mean_delay = align_peaks(
        ecg_peaks, ppg_peaks, fs=fs
    )
    
    # Should find alignments
    assert len(aligned_ecg) > 3
    
    # Delay should be roughly correct
    expected_delay_samples = delay * fs
    assert abs(mean_delay - expected_delay_samples) < 10
    
    print("✅ All detection tests passed!")


if __name__ == "__main__":
    test_integration_ecg_ppg()
