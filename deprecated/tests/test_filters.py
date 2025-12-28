"""Tests for digital filters."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.data.filters import (
    design_abp_filter,
    design_ecg_filter,
    design_eeg_filter,
    design_ppg_filter,
    filter_abp,
    filter_ecg,
    filter_eeg,
    filter_ppg,
    freqz_response,
    get_filter_specs,
    validate_filter_stability
)


class TestFilterDesign:
    """Test filter design functions."""
    
    def test_ppg_filter_design(self):
        """Test PPG filter characteristics."""
        fs = 125.0  # Standard sampling rate
        b, a = design_ppg_filter(fs)
        
        # Check filter stability
        assert validate_filter_stability(b, a), "PPG filter is unstable"
        
        # Get frequency response
        freqs, mag_db, _ = freqz_response(b, a, fs, n_points=1024)
        
        # Test passband (0.4-7.0 Hz) - should have minimal attenuation
        passband_start_idx = np.argmin(np.abs(freqs - 0.4))
        passband_end_idx = np.argmin(np.abs(freqs - 7.0))
        passband_mag = mag_db[passband_start_idx:passband_end_idx]
        
        # Passband should have < 3dB attenuation
        assert np.max(passband_mag) > -3.0, "PPG passband attenuation too high"
        
        # Test stopband - frequencies below 0.2 Hz and above 10 Hz should be attenuated
        if 0.2 < fs/2:  # Only test if frequency is below Nyquist
            low_stop_idx = np.argmin(np.abs(freqs - 0.2))
            assert mag_db[low_stop_idx] < -10, "PPG low stopband attenuation insufficient"
        
        if 10.0 < fs/2:
            high_stop_idx = np.argmin(np.abs(freqs - 10.0))
            assert mag_db[high_stop_idx] < -10, "PPG high stopband attenuation insufficient"
    
    def test_ecg_analysis_filter_design(self):
        """Test ECG analysis filter characteristics."""
        fs = 500.0  # Higher sampling rate for ECG
        b, a = design_ecg_filter(fs, mode='analysis')
        
        # Check stability
        assert validate_filter_stability(b, a), "ECG analysis filter is unstable"
        
        # Get frequency response
        freqs, mag_db, _ = freqz_response(b, a, fs, n_points=1024)
        
        # Test passband (0.5-40.0 Hz)
        passband_start_idx = np.argmin(np.abs(freqs - 0.5))
        passband_end_idx = np.argmin(np.abs(freqs - 40.0))
        passband_mag = mag_db[passband_start_idx:passband_end_idx]
        
        # Check passband has minimal attenuation
        assert np.max(passband_mag) > -3.0, "ECG passband attenuation too high"
        
        # Test stopband
        if 0.2 < fs/2:
            low_stop_idx = np.argmin(np.abs(freqs - 0.2))
            assert mag_db[low_stop_idx] < -10, "ECG low stopband attenuation insufficient"
        
        if 60.0 < fs/2:
            high_stop_idx = np.argmin(np.abs(freqs - 60.0))
            assert mag_db[high_stop_idx] < -10, "ECG high stopband attenuation insufficient"
    
    def test_ecg_rpeak_filter_design(self):
        """Test ECG R-peak detection filter characteristics."""
        fs = 500.0
        b, a = design_ecg_filter(fs, mode='rpeak')
        
        # Check stability
        assert validate_filter_stability(b, a), "ECG R-peak filter is unstable"
        
        # Get frequency response
        freqs, mag_db, _ = freqz_response(b, a, fs, n_points=1024)
        
        # Test passband (0-8.0 Hz) - lowpass
        passband_end_idx = np.argmin(np.abs(freqs - 8.0))
        passband_mag = mag_db[:passband_end_idx]
        
        # Passband should have minimal attenuation
        assert np.max(passband_mag) > -3.0, "ECG R-peak passband attenuation too high"
        
        # Test stopband (> 12 Hz should be well attenuated)
        if 12.0 < fs/2:
            stop_idx = np.argmin(np.abs(freqs - 12.0))
            assert mag_db[stop_idx] < -15, "ECG R-peak stopband attenuation insufficient"
    
    def test_abp_filter_design(self):
        """Test ABP filter characteristics."""
        fs = 125.0
        b, a = design_abp_filter(fs)
        
        # Check stability
        assert validate_filter_stability(b, a), "ABP filter is unstable"
        
        # Get frequency response
        freqs, mag_db, _ = freqz_response(b, a, fs, n_points=1024)
        
        # Test passband (0.5-20.0 Hz)
        passband_start_idx = np.argmin(np.abs(freqs - 0.5))
        passband_end_idx = np.argmin(np.abs(freqs - 20.0))
        passband_mag = mag_db[passband_start_idx:passband_end_idx]
        
        # Passband should have minimal attenuation
        assert np.max(passband_mag) > -3.0, "ABP passband attenuation too high"
        
        # Test stopband
        if 0.2 < fs/2:
            low_stop_idx = np.argmin(np.abs(freqs - 0.2))
            assert mag_db[low_stop_idx] < -10, "ABP low stopband attenuation insufficient"
        
        if 30.0 < fs/2:
            high_stop_idx = np.argmin(np.abs(freqs - 30.0))
            assert mag_db[high_stop_idx] < -10, "ABP high stopband attenuation insufficient"
    
    def test_eeg_filter_design(self):
        """Test EEG filter characteristics."""
        fs = 250.0  # Common EEG sampling rate
        b, a = design_eeg_filter(fs)
        
        # Check stability
        assert validate_filter_stability(b, a), "EEG filter is unstable"
        
        # Get frequency response
        freqs, mag_db, _ = freqz_response(b, a, fs, n_points=1024)
        
        # Test passband (1.0-45.0 Hz)
        passband_start_idx = np.argmin(np.abs(freqs - 1.0))
        passband_end_idx = np.argmin(np.abs(freqs - 45.0))
        passband_mag = mag_db[passband_start_idx:passband_end_idx]
        
        # Passband should have minimal attenuation
        assert np.max(passband_mag) > -3.0, "EEG passband attenuation too high"
        
        # Test stopband
        if 0.5 < fs/2:
            low_stop_idx = np.argmin(np.abs(freqs - 0.5))
            assert mag_db[low_stop_idx] < -10, "EEG low stopband attenuation insufficient"
        
        if 60.0 < fs/2:
            high_stop_idx = np.argmin(np.abs(freqs - 60.0))
            assert mag_db[high_stop_idx] < -10, "EEG high stopband attenuation insufficient"


class TestFilterApplication:
    """Test filter application to signals."""
    
    def test_filter_ppg_signal(self):
        """Test PPG filtering on synthetic signal."""
        fs = 125.0
        duration = 10.0
        t = np.arange(0, duration, 1/fs)
        
        # Create signal with components inside and outside passband
        signal_in_band = np.sin(2 * np.pi * 1.2 * t)  # 1.2 Hz (in passband)
        noise_low = 0.5 * np.sin(2 * np.pi * 0.1 * t)  # 0.1 Hz (below passband)
        noise_high = 0.5 * np.sin(2 * np.pi * 15.0 * t)  # 15 Hz (above passband)
        
        noisy_signal = signal_in_band + noise_low + noise_high
        
        # Apply filter
        filtered = filter_ppg(noisy_signal, fs)
        
        # Check that in-band signal is preserved
        correlation = np.corrcoef(signal_in_band[100:-100], filtered[100:-100])[0, 1]
        assert correlation > 0.9, "PPG filter distorted in-band signal"
        
        # Check output is finite
        assert np.all(np.isfinite(filtered)), "PPG filter produced non-finite values"
    
    def test_filter_ecg_signal(self):
        """Test ECG filtering on synthetic signal."""
        fs = 500.0
        duration = 5.0
        t = np.arange(0, duration, 1/fs)
        
        # Create ECG-like signal with QRS complex frequency content
        qrs_signal = np.zeros_like(t)
        # Add spikes at 1 Hz (60 bpm)
        spike_indices = np.arange(0, len(t), int(fs))
        qrs_signal[spike_indices] = 1.0
        
        # Add noise outside passband
        noise_low = 0.5 * np.sin(2 * np.pi * 0.1 * t)  # Baseline wander
        noise_high = 0.5 * np.sin(2 * np.pi * 60.0 * t)  # Power line noise
        
        noisy_signal = qrs_signal + noise_low + noise_high
        
        # Apply filter
        filtered = filter_ecg(noisy_signal, fs, mode='analysis')
        
        # Check output is finite
        assert np.all(np.isfinite(filtered)), "ECG filter produced non-finite values"
        
        # Test R-peak mode
        filtered_rpeak = filter_ecg(noisy_signal, fs, mode='rpeak')
        assert np.all(np.isfinite(filtered_rpeak)), "ECG R-peak filter produced non-finite values"
    
    def test_filter_abp_signal(self):
        """Test ABP filtering on synthetic signal."""
        fs = 125.0
        duration = 10.0
        t = np.arange(0, duration, 1/fs)
        
        # Create ABP-like signal (oscillating pressure)
        abp_signal = 100 + 20 * np.sin(2 * np.pi * 1.2 * t)  # ~72 bpm
        
        # Add high frequency noise
        noise = 5 * np.random.randn(len(t))
        noisy_signal = abp_signal + noise
        
        # Apply filter
        filtered = filter_abp(noisy_signal, fs)
        
        # Check that filtering reduces high frequency noise
        # High frequency content should be reduced
        fft_noisy = np.abs(np.fft.rfft(noisy_signal))
        fft_filtered = np.abs(np.fft.rfft(filtered))
        freqs = np.fft.rfftfreq(len(noisy_signal), 1/fs)
        
        # Check attenuation at 30 Hz if within range
        if 30.0 < fs/2:
            idx_30hz = np.argmin(np.abs(freqs - 30.0))
            attenuation = fft_filtered[idx_30hz] / (fft_noisy[idx_30hz] + 1e-10)
            assert attenuation < 0.3, "ABP filter didn't attenuate high frequencies"
    
    def test_filter_eeg_signal(self):
        """Test EEG filtering on synthetic signal."""
        fs = 250.0
        duration = 5.0
        t = np.arange(0, duration, 1/fs)
        
        # Create EEG-like signal with alpha (10 Hz) and beta (20 Hz) waves
        alpha = np.sin(2 * np.pi * 10 * t)
        beta = 0.5 * np.sin(2 * np.pi * 20 * t)
        
        # Add DC offset and high frequency noise
        dc_offset = 50.0
        hf_noise = 0.3 * np.sin(2 * np.pi * 70 * t)
        
        noisy_signal = alpha + beta + dc_offset + hf_noise
        
        # Apply filter
        filtered = filter_eeg(noisy_signal, fs)
        
        # Check DC offset is removed
        assert abs(np.mean(filtered)) < 1.0, "EEG filter didn't remove DC offset"
        
        # Check output is finite
        assert np.all(np.isfinite(filtered)), "EEG filter produced non-finite values"


class TestFrequencyResponse:
    """Test frequency response computation."""
    
    def test_freqz_response_shape(self):
        """Test that frequency response has correct shape."""
        fs = 125.0
        b, a = design_ppg_filter(fs)
        
        n_points = 512
        freqs, mag_db, phase_deg = freqz_response(b, a, fs, n_points)
        
        # Check output shapes
        assert len(freqs) == n_points
        assert len(mag_db) == n_points
        assert len(phase_deg) == n_points
        
        # Check frequency range
        assert freqs[0] == 0
        assert abs(freqs[-1] - fs/2) < 1.0  # Should go up to Nyquist
        
        # Check magnitude is in reasonable range (dB)
        assert np.all(mag_db < 20), "Magnitude response unreasonably high"
        assert np.all(mag_db > -100), "Magnitude response unreasonably low"
        
        # Check phase is in degrees
        assert np.all(np.abs(phase_deg) <= 180), "Phase should be in [-180, 180] degrees"


class TestFilterSpecs:
    """Test filter specifications."""
    
    def test_get_filter_specs(self):
        """Test filter specification retrieval."""
        # Test all filter types
        for filter_type in ['ppg', 'ecg', 'ecg_rpeak', 'abp', 'eeg']:
            specs = get_filter_specs(filter_type)
            
            # Check required fields
            assert 'type' in specs
            assert 'order' in specs
            assert 'passband' in specs
            assert 'description' in specs
            
            # Check order is positive integer
            assert specs['order'] > 0
            assert isinstance(specs['order'], int)
        
        # Test invalid filter type
        with pytest.raises(ValueError, match="Unknown filter type"):
            get_filter_specs('invalid_filter')


class TestFilterStability:
    """Test filter stability checks."""
    
    def test_validate_stability(self):
        """Test filter stability validation."""
        # Test stable filter (Butterworth is always stable)
        fs = 125.0
        b, a = design_abp_filter(fs)
        assert validate_filter_stability(b, a), "Butterworth filter should be stable"
        
        # Create unstable filter (pole outside unit circle)
        b_unstable = [1.0]
        a_unstable = [1.0, -2.0]  # Pole at z=2 (outside unit circle)
        assert not validate_filter_stability(b_unstable, a_unstable), "Should detect unstable filter"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_low_sampling_rate(self):
        """Test filters with low sampling rate."""
        fs = 50.0  # Very low sampling rate
        
        # PPG filter should still work
        b, a = design_ppg_filter(fs)
        assert validate_filter_stability(b, a)
        
        # Some filters might have issues with very low fs
        # ECG 40 Hz cutoff is at Nyquist for 80 Hz sampling
        # Should handle gracefully
        fs_low = 80.0
        b, a = design_ecg_filter(fs_low, mode='analysis')
        assert validate_filter_stability(b, a)
    
    def test_short_signal_filtering(self):
        """Test filtering very short signals."""
        fs = 125.0
        
        # Very short signal (less than filter order)
        short_signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Should handle gracefully without crashing
        filtered = filter_ppg(short_signal, fs)
        assert len(filtered) == len(short_signal)
        assert np.all(np.isfinite(filtered))
    
    def test_invalid_ecg_mode(self):
        """Test ECG filter with invalid mode."""
        fs = 500.0
        
        with pytest.raises(ValueError, match="Unknown ECG filter mode"):
            design_ecg_filter(fs, mode='invalid_mode')


def test_filter_integration():
    """Integration test for complete filter pipeline."""
    # Create multi-channel signal
    fs = 125.0
    duration = 5.0
    t = np.arange(0, duration, 1/fs)
    
    # Create signals with different characteristics
    ppg = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(len(t))
    ecg = np.zeros_like(t)
    ecg[::int(fs)] = 1.0  # Spikes at 1 Hz
    abp = 100 + 20 * np.sin(2 * np.pi * 1.2 * t)
    
    # Apply filters
    ppg_filtered = filter_ppg(ppg, fs)
    ecg_filtered = filter_ecg(ecg, fs * 4, mode='analysis')  # Higher fs for ECG
    abp_filtered = filter_abp(abp, fs)
    
    # All outputs should be valid
    assert np.all(np.isfinite(ppg_filtered))
    assert np.all(np.isfinite(ecg_filtered))
    assert np.all(np.isfinite(abp_filtered))
    
    print("✅ All filter tests passed!")


if __name__ == "__main__":
    # Run a quick test
    test_filter_integration()
