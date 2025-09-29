"""Signal quality assessment and artifact detection.

Implements SQI metrics and quality gates for biosignals.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal, stats


def ecg_sqi(
    ecg: np.ndarray,
    peaks: np.ndarray,
    fs: float = 125.0,
    window_s: float = 10.0
) -> float:
    """Compute ECG Signal Quality Index (SQI).
    
    Args:
        ecg: ECG signal array.
        peaks: R-peak indices from detection.
        fs: Sampling frequency.
        window_s: Window duration for quality assessment.
        
    Returns:
        SQI score between 0 and 1 (higher is better).
        
    Note:
        Combines multiple quality metrics:
        - Peak regularity
        - Signal-to-noise ratio
        - Baseline stability
    """
    if len(ecg) == 0:
        return 0.0
    
    if len(peaks) < 3:
        # Not enough peaks for quality assessment
        return 0.0
    
    scores = []
    
    # 1. Peak regularity score
    intervals = np.diff(peaks) / fs
    if len(intervals) > 0:
        # Coefficient of variation of RR intervals
        cv = np.std(intervals) / (np.mean(intervals) + 1e-10)
        # Convert to score (lower CV is better)
        regularity_score = np.exp(-2 * cv)
        regularity_score = np.clip(regularity_score, 0, 1)
        scores.append(regularity_score)
    
    # 2. Signal-to-noise ratio
    # Estimate noise from high-frequency components
    if fs > 80:
        # High-pass filter at 40 Hz to get noise
        sos = signal.butter(4, 40, btype='high', fs=fs, output='sos')
        noise = signal.sosfilt(sos, ecg)
        noise_power = np.var(noise)
        signal_power = np.var(ecg)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        # Convert to score
        snr_score = 1 / (1 + np.exp(-(snr - 10) / 5))  # Sigmoid centered at 10 dB
        scores.append(snr_score)
    
    # 3. Baseline wander assessment
    if len(ecg) > int(fs * 2):  # Need at least 2 seconds
        # Low-pass filter at 0.5 Hz to get baseline
        sos = signal.butter(4, 0.5, btype='low', fs=fs, output='sos')
        baseline = signal.sosfilt(sos, ecg)
        baseline_var = np.var(baseline)
        # Convert to score (lower variance is better)
        baseline_score = np.exp(-baseline_var / (np.var(ecg) + 1e-10))
        baseline_score = np.clip(baseline_score, 0, 1)
        scores.append(baseline_score)
    
    # 4. Peak amplitude consistency
    if len(peaks) > 0:
        peak_amplitudes = ecg[peaks]
        if len(peak_amplitudes) > 1:
            # Coefficient of variation of peak amplitudes
            amp_cv = np.std(peak_amplitudes) / (np.mean(np.abs(peak_amplitudes)) + 1e-10)
            amp_score = np.exp(-2 * amp_cv)
            amp_score = np.clip(amp_score, 0, 1)
            scores.append(amp_score)
    
    # Combine scores
    if len(scores) > 0:
        sqi = np.mean(scores)
    else:
        sqi = 0.0
    
    return float(sqi)


def template_corr(
    ecg: np.ndarray,
    peaks: np.ndarray,
    fs: float = 125.0,
    window_ms: int = 200
) -> float:
    """Compute template correlation for ECG beats.
    
    Args:
        ecg: ECG signal array.
        peaks: R-peak indices.
        fs: Sampling frequency.
        window_ms: Window around R-peak in milliseconds.
        
    Returns:
        Mean correlation coefficient between beats and template.
        
    Note:
        Higher correlation indicates more consistent beat morphology.
    """
    if len(peaks) < 3:
        return 0.0
    
    window_samples = int(window_ms * fs / 1000)
    half_window = window_samples // 2
    
    # Extract beats around R-peaks
    beats = []
    for peak in peaks:
        start = peak - half_window
        end = peak + half_window
        
        if start >= 0 and end < len(ecg):
            beat = ecg[start:end]
            if len(beat) == window_samples:
                beats.append(beat)
    
    if len(beats) < 2:
        return 0.0
    
    beats = np.array(beats)
    
    # Compute template as median beat
    template = np.median(beats, axis=0)
    
    # Compute correlation of each beat with template
    correlations = []
    for beat in beats:
        corr = np.corrcoef(beat, template)[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)
    
    if len(correlations) > 0:
        mean_corr = np.mean(correlations)
        # Ensure positive correlation
        mean_corr = max(0, mean_corr)
    else:
        mean_corr = 0.0
    
    return float(mean_corr)


def ppg_ssqi(
    ppg: np.ndarray,
    fs: float = 125.0
) -> float:
    """Compute PPG Skewness-based Signal Quality Index (sSQI).
    
    Args:
        ppg: PPG signal array.
        fs: Sampling frequency.
        
    Returns:
        sSQI score between 0 and 1.
        
    Note:
        Based on the skewness of PPG signal distribution.
        Good quality PPG typically has negative skewness.
    """
    if len(ppg) < int(fs):  # Need at least 1 second
        return 0.0
    
    # Remove DC component
    ppg_ac = ppg - np.mean(ppg)
    
    # Calculate skewness
    skew = stats.skew(ppg_ac)
    
    # Good PPG has negative skewness (typically -0.5 to -2)
    # Convert to quality score
    if skew < 0:
        # Negative skewness is good
        ssqi = 1 / (1 + np.exp(2 * (skew + 2)))  # Sigmoid centered at -2
    else:
        # Positive skewness is bad
        ssqi = np.exp(-skew)
    
    # Additional checks
    
    # 1. Check for sufficient signal variation
    if np.std(ppg_ac) < 0.01 * np.max(np.abs(ppg_ac)):
        ssqi *= 0.5  # Penalize flat signals
    
    # 2. Check for clipping
    max_val = np.max(ppg)
    min_val = np.min(ppg)
    n_max = np.sum(ppg == max_val)
    n_min = np.sum(ppg == min_val)
    
    clipping_ratio = (n_max + n_min) / len(ppg)
    if clipping_ratio > 0.01:  # More than 1% clipped
        ssqi *= (1 - clipping_ratio)
    
    return float(np.clip(ssqi, 0, 1))


def ppg_abp_corr(
    ppg: np.ndarray,
    abp: np.ndarray,
    fs: float = 125.0,
    max_lag_s: float = 0.5
) -> float:
    """Compute correlation between PPG and ABP signals.
    
    Args:
        ppg: PPG signal array.
        abp: ABP signal array.
        fs: Sampling frequency.
        max_lag_s: Maximum lag to search for correlation.
        
    Returns:
        Maximum correlation coefficient.
        
    Note:
        Accounts for pulse transit time delay between signals.
    """
    if len(ppg) != len(abp):
        # Signals must be same length
        min_len = min(len(ppg), len(abp))
        ppg = ppg[:min_len]
        abp = abp[:min_len]
    
    if len(ppg) < int(fs):  # Need at least 1 second
        return 0.0
    
    # Normalize signals
    ppg_norm = (ppg - np.mean(ppg)) / (np.std(ppg) + 1e-10)
    abp_norm = (abp - np.mean(abp)) / (np.std(abp) + 1e-10)
    
    # Compute cross-correlation
    max_lag_samples = int(max_lag_s * fs)
    
    # Use shorter segment for faster computation
    segment_len = min(len(ppg_norm), int(10 * fs))  # Max 10 seconds
    ppg_segment = ppg_norm[:segment_len]
    abp_segment = abp_norm[:segment_len]
    
    # Compute correlation at different lags
    correlations = []
    for lag in range(-max_lag_samples, max_lag_samples + 1):
        if lag < 0:
            # PPG leads ABP
            ppg_shift = ppg_segment[-lag:]
            abp_shift = abp_segment[:len(ppg_shift)]
        else:
            # ABP leads PPG
            abp_shift = abp_segment[lag:]
            ppg_shift = ppg_segment[:len(abp_shift)]
        
        if len(ppg_shift) > int(fs/2):  # At least 0.5 seconds
            corr = np.corrcoef(ppg_shift, abp_shift)[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
    
    if len(correlations) > 0:
        max_corr = np.max(correlations)
        # Ensure positive correlation
        max_corr = max(0, max_corr)
    else:
        max_corr = 0.0
    
    return float(max_corr)


def hard_artifacts(
    x: np.ndarray,
    fs: float = 125.0,
    flatline_s: float = 2.0,
    saturation_run: int = 30,
    identical_samples: int = 5
) -> Dict[str, bool]:
    """Detect hard artifacts in signal.
    
    Args:
        x: Signal array.
        fs: Sampling frequency.
        flatline_s: Maximum flatline duration in seconds.
        saturation_run: Maximum consecutive saturated samples.
        identical_samples: Maximum consecutive identical samples.
        
    Returns:
        Dictionary with artifact flags:
        - flatline: True if flatline > threshold
        - saturation: True if saturation run > threshold
        - identical: True if > threshold identical consecutive samples
    """
    artifacts = {
        'flatline': False,
        'saturation': False,
        'identical': False
    }
    
    if len(x) == 0:
        return artifacts
    
    # 1. Flatline detection
    flatline_samples = int(flatline_s * fs)
    if len(x) >= flatline_samples:
        # Use rolling window to check for flat segments
        for i in range(len(x) - flatline_samples + 1):
            segment = x[i:i + flatline_samples]
            if np.std(segment) < 1e-6:  # Essentially flat
                artifacts['flatline'] = True
                break
    
    # 2. Saturation detection
    # Assume saturation at min/max values
    max_val = np.max(x)
    min_val = np.min(x)
    
    # Count consecutive saturated samples
    for val in [max_val, min_val]:
        saturated = (x == val).astype(int)
        if np.any(saturated):
            # Find runs of saturated values
            runs = np.diff(np.concatenate(([0], saturated, [0])))
            run_starts = np.where(runs == 1)[0]
            run_ends = np.where(runs == -1)[0]
            run_lengths = run_ends - run_starts
            
            if len(run_lengths) > 0 and np.max(run_lengths) > saturation_run:
                artifacts['saturation'] = True
                break
    
    # 3. Identical consecutive samples
    if len(x) > identical_samples:
        # Check for runs of identical values
        diffs = np.diff(x)
        zero_runs = (diffs == 0).astype(int)
        
        if np.any(zero_runs):
            # Find runs of zeros (identical consecutive values)
            runs = np.diff(np.concatenate(([0], zero_runs, [0])))
            run_starts = np.where(runs == 1)[0]
            run_ends = np.where(runs == -1)[0]
            run_lengths = run_ends - run_starts
            
            if len(run_lengths) > 0 and np.max(run_lengths) > identical_samples:
                artifacts['identical'] = True
    
    return artifacts


def window_accept(
    ecg: Optional[np.ndarray] = None,
    ppg: Optional[np.ndarray] = None,
    abp: Optional[np.ndarray] = None,
    ecg_peaks: Optional[np.ndarray] = None,
    ppg_peaks: Optional[np.ndarray] = None,
    fs: float = 125.0,
    window_s: float = 10.0,
    ecg_sqi_min: float = 0.9,
    template_corr_min: float = 0.8,
    ppg_ssqi_min: float = 0.8,
    ppg_abp_corr_min: float = 0.9,
    min_cycles: int = 3,
    check_artifacts: bool = True
) -> Tuple[bool, List[str]]:
    """Determine if a signal window passes quality gates.
    
    Args:
        ecg: ECG signal window (optional).
        ppg: PPG signal window (optional).
        abp: ABP signal window (optional).
        ecg_peaks: Detected ECG R-peaks (optional).
        ppg_peaks: Detected PPG peaks (optional).
        fs: Sampling frequency.
        window_s: Window duration in seconds.
        ecg_sqi_min: Minimum ECG SQI threshold.
        template_corr_min: Minimum template correlation.
        ppg_ssqi_min: Minimum PPG sSQI threshold.
        ppg_abp_corr_min: Minimum PPG-ABP correlation.
        min_cycles: Minimum required cardiac cycles.
        check_artifacts: Whether to check for hard artifacts.
        
    Returns:
        Tuple of (accept, reasons):
        - accept: Boolean indicating if window passes all gates
        - reasons: List of failure reasons (empty if accepted)
    """
    reasons = []
    
    # Check ECG quality
    if ecg is not None:
        # Check artifacts
        if check_artifacts:
            ecg_artifacts = hard_artifacts(ecg, fs)
            if ecg_artifacts['flatline']:
                reasons.append("ECG flatline detected")
            if ecg_artifacts['saturation']:
                reasons.append("ECG saturation detected")
            if ecg_artifacts['identical']:
                reasons.append("ECG identical samples detected")
        
        # Check SQI
        if ecg_peaks is not None:
            # Check minimum cycles
            if len(ecg_peaks) < min_cycles:
                reasons.append(f"ECG cycles {len(ecg_peaks)} < {min_cycles}")
            
            # Check SQI
            sqi = ecg_sqi(ecg, ecg_peaks, fs, window_s)
            if sqi < ecg_sqi_min:
                reasons.append(f"ECG SQI {sqi:.2f} < {ecg_sqi_min}")
            
            # Check template correlation
            corr = template_corr(ecg, ecg_peaks, fs)
            if corr < template_corr_min:
                reasons.append(f"ECG template correlation {corr:.2f} < {template_corr_min}")
    
    # Check PPG quality
    if ppg is not None:
        # Check artifacts
        if check_artifacts:
            ppg_artifacts = hard_artifacts(ppg, fs)
            if ppg_artifacts['flatline']:
                reasons.append("PPG flatline detected")
            if ppg_artifacts['saturation']:
                reasons.append("PPG saturation detected")
            if ppg_artifacts['identical']:
                reasons.append("PPG identical samples detected")
        
        # Check minimum cycles
        if ppg_peaks is not None and len(ppg_peaks) < min_cycles:
            reasons.append(f"PPG cycles {len(ppg_peaks)} < {min_cycles}")
        
        # Check sSQI
        ssqi = ppg_ssqi(ppg, fs)
        if ssqi < ppg_ssqi_min:
            reasons.append(f"PPG sSQI {ssqi:.2f} < {ppg_ssqi_min}")
        
        # Check PPG-ABP correlation if both present
        if abp is not None:
            corr = ppg_abp_corr(ppg, abp, fs)
            if corr < ppg_abp_corr_min:
                reasons.append(f"PPG-ABP correlation {corr:.2f} < {ppg_abp_corr_min}")
    
    # Check ABP quality
    if abp is not None:
        # Check artifacts
        if check_artifacts:
            abp_artifacts = hard_artifacts(abp, fs)
            if abp_artifacts['flatline']:
                reasons.append("ABP flatline detected")
            if abp_artifacts['saturation']:
                reasons.append("ABP saturation detected")
            if abp_artifacts['identical']:
                reasons.append("ABP identical samples detected")
    
    # Window is accepted if no failure reasons
    accept = len(reasons) == 0
    
    return accept, reasons


def compute_quality_metrics(
    ecg: Optional[np.ndarray] = None,
    ppg: Optional[np.ndarray] = None,
    abp: Optional[np.ndarray] = None,
    ecg_peaks: Optional[np.ndarray] = None,
    ppg_peaks: Optional[np.ndarray] = None,
    fs: float = 125.0
) -> Dict[str, float]:
    """Compute all quality metrics for signals.
    
    Args:
        ecg: ECG signal (optional).
        ppg: PPG signal (optional).
        abp: ABP signal (optional).
        ecg_peaks: ECG R-peaks (optional).
        ppg_peaks: PPG peaks (optional).
        fs: Sampling frequency.
        
    Returns:
        Dictionary of quality metrics.
    """
    metrics = {}
    
    if ecg is not None and ecg_peaks is not None:
        metrics['ecg_sqi'] = ecg_sqi(ecg, ecg_peaks, fs)
        metrics['ecg_template_corr'] = template_corr(ecg, ecg_peaks, fs)
        metrics['ecg_peak_count'] = len(ecg_peaks)
    
    if ppg is not None:
        metrics['ppg_ssqi'] = ppg_ssqi(ppg, fs)
        if ppg_peaks is not None:
            metrics['ppg_peak_count'] = len(ppg_peaks)
    
    if ppg is not None and abp is not None:
        metrics['ppg_abp_corr'] = ppg_abp_corr(ppg, abp, fs)
    
    return metrics
