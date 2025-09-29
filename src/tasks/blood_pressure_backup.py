"""Blood Pressure Estimation from PPG.

Estimates SBP/DBP from PPG waveform using ABP as ground truth.
Benchmark: SBP MAE 2.16 mmHg, DBP MAE 1.12 mmHg (AAMI Grade A)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

from .base import RegressionTask, TaskConfig, Benchmark


class BloodPressureEstimationTask(RegressionTask):
    """Estimate blood pressure (SBP/DBP) from PPG waveform.
    
    Clinical Standards:
        - AAMI: ME ≤5 mmHg, SD ≤8 mmHg (tested on ≥85 subjects)
        - BHS Grade A: ME ≤5 mmHg, SD ≤8 mmHg
        - BHS Grade B: ME ≤10 mmHg, SD ≤15 mmHg
        
    Published Benchmarks:
        - PulseDB (Wang et al. 2023): 5.2M segments, 3458 VitalDB cases
        - Pan et al. (2024): SBP 2.16±1.53 mmHg, DBP 1.12±0.59 mmHg (AAMI Grade A)
        
    Label Generation:
        - SBP: Amplitude at systolic peaks (Elgendi algorithm)
        - DBP: Amplitude at diastolic troughs (turning points)
        - Quality control: PPG-ABP correlation > 0.9
    """
    
    def __init__(
        self,
        target: str = 'both',  # 'sbp', 'dbp', or 'both'
        sampling_rate: float = 125.0,
        segment_duration_s: float = 10.0,
        ppg_abp_corr_threshold: float = 0.9
    ):
        assert target in ['sbp', 'dbp', 'both', 'map']
        
        target_dim = 2 if target == 'both' else 1
        
        config = TaskConfig(
            name=f"blood_pressure_{target}",
            task_type="regression",
            target_dim=target_dim,
            window_size_s=segment_duration_s,
            sampling_rate=sampling_rate,
            required_channels=['PLETH', 'ART'],  # PPG and arterial waveform
            min_sqi=0.8
        )
        super().__init__(config)
        
        self.target = target
        self.ppg_abp_corr_threshold = ppg_abp_corr_threshold
    
    def _load_benchmarks(self):
        """Load published benchmarks for BP estimation."""
        self.benchmarks = [
            Benchmark(
                paper="PulseDB (Wang et al.)",
                year=2023,
                dataset="VitalDB+MIMIC-III",
                n_patients=3458,
                metrics={
                    'sbp_mae': 5.2,
                    'dbp_mae': 3.1,
                    'n_segments': 5_200_000
                },
                method="Cleaned 10-second segments",
                notes="Definitive benchmark dataset"
            ),
            Benchmark(
                paper="Pan et al.",
                year=2024,
                dataset="VitalDB",
                n_patients=1200,
                metrics={
                    'sbp_mae': 2.16,
                    'sbp_std': 1.53,
                    'dbp_mae': 1.12,
                    'dbp_std': 0.59
                },
                method="Continuous waveform reconstruction",
                notes="Best VitalDB performance, AAMI Grade A"
            ),
            Benchmark(
                paper="AAMI Standard",
                year=2013,
                dataset="Standard",
                n_patients=85,
                metrics={
                    'sbp_me': 5.0,
                    'sbp_sd': 8.0,
                    'dbp_me': 5.0,
                    'dbp_sd': 8.0
                },
                method="Regulatory requirement",
                notes="Minimum for clinical use"
            ),
            Benchmark(
                paper="BHS Grade A",
                year=1993,
                dataset="Standard",
                n_patients=85,
                metrics={
                    'sbp_me': 5.0,
                    'sbp_sd': 8.0,
                    'dbp_me': 5.0,
                    'dbp_sd': 8.0
                },
                method="British Hypertension Society",
                notes="Gold standard"
            )
        ]
    
    def generate_labels(
        self,
        case_id: str,
        signals: Dict[str, np.ndarray],
        clinical_data: pd.Series,
        fs: float = 125.0
    ) -> Dict[str, np.ndarray]:
        """Generate SBP/DBP labels from ABP waveform using Elgendi algorithm.
        
        Label Generation Algorithm:
        1. Detect systolic peaks in ABP waveform
        2. Detect diastolic troughs (turning points) in ABP
        3. Extract SBP as amplitude at systolic peaks
        4. Extract DBP as amplitude at diastolic troughs
        5. Quality control: PPG-ABP correlation > threshold
        
        Args:
            case_id: VitalDB case identifier
            signals: Dictionary containing 'PLETH' (PPG) and 'ART' (ABP)
            clinical_data: Clinical parameters
            fs: Sampling frequency
            
        Returns:
            Dictionary with 'sbp', 'dbp', 'map' arrays (one value per cardiac cycle)
        """
        if 'PLETH' not in signals or 'ART' not in signals:
            raise ValueError("Both PLETH and ART signals required for BP estimation")
        
        ppg = signals['PLETH']
        abp = signals['ART']
        
        # Quality check: PPG-ABP correlation
        corr = self._compute_ppg_abp_correlation(ppg, abp, fs)
        if corr < self.ppg_abp_corr_threshold:
            raise ValueError(
                f"PPG-ABP correlation {corr:.3f} below threshold "
                f"{self.ppg_abp_corr_threshold}"
            )
        
        # Detect peaks and troughs in ABP
        systolic_peaks = self._detect_systolic_peaks(abp, fs)
        diastolic_troughs = self._detect_diastolic_troughs(abp, fs)
        
        # Extract BP values
        sbp_values = abp[systolic_peaks]
        dbp_values = abp[diastolic_troughs]
        
        # Align peaks and troughs to create beat-by-beat BP
        sbp_aligned, dbp_aligned = self._align_bp_values(
            sbp_values, dbp_values, systolic_peaks, diastolic_troughs
        )
        
        # Compute MAP
        map_aligned = dbp_aligned + (sbp_aligned - dbp_aligned) / 3
        
        labels = {
            'sbp': sbp_aligned,
            'dbp': dbp_aligned,
            'map': map_aligned,
            'systolic_peaks': systolic_peaks,
            'diastolic_troughs': diastolic_troughs,
            'ppg_abp_correlation': corr
        }
        
        return labels
    
    def _detect_systolic_peaks(
        self,
        abp: np.ndarray,
        fs: float
    ) -> np.ndarray:
        """Detect systolic peaks in ABP waveform.
        
        Uses scipy.signal.find_peaks with physiologically informed parameters.
        
        Args:
            abp: ABP waveform
            fs: Sampling frequency
            
        Returns:
            Array of peak indices
        """
        # Minimum distance between peaks (40 bpm = 1.5 seconds)
        min_distance = int(1.5 * fs)
        
        # Find peaks with minimum distance constraint
        peaks, properties = scipy_signal.find_peaks(
            abp,
            distance=min_distance,
            prominence=10,  # Minimum 10 mmHg prominence
            height=60  # Minimum 60 mmHg (typical diastolic minimum)
        )
        
        return peaks
    
    def _detect_diastolic_troughs(
        self,
        abp: np.ndarray,
        fs: float
    ) -> np.ndarray:
        """Detect diastolic troughs in ABP waveform.
        
        Uses inverted peak detection (finding minima).
        
        Args:
            abp: ABP waveform
            fs: Sampling frequency
            
        Returns:
            Array of trough indices
        """
        # Invert signal to find troughs as peaks
        inverted_abp = -abp
        
        min_distance = int(1.5 * fs)
        
        troughs, properties = scipy_signal.find_peaks(
            inverted_abp,
            distance=min_distance,
            prominence=10
        )
        
        return troughs
    
    def _align_bp_values(
        self,
        sbp_values: np.ndarray,
        dbp_values: np.ndarray,
        systolic_peaks: np.ndarray,
        diastolic_troughs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Align SBP and DBP values beat-by-beat.
        
        Each cardiac cycle should have one systolic peak and one diastolic trough.
        This function pairs them appropriately.
        
        Args:
            sbp_values: SBP values at peaks
            dbp_values: DBP values at troughs
            systolic_peaks: Indices of systolic peaks
            diastolic_troughs: Indices of diastolic troughs
            
        Returns:
            Tuple of (sbp_aligned, dbp_aligned)
        """
        # For each systolic peak, find the nearest preceding diastolic trough
        n_beats = len(systolic_peaks)
        sbp_aligned = np.zeros(n_beats)
        dbp_aligned = np.zeros(n_beats)
        
        for i, peak_idx in enumerate(systolic_peaks):
            sbp_aligned[i] = sbp_values[i]
            
            # Find nearest trough before this peak
            troughs_before = diastolic_troughs[diastolic_troughs < peak_idx]
            if len(troughs_before) > 0:
                trough_idx = troughs_before[-1]
                trough_position = np.where(diastolic_troughs == trough_idx)[0][0]
                dbp_aligned[i] = dbp_values[trough_position]
            else:
                # Use first trough if no trough before peak
                dbp_aligned[i] = dbp_values[0] if len(dbp_values) > 0 else 60.0
        
        return sbp_aligned, dbp_aligned
    
    def _compute_ppg_abp_correlation(
        self,
        ppg: np.ndarray,
        abp: np.ndarray,
        fs: float,
        max_lag_s: float = 0.5
    ) -> float:
        """Compute correlation between PPG and ABP with lag compensation.
        
        Args:
            ppg: PPG signal
            abp: ABP signal
            fs: Sampling frequency
            max_lag_s: Maximum lag to search
            
        Returns:
            Maximum correlation coefficient
        """
        # Normalize signals
        ppg_norm = (ppg - np.mean(ppg)) / (np.std(ppg) + 1e-10)
        abp_norm = (abp - np.mean(abp)) / (np.std(abp) + 1e-10)
        
        # Compute cross-correlation with lags
        max_lag_samples = int(max_lag_s * fs)
        
        correlations = []
        for lag in range(-max_lag_samples, max_lag_samples + 1):
            if lag < 0:
                ppg_shift = ppg_norm[-lag:]
                abp_shift = abp_norm[:len(ppg_shift)]
            else:
                abp_shift = abp_norm[lag:]
                ppg_shift = ppg_norm[:len(abp_shift)]
            
            if len(ppg_shift) > int(fs):  # At least 1 second
                corr = np.corrcoef(ppg_shift, abp_shift)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        
        return max(correlations) if correlations else 0.0
    
    def evaluate(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        return_detailed: bool = False
    ) -> Dict[str, float]:
        """Evaluate BP estimation using AAMI/BHS standards.
        
        Args:
            predictions: Predicted BP values [n_samples, n_targets]
            targets: Ground truth BP values [n_samples, n_targets]
            return_detailed: Whether to return detailed metrics
            
        Returns:
            Dictionary of evaluation metrics including AAMI compliance
        """
        # Standard regression metrics
        metrics = super().evaluate(predictions, targets, return_detailed)
        
        # AAMI/BHS specific metrics
        if predictions.ndim > 1:
            # Multi-output (SBP and DBP)
            for i, bp_type in enumerate(['sbp', 'dbp']):
                pred = predictions[:, i]
                target = targets[:, i]
                
                # Mean error and standard deviation
                errors = pred - target
                me = np.mean(errors)
                sd = np.std(errors)
                
                metrics[f'{bp_type}_me'] = me
                metrics[f'{bp_type}_sd'] = sd
                metrics[f'{bp_type}_mae'] = np.mean(np.abs(errors))
                
                # AAMI compliance
                aami_compliant = (abs(me) <= 5.0) and (sd <= 8.0)
                metrics[f'{bp_type}_aami_compliant'] = aami_compliant
                
                # BHS grade
                bhs_grade = self._compute_bhs_grade(errors)
                metrics[f'{bp_type}_bhs_grade'] = bhs_grade
        else:
            # Single output
            errors = predictions - targets
            me = np.mean(errors)
            sd = np.std(errors)
            
            metrics['me'] = me
            metrics['sd'] = sd
            metrics['aami_compliant'] = (abs(me) <= 5.0) and (sd <= 8.0)
            metrics['bhs_grade'] = self._compute_bhs_grade(errors)
        
        return metrics
    
    def _compute_bhs_grade(self, errors: np.ndarray) -> str:
        """Compute British Hypertension Society grade.
        
        Grade criteria based on cumulative percentage of errors:
        - Grade A: ≤5 mmHg (60%), ≤10 mmHg (85%), ≤15 mmHg (95%)
        - Grade B: ≤5 mmHg (50%), ≤10 mmHg (75%), ≤15 mmHg (90%)
        - Grade C: ≤5 mmHg (40%), ≤10 mmHg (65%), ≤15 mmHg (85%)
        - Grade D: Worse than Grade C
        
        Args:
            errors: Array of prediction errors
            
        Returns:
            BHS grade ('A', 'B', 'C', or 'D')
        """
        abs_errors = np.abs(errors)
        
        pct_5 = 100 * np.mean(abs_errors <= 5)
        pct_10 = 100 * np.mean(abs_errors <= 10)
        pct_15 = 100 * np.mean(abs_errors <= 15)
        
        if pct_5 >= 60 and pct_10 >= 85 and pct_15 >= 95:
            return 'A'
        elif pct_5 >= 50 and pct_10 >= 75 and pct_15 >= 90:
            return 'B'
        elif pct_5 >= 40 and pct_10 >= 65 and pct_15 >= 85:
            return 'C'
        else:
            return 'D'
    
    def extract_ppg_features(
        self,
        ppg: np.ndarray,
        fs: float = 125.0
    ) -> Dict[str, float]:
        """Extract features from PPG waveform for BP estimation.
        
        Common features used in BP estimation:
        - Pulse arrival time (PAT)
        - Pulse transit time (PTT)
        - Pulse wave velocity (PWV)
        - PPG morphological features (amplitude, width, area)
        
        Args:
            ppg: PPG waveform
            fs: Sampling frequency
            
        Returns:
            Dictionary of PPG features
        """
        features = {}
        
        # Detect PPG peaks
        min_distance = int(0.5 * fs)  # Minimum 0.5s between peaks
        peaks, properties = scipy_signal.find_peaks(
            ppg,
            distance=min_distance,
            prominence=0.1 * (np.max(ppg) - np.min(ppg))
        )
        
        if len(peaks) < 2:
            # Not enough peaks, return zeros
            return {k: 0.0 for k in [
                'ppg_amplitude_mean', 'ppg_amplitude_std',
                'ppg_pulse_rate', 'ppg_pulse_interval_std'
            ]}
        
        # Pulse amplitude
        peak_amplitudes = ppg[peaks]
        features['ppg_amplitude_mean'] = np.mean(peak_amplitudes)
        features['ppg_amplitude_std'] = np.std(peak_amplitudes)
        features['ppg_amplitude_cv'] = (
            features['ppg_amplitude_std'] / (features['ppg_amplitude_mean'] + 1e-6)
        )
        
        # Pulse rate
        pulse_intervals = np.diff(peaks) / fs
        features['ppg_pulse_rate'] = 60.0 / np.mean(pulse_intervals)
        features['ppg_pulse_interval_std'] = np.std(pulse_intervals)
        
        # Pulse width (at 50% amplitude)
        # Area under curve
        
        return features
