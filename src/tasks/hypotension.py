"""Intraoperative Hypotension Prediction Task.

Predicts MAP < 65 mmHg sustained for ≥60 seconds.
Benchmark: AUROC 0.90-0.93, AUPRC 0.70-0.88
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import signal

from .base import ClassificationTask, TaskConfig, Benchmark


class HypotensionPredictionTask(ClassificationTask):
    """Predict intraoperative hypotension 5-15 minutes before onset.
    
    Clinical Definition:
        - MAP < 65 mmHg sustained for ≥60 seconds
        - Prediction windows: 5, 10, or 15 minutes before onset
        
    Published Benchmarks:
        - STEP-OP (Choe et al. 2021): AUPRC 0.716, 18,813 patients
        - Jo et al. (2022): AUROC 0.935, AUPRC 0.882, ABP+EEG
        - Jeong et al. (2024): AUROC 0.917, non-invasive (ECG+PPG+Cap+BIS)
    """
    
    def __init__(
        self,
        prediction_window_min: int = 5,
        map_threshold: float = 65.0,
        sustained_duration_s: float = 60.0,
        sampling_rate: float = 2.0  # 2 Hz for MAP (2-second intervals)
    ):
        config = TaskConfig(
            name=f"hypotension_prediction_{prediction_window_min}min",
            task_type="classification",
            num_classes=2,
            clinical_threshold=map_threshold,
            prediction_window_s=prediction_window_min * 60,
            window_size_s=10.0,  # Input window size
            sampling_rate=sampling_rate,
            required_channels=['ART_MBP'],  # Mean arterial pressure
            min_sqi=0.8
        )
        super().__init__(config)
        
        self.prediction_window_min = prediction_window_min
        self.map_threshold = map_threshold
        self.sustained_duration_s = sustained_duration_s
        self.map_sampling_rate = sampling_rate
    
    def _load_benchmarks(self):
        """Load published benchmarks for hypotension prediction."""
        self.benchmarks = [
            Benchmark(
                paper="STEP-OP (Choe et al.)",
                year=2021,
                dataset="VitalDB",
                n_patients=18813,
                metrics={'auprc': 0.716, 'auroc': 0.90},
                method="Weighted CNN-RNN ensemble",
                notes="Most rigorous label generation"
            ),
            Benchmark(
                paper="Jo et al.",
                year=2022,
                dataset="VitalDB",
                n_patients=5230,
                metrics={'auroc': 0.935, 'auprc': 0.882},
                method="ABP waveform + EEG",
                notes="5-minute prediction window"
            ),
            Benchmark(
                paper="Jeong et al.",
                year=2024,
                dataset="VitalDB",
                n_patients=3200,
                metrics={'auroc': 0.917, 'auprc': 0.85},
                method="Non-invasive (ECG+PPG+Cap+BIS)",
                notes="No arterial line needed"
            ),
            Benchmark(
                paper="Target Performance",
                year=2025,
                dataset="VitalDB",
                n_patients=0,
                metrics={'auroc': 0.90, 'auprc': 0.70},
                method="Minimum acceptable performance",
                notes="Clinical deployment threshold"
            )
        ]
    
    def generate_labels(
        self,
        case_id: str,
        signals: Dict[str, np.ndarray],
        clinical_data: pd.Series,
        fs: float = 2.0
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Generate hypotension labels using consensus definition.
        
        Label Generation Algorithm:
        1. Extract MAP values (2-second intervals from Solar8000/ART_MBP)
        2. Identify all 1-minute intervals where MAP < 65 mmHg
        3. Group consecutive hypotensive minutes into episodes
        4. Create prediction windows N minutes before each episode
        5. Label windows as positive (1) if hypotension occurs within prediction window
        
        Args:
            case_id: VitalDB case identifier
            signals: Dictionary containing 'ART_MBP' (mean arterial pressure)
            clinical_data: Clinical parameters (not used for this task)
            fs: Sampling frequency for MAP (default 2 Hz)
            
        Returns:
            Tuple of (labels, episodes):
                - labels: Binary array [n_windows] indicating hypotension risk
                - episodes: List of hypotension episode dicts with start/end times
        """
        if 'ART_MBP' not in signals:
            raise ValueError("ART_MBP signal required for hypotension prediction")
        
        map_signal = signals['ART_MBP']
        
        # Step 1: Identify hypotensive samples (MAP < threshold)
        hypotensive_mask = map_signal < self.map_threshold
        
        # Step 2: Find sustained hypotension (≥60 seconds)
        sustained_samples = int(self.sustained_duration_s * fs)
        episodes = self._find_hypotension_episodes(
            hypotensive_mask, sustained_samples, fs
        )
        
        # Step 3: Create prediction windows
        prediction_samples = int(self.prediction_window_min * 60 * fs)
        labels, valid_windows = self._create_prediction_labels(
            map_signal, episodes, prediction_samples, fs
        )
        
        return labels, episodes
    
    def _find_hypotension_episodes(
        self,
        hypotensive_mask: np.ndarray,
        min_duration_samples: int,
        fs: float
    ) -> List[Dict]:
        """Find sustained hypotension episodes.
        
        Args:
            hypotensive_mask: Boolean array of hypotensive samples
            min_duration_samples: Minimum episode duration in samples
            fs: Sampling frequency
            
        Returns:
            List of episode dictionaries with start_time, end_time, duration
        """
        episodes = []
        
        # Find runs of hypotensive samples
        changes = np.diff(np.concatenate(([0], hypotensive_mask.astype(int), [0])))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        
        for start, end in zip(starts, ends):
            duration = end - start
            if duration >= min_duration_samples:
                episodes.append({
                    'start_idx': int(start),
                    'end_idx': int(end),
                    'start_time_s': float(start / fs),
                    'end_time_s': float(end / fs),
                    'duration_s': float(duration / fs),
                    'min_map': float(np.min(hypotensive_mask[start:end]))
                })
        
        return episodes
    
    def _create_prediction_labels(
        self,
        map_signal: np.ndarray,
        episodes: List[Dict],
        prediction_window_samples: int,
        fs: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create binary labels for prediction windows.
        
        Args:
            map_signal: Full MAP signal
            episodes: List of hypotension episodes
            prediction_window_samples: Prediction window size in samples
            fs: Sampling frequency
            
        Returns:
            Tuple of (labels, valid_window_mask):
                - labels: Binary labels [n_samples] indicating future hypotension
                - valid_window_mask: Boolean mask of valid prediction windows
        """
        n_samples = len(map_signal)
        labels = np.zeros(n_samples, dtype=int)
        
        # For each episode, mark the prediction window as positive
        for episode in episodes:
            start_idx = episode['start_idx']
            
            # Mark the prediction window before the episode
            window_start = max(0, start_idx - prediction_window_samples)
            window_end = start_idx
            
            if window_end > window_start:
                labels[window_start:window_end] = 1
        
        # Valid windows exclude the very start and end of the recording
        # and exclude periods already in hypotension
        valid_mask = np.ones(n_samples, dtype=bool)
        valid_mask[:prediction_window_samples] = False  # Can't predict at start
        valid_mask[-prediction_window_samples:] = False  # Can't predict at end
        
        # Exclude periods during hypotension
        for episode in episodes:
            valid_mask[episode['start_idx']:episode['end_idx']] = False
        
        return labels, valid_mask
    
    def extract_clinical_features(
        self,
        map_signal: np.ndarray,
        window_size_s: float = 300.0,  # 5 minutes
        fs: float = 2.0
    ) -> Dict[str, float]:
        """Extract clinical features from MAP signal for prediction.
        
        Features commonly used in hypotension prediction:
        - Mean, std, min, max MAP
        - MAP trend (slope)
        - MAP variability
        - Time below threshold
        
        Args:
            map_signal: MAP signal window
            window_size_s: Window size in seconds
            fs: Sampling frequency
            
        Returns:
            Dictionary of clinical features
        """
        features = {}
        
        # Basic statistics
        features['map_mean'] = np.mean(map_signal)
        features['map_std'] = np.std(map_signal)
        features['map_min'] = np.min(map_signal)
        features['map_max'] = np.max(map_signal)
        features['map_range'] = features['map_max'] - features['map_min']
        
        # Trend (linear regression slope)
        time_points = np.arange(len(map_signal))
        if len(map_signal) > 1:
            slope, _ = np.polyfit(time_points, map_signal, 1)
            features['map_trend'] = slope
        else:
            features['map_trend'] = 0.0
        
        # Variability
        if len(map_signal) > 1:
            features['map_cv'] = features['map_std'] / (features['map_mean'] + 1e-6)
        else:
            features['map_cv'] = 0.0
        
        # Time below clinical threshold
        below_threshold = map_signal < self.map_threshold
        features['time_below_threshold_pct'] = 100 * np.mean(below_threshold)
        
        # Rate of change
        if len(map_signal) > 1:
            dmap = np.diff(map_signal) * fs  # mmHg/s
            features['map_rate_mean'] = np.mean(np.abs(dmap))
            features['map_rate_max'] = np.max(np.abs(dmap))
        else:
            features['map_rate_mean'] = 0.0
            features['map_rate_max'] = 0.0
        
        return features
    
    def compute_episode_statistics(
        self,
        episodes: List[Dict]
    ) -> Dict[str, float]:
        """Compute statistics about hypotension episodes.
        
        Args:
            episodes: List of hypotension episode dictionaries
            
        Returns:
            Dictionary of episode statistics
        """
        if not episodes:
            return {
                'n_episodes': 0,
                'total_duration_s': 0.0,
                'mean_duration_s': 0.0,
                'max_duration_s': 0.0,
                'episode_rate_per_hour': 0.0
            }
        
        durations = [ep['duration_s'] for ep in episodes]
        
        stats = {
            'n_episodes': len(episodes),
            'total_duration_s': sum(durations),
            'mean_duration_s': np.mean(durations),
            'median_duration_s': np.median(durations),
            'std_duration_s': np.std(durations),
            'max_duration_s': max(durations),
            'min_duration_s': min(durations)
        }
        
        # Episode rate (assumes episodes span the full recording)
        # Try to get recording duration from last episode's end_time_s
        if episodes and 'end_time_s' in episodes[-1]:
            recording_duration_s = episodes[-1]['end_time_s']
            if recording_duration_s > 0:
                stats['episode_rate_per_hour'] = (
                    len(episodes) / recording_duration_s * 3600
                )
        else:
            # If end_time_s not available, can't compute episode rate
            stats['episode_rate_per_hour'] = 0.0
        
        return stats


def load_map_from_vitaldb(
    case_id: str,
    start_sec: float = 0,
    duration_sec: Optional[float] = None,
    use_cache: bool = True
) -> Tuple[np.ndarray, float]:
    """Load MAP signal from VitalDB.
    
    Args:
        case_id: VitalDB case identifier
        start_sec: Start time in seconds
        duration_sec: Duration to load (None = full case)
        use_cache: Whether to use cached data
        
    Returns:
        Tuple of (map_signal, fs)
    """
    import vitaldb
    
    try:
        case_id = int(case_id)
    except:
        pass
    
    # Try Solar8000/ART_MBP first (2-second intervals)
    try:
        vf = vitaldb.VitalFile(case_id)
        tracks = vf.get_track_names()
        
        map_track = None
        if 'Solar8000/ART_MBP' in tracks:
            map_track = 'Solar8000/ART_MBP'
            fs = 0.5  # 2-second intervals
        elif 'SNUADC/ART' in tracks:
            # Can compute MAP from arterial waveform
            map_track = 'SNUADC/ART'
            fs = 500.0  # Will need to compute running mean
        
        if map_track is None:
            raise ValueError(f"No MAP track found in case {case_id}")
        
        end_sec = start_sec + duration_sec if duration_sec else start_sec + 3600
        
        data = vf.to_numpy([map_track], start_sec, end_sec)
        
        if data is None or len(data) == 0:
            raise ValueError(f"No data returned for {map_track}")
        
        signal = data[:, 0] if data.ndim == 2 else data
        
        # If arterial waveform, compute running MAP
        if 'SNUADC' in map_track:
            # MAP ≈ DBP + 1/3(SBP - DBP) ≈ running mean with appropriate window
            window_size = int(2 * fs)  # 2-second window
            map_signal = np.convolve(signal, np.ones(window_size)/window_size, mode='valid')
            # Downsample to 0.5 Hz (2-second intervals)
            downsample_factor = int(fs / 0.5)
            map_signal = map_signal[::downsample_factor]
            fs = 0.5
        else:
            map_signal = signal
        
        return map_signal, fs
        
    except Exception as e:
        raise ValueError(f"Error loading MAP from case {case_id}: {e}")
