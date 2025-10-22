"""
Quality Proxy Indicators for Self-Supervised Learning.

This module computes signal quality indicators (SQI) that can be used as
pseudo-labels for contrastive learning without requiring ground-truth quality labels.

These metrics approximate signal quality based on:
- Signal-to-noise ratio
- Peak detection reliability
- Temporal stability
- Frequency domain characteristics

Author: Claude Code
Date: 2025-10-21
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class QualityProxyComputer:
    """
    Computes quality proxy scores for PPG/ECG signals without labels.

    These scores are used to create positive/negative pairs for contrastive SSL:
    - High similarity in quality scores → positive pairs
    - Low similarity in quality scores → negative pairs

    Args:
        fs: Sampling frequency in Hz (default: 125 Hz)
        device: Device to run computations on

    Example:
        >>> qpc = QualityProxyComputer(fs=125)
        >>> signals = torch.randn(32, 2, 1024)  # [batch, channels, time]
        >>> quality_scores = qpc.compute_batch(signals)
        >>> quality_scores.shape
        torch.Size([32])
    """

    def __init__(self, fs: float = 125.0, device: str = 'cpu'):
        self.fs = fs
        self.device = device

        # Quality score weights (tuned for PPG signals)
        self.weights = {
            'snr': 0.3,           # Signal-to-noise ratio
            'stability': 0.25,    # Temporal stability
            'periodicity': 0.25,  # Periodic consistency
            'amplitude': 0.2      # Amplitude characteristics
        }

    def compute_batch(self, signals: torch.Tensor) -> torch.Tensor:
        """
        Compute quality scores for a batch of signals.

        Args:
            signals: Tensor of shape [batch, channels, time]

        Returns:
            Quality scores of shape [batch], range [0, 1]
            Higher score = better quality
        """
        batch_size = signals.size(0)
        quality_scores = []

        for i in range(batch_size):
            # For multi-channel signals, use first channel (PPG)
            signal = signals[i, 0, :]  # [time]
            score = self.compute_single(signal)
            quality_scores.append(score)

        return torch.tensor(quality_scores, device=signals.device)

    def compute_single(self, signal: torch.Tensor) -> float:
        """
        Compute quality score for a single signal.

        Args:
            signal: Tensor of shape [time]

        Returns:
            Quality score in [0, 1]
        """
        # Move to CPU for scipy operations
        signal_np = signal.detach().cpu().numpy()

        # Compute individual quality indicators
        snr_score = self._compute_snr(signal_np)
        stability_score = self._compute_stability(signal_np)
        periodicity_score = self._compute_periodicity(signal_np)
        amplitude_score = self._compute_amplitude_quality(signal_np)

        # Weighted combination
        total_score = (
            self.weights['snr'] * snr_score +
            self.weights['stability'] * stability_score +
            self.weights['periodicity'] * periodicity_score +
            self.weights['amplitude'] * amplitude_score
        )

        return float(np.clip(total_score, 0, 1))

    def _compute_snr(self, signal: np.ndarray) -> float:
        """
        Estimate signal-to-noise ratio using high-frequency content.

        Higher SNR indicates cleaner signal with less noise.
        """
        # Signal power (low-pass: 0.5-5 Hz for PPG)
        nyquist = self.fs / 2
        low_cutoff = 0.5 / nyquist
        high_cutoff = 5.0 / nyquist

        # Simple frequency domain approach
        fft = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), 1/self.fs)

        # Signal band (0.5-5 Hz)
        signal_band = (freqs >= 0.5) & (freqs <= 5.0)
        signal_power = np.sum(np.abs(fft[signal_band])**2)

        # Noise band (> 10 Hz)
        noise_band = freqs > 10.0
        noise_power = np.sum(np.abs(fft[noise_band])**2) + 1e-8

        snr = signal_power / noise_power

        # Normalize to [0, 1] using sigmoid
        snr_score = 1 / (1 + np.exp(-0.1 * (np.log10(snr + 1) - 2)))

        return snr_score

    def _compute_stability(self, signal: np.ndarray) -> float:
        """
        Compute temporal stability using windowed variance.

        Stable signals have consistent variance across time.
        """
        # Divide signal into windows
        window_size = int(self.fs * 2)  # 2-second windows
        num_windows = len(signal) // window_size

        if num_windows < 2:
            return 0.5  # Not enough data

        # Compute variance for each window
        variances = []
        for i in range(num_windows):
            start = i * window_size
            end = start + window_size
            window = signal[start:end]
            variances.append(np.var(window))

        variances = np.array(variances)

        # Coefficient of variation of variances
        # Low CoV = stable signal
        mean_var = np.mean(variances) + 1e-8
        std_var = np.std(variances)
        cov = std_var / mean_var

        # Convert to score: lower CoV = higher quality
        stability_score = np.exp(-cov)

        return np.clip(stability_score, 0, 1)

    def _compute_periodicity(self, signal: np.ndarray) -> float:
        """
        Assess periodic consistency using autocorrelation.

        PPG signals should be periodic (heart beats).
        """
        # Compute autocorrelation
        signal_norm = signal - np.mean(signal)
        autocorr = np.correlate(signal_norm, signal_norm, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Keep positive lags
        autocorr = autocorr / autocorr[0]  # Normalize

        # Find peaks in autocorrelation (should appear at heart rate intervals)
        # Expected heart rate: 0.5-3 Hz → period: 0.33-2 seconds
        min_lag = int(self.fs * 0.33)  # 0.33 sec
        max_lag = int(self.fs * 2.0)   # 2.0 sec

        if max_lag >= len(autocorr):
            return 0.5

        # Find max autocorrelation in expected range
        autocorr_range = autocorr[min_lag:max_lag]
        max_autocorr = np.max(autocorr_range)

        # Higher autocorrelation = more periodic = better quality
        periodicity_score = max_autocorr

        return np.clip(periodicity_score, 0, 1)

    def _compute_amplitude_quality(self, signal: np.ndarray) -> float:
        """
        Assess signal amplitude characteristics.

        Too low = poor contact, too high = saturation
        """
        # Compute signal range after removing DC
        signal_ac = signal - np.mean(signal)
        signal_range = np.max(signal_ac) - np.min(signal_ac)

        # Expected range for normalized PPG: 0.1 to 2.0
        # (signals are typically z-score normalized)
        optimal_range = 1.0

        # Distance from optimal
        range_diff = np.abs(signal_range - optimal_range)

        # Convert to score
        amplitude_score = np.exp(-range_diff)

        return np.clip(amplitude_score, 0, 1)

    def compute_pairwise_similarity(
        self,
        quality_scores: torch.Tensor,
        temperature: float = 0.1
    ) -> torch.Tensor:
        """
        Compute pairwise quality similarity matrix.

        Used to determine positive/negative pairs for contrastive learning.

        Args:
            quality_scores: Tensor of shape [batch]
            temperature: Temperature for softmax (lower = sharper)

        Returns:
            Similarity matrix of shape [batch, batch]
            Values in [0, 1], higher = more similar quality

        Example:
            >>> scores = torch.tensor([0.8, 0.75, 0.3, 0.85])
            >>> sim = qpc.compute_pairwise_similarity(scores)
            >>> # sim[0, 1] high (both good quality)
            >>> # sim[0, 2] low (different quality)
        """
        batch_size = quality_scores.size(0)

        # Compute absolute differences
        scores_expanded = quality_scores.unsqueeze(1)  # [batch, 1]
        diff = torch.abs(scores_expanded - quality_scores.unsqueeze(0))  # [batch, batch]

        # Convert difference to similarity
        # Small difference = high similarity
        similarity = torch.exp(-diff / temperature)

        return similarity


class ContrastivePairSampler:
    """
    Sample positive and negative pairs based on quality scores.

    Args:
        positive_threshold: Quality similarity threshold for positive pairs
        negative_threshold: Quality similarity threshold for negative pairs

    Example:
        >>> sampler = ContrastivePairSampler(positive_threshold=0.8)
        >>> quality_scores = torch.tensor([0.9, 0.85, 0.3, 0.88])
        >>> pos_pairs, neg_pairs = sampler.sample_pairs(quality_scores)
    """

    def __init__(
        self,
        positive_threshold: float = 0.8,
        negative_threshold: float = 0.3
    ):
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold

    def sample_pairs(
        self,
        quality_scores: torch.Tensor,
        similarity_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample positive and negative pairs.

        Args:
            quality_scores: Quality scores [batch]
            similarity_matrix: Pairwise similarity [batch, batch]

        Returns:
            positive_pairs: Indices of positive pairs [num_pos, 2]
            negative_pairs: Indices of negative pairs [num_neg, 2]
        """
        batch_size = quality_scores.size(0)

        # Create index grid
        i_idx, j_idx = torch.meshgrid(
            torch.arange(batch_size),
            torch.arange(batch_size),
            indexing='ij'
        )

        # Exclude self-pairs
        mask = i_idx != j_idx

        # Positive pairs: high similarity
        pos_mask = (similarity_matrix > self.positive_threshold) & mask
        pos_i = i_idx[pos_mask]
        pos_j = j_idx[pos_mask]
        positive_pairs = torch.stack([pos_i, pos_j], dim=1)

        # Negative pairs: low similarity
        neg_mask = (similarity_matrix < self.negative_threshold) & mask
        neg_i = i_idx[neg_mask]
        neg_j = j_idx[neg_mask]
        negative_pairs = torch.stack([neg_i, neg_j], dim=1)

        return positive_pairs, negative_pairs


def compute_quality_aware_targets(
    signals: torch.Tensor,
    fs: float = 125.0,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience function to compute quality scores and similarity matrix.

    Args:
        signals: Batch of signals [batch, channels, time]
        fs: Sampling frequency
        device: Device for computation

    Returns:
        quality_scores: Quality scores [batch]
        similarity_matrix: Pairwise similarity [batch, batch]

    Example:
        >>> signals = torch.randn(32, 2, 1024)
        >>> scores, sim = compute_quality_aware_targets(signals)
    """
    qpc = QualityProxyComputer(fs=fs, device=device)

    quality_scores = qpc.compute_batch(signals)
    similarity_matrix = qpc.compute_pairwise_similarity(quality_scores)

    return quality_scores, similarity_matrix
