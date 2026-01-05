"""Normalization utilities for TTM-HAR, including RevIN."""

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN).

    Normalizes input time series to zero mean and unit variance per channel,
    with the ability to reverse the normalization after model processing.

    This is particularly useful for time series foundation models to handle
    distribution shifts while preserving the ability to interpret outputs in
    original scale.

    Reference: Kim et al. "Reversible Instance Normalization for Accurate
    Time-Series Forecasting against Distribution Shift" (NeurIPS 2022)

    Args:
        num_channels: Number of input channels
        epsilon: Small constant for numerical stability
        affine: If True, learns affine transformation parameters

    Example:
        >>> revin = RevIN(num_channels=3)
        >>> x = torch.randn(32, 512, 3)  # (batch, time, channels)
        >>> x_norm = revin(x, mode='norm')
        >>> # ... model processing ...
        >>> x_denorm = revin(x_norm, mode='denorm')
    """

    def __init__(
        self,
        num_channels: int,
        epsilon: float = 1e-8,
        affine: bool = False,
    ) -> None:
        """Initialize RevIN layer."""
        super().__init__()

        self.num_channels = num_channels
        self.epsilon = epsilon
        self.affine = affine

        # Learnable affine parameters (optional)
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_channels))
            self.affine_bias = nn.Parameter(torch.zeros(num_channels))

        # Buffers to store statistics for denormalization
        self.register_buffer("mean", torch.zeros(num_channels))
        self.register_buffer("std", torch.ones(num_channels))

    def forward(
        self, x: torch.Tensor, mode: str = "norm"
    ) -> torch.Tensor:
        """
        Apply RevIN normalization or denormalization.

        Args:
            x: Input tensor of shape (B, L, C) where:
               B = batch size, L = sequence length, C = num_channels
            mode: Either "norm" (normalize) or "denorm" (denormalize)

        Returns:
            Normalized or denormalized tensor

        Raises:
            ValueError: If input shape is invalid or mode is unknown
        """
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3D input (batch, length, channels).\n"
                f"  Received: {x.dim()}D with shape {x.shape}\n"
                f"  Hint: Reshape to (batch_size, sequence_length, num_channels)"
            )

        if x.shape[2] != self.num_channels:
            raise ValueError(
                f"Channel mismatch.\n"
                f"  Expected: {self.num_channels} channels\n"
                f"  Received: {x.shape[2]} channels"
            )

        if mode == "norm":
            return self._normalize(x)
        elif mode == "denorm":
            return self._denormalize(x)
        else:
            raise ValueError(
                f"Unknown mode: {mode}\n" f"  Supported modes: ['norm', 'denorm']"
            )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input to zero mean and unit variance per channel.

        Args:
            x: Input tensor (B, L, C)

        Returns:
            Normalized tensor
        """
        # Compute statistics over time dimension (dim=1)
        mean = torch.mean(x, dim=1, keepdim=True)  # (B, 1, C)
        std = torch.sqrt(torch.var(x, dim=1, keepdim=True) + self.epsilon)  # (B, 1, C)

        # Store for denormalization
        self.mean = mean.detach().squeeze(1)[0]  # Store first sample's stats
        self.std = std.detach().squeeze(1)[0]

        # Normalize
        x_norm = (x - mean) / std

        # Apply affine transformation if enabled
        if self.affine:
            x_norm = x_norm * self.affine_weight + self.affine_bias

        return x_norm

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reverse normalization to original scale.

        Args:
            x: Normalized tensor (B, L, C)

        Returns:
            Denormalized tensor
        """
        # Reverse affine transformation if enabled
        if self.affine:
            x = (x - self.affine_bias) / self.affine_weight

        # Denormalize using stored statistics
        mean = self.mean.unsqueeze(0).unsqueeze(1)  # (1, 1, C)
        std = self.std.unsqueeze(0).unsqueeze(1)  # (1, 1, C)

        x_denorm = x * std + mean

        return x_denorm


def normalize_window(
    window: np.ndarray,
    method: str = "zscore",
    epsilon: float = 1e-8,
    axis: int = 0,
) -> Tuple[np.ndarray, dict]:
    """
    Normalize a single window of data.

    Args:
        window: Input window of shape (length, channels)
        method: Normalization method:
            - "zscore": Zero mean, unit variance (per channel)
            - "minmax": Scale to [0, 1] range (per channel)
            - "robust": Use median and IQR (robust to outliers)
        epsilon: Small constant for numerical stability
        axis: Axis along which to compute statistics (0 for time)

    Returns:
        Tuple of (normalized_window, statistics_dict) where statistics_dict
        contains the parameters needed to reverse normalization

    Example:
        >>> window = np.random.randn(512, 3)
        >>> norm_window, stats = normalize_window(window, method="zscore")
        >>> # Verify zero mean, unit variance
        >>> np.allclose(norm_window.mean(axis=0), 0, atol=1e-6)
        True
    """
    if method == "zscore":
        mean = np.mean(window, axis=axis, keepdims=True)
        std = np.std(window, axis=axis, keepdims=True) + epsilon
        normalized = (window - mean) / std

        stats = {"method": "zscore", "mean": mean, "std": std}

    elif method == "minmax":
        min_val = np.min(window, axis=axis, keepdims=True)
        max_val = np.max(window, axis=axis, keepdims=True)
        range_val = max_val - min_val + epsilon
        normalized = (window - min_val) / range_val

        stats = {"method": "minmax", "min": min_val, "max": max_val}

    elif method == "robust":
        median = np.median(window, axis=axis, keepdims=True)
        q75 = np.percentile(window, 75, axis=axis, keepdims=True)
        q25 = np.percentile(window, 25, axis=axis, keepdims=True)
        iqr = q75 - q25 + epsilon
        normalized = (window - median) / iqr

        stats = {"method": "robust", "median": median, "iqr": iqr}

    else:
        raise ValueError(
            f"Unknown normalization method: {method}\n"
            f"  Supported methods: ['zscore', 'minmax', 'robust']"
        )

    return normalized, stats


def denormalize_window(window: np.ndarray, stats: dict) -> np.ndarray:
    """
    Reverse normalization using stored statistics.

    Args:
        window: Normalized window
        stats: Statistics dictionary from normalize_window()

    Returns:
        Denormalized window

    Example:
        >>> window = np.random.randn(512, 3)
        >>> norm_window, stats = normalize_window(window)
        >>> denorm_window = denormalize_window(norm_window, stats)
        >>> np.allclose(window, denorm_window)
        True
    """
    method = stats["method"]

    if method == "zscore":
        return window * stats["std"] + stats["mean"]

    elif method == "minmax":
        range_val = stats["max"] - stats["min"]
        return window * range_val + stats["min"]

    elif method == "robust":
        return window * stats["iqr"] + stats["median"]

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def validate_normalization(
    original: np.ndarray,
    normalized: np.ndarray,
    method: str,
    tolerance: float = 1e-5,
) -> Tuple[bool, str]:
    """
    Validate that normalization was performed correctly.

    Args:
        original: Original window
        normalized: Normalized window
        method: Normalization method used
        tolerance: Tolerance for numerical checks

    Returns:
        Tuple of (is_valid, message)

    Example:
        >>> window = np.random.randn(512, 3)
        >>> norm, stats = normalize_window(window, "zscore")
        >>> is_valid, msg = validate_normalization(window, norm, "zscore")
        >>> assert is_valid, msg
    """
    if method == "zscore":
        # Check zero mean
        mean = np.mean(normalized, axis=0)
        if not np.allclose(mean, 0, atol=tolerance):
            return False, (
                f"Mean not close to zero after z-score normalization.\n"
                f"  Mean: {mean}\n"
                f"  Expected: ~0\n"
                f"  Tolerance: {tolerance}"
            )

        # Check unit variance
        std = np.std(normalized, axis=0)
        if not np.allclose(std, 1, atol=tolerance):
            return False, (
                f"Std not close to one after z-score normalization.\n"
                f"  Std: {std}\n"
                f"  Expected: ~1\n"
                f"  Tolerance: {tolerance}"
            )

    elif method == "minmax":
        # Check range [0, 1]
        min_val = np.min(normalized, axis=0)
        max_val = np.max(normalized, axis=0)

        if not np.allclose(min_val, 0, atol=tolerance):
            return False, f"Min not close to 0: {min_val}"

        if not np.allclose(max_val, 1, atol=tolerance):
            return False, f"Max not close to 1: {max_val}"

    # Test reversibility
    norm, stats = normalize_window(original, method)
    denorm = denormalize_window(norm, stats)

    if not np.allclose(original, denorm, atol=tolerance):
        max_diff = np.max(np.abs(original - denorm))
        return False, (
            f"Normalization not reversible.\n"
            f"  Max difference: {max_diff}\n"
            f"  Tolerance: {tolerance}"
        )

    return True, "Normalization validation passed"
