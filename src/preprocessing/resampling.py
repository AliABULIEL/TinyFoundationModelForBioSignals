"""Signal resampling utilities for TTM-HAR."""

import logging
from typing import Tuple

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


def resample_signal(
    data: np.ndarray,
    source_rate: int,
    target_rate: int,
    method: str = "polyphase",
    axis: int = 0,
) -> np.ndarray:
    """
    Resample signal from source sampling rate to target sampling rate.

    Uses polyphase filtering (scipy.signal.resample_poly) for high-quality resampling
    with minimal aliasing and energy preservation.

    Args:
        data: Input signal array of shape (..., samples, ...) where axis specifies
              the time dimension
        source_rate: Original sampling rate in Hz
        target_rate: Target sampling rate in Hz
        method: Resampling method ("polyphase" only currently supported)
        axis: Axis along which to resample (default: 0)

    Returns:
        Resampled signal array with updated length along specified axis

    Raises:
        ValueError: If sampling rates are invalid or method is unsupported

    Example:
        >>> # Resample from 100Hz to 30Hz
        >>> signal_100hz = np.random.randn(1000, 3)  # 10 seconds, 3 channels
        >>> signal_30hz = resample_signal(signal_100hz, 100, 30)
        >>> signal_30hz.shape
        (300, 3)
    """
    # Validate inputs
    if source_rate <= 0:
        raise ValueError(
            f"source_rate must be positive.\n"
            f"  Received: {source_rate}\n"
            f"  Hint: Common values are 30, 50, 100 Hz"
        )

    if target_rate <= 0:
        raise ValueError(
            f"target_rate must be positive.\n"
            f"  Received: {target_rate}\n"
            f"  Hint: Common values are 30, 50, 100 Hz"
        )

    if method != "polyphase":
        raise ValueError(
            f"Unsupported resampling method: {method}\n"
            f"  Supported methods: ['polyphase']"
        )

    # If rates are equal, return copy
    if source_rate == target_rate:
        logger.debug(f"Source and target rates are equal ({source_rate} Hz), returning copy")
        return data.copy()

    # Compute resampling ratio
    gcd = np.gcd(source_rate, target_rate)
    up = target_rate // gcd
    down = source_rate // gcd

    logger.debug(
        f"Resampling from {source_rate}Hz to {target_rate}Hz "
        f"(upsample {up}x, downsample {down}x)"
    )

    # Resample using polyphase filtering
    try:
        resampled = signal.resample_poly(data, up, down, axis=axis)
    except Exception as e:
        raise RuntimeError(
            f"Resampling failed.\n"
            f"  Source rate: {source_rate} Hz\n"
            f"  Target rate: {target_rate} Hz\n"
            f"  Input shape: {data.shape}\n"
            f"  Error: {e}"
        ) from e

    # Verify energy preservation (Parseval's theorem)
    energy_before = np.sum(data**2)
    energy_after = np.sum(resampled**2)
    energy_ratio = energy_after / (energy_before + 1e-10)

    if not (0.95 <= energy_ratio <= 1.05):
        logger.warning(
            f"Resampling may have altered signal energy significantly. "
            f"Energy ratio: {energy_ratio:.3f} (expected ~1.0)"
        )

    logger.debug(
        f"Resampled signal: {data.shape} -> {resampled.shape}, "
        f"energy ratio: {energy_ratio:.3f}"
    )

    return resampled


def compute_target_length(
    source_length: int, source_rate: int, target_rate: int
) -> int:
    """
    Compute expected length after resampling.

    Args:
        source_length: Original signal length (number of samples)
        source_rate: Original sampling rate in Hz
        target_rate: Target sampling rate in Hz

    Returns:
        Expected length after resampling

    Example:
        >>> # 10 seconds at 100Hz -> 300 samples at 30Hz
        >>> compute_target_length(1000, 100, 30)
        300
    """
    duration_seconds = source_length / source_rate
    target_length = int(np.round(duration_seconds * target_rate))
    return target_length


def validate_resampling(
    original: np.ndarray,
    resampled: np.ndarray,
    source_rate: int,
    target_rate: int,
    tolerance: float = 0.01,
) -> Tuple[bool, str]:
    """
    Validate that resampling was performed correctly.

    Checks:
    1. Output length matches expected length
    2. Energy preservation (within tolerance)

    Args:
        original: Original signal
        resampled: Resampled signal
        source_rate: Original sampling rate
        target_rate: Target sampling rate
        tolerance: Acceptable energy deviation (default: 1%)

    Returns:
        Tuple of (is_valid, message)

    Example:
        >>> original = np.random.randn(1000, 3)
        >>> resampled = resample_signal(original, 100, 30)
        >>> is_valid, msg = validate_resampling(original, resampled, 100, 30)
        >>> assert is_valid, msg
    """
    # Check length
    expected_length = compute_target_length(original.shape[0], source_rate, target_rate)

    if abs(resampled.shape[0] - expected_length) > 1:
        return False, (
            f"Resampled length mismatch.\n"
            f"  Expected: {expected_length}\n"
            f"  Received: {resampled.shape[0]}\n"
            f"  Difference: {abs(resampled.shape[0] - expected_length)}"
        )

    # Check energy preservation
    energy_before = np.sum(original**2)
    energy_after = np.sum(resampled**2)
    energy_ratio = energy_after / (energy_before + 1e-10)

    if abs(energy_ratio - 1.0) > tolerance:
        return False, (
            f"Energy preservation violated.\n"
            f"  Energy ratio: {energy_ratio:.4f}\n"
            f"  Tolerance: {tolerance}\n"
            f"  Deviation: {abs(energy_ratio - 1.0):.4f}"
        )

    return True, "Resampling validation passed"
