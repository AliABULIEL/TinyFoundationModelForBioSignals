"""Gravity removal utilities for accelerometry data."""

import logging
from typing import Optional

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


def remove_gravity(
    data: np.ndarray,
    sampling_rate: float,
    method: str = "highpass",
    cutoff_freq: float = 0.5,
    order: int = 4,
) -> np.ndarray:
    """
    Remove gravitational component from accelerometry data.

    Separates gravitational acceleration (low-frequency) from body movement
    acceleration (high-frequency) using filtering.

    Args:
        data: Input accelerometry data of shape (num_samples, 3) where
              columns are X, Y, Z acceleration in g or m/s^2
        sampling_rate: Sampling frequency in Hz
        method: Filtering method:
            - "highpass": High-pass Butterworth filter
            - "none": No filtering (returns copy of input)
        cutoff_freq: Cutoff frequency for high-pass filter in Hz
                     (typical: 0.5 Hz for human movement)
        order: Filter order (higher = sharper cutoff, default: 4)

    Returns:
        Filtered acceleration data (body movement only) with same shape as input

    Raises:
        ValueError: If inputs are invalid

    Example:
        >>> # Raw accelerometry includes gravity + body movement
        >>> raw_acc = load_accelerometry()  # shape (10000, 3)
        >>> # Remove gravity to isolate body movement
        >>> body_acc = remove_gravity(raw_acc, sampling_rate=100, cutoff_freq=0.5)
    """
    # Validate inputs
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(
            f"Expected data shape (num_samples, 3).\n"
            f"  Received: {data.shape}\n"
            f"  Hint: Reshape to (num_samples, 3) for X, Y, Z axes"
        )

    if sampling_rate <= 0:
        raise ValueError(
            f"sampling_rate must be positive.\n" f"  Received: {sampling_rate}"
        )

    if method == "none":
        logger.debug("Gravity removal disabled (method='none')")
        return data.copy()

    elif method == "highpass":
        return _apply_highpass_filter(data, sampling_rate, cutoff_freq, order)

    else:
        raise ValueError(
            f"Unknown gravity removal method: {method}\n"
            f"  Supported methods: ['highpass', 'none']"
        )


def _apply_highpass_filter(
    data: np.ndarray,
    sampling_rate: float,
    cutoff_freq: float,
    order: int,
) -> np.ndarray:
    """
    Apply high-pass Butterworth filter to remove gravity.

    Args:
        data: Input data (num_samples, 3)
        sampling_rate: Sampling rate in Hz
        cutoff_freq: Cutoff frequency in Hz
        order: Filter order

    Returns:
        Filtered data
    """
    # Validate cutoff frequency
    nyquist = sampling_rate / 2.0

    if cutoff_freq >= nyquist:
        raise ValueError(
            f"Cutoff frequency ({cutoff_freq} Hz) must be less than "
            f"Nyquist frequency ({nyquist} Hz).\n"
            f"  Hint: Reduce cutoff_freq or increase sampling_rate"
        )

    if cutoff_freq <= 0:
        raise ValueError(
            f"Cutoff frequency must be positive.\n"
            f"  Received: {cutoff_freq}\n"
            f"  Hint: Typical values: 0.5 Hz for human movement"
        )

    # Design high-pass Butterworth filter
    try:
        sos = signal.butter(
            order, cutoff_freq, btype="highpass", fs=sampling_rate, output="sos"
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to design high-pass filter.\n"
            f"  Sampling rate: {sampling_rate} Hz\n"
            f"  Cutoff frequency: {cutoff_freq} Hz\n"
            f"  Order: {order}\n"
            f"  Error: {e}"
        ) from e

    # Apply filter to each axis
    filtered = np.zeros_like(data)

    for i in range(3):
        try:
            filtered[:, i] = signal.sosfiltfilt(sos, data[:, i])
        except Exception as e:
            raise RuntimeError(
                f"Failed to apply filter to axis {i}.\n" f"  Error: {e}"
            ) from e

    logger.debug(
        f"Applied high-pass filter: cutoff={cutoff_freq}Hz, "
        f"order={order}, fs={sampling_rate}Hz"
    )

    return filtered


def compute_gravity_component(
    data: np.ndarray,
    sampling_rate: float,
    cutoff_freq: float = 0.5,
    order: int = 4,
) -> np.ndarray:
    """
    Compute the gravitational component (low-frequency) of accelerometry.

    This is the complement of remove_gravity() - instead of returning
    high-frequency body movement, returns low-frequency gravity.

    Args:
        data: Input accelerometry data (num_samples, 3)
        sampling_rate: Sampling rate in Hz
        cutoff_freq: Cutoff frequency for low-pass filter
        order: Filter order

    Returns:
        Gravity component (num_samples, 3)

    Example:
        >>> raw = load_accelerometry()
        >>> gravity = compute_gravity_component(raw, 100)
        >>> body = remove_gravity(raw, 100)
        >>> # Verify: raw ≈ gravity + body
        >>> np.allclose(raw, gravity + body)
        True
    """
    # Validate inputs
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(
            f"Expected data shape (num_samples, 3).\n" f"  Received: {data.shape}"
        )

    # Design low-pass Butterworth filter
    nyquist = sampling_rate / 2.0

    if cutoff_freq >= nyquist:
        raise ValueError(
            f"Cutoff frequency ({cutoff_freq} Hz) >= Nyquist ({nyquist} Hz)"
        )

    sos = signal.butter(
        order, cutoff_freq, btype="lowpass", fs=sampling_rate, output="sos"
    )

    # Apply filter to each axis
    gravity = np.zeros_like(data)

    for i in range(3):
        gravity[:, i] = signal.sosfiltfilt(sos, data[:, i])

    return gravity


def validate_gravity_removal(
    original: np.ndarray,
    body: np.ndarray,
    gravity: Optional[np.ndarray] = None,
    tolerance: float = 0.1,
) -> tuple[bool, str]:
    """
    Validate gravity removal by checking energy distribution.

    Args:
        original: Original accelerometry
        body: Body movement (after gravity removal)
        gravity: Gravity component (if available)
        tolerance: Acceptable energy deviation

    Returns:
        Tuple of (is_valid, message)
    """
    # Check shapes match
    if original.shape != body.shape:
        return False, f"Shape mismatch: {original.shape} vs {body.shape}"

    # If gravity is provided, check that original ≈ gravity + body
    if gravity is not None:
        reconstructed = gravity + body
        max_diff = np.max(np.abs(original - reconstructed))

        if max_diff > tolerance:
            return False, (
                f"Gravity + Body doesn't reconstruct original.\n"
                f"  Max difference: {max_diff}\n"
                f"  Tolerance: {tolerance}"
            )

    # Check that body movement has reduced low-frequency energy
    # (This is a heuristic check - body should have less DC component)
    original_mean = np.abs(np.mean(original, axis=0))
    body_mean = np.abs(np.mean(body, axis=0))

    if np.any(body_mean > original_mean):
        logger.warning(
            f"Body movement has higher DC component than original. "
            f"This may indicate filtering issues."
        )

    return True, "Gravity removal validation passed"
