"""Windowing utilities for converting continuous signals to fixed-length windows."""

import logging
from typing import Literal, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def create_windows(
    data: np.ndarray,
    labels: np.ndarray,
    window_length: int,
    stride: int,
    label_strategy: Literal["center", "mode", "first", "last"] = "center",
    boundary_handling: Literal["drop_incomplete", "pad_zeros", "pad_repeat"] = "drop_incomplete",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create fixed-length windows from continuous time series data.

    Args:
        data: Input data of shape (num_samples, num_channels)
        labels: Input labels of shape (num_samples,)
        window_length: Length of each window in samples
        stride: Stride between consecutive windows (samples)
        label_strategy: How to determine label for each window:
            - "center": Use label at center of window
            - "mode": Use most frequent label in window
            - "first": Use first label in window
            - "last": Use last label in window
        boundary_handling: How to handle incomplete windows at end:
            - "drop_incomplete": Drop windows that don't have full length
            - "pad_zeros": Pad incomplete windows with zeros
            - "pad_repeat": Pad incomplete windows by repeating last sample

    Returns:
        Tuple of (windowed_data, window_labels) where:
            - windowed_data: shape (num_windows, window_length, num_channels)
            - window_labels: shape (num_windows,)

    Raises:
        ValueError: If inputs are invalid

    Example:
        >>> data = np.random.randn(1000, 3)  # 1000 samples, 3 channels
        >>> labels = np.random.randint(0, 5, 1000)
        >>> windows, win_labels = create_windows(data, labels, 512, 256)
        >>> windows.shape
        (2, 512, 3)  # 2 windows with 50% overlap
    """
    # Validate inputs
    if data.ndim != 2:
        raise ValueError(
            f"Expected data to be 2D (samples, channels).\n"
            f"  Received: {data.ndim}D with shape {data.shape}\n"
            f"  Hint: Reshape data to (num_samples, num_channels)"
        )

    if labels.ndim != 1:
        raise ValueError(
            f"Expected labels to be 1D.\n"
            f"  Received: {labels.ndim}D with shape {labels.shape}\n"
            f"  Hint: Flatten labels to (num_samples,)"
        )

    if data.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Data and labels must have same length.\n"
            f"  Data length: {data.shape[0]}\n"
            f"  Labels length: {labels.shape[0]}"
        )

    if window_length <= 0:
        raise ValueError(
            f"window_length must be positive.\n" f"  Received: {window_length}"
        )

    if stride <= 0:
        raise ValueError(f"stride must be positive.\n" f"  Received: {stride}")

    num_samples, num_channels = data.shape

    # Calculate number of complete windows
    if num_samples < window_length:
        if boundary_handling == "drop_incomplete":
            logger.warning(
                f"Signal length ({num_samples}) < window_length ({window_length}). "
                f"Returning 0 windows."
            )
            return np.empty((0, window_length, num_channels)), np.empty((0,), dtype=labels.dtype)
        else:
            num_windows = 1
    else:
        num_windows = (num_samples - window_length) // stride + 1

    # Initialize output arrays
    windows = np.zeros((num_windows, window_length, num_channels), dtype=data.dtype)
    window_labels = np.zeros(num_windows, dtype=labels.dtype)

    # Create windows
    for i in range(num_windows):
        start_idx = i * stride
        end_idx = start_idx + window_length

        if end_idx <= num_samples:
            # Complete window
            windows[i] = data[start_idx:end_idx]
            window_labels[i] = _get_window_label(
                labels[start_idx:end_idx], label_strategy
            )

        else:
            # Incomplete window (only if boundary_handling != "drop_incomplete")
            if boundary_handling == "drop_incomplete":
                # This shouldn't happen with correct num_windows calculation
                logger.warning("Unexpected incomplete window with drop_incomplete strategy")
                break

            elif boundary_handling == "pad_zeros":
                # Pad with zeros
                available = num_samples - start_idx
                windows[i, :available] = data[start_idx:]
                windows[i, available:] = 0
                window_labels[i] = _get_window_label(
                    labels[start_idx:], label_strategy
                )

            elif boundary_handling == "pad_repeat":
                # Pad by repeating last sample
                available = num_samples - start_idx
                windows[i, :available] = data[start_idx:]
                windows[i, available:] = data[-1]
                window_labels[i] = _get_window_label(
                    labels[start_idx:], label_strategy
                )

    logger.debug(
        f"Created {num_windows} windows: "
        f"window_length={window_length}, stride={stride}, "
        f"overlap={window_length - stride}"
    )

    return windows, window_labels


def _get_window_label(
    window_labels: np.ndarray, strategy: Literal["center", "mode", "first", "last"]
) -> int:
    """
    Get single label for a window based on strategy.

    Args:
        window_labels: Labels for all samples in window
        strategy: Label selection strategy

    Returns:
        Single label for the window
    """
    if strategy == "center":
        # Use label at center of window
        return window_labels[len(window_labels) // 2]

    elif strategy == "mode":
        # Use most frequent label
        counts = np.bincount(window_labels)
        return np.argmax(counts)

    elif strategy == "first":
        # Use first label
        return window_labels[0]

    elif strategy == "last":
        # Use last label
        return window_labels[-1]

    else:
        raise ValueError(f"Unknown label strategy: {strategy}")


def compute_num_windows(
    signal_length: int, window_length: int, stride: int
) -> int:
    """
    Compute the number of complete windows that will be created.

    Args:
        signal_length: Total length of signal
        window_length: Length of each window
        stride: Stride between windows

    Returns:
        Number of complete windows

    Example:
        >>> # 1000 samples, 512-length windows, 256 stride
        >>> compute_num_windows(1000, 512, 256)
        2
        >>> # With no overlap
        >>> compute_num_windows(1000, 512, 512)
        1
    """
    if signal_length < window_length:
        return 0
    return max(0, (signal_length - window_length) // stride + 1)


def validate_windowing(
    data: np.ndarray,
    windows: np.ndarray,
    window_length: int,
    stride: int,
) -> Tuple[bool, str]:
    """
    Validate that windowing was performed correctly.

    Args:
        data: Original data
        windows: Windowed data
        window_length: Window length used
        stride: Stride used

    Returns:
        Tuple of (is_valid, message)

    Example:
        >>> data = np.random.randn(1000, 3)
        >>> labels = np.zeros(1000, dtype=int)
        >>> windows, _ = create_windows(data, labels, 512, 256)
        >>> is_valid, msg = validate_windowing(data, windows, 512, 256)
        >>> assert is_valid, msg
    """
    # Check window shape
    expected_num_windows = compute_num_windows(data.shape[0], window_length, stride)

    if windows.shape[0] != expected_num_windows:
        return False, (
            f"Number of windows mismatch.\n"
            f"  Expected: {expected_num_windows}\n"
            f"  Received: {windows.shape[0]}"
        )

    if windows.shape[1] != window_length:
        return False, (
            f"Window length mismatch.\n"
            f"  Expected: {window_length}\n"
            f"  Received: {windows.shape[1]}"
        )

    if windows.shape[2] != data.shape[1]:
        return False, (
            f"Number of channels mismatch.\n"
            f"  Expected: {data.shape[1]}\n"
            f"  Received: {windows.shape[2]}"
        )

    # Verify first window matches original data
    if windows.shape[0] > 0:
        first_window = windows[0]
        original_first = data[:window_length]

        if not np.array_equal(first_window, original_first):
            return False, (
                f"First window doesn't match original data.\n"
                f"  Max difference: {np.max(np.abs(first_window - original_first))}"
            )

    return True, "Windowing validation passed"


def merge_overlapping_predictions(
    window_preds: np.ndarray,
    window_starts: np.ndarray,
    total_length: int,
    strategy: Literal["vote", "average"] = "vote",
) -> np.ndarray:
    """
    Merge predictions from overlapping windows back into continuous sequence.

    Useful for converting window-level predictions back to sample-level predictions.

    Args:
        window_preds: Predictions for each window, shape (num_windows,) for hard labels
                     or (num_windows, num_classes) for soft probabilities
        window_starts: Start index of each window
        total_length: Total length of original signal
        strategy: Merging strategy:
            - "vote": Majority voting for overlapping regions (for hard labels)
            - "average": Average probabilities (for soft predictions)

    Returns:
        Merged predictions of length total_length

    Example:
        >>> window_preds = np.array([0, 1, 1])  # 3 overlapping windows
        >>> window_starts = np.array([0, 256, 512])
        >>> merged = merge_overlapping_predictions(window_preds, window_starts, 1000)
    """
    if window_preds.ndim == 1:
        # Hard labels
        result = np.zeros(total_length, dtype=window_preds.dtype)
        counts = np.zeros(total_length, dtype=int)

        window_length = total_length // len(window_starts)  # Approximate

        for start, pred in zip(window_starts, window_preds):
            end = min(start + window_length, total_length)
            if strategy == "vote":
                result[start:end] = pred
                counts[start:end] += 1
            else:
                raise ValueError(f"Strategy '{strategy}' not supported for hard labels")

        return result

    else:
        # Soft probabilities
        num_classes = window_preds.shape[1]
        result = np.zeros((total_length, num_classes), dtype=window_preds.dtype)
        counts = np.zeros(total_length, dtype=int)

        window_length = total_length // len(window_starts)

        for start, pred in zip(window_starts, window_preds):
            end = min(start + window_length, total_length)
            result[start:end] += pred
            counts[start:end] += 1

        # Average probabilities
        counts = np.maximum(counts, 1)  # Avoid division by zero
        result = result / counts[:, np.newaxis]

        return result
