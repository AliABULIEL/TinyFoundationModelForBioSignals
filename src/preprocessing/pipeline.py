"""Preprocessing pipeline that chains multiple operations."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.preprocessing.gravity import remove_gravity
from src.preprocessing.normalization import normalize_window
from src.preprocessing.resampling import resample_signal
from src.preprocessing.windowing import create_windows

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Preprocessing pipeline for accelerometry data.

    Applies a sequence of transformations:
    1. Resampling (optional)
    2. Gravity removal (optional)
    3. Windowing
    4. Normalization (optional)

    Args:
        config: Configuration dictionary with preprocessing parameters

    Example:
        >>> config = {
        >>>     "sampling_rate_original": 100,
        >>>     "sampling_rate_target": 30,
        >>>     "context_length": 512,
        >>>     "window_stride_train": 256,
        >>>     "normalization": {"method": "zscore"},
        >>> }
        >>> pipeline = PreprocessingPipeline(config)
        >>> data = np.random.randn(10000, 3)
        >>> labels = np.zeros(10000, dtype=int)
        >>> windows, win_labels = pipeline.process(data, labels, is_training=True)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize preprocessing pipeline."""
        self.config = config

        # Extract configuration
        self.sampling_rate_original = config.get("sampling_rate_original", 100)
        self.sampling_rate_target = config.get("sampling_rate_target", 30)
        self.context_length = config.get("context_length", 512)
        self.window_stride_train = config.get("window_stride_train", 256)
        self.window_stride_eval = config.get("window_stride_eval", 512)

        # Resampling
        self.enable_resampling = (
            self.sampling_rate_target != self.sampling_rate_original
        )
        self.resampling_method = config.get("resampling_method", "polyphase")

        # Gravity removal
        gravity_config = config.get("gravity_removal", {})
        self.enable_gravity_removal = gravity_config.get("enabled", False)
        self.gravity_method = gravity_config.get("method", "highpass")
        self.gravity_cutoff = gravity_config.get("cutoff_freq", 0.5)

        # Normalization
        norm_config = config.get("normalization", {})
        self.enable_normalization = norm_config.get("method") is not None
        self.norm_method = norm_config.get("method", "zscore")
        self.norm_epsilon = norm_config.get("epsilon", 1e-8)

        logger.info(
            f"Initialized preprocessing pipeline:\n"
            f"  Resampling: {self.sampling_rate_original}Hz -> {self.sampling_rate_target}Hz "
            f"({'enabled' if self.enable_resampling else 'disabled'})\n"
            f"  Gravity removal: {'enabled' if self.enable_gravity_removal else 'disabled'}\n"
            f"  Windowing: length={self.context_length}, "
            f"stride_train={self.window_stride_train}, stride_eval={self.window_stride_eval}\n"
            f"  Normalization: {self.norm_method if self.enable_normalization else 'disabled'}"
        )

    def process(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        is_training: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process continuous signal through preprocessing pipeline.

        Args:
            data: Input data of shape (num_samples, num_channels)
            labels: Input labels of shape (num_samples,)
            is_training: Whether processing for training (affects stride)

        Returns:
            Tuple of (windowed_data, window_labels) where:
                - windowed_data: (num_windows, context_length, num_channels)
                - window_labels: (num_windows,)

        Raises:
            ValueError: If inputs are invalid
        """
        logger.debug(
            f"Starting preprocessing: input shape={data.shape}, "
            f"is_training={is_training}"
        )

        # Validate inputs
        self._validate_inputs(data, labels)

        # Step 1: Resampling
        if self.enable_resampling:
            data = resample_signal(
                data,
                source_rate=self.sampling_rate_original,
                target_rate=self.sampling_rate_target,
                method=self.resampling_method,
                axis=0,
            )

            # Resample labels (using nearest neighbor)
            ratio = self.sampling_rate_target / self.sampling_rate_original
            new_length = int(len(labels) * ratio)
            label_indices = np.round(
                np.linspace(0, len(labels) - 1, new_length)
            ).astype(int)
            labels = labels[label_indices]

            logger.debug(f"After resampling: shape={data.shape}")

        # Step 2: Gravity removal
        if self.enable_gravity_removal:
            data = remove_gravity(
                data,
                sampling_rate=self.sampling_rate_target,
                method=self.gravity_method,
                cutoff_freq=self.gravity_cutoff,
            )
            logger.debug("Applied gravity removal")

        # Step 3: Windowing
        stride = self.window_stride_train if is_training else self.window_stride_eval

        windows, window_labels = create_windows(
            data,
            labels,
            window_length=self.context_length,
            stride=stride,
            label_strategy="center",
            boundary_handling="drop_incomplete",
        )

        logger.debug(
            f"After windowing: num_windows={windows.shape[0]}, "
            f"window_shape={windows.shape[1:]}"
        )

        # Step 4: Normalization (per window)
        if self.enable_normalization:
            windows = self._normalize_windows(windows)
            logger.debug("Applied normalization")

        # Validate output shapes
        self._validate_outputs(windows, window_labels)

        logger.info(
            f"Preprocessing complete: {data.shape[0]} samples -> "
            f"{windows.shape[0]} windows of length {self.context_length}"
        )

        return windows, window_labels

    def _normalize_windows(self, windows: np.ndarray) -> np.ndarray:
        """
        Normalize each window independently.

        Args:
            windows: Array of shape (num_windows, window_length, num_channels)

        Returns:
            Normalized windows
        """
        num_windows = windows.shape[0]
        normalized = np.zeros_like(windows)

        for i in range(num_windows):
            normalized[i], _ = normalize_window(
                windows[i],
                method=self.norm_method,
                epsilon=self.norm_epsilon,
                axis=0,
            )

        return normalized

    def _validate_inputs(self, data: np.ndarray, labels: np.ndarray) -> None:
        """Validate input data and labels."""
        if data.ndim != 2:
            raise ValueError(
                f"Expected 2D data (samples, channels).\n"
                f"  Received: {data.ndim}D with shape {data.shape}"
            )

        if labels.ndim != 1:
            raise ValueError(
                f"Expected 1D labels.\n" f"  Received: {labels.ndim}D with shape {labels.shape}"
            )

        if data.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Data and labels length mismatch.\n"
                f"  Data: {data.shape[0]}\n"
                f"  Labels: {labels.shape[0]}"
            )

        if data.shape[1] != 3:
            logger.warning(
                f"Expected 3 channels (X, Y, Z) but got {data.shape[1]}. "
                f"Proceeding anyway."
            )

    def _validate_outputs(
        self, windows: np.ndarray, window_labels: np.ndarray
    ) -> None:
        """Validate output shapes."""
        expected_shape = (windows.shape[0], self.context_length, windows.shape[2])

        if windows.shape != expected_shape:
            raise ValueError(
                f"Output window shape mismatch.\n"
                f"  Expected: {expected_shape}\n"
                f"  Received: {windows.shape}"
            )

        if len(window_labels) != windows.shape[0]:
            raise ValueError(
                f"Number of windows and labels mismatch.\n"
                f"  Windows: {windows.shape[0]}\n"
                f"  Labels: {len(window_labels)}"
            )

    def get_output_shape(self, input_length: int) -> Tuple[int, int, int]:
        """
        Compute output shape for given input length.

        Args:
            input_length: Length of input signal

        Returns:
            Tuple of (num_windows, context_length, num_channels)
        """
        # Account for resampling
        if self.enable_resampling:
            ratio = self.sampling_rate_target / self.sampling_rate_original
            length_after_resample = int(input_length * ratio)
        else:
            length_after_resample = input_length

        # Compute number of windows (using training stride as default)
        if length_after_resample < self.context_length:
            num_windows = 0
        else:
            num_windows = (
                length_after_resample - self.context_length
            ) // self.window_stride_train + 1

        return (num_windows, self.context_length, 3)
