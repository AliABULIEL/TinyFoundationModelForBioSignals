"""Data augmentation transforms for accelerometry data."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class AugmentationPipeline:
    """
    Composable augmentation pipeline for accelerometry data.

    Applies a sequence of augmentation transforms with specified probabilities.

    Args:
        transforms: List of transform objects
        p: Probability of applying each transform (default: 0.5)

    Example:
        >>> pipeline = AugmentationPipeline([
        >>>     ScaleAugmentation(scale_range=(0.9, 1.1)),
        >>>     NoiseAugmentation(noise_std=0.01),
        >>> ], p=0.5)
        >>> augmented = pipeline(sample)
    """

    def __init__(
        self,
        transforms: List[callable],
        p: float = 0.5,
    ) -> None:
        """Initialize augmentation pipeline."""
        self.transforms = transforms
        self.p = p

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply augmentations to sample.

        Args:
            sample: Dictionary with 'signal' and 'label' keys

        Returns:
            Augmented sample
        """
        for transform in self.transforms:
            if np.random.rand() < self.p:
                sample = transform(sample)

        return sample


class ScaleAugmentation:
    """
    Scale augmentation for accelerometry data.

    Randomly scales the signal magnitude to simulate different sensor gains
    or movement intensities.

    Args:
        scale_range: Tuple of (min_scale, max_scale)

    Example:
        >>> aug = ScaleAugmentation(scale_range=(0.9, 1.1))
        >>> augmented = aug(sample)
    """

    def __init__(self, scale_range: tuple = (0.9, 1.1)) -> None:
        """Initialize scale augmentation."""
        self.scale_min, self.scale_max = scale_range

        if self.scale_min >= self.scale_max:
            raise ValueError(
                f"scale_min must be < scale_max.\n"
                f"  Received: ({self.scale_min}, {self.scale_max})"
            )

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply scale augmentation."""
        scale_factor = np.random.uniform(self.scale_min, self.scale_max)

        sample["signal"] = sample["signal"] * scale_factor

        return sample


class NoiseAugmentation:
    """
    Add Gaussian noise to accelerometry signal.

    Simulates sensor noise and measurement uncertainty.

    Args:
        noise_std: Standard deviation of Gaussian noise

    Example:
        >>> aug = NoiseAugmentation(noise_std=0.01)
        >>> augmented = aug(sample)
    """

    def __init__(self, noise_std: float = 0.01) -> None:
        """Initialize noise augmentation."""
        if noise_std < 0:
            raise ValueError(f"noise_std must be non-negative, got {noise_std}")

        self.noise_std = noise_std

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply noise augmentation."""
        signal = sample["signal"]

        noise = torch.randn_like(signal) * self.noise_std
        sample["signal"] = signal + noise

        return sample


class RotationAugmentation:
    """
    Random 3D rotation augmentation for accelerometry data.

    Simulates different sensor orientations on the wrist.

    Args:
        max_angle_deg: Maximum rotation angle in degrees

    Example:
        >>> aug = RotationAugmentation(max_angle_deg=15)
        >>> augmented = aug(sample)
    """

    def __init__(self, max_angle_deg: float = 15.0) -> None:
        """Initialize rotation augmentation."""
        self.max_angle_deg = max_angle_deg

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply rotation augmentation."""
        signal = sample["signal"]  # Shape: (length, 3)

        # Generate random rotation matrix
        rotation_matrix = self._random_rotation_matrix()
        rotation_matrix = torch.from_numpy(rotation_matrix).float()

        # Apply rotation: (L, 3) @ (3, 3)^T = (L, 3)
        sample["signal"] = torch.matmul(signal, rotation_matrix.T)

        return sample

    def _random_rotation_matrix(self) -> np.ndarray:
        """
        Generate random 3D rotation matrix.

        Returns:
            3x3 rotation matrix
        """
        # Random rotation angle
        angle = np.random.uniform(-self.max_angle_deg, self.max_angle_deg)
        angle_rad = np.deg2rad(angle)

        # Random rotation axis (normalized)
        axis = np.random.randn(3)
        axis = axis / (np.linalg.norm(axis) + 1e-10)

        # Rodrigues' rotation formula
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])

        R = (
            np.eye(3)
            + np.sin(angle_rad) * K
            + (1 - np.cos(angle_rad)) * np.matmul(K, K)
        )

        return R


class TimeWarpAugmentation:
    """
    Time warping augmentation.

    Randomly stretches or compresses segments of the time series to simulate
    variations in movement speed.

    Args:
        warp_range: Tuple of (min_warp, max_warp) where 1.0 is no warping

    Example:
        >>> aug = TimeWarpAugmentation(warp_range=(0.8, 1.2))
        >>> augmented = aug(sample)
    """

    def __init__(self, warp_range: tuple = (0.9, 1.1)) -> None:
        """Initialize time warp augmentation."""
        self.warp_min, self.warp_max = warp_range

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply time warp augmentation."""
        signal = sample["signal"]  # Shape: (length, 3)
        length = signal.shape[0]

        # Random warp factor
        warp_factor = np.random.uniform(self.warp_min, self.warp_max)

        # Compute new length
        new_length = int(length * warp_factor)

        if new_length == length:
            return sample

        # Resample to new length using linear interpolation
        original_indices = torch.linspace(0, length - 1, length)
        new_indices = torch.linspace(0, length - 1, new_length)

        warped_signal = torch.zeros((new_length, 3))

        for ch in range(3):
            warped_signal[:, ch] = torch.nn.functional.interpolate(
                signal[:, ch].unsqueeze(0).unsqueeze(0),
                size=new_length,
                mode='linear',
                align_corners=True
            ).squeeze()

        # Resize back to original length
        final_signal = torch.nn.functional.interpolate(
            warped_signal.T.unsqueeze(0),
            size=length,
            mode='linear',
            align_corners=True
        ).squeeze(0).T

        sample["signal"] = final_signal

        return sample


class ChannelShuffleAugmentation:
    """
    Randomly shuffle X, Y, Z channels.

    Simulates different sensor orientations.

    Args:
        p_shuffle: Probability of shuffling

    Example:
        >>> aug = ChannelShuffleAugmentation(p_shuffle=0.3)
        >>> augmented = aug(sample)
    """

    def __init__(self, p_shuffle: float = 0.3) -> None:
        """Initialize channel shuffle augmentation."""
        self.p_shuffle = p_shuffle

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply channel shuffle augmentation."""
        if np.random.rand() < self.p_shuffle:
            signal = sample["signal"]  # Shape: (length, 3)

            # Random permutation of channels
            perm = torch.randperm(3)
            sample["signal"] = signal[:, perm]

        return sample


class MaskingAugmentation:
    """
    Random masking augmentation (similar to SpecAugment for time series).

    Randomly masks segments of the time series to improve robustness.

    Args:
        mask_ratio: Fraction of time steps to mask
        mask_value: Value to use for masked regions (default: 0)

    Example:
        >>> aug = MaskingAugmentation(mask_ratio=0.1)
        >>> augmented = aug(sample)
    """

    def __init__(self, mask_ratio: float = 0.1, mask_value: float = 0.0) -> None:
        """Initialize masking augmentation."""
        if not 0 <= mask_ratio <= 1:
            raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")

        self.mask_ratio = mask_ratio
        self.mask_value = mask_value

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply masking augmentation."""
        signal = sample["signal"]  # Shape: (length, 3)
        length = signal.shape[0]

        num_masked = int(length * self.mask_ratio)

        if num_masked > 0:
            # Random start position
            start = np.random.randint(0, length - num_masked + 1)
            end = start + num_masked

            # Apply mask
            signal[start:end] = self.mask_value

        sample["signal"] = signal

        return sample


class MagnitudeWarpAugmentation:
    """
    Magnitude warping augmentation.

    Applies smooth random scaling along the time axis using cubic spline.

    Args:
        sigma: Standard deviation of the random cubic spline
        num_knots: Number of knots for the spline

    Example:
        >>> aug = MagnitudeWarpAugmentation(sigma=0.2, num_knots=4)
        >>> augmented = aug(sample)
    """

    def __init__(self, sigma: float = 0.2, num_knots: int = 4) -> None:
        """Initialize magnitude warp augmentation."""
        self.sigma = sigma
        self.num_knots = num_knots

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply magnitude warp augmentation."""
        signal = sample["signal"]  # Shape: (length, 3)
        length = signal.shape[0]

        # Generate random warp curve
        from scipy.interpolate import CubicSpline

        # Random knot positions
        knot_indices = np.linspace(0, length - 1, self.num_knots, dtype=int)
        knot_values = np.random.normal(1.0, self.sigma, self.num_knots)

        # Create smooth warp curve
        cs = CubicSpline(knot_indices, knot_values)
        warp_curve = cs(np.arange(length))

        # Apply warp
        warp_curve = torch.from_numpy(warp_curve).float()
        sample["signal"] = signal * warp_curve.unsqueeze(1)

        return sample


def get_transform(config: Optional[Dict[str, Any]] = None) -> Optional[AugmentationPipeline]:
    """
    Get augmentation pipeline from configuration.

    Args:
        config: Augmentation configuration dictionary

    Returns:
        AugmentationPipeline or None if augmentation is disabled

    Example:
        >>> config = {
        >>>     "enabled": True,
        >>>     "p": 0.5,
        >>>     "transforms": {
        >>>         "scale": {"scale_range": [0.9, 1.1]},
        >>>         "noise": {"noise_std": 0.01},
        >>>     }
        >>> }
        >>> transform = get_transform(config)
    """
    if config is None or not config.get("enabled", False):
        return None

    transforms = []

    # Parse transform configurations
    transform_configs = config.get("transforms", {})

    if "scale" in transform_configs:
        scale_cfg = transform_configs["scale"]
        transforms.append(ScaleAugmentation(**scale_cfg))

    if "noise" in transform_configs:
        noise_cfg = transform_configs["noise"]
        transforms.append(NoiseAugmentation(**noise_cfg))

    if "rotation" in transform_configs:
        rotation_cfg = transform_configs["rotation"]
        transforms.append(RotationAugmentation(**rotation_cfg))

    if "time_warp" in transform_configs:
        warp_cfg = transform_configs["time_warp"]
        transforms.append(TimeWarpAugmentation(**warp_cfg))

    if "channel_shuffle" in transform_configs:
        shuffle_cfg = transform_configs["channel_shuffle"]
        transforms.append(ChannelShuffleAugmentation(**shuffle_cfg))

    if "masking" in transform_configs:
        mask_cfg = transform_configs["masking"]
        transforms.append(MaskingAugmentation(**mask_cfg))

    if "magnitude_warp" in transform_configs:
        mag_warp_cfg = transform_configs["magnitude_warp"]
        transforms.append(MagnitudeWarpAugmentation(**mag_warp_cfg))

    if not transforms:
        logger.warning("Augmentation enabled but no transforms configured")
        return None

    p = config.get("p", 0.5)
    pipeline = AugmentationPipeline(transforms, p=p)

    logger.info(f"Created augmentation pipeline with {len(transforms)} transforms (p={p})")

    return pipeline
