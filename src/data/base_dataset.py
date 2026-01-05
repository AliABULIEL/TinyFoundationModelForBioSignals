"""Abstract base class for accelerometry datasets."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BaseAccelerometryDataset(ABC, Dataset):
    """
    Abstract base class for accelerometry-based HAR datasets.

    All dataset adapters (CAPTURE-24, WISDM, PAMAP2, etc.) must inherit from
    this class and implement the abstract methods.

    This ensures:
    1. Consistent interface across datasets
    2. Easy addition of new datasets
    3. Separation of dataset-specific logic from core pipeline

    Args:
        data_path: Path to dataset directory
        participant_ids: List of participant IDs to include (None = all)
        transform: Optional transform to apply to samples

    Example:
        >>> # Implement a new dataset adapter
        >>> class MyDataset(BaseAccelerometryDataset):
        >>>     def load_participant(self, pid):
        >>>         # Load from your format
        >>>         return signal, labels
        >>>     # ... implement other methods ...
    """

    def __init__(
        self,
        data_path: str,
        participant_ids: Optional[List[str]] = None,
        transform: Optional[callable] = None,
    ) -> None:
        """Initialize base dataset."""
        super().__init__()

        self.data_path = Path(data_path)
        self.transform = transform

        # Validate data path
        if not self.data_path.exists():
            logger.warning(
                f"Dataset path does not exist: {self.data_path}\n"
                f"  This will cause errors when loading data.\n"
                f"  Hint: Download dataset or check path configuration"
            )

        # Get participant IDs
        all_participants = self.get_participant_ids()

        if participant_ids is None:
            self.participant_ids = all_participants
        else:
            # Validate that requested participants exist
            invalid = set(participant_ids) - set(all_participants)
            if invalid:
                raise ValueError(
                    f"Invalid participant IDs: {invalid}\n"
                    f"  Available: {all_participants[:10]}... (showing first 10)"
                )
            self.participant_ids = participant_ids

        logger.info(
            f"Initialized {self.__class__.__name__}: "
            f"{len(self.participant_ids)} participants"
        )

    @abstractmethod
    def load_participant(self, participant_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load raw accelerometry data and labels for a participant.

        Args:
            participant_id: Participant identifier

        Returns:
            Tuple of (signal, labels) where:
                - signal: np.ndarray of shape (num_samples, 3) - X, Y, Z acceleration
                - labels: np.ndarray of shape (num_samples,) - integer class labels

        Raises:
            FileNotFoundError: If participant data doesn't exist
            ValueError: If data format is invalid

        Note:
            This method must handle dataset-specific file formats, missing data,
            and any necessary preprocessing (e.g., unit conversions).
        """
        pass

    @abstractmethod
    def get_participant_ids(self) -> List[str]:
        """
        Get list of all available participant IDs.

        Returns:
            List of participant IDs as strings

        Example:
            >>> dataset.get_participant_ids()
            ['P001', 'P002', 'P003', ...]
        """
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, any]:
        """
        Get dataset metadata.

        Returns:
            Dictionary containing:
                - sampling_rate: Original sampling rate (Hz)
                - num_channels: Number of accelerometry channels (should be 3)
                - sensor_info: Information about sensor device
                - wear_location: Body location of sensor
                - dynamic_range: Measurement range (g or m/s^2)
                - any other dataset-specific metadata

        Example:
            >>> metadata = dataset.get_metadata()
            >>> metadata['sampling_rate']
            100
        """
        pass

    @abstractmethod
    def get_label_map(self) -> Dict[int, str]:
        """
        Get mapping from class IDs to class names.

        Returns:
            Dictionary mapping integer class IDs to string class names

        Example:
            >>> label_map = dataset.get_label_map()
            >>> label_map[0]
            'Sleep'
        """
        pass

    def __len__(self) -> int:
        """
        Return total number of participants.

        Note:
            Subclasses can override this if they pre-compute windows
            and want to return the total number of windows instead.
        """
        return len(self.participant_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item (participant or window).

        Default implementation loads full participant data. Subclasses should
        override this to return pre-windowed samples if windowing is performed
        during initialization.

        Args:
            idx: Index of item to retrieve

        Returns:
            Dictionary containing:
                - 'signal': torch.Tensor of shape (length, 3) or (context_length, 3)
                - 'label': torch.Tensor (single label or label sequence)
                - 'participant_id': str
                - any other metadata

        """
        participant_id = self.participant_ids[idx]

        # Load participant data
        signal, labels = self.load_participant(participant_id)

        # Convert to tensors
        signal = torch.from_numpy(signal).float()

        # For continuous data, we'll return the first label as representative
        # (Actual windowing happens in DataModule)
        if labels.ndim == 1 and len(labels) > 0:
            label = torch.tensor(labels[0], dtype=torch.long)
        else:
            label = torch.from_numpy(labels).long()

        # Create sample dictionary
        sample = {
            "signal": signal,
            "label": label,
            "participant_id": participant_id,
        }

        # Apply transform if provided
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def get_participant_data(self, participant_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convenience method to get data for a specific participant.

        Args:
            participant_id: Participant identifier

        Returns:
            Tuple of (signal, labels)
        """
        return self.load_participant(participant_id)

    def get_num_classes(self) -> int:
        """
        Get number of activity classes.

        Returns:
            Number of classes
        """
        return len(self.get_label_map())

    def get_class_distribution(self) -> Dict[int, float]:
        """
        Compute class distribution across all participants.

        Returns:
            Dictionary mapping class IDs to their frequency

        Note:
            This can be slow for large datasets as it loads all data.
            Subclasses should cache this if possible.
        """
        logger.info("Computing class distribution (this may take a while)...")

        label_counts = {}
        total_samples = 0

        for pid in self.participant_ids:
            _, labels = self.load_participant(pid)

            for label in np.unique(labels):
                count = np.sum(labels == label)
                label_counts[label] = label_counts.get(label, 0) + count
                total_samples += count

        # Convert to frequencies
        distribution = {
            label: count / total_samples for label, count in label_counts.items()
        }

        logger.info(f"Class distribution computed across {total_samples} samples")

        return distribution

    def compute_class_weights(self, method: str = "inverse_freq") -> np.ndarray:
        """
        Compute class weights for handling imbalance.

        Args:
            method: Weighting method:
                - "inverse_freq": Weight inversely proportional to frequency
                - "balanced": Similar to sklearn's "balanced" mode

        Returns:
            Array of class weights of length num_classes

        Example:
            >>> weights = dataset.compute_class_weights()
            >>> weights  # Higher weights for rare classes
            array([1.2, 1.0, 3.5, 8.0, 15.0])
        """
        distribution = self.get_class_distribution()
        num_classes = self.get_num_classes()

        # Ensure all classes are represented
        frequencies = np.array([distribution.get(i, 1e-10) for i in range(num_classes)])

        if method == "inverse_freq":
            weights = 1.0 / (frequencies + 1e-10)
            # Normalize so minimum weight is 1.0
            weights = weights / np.min(weights)

        elif method == "balanced":
            weights = num_classes / (num_classes * frequencies + 1e-10)

        else:
            raise ValueError(f"Unknown weighting method: {method}")

        logger.info(f"Computed class weights using '{method}': {weights}")

        return weights
