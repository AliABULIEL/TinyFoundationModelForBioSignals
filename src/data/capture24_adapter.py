"""CAPTURE-24 dataset adapter for TTM-HAR."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.base_dataset import BaseAccelerometryDataset
from src.data.label_mappings import CAPTURE24_5CLASS, get_label_mapping

logger = logging.getLogger(__name__)


class CAPTURE24Dataset(BaseAccelerometryDataset):
    """
    CAPTURE-24 dataset adapter.

    CAPTURE-24 is a large-scale wrist-worn accelerometry dataset for 24-hour
    activity recognition collected from ~150 participants wearing Axivity AX3
    sensors on the dominant wrist.

    Data format expectations:
        - Root directory contains participant folders: P001/, P002/, etc.
        - Each participant folder contains:
            - accelerometry.csv (or .npy): Time series data with X, Y, Z columns
            - labels.csv (or .npy): Corresponding activity labels
            - metadata.json (optional): Participant-specific metadata

    Args:
        data_path: Path to CAPTURE-24 root directory
        participant_ids: List of participant IDs to load (None = all)
        num_classes: Number of activity classes (5 or 8)
        transform: Optional transform to apply to samples
        use_synthetic: If True and data doesn't exist, generate synthetic data for testing

    Example:
        >>> dataset = CAPTURE24Dataset(
        >>>     data_path="data/capture24",
        >>>     num_classes=5
        >>> )
        >>> signal, labels = dataset.load_participant("P001")
    """

    def __init__(
        self,
        data_path: str,
        participant_ids: Optional[List[str]] = None,
        num_classes: int = 5,
        transform: Optional[callable] = None,
        use_synthetic: bool = False,
    ) -> None:
        """Initialize CAPTURE-24 dataset."""
        self.num_classes = num_classes
        self.use_synthetic = use_synthetic

        # Get label mapping
        self.label_mapping = get_label_mapping("capture24", num_classes)

        # Initialize base class
        super().__init__(
            data_path=data_path,
            participant_ids=participant_ids,
            transform=transform,
        )

        logger.info(
            f"Initialized CAPTURE-24 dataset: "
            f"{len(self.participant_ids)} participants, "
            f"{num_classes} classes"
        )

    def load_participant(self, participant_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load accelerometry data and labels for a participant.

        Args:
            participant_id: Participant ID (e.g., "P001")

        Returns:
            Tuple of (signal, labels) where:
                - signal: np.ndarray of shape (num_samples, 3) - X, Y, Z acceleration
                - labels: np.ndarray of shape (num_samples,) - integer class labels

        Raises:
            FileNotFoundError: If participant data doesn't exist
            ValueError: If data format is invalid
        """
        participant_dir = self.data_path / participant_id

        # Check if data exists
        if not participant_dir.exists():
            if self.use_synthetic:
                logger.warning(
                    f"Participant {participant_id} not found at {participant_dir}. "
                    f"Generating synthetic data for testing."
                )
                return self._generate_synthetic_data()
            else:
                raise FileNotFoundError(
                    f"Participant directory not found: {participant_dir}\n"
                    f"  Hint: Download CAPTURE-24 dataset or enable use_synthetic=True\n"
                    f"  Expected structure: {self.data_path}/P001/, P002/, ..."
                )

        # Try loading accelerometry data
        signal = self._load_accelerometry(participant_dir)

        # Try loading labels
        labels = self._load_labels(participant_dir)

        # Validate shapes match
        if signal.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Signal and labels length mismatch for {participant_id}.\n"
                f"  Signal: {signal.shape[0]} samples\n"
                f"  Labels: {labels.shape[0]} samples\n"
                f"  Hint: Check that data files are synchronized"
            )

        # Validate signal shape
        if signal.shape[1] != 3:
            raise ValueError(
                f"Expected 3 channels (X, Y, Z), got {signal.shape[1]}\n"
                f"  Hint: Reshape data to (num_samples, 3)"
            )

        # Validate label range
        unique_labels = np.unique(labels)
        invalid_labels = [l for l in unique_labels if l < 0 or l >= self.num_classes]
        if invalid_labels:
            raise ValueError(
                f"Invalid label values for {participant_id}: {invalid_labels}\n"
                f"  Valid range: [0, {self.num_classes-1}]\n"
                f"  Hint: Check label mapping or num_classes configuration"
            )

        logger.debug(
            f"Loaded {participant_id}: "
            f"{signal.shape[0]} samples, "
            f"{len(unique_labels)} unique labels"
        )

        return signal, labels

    def _load_accelerometry(self, participant_dir: Path) -> np.ndarray:
        """Load accelerometry data from participant directory."""
        # Try different file formats
        for filename in ["accelerometry.npy", "accelerometry.csv", "acc.npy", "data.npy"]:
            filepath = participant_dir / filename

            if filepath.exists():
                if filename.endswith(".npy"):
                    signal = np.load(filepath)
                elif filename.endswith(".csv"):
                    df = pd.read_csv(filepath)
                    # Try common column names
                    if all(col in df.columns for col in ["X", "Y", "Z"]):
                        signal = df[["X", "Y", "Z"]].values
                    elif all(col in df.columns for col in ["x", "y", "z"]):
                        signal = df[["x", "y", "z"]].values
                    elif len(df.columns) >= 3:
                        # Assume first 3 columns are X, Y, Z
                        signal = df.iloc[:, :3].values
                        logger.warning(
                            f"Using first 3 columns as X, Y, Z from {filepath}"
                        )
                    else:
                        raise ValueError(
                            f"Cannot find X, Y, Z columns in {filepath}\n"
                            f"  Available columns: {df.columns.tolist()}"
                        )

                logger.debug(f"Loaded accelerometry from {filepath}")
                return signal.astype(np.float32)

        raise FileNotFoundError(
            f"No accelerometry file found in {participant_dir}\n"
            f"  Expected files: accelerometry.npy, accelerometry.csv, acc.npy, or data.npy"
        )

    def _load_labels(self, participant_dir: Path) -> np.ndarray:
        """Load labels from participant directory."""
        # Try different file formats
        for filename in ["labels.npy", "labels.csv", "annotations.npy", "annotations.csv"]:
            filepath = participant_dir / filename

            if filepath.exists():
                if filename.endswith(".npy"):
                    labels = np.load(filepath)
                elif filename.endswith(".csv"):
                    df = pd.read_csv(filepath)
                    # Try common column names
                    if "label" in df.columns:
                        labels = df["label"].values
                    elif "activity" in df.columns:
                        labels = df["activity"].values
                    elif "class" in df.columns:
                        labels = df["class"].values
                    elif len(df.columns) == 1:
                        labels = df.iloc[:, 0].values
                    else:
                        raise ValueError(
                            f"Cannot find label column in {filepath}\n"
                            f"  Available columns: {df.columns.tolist()}"
                        )

                logger.debug(f"Loaded labels from {filepath}")
                return labels.astype(np.int64)

        raise FileNotFoundError(
            f"No labels file found in {participant_dir}\n"
            f"  Expected files: labels.npy, labels.csv, annotations.npy, or annotations.csv"
        )

    def _generate_synthetic_data(
        self, duration_hours: float = 24.0, sampling_rate: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic accelerometry data for testing.

        Args:
            duration_hours: Duration of synthetic data in hours
            sampling_rate: Sampling rate in Hz

        Returns:
            Tuple of (signal, labels)
        """
        num_samples = int(duration_hours * 3600 * sampling_rate)

        # Generate realistic-looking accelerometry with ~1g mean (gravity)
        # and small variations (body movement)
        signal = np.random.randn(num_samples, 3) * 0.3 + np.array([0.0, 0.0, 1.0])

        # Generate random labels with realistic distribution
        # Sleep/Sedentary: 70%, Light: 20%, Moderate: 8%, Vigorous: 2%
        label_probs = [0.35, 0.35, 0.20, 0.08, 0.02]
        if self.num_classes < 5:
            label_probs = label_probs[:self.num_classes]
            label_probs = np.array(label_probs) / np.sum(label_probs)

        labels = np.random.choice(
            self.num_classes,
            size=num_samples,
            p=label_probs[:self.num_classes]
        )

        logger.info(
            f"Generated synthetic data: {num_samples} samples "
            f"({duration_hours}h at {sampling_rate}Hz)"
        )

        return signal.astype(np.float32), labels.astype(np.int64)

    def get_participant_ids(self) -> List[str]:
        """
        Get list of all available participant IDs.

        Returns:
            List of participant IDs sorted alphabetically
        """
        if not self.data_path.exists():
            logger.warning(
                f"Data path does not exist: {self.data_path}\n"
                f"  Returning empty participant list."
            )
            if self.use_synthetic:
                # Return synthetic participant IDs for testing
                return [f"P{i:03d}" for i in range(1, 11)]  # P001 to P010
            return []

        # Find all directories that match participant ID pattern (e.g., P001, P002)
        participant_dirs = [
            d.name for d in self.data_path.iterdir()
            if d.is_dir() and (d.name.startswith("P") or d.name.isdigit())
        ]

        if not participant_dirs:
            logger.warning(
                f"No participant directories found in {self.data_path}\n"
                f"  Expected pattern: P001/, P002/, ..."
            )
            if self.use_synthetic:
                return [f"P{i:03d}" for i in range(1, 11)]

        return sorted(participant_dirs)

    def get_metadata(self) -> Dict[str, any]:
        """
        Get CAPTURE-24 dataset metadata.

        Returns:
            Dictionary containing dataset metadata
        """
        return {
            "dataset_name": "CAPTURE-24",
            "sampling_rate": 100,  # Hz
            "num_channels": 3,
            "sensor_info": "Axivity AX3",
            "wear_location": "dominant_wrist",
            "dynamic_range": 8,  # ±8g
            "num_participants": len(self.participant_ids),
            "num_classes": self.num_classes,
            "duration_per_participant": "~24 hours",
        }

    def get_label_map(self) -> Dict[int, str]:
        """
        Get mapping from class IDs to class names.

        Returns:
            Dictionary mapping integer class IDs to string class names
        """
        return self.label_mapping
