"""DataModule for managing datasets and dataloaders.

This module provides the data pipeline for HAR training.
Only real data is supported - no synthetic data generation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.data.base_dataset import BaseAccelerometryDataset
from src.data.capture24_adapter import CAPTURE24Dataset
from src.data.transforms import get_transform
from src.preprocessing.pipeline import PreprocessingPipeline

logger = logging.getLogger(__name__)


def worker_init_fn(worker_id: int) -> None:
    """
    Initialize each DataLoader worker with a unique random seed.

    This ensures reproducibility while allowing different workers
    to generate different random augmentations.

    Args:
        worker_id: Worker process ID
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    import random
    random.seed(worker_seed)


class WindowedDataset(Dataset):
    """
    Lazy-loading windowed dataset for accelerometry data.

    ⚠️ CRITICAL: This dataset implements STRICT LAZY LOADING for scalability.
    It stores ONLY metadata during initialization and reads windows from disk
    on-demand in __getitem__. This allows handling datasets with >150 participants
    on systems with limited RAM (32GB).

    Architecture:
    - __init__: Scans files, stores window metadata (participant_id, start_index, label)
    - __getitem__: Reads specific window from disk, applies preprocessing on-the-fly

    ANTI-PATTERNS PROHIBITED:
    - Storing full signals in memory
    - Pre-computing all windows at init time
    - Eager concatenation of participant data

    Args:
        base_dataset: Base accelerometry dataset
        preprocessing_pipeline: Preprocessing pipeline
        participant_indices: Indices of participants to include
        is_training: Whether this is training data (affects windowing stride)
        transform: Optional augmentation transform

    Example:
        >>> base_ds = CAPTURE24Dataset("data/capture24")
        >>> pipeline = PreprocessingPipeline(config)
        >>> windowed_ds = WindowedDataset(base_ds, pipeline, [0, 1, 2], is_training=True)
        >>> # No data loaded yet - only metadata stored
        >>> sample = windowed_ds[0]  # Loads single window on demand
    """

    def __init__(
        self,
        base_dataset: BaseAccelerometryDataset,
        preprocessing_pipeline: PreprocessingPipeline,
        participant_indices: List[int],
        is_training: bool = True,
        transform: Optional[callable] = None,
    ) -> None:
        """Initialize lazy windowed dataset with metadata only."""
        self.base_dataset = base_dataset
        self.preprocessing_pipeline = preprocessing_pipeline
        self.participant_indices = participant_indices
        self.is_training = is_training
        self.transform = transform

        # ═══════════════════════════════════════════════════════════════
        # LAZY LOADING: Store ONLY metadata, NO signal data
        # ═══════════════════════════════════════════════════════════════
        logger.info(
            f"Building window metadata index for {len(participant_indices)} participants "
            f"(is_training={is_training})... [LAZY MODE]"
        )

        # Metadata storage (lightweight)
        self.window_metadata = []  # List of (participant_id, window_start, window_end, label)
        self.participant_ids = []  # participant_id for each window
        self.labels = []  # label for each window

        for idx in participant_indices:
            participant_id = base_dataset.participant_ids[idx]

            try:
                # Load ONLY labels to determine window positions
                # We'll load signals on-demand in __getitem__
                signal, labels = base_dataset.load_participant(participant_id)

                # Get window parameters
                context_length = preprocessing_pipeline.context_length
                stride = (
                    preprocessing_pipeline.window_stride_train
                    if is_training
                    else preprocessing_pipeline.window_stride_eval
                )

                # Calculate window positions WITHOUT actually windowing
                signal_length = len(signal)

                # Apply resampling scaling to signal length
                if preprocessing_pipeline.enable_resampling:
                    ratio = (
                        preprocessing_pipeline.sampling_rate_target
                        / preprocessing_pipeline.sampling_rate_original
                    )
                    signal_length = int(signal_length * ratio)

                # Compute window start positions
                num_windows = max(0, (signal_length - context_length) // stride + 1)

                for window_idx in range(num_windows):
                    start = window_idx * stride
                    end = start + context_length

                    # Get label for this window (center sample)
                    label_idx = start + context_length // 2
                    if label_idx < len(labels):
                        window_label = labels[label_idx]
                    else:
                        window_label = labels[-1]

                    # Store metadata ONLY (no signal data)
                    self.window_metadata.append({
                        "participant_id": participant_id,
                        "participant_idx": idx,
                        "window_start": start,
                        "window_end": end,
                        "label": int(window_label),
                    })
                    self.participant_ids.append(participant_id)
                    self.labels.append(int(window_label))

                logger.debug(
                    f"  {participant_id}: {num_windows} windows indexed "
                    f"(signal_length={signal_length}, stride={stride})"
                )

            except Exception as e:
                logger.error(
                    f"Failed to index participant {participant_id}: {e}\n"
                    f"  Skipping this participant."
                )
                continue

        # Convert labels to numpy for efficient class distribution computation
        self.labels = np.array(self.labels, dtype=np.int64)

        logger.info(
            f"✓ Lazy dataset ready: {len(self.window_metadata)} windows indexed from "
            f"{len(participant_indices)} participants (0 MB signal data in memory)"
        )

        # Log class distribution
        unique, counts = np.unique(self.labels, return_counts=True)
        for label, count in zip(unique, counts):
            logger.debug(f"  Class {label}: {count} windows ({count/len(self.labels)*100:.1f}%)")

    def __len__(self) -> int:
        """Return total number of windows."""
        return len(self.window_metadata)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single window by loading it from disk on-demand.

        This method:
        1. Retrieves window metadata (participant_id, start, end)
        2. Loads ONLY the relevant participant's raw data
        3. Applies preprocessing (resampling, gravity removal, normalization)
        4. Extracts the specific window
        5. Returns as tensor

        Args:
            idx: Window index

        Returns:
            Dictionary containing:
                - 'signal': torch.Tensor of shape (context_length, num_channels)
                - 'label': torch.Tensor (scalar)
                - 'participant_id': str
        """
        # Get window metadata
        meta = self.window_metadata[idx]
        participant_id = meta["participant_id"]
        window_start = meta["window_start"]
        window_end = meta["window_end"]
        label = meta["label"]

        # Load participant data (this is the I/O operation)
        signal, labels = self.base_dataset.load_participant(participant_id)

        # Apply preprocessing pipeline
        windowed_data, window_labels = self.preprocessing_pipeline.process(
            signal, labels, is_training=self.is_training
        )

        # Extract the specific window
        # We need to find which window corresponds to our start position
        stride = (
            self.preprocessing_pipeline.window_stride_train
            if self.is_training
            else self.preprocessing_pipeline.window_stride_eval
        )
        window_idx_in_participant = window_start // stride

        # Safety check
        if window_idx_in_participant >= len(windowed_data):
            logger.warning(
                f"Window index {window_idx_in_participant} out of range for participant "
                f"{participant_id} (has {len(windowed_data)} windows). Using last window."
            )
            window_idx_in_participant = len(windowed_data) - 1

        window = windowed_data[window_idx_in_participant]

        # Convert to tensor
        signal_tensor = torch.from_numpy(window).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        sample = {
            "signal": signal_tensor,
            "label": label_tensor,
            "participant_id": participant_id,
        }

        # Apply augmentation transform if provided
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def get_class_distribution(self) -> Dict[int, float]:
        """Get class distribution in this dataset."""
        unique, counts = np.unique(self.labels, return_counts=True)
        total = len(self.labels)
        return {int(label): count / total for label, count in zip(unique, counts)}


class HARDataModule:
    """
    DataModule for Human Activity Recognition.

    Manages dataset creation, splitting, preprocessing, and DataLoader instantiation.
    Ensures subject-independent splits (no subject appears in multiple splits).

    Args:
        config: Complete configuration dictionary
        dataset_name: Name of dataset to use ("capture24", etc.)

    Example:
        >>> config = load_config("configs/default.yaml")
        >>> datamodule = HARDataModule(config, dataset_name="capture24")
        >>> datamodule.setup()
        >>> train_loader = datamodule.train_dataloader()
    """

    def __init__(
        self,
        config: Dict,
        dataset_name: str = "capture24",
    ) -> None:
        """Initialize DataModule."""
        self.config = config
        self.dataset_name = dataset_name

        # Extract configurations
        self.dataset_config = config.get("dataset", {})
        self.preprocessing_config = config.get("preprocessing", {})
        self.training_config = config.get("training", {})
        self.hardware_config = config.get("hardware", {})

        # Initialize preprocessing pipeline
        self.preprocessing_pipeline = PreprocessingPipeline(self.preprocessing_config)

        # Initialize transforms
        aug_config = self.training_config.get("augmentation", None)
        self.train_transform = get_transform(aug_config) if aug_config else None
        self.val_transform = None  # No augmentation for validation/test

        # Datasets (will be created in setup())
        self.base_dataset: Optional[BaseAccelerometryDataset] = None
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        # Class weights
        self.class_weights: Optional[torch.Tensor] = None

        logger.info(
            f"Initialized HARDataModule for {dataset_name} dataset"
        )

    def setup(self) -> None:
        """
        Set up datasets and compute splits.

        This should be called before accessing dataloaders.
        """
        logger.info("Setting up DataModule...")

        # Create base dataset
        self.base_dataset = self._create_base_dataset()

        # Compute subject-independent splits
        train_indices, val_indices, test_indices = self._compute_splits()

        # Log split information
        logger.info(
            f"Subject-independent splits:\n"
            f"  Train: {len(train_indices)} participants\n"
            f"  Val:   {len(val_indices)} participants\n"
            f"  Test:  {len(test_indices)} participants"
        )

        # Verify no subject leakage
        self._verify_no_leakage(train_indices, val_indices, test_indices)

        # Create windowed datasets
        self.train_dataset = WindowedDataset(
            self.base_dataset,
            self.preprocessing_pipeline,
            train_indices,
            is_training=True,
            transform=self.train_transform,
        )

        self.val_dataset = WindowedDataset(
            self.base_dataset,
            self.preprocessing_pipeline,
            val_indices,
            is_training=False,
            transform=self.val_transform,
        )

        self.test_dataset = WindowedDataset(
            self.base_dataset,
            self.preprocessing_pipeline,
            test_indices,
            is_training=False,
            transform=self.val_transform,
        )

        # Compute class weights from training set
        self.class_weights = self._compute_class_weights()

        logger.info("DataModule setup complete")

    def _create_base_dataset(self) -> BaseAccelerometryDataset:
        """Create base dataset based on dataset_name."""
        data_path = self.dataset_config.get("data_path", "data/capture24")
        num_classes = self.dataset_config.get("num_classes", 5)

        if self.dataset_name.lower() == "capture24":
            dataset = CAPTURE24Dataset(
                data_path=data_path,
                num_classes=num_classes,
            )
        else:
            raise ValueError(
                f"Unknown dataset: {self.dataset_name}\n"
                f"  Supported datasets: ['capture24']\n"
                f"  Hint: Implement new dataset adapter for other datasets"
            )

        logger.info(
            f"Created {self.dataset_name} dataset: "
            f"{len(dataset)} participants, {num_classes} classes"
        )

        return dataset

    def _compute_splits(self) -> Tuple[List[int], List[int], List[int]]:
        """
        Compute subject-independent train/val/test splits.

        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        num_participants = len(self.base_dataset)

        # Get split ratios
        train_split = self.dataset_config.get("train_split", 0.7)
        val_split = self.dataset_config.get("val_split", 0.15)
        test_split = self.dataset_config.get("test_split", 0.15)

        # Validate ratios
        total = train_split + val_split + test_split
        if not np.isclose(total, 1.0):
            logger.warning(
                f"Split ratios don't sum to 1.0 (got {total}). Normalizing..."
            )
            train_split /= total
            val_split /= total
            test_split /= total

        # Compute split sizes
        n_train = int(num_participants * train_split)
        n_val = int(num_participants * val_split)
        n_test = num_participants - n_train - n_val

        # Shuffle participants
        rng = np.random.RandomState(self.config.get("experiment", {}).get("seed", 42))
        all_indices = np.arange(num_participants)
        rng.shuffle(all_indices)

        # Split
        train_indices = all_indices[:n_train].tolist()
        val_indices = all_indices[n_train:n_train + n_val].tolist()
        test_indices = all_indices[n_train + n_val:].tolist()

        return train_indices, val_indices, test_indices

    def _verify_no_leakage(
        self,
        train_indices: List[int],
        val_indices: List[int],
        test_indices: List[int],
    ) -> None:
        """
        Verify no subject appears in multiple splits.

        Raises:
            AssertionError: If subject leakage is detected
        """
        train_set = set(train_indices)
        val_set = set(val_indices)
        test_set = set(test_indices)

        train_val_overlap = train_set & val_set
        train_test_overlap = train_set & test_set
        val_test_overlap = val_set & test_set

        if train_val_overlap:
            raise AssertionError(
                f"Subject leakage between train and val: {train_val_overlap}"
            )

        if train_test_overlap:
            raise AssertionError(
                f"Subject leakage between train and test: {train_test_overlap}"
            )

        if val_test_overlap:
            raise AssertionError(
                f"Subject leakage between val and test: {val_test_overlap}"
            )

        logger.debug("✓ No subject leakage detected")

    def _compute_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for handling imbalance.

        Returns:
            Tensor of class weights
        """
        # Check if weights are provided in config
        config_weights = self.dataset_config.get("class_weights", None)

        if config_weights is not None:
            weights = torch.tensor(config_weights, dtype=torch.float32)
            logger.info(f"Using class weights from config: {weights}")
            return weights

        # Compute from training set distribution
        distribution = self.train_dataset.get_class_distribution()
        num_classes = self.base_dataset.get_num_classes()

        # Inverse frequency weighting
        frequencies = np.array([distribution.get(i, 1e-10) for i in range(num_classes)])
        weights = 1.0 / (frequencies + 1e-10)

        # Normalize so minimum weight is 1.0
        weights = weights / np.min(weights)

        weights = torch.from_numpy(weights).float()

        logger.info(f"Computed class weights: {weights}")

        return weights

    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader."""
        if self.train_dataset is None:
            raise RuntimeError("Must call setup() before accessing dataloaders")

        batch_size = self.training_config.get("batch_size", 64)
        num_workers = self.hardware_config.get("num_workers", 4)
        pin_memory = self.hardware_config.get("pin_memory", True)

        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,  # Drop incomplete batches for stable training
            worker_init_fn=worker_init_fn,  # Reproducible worker seeding
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader."""
        if self.val_dataset is None:
            raise RuntimeError("Must call setup() before accessing dataloaders")

        batch_size = self.training_config.get("batch_size", 64)
        num_workers = self.hardware_config.get("num_workers", 4)
        pin_memory = self.hardware_config.get("pin_memory", True)

        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=worker_init_fn,  # Reproducible worker seeding
        )

    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader."""
        if self.test_dataset is None:
            raise RuntimeError("Must call setup() before accessing dataloaders")

        batch_size = self.training_config.get("batch_size", 64)
        num_workers = self.hardware_config.get("num_workers", 4)
        pin_memory = self.hardware_config.get("pin_memory", True)

        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=worker_init_fn,  # Reproducible worker seeding
        )

    def get_num_classes(self) -> int:
        """Get number of activity classes."""
        return self.base_dataset.get_num_classes()

    def get_class_weights(self) -> torch.Tensor:
        """Get class weights for loss function."""
        if self.class_weights is None:
            raise RuntimeError("Must call setup() before accessing class weights")

        return self.class_weights

    def get_label_map(self) -> Dict[int, str]:
        """Get label mapping."""
        return self.base_dataset.get_label_map()
