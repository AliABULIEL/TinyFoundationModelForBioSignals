"""CAPTURE-24 dataset adapter for TTM-HAR.

This module provides the dataset adapter for CAPTURE-24 accelerometry data.
Real data is REQUIRED - no synthetic data generation is supported.

Supports both data formats:
- File format: P001.csv.gz, P002.csv.gz, ... (official CAPTURE-24 format)
- Directory format: P001/, P002/, ... (preprocessed format)
"""

import logging
import re
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

    Supports two data formats:
        1. Official CAPTURE-24 format (P001.csv.gz, P002.csv.gz, ...)
           - Single compressed CSV per participant
           - Columns: time, x, y, z, annotation (+ label columns like Walmsley2020)
        
        2. Preprocessed directory format (P001/, P002/, ...)
           - Each participant folder contains:
             - accelerometry.csv (or .npy): Time series data with X, Y, Z columns
             - labels.csv (or .npy): Corresponding activity labels

    Args:
        data_path: Path to CAPTURE-24 root directory
        participant_ids: List of participant IDs to load (None = all)
        num_classes: Number of activity classes (5 or 8)
        transform: Optional transform to apply to samples
        target_sampling_rate: Target sampling rate in Hz (default 30 for TTM)

    Raises:
        FileNotFoundError: If data_path does not exist or contains no data

    Example:
        >>> dataset = CAPTURE24Dataset(
        >>>     data_path="data/capture24",
        >>>     num_classes=5
        >>> )
        >>> signal, labels = dataset.load_participant("P001")
    """

    # Class constants
    ORIGINAL_SAMPLING_RATE = 100  # Hz
    DEFAULT_TARGET_RATE = 30  # Hz (for TTM compatibility)

    def __init__(
        self,
        data_path: str,
        participant_ids: Optional[List[str]] = None,
        num_classes: int = 5,
        transform: Optional[callable] = None,
        target_sampling_rate: int = 30,
    ) -> None:
        """Initialize CAPTURE-24 dataset."""
        self.num_classes = num_classes
        self.target_sampling_rate = target_sampling_rate
        self._data_format = None  # Will be detected in get_participant_ids()

        # Get label mapping
        self.label_mapping = get_label_mapping("capture24", num_classes)

        # Validate data path exists
        data_path_obj = Path(data_path)
        if not data_path_obj.exists():
            raise FileNotFoundError(
                f"\n{'=' * 80}\n"
                f"❌ CAPTURE-24 DATA NOT FOUND\n"
                f"{'=' * 80}\n\n"
                f"Data path does not exist: {data_path}\n\n"
                f"REQUIRED: Download the CAPTURE-24 dataset first.\n\n"
                f"DOWNLOAD INSTRUCTIONS:\n"
                f"─────────────────────────────────────────────────────────────────────────────\n"
                f"1. Visit: https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16f6bab45f\n"
                f"2. Download the dataset files\n"
                f"3. Extract to: {data_path}\n"
                f"4. Data can be in either format:\n"
                f"   - P001.csv.gz, P002.csv.gz, ... (official format)\n"
                f"   - P001/, P002/, ... directories (preprocessed format)\n"
                f"─────────────────────────────────────────────────────────────────────────────\n"
                f"{'=' * 80}\n"
            )

        # Initialize base class
        super().__init__(
            data_path=data_path,
            participant_ids=participant_ids,
            transform=transform,
        )

        # Validate we have data
        if len(self.participant_ids) == 0:
            raise FileNotFoundError(
                f"\n{'=' * 80}\n"
                f"❌ NO PARTICIPANTS FOUND\n"
                f"{'=' * 80}\n\n"
                f"Data path exists but contains no participant data: {data_path}\n\n"
                f"Expected formats:\n"
                f"  Format 1 (Official): P001.csv.gz, P002.csv.gz, ...\n"
                f"  Format 2 (Preprocessed):\n"
                f"    {data_path}/\n"
                f"      ├── P001/\n"
                f"      │   ├── accelerometry.npy\n"
                f"      │   └── labels.npy\n"
                f"      └── P002/ ...\n\n"
                f"Please download and extract the CAPTURE-24 dataset correctly.\n"
                f"{'=' * 80}\n"
            )

        logger.info(
            f"Initialized CAPTURE-24 dataset: "
            f"{len(self.participant_ids)} participants, "
            f"{num_classes} classes, "
            f"format={self._data_format}"
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
        # Route to appropriate loader based on detected format
        if self._data_format == 'csv.gz':
            return self._load_from_csv_gz(participant_id)
        elif self._data_format == 'csv':
            return self._load_from_csv(participant_id)
        else:
            return self._load_from_directory(participant_id)

    def _load_from_csv_gz(self, participant_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from compressed CSV file (official CAPTURE-24 format)."""
        csv_path = self.data_path / f"{participant_id}.csv.gz"
        
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Participant file not found: {csv_path}\n"
                f"  Hint: Ensure CAPTURE-24 dataset is downloaded and extracted"
            )
        
        return self._parse_capture24_csv(csv_path)

    def _load_from_csv(self, participant_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from uncompressed CSV file."""
        csv_path = self.data_path / f"{participant_id}.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Participant file not found: {csv_path}\n"
                f"  Hint: Ensure CAPTURE-24 dataset is downloaded and extracted"
            )
        
        return self._parse_capture24_csv(csv_path)

    def _parse_capture24_csv(self, csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse a CAPTURE-24 CSV file (compressed or uncompressed).
        
        CAPTURE-24 CSV format:
        - time: timestamp
        - x, y, z: accelerometer readings (g)
        - annotation: raw activity annotation string
        - Walmsley2020, Willetts2018, etc.: label columns
        """
        logger.debug(f"Loading data from {csv_path}")
        
        # Read CSV (pandas handles .gz automatically)
        # FIX: Use low_memory=False to avoid mixed type warnings on large files
        df = pd.read_csv(csv_path, low_memory=False)
        
        # Find accelerometer columns
        accel_cols = self._find_accel_columns(df)
        signal = df[accel_cols].values.astype(np.float32)
        
        # Handle NaN values
        if np.any(np.isnan(signal)):
            nan_count = np.sum(np.isnan(signal))
            logger.warning(f"Found {nan_count} NaN values in {csv_path.name}, filling with 0")
            signal = np.nan_to_num(signal, nan=0.0)
        
        # Extract labels
        labels = self._extract_labels_from_df(df)
        
        # Resample if needed (100Hz -> target rate)
        if self.target_sampling_rate != self.ORIGINAL_SAMPLING_RATE:
            signal, labels = self._resample_data(
                signal, labels,
                self.ORIGINAL_SAMPLING_RATE,
                self.target_sampling_rate
            )
        
        # Validate
        self._validate_data(signal, labels, csv_path.stem)
        
        return signal, labels

    def _find_accel_columns(self, df: pd.DataFrame) -> List[str]:
        """Find accelerometer column names in the dataframe."""
        column_sets = [
            ['x', 'y', 'z'],
            ['X', 'Y', 'Z'],
            ['acc_x', 'acc_y', 'acc_z'],
            ['accel_x', 'accel_y', 'accel_z'],
        ]
        
        for cols in column_sets:
            if all(c in df.columns for c in cols):
                return cols
        
        # Try to find columns with x, y, z in name
        x_col = [c for c in df.columns if 'x' in c.lower() and ('acc' in c.lower() or c.lower() == 'x')]
        y_col = [c for c in df.columns if 'y' in c.lower() and ('acc' in c.lower() or c.lower() == 'y')]
        z_col = [c for c in df.columns if 'z' in c.lower() and ('acc' in c.lower() or c.lower() == 'z')]
        
        if x_col and y_col and z_col:
            return [x_col[0], y_col[0], z_col[0]]
        
        raise ValueError(
            f"Could not find accelerometer columns.\n"
            f"  Available columns: {list(df.columns)}\n"
            f"  Expected: x, y, z or X, Y, Z"
        )

    def _extract_labels_from_df(self, df: pd.DataFrame) -> np.ndarray:
        """Extract activity labels from dataframe."""
        # Priority order for label columns based on num_classes
        if self.num_classes == 5:
            label_cols = ['Walmsley2020', 'walmsley', 'label_5class', 'label']
        else:
            label_cols = ['Willetts2018', 'WillettsSpecific2018', 'willetts', 'label_8class', 'label']
        
        # Also try generic columns
        label_cols.extend(['annotation', 'activity', 'class'])
        
        # Find first available column
        label_col = None
        for col in label_cols:
            if col in df.columns:
                label_col = col
                break
        
        if label_col is None:
            logger.warning(f"No label column found. Available: {list(df.columns)}")
            return np.zeros(len(df), dtype=np.int64)
        
        raw_labels = df[label_col].values
        
        # Convert string labels to integers if needed
        if raw_labels.dtype == object or (len(raw_labels) > 0 and isinstance(raw_labels[0], str)):
            labels = self._convert_string_labels(raw_labels)
        else:
            labels = raw_labels.astype(np.int64)
        
        # Clip to valid range
        labels = np.clip(labels, 0, self.num_classes - 1)
        
        return labels

    def _convert_string_labels(self, string_labels: np.ndarray) -> np.ndarray:
        """Convert string activity labels to integer class indices."""
        labels = np.zeros(len(string_labels), dtype=np.int64)
        
        # Walmsley 5-class mapping
        label_map = {
            'sleep': 0,
            'sedentary': 1,
            'light': 2,
            'moderate-vigorous': 3,
            'bicycling': 4,
        }
        
        for i, label in enumerate(string_labels):
            if pd.isna(label) or label == '':
                labels[i] = 0
            else:
                label_lower = str(label).lower().strip()
                
                if label_lower in label_map:
                    labels[i] = label_map[label_lower]
                elif any(k in label_lower for k in ['sleep', 'night']):
                    labels[i] = 0
                elif any(k in label_lower for k in ['sit', 'stand', 'sedentary', 'stationary']):
                    labels[i] = 1
                elif any(k in label_lower for k in ['light', 'household']):
                    labels[i] = 2
                elif any(k in label_lower for k in ['walk', 'stair', 'run', 'moderate', 'vigorous']):
                    labels[i] = 3
                elif any(k in label_lower for k in ['bike', 'cycl']):
                    labels[i] = 4
                else:
                    labels[i] = 2  # Default to light activity
        
        return labels

    def _resample_data(
        self,
        signal: np.ndarray,
        labels: np.ndarray,
        original_rate: int,
        target_rate: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample signal and labels to target sampling rate.
        
        Uses scipy's resample_poly for high-quality resampling.
        FIX: Uses actual resample_poly output length to avoid shape mismatch.
        """
        from scipy.signal import resample_poly
        from math import gcd
        
        g = gcd(original_rate, target_rate)
        up = target_rate // g
        down = original_rate // g
        
        # FIX: Resample first channel to get the ACTUAL output length
        # (resample_poly can produce length that differs by 1 from calculated)
        resampled_ch0 = resample_poly(signal[:, 0], up, down)
        n_samples_new = len(resampled_ch0)
        
        # Allocate output array with correct size
        resampled_signal = np.zeros((n_samples_new, signal.shape[1]), dtype=np.float32)
        resampled_signal[:, 0] = resampled_ch0
        
        # Resample remaining channels
        for ch in range(1, signal.shape[1]):
            resampled_ch = resample_poly(signal[:, ch], up, down)
            # Truncate or pad to match first channel length (handles off-by-one)
            if len(resampled_ch) >= n_samples_new:
                resampled_signal[:, ch] = resampled_ch[:n_samples_new]
            else:
                resampled_signal[:len(resampled_ch), ch] = resampled_ch
        
        # Resample labels with nearest neighbor to EXACT same length as signal
        new_indices = np.linspace(0, len(labels) - 1, n_samples_new)
        resampled_labels = labels[np.round(new_indices).astype(int)]
        
        return resampled_signal, resampled_labels

    def _load_from_directory(self, participant_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from participant directory (preprocessed format)."""
        participant_dir = self.data_path / participant_id

        if not participant_dir.exists():
            raise FileNotFoundError(
                f"Participant directory not found: {participant_dir}\n"
                f"  Hint: Ensure CAPTURE-24 dataset is downloaded and extracted"
            )

        if participant_dir.is_dir():
            dir_contents = list(participant_dir.iterdir())
            if len(dir_contents) == 0:
                raise ValueError(
                    f"Participant directory is empty: {participant_dir}\n"
                    f"  Expected files: accelerometry.npy/.csv and labels.npy/.csv"
                )

        # Load accelerometry data
        signal = self._load_accelerometry(participant_dir)

        # Load labels
        labels = self._load_labels(participant_dir)

        # Validate
        self._validate_data(signal, labels, participant_id)

        return signal, labels

    def _load_accelerometry(self, participant_dir: Path) -> np.ndarray:
        """Load accelerometry data from participant directory."""
        for filename in ["accelerometry.npy", "accelerometry.csv", "acc.npy", "data.npy"]:
            filepath = participant_dir / filename

            if filepath.exists():
                if filename.endswith(".npy"):
                    signal = np.load(filepath)
                elif filename.endswith(".csv"):
                    df = pd.read_csv(filepath)
                    if all(col in df.columns for col in ["X", "Y", "Z"]):
                        signal = df[["X", "Y", "Z"]].values
                    elif all(col in df.columns for col in ["x", "y", "z"]):
                        signal = df[["x", "y", "z"]].values
                    elif len(df.columns) >= 3:
                        signal = df.iloc[:, :3].values
                        logger.warning(f"Using first 3 columns as X, Y, Z from {filepath}")
                    else:
                        raise ValueError(f"Cannot find X, Y, Z columns in {filepath}")

                logger.debug(f"Loaded accelerometry from {filepath}")
                return signal.astype(np.float32)

        raise FileNotFoundError(
            f"No accelerometry file found in {participant_dir}\n"
            f"  Expected files: accelerometry.npy, accelerometry.csv, acc.npy, or data.npy"
        )

    def _load_labels(self, participant_dir: Path) -> np.ndarray:
        """Load labels from participant directory."""
        for filename in ["labels.npy", "labels.csv", "annotations.npy", "annotations.csv"]:
            filepath = participant_dir / filename

            if filepath.exists():
                if filename.endswith(".npy"):
                    labels = np.load(filepath)
                elif filename.endswith(".csv"):
                    df = pd.read_csv(filepath)
                    if "label" in df.columns:
                        labels = df["label"].values
                    elif "activity" in df.columns:
                        labels = df["activity"].values
                    elif "class" in df.columns:
                        labels = df["class"].values
                    elif len(df.columns) == 1:
                        labels = df.iloc[:, 0].values
                    else:
                        raise ValueError(f"Cannot find label column in {filepath}")

                logger.debug(f"Loaded labels from {filepath}")
                return labels.astype(np.int64)

        raise FileNotFoundError(
            f"No labels file found in {participant_dir}\n"
            f"  Expected files: labels.npy, labels.csv, annotations.npy, or annotations.csv"
        )

    def _validate_data(self, signal: np.ndarray, labels: np.ndarray, participant_id: str):
        """Validate loaded data."""
        if signal.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Signal and labels length mismatch for {participant_id}.\n"
                f"  Signal: {signal.shape[0]} samples\n"
                f"  Labels: {labels.shape[0]} samples"
            )

        if signal.shape[1] != 3:
            raise ValueError(
                f"Expected 3 channels (X, Y, Z), got {signal.shape[1]} for {participant_id}"
            )

        unique_labels = np.unique(labels)
        invalid_labels = [l for l in unique_labels if l < 0 or l >= self.num_classes]
        if invalid_labels:
            logger.warning(
                f"Found labels outside valid range for {participant_id}: {invalid_labels}. "
                f"Clipping to [0, {self.num_classes-1}]"
            )

    def get_participant_ids(self) -> List[str]:
        """
        Get list of all available participant IDs.

        Supports both formats:
        - P001.csv.gz files (official CAPTURE-24)
        - P001/ directories (preprocessed)

        Returns:
            List of participant IDs sorted alphabetically
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")

        # Method 1: Look for P*.csv.gz files (official CAPTURE-24 format)
        csv_gz_files = sorted(self.data_path.glob("P*.csv.gz"))
        if csv_gz_files:
            participant_ids = []
            for f in csv_gz_files:
                match = re.match(r'(P\d{3})\.csv\.gz', f.name)
                if match:
                    participant_ids.append(match.group(1))
            if participant_ids:
                self._data_format = 'csv.gz'
                logger.info(f"Found {len(participant_ids)} participant files (P*.csv.gz format)")
                return sorted(participant_ids)

        # Method 2: Look for P*.csv files (uncompressed)
        csv_files = sorted(self.data_path.glob("P*.csv"))
        if csv_files:
            participant_ids = []
            for f in csv_files:
                match = re.match(r'(P\d{3})\.csv', f.name)
                if match:
                    participant_ids.append(match.group(1))
            if participant_ids:
                self._data_format = 'csv'
                logger.info(f"Found {len(participant_ids)} participant files (P*.csv format)")
                return sorted(participant_ids)

        # Method 3: Look for directories (preprocessed format)
        participant_dirs = [
            d.name for d in self.data_path.iterdir()
            if d.is_dir() and (d.name.startswith("P") or d.name.isdigit())
        ]

        if participant_dirs:
            self._data_format = 'directory'
            logger.info(f"Found {len(participant_dirs)} participant directories")
            return sorted(participant_dirs)

        # Nothing found
        raise FileNotFoundError(
            f"No participant data found in {self.data_path}\n"
            f"  Expected formats:\n"
            f"    - P001.csv.gz, P002.csv.gz, ... (official CAPTURE-24)\n"
            f"    - P001/, P002/, ... (preprocessed directories)"
        )

    def get_metadata(self) -> Dict[str, any]:
        """Get CAPTURE-24 dataset metadata."""
        return {
            "dataset_name": "CAPTURE-24",
            "sampling_rate": self.ORIGINAL_SAMPLING_RATE,
            "target_sampling_rate": self.target_sampling_rate,
            "num_channels": 3,
            "sensor_info": "Axivity AX3",
            "wear_location": "dominant_wrist",
            "dynamic_range": 8,  # ±8g
            "num_participants": len(self.participant_ids),
            "num_classes": self.num_classes,
            "duration_per_participant": "~24 hours",
            "data_format": self._data_format,
        }

    def get_label_map(self) -> Dict[int, str]:
        """Get mapping from class IDs to class names."""
        return self.label_mapping
