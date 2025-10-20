"""
Unified Window Data Loader

PyTorch Dataset for loading individual NPZ window files with embedded labels.
Supports both VitalDB and BUT-PPG formats with consistent interface.

Usage:
    # VitalDB quality classification
    dataset = UnifiedWindowDataset(
        data_dir='data/processed/vitaldb/windows_with_labels/train',
        task='mortality',
        channels=['PPG', 'ECG']
    )

    # BUT-PPG blood pressure regression
    dataset = UnifiedWindowDataset(
        data_dir='data/processed/butppg/windows_with_labels/train',
        task='blood_pressure',
        channels=['PPG', 'ECG'],
        filter_missing=True
    )
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union


class UnifiedWindowDataset(Dataset):
    """Unified dataset for loading individual NPZ window files.

    Supports both VitalDB and BUT-PPG formats with embedded labels.

    Args:
        data_dir: Directory containing window_*.npz files
        task: Task name ('quality', 'hr', 'blood_pressure', 'mortality', 'icu_days', etc.)
        channels: List of channel names to load (e.g., ['PPG', 'ECG'])
        filter_missing: If True, exclude samples with missing labels
        return_metadata: If True, return metadata dict along with signal and label
        transform: Optional transform to apply to signals

    Returns:
        If return_metadata=False: (signal, label)
        If return_metadata=True: (signal, label, metadata)

        - signal: Tensor [C, T] where C=num channels, T=timesteps
        - label: Scalar or tuple of scalars (depends on task)
        - metadata: Dict with record/case ID, quality scores, etc.
    """

    # Task definitions
    TASK_CONFIGS = {
        # BUT-PPG tasks
        'quality': {'label_keys': ['quality'], 'type': 'classification'},
        'hr': {'label_keys': ['hr'], 'type': 'regression'},
        'motion': {'label_keys': ['motion'], 'type': 'classification'},
        'blood_pressure': {'label_keys': ['bp_systolic', 'bp_diastolic'], 'type': 'regression'},
        'bp_systolic': {'label_keys': ['bp_systolic'], 'type': 'regression'},
        'bp_diastolic': {'label_keys': ['bp_diastolic'], 'type': 'regression'},
        'spo2': {'label_keys': ['spo2'], 'type': 'regression'},
        'glycaemia': {'label_keys': ['glycaemia'], 'type': 'regression'},

        # VitalDB tasks (case-level)
        'mortality': {'label_keys': ['death_inhosp'], 'type': 'classification'},
        'icu_days': {'label_keys': ['icu_days'], 'type': 'regression'},
        'emergency': {'label_keys': ['emergency'], 'type': 'classification'},
        'asa': {'label_keys': ['asa'], 'type': 'classification'},

        # Multi-task
        'all_butppg': {
            'label_keys': ['quality', 'hr', 'motion', 'bp_systolic', 'bp_diastolic', 'spo2', 'glycaemia'],
            'type': 'multi-task'
        },
        'all_vitaldb': {
            'label_keys': ['death_inhosp', 'icu_days', 'emergency', 'asa'],
            'type': 'multi-task'
        }
    }

    # Channel definitions
    VITALDB_CHANNELS = {
        'PPG': 0,
        'ECG': 1
    }

    BUTPPG_CHANNELS = {
        'ACC_X': 0,
        'ACC_Y': 1,
        'ACC_Z': 2,
        'PPG': 3,
        'ECG': 4
    }

    def __init__(
        self,
        data_dir: Union[str, Path],
        task: str = 'quality',
        channels: Optional[List[str]] = None,
        filter_missing: bool = True,
        return_metadata: bool = False,
        transform: Optional[callable] = None
    ):
        self.data_dir = Path(data_dir)
        self.task = task
        self.filter_missing = filter_missing
        self.return_metadata = return_metadata
        self.transform = transform

        # Validate task
        if task not in self.TASK_CONFIGS:
            raise ValueError(f"Unknown task '{task}'. Available: {list(self.TASK_CONFIGS.keys())}")

        self.task_config = self.TASK_CONFIGS[task]

        # Find all window files
        self.window_files = sorted(self.data_dir.glob('window_*.npz'))
        if len(self.window_files) == 0:
            raise FileNotFoundError(f"No window files found in {self.data_dir}")

        # Detect dataset type and available channels
        sample = np.load(self.window_files[0])
        signal_shape = sample['signal'].shape

        if signal_shape[0] == 2:
            self.dataset_type = 'vitaldb'
            self.available_channels = self.VITALDB_CHANNELS
            default_channels = ['PPG', 'ECG']
        elif signal_shape[0] == 5:
            self.dataset_type = 'butppg'
            self.available_channels = self.BUTPPG_CHANNELS
            default_channels = ['PPG', 'ECG']
        else:
            raise ValueError(f"Unexpected signal shape: {signal_shape}")

        # Set channels
        if channels is None:
            channels = default_channels
        self.channels = channels

        # Validate channels
        for ch in self.channels:
            if ch not in self.available_channels:
                raise ValueError(f"Channel '{ch}' not available. Available: {list(self.available_channels.keys())}")

        self.channel_indices = [self.available_channels[ch] for ch in self.channels]

        # Filter samples with missing labels
        if self.filter_missing:
            self.valid_indices = self._find_valid_samples()
            print(f"Loaded {len(self.valid_indices)}/{len(self.window_files)} samples with valid labels for task '{task}'")
        else:
            self.valid_indices = list(range(len(self.window_files)))
            print(f"Loaded {len(self.window_files)} samples (no label filtering)")

    def _find_valid_samples(self) -> List[int]:
        """Find samples with valid (non-missing) labels."""
        valid_indices = []

        for idx, window_file in enumerate(self.window_files):
            try:
                data = np.load(window_file)

                # Check if all required labels are present and valid
                labels_valid = True
                for label_key in self.task_config['label_keys']:
                    if label_key not in data:
                        labels_valid = False
                        break

                    label_value = data[label_key]

                    # Check for missing value (-1, NaN, or inf)
                    if isinstance(label_value, (np.ndarray, np.generic)):
                        label_value = label_value.item()

                    if np.isnan(label_value) or np.isinf(label_value) or label_value == -1:
                        labels_valid = False
                        break

                if labels_valid:
                    valid_indices.append(idx)

            except Exception as e:
                print(f"Warning: Error loading {window_file}: {e}")
                continue

        return valid_indices

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor],
                                              Tuple[torch.Tensor, torch.Tensor, Dict]]:
        """Get a single window sample.

        Returns:
            signal: [C, T] tensor
            label: Scalar or tuple (depends on task)
            metadata: Dict (if return_metadata=True)
        """
        # Map to actual file index
        file_idx = self.valid_indices[idx]
        window_file = self.window_files[file_idx]

        # Load data
        data = np.load(window_file)

        # Extract signal and select channels
        signal = data['signal']  # [n_channels, T]
        signal = signal[self.channel_indices, :]  # Select requested channels

        # Convert to tensor
        signal = torch.from_numpy(signal).float()

        # Apply transform if provided
        if self.transform is not None:
            signal = self.transform(signal)

        # Extract labels
        label_keys = self.task_config['label_keys']

        if len(label_keys) == 1:
            # Single label
            label = data[label_keys[0]]
            if isinstance(label, (np.ndarray, np.generic)):
                label = label.item()

            # Convert to tensor
            if self.task_config['type'] == 'classification':
                label = torch.tensor(label, dtype=torch.long)
            else:
                label = torch.tensor(label, dtype=torch.float32)

        else:
            # Multiple labels (e.g., BP systolic + diastolic)
            labels = []
            for label_key in label_keys:
                lbl = data[label_key]
                if isinstance(lbl, (np.ndarray, np.generic)):
                    lbl = lbl.item()
                labels.append(lbl)

            label = torch.tensor(labels, dtype=torch.float32)

        # Extract metadata if requested
        if self.return_metadata:
            metadata = {
                'file': str(window_file.name),
                'fs': int(data['fs']),
                'window_idx': int(data['window_idx']),
                'ppg_quality': float(data['ppg_quality']),
                'ecg_quality': float(data['ecg_quality']),
            }

            # Add record/case ID
            if 'record_id' in data:
                metadata['record_id'] = str(data['record_id'])
            elif 'case_id' in data:
                metadata['case_id'] = int(data['case_id'])

            # Add demographics if available
            for key in ['age', 'sex', 'bmi', 'height', 'weight']:
                if key in data:
                    val = data[key]
                    if isinstance(val, (np.ndarray, np.generic)):
                        val = val.item()
                    metadata[key] = val

            return signal, label, metadata

        else:
            return signal, label

    def get_label_stats(self) -> Dict:
        """Compute label statistics across dataset."""
        stats = {}

        for label_key in self.task_config['label_keys']:
            values = []

            for idx in self.valid_indices:
                window_file = self.window_files[idx]
                data = np.load(window_file)

                if label_key in data:
                    val = data[label_key]
                    if isinstance(val, (np.ndarray, np.generic)):
                        val = val.item()

                    if not (np.isnan(val) or np.isinf(val) or val == -1):
                        values.append(val)

            if len(values) > 0:
                values = np.array(values)
                stats[label_key] = {
                    'count': len(values),
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max())
                }
            else:
                stats[label_key] = {'count': 0}

        return stats


def create_dataloaders(
    train_dir: Union[str, Path],
    val_dir: Union[str, Path],
    test_dir: Union[str, Path],
    task: str = 'quality',
    channels: Optional[List[str]] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    filter_missing: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders.

    Args:
        train_dir: Training window directory
        val_dir: Validation window directory
        test_dir: Test window directory
        task: Task name
        channels: Channel list
        batch_size: Batch size
        num_workers: Number of workers
        filter_missing: Filter missing labels

    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = UnifiedWindowDataset(
        train_dir,
        task=task,
        channels=channels,
        filter_missing=filter_missing
    )

    val_dataset = UnifiedWindowDataset(
        val_dir,
        task=task,
        channels=channels,
        filter_missing=filter_missing
    )

    test_dataset = UnifiedWindowDataset(
        test_dir,
        task=task,
        channels=channels,
        filter_missing=filter_missing
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Example usage and testing
    import sys

    if len(sys.argv) < 2:
        print("Usage: python unified_window_loader.py <data_dir>")
        print("\nExample:")
        print("  python unified_window_loader.py data/processed/butppg/windows_with_labels/train")
        sys.exit(1)

    data_dir = sys.argv[1]

    print("="*80)
    print("UNIFIED WINDOW DATASET TEST")
    print("="*80)

    # Test dataset loading
    dataset = UnifiedWindowDataset(
        data_dir=data_dir,
        task='quality',
        channels=['PPG', 'ECG'],
        filter_missing=True
    )

    print(f"\nDataset info:")
    print(f"  Type: {dataset.dataset_type}")
    print(f"  Samples: {len(dataset)}")
    print(f"  Channels: {dataset.channels}")
    print(f"  Task: {dataset.task}")

    # Get sample
    if len(dataset) > 0:
        signal, label = dataset[0]
        print(f"\nSample 0:")
        print(f"  Signal shape: {signal.shape}")
        print(f"  Label: {label}")

        # Test with metadata
        dataset_meta = UnifiedWindowDataset(
            data_dir=data_dir,
            task='quality',
            channels=['PPG', 'ECG'],
            filter_missing=True,
            return_metadata=True
        )

        signal, label, metadata = dataset_meta[0]
        print(f"\nMetadata:")
        for key, val in metadata.items():
            print(f"    {key}: {val}")

    # Label statistics
    print("\nLabel statistics:")
    stats = dataset.get_label_stats()
    for label_key, label_stats in stats.items():
        print(f"  {label_key}:")
        for key, val in label_stats.items():
            print(f"    {key}: {val}")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
