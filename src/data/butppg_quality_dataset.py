"""Quality-Stratified BUT-PPG Dataset for SSL.

Extends the standard BUT-PPG dataset with quality stratification for
quality-aware contrastive learning. Ensures balanced sampling across
quality levels (low/medium/high) to prevent mode collapse.

Author: Claude Code
Date: 2025-10-22
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from src.data.butppg_dataset import BUTPPGDataset
from src.ssl.quality_proxy import QualityProxyComputer


class QualityStratifiedBUTPPGDataset(Dataset):
    """BUT-PPG dataset with quality stratification for SSL.

    Features:
    1. Computes quality scores on-the-fly or loads precomputed scores
    2. Stratifies samples into quality bins (low/medium/high)
    3. Supports balanced sampling across quality levels
    4. Compatible with quality-aware contrastive learning

    Args:
        data_dir: Path to BUT-PPG data
                  - RAW mode: Path to WFDB files
                  - PREPROCESSED mode: Path to window_*.npz files
        split: 'train', 'val', or 'test'
        modality: Modality specification ('ppg', 'all', etc.)
        fs: Sampling rate (default: 125 Hz to match VitalDB)
        window_sec: Window duration in seconds
        mode: 'raw' or 'preprocessed'
        quality_bins: Number of quality stratification bins (default: 3)
        precompute_quality: If True, precompute all quality scores
        quality_cache_file: Path to save/load precomputed quality scores
        **kwargs: Additional arguments for BUTPPGDataset

    Example:
        >>> dataset = QualityStratifiedBUTPPGDataset(
        ...     data_dir='data/processed/butppg/windows_with_labels',
        ...     split='train',
        ...     modality='all',
        ...     mode='preprocessed',
        ...     quality_bins=3,
        ...     precompute_quality=True
        ... )
        >>> sample = dataset[0]
        >>> print(sample.keys())
        dict_keys(['signal', 'quality_score', 'quality_bin', 'idx'])
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = 'train',
        modality: Union[str, List[str]] = 'all',
        fs: float = 125.0,
        window_sec: float = 10.0,
        mode: str = 'preprocessed',
        quality_bins: int = 3,
        precompute_quality: bool = True,
        quality_cache_file: Optional[str] = None,
        **kwargs
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.split = split
        self.fs = fs
        self.quality_bins = quality_bins
        self.mode = mode

        # Initialize base dataset
        self.base_dataset = BUTPPGDataset(
            data_dir=data_dir,
            split=split,
            modality=modality,
            fs=fs,
            window_sec=window_sec,
            mode=mode,
            return_labels=False,  # We'll handle labels ourselves
            **kwargs
        )

        # Initialize quality computer
        self.quality_computer = QualityProxyComputer(fs=fs)

        # Setup quality cache
        if quality_cache_file is None:
            quality_cache_file = self.data_dir / f'quality_scores_{split}.json'
        self.quality_cache_file = Path(quality_cache_file)

        # Quality scores storage
        self.quality_scores = None
        self.quality_bin_indices = None
        self.bin_boundaries = None

        # Precompute or load quality scores
        if precompute_quality:
            self._precompute_quality_scores()
        else:
            # Will compute on-the-fly
            self.quality_scores = {}

    def _precompute_quality_scores(self):
        """Precompute quality scores for all samples."""
        print(f"Precomputing quality scores for {len(self.base_dataset)} samples...")

        # Try loading from cache first
        if self.quality_cache_file.exists():
            print(f"  Loading from cache: {self.quality_cache_file}")
            try:
                with open(self.quality_cache_file, 'r') as f:
                    cache_data = json.load(f)
                self.quality_scores = torch.tensor(cache_data['scores'])
                self.bin_boundaries = cache_data.get('bin_boundaries')
                print(f"  ✓ Loaded {len(self.quality_scores)} quality scores")
                self._stratify_quality()
                return
            except Exception as e:
                print(f"  ⚠️  Cache load failed: {e}, recomputing...")

        # Compute quality scores
        quality_scores = []

        for idx in range(len(self.base_dataset)):
            if idx % 100 == 0:
                print(f"  Progress: {idx}/{len(self.base_dataset)}")

            # Get signal
            sample = self.base_dataset[idx]
            if isinstance(sample, tuple):
                signal = sample[0]  # (seg1, seg2) → use seg1
            else:
                signal = sample

            # Compute quality
            if isinstance(signal, torch.Tensor):
                signal_batch = signal.unsqueeze(0)  # [C, T] → [1, C, T]
            else:
                signal_batch = torch.from_numpy(signal).unsqueeze(0)

            quality = self.quality_computer.compute_batch(signal_batch)
            quality_scores.append(float(quality[0]))

        self.quality_scores = torch.tensor(quality_scores)
        print(f"  ✓ Computed {len(self.quality_scores)} quality scores")

        # Save to cache
        try:
            self.quality_cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_data = {
                'scores': self.quality_scores.tolist(),
                'split': self.split,
                'num_samples': len(self.quality_scores)
            }
            with open(self.quality_cache_file, 'w') as f:
                json.dump(cache_data, f)
            print(f"  ✓ Saved quality scores to {self.quality_cache_file}")
        except Exception as e:
            print(f"  ⚠️  Failed to save cache: {e}")

        # Stratify into bins
        self._stratify_quality()

    def _stratify_quality(self):
        """Stratify samples into quality bins."""
        if self.quality_scores is None:
            return

        # Create bin boundaries
        if self.bin_boundaries is None:
            self.bin_boundaries = torch.linspace(0, 1, self.quality_bins + 1)

        # Assign samples to bins
        self.quality_bin_indices = torch.bucketize(
            self.quality_scores,
            self.bin_boundaries[:-1],
            right=False
        ) - 1
        self.quality_bin_indices = torch.clamp(
            self.quality_bin_indices, 0, self.quality_bins - 1
        )

        # Print statistics
        print(f"\n  Quality Stratification ({self.split}):")
        bin_labels = ['Low', 'Medium', 'High'] if self.quality_bins == 3 else [f'Bin{i}' for i in range(self.quality_bins)]

        for bin_idx in range(self.quality_bins):
            count = (self.quality_bin_indices == bin_idx).sum().item()
            pct = 100 * count / len(self.quality_scores)
            label = bin_labels[bin_idx] if bin_idx < len(bin_labels) else f'Bin{bin_idx}'
            print(f"    {label}: {count} samples ({pct:.1f}%)")

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get sample with quality information.

        Returns:
            sample: Dictionary containing:
                - signal: Signal tensor [C, T]
                - quality_score: Quality score [1]
                - quality_bin: Bin index [1]
                - idx: Sample index [1]
        """
        # Get base sample
        sample = self.base_dataset[idx]

        # Handle paired samples (seg1, seg2)
        if isinstance(sample, tuple):
            signal = sample[0]  # Use first segment
        else:
            signal = sample

        # Convert to tensor if needed
        if not isinstance(signal, torch.Tensor):
            signal = torch.from_numpy(signal).float()

        # Get or compute quality score
        if self.quality_scores is not None:
            quality_score = self.quality_scores[idx]
            quality_bin = self.quality_bin_indices[idx]
        else:
            # Compute on-the-fly
            signal_batch = signal.unsqueeze(0)  # [C, T] → [1, C, T]
            quality_score = self.quality_computer.compute_batch(signal_batch)[0]

            # Compute bin
            if self.bin_boundaries is None:
                self.bin_boundaries = torch.linspace(0, 1, self.quality_bins + 1)
            quality_bin = torch.bucketize(
                quality_score.unsqueeze(0),
                self.bin_boundaries[:-1],
                right=False
            )[0] - 1
            quality_bin = torch.clamp(quality_bin, 0, self.quality_bins - 1)

        return {
            'signal': signal,
            'quality_score': quality_score,
            'quality_bin': quality_bin,
            'idx': torch.tensor(idx, dtype=torch.long)
        }

    def get_bin_indices(self, bin_idx: int) -> List[int]:
        """Get indices of samples in a specific quality bin.

        Args:
            bin_idx: Bin index (0 = low, 1 = medium, 2 = high for 3 bins)

        Returns:
            indices: List of sample indices in the bin
        """
        if self.quality_bin_indices is None:
            raise ValueError("Quality scores not precomputed. Set precompute_quality=True.")

        mask = self.quality_bin_indices == bin_idx
        indices = torch.where(mask)[0].tolist()
        return indices


class BalancedQualitySampler(Sampler):
    """Sampler that ensures balanced sampling across quality bins.

    Oversamples minority bins to ensure equal representation of
    low/medium/high quality samples in each batch.

    Args:
        dataset: QualityStratifiedBUTPPGDataset instance
        samples_per_bin: Number of samples per bin per epoch
        shuffle: Whether to shuffle samples (default: True)

    Example:
        >>> dataset = QualityStratifiedBUTPPGDataset(...)
        >>> sampler = BalancedQualitySampler(dataset, samples_per_bin=1000)
        >>> loader = DataLoader(dataset, batch_size=32, sampler=sampler)
    """

    def __init__(
        self,
        dataset: QualityStratifiedBUTPPGDataset,
        samples_per_bin: Optional[int] = None,
        shuffle: bool = True
    ):
        self.dataset = dataset
        self.shuffle = shuffle
        self.num_bins = dataset.quality_bins

        if dataset.quality_bin_indices is None:
            raise ValueError("Dataset must have precomputed quality scores.")

        # Get indices for each bin
        self.bin_indices = []
        for bin_idx in range(self.num_bins):
            indices = dataset.get_bin_indices(bin_idx)
            self.bin_indices.append(indices)

        # Determine samples per bin
        if samples_per_bin is None:
            # Use size of largest bin
            samples_per_bin = max(len(indices) for indices in self.bin_indices)

        self.samples_per_bin = samples_per_bin
        self.total_samples = samples_per_bin * self.num_bins

    def __iter__(self):
        """Generate sampling indices."""
        all_indices = []

        for bin_idx in range(self.num_bins):
            bin_indices = self.bin_indices[bin_idx]

            # Sample with replacement if needed
            if len(bin_indices) >= self.samples_per_bin:
                # Subsample
                if self.shuffle:
                    sampled = np.random.choice(
                        bin_indices, size=self.samples_per_bin, replace=False
                    )
                else:
                    sampled = bin_indices[:self.samples_per_bin]
            else:
                # Oversample
                if self.shuffle:
                    sampled = np.random.choice(
                        bin_indices, size=self.samples_per_bin, replace=True
                    )
                else:
                    # Repeat indices
                    repeats = (self.samples_per_bin // len(bin_indices)) + 1
                    sampled = (bin_indices * repeats)[:self.samples_per_bin]

            all_indices.extend(sampled.tolist() if isinstance(sampled, np.ndarray) else sampled)

        # Shuffle all indices
        if self.shuffle:
            np.random.shuffle(all_indices)

        return iter(all_indices)

    def __len__(self):
        """Return total number of samples."""
        return self.total_samples
