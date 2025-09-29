"""PyTorch datasets for windowed biosignal data.

Loads preprocessed NPZ windows and provides efficient batching.
"""

import json
import os
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

from ..utils.io import load_npz, load_yaml
from ..utils.seed import worker_init_fn


class RawWindowDataset(Dataset):
    """Dataset for loading preprocessed signal windows from NPZ files.
    
    Expects NPZ files with:
    - 'data': [n_windows, time, channels] array
    - 'labels': [n_windows, ...] array (optional)
    - 'metadata': dict with sampling info (optional)
    """
    
    def __init__(
        self,
        index_manifest: Union[str, Path, List[Dict]],
        split: str = 'train',
        channels_cfg: Optional[Union[str, Dict]] = None,
        windows_cfg: Optional[Union[str, Dict]] = None,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        cache_size: int = 0
    ):
        """Initialize dataset.
        
        Args:
            index_manifest: Path to manifest JSON or list of file dicts
            split: Which split to use ('train', 'val', 'test')
            channels_cfg: Channel configuration (path or dict)
            windows_cfg: Window configuration (path or dict)
            transform: Transform to apply to windows
            target_transform: Transform to apply to labels
            cache_size: Number of files to cache in memory (0 = no cache)
        """
        super().__init__()
        
        # Load manifest
        if isinstance(index_manifest, (str, Path)):
            with open(index_manifest, 'r') as f:
                manifest = json.load(f)
        else:
            manifest = index_manifest
        
        # Filter by split
        if isinstance(manifest, dict):
            if split in manifest:
                self.file_list = manifest[split]
            else:
                # Assume manifest is {split: [{file_info}, ...]}
                raise ValueError(f"Split '{split}' not found in manifest")
        else:
            # Assume manifest is a list of files
            self.file_list = manifest
        
        # Load configs
        if isinstance(channels_cfg, str):
            self.channels_cfg = load_yaml(channels_cfg)
        else:
            self.channels_cfg = channels_cfg or {}
            
        if isinstance(windows_cfg, str):
            self.windows_cfg = load_yaml(windows_cfg)
        else:
            self.windows_cfg = windows_cfg or {}
        
        self.transform = transform
        self.target_transform = target_transform
        
        # Setup cache
        self.cache_size = cache_size
        self.cache = {}
        
        # Build index: map global idx to (file_idx, window_idx)
        self._build_index()
    
    def _build_index(self):
        """Build index mapping from global to file/window indices."""
        self.index_map = []
        self.cumulative_sizes = [0]
        
        for file_idx, file_info in enumerate(self.file_list):
            if isinstance(file_info, dict):
                n_windows = file_info.get('n_windows', 0)
                if n_windows == 0:
                    # Load file to get size
                    data = self._load_file(file_idx)
                    n_windows = data['data'].shape[0]
            else:
                # file_info is a path, need to load to get size
                data = self._load_file(file_idx)
                n_windows = data['data'].shape[0]
            
            for window_idx in range(n_windows):
                self.index_map.append((file_idx, window_idx))
            
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + n_windows)
    
    def _load_file(self, file_idx: int) -> Dict:
        """Load NPZ file, with caching."""
        # Check cache
        if file_idx in self.cache:
            return self.cache[file_idx]
        
        # Get file path
        file_info = self.file_list[file_idx]
        if isinstance(file_info, dict):
            file_path = file_info['path']
        else:
            file_path = file_info
        
        # Load NPZ with pickle support for metadata
        data = load_npz(file_path, allow_pickle=True)
        
        # Add to cache if enabled
        if self.cache_size > 0:
            # Evict oldest if cache full
            if len(self.cache) >= self.cache_size:
                # Simple FIFO eviction
                oldest = min(self.cache.keys())
                del self.cache[oldest]
            
            self.cache[file_idx] = data
        
        return data
    
    def __len__(self) -> int:
        """Return total number of windows across all files."""
        return len(self.index_map)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[torch.Tensor, Dict]]:
        """Get a single window.
        
        Returns:
            Tuple of (window_data, target/metadata)
        """
        # Map global index to file and window
        file_idx, window_idx = self.index_map[idx]
        
        # Load file
        data = self._load_file(file_idx)
        
        # Extract window
        window = data['data'][window_idx]  # [time, channels]
        
        # Get label/target if available
        if 'labels' in data:
            if data['labels'].ndim == 1:
                target = data['labels'][window_idx]
            else:
                target = data['labels'][window_idx]
        else:
            # Return metadata if no labels
            target = {
                'file_idx': file_idx,
                'window_idx': window_idx,
                'global_idx': idx
            }
            if 'metadata' in data:
                target.update(data['metadata'])
        
        # Convert to tensors
        window = torch.from_numpy(window).float()
        
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)
            if target.dtype == torch.float64:
                target = target.float()
        
        # Apply transforms
        if self.transform:
            window = self.transform(window)
        
        if self.target_transform and not isinstance(target, dict):
            target = self.target_transform(target)
        
        return window, target
    
    def get_channel_names(self) -> List[str]:
        """Get channel names from config or first file."""
        if 'core' in self.channels_cfg:
            return self.channels_cfg['core']
        
        # Try to get from first file
        data = self._load_file(0)
        if 'channel_names' in data:
            return data['channel_names'].tolist()
        
        # Default names
        n_channels = data['data'].shape[-1]
        return [f'ch_{i}' for i in range(n_channels)]
    
    def get_sampling_rate(self) -> float:
        """Get sampling rate from config or first file."""
        if 'target_fs_hz' in self.channels_cfg:
            return self.channels_cfg['target_fs_hz']
        
        # Try to get from first file
        data = self._load_file(0)
        if 'metadata' in data and 'fs' in data['metadata']:
            return data['metadata']['fs']
        
        # Default
        return 125.0


class StreamingWindowDataset(Dataset):
    """Dataset that generates windows on-the-fly from raw signals.
    
    More memory-efficient than pre-computing all windows.
    """
    
    def __init__(
        self,
        signal_files: List[Union[str, Path]],
        window_s: float = 10.0,
        stride_s: float = 10.0,
        channels: List[str] = None,
        fs: float = 125.0,
        transform: Optional[callable] = None,
        min_cycles: int = 0
    ):
        """Initialize streaming dataset.
        
        Args:
            signal_files: List of NPZ files containing signals
            window_s: Window duration in seconds
            stride_s: Stride between windows in seconds
            channels: Channel names to use
            fs: Sampling frequency
            transform: Transform to apply to windows
            min_cycles: Minimum cardiac cycles per window
        """
        super().__init__()
        
        self.signal_files = signal_files
        self.window_s = window_s
        self.stride_s = stride_s
        self.channels = channels
        self.fs = fs
        self.transform = transform
        self.min_cycles = min_cycles
        
        # Calculate window/stride in samples
        self.window_samples = int(window_s * fs)
        self.stride_samples = int(stride_s * fs)
        
        # Build index
        self._build_index()
    
    def _build_index(self):
        """Build index of valid windows."""
        self.index_map = []
        
        for file_idx, file_path in enumerate(self.signal_files):
            # Load signal
            data = load_npz(file_path)
            signal = data['data']  # Assume [time, channels]
            
            # Calculate windows
            n_samples = signal.shape[0]
            n_windows = (n_samples - self.window_samples) // self.stride_samples + 1
            
            for window_idx in range(n_windows):
                start = window_idx * self.stride_samples
                end = start + self.window_samples
                
                # Add to index
                self.index_map.append({
                    'file_idx': file_idx,
                    'start': start,
                    'end': end,
                    'window_idx': window_idx
                })
    
    def __len__(self) -> int:
        """Return total number of windows."""
        return len(self.index_map)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """Get a window."""
        # Get window info
        info = self.index_map[idx]
        
        # Load signal
        data = load_npz(self.signal_files[info['file_idx']])
        signal = data['data']
        
        # Extract window
        window = signal[info['start']:info['end']]
        
        # Convert to tensor
        window = torch.from_numpy(window).float()
        
        # Apply transform
        if self.transform:
            window = self.transform(window)
        
        # Return with metadata
        metadata = {
            'file_idx': info['file_idx'],
            'window_idx': info['window_idx'],
            'start_sample': info['start'],
            'end_sample': info['end']
        }
        
        return window, metadata


def custom_collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, Union[torch.Tensor, List]]:
    """Custom collate function for handling mixed data types.
    
    Args:
        batch: List of (data, target) tuples
        
    Returns:
        Batched tensors or lists
    """
    # Separate data and targets
    data = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    # Stack data tensors
    data = torch.stack(data, dim=0)  # [B, T, C]
    
    # Handle targets based on type
    if all(isinstance(t, torch.Tensor) for t in targets):
        # All tensors - stack them
        if targets[0].dim() == 0:
            # Scalar targets
            targets = torch.stack(targets, dim=0)
        else:
            # Multi-dim targets
            targets = torch.stack(targets, dim=0)
    elif all(isinstance(t, dict) for t in targets):
        # All dicts - keep as list
        targets = targets
    else:
        # Mixed or other - keep as list
        targets = targets
    
    return data, targets


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    seed: int = 42
) -> DataLoader:
    """Create a DataLoader with deterministic settings.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU
        drop_last: Whether to drop incomplete last batch
        seed: Random seed for reproducibility
        
    Returns:
        Configured DataLoader
    """
    # Create generator for shuffling
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    # Create worker init function using partial
    worker_fn = partial(worker_init_fn, base_seed=seed) if num_workers > 0 else None
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=custom_collate_fn,
        worker_init_fn=worker_fn,
        generator=generator if shuffle else None
    )
    
    return dataloader


class MultiModalDataset(Dataset):
    """Dataset for multi-modal biosignals with different sampling rates."""
    
    def __init__(
        self,
        ecg_files: Optional[List[str]] = None,
        ppg_files: Optional[List[str]] = None,
        abp_files: Optional[List[str]] = None,
        target_fs: float = 125.0,
        window_s: float = 10.0,
        align: bool = True
    ):
        """Initialize multi-modal dataset.
        
        Args:
            ecg_files: List of ECG signal files
            ppg_files: List of PPG signal files
            abp_files: List of ABP signal files
            target_fs: Target sampling frequency
            window_s: Window duration
            align: Whether to align signals
        """
        super().__init__()
        
        self.ecg_files = ecg_files or []
        self.ppg_files = ppg_files or []
        self.abp_files = abp_files or []
        self.target_fs = target_fs
        self.window_s = window_s
        self.align = align
        
        # Validate same number of files
        n_files = max(
            len(self.ecg_files),
            len(self.ppg_files),
            len(self.abp_files)
        )
        
        self.n_files = n_files
        self.window_samples = int(window_s * target_fs)
    
    def __len__(self) -> int:
        """Return number of file sets."""
        return self.n_files
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get aligned multi-modal signals."""
        output = {}
        
        # Load ECG if available
        if idx < len(self.ecg_files):
            ecg_data = load_npz(self.ecg_files[idx])
            output['ecg'] = torch.from_numpy(ecg_data['data']).float()
        
        # Load PPG if available
        if idx < len(self.ppg_files):
            ppg_data = load_npz(self.ppg_files[idx])
            output['ppg'] = torch.from_numpy(ppg_data['data']).float()
        
        # Load ABP if available
        if idx < len(self.abp_files):
            abp_data = load_npz(self.abp_files[idx])
            output['abp'] = torch.from_numpy(abp_data['data']).float()
        
        return output
