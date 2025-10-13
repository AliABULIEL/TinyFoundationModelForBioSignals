"""VitalDB Dataset for SSL pretraining with modality dropout.

This dataset loads preprocessed window files and applies SSL-specific augmentations:
- Modality dropout (randomly zero entire channels)
- Paired crops for contrastive/consistency objectives
- Subject-level train/val/test splits (no leakage)

Expected Data Structure:
    data_dir/
        train/
            case_0001_win_0000.npz  # Contains {'PPG': [T], 'ECG': [T], ...}
            case_0001_win_0001.npz
            ...
        val/
            ...
        test/
            ...
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import warnings


class VitalDBDataset(Dataset):
    """VitalDB dataset for SSL pretraining with modality dropout.
    
    Loads preprocessed 10-second non-overlapping windows and applies
    modality dropout for robust SSL pretraining.
    
    Args:
        data_dir: Root directory containing train/val/test subdirectories
        split: Split name ('train', 'val', 'test')
        channels: List of channel names to load (default: ['PPG', 'ECG'])
        window_sec: Window duration in seconds (default: 10.0)
        fs: Sampling rate in Hz (default: 125)
        return_pairs: If True, return two views (seg1, seg2) for contrastive learning
        apply_modality_dropout: If True, randomly zero entire channels
        modality_dropout_prob: Probability of dropping each channel (default: 0.25)
        transform: Optional transform to apply to signals
    
    Returns:
        If return_pairs=True: (seg1, seg2) each [C, T]
        If return_pairs=False: seg [C, T]
        
        Where T = window_sec * fs = 1250 for default params
    
    Example:
        >>> dataset = VitalDBDataset(
        ...     data_dir='data/vitaldb_windows',
        ...     split='train',
        ...     channels=['PPG', 'ECG'],
        ...     apply_modality_dropout=True
        ... )
        >>> seg1, seg2 = dataset[0]
        >>> seg1.shape, seg2.shape
        (torch.Size([2, 1250]), torch.Size([2, 1250]))
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        channels: List[str] = ['PPG', 'ECG'],
        window_sec: float = 10.0,
        fs: int = 125,
        return_pairs: bool = True,
        apply_modality_dropout: bool = True,
        modality_dropout_prob: float = 0.25,
        transform: Optional[callable] = None
    ):
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.split = split
        self.channels = channels
        self.window_sec = window_sec
        self.fs = fs
        self.return_pairs = return_pairs
        self.apply_modality_dropout = apply_modality_dropout and (split == 'train')
        self.modality_dropout_prob = modality_dropout_prob
        self.transform = transform
        
        # Expected number of samples
        self.T = int(window_sec * fs)
        self.C = len(channels)
        
        # Find all window files in split directory
        split_dir = self.data_dir / split
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")
        
        self.window_files = sorted(split_dir.glob('*.npz'))
        
        if len(self.window_files) == 0:
            warnings.warn(f"No window files found in {split_dir}")
        
        print(f"VitalDBDataset[{split}]: {len(self.window_files)} windows, "
              f"channels={channels}, T={self.T}")
    
    def __len__(self) -> int:
        """Return number of windows."""
        return len(self.window_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a window with optional modality dropout.
        
        Args:
            idx: Window index
        
        Returns:
            (seg1, seg2): Two views [C, T] if return_pairs=True
            seg: Single view [C, T] if return_pairs=False
        """
        # Load window file
        window_file = self.window_files[idx]
        
        try:
            data = np.load(window_file)
        except Exception as e:
            raise ValueError(f"Error loading {window_file}: {e}")
        
        # Extract channels
        signal_list = []
        for ch in self.channels:
            if ch not in data:
                raise ValueError(f"Channel '{ch}' not found in {window_file}")
            
            signal = data[ch]
            
            # Validate shape
            if signal.ndim == 1:
                signal = signal[:self.T]  # Truncate if needed
            else:
                raise ValueError(f"Expected 1D signal for {ch}, got shape {signal.shape}")
            
            # Pad if too short
            if len(signal) < self.T:
                signal = np.pad(signal, (0, self.T - len(signal)), mode='constant')
            
            signal_list.append(signal)
        
        # Stack channels: [C, T]
        signal = np.stack(signal_list, axis=0).astype(np.float32)
        
        # Convert to torch tensor
        signal = torch.from_numpy(signal)
        
        # Apply transform if provided
        if self.transform is not None:
            signal = self.transform(signal)
        
        # Create two views
        if self.return_pairs:
            seg1 = signal.clone()
            seg2 = signal.clone()
            
            # Apply modality dropout independently to each view
            if self.apply_modality_dropout:
                seg1 = self._apply_modality_dropout(seg1)
                seg2 = self._apply_modality_dropout(seg2)
            
            return seg1, seg2
        else:
            # Single view
            if self.apply_modality_dropout:
                signal = self._apply_modality_dropout(signal)
            
            return signal
    
    def _apply_modality_dropout(self, signal: torch.Tensor) -> torch.Tensor:
        """Randomly zero out entire channels.
        
        Args:
            signal: Input signal [C, T]
        
        Returns:
            signal: Signal with randomly dropped channels [C, T]
        """
        C, T = signal.shape
        
        for c in range(C):
            if torch.rand(1).item() < self.modality_dropout_prob:
                signal[c, :] = 0.0
        
        return signal
    
    def get_stats(self) -> Dict[str, any]:
        """Get dataset statistics.
        
        Returns:
            Dictionary with dataset info
        """
        return {
            'split': self.split,
            'num_windows': len(self),
            'channels': self.channels,
            'num_channels': self.C,
            'window_sec': self.window_sec,
            'sampling_rate': self.fs,
            'samples_per_window': self.T,
            'modality_dropout': self.apply_modality_dropout,
            'dropout_prob': self.modality_dropout_prob if self.apply_modality_dropout else 0.0
        }


def create_vitaldb_dataloaders(
    data_dir: str,
    batch_size: int = 128,
    channels: List[str] = ['PPG', 'ECG'],
    num_workers: int = 4,
    window_sec: float = 10.0,
    fs: int = 125,
    return_pairs: bool = True,
    apply_modality_dropout: bool = True,
    modality_dropout_prob: float = 0.25,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders for VitalDB SSL pretraining.
    
    Args:
        data_dir: Root directory with train/val/test subdirectories
        batch_size: Batch size (default: 128)
        channels: List of channel names (default: ['PPG', 'ECG'])
        num_workers: Number of worker processes (default: 4)
        window_sec: Window duration in seconds (default: 10.0)
        fs: Sampling rate in Hz (default: 125)
        return_pairs: If True, return paired views for contrastive learning
        apply_modality_dropout: Apply modality dropout during training
        modality_dropout_prob: Dropout probability per channel (default: 0.25)
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        (train_loader, val_loader, test_loader)
    
    Example:
        >>> train_loader, val_loader, test_loader = create_vitaldb_dataloaders(
        ...     data_dir='data/vitaldb_windows',
        ...     batch_size=128,
        ...     channels=['PPG', 'ECG']
        ... )
        >>> for seg1, seg2 in train_loader:
        ...     # seg1, seg2: [B, C, T] where B=128, C=2, T=1250
        ...     break
    """
    # Create datasets
    train_dataset = VitalDBDataset(
        data_dir=data_dir,
        split='train',
        channels=channels,
        window_sec=window_sec,
        fs=fs,
        return_pairs=return_pairs,
        apply_modality_dropout=apply_modality_dropout,
        modality_dropout_prob=modality_dropout_prob
    )
    
    val_dataset = VitalDBDataset(
        data_dir=data_dir,
        split='val',
        channels=channels,
        window_sec=window_sec,
        fs=fs,
        return_pairs=return_pairs,
        apply_modality_dropout=False,  # No dropout in val/test
        modality_dropout_prob=0.0
    )
    
    test_dataset = VitalDBDataset(
        data_dir=data_dir,
        split='test',
        channels=channels,
        window_sec=window_sec,
        fs=fs,
        return_pairs=return_pairs,
        apply_modality_dropout=False,  # No dropout in val/test
        modality_dropout_prob=0.0
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop last incomplete batch for SSL
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Print statistics
    print("\n" + "=" * 70)
    print("VitalDB SSL DataLoaders Created")
    print("=" * 70)
    for name, dataset in [('Train', train_dataset), ('Val', val_dataset), ('Test', test_dataset)]:
        stats = dataset.get_stats()
        print(f"{name:6s}: {stats['num_windows']:5d} windows, "
              f"{stats['num_channels']}ch, "
              f"{stats['samples_per_window']} samples, "
              f"dropout={stats['dropout_prob']:.2f}")
    print("=" * 70 + "\n")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    """Quick sanity check."""
    print("Testing VitalDBDataset...")
    print("=" * 70)
    
    # Note: This will fail if no data exists, but shows usage
    try:
        dataset = VitalDBDataset(
            data_dir='data/vitaldb_windows',
            split='train',
            channels=['PPG', 'ECG'],
            apply_modality_dropout=True
        )
        
        print(f"Dataset size: {len(dataset)} windows")
        
        if len(dataset) > 0:
            seg1, seg2 = dataset[0]
            print(f"seg1 shape: {seg1.shape}")
            print(f"seg2 shape: {seg2.shape}")
            print("✓ Dataset loading works!")
    except Exception as e:
        print(f"Note: {e}")
        print("(This is expected if no preprocessed data exists yet)")
    
    print("=" * 70)
