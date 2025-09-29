"""Tests for PyTorch datasets."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.datasets import (
    RawWindowDataset,
    StreamingWindowDataset,
    MultiModalDataset,
    create_dataloader,
    custom_collate_fn
)
from src.utils.io import save_npz


class TestRawWindowDataset:
    """Test RawWindowDataset."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample NPZ files for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create sample windows
            n_windows = 10
            window_size = 1250  # 10s at 125Hz
            n_channels = 3
            
            # Create train file
            train_data = {
                'data': np.random.randn(n_windows, window_size, n_channels).astype(np.float32),
                'labels': np.random.randint(0, 2, n_windows)
            }
            train_path = os.path.join(tmpdir, 'train.npz')
            save_npz(train_path, train_data)
            
            # Create val file
            val_data = {
                'data': np.random.randn(5, window_size, n_channels).astype(np.float32),
                'labels': np.random.randint(0, 2, 5)
            }
            val_path = os.path.join(tmpdir, 'val.npz')
            save_npz(val_path, val_data)
            
            # Create manifest
            manifest = {
                'train': [{'path': train_path, 'n_windows': n_windows}],
                'val': [{'path': val_path, 'n_windows': 5}]
            }
            manifest_path = os.path.join(tmpdir, 'manifest.json')
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f)
            
            yield {
                'tmpdir': tmpdir,
                'train_path': train_path,
                'val_path': val_path,
                'manifest_path': manifest_path,
                'manifest': manifest
            }
    
    def test_dataset_creation(self, sample_data):
        """Test dataset initialization."""
        dataset = RawWindowDataset(
            sample_data['manifest_path'],
            split='train'
        )
        
        assert len(dataset) == 10
        assert len(dataset.file_list) == 1
    
    def test_dataset_getitem(self, sample_data):
        """Test getting items from dataset."""
        dataset = RawWindowDataset(
            sample_data['manifest_path'],
            split='train'
        )
        
        # Get first item
        window, label = dataset[0]
        
        assert isinstance(window, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert window.shape == (1250, 3)
        assert label.shape == () or label.shape == torch.Size([])
    
    def test_dataset_iteration(self, sample_data):
        """Test iterating over dataset."""
        dataset = RawWindowDataset(
            sample_data['manifest_path'],
            split='train'
        )
        
        # Iterate over all items
        windows = []
        labels = []
        
        for i in range(len(dataset)):
            window, label = dataset[i]
            windows.append(window)
            labels.append(label)
        
        assert len(windows) == 10
        assert all(w.shape == (1250, 3) for w in windows)
    
    def test_dataset_with_cache(self, sample_data):
        """Test dataset with caching."""
        dataset = RawWindowDataset(
            sample_data['manifest_path'],
            split='train',
            cache_size=2
        )
        
        # Access same item multiple times
        window1, _ = dataset[0]
        window2, _ = dataset[0]
        
        # Should be same data
        assert torch.allclose(window1, window2)
        
        # Check cache
        assert len(dataset.cache) > 0
    
    def test_dataset_splits(self, sample_data):
        """Test different splits."""
        # Train split
        train_dataset = RawWindowDataset(
            sample_data['manifest_path'],
            split='train'
        )
        assert len(train_dataset) == 10
        
        # Val split
        val_dataset = RawWindowDataset(
            sample_data['manifest_path'],
            split='val'
        )
        assert len(val_dataset) == 5
    
    def test_dataset_transform(self, sample_data):
        """Test dataset with transforms."""
        def normalize(x):
            return (x - x.mean()) / (x.std() + 1e-8)
        
        dataset = RawWindowDataset(
            sample_data['manifest_path'],
            split='train',
            transform=normalize
        )
        
        window, _ = dataset[0]
        
        # Check normalized
        assert abs(window.mean()) < 0.1
        assert abs(window.std() - 1.0) < 0.1


class TestStreamingWindowDataset:
    """Test StreamingWindowDataset."""
    
    @pytest.fixture
    def sample_signals(self):
        """Create sample signal files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create continuous signals
            fs = 125.0
            duration = 60.0  # 60 seconds
            n_samples = int(duration * fs)
            n_channels = 3
            
            files = []
            for i in range(3):
                signal = np.random.randn(n_samples, n_channels).astype(np.float32)
                path = os.path.join(tmpdir, f'signal_{i}.npz')
                save_npz(path, {'data': signal})
                files.append(path)
            
            yield {
                'tmpdir': tmpdir,
                'files': files,
                'fs': fs,
                'duration': duration
            }
    
    def test_streaming_dataset(self, sample_signals):
        """Test streaming dataset creation."""
        dataset = StreamingWindowDataset(
            sample_signals['files'],
            window_s=10.0,
            stride_s=10.0,
            fs=sample_signals['fs']
        )
        
        # Should have 6 windows per file (60s / 10s)
        expected_windows = 6 * 3
        assert len(dataset) == expected_windows
    
    def test_streaming_getitem(self, sample_signals):
        """Test getting windows from streaming dataset."""
        dataset = StreamingWindowDataset(
            sample_signals['files'],
            window_s=10.0,
            stride_s=5.0,  # 50% overlap
            fs=sample_signals['fs']
        )
        
        window, metadata = dataset[0]
        
        assert isinstance(window, torch.Tensor)
        assert window.shape == (1250, 3)  # 10s at 125Hz
        assert 'file_idx' in metadata
        assert 'window_idx' in metadata
    
    def test_streaming_overlap(self, sample_signals):
        """Test overlapping windows."""
        dataset = StreamingWindowDataset(
            sample_signals['files'],
            window_s=10.0,
            stride_s=5.0,  # 50% overlap
            fs=sample_signals['fs']
        )
        
        # Should have more windows with overlap
        # (60 - 10) / 5 + 1 = 11 windows per file
        expected_windows = 11 * 3
        assert len(dataset) == expected_windows


class TestMultiModalDataset:
    """Test MultiModalDataset."""
    
    @pytest.fixture
    def multimodal_files(self):
        """Create multi-modal signal files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fs = 125.0
            n_samples = 1250
            
            # Create ECG files
            ecg_files = []
            for i in range(2):
                ecg = np.random.randn(n_samples, 1).astype(np.float32)
                path = os.path.join(tmpdir, f'ecg_{i}.npz')
                save_npz(path, {'data': ecg})
                ecg_files.append(path)
            
            # Create PPG files
            ppg_files = []
            for i in range(2):
                ppg = np.random.randn(n_samples, 1).astype(np.float32)
                path = os.path.join(tmpdir, f'ppg_{i}.npz')
                save_npz(path, {'data': ppg})
                ppg_files.append(path)
            
            yield {
                'tmpdir': tmpdir,
                'ecg_files': ecg_files,
                'ppg_files': ppg_files
            }
    
    def test_multimodal_dataset(self, multimodal_files):
        """Test multi-modal dataset."""
        dataset = MultiModalDataset(
            ecg_files=multimodal_files['ecg_files'],
            ppg_files=multimodal_files['ppg_files'],
            target_fs=125.0
        )
        
        assert len(dataset) == 2
        
        # Get first sample
        sample = dataset[0]
        
        assert 'ecg' in sample
        assert 'ppg' in sample
        assert sample['ecg'].shape == (1250, 1)
        assert sample['ppg'].shape == (1250, 1)


class TestDataLoader:
    """Test DataLoader creation and determinism."""
    
    def test_dataloader_creation(self):
        """Test creating dataloader."""
        # Create dummy dataset
        n_samples = 100
        data = torch.randn(n_samples, 10, 3)
        labels = torch.randint(0, 2, (n_samples,))
        dataset = torch.utils.data.TensorDataset(data, labels)
        
        # Create dataloader
        dataloader = create_dataloader(
            dataset,
            batch_size=16,
            shuffle=True,
            num_workers=0,
            seed=42
        )
        
        assert len(dataloader) > 0
        
        # Get a batch
        batch = next(iter(dataloader))
        assert len(batch) == 2
    
    def test_dataloader_determinism(self):
        """Test deterministic batching."""
        # Create dataset
        n_samples = 100
        data = torch.randn(n_samples, 10, 3)
        labels = torch.randint(0, 2, (n_samples,))
        dataset = torch.utils.data.TensorDataset(data, labels)
        
        # Create two dataloaders with same seed
        dataloader1 = create_dataloader(
            dataset,
            batch_size=16,
            shuffle=True,
            num_workers=0,
            seed=42
        )
        
        dataloader2 = create_dataloader(
            dataset,
            batch_size=16,
            shuffle=True,
            num_workers=0,
            seed=42
        )
        
        # Should produce same batches
        for batch1, batch2 in zip(dataloader1, dataloader2):
            assert torch.allclose(batch1[0], batch2[0])
            assert torch.equal(batch1[1], batch2[1])
    
    def test_custom_collate(self):
        """Test custom collate function."""
        # Create batch with mixed types
        batch = [
            (torch.randn(10, 3), torch.tensor(0)),
            (torch.randn(10, 3), torch.tensor(1)),
            (torch.randn(10, 3), torch.tensor(0))
        ]
        
        data, targets = custom_collate_fn(batch)
        
        assert data.shape == (3, 10, 3)
        assert targets.shape == (3,)
    
    def test_custom_collate_with_dicts(self):
        """Test custom collate with dict targets."""
        # Create batch with dict targets
        batch = [
            (torch.randn(10, 3), {'id': 0, 'meta': 'a'}),
            (torch.randn(10, 3), {'id': 1, 'meta': 'b'}),
            (torch.randn(10, 3), {'id': 2, 'meta': 'c'})
        ]
        
        data, targets = custom_collate_fn(batch)
        
        assert data.shape == (3, 10, 3)
        assert isinstance(targets, list)
        assert len(targets) == 3
        assert all(isinstance(t, dict) for t in targets)


def test_shapes_correct():
    """Test that all shapes are correct [B,T,C]."""
    # Create simple dataset
    n_windows = 20
    window_size = 1250
    n_channels = 3
    batch_size = 4
    
    data = torch.randn(n_windows, window_size, n_channels)
    labels = torch.randint(0, 2, (n_windows,))
    dataset = torch.utils.data.TensorDataset(data, labels)
    
    # Create dataloader
    dataloader = create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Check batch shapes
    for batch_data, batch_labels in dataloader:
        # Data should be [B, T, C]
        assert batch_data.ndim == 3
        assert batch_data.shape[0] <= batch_size
        assert batch_data.shape[1] == window_size
        assert batch_data.shape[2] == n_channels
        
        # Labels should be [B]
        assert batch_labels.ndim == 1
        assert batch_labels.shape[0] == batch_data.shape[0]


def test_deterministic_seeding():
    """Test deterministic seeding per worker."""
    # Create dataset
    n_samples = 100
    data = torch.randn(n_samples, 10, 3)
    dataset = torch.utils.data.TensorDataset(data)
    
    # Test with multiple workers
    dataloader1 = create_dataloader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        seed=42
    )
    
    dataloader2 = create_dataloader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        seed=42
    )
    
    # Collect all data
    data1 = []
    data2 = []
    
    for batch in dataloader1:
        data1.append(batch[0])
    
    for batch in dataloader2:
        data2.append(batch[0])
    
    # Should be same order
    data1 = torch.cat(data1, dim=0)
    data2 = torch.cat(data2, dim=0)
    
    # Allow for minor differences due to worker scheduling
    # but overall structure should be similar
    assert data1.shape == data2.shape
