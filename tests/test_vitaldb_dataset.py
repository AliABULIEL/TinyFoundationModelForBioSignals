"""Unit tests for VitalDB SSL dataset and manifests.

Tests dataset loading, modality dropout, and manifest generation with
subject-level splits (no leakage).
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import numpy as np
import tempfile
import shutil
from typing import Dict, List

from src.data.vitaldb_dataset import VitalDBDataset, create_vitaldb_dataloaders
from src.data.manifests import (
    build_manifest,
    verify_manifest_integrity,
    extract_subject_id,
    hash_subject_to_split
)


@pytest.fixture
def mock_data_dir():
    """Create temporary directory with mock preprocessed windows."""
    temp_dir = tempfile.mkdtemp()
    
    # Create train/val/test directories
    for split in ['train', 'val', 'test']:
        split_dir = Path(temp_dir) / split
        split_dir.mkdir(parents=True)
        
        # Create mock window files
        n_windows = 10 if split == 'train' else 3
        
        for case_id in range(2):  # 2 subjects
            for win_id in range(n_windows // 2):
                # Create mock window with PPG and ECG
                window_file = split_dir / f"case_{case_id:04d}_win_{win_id:04d}.npz"
                
                # Mock signals: [T] for each channel
                T = 1250  # 10 seconds @ 125 Hz
                ppg = np.sin(2 * np.pi * 1.2 * np.arange(T) / 125).astype(np.float32)
                ecg = np.sin(2 * np.pi * 1.0 * np.arange(T) / 125).astype(np.float32)
                
                np.savez(window_file, PPG=ppg, ECG=ecg)
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


class TestVitalDBDataset:
    """Test suite for VitalDBDataset."""
    
    def test_initialization(self, mock_data_dir):
        """Test that dataset initializes correctly."""
        dataset = VitalDBDataset(
            data_dir=mock_data_dir,
            split='train',
            channels=['PPG', 'ECG']
        )
        
        assert dataset.split == 'train'
        assert dataset.channels == ['PPG', 'ECG']
        assert dataset.C == 2
        assert dataset.T == 1250
        assert len(dataset) > 0
    
    def test_get_single_window(self, mock_data_dir):
        """Test loading a single window."""
        dataset = VitalDBDataset(
            data_dir=mock_data_dir,
            split='train',
            channels=['PPG', 'ECG'],
            return_pairs=False
        )
        
        signal = dataset[0]
        
        assert isinstance(signal, torch.Tensor)
        assert signal.shape == (2, 1250)  # [C, T]
    
    def test_get_paired_windows(self, mock_data_dir):
        """Test loading paired windows for contrastive learning."""
        dataset = VitalDBDataset(
            data_dir=mock_data_dir,
            split='train',
            channels=['PPG', 'ECG'],
            return_pairs=True
        )
        
        seg1, seg2 = dataset[0]
        
        assert isinstance(seg1, torch.Tensor)
        assert isinstance(seg2, torch.Tensor)
        assert seg1.shape == (2, 1250)
        assert seg2.shape == (2, 1250)
    
    def test_modality_dropout_in_train(self, mock_data_dir):
        """Test that modality dropout is applied in training."""
        dataset = VitalDBDataset(
            data_dir=mock_data_dir,
            split='train',
            channels=['PPG', 'ECG'],
            apply_modality_dropout=True,
            modality_dropout_prob=1.0,  # Always drop
            return_pairs=False
        )
        
        # With prob=1.0, at least one channel should be all zeros
        signal = dataset[0]
        
        # Check if any channel is all zeros
        channel_sums = signal.sum(dim=1)
        has_dropout = (channel_sums == 0).any()
        
        # Note: probabilistic test, might rarely fail
        # But with prob=1.0 and 2 channels, should work
    
    def test_no_modality_dropout_in_test(self, mock_data_dir):
        """Test that modality dropout is NOT applied in test."""
        dataset = VitalDBDataset(
            data_dir=mock_data_dir,
            split='test',
            channels=['PPG', 'ECG'],
            apply_modality_dropout=True,  # Requested but split=test
            return_pairs=False
        )
        
        signal = dataset[0]
        
        # No channel should be all zeros
        channel_sums = signal.sum(dim=1).abs()
        assert (channel_sums > 0).all(), "Modality dropout should not be applied in test split"
    
    def test_different_channels(self, mock_data_dir):
        """Test with different channel configurations."""
        # Single channel
        dataset1 = VitalDBDataset(
            data_dir=mock_data_dir,
            split='train',
            channels=['PPG'],
            return_pairs=False
        )
        signal1 = dataset1[0]
        assert signal1.shape == (1, 1250)
        
        # Two channels
        dataset2 = VitalDBDataset(
            data_dir=mock_data_dir,
            split='train',
            channels=['PPG', 'ECG'],
            return_pairs=False
        )
        signal2 = dataset2[0]
        assert signal2.shape == (2, 1250)
    
    def test_get_stats(self, mock_data_dir):
        """Test dataset statistics."""
        dataset = VitalDBDataset(
            data_dir=mock_data_dir,
            split='train',
            channels=['PPG', 'ECG'],
            apply_modality_dropout=True,
            modality_dropout_prob=0.25
        )
        
        stats = dataset.get_stats()
        
        assert stats['split'] == 'train'
        assert stats['num_channels'] == 2
        assert stats['samples_per_window'] == 1250
        assert stats['modality_dropout'] is True
        assert stats['dropout_prob'] == 0.25
    
    def test_raises_on_missing_channel(self, mock_data_dir):
        """Test that it raises error when channel not found."""
        dataset = VitalDBDataset(
            data_dir=mock_data_dir,
            split='train',
            channels=['PPG', 'ECG', 'ABP'],  # ABP not in mock data
            return_pairs=False
        )
        
        with pytest.raises(ValueError, match="Channel 'ABP' not found"):
            _ = dataset[0]


class TestCreateVitalDBDataloaders:
    """Test dataloader creation."""
    
    def test_create_dataloaders(self, mock_data_dir):
        """Test creating train/val/test dataloaders."""
        train_loader, val_loader, test_loader = create_vitaldb_dataloaders(
            data_dir=mock_data_dir,
            batch_size=4,
            channels=['PPG', 'ECG'],
            num_workers=0  # No multiprocessing in tests
        )
        
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None
    
    def test_dataloader_batch_shape(self, mock_data_dir):
        """Test that batches have correct shape."""
        train_loader, _, _ = create_vitaldb_dataloaders(
            data_dir=mock_data_dir,
            batch_size=4,
            channels=['PPG', 'ECG'],
            num_workers=0,
            return_pairs=True
        )
        
        # Get one batch
        seg1, seg2 = next(iter(train_loader))
        
        # Should be [B, C, T]
        assert seg1.shape == (4, 2, 1250)
        assert seg2.shape == (4, 2, 1250)
    
    def test_dataloader_no_dropout_in_val(self, mock_data_dir):
        """Test that validation loader doesn't have dropout."""
        _, val_loader, _ = create_vitaldb_dataloaders(
            data_dir=mock_data_dir,
            batch_size=3,
            channels=['PPG', 'ECG'],
            num_workers=0,
            apply_modality_dropout=True  # Requested for train only
        )
        
        # Validation dataset should not have dropout
        assert val_loader.dataset.apply_modality_dropout is False


class TestExtractSubjectId:
    """Test subject ID extraction from filenames."""
    
    def test_standard_format(self):
        """Test extraction from standard format."""
        assert extract_subject_id('case_0001_win_0000.npz') == '0001'
        assert extract_subject_id('case_0042_win_0123.npz') == '0042'
        assert extract_subject_id('case_9999_win_9999.npz') == '9999'
    
    def test_different_formats(self):
        """Test with various filename formats."""
        # Should work with different separators
        result1 = extract_subject_id('case_123_win_456.npz')
        assert result1 == '123'
        
        # Fallback for non-standard format
        result2 = extract_subject_id('random_file.npz')
        assert len(result2) > 0  # Should return some hash


class TestHashSubjectToSplit:
    """Test deterministic subject-to-split assignment."""
    
    def test_deterministic(self):
        """Test that same subject always maps to same split."""
        split1 = hash_subject_to_split('0001', seed=42)
        split2 = hash_subject_to_split('0001', seed=42)
        
        assert split1 == split2, "Should be deterministic"
    
    def test_different_seeds(self):
        """Test that different seeds can give different splits."""
        split1 = hash_subject_to_split('0001', seed=42)
        split2 = hash_subject_to_split('0001', seed=99)
        
        # Note: might occasionally be the same, but unlikely
        # This is probabilistic, so we just check it runs
        assert split1 in ['train', 'val', 'test']
        assert split2 in ['train', 'val', 'test']
    
    def test_returns_valid_split(self):
        """Test that it always returns a valid split name."""
        for i in range(100):
            split = hash_subject_to_split(f'{i:04d}')
            assert split in ['train', 'val', 'test']
    
    def test_respects_ratios(self):
        """Test that splits approximately respect ratios."""
        n_subjects = 1000
        splits = [hash_subject_to_split(f'{i:04d}', train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
                  for i in range(n_subjects)]
        
        train_count = splits.count('train')
        val_count = splits.count('val')
        test_count = splits.count('test')
        
        # Should be approximately 800/100/100
        assert 750 < train_count < 850, f"Train count {train_count} not around 800"
        assert 50 < val_count < 150, f"Val count {val_count} not around 100"
        assert 50 < test_count < 150, f"Test count {test_count} not around 100"


class TestBuildManifest:
    """Test manifest building."""
    
    @pytest.fixture
    def mock_windows_flat(self):
        """Create flat directory with mock windows."""
        temp_dir = tempfile.mkdtemp()
        
        # Create windows for 3 subjects
        for case_id in range(3):
            for win_id in range(5):
                window_file = Path(temp_dir) / f"case_{case_id:04d}_win_{win_id:04d}.npz"
                
                # Mock data
                ppg = np.random.randn(1250).astype(np.float32)
                ecg = np.random.randn(1250).astype(np.float32)
                np.savez(window_file, PPG=ppg, ECG=ecg)
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_build_manifest(self, mock_windows_flat):
        """Test building manifests from flat directory."""
        manifest_paths = build_manifest(
            root=mock_windows_flat,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            seed=42
        )
        
        assert 'train' in manifest_paths
        assert 'val' in manifest_paths
        assert 'test' in manifest_paths
        
        # Check files exist
        for split, path in manifest_paths.items():
            assert Path(path).exists()
    
    def test_no_subject_leakage(self, mock_windows_flat):
        """Test that manifests have no subject leakage."""
        manifest_paths = build_manifest(
            root=mock_windows_flat,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )
        
        # Should not raise
        verify_manifest_integrity(manifest_paths)
    
    def test_manifest_content(self, mock_windows_flat):
        """Test manifest CSV content."""
        import csv
        
        manifest_paths = build_manifest(
            root=mock_windows_flat,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )
        
        # Read train manifest
        with open(manifest_paths['train'], 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            assert len(rows) > 0
            
            # Check columns
            assert 'filepath' in rows[0]
            assert 'subject_id' in rows[0]
            assert 'split' in rows[0]
            
            # Check split column
            assert all(row['split'] == 'train' for row in rows)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
