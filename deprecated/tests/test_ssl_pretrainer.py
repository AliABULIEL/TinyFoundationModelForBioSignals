"""Unit tests for SSL pretrainer.

Tests SSLTrainer with masking, encoding, decoding, and loss computation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import tempfile
import shutil
import json

from src.ssl.pretrainer import SSLTrainer
from src.ssl.masking import random_masking, block_masking
from src.ssl.objectives import MaskedSignalModeling, MultiResolutionSTFT
from src.models.decoders import ReconstructionHead1D


class MockEncoder(nn.Module):
    """Mock encoder for testing."""
    
    def __init__(self, d_model=192):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(1250, d_model)
    
    def forward(self, x):
        """Encode input [B, C, T] to [B, P, D]."""
        B, C, T = x.shape
        
        # Simple projection
        x_flat = x.reshape(B, -1)[:, :1250]
        features = self.proj(x_flat)
        
        # Return as [B, P=10, D]
        P = 10
        return features.unsqueeze(1).expand(B, P, self.d_model)


@pytest.fixture
def mock_components():
    """Create mock components for testing."""
    encoder = MockEncoder(d_model=192)
    decoder = ReconstructionHead1D(d_model=192, patch_size=125, n_channels=2)
    
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=1e-4
    )
    
    msm_criterion = MaskedSignalModeling(patch_size=125)
    stft_criterion = MultiResolutionSTFT(
        n_ffts=[512, 1024],
        hop_lengths=[128, 256],
        weight=1.0
    )
    
    return {
        'encoder': encoder,
        'decoder': decoder,
        'optimizer': optimizer,
        'msm_criterion': msm_criterion,
        'stft_criterion': stft_criterion
    }


@pytest.fixture
def mock_dataloaders():
    """Create mock dataloaders."""
    # Create synthetic data
    n_samples = 32
    C, T = 2, 1250
    
    # Training data
    X_train = torch.randn(n_samples, C, T)
    train_dataset = TensorDataset(X_train)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    # Validation data
    X_val = torch.randn(16, C, T)
    val_dataset = TensorDataset(X_val)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    return train_loader, val_loader


@pytest.fixture
def temp_save_dir():
    """Create temporary directory for checkpoints."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestSSLTrainer:
    """Test suite for SSLTrainer."""
    
    def test_initialization(self, mock_components):
        """Test trainer initialization."""
        trainer = SSLTrainer(
            encoder=mock_components['encoder'],
            decoder=mock_components['decoder'],
            optimizer=mock_components['optimizer'],
            msm_criterion=mock_components['msm_criterion'],
            device='cpu',
            use_amp=False
        )
        
        assert trainer.device == 'cpu'
        assert trainer.use_amp is False
        assert trainer.gradient_clip == 1.0
        assert trainer.best_val_loss == float('inf')
    
    def test_initialization_with_stft(self, mock_components):
        """Test initialization with STFT loss."""
        trainer = SSLTrainer(
            encoder=mock_components['encoder'],
            decoder=mock_components['decoder'],
            optimizer=mock_components['optimizer'],
            msm_criterion=mock_components['msm_criterion'],
            stft_criterion=mock_components['stft_criterion'],
            stft_weight=0.3,
            device='cpu',
            use_amp=False
        )
        
        assert trainer.stft_criterion is not None
        assert trainer.stft_weight == 0.3
    
    def test_fit_runs(self, mock_components, mock_dataloaders, temp_save_dir):
        """Test that training runs without errors."""
        train_loader, val_loader = mock_dataloaders
        
        trainer = SSLTrainer(
            encoder=mock_components['encoder'],
            decoder=mock_components['decoder'],
            optimizer=mock_components['optimizer'],
            msm_criterion=mock_components['msm_criterion'],
            device='cpu',
            use_amp=False
        )
        
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=2,
            save_dir=temp_save_dir,
            mask_ratio=0.4,
            log_interval=10
        )
        
        # Check history structure
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert 'train_msm' in history
        assert 'val_msm' in history
        
        # Check lengths
        assert len(history['train_loss']) == 2
        assert len(history['val_loss']) == 2
    
    def test_checkpoints_saved(self, mock_components, mock_dataloaders, temp_save_dir):
        """Test that checkpoints are saved."""
        train_loader, val_loader = mock_dataloaders
        
        trainer = SSLTrainer(
            encoder=mock_components['encoder'],
            decoder=mock_components['decoder'],
            optimizer=mock_components['optimizer'],
            msm_criterion=mock_components['msm_criterion'],
            device='cpu',
            use_amp=False
        )
        
        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=2,
            save_dir=temp_save_dir,
            save_interval=1
        )
        
        save_dir = Path(temp_save_dir)
        
        # Check that checkpoints exist
        assert (save_dir / 'best_model.pt').exists()
        assert (save_dir / 'last_model.pt').exists()
        assert (save_dir / 'training_history.json').exists()
    
    def test_load_checkpoint(self, mock_components, mock_dataloaders, temp_save_dir):
        """Test loading checkpoint."""
        train_loader, val_loader = mock_dataloaders
        
        trainer = SSLTrainer(
            encoder=mock_components['encoder'],
            decoder=mock_components['decoder'],
            optimizer=mock_components['optimizer'],
            msm_criterion=mock_components['msm_criterion'],
            device='cpu',
            use_amp=False
        )
        
        # Train briefly
        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=1,
            save_dir=temp_save_dir
        )
        
        # Load checkpoint
        checkpoint_path = Path(temp_save_dir) / 'best_model.pt'
        trainer.load_checkpoint(str(checkpoint_path))
        
        assert trainer.best_val_loss < float('inf')
    
    def test_training_with_stft_loss(self, mock_components, mock_dataloaders, temp_save_dir):
        """Test training with both MSM and STFT losses."""
        train_loader, val_loader = mock_dataloaders
        
        trainer = SSLTrainer(
            encoder=mock_components['encoder'],
            decoder=mock_components['decoder'],
            optimizer=mock_components['optimizer'],
            msm_criterion=mock_components['msm_criterion'],
            stft_criterion=mock_components['stft_criterion'],
            stft_weight=0.3,
            device='cpu',
            use_amp=False
        )
        
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=2,
            save_dir=temp_save_dir
        )
        
        # Check STFT losses are tracked
        assert 'train_stft' in history
        assert 'val_stft' in history
        assert len(history['train_stft']) == 2
        
        # STFT losses should be non-zero
        assert any(x > 0 for x in history['train_stft'])
    
    def test_gradient_clipping(self, mock_components, mock_dataloaders, temp_save_dir):
        """Test that gradient clipping is applied."""
        train_loader, val_loader = mock_dataloaders
        
        # Create trainer with gradient clipping
        trainer = SSLTrainer(
            encoder=mock_components['encoder'],
            decoder=mock_components['decoder'],
            optimizer=mock_components['optimizer'],
            msm_criterion=mock_components['msm_criterion'],
            gradient_clip=1.0,
            device='cpu',
            use_amp=False
        )
        
        # Should run without error
        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=1,
            save_dir=temp_save_dir
        )
    
    def test_different_mask_functions(self, mock_components, mock_dataloaders, temp_save_dir):
        """Test with different masking functions."""
        train_loader, val_loader = mock_dataloaders
        
        # Test with block masking
        trainer = SSLTrainer(
            encoder=mock_components['encoder'],
            decoder=mock_components['decoder'],
            optimizer=mock_components['optimizer'],
            msm_criterion=mock_components['msm_criterion'],
            mask_fn=block_masking,
            device='cpu',
            use_amp=False
        )
        
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=1,
            save_dir=temp_save_dir
        )
        
        assert len(history['train_loss']) == 1
    
    def test_training_history_json(self, mock_components, mock_dataloaders, temp_save_dir):
        """Test that training history is saved as JSON."""
        train_loader, val_loader = mock_dataloaders
        
        trainer = SSLTrainer(
            encoder=mock_components['encoder'],
            decoder=mock_components['decoder'],
            optimizer=mock_components['optimizer'],
            msm_criterion=mock_components['msm_criterion'],
            device='cpu',
            use_amp=False
        )
        
        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=2,
            save_dir=temp_save_dir
        )
        
        # Load and verify JSON
        history_path = Path(temp_save_dir) / 'training_history.json'
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        assert 'train_loss' in history
        assert len(history['train_loss']) == 2
    
    def test_best_model_tracking(self, mock_components, mock_dataloaders, temp_save_dir):
        """Test that best model is tracked correctly."""
        train_loader, val_loader = mock_dataloaders
        
        trainer = SSLTrainer(
            encoder=mock_components['encoder'],
            decoder=mock_components['decoder'],
            optimizer=mock_components['optimizer'],
            msm_criterion=mock_components['msm_criterion'],
            device='cpu',
            use_amp=False
        )
        
        initial_best = trainer.best_val_loss
        
        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=2,
            save_dir=temp_save_dir
        )
        
        # Best val loss should be updated
        assert trainer.best_val_loss < initial_best
    
    def test_paired_data_handling(self, mock_components, temp_save_dir):
        """Test handling of paired data (for contrastive learning)."""
        # Create paired dataset
        n_samples = 16
        C, T = 2, 1250
        
        X1 = torch.randn(n_samples, C, T)
        X2 = torch.randn(n_samples, C, T)
        
        train_dataset = TensorDataset(X1, X2)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        
        val_dataset = TensorDataset(X1[:8], X2[:8])
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
        
        trainer = SSLTrainer(
            encoder=mock_components['encoder'],
            decoder=mock_components['decoder'],
            optimizer=mock_components['optimizer'],
            msm_criterion=mock_components['msm_criterion'],
            device='cpu',
            use_amp=False
        )
        
        # Should handle paired data (uses first view)
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=1,
            save_dir=temp_save_dir
        )
        
        assert len(history['train_loss']) == 1


class TestSSLTrainerCheckpoint:
    """Test checkpoint loading and saving."""
    
    def test_checkpoint_contents(self, mock_components, mock_dataloaders, temp_save_dir):
        """Test that checkpoint contains all necessary information."""
        train_loader, val_loader = mock_dataloaders
        
        trainer = SSLTrainer(
            encoder=mock_components['encoder'],
            decoder=mock_components['decoder'],
            optimizer=mock_components['optimizer'],
            msm_criterion=mock_components['msm_criterion'],
            device='cpu',
            use_amp=False
        )
        
        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=1,
            save_dir=temp_save_dir
        )
        
        # Load checkpoint
        checkpoint_path = Path(temp_save_dir) / 'best_model.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Verify contents
        assert 'epoch' in checkpoint
        assert 'encoder_state_dict' in checkpoint
        assert 'decoder_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'best_val_loss' in checkpoint
        assert 'metrics' in checkpoint
        assert 'config' in checkpoint
    
    def test_checkpoint_config(self, mock_components, mock_dataloaders, temp_save_dir):
        """Test that checkpoint config is saved correctly."""
        train_loader, val_loader = mock_dataloaders
        
        trainer = SSLTrainer(
            encoder=mock_components['encoder'],
            decoder=mock_components['decoder'],
            optimizer=mock_components['optimizer'],
            msm_criterion=mock_components['msm_criterion'],
            gradient_clip=2.0,
            stft_weight=0.5,
            device='cpu',
            use_amp=False
        )
        
        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=1,
            save_dir=temp_save_dir
        )
        
        checkpoint = torch.load(
            Path(temp_save_dir) / 'best_model.pt',
            map_location='cpu'
        )
        
        # Check config
        assert checkpoint['config']['gradient_clip'] == 2.0
        assert checkpoint['config']['stft_weight'] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
