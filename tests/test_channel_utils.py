"""Unit tests for channel inflation utilities."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tempfile
import pytest
import torch
import torch.nn as nn

from src.models.channel_utils import (
    load_pretrained_with_channel_inflate,
    unfreeze_last_n_blocks,
    verify_channel_inflation,
    get_channel_inflation_report,
    _inflate_channel_weights
)


class DummyModel(nn.Module):
    """Dummy model for testing channel inflation."""
    
    def __init__(self, input_channels: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.input_channels = input_channels
        
        # Input projection (channel-dependent)
        self.input_proj = nn.Linear(input_channels, hidden_dim)
        
        # Transformer blocks (channel-independent)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                batch_first=True
            )
            for _ in range(4)
        ])
        
        # Output head
        self.head = nn.Linear(hidden_dim, 2)  # Binary classification
    
    def forward(self, x):
        # x: [batch, time, channels]
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)  # Average over time
        return self.head(x)


def test_inflate_channel_weights():
    """Test basic channel weight inflation."""
    # Create a simple weight tensor
    pretrained = torch.randn(10, 2)  # [features, channels]
    new_shape = torch.zeros(10, 5)
    
    inflated = _inflate_channel_weights(
        pretrained,
        new_shape,
        pretrain_channels=2,
        finetune_channels=5,
        param_name="test_weight"
    )
    
    assert inflated is not None
    assert inflated.shape == new_shape.shape
    
    # Check that first 2 channels match pretrained
    assert torch.allclose(inflated[:, :2], pretrained)
    
    # Check that new channels are initialized (not zero)
    assert inflated[:, 2:].abs().sum() > 0


def test_channel_inflation_pipeline():
    """Test complete channel inflation pipeline."""
    # Create and save a 2-channel model
    model_2ch = DummyModel(input_channels=2)
    
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pt', delete=False) as f:
        checkpoint_path = f.name
        torch.save({
            'model_state_dict': model_2ch.state_dict(),
            'config': {
                'input_channels': 2,
                'hidden_dim': 64
            }
        }, f)
    
    try:
        # Create model config for 5 channels
        model_config = {
            'input_channels': 5,
            'hidden_dim': 64
        }
        
        # This would normally call create_ttm_model, but for testing
        # we'll manually create the inflated model
        model_5ch_state = {}
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        pretrained_state = checkpoint['model_state_dict']
        
        # Create 5-ch model
        model_5ch = DummyModel(input_channels=5)
        new_state = model_5ch.state_dict()
        
        # Manually inflate input_proj weights
        for name, param in pretrained_state.items():
            if name in new_state:
                if param.shape == new_state[name].shape:
                    model_5ch_state[name] = param
                elif 'input_proj' in name and 'weight' in name:
                    # Inflate this parameter
                    inflated = _inflate_channel_weights(
                        param,
                        new_state[name],
                        2, 5,
                        name
                    )
                    if inflated is not None:
                        model_5ch_state[name] = inflated
        
        # Load inflated state
        model_5ch.load_state_dict(model_5ch_state, strict=False)
        
        # Verify some weights transferred
        assert len(model_5ch_state) > 0
        
        print("✓ Channel inflation pipeline test passed")
        
    finally:
        # Cleanup
        Path(checkpoint_path).unlink()


def test_unfreeze_last_n_blocks():
    """Test unfreezing last N blocks."""
    model = DummyModel(input_channels=2)
    
    # Initially freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze last 2 blocks
    unfreeze_last_n_blocks(model, n=2, verbose=False)
    
    # Check that some parameters are now trainable
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable_params) > 0
    
    print("✓ Unfreeze last N blocks test passed")


def test_verify_channel_inflation():
    """Test verification of channel inflation."""
    # Create two identical models
    model_2ch = DummyModel(input_channels=2)
    model_2ch_copy = DummyModel(input_channels=2)
    
    # Copy weights
    model_2ch_copy.load_state_dict(model_2ch.state_dict())
    
    # Verify they match
    result = verify_channel_inflation(model_2ch, model_2ch_copy, verbose=False)
    assert result is True
    
    # Create model with different weights
    model_2ch_diff = DummyModel(input_channels=2)
    
    # Verify they don't match
    result = verify_channel_inflation(model_2ch, model_2ch_diff, verbose=False)
    assert result is False
    
    print("✓ Verify channel inflation test passed")


def test_get_channel_inflation_report():
    """Test channel inflation report generation."""
    model = DummyModel(input_channels=2)
    
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pt', delete=False) as f:
        checkpoint_path = f.name
        torch.save(model.state_dict(), f)
    
    try:
        report = get_channel_inflation_report(
            checkpoint_path,
            pretrain_channels=2,
            finetune_channels=5
        )
        
        assert 'total_params' in report
        assert 'channel_dependent' in report
        assert 'transferable' in report
        assert 'report' in report
        
        assert report['total_params'] > 0
        assert isinstance(report['report'], str)
        
        print("✓ Channel inflation report test passed")
        
    finally:
        Path(checkpoint_path).unlink()


if __name__ == "__main__":
    print("Running channel inflation utility tests...")
    print("=" * 70)
    
    test_inflate_channel_weights()
    test_channel_inflation_pipeline()
    test_unfreeze_last_n_blocks()
    test_verify_channel_inflation()
    test_get_channel_inflation_report()
    
    print("=" * 70)
    print("✓ All tests passed!")
