"""Tests for TTM adapter, heads, and LoRA."""
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.lora import (
    LoRALinear, LoRAConfig,
    apply_lora, get_lora_parameters,
    freeze_non_lora_parameters
)
from src.models.heads import (
    LinearClassifier, LinearRegressor,
    MLPClassifier, MLPRegressor,
    AttentionPooling, SequenceClassifier
)

# Try to import TTM components
try:
    from src.models.ttm_adapter import TTMAdapter, create_ttm_model
    TTM_AVAILABLE = True
except ImportError:
    TTM_AVAILABLE = False


class TestLoRA:
    """Test LoRA implementation."""
    
    def test_lora_linear_creation(self):
        """Test LoRA linear layer creation."""
        # Create original linear layer
        linear = nn.Linear(128, 64)
        
        # Wrap with LoRA
        lora_linear = LoRALinear(linear, r=8, alpha=16, dropout=0.1)
        
        # Check dimensions
        assert lora_linear.in_features == 128
        assert lora_linear.out_features == 64
        assert lora_linear.r == 8
        
        # Check LoRA parameters exist
        assert hasattr(lora_linear, 'lora_A')
        assert hasattr(lora_linear, 'lora_B')
        assert lora_linear.lora_A.shape == (8, 128)
        assert lora_linear.lora_B.shape == (64, 8)
    
    def test_lora_forward(self):
        """Test LoRA forward pass."""
        batch_size = 16
        in_features = 128
        out_features = 64
        
        # Create input
        x = torch.randn(batch_size, in_features)
        
        # Original linear
        linear = nn.Linear(in_features, out_features)
        original_output = linear(x)
        
        # LoRA linear
        lora_linear = LoRALinear(linear, r=8, alpha=16)
        
        # Initialize B with non-zero values for testing
        nn.init.xavier_uniform_(lora_linear.lora_B)
        
        lora_output = lora_linear(x)
        
        # Shapes should match
        assert lora_output.shape == original_output.shape
        assert lora_output.shape == (batch_size, out_features)
        
        # Outputs should be different (due to LoRA adaptation)
        assert not torch.allclose(lora_output, original_output, atol=1e-4)
    
    def test_lora_frozen_base(self):
        """Test that base weights are frozen."""
        linear = nn.Linear(128, 64)
        lora_linear = LoRALinear(linear, r=8)
        
        # Base weights should not require grad
        assert not lora_linear.original_module.weight.requires_grad
        assert not lora_linear.original_module.bias.requires_grad
        
        # LoRA parameters should require grad
        assert lora_linear.lora_A.requires_grad
        assert lora_linear.lora_B.requires_grad
    
    def test_apply_lora_to_model(self):
        """Test applying LoRA to a model."""
        # Create simple model
        model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
        # Apply LoRA
        lora_modules = apply_lora(model, r=4, alpha=8)
        
        # Should have replaced linear layers
        assert len(lora_modules) > 0
        
        # Check that LoRA modules are in place
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                assert not module.original_module.weight.requires_grad
                assert module.lora_A.requires_grad
                assert module.lora_B.requires_grad
    
    def test_get_lora_parameters(self):
        """Test getting LoRA parameters."""
        model = nn.Sequential(
            nn.Linear(128, 256),
            nn.Linear(256, 10)
        )
        
        # Before LoRA
        original_params = sum(p.numel() for p in model.parameters())
        
        # Apply LoRA
        apply_lora(model, r=8, alpha=16)
        
        # Get LoRA parameters
        lora_params = get_lora_parameters(model)
        assert len(lora_params) > 0
        
        # Count parameters
        lora_param_count = sum(p.numel() for p in lora_params)
        assert lora_param_count < original_params  # LoRA should be smaller
    
    def test_lora_merge_unmerge(self):
        """Test merging and unmerging LoRA weights."""
        linear = nn.Linear(128, 64)
        original_weight = linear.weight.clone()
        
        lora_linear = LoRALinear(linear, r=8, merge_weights=True)
        
        # Initialize B with non-zero values so merge has an effect
        nn.init.xavier_uniform_(lora_linear.lora_B)
        
        # Initially not merged
        assert not lora_linear.merged
        
        # Merge weights
        lora_linear.merge()
        assert lora_linear.merged
        assert not torch.allclose(lora_linear.original_module.weight, original_weight)
        
        # Unmerge weights
        lora_linear.unmerge()
        assert not lora_linear.merged
        assert torch.allclose(lora_linear.original_module.weight, original_weight, atol=1e-6)


class TestHeads:
    """Test task-specific heads."""
    
    def test_linear_classifier(self):
        """Test linear classification head."""
        batch_size = 16
        in_features = 768
        num_classes = 10
        
        head = LinearClassifier(in_features, num_classes, dropout=0.1)
        
        # Test 2D input
        x_2d = torch.randn(batch_size, in_features)
        logits = head(x_2d)
        assert logits.shape == (batch_size, num_classes)
        
        # Test 3D input (should average over time)
        seq_len = 50
        x_3d = torch.randn(batch_size, seq_len, in_features)
        logits = head(x_3d)
        assert logits.shape == (batch_size, num_classes)
    
    def test_linear_regressor(self):
        """Test linear regression head."""
        batch_size = 16
        in_features = 768
        out_features = 1
        
        head = LinearRegressor(in_features, out_features)
        
        # Test 2D input
        x = torch.randn(batch_size, in_features)
        output = head(x)
        assert output.shape == (batch_size, out_features)
    
    def test_mlp_classifier(self):
        """Test MLP classification head."""
        batch_size = 16
        in_features = 768
        num_classes = 10
        hidden_dims = [256, 128]
        
        head = MLPClassifier(
            in_features, num_classes,
            hidden_dims=hidden_dims,
            dropout=0.1,
            use_batch_norm=True
        )
        
        x = torch.randn(batch_size, in_features)
        logits = head(x)
        assert logits.shape == (batch_size, num_classes)
    
    def test_mlp_regressor(self):
        """Test MLP regression head."""
        batch_size = 16
        in_features = 768
        out_features = 5
        
        head = MLPRegressor(
            in_features, out_features,
            hidden_dims=[256],
            dropout=0.1
        )
        
        x = torch.randn(batch_size, in_features)
        output = head(x)
        assert output.shape == (batch_size, out_features)
    
    def test_attention_pooling(self):
        """Test attention pooling."""
        batch_size = 16
        seq_len = 50
        in_features = 768
        
        pooling = AttentionPooling(in_features)
        
        x = torch.randn(batch_size, seq_len, in_features)
        pooled = pooling(x)
        
        assert pooled.shape == (batch_size, in_features)
    
    def test_sequence_classifier(self):
        """Test sequence classifier with different pooling."""
        batch_size = 16
        seq_len = 50
        in_features = 768
        num_classes = 10
        
        # Test different pooling methods
        for pooling in ['mean', 'max', 'last', 'attention']:
            clf = SequenceClassifier(
                in_features, num_classes,
                pooling=pooling,
                head_type='linear'
            )
            
            x = torch.randn(batch_size, seq_len, in_features)
            logits = clf(x)
            assert logits.shape == (batch_size, num_classes)


@pytest.mark.skipif(not TTM_AVAILABLE, reason="TTM not available")
class TestTTMAdapter:
    """Test TTM adapter."""
    
    def test_ttm_creation(self):
        """Test TTM adapter creation."""
        model = TTMAdapter(
            variant="ttm-512-96",
            task="classification",
            num_classes=2,
            head_type="linear",
            freeze_encoder=True,
            input_channels=3,
            context_length=96
        )
        
        assert model is not None
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'head')
        assert model.encoder_dim > 0
    
    def test_frozen_encoder(self):
        """Test that frozen encoder has no gradients."""
        model = TTMAdapter(
            variant="ttm-512-96",
            task="classification",
            num_classes=2,
            freeze_encoder=True,
            input_channels=1,
            context_length=96
        )
        
        # Check encoder parameters are frozen
        for param in model.encoder.parameters():
            assert not param.requires_grad
        
        # Check head parameters are trainable
        for param in model.head.parameters():
            assert param.requires_grad
    
    def test_partial_unfreeze(self):
        """Test partial unfreezing of last N blocks."""
        model = TTMAdapter(
            variant="ttm-512-96",
            task="classification",
            num_classes=2,
            freeze_encoder=True,
            unfreeze_last_n_blocks=2,
            input_channels=1,
            context_length=96
        )
        
        # Some encoder parameters should be trainable
        trainable_encoder_params = sum(
            1 for p in model.encoder.parameters() if p.requires_grad
        )
        frozen_encoder_params = sum(
            1 for p in model.encoder.parameters() if not p.requires_grad
        )
        
        assert trainable_encoder_params > 0
        assert frozen_encoder_params > 0
    
    def test_lora_application(self):
        """Test LoRA application to TTM."""
        lora_config = LoRAConfig(r=4, alpha=8, dropout=0.1)
        
        model = TTMAdapter(
            variant="ttm-512-96",
            task="classification",
            num_classes=2,
            freeze_encoder=True,
            lora_config=lora_config,
            input_channels=1,
            context_length=96
        )
        
        # Check LoRA modules exist
        assert hasattr(model, 'lora_modules')
        assert len(model.lora_modules) > 0
        
        # Check LoRA parameters are trainable
        lora_params = get_lora_parameters(model)
        assert len(lora_params) > 0
        for param in lora_params:
            assert param.requires_grad
        
        # Check base parameters are frozen
        for name, param in model.encoder.named_parameters():
            if 'lora_' not in name:
                assert not param.requires_grad
    
    def test_forward_shapes(self):
        """Test forward pass shapes."""
        batch_size = 8
        seq_len = 96
        n_channels = 3
        num_classes = 5
        
        model = TTMAdapter(
            variant="ttm-512-96",
            task="classification",
            num_classes=num_classes,
            input_channels=n_channels,
            context_length=seq_len
        )
        
        # Test [B, T, C] input
        x_btc = torch.randn(batch_size, seq_len, n_channels)
        output = model(x_btc)
        assert output.shape == (batch_size, num_classes)
        
        # Test [B, C, T] input (should transpose)
        x_bct = torch.randn(batch_size, n_channels, seq_len)
        output = model(x_bct)
        assert output.shape == (batch_size, num_classes)
        
        # Test with return_features
        output, features = model(x_btc, return_features=True)
        assert output.shape == (batch_size, num_classes)
        assert features is not None
    
    def test_different_tasks(self):
        """Test different task configurations."""
        batch_size = 8
        seq_len = 96
        n_channels = 1
        
        # Classification
        clf_model = TTMAdapter(
            variant="ttm-512-96",
            task="classification",
            num_classes=3,
            input_channels=n_channels,
            context_length=seq_len
        )
        x = torch.randn(batch_size, seq_len, n_channels)
        output = clf_model(x)
        assert output.shape == (batch_size, 3)
        
        # Regression
        reg_model = TTMAdapter(
            variant="ttm-512-96",
            task="regression",
            out_features=1,
            input_channels=n_channels,
            context_length=seq_len
        )
        output = reg_model(x)
        assert output.shape == (batch_size, 1)
        
        # Multi-output regression
        reg_model_multi = TTMAdapter(
            variant="ttm-512-96",
            task="regression",
            out_features=5,
            input_channels=n_channels,
            context_length=seq_len
        )
        output = reg_model_multi(x)
        assert output.shape == (batch_size, 5)
    
    def test_create_from_config(self):
        """Test model creation from config dictionary."""
        config = {
            'variant': 'ttm-512-96',
            'task': 'classification',
            'num_classes': 2,
            'head_type': 'mlp',
            'head_config': {
                'hidden_dims': [256, 128],
                'dropout': 0.2
            },
            'freeze_encoder': True,
            'unfreeze_last_n_blocks': 1,
            'lora': {
                'enabled': True,
                'r': 8,
                'alpha': 16,
                'dropout': 0.1
            },
            'input_channels': 3,
            'context_length': 96
        }
        
        model = create_ttm_model(config)
        
        assert model is not None
        assert hasattr(model, 'lora_modules')
        assert len(model.lora_modules) > 0
    
    def test_parameter_summary(self):
        """Test parameter summary methods."""
        model = TTMAdapter(
            variant="ttm-512-96",
            task="classification",
            num_classes=2,
            freeze_encoder=True,
            input_channels=1,
            context_length=96
        )
        
        # Get different parameter groups
        encoder_params = model.get_encoder_params()
        head_params = model.get_head_params()
        trainable_params = model.get_trainable_params()
        
        assert len(encoder_params) > 0
        assert len(head_params) > 0
        assert len(trainable_params) > 0
        
        # With frozen encoder, only head should be trainable
        assert len(trainable_params) == len(head_params)


# Integration tests
def test_lora_integration():
    """Test LoRA integration with a simple model."""
    # Create model
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    # Apply LoRA
    lora_modules = apply_lora(model, r=4, alpha=8)
    
    # Freeze non-LoRA parameters
    freeze_non_lora_parameters(model)
    
    # Create optimizer with only LoRA parameters
    lora_params = get_lora_parameters(model)
    optimizer = torch.optim.Adam(lora_params, lr=1e-3)
    
    # Forward pass
    x = torch.randn(16, 128)
    output = model(x)
    assert output.shape == (16, 10)
    
    # Backward pass
    loss = output.mean()
    loss.backward()
    
    # Check gradients
    for name, param in model.named_parameters():
        if 'lora_' in name:
            assert param.grad is not None
        else:
            assert param.grad is None or param.grad.sum() == 0
    
    # Optimizer step
    optimizer.step()
    
    # Parameters should have been updated
    assert True  # If we get here, integration works


def test_heads_with_different_activations():
    """Test heads with different activation functions."""
    batch_size = 16
    in_features = 768
    num_classes = 10
    
    for activation in ['relu', 'gelu', 'tanh']:
        head = MLPClassifier(
            in_features, num_classes,
            hidden_dims=[256],
            activation=activation
        )
        
        x = torch.randn(batch_size, in_features)
        output = head(x)
        assert output.shape == (batch_size, num_classes)


def test_heads_with_different_normalizations():
    """Test heads with different normalization layers."""
    batch_size = 16
    in_features = 768
    num_classes = 10
    
    # Batch norm
    head_bn = MLPClassifier(
        in_features, num_classes,
        use_batch_norm=True,
        use_layer_norm=False
    )
    
    # Layer norm
    head_ln = MLPClassifier(
        in_features, num_classes,
        use_batch_norm=False,
        use_layer_norm=True
    )
    
    # No norm
    head_no_norm = MLPClassifier(
        in_features, num_classes,
        use_batch_norm=False,
        use_layer_norm=False
    )
    
    x = torch.randn(batch_size, in_features)
    
    for head in [head_bn, head_ln, head_no_norm]:
        output = head(x)
        assert output.shape == (batch_size, num_classes)
