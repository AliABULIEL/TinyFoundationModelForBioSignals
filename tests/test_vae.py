"""
Test suite for VAE biosignal model.
Tests VAE architecture, training, and integration with pipeline.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.vae_adapter import (
    BiosignalVAEEncoder,
    BiosignalVAEDecoder,
    BiosignalVAE,
    VAEAdapter
)
from src.models.ttm_adapter import create_ttm_model


class TestVAEComponents:
    """Test individual VAE components."""
    
    def test_encoder_shape(self):
        """Test encoder output shapes."""
        encoder = BiosignalVAEEncoder(input_channels=1, input_length=1250, latent_dim=64)
        x = torch.randn(8, 1, 1250)  # batch=8, channels=1, length=1250
        
        mu, logvar = encoder(x)
        
        assert mu.shape == (8, 64), f"Expected mu shape (8, 64), got {mu.shape}"
        assert logvar.shape == (8, 64), f"Expected logvar shape (8, 64), got {logvar.shape}"
    
    def test_decoder_shape(self):
        """Test decoder output shapes."""
        decoder = BiosignalVAEDecoder(latent_dim=64, output_channels=1, output_length=1250)
        z = torch.randn(8, 64)  # batch=8, latent_dim=64
        
        x_recon = decoder(z)
        
        assert x_recon.shape == (8, 1, 1250), f"Expected shape (8, 1, 1250), got {x_recon.shape}"
    
    def test_vae_forward(self):
        """Test complete VAE forward pass."""
        vae = BiosignalVAE(input_channels=1, input_length=1250, latent_dim=64, beta=1.0)
        x = torch.randn(8, 1, 1250)
        
        x_recon, mu, logvar = vae(x)
        
        assert x_recon.shape == x.shape, f"Reconstruction shape mismatch"
        assert mu.shape == (8, 64), f"Expected mu shape (8, 64), got {mu.shape}"
        assert logvar.shape == (8, 64), f"Expected logvar shape (8, 64), got {logvar.shape}"
    
    def test_vae_loss(self):
        """Test VAE loss computation."""
        vae = BiosignalVAE(input_channels=1, input_length=1250, latent_dim=64, beta=1.0)
        x = torch.randn(8, 1, 1250)
        
        x_recon, mu, logvar = vae(x)
        total_loss, recon_loss, kl_loss = vae.loss_function(x, x_recon, mu, logvar)
        
        assert total_loss > 0, "Total loss should be positive"
        assert recon_loss >= 0, "Reconstruction loss should be non-negative"
        assert kl_loss >= 0, "KL loss should be non-negative"
        assert torch.isfinite(total_loss), "Loss should be finite"
    
    def test_reparameterization(self):
        """Test reparameterization trick."""
        vae = BiosignalVAE()
        mu = torch.zeros(8, 64)
        logvar = torch.zeros(8, 64)
        
        z = vae.reparameterize(mu, logvar)
        
        assert z.shape == (8, 64), f"Expected z shape (8, 64), got {z.shape}"
        # With zero mean and unit variance, samples should be roughly in [-3, 3]
        assert torch.abs(z).max() < 5, "Samples should be reasonable"
    
    def test_generation(self):
        """Test sample generation from prior."""
        vae = BiosignalVAE(latent_dim=64)
        device = torch.device('cpu')
        
        samples = vae.generate(num_samples=4, device=device)
        
        assert samples.shape == (4, 1, 1250), f"Expected shape (4, 1, 1250), got {samples.shape}"


class TestVAEAdapter:
    """Test VAE adapter for pipeline integration."""
    
    def test_adapter_classification(self):
        """Test VAE adapter for classification task."""
        adapter = VAEAdapter(
            task='classification',
            num_classes=2,
            input_channels=1,
            context_length=1250,
            latent_dim=64,
            head_type='linear'
        )
        
        x = torch.randn(8, 1250, 1)  # Pipeline format: [batch, length, channels]
        output = adapter(x)
        
        assert output.shape == (8, 2), f"Expected shape (8, 2), got {output.shape}"
    
    def test_adapter_regression(self):
        """Test VAE adapter for regression task."""
        adapter = VAEAdapter(
            task='regression',
            input_channels=1,
            context_length=1250,
            latent_dim=64,
            head_type='linear'
        )
        
        x = torch.randn(8, 1250, 1)
        output = adapter(x)
        
        assert output.shape == (8, 1), f"Expected shape (8, 1), got {output.shape}"
    
    def test_adapter_with_vae_outputs(self):
        """Test adapter returning VAE outputs."""
        adapter = VAEAdapter(
            task='classification',
            num_classes=2,
            input_channels=1,
            context_length=1250
        )
        
        x = torch.randn(8, 1250, 1)
        output, vae_outputs = adapter(x, return_vae_outputs=True)
        
        assert output.shape == (8, 2), "Classification output shape mismatch"
        assert 'recon' in vae_outputs, "Missing reconstruction in VAE outputs"
        assert 'mu' in vae_outputs, "Missing mu in VAE outputs"
        assert 'logvar' in vae_outputs, "Missing logvar in VAE outputs"
        assert vae_outputs['recon'].shape == (8, 1, 1250), "Reconstruction shape mismatch"
    
    def test_vae_loss_method(self):
        """Test get_vae_loss method."""
        adapter = VAEAdapter(
            task='classification',
            num_classes=2,
            input_channels=1,
            context_length=1250
        )
        
        x = torch.randn(8, 1250, 1)
        total, recon, kl = adapter.get_vae_loss(x)
        
        assert total > 0, "Total loss should be positive"
        assert recon >= 0, "Reconstruction loss should be non-negative"
        assert kl >= 0, "KL loss should be non-negative"
    
    def test_feature_extraction(self):
        """Test latent feature extraction."""
        adapter = VAEAdapter(
            input_channels=1,
            context_length=1250,
            latent_dim=64
        )
        
        x = torch.randn(8, 1250, 1)
        features = adapter.extract_features(x)
        
        assert features.shape == (8, 64), f"Expected features shape (8, 64), got {features.shape}"
    
    def test_frozen_encoder(self):
        """Test encoder freezing functionality."""
        adapter = VAEAdapter(
            task='classification',
            num_classes=2,
            freeze_encoder=True
        )
        
        # Check encoder parameters are frozen
        for param in adapter.vae.encoder.parameters():
            assert not param.requires_grad, "Encoder parameters should be frozen"
        
        # Check head parameters are trainable
        for param in adapter.head.parameters():
            assert param.requires_grad, "Head parameters should be trainable"
    
    def test_mlp_head(self):
        """Test MLP head type."""
        adapter = VAEAdapter(
            task='classification',
            num_classes=2,
            head_type='mlp',
            dropout_rate=0.2
        )
        
        assert isinstance(adapter.head, nn.Sequential), "Head should be Sequential for MLP"
        assert len(adapter.head) > 1, "MLP should have multiple layers"
        
        # Check for dropout layers
        has_dropout = any(isinstance(layer, nn.Dropout) for layer in adapter.head)
        assert has_dropout, "MLP head should contain dropout"


class TestModelFactory:
    """Test model factory function with VAE support."""
    
    def test_create_vae_model(self):
        """Test VAE model creation via factory."""
        config = {
            'model_type': 'vae',
            'task': 'classification',
            'num_classes': 2,
            'input_channels': 1,
            'context_length': 1024,
            'latent_dim': 64,
            'beta': 1.0
        }
        
        model = create_ttm_model(config)
        
        assert isinstance(model, VAEAdapter), "Should create VAEAdapter"
        assert hasattr(model, 'vae'), "Should have VAE attribute"
        assert hasattr(model, 'head'), "Should have task head"
    
    def test_create_ttm_model(self):
        """Test TTM model creation (default)."""
        config = {
            'model_type': 'ttm',
            'task': 'classification',
            'num_classes': 2,
            'input_channels': 1,
            'context_length': 1250
        }
        
        model = create_ttm_model(config)
        
        # Should create TTM model (not VAE)
        assert not isinstance(model, VAEAdapter), "Should not create VAEAdapter"


class TestVAETraining:
    """Test VAE training functionality."""
    
    def test_vae_gradient_flow(self):
        """Test gradient flow through VAE."""
        vae = BiosignalVAE()
        x = torch.randn(4, 1, 1250, requires_grad=True)
        
        x_recon, mu, logvar = vae(x)
        total_loss, _, _ = vae.loss_function(x, x_recon, mu, logvar)
        
        total_loss.backward()
        
        assert x.grad is not None, "Gradients should flow to input"
        assert torch.isfinite(x.grad).all(), "Gradients should be finite"
    
    def test_vae_adapter_training_step(self):
        """Test single training step with VAE adapter."""
        adapter = VAEAdapter(
            task='classification',
            num_classes=2,
            input_channels=1,
            context_length=1250
        )
        
        optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)
        
        # Simulate batch
        x = torch.randn(8, 1250, 1)
        y = torch.randint(0, 2, (8,))
        
        # Forward pass
        output = adapter(x)
        task_loss = nn.CrossEntropyLoss()(output, y)
        
        # VAE loss
        vae_total, _, _ = adapter.get_vae_loss(x)
        
        # Combined loss
        total_loss = task_loss + 0.5 * vae_total
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        assert torch.isfinite(total_loss), "Loss should be finite"
        
        # Check parameters were updated
        for param in adapter.parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), "Gradients should be finite"


def test_different_input_lengths():
    """Test VAE with different input lengths."""
    for length in [625, 1250, 2500]:  # 5s, 10s, 20s at 125Hz
        vae = BiosignalVAE(input_length=length)
        x = torch.randn(4, 1, length)
        
        x_recon, mu, logvar = vae(x)
        
        assert x_recon.shape == x.shape, f"Shape mismatch for length {length}"


def test_different_latent_dims():
    """Test VAE with different latent dimensions."""
    for latent_dim in [32, 64, 128]:
        vae = BiosignalVAE(latent_dim=latent_dim)
        x = torch.randn(4, 1, 1250)
        
        x_recon, mu, logvar = vae(x)
        
        assert mu.shape[1] == latent_dim, f"Latent dim mismatch for {latent_dim}"
        assert logvar.shape[1] == latent_dim, f"Latent dim mismatch for {latent_dim}"


def test_beta_vae():
    """Test beta-VAE with different beta values."""
    for beta in [0.5, 1.0, 2.0]:
        vae = BiosignalVAE(beta=beta)
        x = torch.randn(4, 1, 1250)
        
        x_recon, mu, logvar = vae(x)
        total_loss, recon_loss, kl_loss = vae.loss_function(x, x_recon, mu, logvar)
        
        # Check that beta is applied correctly
        expected_total = recon_loss + beta * kl_loss
        assert torch.isclose(total_loss, expected_total, rtol=1e-5), \
            f"Beta not applied correctly for beta={beta}"


if __name__ == "__main__":
    # Run basic tests
    print("Testing VAE components...")
    test_components = TestVAEComponents()
    test_components.test_encoder_shape()
    test_components.test_decoder_shape()
    test_components.test_vae_forward()
    test_components.test_vae_loss()
    test_components.test_reparameterization()
    test_components.test_generation()
    print("✓ VAE components tests passed")
    
    print("\nTesting VAE adapter...")
    test_adapter = TestVAEAdapter()
    test_adapter.test_adapter_classification()
    test_adapter.test_adapter_regression()
    test_adapter.test_adapter_with_vae_outputs()
    test_adapter.test_vae_loss_method()
    test_adapter.test_feature_extraction()
    test_adapter.test_frozen_encoder()
    test_adapter.test_mlp_head()
    print("✓ VAE adapter tests passed")
    
    print("\nTesting model factory...")
    test_factory = TestModelFactory()
    test_factory.test_create_vae_model()
    print("✓ Model factory tests passed")
    
    print("\nTesting training...")
    test_training = TestVAETraining()
    test_training.test_vae_gradient_flow()
    test_training.test_vae_adapter_training_step()
    print("✓ Training tests passed")
    
    print("\nTesting configurations...")
    test_different_input_lengths()
    test_different_latent_dims()
    test_beta_vae()
    print("✓ Configuration tests passed")
    
    print("\n✅ All tests passed!")
