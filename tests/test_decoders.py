"""Unit tests for reconstruction decoders.

Tests ReconstructionHead1D for SSL masked autoencoder pretraining.
"""

import pytest
import torch
import torch.nn as nn

from src.models.decoders import ReconstructionHead1D, create_reconstruction_head


class TestReconstructionHead1D:
    """Test suite for ReconstructionHead1D."""
    
    def test_initialization(self):
        """Test that decoder initializes correctly."""
        decoder = ReconstructionHead1D(
            d_model=192,
            patch_size=125,
            n_channels=2
        )
        
        assert decoder.d_model == 192
        assert decoder.patch_size == 125
        assert decoder.n_channels == 2
        assert isinstance(decoder, nn.Module)
    
    def test_output_shape(self):
        """Test that output shape is correct."""
        decoder = ReconstructionHead1D(d_model=192, patch_size=125, n_channels=2)
        
        B, P, D = 16, 10, 192
        latents = torch.randn(B, P, D)
        
        reconstructed = decoder(latents)
        
        expected_shape = (B, 2, 1250)  # T = P * patch_size = 10 * 125
        assert reconstructed.shape == expected_shape, \
            f"Expected {expected_shape}, got {reconstructed.shape}"
    
    def test_different_patch_counts(self):
        """Test with different numbers of patches."""
        decoder = ReconstructionHead1D(d_model=192, patch_size=125, n_channels=2)
        
        B, D = 8, 192
        
        for P in [5, 10, 20, 40]:
            latents = torch.randn(B, P, D)
            reconstructed = decoder(latents)
            
            expected_T = P * 125
            assert reconstructed.shape == (B, 2, expected_T), \
                f"For P={P}, expected T={expected_T}, got shape {reconstructed.shape}"
    
    def test_different_channels(self):
        """Test with different numbers of output channels."""
        B, P, D = 8, 10, 192
        
        for n_channels in [1, 2, 3, 5]:
            decoder = ReconstructionHead1D(
                d_model=192,
                patch_size=125,
                n_channels=n_channels
            )
            
            latents = torch.randn(B, P, D)
            reconstructed = decoder(latents)
            
            assert reconstructed.shape == (B, n_channels, 1250), \
                f"For {n_channels} channels, got shape {reconstructed.shape}"
    
    def test_different_patch_sizes(self):
        """Test with different patch sizes."""
        B, P, D = 8, 10, 192
        
        for patch_size in [50, 100, 125, 250]:
            decoder = ReconstructionHead1D(
                d_model=192,
                patch_size=patch_size,
                n_channels=2
            )
            
            latents = torch.randn(B, P, D)
            reconstructed = decoder(latents)
            
            expected_T = P * patch_size
            assert reconstructed.shape == (B, 2, expected_T), \
                f"For patch_size={patch_size}, got shape {reconstructed.shape}"
    
    def test_different_latent_dims(self):
        """Test with different latent dimensions."""
        B, P = 8, 10
        
        for d_model in [64, 128, 192, 256, 512]:
            decoder = ReconstructionHead1D(
                d_model=d_model,
                patch_size=125,
                n_channels=2
            )
            
            latents = torch.randn(B, P, d_model)
            reconstructed = decoder(latents)
            
            assert reconstructed.shape == (B, 2, 1250), \
                f"For d_model={d_model}, got shape {reconstructed.shape}"
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly."""
        decoder = ReconstructionHead1D(d_model=192, patch_size=125, n_channels=2)
        
        B, P, D = 8, 10, 192
        latents = torch.randn(B, P, D, requires_grad=True)
        
        reconstructed = decoder(latents)
        
        # Compute dummy loss
        loss = reconstructed.sum()
        loss.backward()
        
        assert latents.grad is not None, "Gradients should flow to latents"
        assert not torch.all(latents.grad == 0), "Gradients should be non-zero"
    
    def test_batch_size_one(self):
        """Test with batch size of 1 (edge case)."""
        decoder = ReconstructionHead1D(d_model=192, patch_size=125, n_channels=2)
        
        latents = torch.randn(1, 10, 192)
        reconstructed = decoder(latents)
        
        assert reconstructed.shape == (1, 2, 1250)
    
    def test_single_patch(self):
        """Test with single patch (edge case)."""
        decoder = ReconstructionHead1D(d_model=192, patch_size=125, n_channels=2)
        
        latents = torch.randn(8, 1, 192)
        reconstructed = decoder(latents)
        
        assert reconstructed.shape == (8, 2, 125)
    
    def test_raises_on_wrong_input_dim(self):
        """Test that it raises error on wrong input dimensions."""
        decoder = ReconstructionHead1D(d_model=192, patch_size=125, n_channels=2)
        
        # 2D input (missing patch dimension)
        with pytest.raises(ValueError, match="Expected 3D input"):
            decoder(torch.randn(8, 192))
        
        # 4D input (extra dimension)
        with pytest.raises(ValueError, match="Expected 3D input"):
            decoder(torch.randn(8, 10, 192, 1))
    
    def test_raises_on_wrong_latent_dim(self):
        """Test that it raises error when latent dim doesn't match."""
        decoder = ReconstructionHead1D(d_model=192, patch_size=125, n_channels=2)
        
        # Wrong D dimension
        latents = torch.randn(8, 10, 256)  # Expected 192, got 256
        
        with pytest.raises(ValueError, match="doesn't match expected d_model"):
            decoder(latents)
    
    def test_get_output_shape_helper(self):
        """Test get_output_shape helper method."""
        decoder = ReconstructionHead1D(d_model=192, patch_size=125, n_channels=2)
        
        # Test various patch counts
        assert decoder.get_output_shape(10) == (2, 1250)
        assert decoder.get_output_shape(5) == (2, 625)
        assert decoder.get_output_shape(20) == (2, 2500)
    
    def test_device_compatibility(self):
        """Test that decoder works on different devices."""
        decoder = ReconstructionHead1D(d_model=192, patch_size=125, n_channels=2)
        
        # CPU
        latents_cpu = torch.randn(8, 10, 192)
        reconstructed_cpu = decoder(latents_cpu)
        assert reconstructed_cpu.device == latents_cpu.device
        
        # GPU (if available)
        if torch.cuda.is_available():
            decoder_gpu = decoder.cuda()
            latents_gpu = latents_cpu.cuda()
            reconstructed_gpu = decoder_gpu(latents_gpu)
            assert reconstructed_gpu.device == latents_gpu.device
    
    def test_deterministic_output(self):
        """Test that output is deterministic for same input."""
        decoder = ReconstructionHead1D(d_model=192, patch_size=125, n_channels=2)
        
        latents = torch.randn(8, 10, 192)
        
        # Two forward passes with same input
        output1 = decoder(latents.clone())
        output2 = decoder(latents.clone())
        
        assert torch.allclose(output1, output2), \
            "Output should be deterministic for same input"
    
    def test_parameter_count(self):
        """Test that parameter count is reasonable."""
        decoder = ReconstructionHead1D(d_model=192, patch_size=125, n_channels=2)
        
        num_params = sum(p.numel() for p in decoder.parameters())
        
        # Expected: 192 * (2 * 125) + (2 * 125) for linear layer
        expected = 192 * (2 * 125) + (2 * 125)  # weights + bias
        
        assert num_params == expected, \
            f"Expected {expected} parameters, got {num_params}"
        
        # Verify it's lightweight (< 100K params)
        assert num_params < 100_000, \
            f"Decoder should be lightweight, got {num_params} params"


class TestCreateReconstructionHead:
    """Test factory function."""
    
    def test_factory_function(self):
        """Test that factory function creates correct decoder."""
        decoder = create_reconstruction_head(
            encoder_dim=192,
            patch_size=125,
            n_channels=2
        )
        
        assert isinstance(decoder, ReconstructionHead1D)
        assert decoder.d_model == 192
        assert decoder.patch_size == 125
        assert decoder.n_channels == 2
    
    def test_factory_with_different_params(self):
        """Test factory with various parameters."""
        decoder = create_reconstruction_head(
            encoder_dim=256,
            patch_size=100,
            n_channels=3
        )
        
        assert decoder.d_model == 256
        assert decoder.patch_size == 100
        assert decoder.n_channels == 3
        
        # Test forward pass
        latents = torch.randn(4, 5, 256)
        output = decoder(latents)
        assert output.shape == (4, 3, 500)  # 5 patches * 100 = 500


class TestReconstructionHeadIntegration:
    """Integration tests with masking and SSL objectives."""
    
    def test_with_masked_signal(self):
        """Test decoder with masked input (realistic SSL scenario)."""
        from src.ssl.masking import random_masking
        
        decoder = ReconstructionHead1D(d_model=192, patch_size=125, n_channels=2)
        
        # Original signal
        B, C, T = 16, 2, 1250
        original = torch.randn(B, C, T)
        
        # Apply masking
        masked_x, mask_bool = random_masking(original, mask_ratio=0.4, patch_size=125)
        
        # Simulate encoder output (for testing, use random latents)
        P = T // 125
        latents = torch.randn(B, P, 192)
        
        # Reconstruct
        reconstructed = decoder(latents)
        
        # Verify shape matches original
        assert reconstructed.shape == original.shape, \
            f"Reconstructed shape {reconstructed.shape} != original {original.shape}"
    
    def test_with_msm_loss(self):
        """Test decoder output can be used with MSM loss."""
        from src.ssl.objectives import MaskedSignalModeling
        from src.ssl.masking import random_masking
        
        decoder = ReconstructionHead1D(d_model=192, patch_size=125, n_channels=2)
        msm_loss = MaskedSignalModeling(patch_size=125)
        
        # Original signal
        B, C, T = 16, 2, 1250
        original = torch.randn(B, C, T)
        
        # Mask
        _, mask_bool = random_masking(original, mask_ratio=0.4, patch_size=125)
        
        # Simulate reconstruction
        P = 10
        latents = torch.randn(B, P, 192, requires_grad=True)
        reconstructed = decoder(latents)
        
        # Compute loss
        loss = msm_loss(reconstructed, original, mask_bool)
        
        # Verify loss is valid
        assert torch.isfinite(loss), "Loss should be finite"
        assert loss >= 0, "Loss should be non-negative"
        
        # Verify gradients flow
        loss.backward()
        assert latents.grad is not None, "Gradients should flow to encoder output"
    
    def test_encoder_decoder_compatibility(self):
        """Test that decoder is compatible with typical encoder output shapes."""
        # Common TTM encoder output dimensions
        encoder_configs = [
            (192, 125, 2),   # TTM-512
            (256, 125, 2),   # TTM-1024
            (512, 125, 3),   # Larger model, more channels
        ]
        
        for d_model, patch_size, n_channels in encoder_configs:
            decoder = ReconstructionHead1D(
                d_model=d_model,
                patch_size=patch_size,
                n_channels=n_channels
            )
            
            # Simulate encoder output
            B, P = 8, 10
            latents = torch.randn(B, P, d_model)
            
            # Decode
            reconstructed = decoder(latents)
            
            # Verify
            expected_T = P * patch_size
            assert reconstructed.shape == (B, n_channels, expected_T), \
                f"Config {encoder_configs}: got shape {reconstructed.shape}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
