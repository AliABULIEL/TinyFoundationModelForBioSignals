"""Unit tests for SSL loss objectives.

Tests MaskedSignalModeling and MultiResolutionSTFT loss functions.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pytest
import torch
import torch.nn as nn
import numpy as np

from src.ssl.objectives import MaskedSignalModeling, MultiResolutionSTFT
from src.ssl.masking import random_masking


class TestMaskedSignalModeling:
    """Test suite for Masked Signal Modeling loss."""
    
    def test_initialization(self):
        """Test that MSM loss initializes correctly."""
        msm = MaskedSignalModeling(patch_size=125)
        assert msm.patch_size == 125
        assert isinstance(msm, nn.Module)
    
    def test_output_shape(self):
        """Test that loss returns scalar."""
        msm = MaskedSignalModeling(patch_size=125)
        
        B, C, T = 16, 2, 1250
        pred = torch.randn(B, C, T)
        target = torch.randn(B, C, T)
        mask = torch.rand(B, 10) > 0.6  # 40% masked
        
        loss = msm(pred, target, mask)
        
        assert loss.shape == torch.Size([]), "Loss should be scalar"
        assert loss.ndim == 0, "Loss should be 0-dimensional"
    
    def test_loss_positive(self):
        """Test that loss is always positive."""
        msm = MaskedSignalModeling(patch_size=125)
        
        B, C, T = 16, 2, 1250
        pred = torch.randn(B, C, T)
        target = torch.randn(B, C, T)
        mask = torch.rand(B, 10) > 0.6
        
        loss = msm(pred, target, mask)
        
        assert loss >= 0, "Loss should be non-negative"
    
    def test_loss_is_zero_when_identical(self):
        """Test that loss is zero when pred == target."""
        msm = MaskedSignalModeling(patch_size=125)
        
        B, C, T = 16, 2, 1250
        signal = torch.randn(B, C, T)
        mask = torch.rand(B, 10) > 0.6
        
        loss = msm(signal, signal, mask)
        
        assert loss < 1e-6, f"Loss should be near zero for identical inputs, got {loss}"
    
    def test_computes_only_on_masked_patches(self):
        """Test that loss is computed only on masked regions."""
        msm = MaskedSignalModeling(patch_size=125)
        
        B, C, T = 4, 2, 1250
        P = T // 125
        
        # Create target and prediction
        target = torch.ones(B, C, T)
        
        # Prediction is zeros everywhere
        pred = torch.zeros(B, C, T)
        
        # Mask only first patch (should have loss ~1.0)
        mask = torch.zeros(B, P, dtype=torch.bool)
        mask[:, 0] = True
        
        loss = msm(pred, target, mask)
        
        # Loss should be 1.0 (squared error of 1)
        assert abs(loss - 1.0) < 0.01, f"Expected loss ~1.0, got {loss}"
        
        # Now mask all patches
        mask_all = torch.ones(B, P, dtype=torch.bool)
        loss_all = msm(pred, target, mask_all)
        
        # Should still be ~1.0 (MSE of all patches)
        assert abs(loss_all - 1.0) < 0.01, f"Expected loss ~1.0, got {loss_all}"
    
    def test_different_patch_sizes(self):
        """Test with different patch sizes."""
        for patch_size in [50, 125, 250]:
            T = 1250
            if T % patch_size != 0:
                continue
            
            msm = MaskedSignalModeling(patch_size=patch_size)
            
            B, C = 8, 2
            P = T // patch_size
            
            pred = torch.randn(B, C, T)
            target = torch.randn(B, C, T)
            mask = torch.rand(B, P) > 0.6
            
            loss = msm(pred, target, mask)
            
            assert loss >= 0, f"Loss should be non-negative for patch_size={patch_size}"
            assert torch.isfinite(loss), f"Loss should be finite for patch_size={patch_size}"
    
    def test_raises_on_empty_mask(self):
        """Test that it raises error when mask is all False."""
        msm = MaskedSignalModeling(patch_size=125)
        
        B, C, T = 4, 2, 1250
        pred = torch.randn(B, C, T)
        target = torch.randn(B, C, T)
        
        # All False mask
        mask = torch.zeros(B, 10, dtype=torch.bool)
        
        with pytest.raises(ValueError, match="Mask is all False"):
            msm(pred, target, mask)
    
    def test_raises_on_shape_mismatch(self):
        """Test that it raises error on incompatible shapes."""
        msm = MaskedSignalModeling(patch_size=125)
        
        B, C, T = 4, 2, 1250
        pred = torch.randn(B, C, T)
        target = torch.randn(B, C, T + 1)  # Wrong shape
        mask = torch.rand(B, 10) > 0.5
        
        with pytest.raises(AssertionError):
            msm(pred, target, mask)
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly."""
        msm = MaskedSignalModeling(patch_size=125)
        
        B, C, T = 8, 2, 1250
        pred = torch.randn(B, C, T, requires_grad=True)
        target = torch.randn(B, C, T)
        mask = torch.rand(B, 10) > 0.6
        
        loss = msm(pred, target, mask)
        loss.backward()
        
        assert pred.grad is not None, "Gradients should flow to predictions"
        assert not torch.all(pred.grad == 0), "Gradients should be non-zero"
    
    def test_with_real_masking(self):
        """Test integration with actual masking function."""
        msm = MaskedSignalModeling(patch_size=125)
        
        B, C, T = 16, 2, 1250
        
        # Original signal
        original = torch.randn(B, C, T)
        
        # Apply masking
        masked_x, mask_bool = random_masking(original, mask_ratio=0.4, patch_size=125)
        
        # Predict from masked input (identity for testing)
        pred = masked_x.clone()
        
        # Compute loss
        loss = msm(pred, original, mask_bool)
        
        # Loss should be positive since masked regions are zeroed
        assert loss > 0, "Loss should be positive when predicting from masked input"
        assert torch.isfinite(loss), "Loss should be finite"


class TestMultiResolutionSTFT:
    """Test suite for Multi-Resolution STFT loss."""
    
    def test_initialization(self):
        """Test that MR-STFT loss initializes correctly."""
        stft = MultiResolutionSTFT()
        
        assert stft.n_ffts == [512, 1024, 2048]
        assert stft.hop_lengths == [128, 256, 512]
        assert stft.weight == 1.0
        assert isinstance(stft, nn.Module)
    
    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        stft = MultiResolutionSTFT(
            n_ffts=[256, 512],
            hop_lengths=[64, 128],
            weight=0.5,
            use_spectral_convergence=True
        )
        
        assert stft.n_ffts == [256, 512]
        assert stft.hop_lengths == [64, 128]
        assert stft.weight == 0.5
        assert stft.use_spectral_convergence is True
    
    def test_output_shape(self):
        """Test that loss returns scalar."""
        stft = MultiResolutionSTFT()
        
        B, C, T = 16, 2, 1250
        pred = torch.randn(B, C, T)
        target = torch.randn(B, C, T)
        
        loss = stft(pred, target)
        
        assert loss.shape == torch.Size([]), "Loss should be scalar"
        assert loss.ndim == 0, "Loss should be 0-dimensional"
    
    def test_loss_positive(self):
        """Test that loss is always positive."""
        stft = MultiResolutionSTFT()
        
        B, C, T = 16, 2, 1250
        pred = torch.randn(B, C, T)
        target = torch.randn(B, C, T)
        
        loss = stft(pred, target)
        
        assert loss >= 0, "Loss should be non-negative"
    
    def test_loss_is_zero_when_identical(self):
        """Test that loss is near zero when pred == target."""
        stft = MultiResolutionSTFT()
        
        B, C, T = 8, 2, 1250
        signal = torch.randn(B, C, T)
        
        loss = stft(signal, signal)
        
        # Should be very small (numerical precision issues possible)
        assert loss < 0.01, f"Loss should be near zero for identical inputs, got {loss}"
    
    def test_loss_finite(self):
        """Test that loss is always finite."""
        stft = MultiResolutionSTFT()
        
        B, C, T = 16, 2, 1250
        
        # Test with various inputs
        for _ in range(5):
            pred = torch.randn(B, C, T) * 10  # Large values
            target = torch.randn(B, C, T) * 10
            
            loss = stft(pred, target)
            
            assert torch.isfinite(loss), f"Loss should be finite, got {loss}"
    
    def test_multi_channel_input(self):
        """Test with different numbers of channels."""
        stft = MultiResolutionSTFT()
        
        B, T = 8, 1250
        
        for C in [1, 2, 3, 5]:
            pred = torch.randn(B, C, T)
            target = torch.randn(B, C, T)
            
            loss = stft(pred, target)
            
            assert loss >= 0, f"Loss should be non-negative for {C} channels"
            assert torch.isfinite(loss), f"Loss should be finite for {C} channels"
    
    def test_different_resolutions(self):
        """Test with different STFT resolutions."""
        # Single resolution
        stft1 = MultiResolutionSTFT(n_ffts=[512], hop_lengths=[128])
        
        # Two resolutions
        stft2 = MultiResolutionSTFT(n_ffts=[512, 1024], hop_lengths=[128, 256])
        
        # Three resolutions (default)
        stft3 = MultiResolutionSTFT()
        
        B, C, T = 8, 2, 1250
        pred = torch.randn(B, C, T)
        target = torch.randn(B, C, T)
        
        loss1 = stft1(pred, target)
        loss2 = stft2(pred, target)
        loss3 = stft3(pred, target)
        
        assert all(torch.isfinite(l) for l in [loss1, loss2, loss3]), \
            "All losses should be finite"
        assert all(l >= 0 for l in [loss1, loss2, loss3]), \
            "All losses should be non-negative"
    
    def test_weight_parameter(self):
        """Test that weight parameter scales loss correctly."""
        B, C, T = 8, 2, 1250
        pred = torch.randn(B, C, T)
        target = torch.randn(B, C, T)
        
        # Weight = 1.0
        stft1 = MultiResolutionSTFT(weight=1.0)
        loss1 = stft1(pred, target)
        
        # Weight = 0.5
        stft2 = MultiResolutionSTFT(weight=0.5)
        loss2 = stft2(pred, target)
        
        # loss2 should be approximately half of loss1
        ratio = loss2 / loss1
        assert abs(ratio - 0.5) < 0.01, \
            f"Expected ratio ~0.5, got {ratio:.3f}"
    
    def test_spectral_convergence(self):
        """Test spectral convergence option."""
        B, C, T = 8, 2, 1250
        pred = torch.randn(B, C, T)
        target = torch.randn(B, C, T)
        
        # Without spectral convergence
        stft1 = MultiResolutionSTFT(use_spectral_convergence=False)
        loss1 = stft1(pred, target)
        
        # With spectral convergence
        stft2 = MultiResolutionSTFT(use_spectral_convergence=True)
        loss2 = stft2(pred, target)
        
        # Both should be finite
        assert torch.isfinite(loss1) and torch.isfinite(loss2), \
            "Losses should be finite"
        
        # With spectral convergence should generally be higher
        # (but not always, depends on input)
        assert loss2 >= 0, "Loss with spectral convergence should be non-negative"
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly."""
        stft = MultiResolutionSTFT()
        
        B, C, T = 8, 2, 1250
        pred = torch.randn(B, C, T, requires_grad=True)
        target = torch.randn(B, C, T)
        
        loss = stft(pred, target)
        loss.backward()
        
        assert pred.grad is not None, "Gradients should flow to predictions"
        assert not torch.all(pred.grad == 0), "Gradients should be non-zero"
    
    def test_raises_on_shape_mismatch(self):
        """Test that it raises error on incompatible shapes."""
        stft = MultiResolutionSTFT()
        
        B, C, T = 8, 2, 1250
        pred = torch.randn(B, C, T)
        target = torch.randn(B, C, T + 1)  # Wrong shape
        
        with pytest.raises(AssertionError):
            stft(pred, target)
    
    def test_short_signals(self):
        """Test with shorter signals (edge case)."""
        stft = MultiResolutionSTFT(n_ffts=[256], hop_lengths=[64])
        
        B, C, T = 8, 2, 256  # Short signal
        pred = torch.randn(B, C, T)
        target = torch.randn(B, C, T)
        
        loss = stft(pred, target)
        
        assert torch.isfinite(loss), "Loss should be finite for short signals"
        assert loss >= 0, "Loss should be non-negative for short signals"


class TestCombinedLosses:
    """Test combined MSM + STFT loss (typical SSL objective)."""
    
    def test_combined_loss_computation(self):
        """Test combining MSM and STFT losses."""
        msm = MaskedSignalModeling(patch_size=125)
        stft = MultiResolutionSTFT(weight=0.3)
        
        B, C, T = 16, 2, 1250
        
        # Original and masked signals
        original = torch.randn(B, C, T)
        masked_x, mask_bool = random_masking(original, mask_ratio=0.4, patch_size=125)
        
        # Prediction (for testing, just use masked as is)
        pred = masked_x.clone()
        
        # Compute both losses
        loss_msm = msm(pred, original, mask_bool)
        loss_stft = stft(pred, original)
        
        # Combined loss (typical: 0.7 * MSM + 0.3 * STFT)
        loss_combined = 0.7 * loss_msm + 0.3 * loss_stft
        
        assert torch.isfinite(loss_combined), "Combined loss should be finite"
        assert loss_combined >= 0, "Combined loss should be non-negative"
        
        # Individual losses should contribute
        assert loss_msm > 0, "MSM loss should be positive"
        assert loss_stft >= 0, "STFT loss should be non-negative"
    
    def test_combined_gradient_flow(self):
        """Test that gradients flow through both losses."""
        msm = MaskedSignalModeling(patch_size=125)
        stft = MultiResolutionSTFT(weight=0.3)
        
        B, C, T = 8, 2, 1250
        
        original = torch.randn(B, C, T)
        masked_x, mask_bool = random_masking(original, mask_ratio=0.4, patch_size=125)
        
        # Prediction with gradients
        pred = masked_x.clone().requires_grad_(True)
        
        # Combined loss
        loss = 0.7 * msm(pred, original, mask_bool) + 0.3 * stft(pred, original)
        loss.backward()
        
        assert pred.grad is not None, "Gradients should flow"
        assert not torch.all(pred.grad == 0), "Gradients should be non-zero"
        assert torch.all(torch.isfinite(pred.grad)), "Gradients should be finite"
    
    def test_realistic_ssl_scenario(self):
        """Test realistic SSL pretraining scenario."""
        msm = MaskedSignalModeling(patch_size=125)
        stft = MultiResolutionSTFT(n_ffts=[512, 1024, 2048], hop_lengths=[128, 256, 512], weight=1.0)
        
        B, C, T = 16, 2, 1250
        
        # Simulate multiple training steps
        losses = []
        for _ in range(5):
            # Generate batch
            original = torch.randn(B, C, T)
            
            # Mask input
            masked_x, mask_bool = random_masking(original, mask_ratio=0.4, patch_size=125)
            
            # Predict (identity for testing)
            pred = masked_x.clone()
            
            # Compute loss
            loss_msm = msm(pred, original, mask_bool)
            loss_stft = stft(pred, original)
            loss_total = 0.7 * loss_msm + 0.3 * loss_stft
            
            losses.append(loss_total.item())
            
            assert torch.isfinite(loss_total), "Loss should be finite"
        
        # Check losses are reasonable
        assert all(l > 0 for l in losses), "All losses should be positive"
        assert all(l < 100 for l in losses), "Losses should be in reasonable range"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
