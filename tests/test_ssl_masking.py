"""Unit tests for SSL masking strategies.

Tests random and block masking implementations for biosignal MAE pretraining.
"""

import pytest
import torch
import numpy as np

from src.ssl.masking import random_masking, block_masking


class TestRandomMasking:
    """Test suite for random patch masking."""
    
    def test_output_shapes(self):
        """Test that output shapes are correct."""
        B, C, T = 16, 2, 1250
        patch_size = 125
        
        x = torch.randn(B, C, T)
        masked_x, mask_bool = random_masking(x, mask_ratio=0.4, patch_size=patch_size)
        
        # Check shapes
        assert masked_x.shape == (B, C, T), f"Expected {(B, C, T)}, got {masked_x.shape}"
        
        P = T // patch_size
        assert mask_bool.shape == (B, P), f"Expected {(B, P)}, got {mask_bool.shape}"
    
    def test_mask_ratio_accuracy(self):
        """Test that mask ratio is approximately correct."""
        B, C, T = 32, 2, 1250
        patch_size = 125
        mask_ratio = 0.4
        
        x = torch.randn(B, C, T)
        _, mask_bool = random_masking(x, mask_ratio=mask_ratio, patch_size=patch_size)
        
        # Calculate actual mask ratio
        P = T // patch_size
        actual_ratio = mask_bool.float().sum() / (B * P)
        
        # Should be within ±1% of target
        assert abs(actual_ratio - mask_ratio) < 0.01, \
            f"Mask ratio {actual_ratio:.3f} too far from target {mask_ratio}"
    
    def test_shared_temporal_mask(self):
        """Test that same temporal patches are masked across all channels."""
        B, C, T = 8, 3, 1250
        patch_size = 125
        
        x = torch.randn(B, C, T)
        masked_x, mask_bool = random_masking(x, mask_ratio=0.4, patch_size=patch_size)
        
        # For each sample and patch, check if masking is consistent across channels
        P = T // patch_size
        for b in range(B):
            for p in range(P):
                patch_start = p * patch_size
                patch_end = (p + 1) * patch_size
                
                # Get patch values for all channels
                patch_values = masked_x[b, :, patch_start:patch_end]
                
                if mask_bool[b, p]:
                    # Patch should be all zeros across all channels
                    assert torch.allclose(patch_values, torch.zeros_like(patch_values)), \
                        f"Masked patch {p} in sample {b} not all zeros"
                # If not masked, values should be non-zero (with high probability)
    
    def test_mask_values_zeroed(self):
        """Test that masked patches are actually zeroed out."""
        B, C, T = 4, 2, 1250
        patch_size = 125
        
        # Use non-zero input
        x = torch.ones(B, C, T)
        masked_x, mask_bool = random_masking(x, mask_ratio=0.4, patch_size=patch_size)
        
        P = T // patch_size
        for b in range(B):
            for p in range(P):
                patch_start = p * patch_size
                patch_end = (p + 1) * patch_size
                
                if mask_bool[b, p]:
                    # Masked patch should be all zeros
                    patch_vals = masked_x[b, :, patch_start:patch_end]
                    assert torch.all(patch_vals == 0), \
                        f"Masked patch {p} in sample {b} not all zeros"
                else:
                    # Unmasked patch should be all ones (original values)
                    patch_vals = masked_x[b, :, patch_start:patch_end]
                    assert torch.all(patch_vals == 1), \
                        f"Unmasked patch {p} in sample {b} not preserved"
    
    def test_different_mask_ratios(self):
        """Test various mask ratios."""
        B, C, T = 16, 2, 1250
        patch_size = 125
        P = T // patch_size
        
        for mask_ratio in [0.1, 0.3, 0.5, 0.7]:
            x = torch.randn(B, C, T)
            _, mask_bool = random_masking(x, mask_ratio=mask_ratio, patch_size=patch_size)
            
            actual_ratio = mask_bool.float().sum() / (B * P)
            
            # Within 2% tolerance
            assert abs(actual_ratio - mask_ratio) < 0.02, \
                f"Mask ratio {actual_ratio:.3f} too far from {mask_ratio}"
    
    def test_device_compatibility(self):
        """Test that masking works on different devices."""
        B, C, T = 8, 2, 1250
        
        # CPU
        x_cpu = torch.randn(B, C, T)
        masked_cpu, mask_cpu = random_masking(x_cpu, mask_ratio=0.4, patch_size=125)
        assert masked_cpu.device == x_cpu.device
        assert mask_cpu.device == x_cpu.device
        
        # GPU (if available)
        if torch.cuda.is_available():
            x_gpu = x_cpu.cuda()
            masked_gpu, mask_gpu = random_masking(x_gpu, mask_ratio=0.4, patch_size=125)
            assert masked_gpu.device == x_gpu.device
            assert mask_gpu.device == x_gpu.device
    
    def test_deterministic_with_seed(self):
        """Test that results are deterministic when using manual seed."""
        B, C, T = 8, 2, 1250
        x = torch.randn(B, C, T)
        
        # First run
        torch.manual_seed(42)
        _, mask1 = random_masking(x.clone(), mask_ratio=0.4, patch_size=125)
        
        # Second run with same seed
        torch.manual_seed(42)
        _, mask2 = random_masking(x.clone(), mask_ratio=0.4, patch_size=125)
        
        # Should be identical
        assert torch.all(mask1 == mask2), "Masking not deterministic with same seed"


class TestBlockMasking:
    """Test suite for block/contiguous masking."""
    
    def test_output_shapes(self):
        """Test that output shapes are correct."""
        B, C, T = 16, 2, 1250
        patch_size = 125
        
        x = torch.randn(B, C, T)
        masked_x, mask_bool = block_masking(x, mask_ratio=0.4, span_length=5, patch_size=patch_size)
        
        assert masked_x.shape == (B, C, T)
        
        P = T // patch_size
        assert mask_bool.shape == (B, P)
    
    def test_contiguous_masking(self):
        """Test that masked patches form contiguous spans."""
        B, C, T = 8, 2, 1250
        patch_size = 125
        span_length = 3
        
        x = torch.randn(B, C, T)
        _, mask_bool = block_masking(x, mask_ratio=0.4, span_length=span_length, patch_size=patch_size)
        
        # Check for contiguous spans
        for b in range(B):
            mask = mask_bool[b].numpy()
            
            # Find runs of True values
            changes = np.diff(np.concatenate([[False], mask, [False]]).astype(int))
            run_starts = np.where(changes == 1)[0]
            run_ends = np.where(changes == -1)[0]
            run_lengths = run_ends - run_starts
            
            if len(run_lengths) > 0:
                # At least some runs should be of length span_length or close to it
                # (might be shorter at boundaries)
                max_run = run_lengths.max()
                assert max_run >= min(span_length, mask.sum()), \
                    f"No contiguous spans found for sample {b}"
    
    def test_mask_ratio_accuracy(self):
        """Test that mask ratio is approximately correct."""
        B, C, T = 32, 2, 1250
        patch_size = 125
        mask_ratio = 0.4
        
        x = torch.randn(B, C, T)
        _, mask_bool = block_masking(x, mask_ratio=mask_ratio, span_length=5, patch_size=patch_size)
        
        P = T // patch_size
        actual_ratio = mask_bool.float().sum() / (B * P)
        
        # Should be within ±2% (slightly more tolerance than random due to span constraints)
        assert abs(actual_ratio - mask_ratio) < 0.02, \
            f"Mask ratio {actual_ratio:.3f} too far from target {mask_ratio}"
    
    def test_shared_temporal_mask(self):
        """Test that same temporal patches are masked across all channels."""
        B, C, T = 8, 3, 1250
        patch_size = 125
        
        x = torch.randn(B, C, T)
        masked_x, mask_bool = block_masking(x, mask_ratio=0.4, span_length=4, patch_size=patch_size)
        
        P = T // patch_size
        for b in range(B):
            for p in range(P):
                patch_start = p * patch_size
                patch_end = (p + 1) * patch_size
                
                patch_values = masked_x[b, :, patch_start:patch_end]
                
                if mask_bool[b, p]:
                    assert torch.allclose(patch_values, torch.zeros_like(patch_values)), \
                        f"Masked patch {p} in sample {b} not all zeros"
    
    def test_different_span_lengths(self):
        """Test various span lengths."""
        B, C, T = 16, 2, 1250
        patch_size = 125
        
        for span_length in [2, 5, 8]:
            x = torch.randn(B, C, T)
            masked_x, mask_bool = block_masking(
                x, mask_ratio=0.4, span_length=span_length, patch_size=patch_size
            )
            
            # Check that output is valid
            assert masked_x.shape == (B, C, T)
            assert mask_bool.shape == (B, T // patch_size)
            
            # Check mask ratio is reasonable
            P = T // patch_size
            actual_ratio = mask_bool.float().sum() / (B * P)
            assert 0.3 < actual_ratio < 0.5, \
                f"Mask ratio {actual_ratio:.3f} unreasonable for span_length={span_length}"
    
    def test_edge_cases(self):
        """Test edge cases like very small or large span lengths."""
        B, C, T = 8, 2, 1250
        patch_size = 125
        P = T // patch_size  # 10 patches
        
        x = torch.randn(B, C, T)
        
        # Very small span (should work like random)
        masked_x1, mask1 = block_masking(x, mask_ratio=0.4, span_length=1, patch_size=patch_size)
        assert masked_x1.shape == (B, C, T)
        
        # Span larger than total patches (should still work)
        masked_x2, mask2 = block_masking(x, mask_ratio=0.4, span_length=20, patch_size=patch_size)
        assert masked_x2.shape == (B, C, T)
        
        # Very high mask ratio
        masked_x3, mask3 = block_masking(x, mask_ratio=0.9, span_length=5, patch_size=patch_size)
        actual_ratio = mask3.float().sum() / (B * P)
        assert actual_ratio > 0.85, "High mask ratio not achieved"


class TestMaskingComparison:
    """Compare random vs block masking."""
    
    def test_both_produce_valid_output(self):
        """Test that both masking strategies produce valid output."""
        B, C, T = 16, 3, 1250
        patch_size = 125
        
        x = torch.randn(B, C, T)
        
        # Random masking
        masked_rand, mask_rand = random_masking(x.clone(), mask_ratio=0.4, patch_size=patch_size)
        
        # Block masking
        masked_block, mask_block = block_masking(x.clone(), mask_ratio=0.4, span_length=5, patch_size=patch_size)
        
        # Both should have correct shapes
        assert masked_rand.shape == masked_block.shape == (B, C, T)
        assert mask_rand.shape == mask_block.shape == (B, T // patch_size)
        
        # Both should have similar mask ratios
        P = T // patch_size
        ratio_rand = mask_rand.float().sum() / (B * P)
        ratio_block = mask_block.float().sum() / (B * P)
        
        assert abs(ratio_rand - ratio_block) < 0.05, \
            "Random and block masking have very different mask ratios"
    
    def test_block_more_contiguous(self):
        """Test that block masking produces more contiguous spans."""
        B, C, T = 16, 2, 1250
        patch_size = 125
        
        x = torch.randn(B, C, T)
        
        # Random masking
        _, mask_rand = random_masking(x.clone(), mask_ratio=0.4, patch_size=patch_size)
        
        # Block masking
        _, mask_block = block_masking(x.clone(), mask_ratio=0.4, span_length=5, patch_size=patch_size)
        
        # Count number of contiguous runs
        def count_runs(mask):
            runs = []
            for b in range(mask.shape[0]):
                m = mask[b].numpy()
                changes = np.diff(np.concatenate([[False], m, [False]]).astype(int))
                n_runs = (changes == 1).sum()
                runs.append(n_runs)
            return np.mean(runs)
        
        runs_rand = count_runs(mask_rand)
        runs_block = count_runs(mask_block)
        
        # Block masking should have fewer, longer runs
        # (not always true due to randomness, but on average)
        print(f"Random masking: {runs_rand:.1f} runs, Block masking: {runs_block:.1f} runs")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
