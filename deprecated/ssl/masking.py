"""Masking strategies for self-supervised biosignal learning.

Implements patch-based masking for MAE-style pretraining on 1D biosignals.
All masking operates on a shared temporal mask across channels.
"""

import torch
import torch.nn as nn
from typing import Tuple


def random_masking(
    x: torch.Tensor,
    mask_ratio: float = 0.4,
    patch_size: int = 125,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random patch masking with shared temporal mask across channels.
    
    Divides the time dimension into patches and randomly masks a fraction of them.
    The same temporal patches are masked across all channels.
    
    Args:
        x: Input tensor of shape [B, C, T] where:
            B = batch size
            C = number of channels
            T = time dimension (e.g., 1250 for 10s @ 125Hz)
        mask_ratio: Fraction of patches to mask (default: 0.4)
        patch_size: Size of each patch in samples (default: 125 = 1s @ 125Hz)
    
    Returns:
        masked_x: Tensor [B, C, T] with masked patches zeroed out
        mask_bool: Boolean mask [B, P] where True = masked patch
                  P = T // patch_size (number of patches)
    
    Example:
        >>> x = torch.randn(16, 2, 1250)  # 16 samples, 2 channels, 1250 timesteps
        >>> masked_x, mask = random_masking(x, mask_ratio=0.4, patch_size=125)
        >>> masked_x.shape
        torch.Size([16, 2, 1250])
        >>> mask.shape
        torch.Size([16, 10])  # 10 patches
        >>> mask.sum(dim=1).float().mean() / 10  # Should be ~0.4
        tensor(0.4000)
    """
    B, C, T = x.shape
    
    # Calculate number of patches
    P = T // patch_size
    assert T % patch_size == 0, f"T={T} must be divisible by patch_size={patch_size}"
    
    # Number of patches to mask per sample
    num_masked = round(P * mask_ratio)
    
    # Create mask: [B, P]
    mask_bool = torch.zeros(B, P, dtype=torch.bool, device=x.device)
    
    # Randomly select patches to mask for each sample
    for i in range(B):
        masked_indices = torch.randperm(P, device=x.device)[:num_masked]
        mask_bool[i, masked_indices] = True
    
    # Create masked version of x
    masked_x = x.clone()
    
    # Zero out masked patches (shared across channels)
    for p in range(P):
        patch_start = p * patch_size
        patch_end = (p + 1) * patch_size
        # Expand mask to [B, 1] for broadcasting across channels
        patch_mask = mask_bool[:, p].unsqueeze(1)  # [B, 1]
        masked_x[:, :, patch_start:patch_end] = torch.where(
            patch_mask.unsqueeze(2),  # [B, 1, 1]
            torch.zeros_like(masked_x[:, :, patch_start:patch_end]),
            masked_x[:, :, patch_start:patch_end]
        )
    
    return masked_x, mask_bool


def block_masking(
    x: torch.Tensor,
    mask_ratio: float = 0.4,
    span_length: int = 5,
    patch_size: int = 125,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Block/contiguous patch masking with shared temporal mask across channels.
    
    Masks contiguous spans of patches rather than random individual patches.
    This is more challenging for the model and closer to real-world corruptions.
    
    Args:
        x: Input tensor of shape [B, C, T] where:
            B = batch size
            C = number of channels
            T = time dimension (e.g., 1250 for 10s @ 125Hz)
        mask_ratio: Fraction of patches to mask (default: 0.4)
        span_length: Length of contiguous masked spans in patches (default: 5)
        patch_size: Size of each patch in samples (default: 125 = 1s @ 125Hz)
    
    Returns:
        masked_x: Tensor [B, C, T] with masked patches zeroed out
        mask_bool: Boolean mask [B, P] where True = masked patch
                  P = T // patch_size (number of patches)
    
    Example:
        >>> x = torch.randn(16, 2, 1250)  # 16 samples, 2 channels, 1250 timesteps
        >>> masked_x, mask = block_masking(x, mask_ratio=0.4, span_length=5, patch_size=125)
        >>> masked_x.shape
        torch.Size([16, 2, 1250])
        >>> mask.shape
        torch.Size([16, 10])  # 10 patches
        >>> # Verify contiguous masking pattern
        >>> mask[0]  # Example: tensor([False, False, True, True, True, True, True, False, False, False])
    """
    B, C, T = x.shape
    
    # Calculate number of patches
    P = T // patch_size
    assert T % patch_size == 0, f"T={T} must be divisible by patch_size={patch_size}"
    
    # Number of patches to mask per sample
    num_masked = round(P * mask_ratio)
    
    # Create mask: [B, P]
    mask_bool = torch.zeros(B, P, dtype=torch.bool, device=x.device)
    
    # Create contiguous masked spans for each sample
    for i in range(B):
        masked_count = 0
        attempts = 0
        max_attempts = P * 10  # Prevent infinite loops
        
        while masked_count < num_masked and attempts < max_attempts:
            # Randomly select starting position
            start_pos = torch.randint(0, P, (1,), device=x.device).item()
            
            # Calculate actual span length (don't exceed remaining needed masks)
            actual_span = min(span_length, num_masked - masked_count, P - start_pos)
            
            # Mask the span
            mask_bool[i, start_pos:start_pos + actual_span] = True
            masked_count = mask_bool[i].sum().item()
            attempts += 1
        
        # If we couldn't reach exact ratio with spans, fill remaining randomly
        if masked_count < num_masked:
            unmasked_indices = (~mask_bool[i]).nonzero(as_tuple=True)[0]
            if len(unmasked_indices) > 0:
                additional = min(num_masked - masked_count, len(unmasked_indices))
                random_indices = unmasked_indices[torch.randperm(len(unmasked_indices), device=x.device)[:additional]]
                mask_bool[i, random_indices] = True
    
    # Create masked version of x
    masked_x = x.clone()
    
    # Zero out masked patches (shared across channels)
    for p in range(P):
        patch_start = p * patch_size
        patch_end = (p + 1) * patch_size
        # Expand mask to [B, 1] for broadcasting across channels
        patch_mask = mask_bool[:, p].unsqueeze(1)  # [B, 1]
        masked_x[:, :, patch_start:patch_end] = torch.where(
            patch_mask.unsqueeze(2),  # [B, 1, 1]
            torch.zeros_like(masked_x[:, :, patch_start:patch_end]),
            masked_x[:, :, patch_start:patch_end]
        )
    
    return masked_x, mask_bool
