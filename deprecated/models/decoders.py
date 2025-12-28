"""Decoder heads for SSL reconstruction tasks.

Lightweight decoders for masked autoencoder (MAE) style pretraining on biosignals.
Following MAE design principle: asymmetric encoder-decoder where decoder is minimal.
"""

import torch
import torch.nn as nn
from typing import Tuple


class ReconstructionHead1D(nn.Module):
    """Lightweight reconstruction head for SSL masked signal reconstruction.
    
    Takes patch-level latent representations from TTM encoder and projects
    them back to the original signal space for MAE-style pretraining.
    
    Architecture:
    - Single linear layer: D -> (C * patch_size)
    - Reshape and fold patches to recover [B, C, T] signal
    
    No convolutions or complex upsampling - keeps decoder lightweight
    following MAE principle (asymmetric encoder-decoder with simple decoder).
    
    References:
        - He et al. (2022) "Masked Autoencoders Are Scalable Vision Learners"
        - bioFAME (ICLR 2024) for biosignal adaptation
    
    Example:
        >>> # Create decoder for 2-channel biosignals
        >>> decoder = ReconstructionHead1D(d_model=192, patch_size=125, n_channels=2)
        >>> 
        >>> # Encoder output: [B=16, P=10, D=192]
        >>> latents = torch.randn(16, 10, 192)
        >>> 
        >>> # Reconstruct signal
        >>> reconstructed = decoder(latents)
        >>> reconstructed.shape
        torch.Size([16, 2, 1250])  # [B, C, T] where T = P * patch_size
        >>> 
        >>> # Verify shapes
        >>> B, P, D = latents.shape
        >>> B_out, C_out, T_out = reconstructed.shape
        >>> assert B_out == B
        >>> assert C_out == decoder.n_channels
        >>> assert T_out == P * decoder.patch_size
    """
    
    def __init__(
        self,
        d_model: int = 192,
        patch_size: int = 125,
        n_channels: int = 2
    ):
        """Initialize reconstruction head.
        
        Args:
            d_model: Latent dimension from encoder (default: 192 for TTM-512)
            patch_size: Size of each patch in samples (default: 125 = 1s @ 125Hz)
            n_channels: Number of signal channels to reconstruct (default: 2)
        """
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.n_channels = n_channels
        
        # Output dimension: need to reconstruct C channels × patch_size samples per patch
        out_features = n_channels * patch_size
        
        # Linear projection: D -> (C * patch_size)
        self.proj = nn.Linear(d_model, out_features)
        
        # Track parameters for inspection
        self._num_params = sum(p.numel() for p in self.parameters())
    
    def update_patch_size(self, new_patch_size: int):
        """Update patch size and recreate projection layer.
        
        This is needed when TTM's actual patch configuration differs from config.
        
        Args:
            new_patch_size: New patch size to use
        """
        if new_patch_size == self.patch_size:
            return  # No change needed
        
        print(f"[INFO] Decoder: Updating patch_size from {self.patch_size} to {new_patch_size}")
        print(f"[INFO] Decoder: Recreating projection layer...")
        
        old_out_features = self.n_channels * self.patch_size
        new_out_features = self.n_channels * new_patch_size
        
        print(f"[INFO] Decoder: proj out_features {old_out_features} → {new_out_features}")
        
        # Update patch_size
        self.patch_size = new_patch_size
        
        # Recreate projection layer with new output size
        # Get device from existing projection layer
        device = next(self.proj.parameters()).device
        self.proj = nn.Linear(self.d_model, new_out_features).to(device)
        
        print(f"[INFO] Decoder: Projection layer recreated successfully")
    
    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Reconstruct signal from latent patch representations.
        
        Args:
            latents: Patch-level features [B, P, D] from encoder where:
                B = batch size
                P = number of patches
                D = latent dimension (d_model)
        
        Returns:
            reconstructed: Reconstructed signal [B, C, T] where:
                C = n_channels
                T = P * patch_size (total time steps)
        
        Raises:
            ValueError: If input shape is incompatible
        
        Example:
            >>> decoder = ReconstructionHead1D(d_model=192, patch_size=125, n_channels=2)
            >>> latents = torch.randn(16, 10, 192)  # 16 samples, 10 patches, 192-d
            >>> reconstructed = decoder(latents)
            >>> reconstructed.shape
            torch.Size([16, 2, 1250])
        """
        if latents.ndim != 3:
            raise ValueError(f"Expected 3D input [B, P, D], got shape {latents.shape}")
        
        B, P, D = latents.shape
        
        if D != self.d_model:
            raise ValueError(
                f"Input latent dimension {D} doesn't match expected d_model={self.d_model}"
            )
        
        # Project to signal space: [B, P, D] -> [B, P, C*patch_size]
        x = self.proj(latents)  # [B, P, n_channels * patch_size]
        
        # Reshape to separate channels and patch dimension
        # [B, P, C*patch_size] -> [B, P, C, patch_size]
        x = x.reshape(B, P, self.n_channels, self.patch_size)
        
        # Fold patches into time dimension
        # Permute to [B, C, P, patch_size] then reshape to [B, C, T]
        x = x.permute(0, 2, 1, 3)  # [B, C, P, patch_size]
        
        T = P * self.patch_size
        x = x.reshape(B, self.n_channels, T)  # [B, C, T]
        
        return x
    
    def get_output_shape(self, num_patches: int) -> Tuple[int, int]:
        """Get output shape for given number of patches.
        
        Args:
            num_patches: Number of patches (P)
        
        Returns:
            (n_channels, time_steps): Output shape as (C, T)
        
        Example:
            >>> decoder = ReconstructionHead1D(patch_size=125, n_channels=2)
            >>> decoder.get_output_shape(num_patches=10)
            (2, 1250)
        """
        T = num_patches * self.patch_size
        return (self.n_channels, T)
    
    def print_summary(self):
        """Print decoder summary."""
        print("ReconstructionHead1D Summary")
        print("=" * 50)
        print(f"  Latent dimension (d_model):  {self.d_model}")
        print(f"  Patch size:                  {self.patch_size}")
        print(f"  Output channels:             {self.n_channels}")
        print(f"  Parameters:                  {self._num_params:,}")
        print(f"  Output per patch:            {self.n_channels * self.patch_size}")
        print("=" * 50)
        
        # Example I/O
        example_patches = 10
        C, T = self.get_output_shape(example_patches)
        print(f"\nExample: {example_patches} patches -> {C} channels × {T} samples")
    
    def extra_repr(self) -> str:
        """Extra representation for print()."""
        return (
            f"d_model={self.d_model}, patch_size={self.patch_size}, "
            f"n_channels={self.n_channels}, params={self._num_params:,}"
        )


def create_reconstruction_head(
    encoder_dim: int,
    patch_size: int,
    n_channels: int
) -> ReconstructionHead1D:
    """Factory function to create reconstruction head.
    
    Args:
        encoder_dim: Encoder output dimension (d_model)
        patch_size: Patch size in samples
        n_channels: Number of channels to reconstruct
    
    Returns:
        Initialized ReconstructionHead1D
    
    Example:
        >>> head = create_reconstruction_head(
        ...     encoder_dim=192,
        ...     patch_size=125,
        ...     n_channels=2
        ... )
        >>> head.print_summary()
    """
    return ReconstructionHead1D(
        d_model=encoder_dim,
        patch_size=patch_size,
        n_channels=n_channels
    )


if __name__ == "__main__":
    """Quick sanity check of reconstruction head."""
    print("Testing ReconstructionHead1D...")
    print("=" * 70)
    
    # Create decoder
    decoder = ReconstructionHead1D(
        d_model=192,
        patch_size=125,
        n_channels=2
    )
    
    decoder.print_summary()
    
    # Test forward pass
    print("\nTesting forward pass...")
    B, P, D = 16, 10, 192
    latents = torch.randn(B, P, D)
    print(f"Input shape:  {list(latents.shape)} [B, P, D]")
    
    reconstructed = decoder(latents)
    print(f"Output shape: {list(reconstructed.shape)} [B, C, T]")
    
    # Verify dimensions
    assert reconstructed.shape == (B, 2, 1250), f"Unexpected shape: {reconstructed.shape}"
    
    print("\n✓ All checks passed!")
    print("=" * 70)
