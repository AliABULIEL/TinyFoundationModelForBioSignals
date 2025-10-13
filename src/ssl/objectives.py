"""Loss objectives for self-supervised biosignal learning.

Implements MAE-style masked signal modeling and multi-resolution spectral losses
for 1D biosignal foundation models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MaskedSignalModeling(nn.Module):
    """Masked Signal Modeling (MSM) loss for MAE-style pretraining.
    
    Computes Mean Squared Error (MSE) only on masked patches, similar to
    MAE (He et al., 2022) but for 1D biosignals.
    
    The loss is normalized by the number of masked elements to ensure
    consistent scaling regardless of mask ratio.
    
    Example:
        >>> msm_loss = MaskedSignalModeling(patch_size=125)
        >>> pred = torch.randn(16, 2, 1250)  # Reconstructed signal
        >>> target = torch.randn(16, 2, 1250)  # Original signal
        >>> mask = torch.rand(16, 10) > 0.6  # [B, P] boolean mask (40% masked)
        >>> loss = msm_loss(pred, target, mask)
        >>> loss.shape
        torch.Size([])  # Scalar loss
    """
    
    def __init__(self, patch_size: int = 125):
        """Initialize MSM loss.
        
        Args:
            patch_size: Size of each patch in samples (default: 125 = 1s @ 125Hz)
        """
        super().__init__()
        self.patch_size = patch_size
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE loss only on masked patches.
        
        Args:
            pred: Predicted/reconstructed signal [B, C, T]
            target: Original target signal [B, C, T]
            mask: Boolean mask [B, P] where True = masked patch to compute loss on
                  P = T // patch_size
        
        Returns:
            loss: Scalar MSE loss averaged over masked elements
        
        Raises:
            ValueError: If shapes are incompatible or mask is all False
        """
        B, C, T = pred.shape
        assert target.shape == pred.shape, f"Shape mismatch: pred {pred.shape}, target {target.shape}"
        
        P = T // self.patch_size
        assert mask.shape == (B, P), f"Mask shape {mask.shape} incompatible with patches ({B}, {P})"
        
        # Check if there are any masked patches
        if not mask.any():
            raise ValueError("Mask is all False - no patches to compute loss on")
        
        # Compute squared error
        squared_error = (pred - target) ** 2  # [B, C, T]
        
        # Create patch-level mask [B, C, T]
        patch_mask_expanded = torch.zeros_like(squared_error, dtype=torch.bool)
        for p in range(P):
            patch_start = p * self.patch_size
            patch_end = (p + 1) * self.patch_size
            # Broadcast mask[B, P] to [B, C, T] for this patch
            patch_mask_expanded[:, :, patch_start:patch_end] = mask[:, p].unsqueeze(1).unsqueeze(2)
        
        # Compute loss only on masked regions
        masked_squared_error = squared_error[patch_mask_expanded]
        
        # Average over all masked elements
        loss = masked_squared_error.mean()
        
        return loss


class MultiResolutionSTFT(nn.Module):
    """Multi-Resolution STFT loss for spectral reconstruction.
    
    Computes spectral loss at multiple resolutions using STFT, combining:
    1. L1 loss on log-magnitude spectrograms
    2. Optional spectral convergence loss
    
    This encourages the model to preserve spectral characteristics across
    different frequency ranges, improving reconstruction quality.
    
    Reference:
        - Yamamoto et al. (2020) "Parallel WaveGAN"
        - bioFAME (ICLR 2024) for biosignal applications
    
    Example:
        >>> stft_loss = MultiResolutionSTFT(
        ...     n_ffts=[512, 1024, 2048],
        ...     hop_lengths=[128, 256, 512],
        ...     weight=0.3
        ... )
        >>> pred = torch.randn(16, 2, 1250)  # Predicted signal
        >>> target = torch.randn(16, 2, 1250)  # Target signal
        >>> loss = stft_loss(pred, target)
        >>> loss.shape
        torch.Size([])  # Scalar loss
    """
    
    def __init__(
        self,
        n_ffts: list = [512, 1024, 2048],
        hop_lengths: list = [128, 256, 512],
        win_lengths: list = None,
        weight: float = 1.0,
        use_spectral_convergence: bool = False,
    ):
        """Initialize Multi-Resolution STFT loss.
        
        Args:
            n_ffts: List of FFT sizes for different resolutions
            hop_lengths: List of hop lengths corresponding to each FFT size
            win_lengths: List of window lengths (defaults to n_ffts if None)
            weight: Weight for this loss component (default: 1.0)
            use_spectral_convergence: Whether to include spectral convergence loss
        """
        super().__init__()
        assert len(n_ffts) == len(hop_lengths), "n_ffts and hop_lengths must have same length"
        
        self.n_ffts = n_ffts
        self.hop_lengths = hop_lengths
        self.win_lengths = win_lengths if win_lengths is not None else n_ffts
        self.weight = weight
        self.use_spectral_convergence = use_spectral_convergence
        
        assert len(self.win_lengths) == len(n_ffts), "win_lengths must match n_ffts length"
    
    def _compute_stft(
        self,
        x: torch.Tensor,
        n_fft: int,
        hop_length: int,
        win_length: int,
    ) -> torch.Tensor:
        """Compute STFT magnitude spectrogram.
        
        Args:
            x: Input signal [B, C, T]
            n_fft: FFT size
            hop_length: Hop length
            win_length: Window length
        
        Returns:
            Magnitude spectrogram [B, C, F, T'] where F = n_fft//2 + 1
        """
        B, C, T = x.shape
        
        # Reshape to [B*C, T] for processing
        x_flat = x.reshape(B * C, T)
        
        # Create window
        window = torch.hann_window(win_length, device=x.device)
        
        # Compute STFT
        stft = torch.stft(
            x_flat,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
            center=True,
            normalized=False,
        )
        
        # Compute magnitude
        mag = torch.abs(stft)  # [B*C, F, T']
        
        # Reshape back to [B, C, F, T']
        F, T_prime = mag.shape[1], mag.shape[2]
        mag = mag.reshape(B, C, F, T_prime)
        
        return mag
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute multi-resolution STFT loss.
        
        Args:
            pred: Predicted signal [B, C, T]
            target: Target signal [B, C, T]
        
        Returns:
            loss: Scalar loss averaged across resolutions and channels
        
        Raises:
            ValueError: If shapes are incompatible
        """
        assert pred.shape == target.shape, f"Shape mismatch: pred {pred.shape}, target {target.shape}"
        
        total_loss = 0.0
        num_resolutions = len(self.n_ffts)
        
        for n_fft, hop_length, win_length in zip(self.n_ffts, self.hop_lengths, self.win_lengths):
            # Compute STFT magnitudes
            pred_mag = self._compute_stft(pred, n_fft, hop_length, win_length)
            target_mag = self._compute_stft(target, n_fft, hop_length, win_length)
            
            # L1 loss on log-magnitude (add small epsilon for numerical stability)
            epsilon = 1e-8
            log_pred = torch.log(pred_mag + epsilon)
            log_target = torch.log(target_mag + epsilon)
            
            mag_loss = F.l1_loss(log_pred, log_target)
            
            loss_at_resolution = mag_loss
            
            # Optional: Add spectral convergence loss
            if self.use_spectral_convergence:
                # Frobenius norm of difference / Frobenius norm of target
                convergence = torch.norm(pred_mag - target_mag, p='fro') / (torch.norm(target_mag, p='fro') + epsilon)
                loss_at_resolution = loss_at_resolution + convergence
            
            total_loss += loss_at_resolution
        
        # Average across resolutions
        avg_loss = total_loss / num_resolutions
        
        # Apply weight
        weighted_loss = self.weight * avg_loss
        
        return weighted_loss
