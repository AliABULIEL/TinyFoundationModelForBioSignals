"""SSL Pretrainer for masked autoencoder biosignal pretraining.

Implements training loop with:
- Masked signal modeling (MSM)
- Multi-resolution STFT loss
- Automatic mixed precision (AMP)
- Gradient clipping
- Best model checkpointing
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import time
from typing import Dict, Optional, Callable, Tuple, Any
import warnings

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .masking import random_masking, block_masking
from .objectives import MaskedSignalModeling, MultiResolutionSTFT


class SSLTrainer:
    """SSL trainer for masked autoencoder pretraining on biosignals.
    
    Implements training loop with:
    - Input masking (random or block)
    - Encoder → latent representations
    - Decoder → signal reconstruction
    - Loss: MSM + optional STFT
    - AMP for faster training
    - Gradient clipping for stability
    - Best model checkpointing
    
    Args:
        encoder: Encoder model (e.g., TTM)
        decoder: Reconstruction decoder
        optimizer: Optimizer (e.g., AdamW)
        msm_criterion: Masked signal modeling loss
        stft_criterion: Optional multi-resolution STFT loss
        mask_fn: Masking function (random_masking or block_masking)
        device: Device to train on
        use_amp: Use automatic mixed precision
        gradient_clip: Max gradient norm for clipping
        stft_weight: Weight for STFT loss (default: 0.3)
    
    Example:
        >>> from src.models.ttm_adapter import TTMAdapter
        >>> from src.models.decoders import ReconstructionHead1D
        >>> from src.ssl.objectives import MaskedSignalModeling, MultiResolutionSTFT
        >>> 
        >>> encoder = TTMAdapter(...)
        >>> decoder = ReconstructionHead1D(d_model=192, patch_size=125, n_channels=2)
        >>> optimizer = torch.optim.AdamW(
        ...     list(encoder.parameters()) + list(decoder.parameters()),
        ...     lr=1e-4
        ... )
        >>> 
        >>> trainer = SSLTrainer(
        ...     encoder=encoder,
        ...     decoder=decoder,
        ...     optimizer=optimizer,
        ...     msm_criterion=MaskedSignalModeling(patch_size=125),
        ...     stft_criterion=MultiResolutionSTFT(weight=1.0),
        ...     use_amp=True,
        ...     gradient_clip=1.0,
        ...     stft_weight=0.3
        ... )
        >>> 
        >>> history = trainer.fit(
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     num_epochs=100,
        ...     save_dir='artifacts/ssl_pretrain',
        ...     mask_ratio=0.4
        ... )
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        optimizer: torch.optim.Optimizer,
        msm_criterion: MaskedSignalModeling,
        stft_criterion: Optional[MultiResolutionSTFT] = None,
        mask_fn: Callable = random_masking,
        device: str = 'cuda',
        use_amp: bool = True,
        gradient_clip: float = 1.0,
        stft_weight: float = 0.3
    ):
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.optimizer = optimizer
        self.msm_criterion = msm_criterion.to(device)
        self.stft_criterion = stft_criterion.to(device) if stft_criterion is not None else None
        self.mask_fn = mask_fn
        self.device = device
        self.use_amp = use_amp and torch.cuda.is_available()
        self.gradient_clip = gradient_clip
        self.stft_weight = stft_weight
        
        # AMP scaler
        self.scaler = GradScaler() if self.use_amp else None
        
        # Track best model
        self.best_val_loss = float('inf')
        
        print("SSLTrainer initialized:")
        print(f"  Device: {device}")
        print(f"  AMP: {self.use_amp}")
        print(f"  Gradient clip: {gradient_clip}")
        print(f"  STFT weight: {stft_weight if stft_criterion else 'N/A (MSM only)'}")
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        save_dir: str = 'artifacts/foundation_model',
        mask_ratio: float = 0.4,
        log_interval: int = 50,
        save_interval: int = 10
    ) -> Dict[str, list]:
        """Train the SSL model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            mask_ratio: Masking ratio (default: 0.4)
            log_interval: Log every N batches
            save_interval: Save checkpoint every N epochs
        
        Returns:
            history: Dictionary with training history
                {'train_loss': [...], 'val_loss': [...],
                 'train_msm': [...], 'train_stft': [...],
                 'val_msm': [...], 'val_stft': [...]}
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_msm': [],
            'train_stft': [],
            'val_msm': [],
            'val_stft': []
        }
        
        print(f"\nStarting SSL pretraining for {num_epochs} epochs")
        print(f"Mask ratio: {mask_ratio}")
        print("=" * 70)
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Train
            train_metrics = self._train_epoch(
                train_loader,
                mask_ratio=mask_ratio,
                log_interval=log_interval,
                epoch=epoch
            )
            
            # Validate
            val_metrics = self._validate_epoch(
                val_loader,
                mask_ratio=mask_ratio
            )
            
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['train_msm'].append(train_metrics['msm_loss'])
            history['train_stft'].append(train_metrics.get('stft_loss', 0.0))
            history['val_loss'].append(val_metrics['loss'])
            history['val_msm'].append(val_metrics['msm_loss'])
            history['val_stft'].append(val_metrics.get('stft_loss', 0.0))
            
            epoch_time = time.time() - epoch_start
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"MSM: {train_metrics['msm_loss']:.4f}, "
                  f"STFT: {train_metrics.get('stft_loss', 0.0):.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"MSM: {val_metrics['msm_loss']:.4f}, "
                  f"STFT: {val_metrics.get('stft_loss', 0.0):.4f}")
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self._save_checkpoint(
                    save_dir / 'best_model.pt',
                    epoch=epoch,
                    metrics=val_metrics
                )
                print(f"  ✓ Best model saved (val_loss: {val_metrics['loss']:.4f})")
            
            # Periodic checkpoint
            if (epoch + 1) % save_interval == 0:
                self._save_checkpoint(
                    save_dir / f'checkpoint_epoch_{epoch+1}.pt',
                    epoch=epoch,
                    metrics=val_metrics
                )
        
        # Save final model
        self._save_checkpoint(
            save_dir / 'last_model.pt',
            epoch=num_epochs - 1,
            metrics=val_metrics
        )
        
        # Save training history
        with open(save_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        print("\n" + "=" * 70)
        print(f"Training complete! Best val loss: {self.best_val_loss:.4f}")
        print(f"Checkpoints saved to: {save_dir}")
        
        return history
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        mask_ratio: float,
        log_interval: int,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            mask_ratio: Masking ratio
            log_interval: Log every N batches
            epoch: Current epoch number
        
        Returns:
            metrics: Dictionary with epoch metrics
        """
        self.encoder.train()
        self.decoder.train()
        
        total_loss = 0.0
        total_msm = 0.0
        total_stft = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        
        # Check if dataloader is empty
        if len(train_loader) == 0:
            raise ValueError(
                f"Training dataloader is empty! "
                f"Dataset size: {len(train_loader.dataset)}, "
                f"Batch size: {train_loader.batch_size}, "
                f"Drop last: {train_loader.drop_last}. "
                f"Either reduce batch_size or set drop_last=False."
            )
        
        for batch_idx, batch in enumerate(pbar):
            # Unpack batch - handle single tensors, tuples from TensorDataset, and paired inputs
            if isinstance(batch, (tuple, list)):
                inputs = batch[0]  # Use first element
            else:
                inputs = batch
            
            # Move to device
            inputs = inputs.to(self.device)
            B, C, T = inputs.shape
            
            # Forward pass with AMP
            with autocast(enabled=self.use_amp):
                # Apply masking
                masked_inputs, mask_bool = self.mask_fn(
                    inputs,
                    mask_ratio=mask_ratio,
                    patch_size=self.decoder.patch_size
                )
                
                # Encode: [B, C, T] -> [B, P, D]
                # Use get_encoder_output for SSL to preserve patch dimensions
                latents = self.encoder.get_encoder_output(masked_inputs)
                
                # CRITICAL: Sync all components with encoder after first call
                # TTM may output different patch count than config expects
                patch_size_changed = False
                if hasattr(self.encoder, 'patch_size') and self.decoder.patch_size != self.encoder.patch_size:
                    patch_size_changed = True
                    
                    # Use update_patch_size to recreate projection layer
                    self.decoder.update_patch_size(self.encoder.patch_size)
                    
                    # ALSO update MSM criterion patch_size
                    if hasattr(self, 'msm_criterion') and self.msm_criterion.patch_size != self.encoder.patch_size:
                        print(f"[INFO] Syncing MSM criterion patch_size from {self.msm_criterion.patch_size} to {self.encoder.patch_size}")
                        self.msm_criterion.patch_size = self.encoder.patch_size
                    
                    # CRITICAL: Recreate mask with new patch_size!
                    print(f"[INFO] Recreating mask with updated patch_size={self.encoder.patch_size}")
                    masked_inputs, mask_bool = self.mask_fn(
                        inputs,
                        mask_ratio=mask_ratio,
                        patch_size=self.encoder.patch_size
                    )
                    
                    # Re-encode with new mask
                    print(f"[INFO] Re-encoding with updated mask")
                    latents = self.encoder.get_encoder_output(masked_inputs)
                
                # Handle different encoder output formats
                if isinstance(latents, tuple):
                    latents = latents[0]  # Use first output if tuple
                
                # Ensure latents are [B, P, D]
                if latents.ndim == 2:
                    # [B, D] -> [B, 1, D]
                    latents = latents.unsqueeze(1)
                
                # Decode: [B, P, D] -> [B, C, T]
                reconstructed = self.decoder(latents)
                
                # Compute MSM loss
                msm_loss = self.msm_criterion(reconstructed, inputs, mask_bool)
                
                # Compute total loss
                loss = msm_loss
                
                # Add STFT loss if available
                stft_loss = 0.0
                if self.stft_criterion is not None:
                    stft_loss = self.stft_criterion(reconstructed, inputs)
                    loss = loss + self.stft_weight * stft_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters()) + list(self.decoder.parameters()),
                        self.gradient_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                # Gradient clipping
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters()) + list(self.decoder.parameters()),
                        self.gradient_clip
                    )
                
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_msm += msm_loss.item()
            if isinstance(stft_loss, torch.Tensor):
                total_stft += stft_loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / num_batches,
                'msm': total_msm / num_batches,
                'stft': total_stft / num_batches if self.stft_criterion else 0.0
            })
        
        # Return epoch metrics
        metrics = {
            'loss': total_loss / num_batches,
            'msm_loss': total_msm / num_batches,
        }
        
        if self.stft_criterion is not None:
            metrics['stft_loss'] = total_stft / num_batches
        
        return metrics
    
    def _validate_epoch(
        self,
        val_loader: DataLoader,
        mask_ratio: float
    ) -> Dict[str, float]:
        """Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            mask_ratio: Masking ratio
        
        Returns:
            metrics: Dictionary with validation metrics
        """
        self.encoder.eval()
        self.decoder.eval()
        
        total_loss = 0.0
        total_msm = 0.0
        total_stft = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Unpack batch - handle single tensors, tuples from TensorDataset, and paired inputs
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0]  # Use first element
                else:
                    inputs = batch
                
                inputs = inputs.to(self.device)
                
                # Forward pass with AMP
                with autocast(enabled=self.use_amp):
                    # Apply masking
                    masked_inputs, mask_bool = self.mask_fn(
                        inputs,
                        mask_ratio=mask_ratio,
                        patch_size=self.decoder.patch_size
                    )
                    
                    # Encode: [B, C, T] -> [B, P, D]
                    # Use get_encoder_output for SSL to preserve patch dimensions
                    latents = self.encoder.get_encoder_output(masked_inputs)
                    
                    # Sync all components if needed (should already be synced from training)
                    if hasattr(self.encoder, 'patch_size') and self.decoder.patch_size != self.encoder.patch_size:
                        self.decoder.update_patch_size(self.encoder.patch_size)
                        
                        # ALSO sync MSM criterion
                        if hasattr(self, 'msm_criterion') and self.msm_criterion.patch_size != self.encoder.patch_size:
                            self.msm_criterion.patch_size = self.encoder.patch_size
                        
                        # Recreate mask with correct patch_size
                        masked_inputs, mask_bool = self.mask_fn(
                            inputs,
                            mask_ratio=mask_ratio,
                            patch_size=self.encoder.patch_size
                        )
                        
                        # Re-encode with new mask
                        latents = self.encoder.get_encoder_output(masked_inputs)
                    
                    if isinstance(latents, tuple):
                        latents = latents[0]
                    
                    if latents.ndim == 2:
                        latents = latents.unsqueeze(1)
                    
                    # Decode
                    reconstructed = self.decoder(latents)
                    
                    # Compute losses
                    msm_loss = self.msm_criterion(reconstructed, inputs, mask_bool)
                    loss = msm_loss
                    
                    stft_loss = 0.0
                    if self.stft_criterion is not None:
                        stft_loss = self.stft_criterion(reconstructed, inputs)
                        loss = loss + self.stft_weight * stft_loss
                
                # Update metrics
                total_loss += loss.item()
                total_msm += msm_loss.item()
                if isinstance(stft_loss, torch.Tensor):
                    total_stft += stft_loss.item()
                num_batches += 1
        
        # Return validation metrics
        metrics = {
            'loss': total_loss / num_batches,
            'msm_loss': total_msm / num_batches,
        }
        
        if self.stft_criterion is not None:
            metrics['stft_loss'] = total_stft / num_batches
        
        return metrics
    
    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: Dict[str, float]
    ):
        """Save checkpoint.
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            metrics: Validation metrics
        """
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics': metrics,
            'config': {
                'gradient_clip': self.gradient_clip,
                'stft_weight': self.stft_weight,
                'use_amp': self.use_amp
            }
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Loaded checkpoint from: {path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Best val loss: {self.best_val_loss:.4f}")


if __name__ == "__main__":
    """Quick sanity check."""
    print("Testing SSLTrainer...")
    print("=" * 70)
    
    # Create mock encoder and decoder
    class MockEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(1250, 192)
        
        def forward(self, x):
            # [B, C, T] -> [B, P, D]
            B, C, T = x.shape
            x = x.reshape(B, -1)  # Flatten
            x = self.proj(x[:, :1250])  # Project
            return x.unsqueeze(1).expand(B, 10, 192)  # Mock P=10
    
    from src.models.decoders import ReconstructionHead1D
    from src.ssl.objectives import MaskedSignalModeling, MultiResolutionSTFT
    
    encoder = MockEncoder()
    decoder = ReconstructionHead1D(d_model=192, patch_size=125, n_channels=2)
    
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=1e-4
    )
    
    trainer = SSLTrainer(
        encoder=encoder,
        decoder=decoder,
        optimizer=optimizer,
        msm_criterion=MaskedSignalModeling(patch_size=125),
        stft_criterion=MultiResolutionSTFT(weight=1.0),
        device='cpu',
        use_amp=False,
        gradient_clip=1.0,
        stft_weight=0.3
    )
    
    print("\n✓ SSLTrainer initialized successfully!")
    print("=" * 70)
