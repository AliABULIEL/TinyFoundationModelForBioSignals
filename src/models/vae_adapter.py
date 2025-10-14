"""
Variational Autoencoder (VAE) adapter for biosignal representation learning.
Fully integrated with the TTM pipeline for VitalDB data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
import numpy as np


class BiosignalVAEEncoder(nn.Module):
    """Convolutional encoder for biosignal VAE."""
    
    def __init__(self, input_channels: int = 1, input_length: int = 1250, latent_dim: int = 64):
        super().__init__()
        
        # Convolutional layers for temporal feature extraction
        self.conv_layers = nn.Sequential(
            # Conv block 1
            nn.Conv1d(input_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout1d(0.1),
            
            # Conv block 2
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout1d(0.1),
            
            # Conv block 3
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout1d(0.1),
            
            # Conv block 4
            nn.Conv1d(128, 256, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        # Calculate flattened size after convolutions
        # Use formula: output_length = (input_length + 2*padding - kernel_size) / stride + 1
        # For each layer with k=7, s=2, p=3: output = (input + 6 - 7) / 2 + 1 = input/2 + 0.5
        # After 4 layers: 1250 -> 625 -> 312 -> 156 -> 78
        conv_out_length = input_length
        for _ in range(4):  # 4 conv layers
            conv_out_length = (conv_out_length + 2*3 - 7) // 2 + 1
        self.flatten_size = 256 * conv_out_length
        
        # Latent projections
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.
        Args:
            x: Input tensor [batch, channels, length]
        Returns:
            mu: Mean of latent distribution [batch, latent_dim]
            logvar: Log variance of latent distribution [batch, latent_dim]
        """
        h = self.conv_layers(x)
        h = h.view(h.size(0), -1)  # Flatten
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class BiosignalVAEDecoder(nn.Module):
    """Convolutional decoder for biosignal VAE."""
    
    def __init__(self, latent_dim: int = 64, output_channels: int = 1, output_length: int = 1250):
        super().__init__()
        
        self.output_length = output_length
        conv_out_length = output_length // 16
        
        # Project from latent
        self.fc = nn.Linear(latent_dim, 256 * conv_out_length)
        self.conv_out_length = conv_out_length
        
        # Transpose convolutional layers
        self.deconv_layers = nn.Sequential(
            # Deconv block 1
            nn.ConvTranspose1d(256, 128, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout1d(0.1),
            
            # Deconv block 2
            nn.ConvTranspose1d(128, 64, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout1d(0.1),
            
            # Deconv block 3
            nn.ConvTranspose1d(64, 32, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout1d(0.1),
            
            # Deconv block 4
            nn.ConvTranspose1d(32, output_channels, kernel_size=7, stride=2, padding=3, output_padding=1),
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent code to signal.
        Args:
            z: Latent code [batch, latent_dim]
        Returns:
            x_recon: Reconstructed signal [batch, channels, length]
        """
        h = self.fc(z)
        h = h.view(h.size(0), 256, self.conv_out_length)
        x_recon = self.deconv_layers(h)
        
        # Ensure correct output size
        if x_recon.size(-1) != self.output_length:
            x_recon = F.interpolate(x_recon, size=self.output_length, mode='linear', align_corners=False)
        
        return x_recon


class BiosignalVAE(nn.Module):
    """Complete VAE model for biosignal representation learning."""
    
    def __init__(self, 
                 input_channels: int = 1,
                 input_length: int = 1250,
                 latent_dim: int = 64,
                 beta: float = 1.0):
        super().__init__()
        
        self.encoder = BiosignalVAEEncoder(input_channels, input_length, latent_dim)
        self.decoder = BiosignalVAEDecoder(latent_dim, output_channels=input_channels, 
                                          output_length=input_length)
        self.beta = beta  # Weight for KL divergence (beta-VAE)
        self.latent_dim = latent_dim
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from latent distribution.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.
        Args:
            x: Input signal [batch, channels, length]
        Returns:
            x_recon: Reconstructed signal
            mu: Latent mean
            logvar: Latent log variance
        """
        # Encode
        mu, logvar = self.encoder(x)
        
        # Sample
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_recon = self.decoder(z)
        
        return x_recon, mu, logvar
    
    def loss_function(self, x: torch.Tensor, x_recon: torch.Tensor,
                     mu: torch.Tensor, logvar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        VAE loss = Reconstruction loss + KL divergence
        """
        # Reconstruction loss (MSE for continuous signals)
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss with beta weighting
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation (mean)."""
        mu, _ = self.encoder(x)
        return mu
    
    def generate(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Generate new samples from prior."""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decoder(z)
        return samples


class VAEAdapter(nn.Module):
    """
    Adapter to make VAE compatible with TTM pipeline.
    Provides same interface as TTMAdapter for seamless integration.
    """
    
    def __init__(self, 
                 task: str = 'classification',
                 num_classes: int = 2,
                 input_channels: int = 1,
                 context_length: int = 1250,
                 latent_dim: int = 64,
                 freeze_encoder: bool = False,
                 head_type: str = 'linear',
                 beta: float = 1.0,
                 dropout_rate: float = 0.2,
                 **kwargs):
        super().__init__()
        
        self.task = task
        self.num_classes = num_classes
        self.mode = 'vae'  # Identifier for model type
        
        # Create VAE
        self.vae = BiosignalVAE(
            input_channels=input_channels,
            input_length=context_length,
            latent_dim=latent_dim,
            beta=beta
        )
        
        # Task-specific head
        if task == 'classification':
            if head_type == 'linear':
                self.head = nn.Linear(latent_dim, num_classes)
            else:  # MLP
                self.head = nn.Sequential(
                    nn.Linear(latent_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(64, num_classes)
                )
        elif task == 'regression':
            if head_type == 'linear':
                self.head = nn.Linear(latent_dim, 1)
            else:  # MLP
                self.head = nn.Sequential(
                    nn.Linear(latent_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(64, 1)
                )
        else:  # reconstruction only
            self.head = None
        
        # Freeze encoder if specified (for fine-tuning)
        if freeze_encoder:
            for param in self.vae.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor, return_vae_outputs: bool = False) -> torch.Tensor:
        """
        Forward pass compatible with TTM pipeline.
        Args:
            x: Input signal [batch, length, channels] or [batch, channels, length]
            return_vae_outputs: Whether to return VAE reconstruction outputs
        Returns:
            Task output or (task_output, vae_outputs) if return_vae_outputs=True
        """
        # Ensure correct input shape [batch, channels, length]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.size(2) == 1:  # [batch, length, 1]
            x = x.transpose(1, 2)  # -> [batch, 1, length]
        
        # VAE forward
        x_recon, mu, logvar = self.vae(x)
        
        # Task-specific output
        if self.head is not None:
            # Use latent mean for downstream task
            output = self.head(mu)
            
            if return_vae_outputs:
                return output, {'recon': x_recon, 'mu': mu, 'logvar': logvar}
            else:
                return output
        else:
            # Return reconstruction for unsupervised training
            if return_vae_outputs:
                return x_recon, {'recon': x_recon, 'mu': mu, 'logvar': logvar}
            else:
                return x_recon
    
    def get_vae_loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get VAE losses for unsupervised training.
        Args:
            x: Input signal
        Returns:
            total_loss, recon_loss, kl_loss
        """
        # Ensure correct shape
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.size(2) == 1:
            x = x.transpose(1, 2)
        
        x_recon, mu, logvar = self.vae(x)
        return self.vae.loss_function(x, x_recon, mu, logvar)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent features for analysis."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.size(2) == 1:
            x = x.transpose(1, 2)
        
        return self.vae.encode(x)
    
    def generate_samples(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Generate new biosignal samples."""
        return self.vae.generate(num_samples, device)
