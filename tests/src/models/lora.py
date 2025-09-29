"""LoRA (Low-Rank Adaptation) implementation for TTM."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """LoRA adaptation layer."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
    ):
        """Initialize LoRA layer.
        
        Args:
            in_features: Input features.
            out_features: Output features.
            rank: LoRA rank.
            alpha: Scaling factor.
            dropout: Dropout rate.
        """
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Initialize
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Initialize LoRA parameters."""
        nn.init.kaiming_uniform_(self.lora_A, a=torch.sqrt(torch.tensor(5.0)).item())
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation.
        
        Args:
            x: Input tensor.
            
        Returns:
            LoRA delta.
        """
        if self.dropout is not None:
            x = self.dropout(x)
        
        # Compute low-rank adaptation
        lora_out = x @ self.lora_A @ self.lora_B
        return lora_out * self.scaling


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation."""
    
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
    ):
        """Initialize LoRA linear layer.
        
        Args:
            base_layer: Base linear layer.
            rank: LoRA rank.
            alpha: Scaling factor.
            dropout: Dropout rate.
        """
        super().__init__()
        self.base_layer = base_layer
        self.lora = LoRALayer(
            base_layer.in_features,
            base_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        
        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output with LoRA adaptation.
        """
        base_out = self.base_layer(x)
        lora_out = self.lora(x)
        return base_out + lora_out


def add_lora_to_model(
    model: nn.Module,
    target_modules: list,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.1,
) -> nn.Module:
    """Add LoRA layers to target modules in a model.
    
    Args:
        model: Base model.
        target_modules: List of module names to add LoRA to.
        rank: LoRA rank.
        alpha: Scaling factor.
        dropout: Dropout rate.
        
    Returns:
        Model with LoRA layers.
    """
    # TODO: Implement in later prompt
    pass


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Merge LoRA weights back into base model.
    
    Args:
        model: Model with LoRA layers.
        
    Returns:
        Model with merged weights.
    """
    # TODO: Implement in later prompt
    pass


def get_lora_parameters(model: nn.Module) -> list:
    """Get all LoRA parameters from model.
    
    Args:
        model: Model potentially containing LoRA layers.
        
    Returns:
        List of LoRA parameters.
    """
    lora_params = []
    for name, module in model.named_modules():
        if isinstance(module, (LoRALayer, LoRALinear)):
            lora_params.extend(module.parameters())
    return lora_params


def count_lora_parameters(model: nn.Module) -> int:
    """Count number of trainable LoRA parameters.
    
    Args:
        model: Model potentially containing LoRA layers.
        
    Returns:
        Number of trainable LoRA parameters.
    """
    return sum(p.numel() for p in get_lora_parameters(model) if p.requires_grad)
