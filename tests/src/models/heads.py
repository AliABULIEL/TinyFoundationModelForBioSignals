"""Task-specific heads for TTM model."""

from typing import List, Optional

import torch
import torch.nn as nn


class MLPHead(nn.Module):
    """Multi-layer perceptron head for classification/regression."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.2,
        activation: str = "relu",
    ):
        """Initialize MLP head.
        
        Args:
            input_dim: Input dimension.
            hidden_dims: Hidden layer dimensions.
            output_dim: Output dimension.
            dropout: Dropout rate.
            activation: Activation function name.
        """
        super().__init__()
        # TODO: Implement in later prompt
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor.
        """
        # TODO: Implement in later prompt
        pass


class LinearHead(nn.Module):
    """Simple linear head for classification/regression."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bias: bool = True,
    ):
        """Initialize linear head.
        
        Args:
            input_dim: Input dimension.
            output_dim: Output dimension.
            bias: Whether to use bias.
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor.
        """
        return self.linear(x)


class AttentionHead(nn.Module):
    """Attention-based aggregation head."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """Initialize attention head.
        
        Args:
            input_dim: Input dimension.
            output_dim: Output dimension.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
        """
        super().__init__()
        # TODO: Implement in later prompt
        pass
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, dim].
            mask: Optional attention mask.
            
        Returns:
            Output tensor.
        """
        # TODO: Implement in later prompt
        pass


class PoolingHead(nn.Module):
    """Pooling-based aggregation head."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        pooling_type: str = "mean",
        hidden_dim: Optional[int] = None,
    ):
        """Initialize pooling head.
        
        Args:
            input_dim: Input dimension.
            output_dim: Output dimension.
            pooling_type: Type of pooling ('mean', 'max', 'both').
            hidden_dim: Optional hidden dimension for projection.
        """
        super().__init__()
        # TODO: Implement in later prompt
        pass
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, dim].
            mask: Optional mask for valid positions.
            
        Returns:
            Output tensor.
        """
        # TODO: Implement in later prompt
        pass


def create_head(
    head_type: str,
    input_dim: int,
    output_dim: int,
    **kwargs,
) -> nn.Module:
    """Factory function to create task heads.
    
    Args:
        head_type: Type of head ('mlp', 'linear', 'attention', 'pooling').
        input_dim: Input dimension.
        output_dim: Output dimension.
        **kwargs: Additional arguments for the head.
        
    Returns:
        Task head module.
    """
    # TODO: Implement in later prompt
    pass
