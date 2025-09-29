"""Task-specific heads for classification and regression.

Provides simple and MLP-based heads for downstream tasks.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearClassifier(nn.Module):
    """Simple linear classification head."""
    
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        dropout: float = 0.0,
        bias: bool = True
    ):
        """Initialize linear classifier.
        
        Args:
            in_features: Input feature dimension
            num_classes: Number of output classes
            dropout: Dropout probability
            bias: Whether to use bias
        """
        super().__init__()
        
        self.in_features = in_features
        self.num_classes = num_classes
        
        # Dropout layer
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()
        
        # Linear layer
        self.fc = nn.Linear(in_features, num_classes, bias=bias)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc.weight)
        if bias:
            nn.init.zeros_(self.fc.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [B, in_features] or [B, T, in_features]
            
        Returns:
            Logits tensor [B, num_classes]
        """
        # Handle sequence output by averaging
        if x.dim() == 3:
            x = x.mean(dim=1)  # Average over time dimension
        
        x = self.dropout(x)
        logits = self.fc(x)
        return logits


class LinearRegressor(nn.Module):
    """Simple linear regression head."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        dropout: float = 0.0,
        bias: bool = True
    ):
        """Initialize linear regressor.
        
        Args:
            in_features: Input feature dimension
            out_features: Output dimension (default 1 for scalar)
            dropout: Dropout probability
            bias: Whether to use bias
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Dropout layer
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()
        
        # Linear layer
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc.weight)
        if bias:
            nn.init.zeros_(self.fc.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [B, in_features] or [B, T, in_features]
            
        Returns:
            Output tensor [B, out_features]
        """
        # Handle sequence output by averaging
        if x.dim() == 3:
            x = x.mean(dim=1)  # Average over time dimension
        
        x = self.dropout(x)
        output = self.fc(x)
        return output


class MLPClassifier(nn.Module):
    """MLP classification head with batch norm and dropout."""
    
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = 'relu',
        use_batch_norm: bool = True,
        use_layer_norm: bool = False
    ):
        """Initialize MLP classifier.
        
        Args:
            in_features: Input feature dimension
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu', 'tanh')
            use_batch_norm: Whether to use batch normalization
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()
        
        self.in_features = in_features
        self.num_classes = num_classes
        
        if hidden_dims is None:
            # Default: single hidden layer with size between input and output
            hidden_dims = [max(num_classes * 2, in_features // 2)]
        
        # Build layers
        layers = []
        prev_dim = in_features
        
        for hidden_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            # Activation
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            
            prev_dim = hidden_dim
        
        # Final layer (no activation)
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/He initialization."""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [B, in_features] or [B, T, in_features]
            
        Returns:
            Logits tensor [B, num_classes]
        """
        # Handle sequence output by averaging
        if x.dim() == 3:
            x = x.mean(dim=1)  # Average over time dimension
        
        logits = self.mlp(x)
        return logits


class MLPRegressor(nn.Module):
    """MLP regression head with batch norm and dropout."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = 'relu',
        use_batch_norm: bool = True,
        use_layer_norm: bool = False
    ):
        """Initialize MLP regressor.
        
        Args:
            in_features: Input feature dimension
            out_features: Output dimension
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu', 'tanh')
            use_batch_norm: Whether to use batch normalization
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        if hidden_dims is None:
            # Default: single hidden layer
            hidden_dims = [in_features // 2]
        
        # Build layers
        layers = []
        prev_dim = in_features
        
        for hidden_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            # Activation
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            
            prev_dim = hidden_dim
        
        # Final layer (no activation for regression)
        layers.append(nn.Linear(prev_dim, out_features))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/He initialization."""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [B, in_features] or [B, T, in_features]
            
        Returns:
            Output tensor [B, out_features]
        """
        # Handle sequence output by averaging
        if x.dim() == 3:
            x = x.mean(dim=1)  # Average over time dimension
        
        output = self.mlp(x)
        return output


class AttentionPooling(nn.Module):
    """Attention-based pooling for sequence features."""
    
    def __init__(self, in_features: int):
        """Initialize attention pooling.
        
        Args:
            in_features: Input feature dimension
        """
        super().__init__()
        self.attention = nn.Linear(in_features, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [B, T, in_features]
            
        Returns:
            Pooled tensor [B, in_features]
        """
        # Compute attention weights
        attn_weights = self.attention(x)  # [B, T, 1]
        attn_weights = F.softmax(attn_weights, dim=1)  # [B, T, 1]
        
        # Apply attention
        pooled = torch.sum(x * attn_weights, dim=1)  # [B, in_features]
        return pooled


class SequenceClassifier(nn.Module):
    """Classifier for sequence data with various pooling options."""
    
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        pooling: str = 'mean',
        head_type: str = 'linear',
        **head_kwargs
    ):
        """Initialize sequence classifier.
        
        Args:
            in_features: Input feature dimension
            num_classes: Number of output classes
            pooling: Pooling method ('mean', 'max', 'last', 'attention')
            head_type: Type of head ('linear' or 'mlp')
            **head_kwargs: Additional arguments for the head
        """
        super().__init__()
        
        self.pooling = pooling
        
        # Setup pooling
        if pooling == 'attention':
            self.pool = AttentionPooling(in_features)
        else:
            self.pool = None
        
        # Setup head
        if head_type == 'linear':
            self.head = LinearClassifier(in_features, num_classes, **head_kwargs)
        elif head_type == 'mlp':
            self.head = MLPClassifier(in_features, num_classes, **head_kwargs)
        else:
            raise ValueError(f"Unknown head type: {head_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [B, T, in_features]
            
        Returns:
            Logits tensor [B, num_classes]
        """
        # Apply pooling
        if self.pooling == 'mean':
            x = x.mean(dim=1)
        elif self.pooling == 'max':
            x, _ = x.max(dim=1)
        elif self.pooling == 'last':
            x = x[:, -1, :]
        elif self.pooling == 'attention':
            x = self.pool(x)
        
        # Apply head (expects 2D input now)
        logits = self.head(x)
        return logits
