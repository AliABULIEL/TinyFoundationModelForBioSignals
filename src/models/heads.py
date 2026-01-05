"""Classification heads for time series models."""

import logging
from typing import List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ClassificationHead(nn.Module):
    """
    Base class for classification heads.

    All classification heads should inherit from this class.
    """

    def __init__(self, input_dim: int, num_classes: int) -> None:
        """Initialize classification head."""
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features of shape (B, D)

        Returns:
            Logits of shape (B, K) where K = num_classes
        """
        raise NotImplementedError


class LinearHead(ClassificationHead):
    """
    Simple linear classification head.

    Just a single linear layer: features → logits

    Args:
        input_dim: Input feature dimension
        num_classes: Number of output classes
        dropout: Dropout rate (applied before linear layer)

    Example:
        >>> head = LinearHead(input_dim=768, num_classes=5, dropout=0.1)
        >>> features = torch.randn(32, 768)
        >>> logits = head(features)  # Shape: (32, 5)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialize linear head."""
        super().__init__(input_dim, num_classes)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(input_dim, num_classes)

        # Initialize weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        logger.debug(
            f"Initialized LinearHead: {input_dim} -> {num_classes}, dropout={dropout}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Features of shape (B, D)

        Returns:
            Logits of shape (B, K)
        """
        if x.dim() != 2:
            raise ValueError(
                f"Expected 2D input (batch, features).\n"
                f"  Received: {x.dim()}D with shape {x.shape}\n"
                f"  Hint: Apply pooling before classification head"
            )

        x = self.dropout(x)
        logits = self.classifier(x)

        return logits


class MLPHead(ClassificationHead):
    """
    MLP classification head with one or more hidden layers.

    Architecture: features → [hidden layers with dropout/activation] → logits

    Args:
        input_dim: Input feature dimension
        num_classes: Number of output classes
        hidden_dims: List of hidden layer dimensions (e.g., [256, 128])
        dropout: Dropout rate
        activation: Activation function ("relu", "gelu", "silu")

    Example:
        >>> head = MLPHead(
        >>>     input_dim=768,
        >>>     num_classes=5,
        >>>     hidden_dims=[256, 128],
        >>>     dropout=0.1,
        >>>     activation="gelu"
        >>> )
        >>> features = torch.randn(32, 768)
        >>> logits = head(features)  # Shape: (32, 5)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: List[int],
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        """Initialize MLP head."""
        super().__init__(input_dim, num_classes)

        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout

        # Get activation function
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "gelu":
            act_fn = nn.GELU
        elif activation == "silu":
            act_fn = nn.SiLU
        else:
            raise ValueError(
                f"Unknown activation: {activation}\n"
                f"  Supported: ['relu', 'gelu', 'silu']"
            )

        # Build MLP layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                act_fn(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        logger.debug(
            f"Initialized MLPHead: {input_dim} -> {hidden_dims} -> {num_classes}, "
            f"dropout={dropout}, activation={activation}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Features of shape (B, D)

        Returns:
            Logits of shape (B, K)
        """
        if x.dim() != 2:
            raise ValueError(
                f"Expected 2D input (batch, features).\n"
                f"  Received: {x.dim()}D with shape {x.shape}"
            )

        logits = self.mlp(x)

        return logits


class AttentionPoolingHead(ClassificationHead):
    """
    Classification head with attention-based pooling.

    Uses learnable attention to weight different parts of the feature
    representation before classification.

    Architecture:
        features → attention weights → weighted sum → classifier → logits

    Args:
        input_dim: Input feature dimension
        num_classes: Number of output classes
        dropout: Dropout rate

    Example:
        >>> head = AttentionPoolingHead(input_dim=768, num_classes=5)
        >>> features = torch.randn(32, 16, 768)  # (B, seq_len, D)
        >>> logits = head(features)  # Shape: (32, 5)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialize attention pooling head."""
        super().__init__(input_dim, num_classes)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.Tanh(),
            nn.Linear(input_dim // 4, 1),
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(input_dim, num_classes)

        # Initialize weights
        for module in self.attention.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        logger.debug(
            f"Initialized AttentionPoolingHead: {input_dim} -> {num_classes}, "
            f"dropout={dropout}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention pooling.

        Args:
            x: Features of shape (B, D) or (B, L, D) where L is sequence length

        Returns:
            Logits of shape (B, K)
        """
        if x.dim() == 2:
            # Already pooled, just apply classifier
            x = self.dropout(x)
            logits = self.classifier(x)
            return logits

        elif x.dim() == 3:
            # Apply attention pooling
            # x shape: (B, L, D)
            batch_size, seq_len, feat_dim = x.shape

            # Compute attention weights
            attn_scores = self.attention(x)  # (B, L, 1)
            attn_weights = torch.softmax(attn_scores, dim=1)  # (B, L, 1)

            # Weighted sum
            x_pooled = torch.sum(x * attn_weights, dim=1)  # (B, D)

            # Classify
            x_pooled = self.dropout(x_pooled)
            logits = self.classifier(x_pooled)

            return logits

        else:
            raise ValueError(
                f"Expected 2D or 3D input.\n"
                f"  Received: {x.dim()}D with shape {x.shape}"
            )


def get_classification_head(
    head_type: str,
    input_dim: int,
    num_classes: int,
    hidden_dims: Optional[List[int]] = None,
    dropout: float = 0.1,
    activation: str = "gelu",
) -> ClassificationHead:
    """
    Factory function to create classification head.

    Args:
        head_type: Type of head ("linear", "mlp", "attention")
        input_dim: Input feature dimension
        num_classes: Number of output classes
        hidden_dims: Hidden dimensions for MLP head
        dropout: Dropout rate
        activation: Activation function

    Returns:
        ClassificationHead instance

    Raises:
        ValueError: If head_type is unknown

    Example:
        >>> head = get_classification_head(
        >>>     head_type="mlp",
        >>>     input_dim=768,
        >>>     num_classes=5,
        >>>     hidden_dims=[256, 128]
        >>> )
    """
    head_type = head_type.lower()

    if head_type == "linear":
        return LinearHead(
            input_dim=input_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    elif head_type == "mlp":
        if hidden_dims is None or len(hidden_dims) == 0:
            raise ValueError(
                f"MLP head requires hidden_dims to be specified.\n"
                f"  Example: hidden_dims=[256, 128]"
            )

        return MLPHead(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
        )

    elif head_type == "attention":
        return AttentionPoolingHead(
            input_dim=input_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    else:
        raise ValueError(
            f"Unknown head type: {head_type}\n"
            f"  Supported types: ['linear', 'mlp', 'attention']"
        )


class PoolingLayer(nn.Module):
    """
    Pooling layer for reducing sequence features to single vector.

    Args:
        pooling_type: Type of pooling ("mean", "max", "first", "last")

    Example:
        >>> pooling = PoolingLayer("mean")
        >>> x = torch.randn(32, 16, 768)  # (B, L, D)
        >>> x_pooled = pooling(x)  # (32, 768)
    """

    def __init__(self, pooling_type: str = "mean") -> None:
        """Initialize pooling layer."""
        super().__init__()

        self.pooling_type = pooling_type.lower()

        if self.pooling_type not in ["mean", "max", "first", "last"]:
            raise ValueError(
                f"Unknown pooling type: {pooling_type}\n"
                f"  Supported: ['mean', 'max', 'first', 'last']"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply pooling.

        Args:
            x: Input of shape (B, L, D) or (B, D)

        Returns:
            Pooled output of shape (B, D)
        """
        if x.dim() == 2:
            # Already pooled
            return x

        elif x.dim() == 3:
            if self.pooling_type == "mean":
                return torch.mean(x, dim=1)

            elif self.pooling_type == "max":
                return torch.max(x, dim=1)[0]

            elif self.pooling_type == "first":
                return x[:, 0, :]

            elif self.pooling_type == "last":
                return x[:, -1, :]

        else:
            raise ValueError(
                f"Expected 2D or 3D input.\n"
                f"  Received: {x.dim()}D with shape {x.shape}"
            )
