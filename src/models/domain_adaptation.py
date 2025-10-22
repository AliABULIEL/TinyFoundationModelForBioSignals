"""Domain Adaptation for SSL Foundation Models.

Bridges the gap between source domain (VitalDB - hospital ICU) and
target domain (BUT-PPG - smartphone camera) using various adaptation strategies.

Key differences between domains:
- VitalDB: Clinical-grade sensors, controlled environment, ICU patients
- BUT-PPG: Smartphone camera, ambient light, healthy subjects
- Signal quality, noise characteristics, and demographics differ significantly

Adaptation strategies:
1. Projection-based: Learn domain-invariant representations
2. Adversarial: Domain-adversarial neural networks (DANN)
3. Fine-tuning: Progressive unfreezing with different learning rates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DomainProjectionAdapter(nn.Module):
    """Simple projection-based domain adapter.

    Maps SSL encoder features to domain-adapted space using
    a learnable projection with residual connection.

    Architecture:
        features → [Linear → LayerNorm → GELU → Dropout → Linear] + residual → adapted

    Example:
        >>> ssl_encoder = load_ssl_encoder('artifacts/ssl_vitaldb/best_model.pt')
        >>> adapter = DomainProjectionAdapter(d_model=192)
        >>>
        >>> # Extract features from BUT-PPG data
        >>> x = torch.randn(16, 2, 1024)  # BUT-PPG batch
        >>> ssl_features = ssl_encoder.get_encoder_output(x)  # [16, 16, 192]
        >>> adapted = adapter(ssl_features)  # [16, 16, 192]
    """

    def __init__(
        self,
        d_model: int = 192,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.3,
        use_residual: bool = True
    ):
        """Initialize domain projection adapter.

        Args:
            d_model: Input/output feature dimension (from SSL encoder)
            hidden_dim: Hidden layer dimension (default: same as d_model)
            dropout: Dropout probability (default: 0.3)
            use_residual: Whether to use residual connection
        """
        super().__init__()

        self.d_model = d_model
        self.hidden_dim = hidden_dim or d_model
        self.use_residual = use_residual

        # Projection layers
        self.projection = nn.Sequential(
            nn.Linear(d_model, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, d_model)
        )

        # Initialize with small weights for stability
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adapt features to target domain.

        Args:
            x: SSL encoder features [B, P, D] or [B, D]
                B = batch size
                P = number of patches (optional)
                D = feature dimension (d_model)

        Returns:
            adapted: Domain-adapted features, same shape as input
        """
        projected = self.projection(x)

        if self.use_residual:
            return x + projected  # Residual connection
        else:
            return projected


class DomainAdversarialAdapter(nn.Module):
    """Domain-adversarial adapter with gradient reversal.

    Implements Domain-Adversarial Neural Network (DANN) approach:
    - Feature extractor learns domain-invariant representations
    - Domain classifier tries to distinguish source vs target
    - Gradient reversal makes feature extractor fool domain classifier

    Reference:
        Ganin et al. (2016) "Domain-Adversarial Training of Neural Networks"

    Example:
        >>> adapter = DomainAdversarialAdapter(d_model=192)
        >>>
        >>> # Training loop
        >>> ssl_features = ssl_encoder(x)
        >>> adapted, domain_logits = adapter(ssl_features, alpha=0.5)
        >>>
        >>> # Task loss (e.g., classification)
        >>> task_loss = criterion(classifier(adapted), labels)
        >>>
        >>> # Domain adversarial loss
        >>> domain_loss = F.cross_entropy(domain_logits, domain_labels)
        >>>
        >>> total_loss = task_loss + lambda_domain * domain_loss
    """

    def __init__(
        self,
        d_model: int = 192,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.3
    ):
        """Initialize domain adversarial adapter.

        Args:
            d_model: Feature dimension
            hidden_dim: Hidden dimension for adapter (default: same as d_model)
            dropout: Dropout probability
        """
        super().__init__()

        self.d_model = d_model
        self.hidden_dim = hidden_dim or d_model

        # Feature adapter (learns domain-invariant features)
        self.feature_adapter = nn.Sequential(
            nn.Linear(d_model, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, d_model)
        )

        # Domain classifier (tries to distinguish domains)
        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(),  # Key component!
            nn.Linear(d_model, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 2)  # Binary: source vs target
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in [self.feature_adapter, self.domain_classifier]:
            for m in module:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        alpha: float = 1.0,
        return_domain_logits: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with optional domain classification.

        Args:
            x: Input features [B, P, D] or [B, D]
            alpha: Gradient reversal strength (0=no reversal, 1=full reversal)
            return_domain_logits: Whether to return domain classifier logits

        Returns:
            adapted: Adapted features, same shape as input
            domain_logits: Domain classifier logits [B, 2] (if return_domain_logits=True)
        """
        # Adapt features
        adapted = self.feature_adapter(x) + x  # Residual

        if return_domain_logits:
            # Pool patches if needed: [B, P, D] → [B, D]
            if adapted.dim() == 3:
                features_for_domain = adapted.mean(dim=1)
            else:
                features_for_domain = adapted

            # Classify domain (with gradient reversal)
            domain_logits = self.domain_classifier(features_for_domain)
            return adapted, domain_logits
        else:
            return adapted, None


class GradientReversalLayer(nn.Module):
    """Gradient Reversal Layer for domain-adversarial training.

    Forward pass: identity (x → x)
    Backward pass: negates gradients (∂L/∂x → -λ * ∂L/∂x)

    This makes the feature extractor learn domain-invariant features
    by maximizing domain classifier loss (fooling it).
    """

    def __init__(self, alpha: float = 1.0):
        """Initialize GRL.

        Args:
            alpha: Gradient reversal strength (default: 1.0)
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (identity) with gradient reversal in backward."""
        return GradientReversalFunction.apply(x, self.alpha)


class GradientReversalFunction(torch.autograd.Function):
    """Autograd function for gradient reversal."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class ProgressiveFineTuner:
    """Progressive fine-tuning strategy for domain adaptation.

    Fine-tunes SSL encoder with different learning rates:
    - Phase 1: Freeze encoder, train adapter + head (warm-up)
    - Phase 2: Unfreeze top layers, train with low LR
    - Phase 3: Unfreeze all, train end-to-end

    Example:
        >>> ssl_encoder = load_ssl_encoder()
        >>> adapter = DomainProjectionAdapter()
        >>> task_head = nn.Linear(192, 2)
        >>>
        >>> tuner = ProgressiveFineTuner(
        ...     encoder=ssl_encoder,
        ...     adapter=adapter,
        ...     task_head=task_head
        ... )
        >>>
        >>> # Phase 1: Warm-up (5 epochs)
        >>> optimizer = tuner.get_optimizer(phase=1, base_lr=1e-3)
        >>> train(model, optimizer, epochs=5)
        >>>
        >>> # Phase 2: Partial unfreezing (10 epochs)
        >>> optimizer = tuner.get_optimizer(phase=2, base_lr=1e-4)
        >>> train(model, optimizer, epochs=10)
        >>>
        >>> # Phase 3: Full fine-tuning (15 epochs)
        >>> optimizer = tuner.get_optimizer(phase=3, base_lr=1e-5)
        >>> train(model, optimizer, epochs=15)
    """

    def __init__(
        self,
        encoder: nn.Module,
        adapter: nn.Module,
        task_head: nn.Module
    ):
        """Initialize progressive fine-tuner.

        Args:
            encoder: SSL encoder (frozen initially)
            adapter: Domain adapter
            task_head: Task-specific head (classifier/regressor)
        """
        self.encoder = encoder
        self.adapter = adapter
        self.task_head = task_head

    def get_optimizer(
        self,
        phase: int,
        base_lr: float = 1e-4,
        weight_decay: float = 0.01
    ) -> torch.optim.Optimizer:
        """Get optimizer for specified training phase.

        Args:
            phase: Training phase (1, 2, or 3)
            base_lr: Base learning rate
            weight_decay: Weight decay

        Returns:
            Configured optimizer for the phase
        """
        if phase == 1:
            # Phase 1: Freeze encoder, train adapter + head
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.adapter.parameters():
                param.requires_grad = True
            for param in self.task_head.parameters():
                param.requires_grad = True

            optimizer = torch.optim.AdamW([
                {'params': self.adapter.parameters(), 'lr': base_lr},
                {'params': self.task_head.parameters(), 'lr': base_lr}
            ], weight_decay=weight_decay)

            print(f"Phase 1: Encoder frozen, adapter+head trainable (LR={base_lr})")

        elif phase == 2:
            # Phase 2: Unfreeze encoder, use low LR
            for param in self.encoder.parameters():
                param.requires_grad = True

            optimizer = torch.optim.AdamW([
                {'params': self.encoder.parameters(), 'lr': base_lr * 0.1},
                {'params': self.adapter.parameters(), 'lr': base_lr},
                {'params': self.task_head.parameters(), 'lr': base_lr}
            ], weight_decay=weight_decay)

            print(f"Phase 2: Encoder LR={base_lr*0.1}, adapter+head LR={base_lr}")

        elif phase == 3:
            # Phase 3: Full fine-tuning with very low LR
            for param in self.encoder.parameters():
                param.requires_grad = True

            optimizer = torch.optim.AdamW([
                {'params': self.encoder.parameters(), 'lr': base_lr * 0.01},
                {'params': self.adapter.parameters(), 'lr': base_lr * 0.1},
                {'params': self.task_head.parameters(), 'lr': base_lr}
            ], weight_decay=weight_decay)

            print(f"Phase 3: Encoder LR={base_lr*0.01}, adapter LR={base_lr*0.1}, head LR={base_lr}")

        else:
            raise ValueError(f"Invalid phase: {phase}. Must be 1, 2, or 3.")

        return optimizer


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_domain_adapted_model(
    ssl_checkpoint_path: str,
    adaptation_type: str = 'projection',
    num_classes: int = 2,
    device: str = 'cuda'
) -> nn.Module:
    """Create complete domain-adapted model from SSL checkpoint.

    Args:
        ssl_checkpoint_path: Path to SSL pretrained checkpoint
        adaptation_type: Type of adaptation ('projection' or 'adversarial')
        num_classes: Number of output classes for task
        device: Device to load model on

    Returns:
        Complete model with SSL encoder + adapter + task head

    Example:
        >>> model = create_domain_adapted_model(
        ...     'artifacts/ssl_vitaldb/best_model.pt',
        ...     adaptation_type='projection',
        ...     num_classes=2
        ... )
        >>>
        >>> # Use for BUT-PPG fine-tuning
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        >>> train(model, butppg_train_loader, optimizer)
    """
    from src.models.ttm_adapter import TTMAdapter

    # Load SSL checkpoint
    checkpoint = torch.load(ssl_checkpoint_path, map_location=device, weights_only=False)

    # Get encoder config from checkpoint
    if 'config' not in checkpoint:
        raise ValueError(f"Checkpoint missing 'config' key")

    config = checkpoint['config']

    # Create encoder
    encoder = TTMAdapter(
        variant='ibm-granite/granite-timeseries-ttm-r1',
        task='ssl',
        input_channels=2,
        context_length=1024,
        use_real_ttm=True
    )

    # Load SSL weights
    if 'encoder_state_dict' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        print(f"✓ Loaded SSL encoder weights from {ssl_checkpoint_path}")
    else:
        print(f"⚠️  No encoder weights in checkpoint")

    # Create adapter
    d_model = 192  # TTM-Enhanced default
    if adaptation_type == 'projection':
        adapter = DomainProjectionAdapter(d_model=d_model)
        print(f"✓ Created projection adapter")
    elif adaptation_type == 'adversarial':
        adapter = DomainAdversarialAdapter(d_model=d_model)
        print(f"✓ Created adversarial adapter")
    else:
        raise ValueError(f"Unknown adaptation type: {adaptation_type}")

    # Create task head
    task_head = nn.Linear(d_model, num_classes)
    print(f"✓ Created task head ({d_model} → {num_classes})")

    # Combine into full model
    class DomainAdaptedModel(nn.Module):
        def __init__(self, encoder, adapter, head):
            super().__init__()
            self.encoder = encoder
            self.adapter = adapter
            self.head = head

        def forward(self, x):
            # Encode: [B, C, T] → [B, P, D]
            features = self.encoder.get_encoder_output(x)

            # Adapt: [B, P, D] → [B, P, D]
            adapted = self.adapter(features)
            if isinstance(adapted, tuple):
                adapted = adapted[0]  # Handle adversarial case

            # Pool patches: [B, P, D] → [B, D]
            pooled = adapted.mean(dim=1)

            # Classify: [B, D] → [B, num_classes]
            logits = self.head(pooled)
            return logits

    model = DomainAdaptedModel(encoder, adapter, task_head)
    model = model.to(device)

    print(f"\n✓ Domain-adapted model ready!")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model
