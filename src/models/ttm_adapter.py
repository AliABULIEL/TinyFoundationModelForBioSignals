"""TTM (TinyTimeMixers) adapter for biosignal analysis.

Wraps the TinyTimeMixers model from IBM's tsfm library with options for:
- Frozen encoder (foundation model mode)
- Partial unfreezing of last N blocks
- LoRA adaptation for efficient fine-tuning
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

# Try to import from IBM's tsfm library
TTM_AVAILABLE = False
try:
    from tsfm import TinyTimeMixerForPrediction
    TTM_AVAILABLE = True
    print("✓ IBM tsfm library found - TTM models available")
except ImportError:
    warnings.warn(
        "IBM tsfm library not installed. Install with: pip install tsfm[notebooks]",
        ImportWarning
    )

# Fallback: try HuggingFace transformers
if not TTM_AVAILABLE:
    try:
        from transformers import AutoModel, AutoConfig
        TRANSFORMERS_AVAILABLE = True
    except ImportError:
        TRANSFORMERS_AVAILABLE = False
        warnings.warn(
            "Neither tsfm nor transformers installed. Using fallback model.",
            ImportWarning
        )

from .heads import (
    LinearClassifier, LinearRegressor,
    MLPClassifier, MLPRegressor,
    SequenceClassifier
)
from .lora import (
    LoRAConfig, apply_lora,
    freeze_non_lora_parameters,
    get_lora_parameters,
    print_lora_summary
)


class TTMAdapter(nn.Module):
    """Adapter for TinyTimeMixers with flexible training configurations."""
    
    def __init__(
        self,
        variant: str = "ibm-granite/granite-timeseries-ttm-v1",
        task: str = "classification",
        num_classes: Optional[int] = None,
        out_features: Optional[int] = None,
        head_type: str = "linear",
        head_config: Optional[Dict] = None,
        freeze_encoder: bool = True,
        unfreeze_last_n_blocks: int = 0,
        lora_config: Optional[Union[Dict, LoRAConfig]] = None,
        input_channels: int = 1,
        context_length: int = 512,
        prediction_length: int = 96,
        use_tsfm: bool = True,
        **ttm_kwargs
    ):
        """Initialize TTM adapter.
        
        Args:
            variant: Model variant (IBM model ID or HuggingFace ID)
            task: Task type ("classification", "regression", "prediction")
            num_classes: Number of classes for classification
            out_features: Output dimension for regression
            head_type: Type of head ("linear", "mlp", "sequence")
            head_config: Configuration for the head
            freeze_encoder: Whether to freeze the encoder
            unfreeze_last_n_blocks: Number of last blocks to unfreeze
            lora_config: LoRA configuration (dict or LoRAConfig)
            input_channels: Number of input channels
            context_length: Length of input context
            prediction_length: Length of prediction horizon
            use_tsfm: Whether to use IBM's tsfm library (True) or fallback
            **ttm_kwargs: Additional arguments for TTM model
        """
        super().__init__()
        
        self.variant = variant
        self.task = task
        self.freeze_encoder = freeze_encoder
        self.unfreeze_last_n_blocks = unfreeze_last_n_blocks
        self.input_channels = input_channels
        self.context_length = context_length
        self.prediction_length = prediction_length if task == "prediction" else 0
        self.using_real_ttm = False  # Track if we're using real TTM
        
        # Initialize encoder
        if use_tsfm and TTM_AVAILABLE:
            self._init_encoder_tsfm(variant, **ttm_kwargs)
        else:
            warnings.warn("TTM not available, using fallback model")
            self._create_fallback_encoder()
        
        # Get encoder output dimension
        self.encoder_dim = self._get_encoder_dim()
        
        # Initialize task head
        self._init_head(
            task, num_classes, out_features,
            head_type, head_config
        )
        
        # Apply freezing configuration
        self._configure_freezing(freeze_encoder, unfreeze_last_n_blocks)
        
        # Apply LoRA if configured
        if lora_config is not None:
            self._apply_lora(lora_config)
    
    def _init_encoder_tsfm(self, model_id: str, **kwargs):
        """Initialize encoder from IBM's tsfm library.
        
        Args:
            model_id: Model ID (e.g., "ibm-granite/granite-timeseries-ttm-v1")
            **kwargs: Additional model arguments
        """
        try:
            from tsfm import TinyTimeMixerForPrediction
            
            print(f"Loading TTM model: {model_id}")
            
            # Configure TTM model
            self.encoder = TinyTimeMixerForPrediction.from_pretrained(
                model_id,
                context_length=self.context_length,
                prediction_length=self.prediction_length if self.prediction_length > 0 else 96,
                num_input_channels=self.input_channels,
                **kwargs
            )
            
            # Extract backbone if available
            if hasattr(self.encoder, 'backbone'):
                self.backbone = self.encoder.backbone
            else:
                self.backbone = self.encoder
                
            self.using_real_ttm = True
            print(f"✓ Successfully loaded TTM model from {model_id}")
            print(f"  Context length: {self.context_length}")
            print(f"  Input channels: {self.input_channels}")
            
        except Exception as e:
            warnings.warn(f"Failed to load TTM from tsfm: {e}")
            warnings.warn("Falling back to simple encoder")
            self._create_fallback_encoder()
    
    def _create_fallback_encoder(self):
        """Create a simple transformer encoder as fallback."""
        print("⚠️ Using fallback encoder (not pre-trained TTM)")
        hidden_size = 512  # Default hidden size
        
        class SimpleEncoder(nn.Module):
            def __init__(self, input_channels, context_length, hidden_size):
                super().__init__()
                self.input_projection = nn.Linear(input_channels, hidden_size)
                self.pos_embedding = nn.Parameter(
                    torch.randn(1, context_length, hidden_size) * 0.02
                )
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=8,
                    dim_feedforward=hidden_size * 4,
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
                self.config = type('Config', (), {'hidden_size': hidden_size, 'd_model': hidden_size})()
            
            def forward(self, x, output_hidden_states=False):
                # Handle different input shapes
                if x.dim() == 3:
                    # Ensure [B, T, C] format
                    if x.size(-1) != self.input_projection.in_features:
                        # [B, C, T] -> [B, T, C]
                        x = x.transpose(1, 2)
                
                x = self.input_projection(x)
                x = x + self.pos_embedding[:, :x.size(1), :]
                x = self.transformer(x)
                
                if output_hidden_states:
                    return type('Output', (), {'last_hidden_state': x, 'hidden_states': (x,)})()
                return x
        
        self.encoder = SimpleEncoder(
            self.input_channels, self.context_length, hidden_size
        )
        self.backbone = self.encoder
        self.using_real_ttm = False
    
    def _get_encoder_dim(self) -> int:
        """Get the output dimension of the encoder."""
        # Try to get from model config
        if hasattr(self.encoder, 'config'):
            if hasattr(self.encoder.config, 'd_model'):
                return self.encoder.config.d_model
            elif hasattr(self.encoder.config, 'hidden_size'):
                return self.encoder.config.hidden_size
        
        # Default based on variant name
        if '512' in str(self.variant):
            return 512
        elif '1024' in str(self.variant):
            return 1024
        elif '1536' in str(self.variant):
            return 1536
        else:
            return 512  # Default
    
    def _init_head(
        self,
        task: str,
        num_classes: Optional[int],
        out_features: Optional[int],
        head_type: str,
        head_config: Optional[Dict]
    ):
        """Initialize task-specific head."""
        if head_config is None:
            head_config = {}
        
        if task == "classification":
            if num_classes is None:
                raise ValueError("num_classes required for classification")
            
            if head_type == "linear":
                self.head = LinearClassifier(
                    self.encoder_dim, num_classes, **head_config
                )
            elif head_type == "mlp":
                self.head = MLPClassifier(
                    self.encoder_dim, num_classes, **head_config
                )
            elif head_type == "sequence":
                self.head = SequenceClassifier(
                    self.encoder_dim, num_classes,
                    head_type="linear" if "mlp" not in head_config else "mlp",
                    **head_config
                )
            else:
                raise ValueError(f"Unknown head type: {head_type}")
        
        elif task == "regression":
            if out_features is None:
                out_features = 1  # Default to scalar output
            
            if head_type == "linear":
                self.head = LinearRegressor(
                    self.encoder_dim, out_features, **head_config
                )
            elif head_type == "mlp":
                self.head = MLPRegressor(
                    self.encoder_dim, out_features, **head_config
                )
            else:
                raise ValueError(f"Unknown head type: {head_type}")
        
        elif task == "prediction":
            # For time series prediction, use the built-in head if available
            self.head = None
        
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def _configure_freezing(self, freeze_encoder: bool, unfreeze_last_n: int):
        """Configure parameter freezing."""
        if freeze_encoder:
            # Freeze all encoder parameters
            for param in self.encoder.parameters():
                param.requires_grad = False
            
            # Unfreeze last N blocks if specified
            if unfreeze_last_n > 0:
                self._unfreeze_last_blocks(unfreeze_last_n)
    
    def _unfreeze_last_blocks(self, n_blocks: int):
        """Unfreeze the last N transformer blocks."""
        # Find transformer blocks in the model
        blocks = []
        for name, module in self.encoder.named_modules():
            if 'block' in name.lower() or 'layer' in name.lower():
                if isinstance(module, nn.Module) and len(list(module.children())) > 0:
                    blocks.append((name, module))
        
        # Unfreeze last N blocks
        if len(blocks) > 0:
            blocks_to_unfreeze = blocks[-n_blocks:]
            for name, block in blocks_to_unfreeze:
                for param in block.parameters():
                    param.requires_grad = True
    
    def _apply_lora(self, lora_config: Union[Dict, LoRAConfig]):
        """Apply LoRA to the model."""
        if isinstance(lora_config, dict):
            lora_config = LoRAConfig.from_dict(lora_config)
        
        # Apply LoRA to encoder
        lora_modules = apply_lora(
            self.encoder,
            r=lora_config.r,
            alpha=lora_config.alpha,
            dropout=lora_config.dropout,
            target_modules=lora_config.target_modules,
            exclude_modules=lora_config.exclude_modules
        )
        
        self.lora_modules = lora_modules
        
        # Freeze non-LoRA parameters
        freeze_non_lora_parameters(self.encoder)
        
        # Print summary
        print(f"Applied LoRA to {len(lora_modules)} modules")
        print_lora_summary(self)
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through encoder and head.
        
        Args:
            x: Input tensor [B, C, T] or [B, T, C]
            return_features: Whether to return encoder features
            
        Returns:
            Output predictions or (predictions, features) if return_features=True
        """
        # Handle input shape based on whether we're using real TTM
        if self.using_real_ttm:
            # TTM expects specific input format
            # Typically [batch_size, context_length, num_channels]
            if x.dim() == 3 and x.size(1) == self.input_channels:
                # [B, C, T] -> [B, T, C]
                x = x.transpose(1, 2)
            
            # TTM forward pass
            outputs = self.encoder(x)
            
            if hasattr(outputs, 'prediction_outputs'):
                # For prediction task
                features = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else None
                predictions = outputs.prediction_outputs
            else:
                # For classification/regression
                features = outputs
                if self.head is not None:
                    predictions = self.head(features)
                else:
                    predictions = features
        else:
            # Fallback encoder
            if x.dim() == 3 and x.size(1) == self.input_channels:
                # [B, C, T] -> [B, T, C]
                x = x.transpose(1, 2)
            
            features = self.encoder(x)
            
            # Pool over sequence dimension if needed
            if features.dim() == 3 and self.head is not None:
                # Use mean pooling
                features = features.mean(dim=1)
            
            if self.head is not None:
                predictions = self.head(features)
            else:
                predictions = features
        
        if return_features:
            return predictions, features
        else:
            return predictions
    
    def is_using_real_ttm(self) -> bool:
        """Check if using real TTM or fallback."""
        return self.using_real_ttm
    
    def get_encoder_params(self) -> List[nn.Parameter]:
        """Get encoder parameters."""
        return list(self.encoder.parameters())
    
    def get_head_params(self) -> List[nn.Parameter]:
        """Get head parameters."""
        if self.head is not None:
            return list(self.head.parameters())
        return []
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get all trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]
    
    def freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def print_parameter_summary(self):
        """Print summary of parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        head_params = sum(p.numel() for p in self.head.parameters()) if self.head else 0
        
        print("=" * 50)
        print("MODEL PARAMETER SUMMARY")
        print("=" * 50)
        print(f"Using real TTM: {self.using_real_ttm}")
        print(f"Model variant: {self.variant}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Encoder parameters: {encoder_params:,}")
        print(f"Head parameters: {head_params:,}")
        print(f"Trainable percentage: {trainable_params/total_params*100:.2f}%")
        print("=" * 50)


def create_ttm_model(config: Dict) -> TTMAdapter:
    """Create TTM model from configuration dictionary.
    
    Args:
        config: Model configuration
        
    Returns:
        Configured TTMAdapter
    """
    # Extract LoRA config if present
    lora_config = None
    if 'lora' in config and config['lora'].get('enabled', False):
        lora_config = LoRAConfig(
            r=config['lora'].get('r', 8),
            alpha=config['lora'].get('alpha', 16),
            dropout=config['lora'].get('dropout', 0.0),
            target_modules=config['lora'].get('target_modules'),
            exclude_modules=config['lora'].get('exclude_modules')
        )
    
    # Use IBM TTM model ID
    variant = config.get('variant', 'ibm-granite/granite-timeseries-ttm-v1')
    
    # Create model
    model = TTMAdapter(
        variant=variant,
        task=config.get('task', 'classification'),
        num_classes=config.get('num_classes'),
        out_features=config.get('out_features'),
        head_type=config.get('head_type', 'linear'),
        head_config=config.get('head_config', {}),
        freeze_encoder=config.get('freeze_encoder', True),
        unfreeze_last_n_blocks=config.get('unfreeze_last_n_blocks', 0),
        lora_config=lora_config,
        input_channels=config.get('input_channels', 1),
        context_length=config.get('context_length', 512),
        prediction_length=config.get('prediction_length', 96),
        use_tsfm=config.get('use_tsfm', True)  # Default to using tsfm
    )
    
    return model
