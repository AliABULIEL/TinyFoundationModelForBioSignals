"""TTM (TinyTimeMixers) adapter for biosignal analysis.

Wraps the TinyTimeMixers model from Hugging Face with options for:
- Frozen encoder (foundation model mode)
- Partial unfreezing of last N blocks
- LoRA adaptation for efficient fine-tuning

To use TTM from Hugging Face, install:
pip install transformers
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

# Try to import from Hugging Face transformers
try:
    from transformers import AutoModel, AutoConfig
    # For TTM specifically, we might need the PatchTSMixer model
    try:
        from transformers import PatchTSMixerForPrediction, PatchTSMixerModel
        TTM_AVAILABLE = True
    except ImportError:
        # Fallback to AutoModel
        TTM_AVAILABLE = True
        warnings.warn(
            "PatchTSMixer not found in transformers. Using AutoModel fallback.",
            ImportWarning
        )
except ImportError:
    TTM_AVAILABLE = False
    warnings.warn(
        "transformers not installed. Install with: pip install transformers",
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
        variant: str = "ibm/TTM",  # Hugging Face model ID
        task: str = "classification",
        num_classes: Optional[int] = None,
        out_features: Optional[int] = None,
        head_type: str = "linear",
        head_config: Optional[Dict] = None,
        freeze_encoder: bool = True,
        unfreeze_last_n_blocks: int = 0,
        lora_config: Optional[Union[Dict, LoRAConfig]] = None,
        input_channels: int = 1,
        context_length: int = 96,
        prediction_length: int = 0,
        use_huggingface: bool = True,
        **ttm_kwargs
    ):
        """Initialize TTM adapter.
        
        Args:
            variant: HuggingFace model ID or variant name
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
            use_huggingface: Whether to use Hugging Face models
            **ttm_kwargs: Additional arguments for TTM model
        """
        super().__init__()
        
        if not TTM_AVAILABLE:
            raise ImportError(
                "transformers not installed. Install with: pip install transformers"
            )
        
        self.variant = variant
        self.task = task
        self.freeze_encoder = freeze_encoder
        self.unfreeze_last_n_blocks = unfreeze_last_n_blocks
        self.input_channels = input_channels
        self.context_length = context_length
        self.prediction_length = prediction_length
        
        # Initialize encoder from Hugging Face
        self._init_encoder_huggingface(variant, **ttm_kwargs)
        
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
    
    def _init_encoder_huggingface(self, model_id: str, **kwargs):
        """Initialize encoder from Hugging Face.
        
        Args:
            model_id: Hugging Face model ID
            **kwargs: Additional model arguments
        """
        # Map common TTM variants to HF model IDs
        hf_model_map = {
            "ttm-512-96": "ibm/TTM",
            "ttm-1024-96": "ibm/TTM",
            "ttm-1536-96": "ibm/TTM",
        }
        
        # Use mapping if it's a known variant
        if model_id in hf_model_map:
            model_id = hf_model_map[model_id]
        
        try:
            # Try to load as PatchTSMixer (TTM architecture)
            if 'PatchTSMixerForPrediction' in globals():
                self.encoder = PatchTSMixerForPrediction.from_pretrained(
                    model_id,
                    num_input_channels=self.input_channels,
                    context_length=self.context_length,
                    prediction_length=self.prediction_length or self.context_length,
                    **kwargs
                )
            else:
                # Fallback to AutoModel
                self.encoder = AutoModel.from_pretrained(
                    model_id,
                    trust_remote_code=True,  # Some models need this
                    **kwargs
                )
            
            # Get the backbone if available
            if hasattr(self.encoder, 'backbone'):
                self.backbone = self.encoder.backbone
            else:
                self.backbone = self.encoder
                
        except Exception as e:
            warnings.warn(f"Failed to load from HuggingFace: {e}")
            # Create a simple transformer encoder as fallback
            self._create_fallback_encoder()
    
    def _create_fallback_encoder(self):
        """Create a simple transformer encoder as fallback."""
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
        """Initialize task-specific head.
        
        Args:
            task: Task type
            num_classes: Number of classes
            out_features: Output features for regression
            head_type: Type of head
            head_config: Head configuration
        """
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
        """Configure parameter freezing.
        
        Args:
            freeze_encoder: Whether to freeze encoder
            unfreeze_last_n: Number of last blocks to unfreeze
        """
        if freeze_encoder:
            # Freeze all encoder parameters
            for param in self.encoder.parameters():
                param.requires_grad = False
            
            # Unfreeze last N blocks if specified
            if unfreeze_last_n > 0:
                self._unfreeze_last_blocks(unfreeze_last_n)
    
    def _unfreeze_last_blocks(self, n_blocks: int):
        """Unfreeze the last N transformer blocks.
        
        Args:
            n_blocks: Number of blocks to unfreeze
        """
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
        """Apply LoRA to the model.
        
        Args:
            lora_config: LoRA configuration
        """
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
            x: Input tensor [B, T, C] or [B, C, T]
            return_features: Whether to return encoder features
            
        Returns:
            Output predictions or (predictions, features) if return_features=True
        """
        # Most models expect [B, T, C] format
        if x.dim() == 3 and x.size(-1) != self.input_channels:
            # Assume [B, C, T] format, transpose to [B, T, C]
            x = x.transpose(1, 2)
        
        # Get encoder features
        if self.task == "prediction" and hasattr(self.encoder, 'generate'):
            # Use model's prediction method if available
            outputs = self.encoder(x)
            if hasattr(outputs, 'prediction_outputs'):
                features = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else None
                predictions = outputs.prediction_outputs
            else:
                features = None
                predictions = outputs
        else:
            # Extract features for classification/regression
            try:
                outputs = self.encoder(x, output_hidden_states=True)
            except:
                outputs = self.encoder(x)
            
            if hasattr(outputs, 'last_hidden_state'):
                features = outputs.last_hidden_state
            elif hasattr(outputs, 'hidden_states'):
                features = outputs.hidden_states[-1]
            elif isinstance(outputs, torch.Tensor):
                features = outputs
            else:
                # Fallback: use the output directly
                features = outputs
            
            # Apply task head
            if self.head is not None:
                predictions = self.head(features)
            else:
                predictions = features
        
        if return_features:
            return predictions, features
        else:
            return predictions
    
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
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Encoder parameters: {encoder_params:,}")
        print(f"Head parameters: {head_params:,}")
        print(f"Trainable percentage: {trainable_params/total_params*100:.2f}%")


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
    
    # Update variant to use HuggingFace model IDs
    variant = config.get('variant', 'ibm/TTM')
    
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
        context_length=config.get('context_length', 96),
        prediction_length=config.get('prediction_length', 0)
    )
    
    return model
