"""Fixed TTM adapter with proper dimension handling for classification"""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

# Try to import the REAL TTM from tsfm_public
TTM_AVAILABLE = False
try:
    from tsfm_public import get_model, count_parameters
    from tsfm_public.models.tinytimemixer.modeling_tinytimemixer import (
        TinyTimeMixerForPrediction
    )
    TTM_AVAILABLE = True
    print("✓ Real TTM (tsfm_public) available - using IBM's pre-trained model")
except ImportError as e:
    warnings.warn(
        f"tsfm_public not available: {e}\nUsing fallback model",
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
    """Adapter for TinyTimeMixers with proper dimension handling."""
    
    def __init__(
        self,
        variant: str = "ibm-granite/granite-timeseries-ttm-r1",
        task: str = "classification",
        num_classes: Optional[int] = None,
        out_features: Optional[int] = None,
        head_type: str = "linear",
        head_config: Optional[Dict] = None,
        freeze_encoder: bool = True,
        unfreeze_last_n_blocks: int = 0,
        lora_config: Optional[Union[Dict, LoRAConfig]] = None,
        input_channels: int = 3,
        context_length: int = 512,
        prediction_length: int = 96,
        use_real_ttm: bool = True,
        decoder_mode: str = "mix_channel",
        **ttm_kwargs
    ):
        super().__init__()
        
        self.variant = variant
        self.task = task
        self.freeze_encoder = freeze_encoder
        self.unfreeze_last_n_blocks = unfreeze_last_n_blocks
        self.input_channels = input_channels
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.using_real_ttm = False
        self.encoder_dim = 192  # TTM hidden size
        
        # Initialize encoder
        if use_real_ttm and TTM_AVAILABLE:
            self._init_real_ttm(variant, decoder_mode, **ttm_kwargs)
        else:
            warnings.warn("Real TTM not available, using fallback model")
            self._create_fallback_encoder()
        
        # Initialize task head for classification/regression
        if task != "prediction":
            self._init_head(
                task, num_classes, out_features,
                head_type, head_config
            )
        else:
            self.head = None
        
        # Apply freezing configuration
        if self.using_real_ttm and freeze_encoder:
            self._configure_freezing()
        
        # Apply LoRA if configured
        if lora_config is not None and self.using_real_ttm:
            self._apply_lora(lora_config)
    
    def _init_real_ttm(self, model_id: str, decoder_mode: str, **kwargs):
        """Initialize the REAL TTM model using tsfm_public."""
        try:
            print(f"Loading real TTM model: {model_id}")
            
            # Load TTM model
            self.encoder = get_model(
                model_id,
                context_length=self.context_length,
                prediction_length=self.prediction_length,
                num_input_channels=self.input_channels,
                decoder_mode=decoder_mode,
                **kwargs
            )
            
            self.backbone = self.encoder.backbone if hasattr(self.encoder, 'backbone') else self.encoder
            self.using_real_ttm = True
            
            # Print model info
            param_count = count_parameters(self.encoder)
            print(f"✓ Real TTM loaded successfully!")
            print(f"  Model: {model_id}")
            print(f"  Parameters: {param_count:,}")
            print(f"  Context length: {self.context_length}")
            print(f"  Input channels: {self.input_channels}")
            print(f"  Decoder mode: {decoder_mode}")
            
        except Exception as e:
            warnings.warn(f"Failed to load real TTM: {e}")
            warnings.warn("Falling back to simple encoder")
            self._create_fallback_encoder()
    
    def _create_fallback_encoder(self):
        """Create a simple CNN encoder as fallback."""
        print("⚠️ Using fallback CNN encoder (not pre-trained TTM)")
        
        class SimpleCNNEncoder(nn.Module):
            def __init__(self, input_channels, hidden_size=192):
                super().__init__()
                self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2)
                self.bn1 = nn.BatchNorm1d(64)
                self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2)
                self.bn2 = nn.BatchNorm1d(128)
                self.conv3 = nn.Conv1d(128, hidden_size, kernel_size=3, stride=2)
                self.bn3 = nn.BatchNorm1d(hidden_size)
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.hidden_size = hidden_size
            
            def forward(self, x):
                # Handle input shape [B, T, C] or [B, C, T]
                if x.dim() == 3 and x.size(-1) == self.conv1.in_channels:
                    x = x.transpose(1, 2)  # [B, T, C] -> [B, C, T]
                
                x = torch.relu(self.bn1(self.conv1(x)))
                x = torch.relu(self.bn2(self.conv2(x)))
                x = torch.relu(self.bn3(self.conv3(x)))
                return self.pool(x).squeeze(-1)
        
        self.encoder = SimpleCNNEncoder(self.input_channels)
        self.backbone = self.encoder
        self.using_real_ttm = False
    
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
        
        # For MLP head, ensure batch_norm is properly handled
        if head_type == "mlp":
            # Default to no batch_norm to avoid dimension issues
            head_config['batch_norm'] = head_config.get('batch_norm', False)
        
        # Calculate input dimension for head
        # For real TTM: channels * patches * hidden = 3 * 8 * 192 after flattening
        # We'll pool this down to just hidden_dim
        input_dim = self.encoder_dim
        
        if task == "classification":
            if num_classes is None:
                raise ValueError("num_classes required for classification")
            
            if head_type == "linear":
                self.head = LinearClassifier(
                    input_dim, num_classes, **head_config
                )
            elif head_type == "mlp":
                self.head = MLPClassifier(
                    input_dim, num_classes, **head_config
                )
            else:
                raise ValueError(f"Unknown head type: {head_type}")
        
        elif task == "regression":
            if out_features is None:
                out_features = 1
            
            if head_type == "linear":
                self.head = LinearRegressor(
                    input_dim, out_features, **head_config
                )
            elif head_type == "mlp":
                self.head = MLPRegressor(
                    input_dim, out_features, **head_config
                )
            else:
                raise ValueError(f"Unknown head type: {head_type}")
    
    def _configure_freezing(self):
        """Configure parameter freezing for the backbone."""
        if self.using_real_ttm:
            print("Freezing TTM backbone...")
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            # Keep decoder unfrozen if it exists
            if hasattr(self.encoder, 'decoder'):
                print("Keeping decoder unfrozen for fine-tuning")
                for param in self.encoder.decoder.parameters():
                    param.requires_grad = True
            
            # Count parameters after freezing
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"After freezing: {trainable_params:,}/{total_params:,} trainable params")
    
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
            Output predictions or (predictions, features)
        """
        if self.using_real_ttm:
            # Real TTM expects [batch, time, channels]
            if x.dim() == 3 and x.size(1) == self.input_channels:
                x = x.transpose(1, 2)  # [B, C, T] -> [B, T, C]
            
            # For prediction task, use the full model
            if self.task == "prediction":
                outputs = self.encoder(x)
                if hasattr(outputs, 'prediction_outputs'):
                    return outputs.prediction_outputs
                return outputs
            
            # For classification/regression, extract backbone features
            backbone_output = self.backbone(x)
            
            # Extract last_hidden_state: [batch, channels, patches, hidden]
            if hasattr(backbone_output, 'last_hidden_state'):
                features = backbone_output.last_hidden_state
                # Shape: [batch, 3, 8, 192]
                
                # Pool over channels and patches to get [batch, hidden]
                # Option 1: Mean pool over channels and patches
                features = features.mean(dim=[1, 2])  # [batch, 192]
                
                # Alternative Option 2: Flatten and use linear projection
                # batch_size = features.size(0)
                # features = features.view(batch_size, -1)  # [batch, 3*8*192]
                # Then you'd need a projection layer to reduce dimensionality
            else:
                # Fallback if structure is different
                features = backbone_output
                if features.dim() > 2:
                    batch_size = features.size(0)
                    features = features.view(batch_size, -1)
                    # Take first encoder_dim dimensions
                    features = features[:, :self.encoder_dim]
            
            # Pass through task head
            if self.head is not None:
                predictions = self.head(features)
            else:
                predictions = features
                
        else:
            # Fallback encoder (already returns 2D features)
            features = self.encoder(x)
            
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
    
    def print_parameter_summary(self):
        """Print summary of parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        if self.using_real_ttm:
            backbone_params = sum(p.numel() for p in self.backbone.parameters())
        else:
            backbone_params = sum(p.numel() for p in self.encoder.parameters())
            
        head_params = sum(p.numel() for p in self.head.parameters()) if self.head else 0
        
        print("=" * 50)
        print("MODEL PARAMETER SUMMARY")
        print("=" * 50)
        print(f"Using real TTM: {self.using_real_ttm}")
        print(f"Model variant: {self.variant}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Backbone parameters: {backbone_params:,}")
        print(f"Head parameters: {head_params:,}")
        print(f"Trainable percentage: {trainable_params/total_params*100:.2f}%")
        print("=" * 50)


def create_ttm_model(config: Dict) -> TTMAdapter:
    """Create TTM model from configuration dictionary.
    
    Args:
        config: Model configuration
        
    Returns:
        Configured TTMAdapter with real TTM
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
    
    # Use IBM's real TTM model
    variant = config.get('variant', 'ibm-granite/granite-timeseries-ttm-r1')
    
    # Create model with real TTM
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
        input_channels=config.get('input_channels', 3),
        context_length=config.get('context_length', 512),
        prediction_length=config.get('prediction_length', 96),
        use_real_ttm=True,  # Always try to use real TTM
        decoder_mode=config.get('decoder_mode', 'mix_channel')
    )
    
    return model
