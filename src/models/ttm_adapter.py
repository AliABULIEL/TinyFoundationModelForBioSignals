"""Fixed TTM adapter with proper dimension handling for classification and SSL"""

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
    """Adapter for TinyTimeMixers with proper dimension handling.
    
    Supports:
    - SSL pretraining with masked signal modeling
    - Downstream classification/regression tasks
    - Flexible context_length and patch_size configuration
    
    Args:
        variant: HuggingFace model ID (default: ibm-granite/granite-timeseries-ttm-r1)
        task: Task type (classification, regression, prediction, ssl)
        num_classes: Number of classes for classification
        out_features: Output features for regression
        head_type: Type of task head (linear, mlp, sequence)
        head_config: Configuration dict for task head
        freeze_encoder: Whether to freeze encoder weights
        unfreeze_last_n_blocks: Number of final blocks to unfreeze
        lora_config: LoRA configuration for parameter-efficient fine-tuning
        input_channels: Number of input channels (2 for SSL, 5 for fine-tuning)
        context_length: Input sequence length (default: 1250 for 10s @ 125Hz)
        patch_size: Size of each patch in samples (default: 125 for 1s patches)
        prediction_length: Output length for prediction tasks
        use_real_ttm: Whether to use real TTM or fallback
        decoder_mode: TTM decoder mode (mix_channel, etc.)
    
    Example:
        >>> # SSL pretraining setup
        >>> model = TTMAdapter(
        ...     task='ssl',
        ...     input_channels=2,
        ...     context_length=1250,
        ...     patch_size=125,
        ...     freeze_encoder=False
        ... )
        
        >>> # Fine-tuning setup
        >>> model = TTMAdapter(
        ...     task='classification',
        ...     num_classes=2,
        ...     input_channels=5,
        ...     context_length=1250,
        ...     freeze_encoder=True
        ... )
    """
    
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
        input_channels: int = 2,
        context_length: int = 1250,
        patch_size: int = 125,
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
        self.patch_size = patch_size
        self.prediction_length = prediction_length
        self.using_real_ttm = False
        
        # Calculate encoder dimension based on configuration
        # For pretrained TTM variants, use their d_model values
        # For custom configs, calculate based on patch_size
        if context_length == 512 and patch_size == 64:
            # TTM-Base pretrained
            self.encoder_dim = 192  # TTM-Base d_model
        elif context_length == 1024 and patch_size == 128:
            # TTM-Enhanced pretrained
            self.encoder_dim = 192  # TTM-Enhanced d_model
        elif context_length == 1536 and patch_size == 128:
            # TTM-Advanced pretrained  
            self.encoder_dim = 192  # TTM-Advanced d_model
        else:
            # Custom configuration - calculate d_model based on patch_size
            if patch_size <= 32:
                self.encoder_dim = 64
            elif patch_size <= 64:
                self.encoder_dim = 256
            elif patch_size <= 128:
                self.encoder_dim = 384  # For patch_size=125
            else:
                self.encoder_dim = 512
        
        # Calculate number of patches
        self.num_patches = context_length // patch_size
        assert context_length % patch_size == 0, \
            f"context_length ({context_length}) must be divisible by patch_size ({patch_size})"
        
        print(f"Initializing TTM with:")
        print(f"  - Input channels: {input_channels}")
        print(f"  - Context length: {context_length}")
        print(f"  - Patch size: {patch_size}")
        print(f"  - Number of patches: {self.num_patches}")
        print(f"  - Encoder dimension: {self.encoder_dim}")
        
        # Initialize encoder
        if use_real_ttm and TTM_AVAILABLE:
            self._init_real_ttm(variant, decoder_mode, **ttm_kwargs)
        else:
            warnings.warn("Real TTM not available, using fallback model")
            self._create_fallback_encoder()
        
        # Initialize task head for classification/regression
        if task not in ["prediction", "ssl"]:
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
        """Initialize the REAL TTM model using tsfm_public.
        
        Args:
            model_id: HuggingFace model identifier
            decoder_mode: Decoder configuration for TTM
            **kwargs: Additional arguments passed to get_model
        """
        try:
            print(f"Loading real TTM model: {model_id}")
            
            # For custom context_length, we need to check if we can use pretrained weights
            # IBM's pretrained TTM variants:
            # - TTM-Base: context=512, patch=64
            # - TTM-Enhanced: context=1024, patch=128  ← BEST for biosignals!
            # - TTM-Advanced: context=1536, patch=128
            
            can_use_pretrained = False
            pretrained_variant = None
            
            # Check if dimensions match any pretrained variant
            if self.context_length == 512 and self.patch_size == 64:
                can_use_pretrained = True
                pretrained_variant = "TTM-Base"
            elif self.context_length == 1024 and self.patch_size == 128:
                can_use_pretrained = True
                pretrained_variant = "TTM-Enhanced"
            elif self.context_length == 1536 and self.patch_size == 128:
                can_use_pretrained = True
                pretrained_variant = "TTM-Advanced"
            
            if can_use_pretrained:
                print(f"  ✓ Dimensions match {pretrained_variant} - loading pretrained weights!")
                # Use get_model to load pretrained weights
                self.encoder = get_model(
                    model_id,
                    context_length=self.context_length,
                    prediction_length=self.prediction_length,
                    num_input_channels=self.input_channels,
                    decoder_mode=decoder_mode,
                    **kwargs
                )
                print(f"  ✓ Successfully loaded {pretrained_variant} pretrained weights!")
            else:
                print(f"  Note: Dimensions (context={self.context_length}, patch={self.patch_size}) don't match pretrained variants")
                print(f"  Recommended: Use context=1024, patch=128 for TTM-Enhanced with biosignals (8.192s @ 125Hz)")
                print(f"  Creating fresh TTM architecture without pretrained weights...")
                
                # Import config class
                from tsfm_public.models.tinytimemixer.configuration_tinytimemixer import TinyTimeMixerConfig
                
                # Calculate appropriate d_model for custom patch_length
                # Rule: d_model should be 2-5X of patch_length AND divisible by 8
                # For patch_length=125: use 384 (3X) or 512 (4X)
                if self.patch_size <= 32:
                    d_model = 64  # Original TTM default for small patches
                elif self.patch_size <= 64:
                    d_model = 256  # 4X for medium patches
                elif self.patch_size <= 128:
                    d_model = 384  # 3X for large patches (your case: 125)
                else:
                    d_model = 512  # For very large patches
                
                config = TinyTimeMixerConfig(
                    context_length=self.context_length,
                    patch_length=self.patch_size,
                    num_input_channels=self.input_channels,
                    prediction_length=self.prediction_length,
                    d_model=d_model,  # Dynamically calculated based on patch_length
                    num_layers=8,  # Standard TTM depth
                    expansion_factor=2,  # TTM default
                    dropout=0.2,
                    head_dropout=0.2,
                    mode="common_channel",  # TTM mode for pretraining (channel-independent)
                    decoder_mode=decoder_mode,
                    scaling="std",  # TTM default
                    adaptive_patching_levels=0,  # DISABLED for custom configs (no multi-resolution)
                    use_positional_encoding=False,  # Not needed for biosignals
                    resolution_prefix_tuning=False,  # Not needed for single resolution
                )
                
                # Calculate num_patches
                config.num_patches = self.num_patches
                
                # Log the configuration
                print(f"  Calculated d_model={d_model} (for patch_length={self.patch_size})")
                print(f"  Fresh config: num_patches={config.num_patches}, patch_length={config.patch_length}")
                print(f"  Fresh config: context_length={config.context_length}, d_model={config.d_model}")
                print(f"  Fresh config: num_layers={config.num_layers}, adaptive_patching_levels={config.adaptive_patching_levels}")
                
                # Initialize model from fresh config
                self.encoder = TinyTimeMixerForPrediction(config)
            
            self.backbone = self.encoder.backbone if hasattr(self.encoder, 'backbone') else self.encoder
            self.using_real_ttm = True
            
            # Print model info
            try:
                param_count = count_parameters(self.encoder)
            except:
                param_count = sum(p.numel() for p in self.encoder.parameters())
            
            print(f"✓ Real TTM loaded successfully!")
            print(f"  Model: {model_id}")
            print(f"  Parameters: {param_count:,}")
            print(f"  Context length: {self.context_length}")
            print(f"  Input channels: {self.input_channels}")
            print(f"  Decoder mode: {decoder_mode}")
            print(f"  Expected patches: {self.num_patches}")
            print(f"  Using pretrained weights: {can_use_pretrained}")
            
        except Exception as e:
            warnings.warn(f"Failed to load real TTM: {e}")
            warnings.warn("Falling back to simple encoder")
            self._create_fallback_encoder()
    
    def _create_fallback_encoder(self):
        """Create a simple CNN encoder as fallback."""
        print("⚠️ Using fallback CNN encoder (not pre-trained TTM)")
        
        class SimpleCNNEncoder(nn.Module):
            def __init__(self, input_channels, hidden_size=192, context_length=1250):
                super().__init__()
                self.context_length = context_length
                self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3)
                self.bn1 = nn.BatchNorm1d(64)
                self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
                self.bn2 = nn.BatchNorm1d(128)
                self.conv3 = nn.Conv1d(128, hidden_size, kernel_size=3, stride=2, padding=1)
                self.bn3 = nn.BatchNorm1d(hidden_size)
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.hidden_size = hidden_size
            
            def forward(self, x):
                # Handle input shape [B, T, C] or [B, C, T]
                if x.dim() == 3:
                    if x.size(1) == self.context_length:
                        # [B, T, C] -> [B, C, T]
                        x = x.transpose(1, 2)
                    elif x.size(-1) != self.context_length:
                        # Assume [B, C, T] is correct
                        pass
                
                x = torch.relu(self.bn1(self.conv1(x)))
                x = torch.relu(self.bn2(self.conv2(x)))
                x = torch.relu(self.bn3(self.conv3(x)))
                return self.pool(x).squeeze(-1)
        
        self.encoder = SimpleCNNEncoder(
            self.input_channels,
            hidden_size=self.encoder_dim,
            context_length=self.context_length
        )
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
        
        # For MLP head, ensure use_batch_norm is properly handled
        if head_type == "mlp":
            # Default to no batch_norm to avoid dimension issues
            head_config['use_batch_norm'] = head_config.get('use_batch_norm', False)
        
        # Calculate input dimension for head
        # After pooling: [batch, hidden_dim]
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
        """Apply LoRA to the model with TTM-specific targeting."""
        if isinstance(lora_config, dict):
            lora_config = LoRAConfig.from_dict(lora_config)
        
        # If no target modules specified, use TTM-specific defaults
        target_modules = lora_config.target_modules
        if target_modules is None or len(target_modules) == 0:
            # Target TTM's attention and mixer layers by default
            target_modules = [
                'attention',      # Attention layers
                'mixer',          # Time mixer layers
                'dense',          # Dense/FF layers
                'query',          # Query projections
                'key',            # Key projections
                'value',          # Value projections
                'output'          # Output projections
            ]
            print(f"Using default TTM target modules: {target_modules}")
        
        exclude_modules = lora_config.exclude_modules
        if exclude_modules is None:
            # Exclude normalization and embedding layers
            exclude_modules = ['norm', 'ln', 'layernorm', 'embed', 'head']
        
        # Apply LoRA to backbone (not decoder/head)
        target = self.backbone if hasattr(self, 'backbone') else self.encoder
        
        lora_modules = apply_lora(
            target,
            r=lora_config.r,
            alpha=lora_config.alpha,
            dropout=lora_config.dropout,
            target_modules=target_modules,
            exclude_modules=exclude_modules
        )
        
        self.lora_modules = lora_modules
        
        # Freeze non-LoRA parameters in the backbone
        freeze_non_lora_parameters(target)
        
        # Validate LoRA was applied
        if len(lora_modules) == 0:
            warnings.warn(
                "No LoRA modules were created. This might indicate that the "
                "target_modules don't match the model architecture. "
                "Try inspecting model.named_modules() to see available names."
            )
        else:
            print(f"✓ Applied LoRA to {len(lora_modules)} modules:")
            for name in list(lora_modules.keys())[:5]:  # Show first 5
                print(f"  - {name}")
            if len(lora_modules) > 5:
                print(f"  ... and {len(lora_modules) - 5} more")
        
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
            
            # For classification/regression/SSL, extract backbone features
            backbone_output = self.backbone(x)
            
            # Extract last_hidden_state
            # Expected shape: [batch, channels, patches, hidden]
            # For context_length=1250, patch_size=125: [batch, input_channels, 10, 192]
            if hasattr(backbone_output, 'last_hidden_state'):
                features = backbone_output.last_hidden_state
                
                # Pool over channels and patches to get [batch, hidden]
                # Average pooling maintains scale better than flattening
                features = features.mean(dim=[1, 2])  # [batch, encoder_dim]
                
            else:
                # Fallback if structure is different
                features = backbone_output
                if features.dim() > 2:
                    batch_size = features.size(0)
                    # Reshape to [batch, -1]
                    features = features.view(batch_size, -1)
                    # Project or select first encoder_dim dimensions
                    if features.size(1) > self.encoder_dim:
                        features = features[:, :self.encoder_dim]
            
            # Pass through task head if not SSL
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
    
    def get_encoder_output(self, x: torch.Tensor) -> torch.Tensor:
        """Get encoder output for SSL pretraining.
        
        Args:
            x: Input tensor [B, C, T] or [B, T, C]
            
        Returns:
            Encoder features [B, P, D] where:
                B = batch size
                P = number of patches
                D = encoder dimension (d_model)
        """
        if self.using_real_ttm:
            # Real TTM expects [batch, time, channels]
            if x.dim() == 3 and x.size(1) == self.input_channels:
                x = x.transpose(1, 2)  # [B, C, T] -> [B, T, C]
            
            B = x.size(0)
            backbone_output = self.backbone(x)
            
            # Extract last_hidden_state
            if hasattr(backbone_output, 'last_hidden_state'):
                features = backbone_output.last_hidden_state
                
                # Debug: Understand TTM's output structure
                # print(f"[DEBUG] TTM last_hidden_state shape: {features.shape}")
                # print(f"[DEBUG] Expected patches: {self.num_patches}, channels: {self.input_channels}")
                
                # Handle different possible TTM output formats
                if features.dim() == 4:
                    # Check if format is [B, C, P, D] where C matches input_channels
                    if features.size(1) == self.input_channels:
                        actual_patches = features.size(2)
                        # print(f"[DEBUG] Format: [B, C, P, D] = [{B}, {features.size(1)}, {actual_patches}, {features.size(3)}]")

                        # Check if TTM's patch count matches our expectation
                        if actual_patches != self.num_patches:
                            # print(f"[WARNING] TTM outputs {actual_patches} patches, but config expects {self.num_patches}")
                            # print(f"[INFO] TTM is using patch_length={self.context_length // actual_patches} internally")
                            # print(f"[INFO] Updating num_patches from {self.num_patches} to {actual_patches}")
                            # print(f"[INFO] Updating patch_size from {self.patch_size} to {self.context_length // actual_patches}")
                            pass
                            
                            # Update the model's patch configuration to match TTM's actual output
                            self.num_patches = actual_patches
                            self.patch_size = self.context_length // actual_patches
                        
                        # Average over channels: [B, C, P, D] -> [B, P, D]
                        features = features.mean(dim=1)
                    
                    # Case 2: [B, P, C, D] - patches, channels, d_model
                    elif features.size(1) == self.num_patches and features.size(2) == self.input_channels:
                        # print(f"[DEBUG] Format: [B, P, C, D] = [{B}, {features.size(1)}, {features.size(2)}, {features.size(3)}]")
                        # Average over channels: [B, P, C, D] -> [B, P, D]
                        features = features.mean(dim=2)

                    # Case 3: [B, C, P', D] where P' doesn't match expected but C matches
                    elif features.size(1) == self.input_channels:
                        actual_patches = features.size(2)
                        # print(f"[DEBUG] Format: [B, C, P', D] with P'={actual_patches} (expected P={self.num_patches})")
                        # print(f"[INFO] Auto-adjusting to TTM's actual patch count")
                        self.num_patches = actual_patches
                        self.patch_size = self.context_length // actual_patches
                        features = features.mean(dim=1)

                    else:
                        # print(f"[WARNING] Unexpected 4D shape: {features.shape}")
                        # print(f"[WARNING] Trying to auto-detect format...")
                        # Try to detect: if dim1 looks like patches (8-32 range) and dim2 looks like channels (1-10 range)
                        if 4 <= features.size(1) <= 64 and 1 <= features.size(2) <= 10:
                            # print(f"[INFO] Detected as [B, P, C, D], averaging over dim=2")
                            features = features.mean(dim=2)
                        else:
                            # print(f"[INFO] Assuming [B, C, P, D], averaging over dim=1")
                            actual_patches = features.size(2)
                            if actual_patches != self.num_patches:
                                # print(f"[INFO] Updating num_patches to {actual_patches}")
                                self.num_patches = actual_patches
                                self.patch_size = self.context_length // actual_patches
                            features = features.mean(dim=1)
                
                elif features.dim() == 3:
                    # Case 3: [B, P, D] - already in correct format
                    if features.size(1) == self.num_patches:
                        # print(f"[DEBUG] Format: [B, P, D] = [{B}, {features.size(1)}, {features.size(2)}] - already correct!")
                        pass  # No transformation needed

                    # Case 4: [B, C*P, D] - channels and patches flattened together
                    elif features.size(1) == self.num_patches * self.input_channels:
                        # print(f"[DEBUG] Format: [B, C*P, D] = [{B}, {features.size(1)}, {features.size(2)}]")
                        # print(f"[DEBUG] Reshaping to [B, C, P, D] and averaging over channels...")
                        # Reshape: [B, C*P, D] -> [B, C, P, D]
                        features = features.reshape(B, self.input_channels, self.num_patches, features.size(-1))
                        # Average over channels: [B, C, P, D] -> [B, P, D]
                        features = features.mean(dim=1)

                    else:
                        # print(f"[WARNING] Unexpected 3D shape: {features.shape}")
                        # print(f"[WARNING] Expected dim1={self.num_patches} or {self.num_patches * self.input_channels}")
                        # Try to use as-is if matches expected patches
                        if features.size(1) < self.num_patches * 2:
                            # print(f"[WARNING] Using as-is since dim1={features.size(1)} is reasonable")
                            pass
                        else:
                            raise ValueError(
                                f"Cannot handle shape {features.shape}. "
                                f"Expected [B={B}, P={self.num_patches}, D] or "
                                f"[B={B}, C*P={self.num_patches * self.input_channels}, D]"
                            )
                
                else:
                    raise ValueError(f"Unexpected TTM output dimensions: {features.dim()}D with shape {features.shape}")

                # print(f"[DEBUG] Final encoder output: {features.shape}")
                # print(f"[DEBUG] Expected: [B={B}, P={self.num_patches}, D={self.encoder_dim}]")
                # print(f"[INFO] Encoder patch_size is now: {self.patch_size}")
                # print(f"[INFO] Encoder num_patches is now: {self.num_patches}")
                
                # Verify output shape
                if features.dim() != 3:
                    raise ValueError(f"Encoder output must be 3D [B, P, D], got {features.dim()}D: {features.shape}")
                
                if features.size(0) != B:
                    raise ValueError(f"Batch size mismatch: got {features.size(0)}, expected {B}")
                
                # if features.size(1) != self.num_patches:
                #     print(f"[WARNING] Patch count mismatch: got {features.size(1)}, expected {self.num_patches}")
                #     print(f"[WARNING] This will cause decoder output size = {features.size(1) * self.patch_size} instead of {self.num_patches * self.patch_size}")
                
                return features
            else:
                raise ValueError("Backbone output doesn't have expected structure for SSL")
        else:
            raise NotImplementedError("SSL not supported with fallback encoder")
    
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
        print(f"Context length: {self.context_length}")
        print(f"Patch size: {self.patch_size}")
        print(f"Number of patches: {self.num_patches}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Backbone parameters: {backbone_params:,}")
        print(f"Head parameters: {head_params:,}")
        print(f"Trainable percentage: {trainable_params/total_params*100:.2f}%")
        print("=" * 50)
    
    def inspect_modules(self, show_all: bool = False, max_display: int = 20):
        """Inspect module names for LoRA targeting.
        
        Args:
            show_all: Show all modules (otherwise only Linear modules)
            max_display: Maximum number of modules to display
        """
        print("\n" + "=" * 50)
        print("MODULE INSPECTION (for LoRA targeting)")
        print("=" * 50)
        
        target = self.backbone if hasattr(self, 'backbone') else self.encoder
        linear_modules = []
        all_modules = []
        
        for name, module in target.named_modules():
            all_modules.append((name, type(module).__name__))
            if isinstance(module, nn.Linear):
                linear_modules.append((name, module.in_features, module.out_features))
        
        print(f"\nTotal modules: {len(all_modules)}")
        print(f"Linear modules: {len(linear_modules)}")
        
        if show_all:
            print(f"\nAll modules (showing {min(max_display, len(all_modules))}/{len(all_modules)}):")
            for name, mod_type in all_modules[:max_display]:
                print(f"  {name:60s} -> {mod_type}")
        else:
            print(f"\nLinear modules (showing {min(max_display, len(linear_modules))}/{len(linear_modules)}):")
            for name, in_feat, out_feat in linear_modules[:max_display]:
                print(f"  {name:60s} -> [{in_feat}, {out_feat}]")
        
        if len(linear_modules) > max_display or (show_all and len(all_modules) > max_display):
            print(f"  ... ({len(linear_modules) - max_display} more)")
        
        print("\nTo target specific modules with LoRA, use patterns like:")
        print("  target_modules: ['attention', 'mixer', 'dense', 'query', 'value']")
        print("=" * 50 + "\n")


def create_ttm_model(config: Dict):
    """Create model from configuration dictionary.
    
    Args:
        config: Model configuration containing:
            - variant: Model variant string
            - task: Task type (classification, regression, prediction, ssl)
            - input_channels: Number of input channels
            - context_length: Input sequence length
            - patch_size: Patch size in samples
            - num_classes: Number of classes (for classification)
            - freeze_encoder: Whether to freeze encoder
            - lora: LoRA configuration
            
    Returns:
        Configured TTMAdapter model
    """
    # Check model type
    model_type = config.get('model_type', 'ttm')
    
    if model_type == 'vae':
        # Use VAE model
        from .vae_adapter import VAEAdapter
        return VAEAdapter(**config)
    
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
        input_channels=config.get('input_channels', 2),
        context_length=config.get('context_length', 1250),
        patch_size=config.get('patch_size', 125),
        prediction_length=config.get('prediction_length', 96),
        use_real_ttm=True,  # Always try to use real TTM
        decoder_mode=config.get('decoder_mode', 'mix_channel')
    )
    
    return model
