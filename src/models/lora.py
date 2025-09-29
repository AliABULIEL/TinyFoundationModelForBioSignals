"""LoRA (Low-Rank Adaptation) implementation for efficient fine-tuning.

Based on: https://arxiv.org/abs/2106.09685
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """LoRA adapter for nn.Linear layers.
    
    Implements low-rank decomposition: W' = W + BA/r
    where W is frozen, B and A are trainable low-rank matrices.
    """
    
    def __init__(
        self,
        original_module: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        merge_weights: bool = False
    ):
        """Initialize LoRA adapter.
        
        Args:
            original_module: Original nn.Linear module to adapt
            r: Rank of decomposition
            alpha: Scaling factor (lora_alpha/r)
            dropout: Dropout probability
            merge_weights: Whether to merge adapter weights
        """
        super().__init__()
        
        self.original_module = original_module
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.merge_weights = merge_weights
        self.merged = False
        
        # Get dimensions
        self.in_features = original_module.in_features
        self.out_features = original_module.out_features
        
        # Freeze original weights
        for param in original_module.parameters():
            param.requires_grad = False
        
        # Create low-rank matrices
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
            self.scaling = alpha / r
            
            # Initialize A with Kaiming uniform and B with zeros
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            
            # Optional dropout
            if dropout > 0:
                self.lora_dropout = nn.Dropout(p=dropout)
            else:
                self.lora_dropout = nn.Identity()
        else:
            # r=0 means no LoRA, just frozen original
            self.lora_A = None
            self.lora_B = None
            self.scaling = 1.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        if self.r > 0 and not self.merged:
            # Original forward pass
            result = self.original_module(x)
            
            # Add LoRA adaptation
            x_dropout = self.lora_dropout(x)
            # Use weight matrices directly for computation
            lora_out = x_dropout @ self.lora_A.t()  # [B, ..., r]
            lora_out = lora_out @ self.lora_B.t()   # [B, ..., out]
            result = result + lora_out * self.scaling
            
            return result
        else:
            # No LoRA or weights merged
            return self.original_module(x)
    
    def merge(self):
        """Merge LoRA weights into original weights."""
        if self.r > 0 and not self.merged:
            # Compute merged weight: W' = W + BA * scaling
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.original_module.weight.data += delta_w
            self.merged = True
    
    def unmerge(self):
        """Unmerge LoRA weights from original weights."""
        if self.r > 0 and self.merged:
            # Subtract merged weight
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.original_module.weight.data -= delta_w
            self.merged = False
    
    def train(self, mode: bool = True):
        """Set training mode."""
        super().train(mode)
        # Unmerge weights when training
        if mode and self.merge_weights and self.merged:
            self.unmerge()
        return self
    
    def eval(self):
        """Set evaluation mode."""
        super().eval()
        # Optionally merge weights when evaluating
        if self.merge_weights and not self.merged:
            self.merge()
        return self


def apply_lora(
    model: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
    exclude_modules: Optional[List[str]] = None
) -> Dict[str, LoRALinear]:
    """Apply LoRA to specified modules in a model.
    
    Args:
        model: PyTorch model
        r: LoRA rank
        alpha: LoRA alpha scaling
        dropout: LoRA dropout
        target_modules: Module name patterns to target (e.g., ['mixer', 'mlp'])
        exclude_modules: Module name patterns to exclude
        
    Returns:
        Dictionary of replaced modules
    """
    if target_modules is None:
        # Default: target all Linear layers except layer norm
        target_modules = []
    
    if exclude_modules is None:
        exclude_modules = ['norm', 'ln', 'layernorm']
    
    lora_modules = {}
    modules_to_replace = []
    
    # First pass: identify modules to replace
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if should target this module
            if target_modules:
                should_target = any(target in name.lower() for target in target_modules)
            else:
                # If no specific targets, target all Linear layers
                should_target = True
                
            should_exclude = any(exclude in name.lower() for exclude in exclude_modules)
            
            if should_target and not should_exclude:
                modules_to_replace.append((name, module))
    
    # Second pass: replace modules
    for name, module in modules_to_replace:
        # Create LoRA module
        lora_module = LoRALinear(
            module, r=r, alpha=alpha, dropout=dropout
        )
        
        # Parse the module path
        parts = name.split('.')
        
        # Navigate to parent and replace
        parent = model
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)
        
        # Replace the module
        if parts[-1].isdigit():
            parent[int(parts[-1])] = lora_module
        else:
            setattr(parent, parts[-1], lora_module)
        
        lora_modules[name] = lora_module
    
    return lora_modules


def mark_lora_parameters(model: nn.Module):
    """Mark LoRA parameters for easy identification.
    
    Args:
        model: Model with LoRA modules
    """
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.is_lora_param = True
        else:
            param.is_lora_param = False


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Get only LoRA parameters from model.
    
    Args:
        model: Model with LoRA modules
        
    Returns:
        List of LoRA parameters
    """
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_params.append(param)
    return lora_params


def freeze_non_lora_parameters(model: nn.Module):
    """Freeze all non-LoRA parameters.
    
    Args:
        model: Model with LoRA modules
    """
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False


def print_lora_summary(model: nn.Module):
    """Print summary of LoRA parameters.
    
    Args:
        model: Model with LoRA modules
    """
    total_params = 0
    lora_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        if 'lora_' in name:
            lora_params += param_count
        
        if param.requires_grad:
            trainable_params += param_count
    
    print(f"Total parameters: {total_params:,}")
    print(f"LoRA parameters: {lora_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"LoRA percentage: {lora_params/total_params*100:.2f}%")
    print(f"Trainable percentage: {trainable_params/total_params*100:.2f}%")


class LoRAConfig:
    """Configuration for LoRA adapters."""
    
    def __init__(
        self,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
        exclude_modules: Optional[List[str]] = None,
        merge_weights: bool = False
    ):
        """Initialize LoRA configuration.
        
        Args:
            r: Rank of decomposition
            alpha: Scaling factor
            dropout: Dropout probability
            target_modules: Modules to apply LoRA to
            exclude_modules: Modules to exclude from LoRA
            merge_weights: Whether to merge weights during eval
        """
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules
        self.exclude_modules = exclude_modules
        self.merge_weights = merge_weights
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'LoRAConfig':
        """Create from dictionary configuration."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'r': self.r,
            'alpha': self.alpha,
            'dropout': self.dropout,
            'target_modules': self.target_modules,
            'exclude_modules': self.exclude_modules,
            'merge_weights': self.merge_weights
        }
