"""Channel inflation utilities for transfer learning between different channel counts.

Handles the transition from SSL pretraining (2 channels: PPG+ECG) to downstream
fine-tuning (5 channels: ACC_X, ACC_Y, ACC_Z, PPG, ECG).

Key functions:
- load_pretrained_with_channel_inflate: Load 2-ch checkpoint and inflate to 5-ch
- unfreeze_last_n_blocks: Progressive unfreezing for fine-tuning
- verify_channel_inflation: Verify weight transfer correctness
"""

import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


def load_pretrained_with_channel_inflate(
    checkpoint_path: str,
    pretrain_channels: int = 2,
    finetune_channels: int = 5,
    freeze_pretrained: bool = True,
    model_config: Optional[Dict] = None,
    device: str = "cpu",
    strict: bool = False
) -> nn.Module:
    """Load pretrained model and inflate channels for fine-tuning.
    
    This function handles the critical task of transferring knowledge from a
    model pretrained on fewer channels (e.g., 2: PPG+ECG) to a model that needs
    to process more channels (e.g., 5: ACC_X,Y,Z + PPG+ECG).
    
    Strategy:
    1. Load the pretrained checkpoint (2 channels)
    2. Create a fresh model with target channel count (5 channels)
    3. Transfer all non-channel-dependent weights exactly
    4. For channel-dependent layers (input embeddings/projections):
       - Copy weights for shared channels (PPG, ECG)
       - Initialize new channels (ACC_X, ACC_Y, ACC_Z) from mean of existing
    5. Optionally freeze pretrained weights, keeping new parts trainable
    
    Args:
        checkpoint_path: Path to pretrained model checkpoint
        pretrain_channels: Number of channels in pretrained model (default: 2)
        finetune_channels: Number of channels for fine-tuning (default: 5)
        freeze_pretrained: If True, freeze all pretrained parameters except
                          new channel weights and task head (default: True)
        model_config: Configuration dict for creating the fine-tuning model.
                     If None, attempts to load from checkpoint metadata.
        device: Device to load model on (default: "cpu")
        strict: If True, require exact parameter name matching (default: False)
    
    Returns:
        Model with inflated channels, ready for fine-tuning
        
    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        ValueError: If channel inflation is invalid (e.g., pretrain >= finetune)
        RuntimeError: If weight transfer fails
    
    Example:
        >>> # Load 2-ch pretrained model and inflate to 5-ch
        >>> model_config = {
        ...     'variant': 'ibm-granite/granite-timeseries-ttm-r1',
        ...     'task': 'classification',
        ...     'num_classes': 2,
        ...     'input_channels': 5,
        ...     'context_length': 1250,
        ...     'freeze_encoder': False  # Will be set by this function
        ... }
        >>> model = load_pretrained_with_channel_inflate(
        ...     checkpoint_path='checkpoints/ssl_pretrained.pt',
        ...     pretrain_channels=2,
        ...     finetune_channels=5,
        ...     freeze_pretrained=True,
        ...     model_config=model_config
        ... )
        >>> # Model now has 5 input channels with pretrained weights frozen
    """
    # Validation
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if finetune_channels <= pretrain_channels:
        raise ValueError(
            f"finetune_channels ({finetune_channels}) must be > "
            f"pretrain_channels ({pretrain_channels})"
        )
    
    print("=" * 70)
    print("CHANNEL INFLATION: {} → {} channels".format(pretrain_channels, finetune_channels))
    print("=" * 70)
    
    # Load checkpoint
    print(f"\n1. Loading pretrained checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract state dict and config
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            pretrained_state = checkpoint['model_state_dict']
            saved_config = checkpoint.get('config', {})
        elif 'state_dict' in checkpoint:
            pretrained_state = checkpoint['state_dict']
            saved_config = checkpoint.get('config', {})
        else:
            pretrained_state = checkpoint
            saved_config = {}
    else:
        pretrained_state = checkpoint.state_dict()
        saved_config = {}
    
    print(f"   ✓ Loaded {len(pretrained_state)} parameters from checkpoint")
    
    # Determine model config
    if model_config is None:
        if saved_config:
            model_config = saved_config.copy()
            model_config['input_channels'] = finetune_channels
            print(f"   ✓ Using config from checkpoint, updated channels to {finetune_channels}")
        else:
            raise ValueError(
                "model_config must be provided if checkpoint doesn't contain config"
            )
    else:
        # Ensure config has correct channel count
        model_config = model_config.copy()
        model_config['input_channels'] = finetune_channels
    
    # Create new model with target channel count
    print(f"\n2. Creating new model with {finetune_channels} input channels")
    from .ttm_adapter import create_ttm_model
    
    # Temporarily disable freezing - we'll handle it after inflation
    model_config['freeze_encoder'] = False
    model = create_ttm_model(model_config)
    
    print(f"   ✓ Created model: {model.__class__.__name__}")
    
    # Get new model's state dict
    new_state = model.state_dict()
    
    # Perform channel inflation
    print(f"\n3. Inflating channels: {pretrain_channels} → {finetune_channels}")
    inflated_state = {}
    channel_dependent_params = []
    transferred_params = []
    skipped_params = []
    
    for param_name, pretrained_param in pretrained_state.items():
        # Check if parameter exists in new model
        if param_name not in new_state:
            if not strict:
                skipped_params.append(param_name)
                continue
            else:
                raise RuntimeError(f"Parameter {param_name} not found in new model")
        
        new_param = new_state[param_name]
        
        # Check if shapes match (non-channel-dependent)
        if pretrained_param.shape == new_param.shape:
            # Direct copy - no channel dimension affected
            inflated_state[param_name] = pretrained_param.clone()
            transferred_params.append(param_name)
        
        else:
            # Shape mismatch - likely channel-dependent
            # Try to intelligently inflate
            
            # Check if this is an input/embedding layer
            # Common patterns: "encoder.input", "backbone.input", "embed", "in_proj"
            is_input_layer = any(keyword in param_name.lower() 
                                for keyword in ['input', 'embed', 'in_proj', 'patch_embed'])
            
            if is_input_layer and len(pretrained_param.shape) >= 2:
                # Attempt channel inflation
                inflated_param = _inflate_channel_weights(
                    pretrained_param,
                    new_param,
                    pretrain_channels,
                    finetune_channels,
                    param_name
                )
                
                if inflated_param is not None:
                    inflated_state[param_name] = inflated_param
                    channel_dependent_params.append(param_name)
                    print(f"   ✓ Inflated: {param_name}")
                    print(f"      {pretrained_param.shape} → {inflated_param.shape}")
                else:
                    warnings.warn(
                        f"Could not inflate {param_name}: {pretrained_param.shape} "
                        f"→ {new_param.shape}"
                    )
                    # Keep randomly initialized weights
                    skipped_params.append(param_name)
            else:
                # Shape mismatch but not an input layer - skip or warn
                if strict:
                    raise RuntimeError(
                        f"Shape mismatch for {param_name}: "
                        f"pretrained {pretrained_param.shape} vs "
                        f"new {new_param.shape}"
                    )
                else:
                    skipped_params.append(param_name)
    
    # Load inflated state dict
    print(f"\n4. Loading inflated weights into new model")
    model.load_state_dict(inflated_state, strict=False)
    
    print(f"   ✓ Transferred: {len(transferred_params)} parameters (exact match)")
    print(f"   ✓ Inflated: {len(channel_dependent_params)} parameters (channel-dependent)")
    if skipped_params:
        print(f"   ⚠ Skipped: {len(skipped_params)} parameters (shape mismatch or not found)")
        if len(skipped_params) <= 5:
            for name in skipped_params:
                print(f"      - {name}")
    
    # Apply freezing if requested
    if freeze_pretrained:
        print(f"\n5. Freezing pretrained parameters")
        _freeze_pretrained_weights(
            model,
            channel_dependent_params,
            finetune_channels,
            pretrain_channels
        )
    else:
        print(f"\n5. Keeping all parameters trainable")
    
    # Print summary
    _print_inflation_summary(model, pretrain_channels, finetune_channels)
    
    print("=" * 70)
    print("✓ Channel inflation complete")
    print("=" * 70)
    
    return model


def _inflate_channel_weights(
    pretrained_param: torch.Tensor,
    new_param: torch.Tensor,
    pretrain_channels: int,
    finetune_channels: int,
    param_name: str
) -> Optional[torch.Tensor]:
    """Inflate channel-dependent weights.
    
    Strategy for inflating from 2→5 channels:
    - Channels 0,1 (PPG, ECG): Copy from pretrained
    - Channels 2,3,4 (ACC_X, ACC_Y, ACC_Z): Initialize from mean of channels 0,1
    
    Args:
        pretrained_param: Pretrained parameter tensor
        new_param: New model parameter tensor (for shape reference)
        pretrain_channels: Original channel count
        finetune_channels: Target channel count
        param_name: Parameter name for logging
    
    Returns:
        Inflated parameter tensor, or None if inflation not possible
    """
    # Create new tensor with target shape
    inflated = torch.zeros_like(new_param)
    
    # Try to determine which dimension is the channel dimension
    # Common patterns:
    # - Conv1d weights: [out_channels, in_channels, kernel_size]
    # - Linear weights: [out_features, in_features]
    # - Embedding: [num_embeddings, embedding_dim]
    
    pretrain_shape = pretrained_param.shape
    new_shape = new_param.shape
    
    # Find which dimension changed
    channel_dim = None
    for dim in range(len(pretrain_shape)):
        if pretrain_shape[dim] != new_shape[dim]:
            # Check if this dimension matches channel counts
            if pretrain_shape[dim] == pretrain_channels and new_shape[dim] == finetune_channels:
                channel_dim = dim
                break
    
    if channel_dim is None:
        warnings.warn(
            f"Could not determine channel dimension for {param_name}: "
            f"{pretrain_shape} → {new_shape}"
        )
        return None
    
    # Inflate based on channel dimension location
    if channel_dim == 0:
        # Channel is first dimension (e.g., out_channels in Conv)
        # Copy pretrained channels
        inflated[:pretrain_channels] = pretrained_param
        
        # Initialize new channels from mean of pretrained
        mean_init = pretrained_param.mean(dim=0, keepdim=True)
        for i in range(pretrain_channels, finetune_channels):
            # Add small noise to break symmetry
            noise = torch.randn_like(mean_init) * 0.01
            inflated[i] = mean_init.squeeze(0) + noise.squeeze(0)
    
    elif channel_dim == 1:
        # Channel is second dimension (e.g., in_channels in Conv/Linear)
        # Copy pretrained channels
        inflated[:, :pretrain_channels] = pretrained_param
        
        # Initialize new channels from mean of pretrained
        mean_init = pretrained_param.mean(dim=1, keepdim=True)
        for i in range(pretrain_channels, finetune_channels):
            # Add small noise to break symmetry
            noise = torch.randn_like(mean_init) * 0.01
            inflated[:, i] = mean_init.squeeze(1) + noise.squeeze(1)
    
    else:
        warnings.warn(
            f"Unexpected channel dimension {channel_dim} for {param_name}"
        )
        return None
    
    return inflated


def _freeze_pretrained_weights(
    model: nn.Module,
    channel_dependent_params: list,
    finetune_channels: int,
    pretrain_channels: int
) -> None:
    """Freeze pretrained weights while keeping new parameters trainable.
    
    Strategy:
    - Freeze all encoder/backbone parameters
    - Keep task head trainable
    - Keep newly initialized channel weights trainable
    
    Args:
        model: Model to apply freezing to
        channel_dependent_params: List of channel-dependent parameter names
        finetune_channels: Target channel count
        pretrain_channels: Original channel count
    """
    # Freeze encoder/backbone
    if hasattr(model, 'backbone'):
        for param in model.backbone.parameters():
            param.requires_grad = False
    elif hasattr(model, 'encoder'):
        for param in model.encoder.parameters():
            param.requires_grad = False
    
    # Keep head trainable
    if hasattr(model, 'head') and model.head is not None:
        for param in model.head.parameters():
            param.requires_grad = True
    
    # For channel-dependent layers, we ideally want to freeze the pretrained
    # part and unfreeze the new part, but this requires parameter splitting
    # which is complex. For now, we unfreeze the entire layer.
    # TODO: Implement partial freezing for channel-inflated layers
    for param_name in channel_dependent_params:
        param = dict(model.named_parameters())[param_name]
        param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"   ✓ Frozen pretrained parameters")
    print(f"   ✓ Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")


def _print_inflation_summary(
    model: nn.Module,
    pretrain_channels: int,
    finetune_channels: int
) -> None:
    """Print summary of channel inflation."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n6. Summary:")
    print(f"   Channels: {pretrain_channels} → {finetune_channels}")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Frozen parameters: {total_params - trainable_params:,}")
    print(f"   Trainable %: {trainable_params/total_params*100:.1f}%")


def unfreeze_last_n_blocks(
    model: nn.Module,
    n: int = 2,
    verbose: bool = True
) -> None:
    """Unfreeze the last N transformer blocks for progressive fine-tuning.
    
    This function implements progressive unfreezing, a technique where we gradually
    unfreeze layers of a pretrained model starting from the output end. This helps
    prevent catastrophic forgetting while adapting to new tasks.
    
    Strategy:
    1. Identify transformer blocks in the model
    2. Unfreeze the last N blocks
    3. Keep earlier blocks frozen
    4. Keep task head unfrozen
    
    Args:
        model: Model to unfreeze blocks in
        n: Number of blocks to unfreeze from the end (default: 2)
        verbose: Whether to print unfreezing information
    
    Example:
        >>> # Initially, encoder is frozen
        >>> model = load_pretrained_with_channel_inflate(...)
        >>> 
        >>> # Unfreeze last 2 blocks
        >>> unfreeze_last_n_blocks(model, n=2)
        >>> 
        >>> # Train with these blocks unfrozen...
        >>> 
        >>> # Later, unfreeze more blocks
        >>> unfreeze_last_n_blocks(model, n=4)
    """
    if verbose:
        print("=" * 70)
        print(f"UNFREEZING LAST {n} BLOCKS")
        print("=" * 70)
    
    # Get backbone/encoder
    backbone = None
    if hasattr(model, 'backbone'):
        backbone = model.backbone
    elif hasattr(model, 'encoder'):
        backbone = model.encoder
    else:
        raise ValueError("Model doesn't have 'backbone' or 'encoder' attribute")
    
    # Find transformer blocks
    # Common patterns: "layer", "block", "transformer_layer"
    blocks = []
    for name, module in backbone.named_modules():
        # Check if this looks like a transformer block
        if any(keyword in name.lower() for keyword in ['layer', 'block']):
            # Make sure it's a direct child (not nested)
            if '.' not in name or name.count('.') == 1:
                blocks.append((name, module))
    
    if not blocks:
        warnings.warn("Could not find transformer blocks in model")
        if verbose:
            print("⚠ No transformer blocks found")
            print("Available modules:")
            for name, _ in backbone.named_modules():
                print(f"  - {name}")
        return
    
    if verbose:
        print(f"\nFound {len(blocks)} transformer blocks:")
        for name, _ in blocks[:5]:  # Show first 5
            print(f"  - {name}")
        if len(blocks) > 5:
            print(f"  ... and {len(blocks) - 5} more")
    
    # Determine which blocks to unfreeze
    n_to_unfreeze = min(n, len(blocks))
    blocks_to_unfreeze = blocks[-n_to_unfreeze:]  # Last N blocks
    
    if verbose:
        print(f"\nUnfreezing last {n_to_unfreeze} blocks:")
    
    # Unfreeze selected blocks
    for name, module in blocks_to_unfreeze:
        for param in module.parameters():
            param.requires_grad = True
        if verbose:
            print(f"  ✓ Unfrozen: {name}")
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if verbose:
        print(f"\nParameter summary:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Frozen: {total_params - trainable_params:,}")
        print(f"  Trainable %: {trainable_params/total_params*100:.1f}%")
        print("=" * 70)


def verify_channel_inflation(
    model_2ch: nn.Module,
    model_5ch: nn.Module,
    verbose: bool = True
) -> bool:
    """Verify that channel inflation was performed correctly.
    
    Checks that all non-channel-dependent parameters are identical between
    the 2-channel and 5-channel models, confirming successful weight transfer.
    
    Args:
        model_2ch: Original 2-channel model
        model_5ch: Inflated 5-channel model
        verbose: Whether to print verification details
    
    Returns:
        True if verification passes, False otherwise
    
    Example:
        >>> # Load original 2-ch model
        >>> model_2ch = create_ttm_model({'input_channels': 2, ...})
        >>> model_2ch.load_state_dict(torch.load('pretrained.pt'))
        >>> 
        >>> # Create inflated 5-ch model
        >>> model_5ch = load_pretrained_with_channel_inflate(
        ...     'pretrained.pt',
        ...     pretrain_channels=2,
        ...     finetune_channels=5
        ... )
        >>> 
        >>> # Verify inflation
        >>> verify_channel_inflation(model_2ch, model_5ch)
        True
    """
    if verbose:
        print("=" * 70)
        print("VERIFYING CHANNEL INFLATION")
        print("=" * 70)
    
    state_2ch = model_2ch.state_dict()
    state_5ch = model_5ch.state_dict()
    
    matching_params = []
    mismatched_params = []
    shape_diff_params = []
    
    for name, param_2ch in state_2ch.items():
        if name not in state_5ch:
            if verbose:
                print(f"⚠ Parameter not in 5-ch model: {name}")
            continue
        
        param_5ch = state_5ch[name]
        
        # Check if shapes match
        if param_2ch.shape == param_5ch.shape:
            # Shapes match - check if values match
            if torch.allclose(param_2ch, param_5ch, rtol=1e-5, atol=1e-8):
                matching_params.append(name)
            else:
                mismatched_params.append(name)
        else:
            # Shapes differ - expected for channel-dependent layers
            shape_diff_params.append((name, param_2ch.shape, param_5ch.shape))
    
    if verbose:
        print(f"\nVerification results:")
        print(f"  ✓ Matching parameters: {len(matching_params)}")
        print(f"  ✗ Mismatched parameters: {len(mismatched_params)}")
        print(f"  ⟷ Shape-different parameters: {len(shape_diff_params)}")
        
        if shape_diff_params:
            print(f"\nChannel-dependent parameters (expected):")
            for name, shape_2ch, shape_5ch in shape_diff_params[:5]:
                print(f"  - {name}: {shape_2ch} → {shape_5ch}")
            if len(shape_diff_params) > 5:
                print(f"  ... and {len(shape_diff_params) - 5} more")
        
        if mismatched_params:
            print(f"\n⚠ WARNING: Found {len(mismatched_params)} parameters with "
                  f"matching shape but different values:")
            for name in mismatched_params[:5]:
                print(f"  - {name}")
            if len(mismatched_params) > 5:
                print(f"  ... and {len(mismatched_params) - 5} more")
    
    # Verification passes if we have matching params and no unexpected mismatches
    success = len(matching_params) > 0 and len(mismatched_params) == 0
    
    if verbose:
        if success:
            print(f"\n✓ Verification PASSED")
        else:
            print(f"\n✗ Verification FAILED")
        print("=" * 70)
    
    return success


def get_channel_inflation_report(
    checkpoint_path: str,
    pretrain_channels: int = 2,
    finetune_channels: int = 5
) -> Dict[str, any]:
    """Generate a detailed report about channel inflation without loading models.
    
    Analyzes a checkpoint and predicts what will happen during channel inflation.
    
    Args:
        checkpoint_path: Path to pretrained checkpoint
        pretrain_channels: Original channel count
        finetune_channels: Target channel count
    
    Returns:
        Dictionary with inflation analysis:
        - 'total_params': Total parameters in checkpoint
        - 'channel_dependent': List of likely channel-dependent parameters
        - 'transferable': List of directly transferable parameters
        - 'report': Human-readable report string
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint.state_dict()
    
    # Analyze parameters
    total_params = 0
    channel_dependent = []
    transferable = []
    
    for name, param in state_dict.items():
        total_params += param.numel()
        
        # Check if likely channel-dependent
        has_channel_in_shape = pretrain_channels in param.shape
        is_input_layer = any(kw in name.lower() for kw in ['input', 'embed', 'in_proj'])
        
        if has_channel_in_shape and is_input_layer:
            channel_dependent.append({
                'name': name,
                'shape': tuple(param.shape),
                'numel': param.numel()
            })
        else:
            transferable.append({
                'name': name,
                'shape': tuple(param.shape),
                'numel': param.numel()
            })
    
    # Generate report
    report = []
    report.append("=" * 70)
    report.append(f"CHANNEL INFLATION ANALYSIS: {checkpoint_path.name}")
    report.append("=" * 70)
    report.append(f"Channels: {pretrain_channels} → {finetune_channels}")
    report.append(f"Total parameters: {total_params:,}")
    report.append(f"\nDirectly transferable: {len(transferable)} parameters")
    report.append(f"Channel-dependent: {len(channel_dependent)} parameters")
    
    if channel_dependent:
        report.append(f"\nChannel-dependent parameters:")
        for param_info in channel_dependent:
            report.append(f"  - {param_info['name']}: {param_info['shape']}")
    
    report.append("=" * 70)
    
    return {
        'total_params': total_params,
        'channel_dependent': channel_dependent,
        'transferable': transferable,
        'report': '\n'.join(report)
    }


if __name__ == "__main__":
    """Quick demonstration of channel inflation utilities."""
    print("Channel Inflation Utilities")
    print("=" * 70)
    print("\nThis module provides utilities for inflating channel counts")
    print("when transferring from SSL pretraining to downstream tasks.")
    print("\nKey functions:")
    print("  - load_pretrained_with_channel_inflate()")
    print("  - unfreeze_last_n_blocks()")
    print("  - verify_channel_inflation()")
    print("  - get_channel_inflation_report()")
    print("\nSee function docstrings for detailed usage examples.")
    print("=" * 70)
