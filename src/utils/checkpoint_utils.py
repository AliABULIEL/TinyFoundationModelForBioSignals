"""Checkpoint utilities for safe loading/saving with architecture verification.

This module provides robust checkpoint loading that handles architecture mismatches
and provides clear error messages when things go wrong.
"""

import torch
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

from src.models.ttm_adapter import TTMAdapter


def load_ssl_checkpoint_safe(
    checkpoint_path: str,
    device: str = 'cuda',
    target_config: Optional[Dict] = None,
    strict: bool = False,
    verbose: bool = True
) -> Tuple[TTMAdapter, Dict, Dict]:
    """
    Safely load SSL checkpoint with comprehensive architecture validation.

    This function:
    1. Loads the checkpoint file
    2. Extracts and validates architecture metadata
    3. Creates a model with matching architecture
    4. Loads weights with proper error handling
    5. Verifies the loaded model works correctly

    Args:
        checkpoint_path: Path to SSL checkpoint file
        device: Device to load model on ('cuda' or 'cpu')
        target_config: Optional dict to override architecture detection
                      Keys: context_length, patch_size, d_model, input_channels
        strict: If True, raise error on any weight mismatch
               If False, allow partial loading (for fine-tuning)
        verbose: If True, print detailed loading information

    Returns:
        encoder: Loaded TTMAdapter model
        architecture_config: Dict with actual architecture used
        metrics: Dict with checkpoint metrics (loss, epoch, etc.)

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        ValueError: If architecture cannot be determined or is invalid
        RuntimeError: If weights fail to load

    Example:
        >>> encoder, config, metrics = load_ssl_checkpoint_safe(
        ...     'artifacts/ssl/best_model.pt',
        ...     device='cuda',
        ...     verbose=True
        ... )
        >>> print(f"Loaded model with patch_size={config['patch_size']}")
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if verbose:
        print(f"\n{'='*80}")
        print(f"LOADING SSL CHECKPOINT")
        print(f"{'='*80}")
        print(f"Path: {checkpoint_path}")

    # 1. Load checkpoint file
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if verbose:
            print(f"✓ Checkpoint loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")

    # 2. Extract encoder state dict
    if 'encoder_state_dict' in checkpoint:
        encoder_state = checkpoint['encoder_state_dict']
    elif 'model_state_dict' in checkpoint:
        # Extract encoder part
        encoder_state = {
            k.replace('encoder.', ''): v
            for k, v in checkpoint['model_state_dict'].items()
            if k.startswith('encoder.')
        }
    else:
        encoder_state = checkpoint

    if verbose:
        print(f"✓ Found {len(encoder_state)} encoder parameters")

    # 3. Detect architecture from checkpoint
    if target_config is not None:
        # Use provided config
        architecture_config = target_config.copy()
        if verbose:
            print(f"ℹ️  Using provided target config")
    elif 'architecture' in checkpoint:
        # Use saved architecture metadata (NEW FORMAT)
        architecture_config = checkpoint['architecture'].copy()
        if verbose:
            print(f"✓ Found saved architecture metadata")
    else:
        # Fallback: Detect from state dict shapes (OLD FORMAT)
        if verbose:
            print(f"⚠️  No architecture metadata found, detecting from weights...")
        architecture_config = _detect_architecture_from_state_dict(encoder_state, verbose=verbose)

    # Validate required fields
    required_fields = ['context_length', 'patch_size', 'd_model', 'input_channels']
    missing_fields = [f for f in required_fields if f not in architecture_config]
    if missing_fields:
        raise ValueError(f"Missing required architecture fields: {missing_fields}")

    if verbose:
        print(f"\n{'='*80}")
        print(f"DETECTED ARCHITECTURE")
        print(f"{'='*80}")
        for key, value in architecture_config.items():
            print(f"  {key}: {value}")
        print(f"{'='*80}")

    # 4. Create model with detected architecture
    try:
        encoder = TTMAdapter(
            variant=architecture_config.get('variant', 'ibm-granite/granite-timeseries-ttm-r1'),
            task='ssl',
            input_channels=architecture_config['input_channels'],
            context_length=architecture_config['context_length'],
            patch_size=architecture_config['patch_size'],
            d_model=architecture_config['d_model'],
            use_real_ttm=True,
            force_from_scratch=True  # Don't load IBM pretrained
        ).to(device)

        if verbose:
            print(f"✓ Created TTMAdapter model")
    except Exception as e:
        raise RuntimeError(f"Failed to create model: {e}")

    # 5. Verify model architecture matches checkpoint
    actual_config = encoder.get_architecture_config()
    matches, message = encoder.verify_architecture(architecture_config)

    if not matches:
        if strict:
            raise ValueError(f"Architecture mismatch!\n{message}")
        elif verbose:
            warnings.warn(f"Architecture mismatch (continuing anyway):\n{message}")
    elif verbose:
        print(f"✓ {message}")

    # 6. Load weights
    try:
        # Filter to only load encoder backbone weights (skip decoder)
        encoder_backbone_weights = {
            k: v for k, v in encoder_state.items()
            if k.startswith('encoder.backbone') or k.startswith('backbone')
        }

        if verbose:
            print(f"\nLoading weights...")
            print(f"  Backbone weights: {len(encoder_backbone_weights)}")

        missing_keys, unexpected_keys = encoder.load_state_dict(
            encoder_backbone_weights,
            strict=False
        )

        if verbose:
            print(f"✓ Weights loaded")
            print(f"  Missing keys: {len(missing_keys)} (decoder/head, expected)")
            print(f"  Unexpected keys: {len(unexpected_keys)}")

        if strict and (missing_keys or unexpected_keys):
            raise RuntimeError(
                f"Weight loading mismatch:\n"
                f"  Missing: {missing_keys[:5]}\n"
                f"  Unexpected: {unexpected_keys[:5]}"
            )
    except Exception as e:
        raise RuntimeError(f"Failed to load weights: {e}")

    # 7. Verify model works with a test forward pass
    try:
        test_input = torch.randn(
            2,
            architecture_config['input_channels'],
            architecture_config['context_length']
        ).to(device)

        with torch.no_grad():
            test_output = encoder.get_encoder_output(test_input)

        expected_patches = architecture_config['context_length'] // architecture_config['patch_size']
        if test_output.shape != (2, expected_patches, architecture_config['d_model']):
            raise ValueError(
                f"Output shape mismatch: got {test_output.shape}, "
                f"expected (2, {expected_patches}, {architecture_config['d_model']})"
            )

        if verbose:
            print(f"✓ Test forward pass successful")
            print(f"  Output shape: {test_output.shape}")
    except Exception as e:
        raise RuntimeError(f"Model verification failed: {e}")

    # 8. Extract metrics
    metrics = {
        'epoch': checkpoint.get('epoch', 0),
        'val_loss': checkpoint.get('val_loss', checkpoint.get('best_val_loss', None)),
        'train_loss': checkpoint.get('train_loss', None),
    }

    if verbose:
        print(f"\n{'='*80}")
        print(f"✅ CHECKPOINT LOADED SUCCESSFULLY")
        print(f"{'='*80}")
        if metrics['val_loss'] is not None:
            print(f"Validation loss: {metrics['val_loss']:.4f}")
        print(f"Epoch: {metrics['epoch']}")
        print(f"{'='*80}\n")

    return encoder, architecture_config, metrics


def _detect_architecture_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    verbose: bool = True
) -> Dict:
    """
    Detect architecture from checkpoint state dict by inspecting tensor shapes.

    This is a fallback for old checkpoints that don't have architecture metadata.

    Args:
        state_dict: Model state dictionary
        verbose: Print detection process

    Returns:
        config: Detected architecture configuration
    """
    config = {}

    # Detect d_model and patch_size from patcher weights
    # TTM patcher: [d_model, patch_size * input_channels]
    # For 2-channel input: [d_model, 2 * patch_size]
    patcher_keys = [k for k in state_dict.keys() if 'patcher.weight' in k and 'backbone' in k]
    if patcher_keys:
        patcher_weight = state_dict[patcher_keys[0]]
        config['d_model'] = patcher_weight.shape[0]
        patch_dim = patcher_weight.shape[1]

        # FIX: Account for input_channels when detecting patch_size
        # Assume 2 channels (PPG+ECG) for biosignals
        input_channels = 2
        config['input_channels'] = input_channels
        config['patch_size'] = patch_dim // input_channels

        if verbose:
            print(f"  ✓ Detected from patcher: d_model={config['d_model']}, "
                  f"patch_dim={patch_dim}, patch_size={config['patch_size']} "
                  f"(={patch_dim}/{input_channels} channels)")
    else:
        raise ValueError("Could not find patcher weights in checkpoint")

    # Detect num_patches from backbone encoder patch mixer
    # Patch mixer MLP operates on num_patches dimension
    patch_mixer_keys = [k for k in state_dict.keys()
                        if 'backbone.encoder' in k and 'patch_mixer.mlp.fc1.weight' in k]
    if patch_mixer_keys:
        # Use first encoder layer
        first_key = sorted(patch_mixer_keys)[0]
        mlp_weight = state_dict[first_key]
        # MLP weight shape: [out_features, in_features] where in_features = num_patches
        num_patches = mlp_weight.shape[1]
        config['num_patches'] = num_patches
        config['context_length'] = num_patches * config['patch_size']

        if verbose:
            print(f"  ✓ Detected from patch_mixer: num_patches={num_patches}, "
                  f"context_length={config['context_length']}")
    else:
        # Fallback: try to detect from head dimensions
        head_keys = [k for k in state_dict.keys() if 'head.base_forecast_block.weight' in k]
        if head_keys:
            head_weight = state_dict[head_keys[0]]
            input_size = head_weight.shape[1]  # [output_size, input_size]
            # input_size ≈ d_model * num_patches
            num_patches = round(input_size / config['d_model'])
            config['num_patches'] = num_patches
            config['context_length'] = num_patches * config['patch_size']

            if verbose:
                print(f"  ✓ Detected from head: num_patches≈{num_patches}, "
                      f"context_length={config['context_length']}")
        else:
            # Last resort: assume standard TTM-Enhanced
            config['context_length'] = 1024
            config['num_patches'] = config['context_length'] // config['patch_size']
            if verbose:
                print(f"  ⚠️  No patch_mixer or head found, assuming context_length=1024")
                print(f"     num_patches={config['num_patches']}")

    # Add other defaults
    config['variant'] = 'ibm-granite/granite-timeseries-ttm-r1'
    config['using_real_ttm'] = True

    return config


def save_ssl_checkpoint(
    checkpoint_path: str,
    epoch: int,
    encoder: TTMAdapter,
    decoder: Optional[Any],
    optimizer: Any,
    metrics: Dict,
    args: Optional[Dict] = None,
    additional_data: Optional[Dict] = None
) -> None:
    """
    Save SSL checkpoint with comprehensive architecture metadata.

    This ensures the checkpoint can be loaded reliably with correct architecture.

    Args:
        checkpoint_path: Where to save checkpoint
        epoch: Current training epoch
        encoder: TTMAdapter encoder model
        decoder: Optional decoder model
        optimizer: Optimizer state
        metrics: Training metrics (loss, etc.)
        args: Optional training arguments
        additional_data: Any additional data to save

    Example:
        >>> save_ssl_checkpoint(
        ...     'artifacts/ssl/best_model.pt',
        ...     epoch=50,
        ...     encoder=encoder,
        ...     decoder=decoder,
        ...     optimizer=optimizer,
        ...     metrics={'val_loss': 0.123, 'train_loss': 0.145},
        ...     args=vars(args)
        ... )
    """
    # Get actual architecture from model
    architecture_config = encoder.get_architecture_config()

    # Build checkpoint
    checkpoint = {
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'architecture': architecture_config,  # ← CRITICAL: Actual architecture
    }

    # Add decoder if present
    if decoder is not None:
        checkpoint['decoder_state_dict'] = decoder.state_dict()

    # Add training arguments
    if args is not None:
        checkpoint['args'] = args

    # Add any additional data
    if additional_data is not None:
        checkpoint.update(additional_data)

    # Save
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)

    print(f"✓ Checkpoint saved: {checkpoint_path}")
    print(f"  Architecture: context={architecture_config['context_length']}, "
          f"patch={architecture_config['patch_size']}, d_model={architecture_config['d_model']}")
