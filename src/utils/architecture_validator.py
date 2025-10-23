#!/usr/bin/env python3
"""Architecture validation utilities for biosignal models.

This module provides functions to validate architecture compatibility between
different pipeline stages (SSL → fine-tuning) and generate detailed architecture
reports for debugging.

Key Features:
- Extract actual runtime architecture from models (not constructor params)
- Validate compatibility between checkpoints and expected config
- Generate detailed architecture reports
- Check if stages can load each other's checkpoints

Usage:
    # Get actual architecture from model
    arch = get_actual_architecture(encoder)

    # Validate compatibility
    is_compatible, msg = validate_stage_compatibility(
        stage2_checkpoint='artifacts/ssl/best_model.pt',
        expected_config={'context_length': 1024, 'patch_size': 128}
    )

    # Generate architecture report
    report = create_architecture_report('artifacts/ssl/best_model.pt')
    print(report)
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

import torch
import torch.nn as nn


# IBM TTM Pretrained Variants
TTM_VARIANTS = {
    'TTM-Base': {
        'variant': 'ibm-granite/granite-timeseries-ttm-r1',
        'context_length': 512,
        'patch_size': 64,
        'd_model': 192,
        'num_patches': 8,  # 512 / 64
        'description': 'Base variant for shorter sequences'
    },
    'TTM-Enhanced': {
        'variant': 'ibm-granite/granite-timeseries-ttm-r1',
        'context_length': 1024,
        'patch_size': 128,
        'd_model': 192,
        'num_patches': 8,  # 1024 / 128
        'description': 'Enhanced variant for biosignals (recommended)'
    },
    'TTM-Advanced': {
        'variant': 'ibm-granite/granite-timeseries-ttm-r1',
        'context_length': 1536,
        'patch_size': 128,
        'd_model': 192,
        'num_patches': 12,  # 1536 / 128
        'description': 'Advanced variant for longer sequences'
    }
}


def get_actual_architecture(model: nn.Module) -> Dict[str, Any]:
    """Get ACTUAL runtime architecture from model, not constructor parameters.

    For TTM models, this is critical because TTM uses adaptive patching where
    the actual patch size used internally can differ from the constructor value.

    Args:
        model: PyTorch model (TTMAdapter or similar)

    Returns:
        architecture: Dict with actual runtime configuration

    Example:
        >>> arch = get_actual_architecture(encoder)
        >>> print(f"Actual patch size: {arch['patch_size']}")
        128
        >>> # Even if constructor was called with patch_size=64!
    """
    architecture = {}

    # Prefer model's own method if available
    if hasattr(model, 'get_architecture_config'):
        return model.get_architecture_config()

    # Fallback: Manual extraction
    if hasattr(model, 'context_length'):
        architecture['context_length'] = model.context_length

    if hasattr(model, 'input_channels'):
        architecture['input_channels'] = model.input_channels

    # For TTM models, get ACTUAL patch size from backbone config
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'config'):
        config = model.backbone.config
        if hasattr(config, 'patch_length'):
            architecture['patch_size'] = config.patch_length
            architecture['actual_patch_size'] = config.patch_length
        if hasattr(config, 'd_model'):
            architecture['d_model'] = config.d_model
    elif hasattr(model, 'patch_size'):
        # Fallback to model attribute
        architecture['patch_size'] = model.patch_size

    # Get d_model
    if 'd_model' not in architecture:
        if hasattr(model, 'd_model'):
            architecture['d_model'] = model.d_model
        elif hasattr(model, 'encoder_dim'):
            architecture['d_model'] = model.encoder_dim

    # Calculate num_patches
    if 'context_length' in architecture and 'patch_size' in architecture:
        architecture['num_patches'] = architecture['context_length'] // architecture['patch_size']

    # Get variant info
    if hasattr(model, 'variant'):
        architecture['variant'] = model.variant
    if hasattr(model, 'using_real_ttm'):
        architecture['using_real_ttm'] = model.using_real_ttm

    return architecture


def validate_architecture_compatibility(
    actual_arch: Dict[str, Any],
    expected_arch: Dict[str, Any],
    critical_keys: Optional[list] = None
) -> Tuple[bool, str]:
    """Validate that actual architecture matches expected configuration.

    Args:
        actual_arch: Actual architecture from model/checkpoint
        expected_arch: Expected architecture configuration
        critical_keys: Keys that MUST match (default: context_length, patch_size, d_model)

    Returns:
        is_compatible: Whether architectures are compatible
        message: Detailed message about compatibility

    Example:
        >>> is_compatible, msg = validate_architecture_compatibility(
        ...     actual_arch={'context_length': 1024, 'patch_size': 128, 'd_model': 192},
        ...     expected_arch={'context_length': 1024, 'patch_size': 128}
        ... )
        >>> print(msg)
        ✓ Architecture compatible
    """
    if critical_keys is None:
        critical_keys = ['context_length', 'patch_size', 'd_model']

    mismatches = []

    for key in critical_keys:
        expected_value = expected_arch.get(key)
        actual_value = actual_arch.get(key)

        if expected_value is not None and actual_value is not None:
            if expected_value != actual_value:
                mismatches.append(
                    f"{key}: expected={expected_value}, actual={actual_value}"
                )

    if mismatches:
        message = "❌ Architecture mismatch:\n" + "\n".join(f"  • {m}" for m in mismatches)
        return False, message
    else:
        return True, "✓ Architecture compatible"


def validate_stage_compatibility(
    stage1_checkpoint: Optional[str] = None,
    stage1_arch: Optional[Dict] = None,
    stage2_checkpoint: Optional[str] = None,
    stage2_arch: Optional[Dict] = None,
    expected_config: Optional[Dict] = None
) -> Tuple[bool, str]:
    """Validate compatibility between pipeline stages.

    Checks that Stage 1 checkpoint can be loaded for Stage 2, and Stage 2
    checkpoint can be loaded for Stage 3 fine-tuning.

    Args:
        stage1_checkpoint: Path to Stage 1 checkpoint (optional)
        stage1_arch: Stage 1 architecture dict (optional, alternative to checkpoint)
        stage2_checkpoint: Path to Stage 2 checkpoint (optional)
        stage2_arch: Stage 2 architecture dict (optional)
        expected_config: Expected configuration (optional)

    Returns:
        is_compatible: Whether stages are compatible
        message: Detailed compatibility message

    Example:
        >>> is_compat, msg = validate_stage_compatibility(
        ...     stage2_checkpoint='artifacts/ssl/best_model.pt',
        ...     expected_config={'context_length': 1024, 'patch_size': 128}
        ... )
    """
    # Load architectures from checkpoints if needed
    if stage1_checkpoint and stage1_arch is None:
        ckpt = torch.load(stage1_checkpoint, map_location='cpu', weights_only=False)
        stage1_arch = ckpt.get('architecture', {})

    if stage2_checkpoint and stage2_arch is None:
        ckpt = torch.load(stage2_checkpoint, map_location='cpu', weights_only=False)
        stage2_arch = ckpt.get('architecture', {})

    # Validate against expected config
    if expected_config:
        if stage2_arch:
            return validate_architecture_compatibility(stage2_arch, expected_config)
        elif stage1_arch:
            return validate_architecture_compatibility(stage1_arch, expected_config)

    # Validate Stage 1 → Stage 2 compatibility
    if stage1_arch and stage2_arch:
        # Check critical dimensions match
        return validate_architecture_compatibility(stage2_arch, stage1_arch)

    return True, "✓ No validation constraints provided"


def create_architecture_report(
    checkpoint_path: Optional[str] = None,
    model: Optional[nn.Module] = None,
    architecture: Optional[Dict] = None
) -> str:
    """Generate detailed architecture report for debugging.

    Args:
        checkpoint_path: Path to checkpoint (optional)
        model: Model instance (optional)
        architecture: Architecture dict (optional)

    Returns:
        report: Formatted architecture report

    Example:
        >>> report = create_architecture_report(checkpoint_path='artifacts/ssl/best_model.pt')
        >>> print(report)
        ================================================================================
        ARCHITECTURE REPORT
        ================================================================================
        Source: artifacts/ssl/best_model.pt
        ...
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("ARCHITECTURE REPORT")
    report_lines.append("=" * 80)

    # Get architecture from provided source
    if checkpoint_path:
        report_lines.append(f"Source: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        architecture = ckpt.get('architecture', {})
        has_metadata = 'architecture' in ckpt
    elif model:
        report_lines.append("Source: Model instance")
        architecture = get_actual_architecture(model)
        has_metadata = True
    elif architecture:
        report_lines.append("Source: Provided architecture dict")
        has_metadata = True
    else:
        return "Error: Must provide checkpoint_path, model, or architecture"

    report_lines.append("")

    # Metadata status
    if checkpoint_path:
        if has_metadata:
            report_lines.append("✓ Checkpoint has NEW format (architecture metadata present)")
        else:
            report_lines.append("⚠️  Checkpoint uses OLD format (no architecture metadata)")
        report_lines.append("")

    # Core architecture
    report_lines.append("Core Architecture:")
    report_lines.append("-" * 80)

    core_keys = [
        ('context_length', 'Input sequence length (samples)'),
        ('patch_size', 'Patch size (samples per patch)'),
        ('num_patches', 'Number of patches'),
        ('d_model', 'Model embedding dimension'),
        ('input_channels', 'Number of input channels'),
    ]

    for key, description in core_keys:
        value = architecture.get(key, 'N/A')
        report_lines.append(f"  {key:20s}: {str(value):10s}  # {description}")

    # TTM-specific info
    report_lines.append("")
    report_lines.append("TTM Configuration:")
    report_lines.append("-" * 80)

    ttm_keys = [
        ('variant', 'HuggingFace model ID'),
        ('using_real_ttm', 'Using IBM TTM (vs fallback)'),
        ('adaptive_patching', 'Adaptive patching enabled'),
        ('num_layers', 'Number of encoder layers'),
    ]

    for key, description in ttm_keys:
        value = architecture.get(key, 'N/A')
        report_lines.append(f"  {key:20s}: {str(value):10s}  # {description}")

    # Identify variant
    report_lines.append("")
    report_lines.append("Variant Identification:")
    report_lines.append("-" * 80)

    context = architecture.get('context_length')
    patch = architecture.get('patch_size')

    matched_variant = None
    for variant_name, variant_info in TTM_VARIANTS.items():
        if (variant_info['context_length'] == context and
            variant_info['patch_size'] == patch):
            matched_variant = variant_name
            break

    if matched_variant:
        report_lines.append(f"  Matches: {matched_variant}")
        report_lines.append(f"  Description: {TTM_VARIANTS[matched_variant]['description']}")
    else:
        report_lines.append(f"  Matches: Custom configuration")
        report_lines.append(f"  Note: Does not match standard TTM variants")

    # Recommendations
    report_lines.append("")
    report_lines.append("Recommendations:")
    report_lines.append("-" * 80)

    if context == 1024 and patch == 128:
        report_lines.append("  ✓ Using TTM-Enhanced (RECOMMENDED for biosignals)")
    elif has_metadata:
        report_lines.append("  ✓ Architecture metadata present - checkpoint is production-ready")
    else:
        report_lines.append("  ⚠️  Consider re-training with new checkpoint format")
        report_lines.append("     This will save proper architecture metadata")

    report_lines.append("=" * 80)

    return "\n".join(report_lines)


def get_ttm_variant_config(variant_name: str) -> Dict[str, Any]:
    """Get configuration for a specific TTM variant.

    Args:
        variant_name: Variant name ('TTM-Base', 'TTM-Enhanced', 'TTM-Advanced')

    Returns:
        config: Configuration dictionary

    Raises:
        ValueError: If variant name not recognized

    Example:
        >>> config = get_ttm_variant_config('TTM-Enhanced')
        >>> print(f"Context length: {config['context_length']}")
        1024
    """
    if variant_name not in TTM_VARIANTS:
        available = ', '.join(TTM_VARIANTS.keys())
        raise ValueError(f"Unknown variant '{variant_name}'. Available: {available}")

    return TTM_VARIANTS[variant_name].copy()


def recommend_ttm_variant(
    context_length: int,
    patch_size: Optional[int] = None
) -> Tuple[str, Dict[str, Any]]:
    """Recommend best TTM variant for given configuration.

    Args:
        context_length: Desired input sequence length
        patch_size: Desired patch size (optional)

    Returns:
        variant_name: Recommended variant name
        config: Full variant configuration

    Example:
        >>> variant, config = recommend_ttm_variant(context_length=1024)
        >>> print(f"Recommended: {variant}")
        TTM-Enhanced
    """
    # Find best match based on context_length
    best_match = None
    min_diff = float('inf')

    for variant_name, variant_config in TTM_VARIANTS.items():
        diff = abs(variant_config['context_length'] - context_length)
        if patch_size:
            # Also check patch_size if provided
            if variant_config['patch_size'] == patch_size:
                diff = diff * 0.5  # Prefer exact patch_size match

        if diff < min_diff:
            min_diff = diff
            best_match = variant_name

    return best_match, TTM_VARIANTS[best_match].copy()
