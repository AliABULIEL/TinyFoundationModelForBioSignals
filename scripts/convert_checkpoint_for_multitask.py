#!/usr/bin/env python3
"""
Convert multi-scale classifier checkpoint to simple encoder for multi-task evaluation.

This extracts only the encoder backbone weights from a fine-tuned checkpoint,
discarding the task-specific classification head, pooling, and fusion layers.
"""

import torch
import argparse
import os
from pathlib import Path


def convert_checkpoint(input_path, output_path, verbose=True):
    """
    Extract only encoder weights from multi-scale classifier checkpoint.

    Args:
        input_path: Path to fine-tuned checkpoint with multi-scale head
        output_path: Path to save converted encoder-only checkpoint
        verbose: Print detailed progress
    """
    if verbose:
        print("="*70)
        print("CHECKPOINT CONVERSION FOR MULTI-TASK EVALUATION")
        print("="*70)
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")
        print()

    # Load checkpoint
    if verbose:
        print("Loading checkpoint...")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Checkpoint not found: {input_path}")

    # PyTorch 2.6+ requires weights_only=False for pickled checkpoints
    try:
        checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions
        checkpoint = torch.load(input_path, map_location='cpu')

    # Determine checkpoint structure
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    if verbose:
        print(f"✓ Loaded checkpoint with {len(state_dict)} parameters")
        print()

    # Extract only encoder weights
    encoder_weights = {}
    excluded_keys = []

    for key, value in state_dict.items():
        # Keep only encoder weights, exclude task-specific components
        if 'encoder' in key:
            # Exclude these components:
            exclude_patterns = [
                'classifier',     # Classification head
                'pooling',        # Attention pooling
                'fusion',         # Multi-scale fusion
                'attention',      # Attention weights
                'head',          # Task head
                'projection',    # Projection layers
                'adapter'        # Adapter layers
            ]

            if any(pattern in key for pattern in exclude_patterns):
                excluded_keys.append(key)
                continue

            # Remove 'encoder.' prefix if present for compatibility
            new_key = key.replace('encoder.', '')
            encoder_weights[new_key] = value
        else:
            excluded_keys.append(key)

    if verbose:
        print(f"✓ Extracted {len(encoder_weights)} encoder parameters")
        print(f"✓ Excluded {len(excluded_keys)} task-specific parameters")
        print()

        if excluded_keys and verbose:
            print("Excluded components:")
            excluded_types = {}
            for key in excluded_keys:
                component = key.split('.')[0]
                excluded_types[component] = excluded_types.get(component, 0) + 1
            for component, count in sorted(excluded_types.items()):
                print(f"  - {component}: {count} parameters")
            print()

    # Detect architecture from weights
    architecture = detect_architecture(encoder_weights)

    if verbose:
        print("Detected architecture:")
        for key, value in architecture.items():
            print(f"  {key}: {value}")
        print()

    # Create converted checkpoint
    converted = {
        'encoder_state_dict': encoder_weights,
        'architecture': architecture,
        'converted_from': input_path,
        'conversion_info': {
            'original_params': len(state_dict),
            'encoder_params': len(encoder_weights),
            'excluded_params': len(excluded_keys)
        }
    }

    # Save converted checkpoint
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    torch.save(converted, output_path)

    if verbose:
        print("="*70)
        print("✅ CONVERSION COMPLETE")
        print("="*70)
        print(f"Saved to: {output_path}")
        print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        print()
        print("This checkpoint can now be used with finetune_enhanced.py for multi-task evaluation.")
        print("="*70)

    return converted


def detect_architecture(encoder_weights):
    """
    Detect architecture parameters from encoder weights.

    Args:
        encoder_weights: Dictionary of encoder parameters

    Returns:
        Dictionary with architecture configuration
    """
    architecture = {
        'context_length': 512,
        'patch_size': 64,
        'd_model': 192,
        'num_channels': 2,
        'num_patches': 8,
        'using_real_ttm': True
    }

    # Try to detect d_model from weights
    for key, value in encoder_weights.items():
        if 'patcher' in key and 'weight' in key:
            if len(value.shape) >= 2:
                architecture['d_model'] = value.shape[0]
                architecture['patch_size'] = value.shape[1] // architecture['num_channels']
                break

    # Calculate num_patches
    architecture['num_patches'] = architecture['context_length'] // architecture['patch_size']

    return architecture


def main():
    parser = argparse.ArgumentParser(
        description='Convert multi-scale classifier checkpoint to simple encoder for multi-task evaluation'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Input checkpoint path (from finetune_butppg.py)'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output checkpoint path (for finetune_enhanced.py)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    try:
        convert_checkpoint(
            args.input,
            args.output,
            verbose=not args.quiet
        )
        return 0
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
