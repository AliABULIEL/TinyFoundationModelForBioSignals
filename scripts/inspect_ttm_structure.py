#!/usr/bin/env python3
"""Inspect TTM model structure to identify mixer blocks for unfreezing."""

import torch
import sys
sys.path.insert(0, '/Users/aliab/Desktop/TinyFoundationModelForBioSignals')

from src.models.ttm_adapter import TTMAdapter

def inspect_ttm_structure():
    """Create TTM model and print its structure."""
    print("="*80)
    print("TTM Model Structure Analysis")
    print("="*80)

    # Create TTM model
    model = TTMAdapter(
        context_length=1024,
        input_channels=2,
        patch_size=128,
        d_model=192,
        task='classification',
        num_classes=2,
        use_real_ttm=True
    )

    print("\n1. Full Module Hierarchy (all named_modules):")
    print("-"*80)
    for name, module in model.named_modules():
        module_type = module.__class__.__name__
        # Count parameters
        num_params = sum(p.numel() for p in module.parameters())
        if num_params > 0 and '.' in name:  # Skip root and empty modules
            print(f"{name:60s} {module_type:30s} {num_params:>10,} params")

    print("\n2. Encoder/Backbone Structure:")
    print("-"*80)
    if hasattr(model, 'encoder'):
        encoder = model.encoder
        print(f"✓ Model has 'encoder' attribute: {encoder.__class__.__name__}")

        if hasattr(encoder, 'backbone'):
            backbone = encoder.backbone
            print(f"✓ Encoder has 'backbone' attribute: {backbone.__class__.__name__}")

            print("\nBackbone named_modules:")
            for name, module in backbone.named_modules():
                if 'mixer' in name.lower():
                    num_params = sum(p.numel() for p in module.parameters())
                    print(f"  {name:60s} {module.__class__.__name__:30s} {num_params:>10,} params")

    print("\n3. Identify Mixer Blocks:")
    print("-"*80)

    # Try different patterns
    patterns = [
        ('mixers.', 'TTM MLP-Mixer blocks'),
        ('layer', 'Standard transformer layers'),
        ('block', 'Standard transformer blocks'),
        ('mlp_mixer_encoder.mixers.', 'Full mixer path'),
    ]

    backbone = model.encoder.backbone if hasattr(model, 'encoder') and hasattr(model.encoder, 'backbone') else model

    for pattern, description in patterns:
        print(f"\nPattern: '{pattern}' ({description})")
        blocks = []
        for name, module in backbone.named_modules():
            if pattern in name.lower():
                # Check if this is a leaf block (ends with digit)
                parts = name.split('.')
                if parts[-1].isdigit():
                    blocks.append(name)

        if blocks:
            print(f"  ✓ Found {len(blocks)} blocks:")
            for block_name in blocks[:5]:
                print(f"    - {block_name}")
            if len(blocks) > 5:
                print(f"    ... and {len(blocks) - 5} more")
        else:
            print(f"  ✗ No blocks found")

    print("\n4. Recommended Unfreezing Pattern:")
    print("-"*80)

    # Find mixer blocks
    mixer_blocks = []
    for name, module in backbone.named_modules():
        if 'mixers.' in name and name.split('.')[-1].isdigit():
            mixer_blocks.append((name, module))

    if mixer_blocks:
        print(f"✓ Detected {len(mixer_blocks)} TTM MLP-Mixer blocks")
        print(f"\nFirst 3 blocks:")
        for name, _ in mixer_blocks[:3]:
            print(f"  - {name}")
        print(f"\nLast 3 blocks (would unfreeze these first):")
        for name, _ in mixer_blocks[-3:]:
            print(f"  - {name}")

        print(f"\nPattern to use in code:")
        print(f"  if 'mixers.' in name and name.split('.')[-1].isdigit():")
    else:
        print("✗ No mixer blocks found - may need different pattern")

    print("\n" + "="*80)

if __name__ == '__main__':
    inspect_ttm_structure()
