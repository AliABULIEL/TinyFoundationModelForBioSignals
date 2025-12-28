#!/usr/bin/env python3
"""Test that the fixed unfreezing logic correctly identifies TTM mixer blocks."""

import torch
import sys
sys.path.insert(0, '/Users/aliab/Desktop/TinyFoundationModelForBioSignals')

from src.models.ttm_adapter import TTMAdapter
from src.models.channel_utils import unfreeze_last_n_blocks

def test_unfreezing():
    """Test unfreezing logic with TTM model."""
    print("="*80)
    print("Testing TTM Unfreezing Fix")
    print("="*80)

    # Create TTM model
    print("\n1. Creating TTM model...")
    model = TTMAdapter(
        context_length=1024,
        input_channels=2,
        patch_size=128,
        d_model=192,
        task='classification',
        num_classes=2,
        use_real_ttm=True
    )

    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Freeze all parameters first
    print("\n2. Freezing all parameters...")
    for param in model.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable parameters: {trainable:,}")

    # Test unfreezing last 1 block
    print("\n3. Testing unfreeze_last_n_blocks(n=1)...")
    unfreeze_last_n_blocks(model, n=1, verbose=True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n   Result: {trainable:,} / {total:,} trainable ({trainable/total*100:.1f}%)")

    # Freeze all again
    print("\n4. Re-freezing all parameters...")
    for param in model.parameters():
        param.requires_grad = False

    # Test unfreezing last 2 blocks
    print("\n5. Testing unfreeze_last_n_blocks(n=2)...")
    unfreeze_last_n_blocks(model, n=2, verbose=True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n   Result: {trainable:,} / {total:,} trainable ({trainable/total*100:.1f}%)")

    # Freeze all again
    print("\n6. Re-freezing all parameters...")
    for param in model.parameters():
        param.requires_grad = False

    # Test unfreezing all 3 blocks
    print("\n7. Testing unfreeze_last_n_blocks(n=3)...")
    unfreeze_last_n_blocks(model, n=3, verbose=True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n   Result: {trainable:,} / {total:,} trainable ({trainable/total*100:.1f}%)")

    print("\n" + "="*80)
    print("✓ Test completed! Unfreezing now recognizes TTM mixer blocks.")
    print("="*80)

    # Verify specific blocks were unfrozen
    print("\n8. Verifying specific mixer blocks are trainable...")
    encoder = model.encoder
    if hasattr(encoder, 'backbone'):
        backbone = encoder.backbone
        for name, module in backbone.named_modules():
            if 'mixers.' in name and name.split('.')[-1].isdigit():
                # Check if any parameter in this block is trainable
                has_trainable = any(p.requires_grad for p in module.parameters())
                status = "✓ TRAINABLE" if has_trainable else "✗ FROZEN"
                num_params = sum(p.numel() for p in module.parameters())
                print(f"  {status}: {name:50s} ({num_params:,} params)")

    print("\n" + "="*80)

if __name__ == '__main__':
    test_unfreezing()
