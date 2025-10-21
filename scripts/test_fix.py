#!/usr/bin/env python3
"""Test if the fix works."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.models.ttm_adapter import TTMAdapter

# Load SSL checkpoint
ckpt = torch.load('artifacts/butppg_ssl/best_model.pt', map_location='cpu', weights_only=False)
ssl_state = ckpt['encoder_state_dict']

# Create backbone dict with BOTH prefixes (matching the fix)
backbone_state_dict = {}
for key, value in ssl_state.items():
    if 'encoder.backbone' in key:
        # Add with original key
        backbone_state_dict[key] = value
        # Add with stripped prefix (only first occurrence!)
        stripped_key = key.replace('encoder.', '', 1)
        backbone_state_dict[stripped_key] = value

print(f"Created backbone_state_dict with {len(backbone_state_dict)} keys")

# Create model
model = TTMAdapter(
    variant='ibm-granite/granite-timeseries-ttm-r1',
    task='classification',
    num_classes=2,
    input_channels=2,
    context_length=1024,
    patch_size=128,
    d_model=192,
    freeze_encoder=False
)

# Trigger auto-adaptation
with torch.no_grad():
    _ = model.get_encoder_output(torch.randn(1, 2, 1024))

print(f"Model patch_size after adaptation: {model.patch_size}")

# Try loading
missing, unexpected = model.load_state_dict(backbone_state_dict, strict=False)

backbone_missing = [k for k in missing if 'backbone' in k and 'encoder' in k]
head_missing = [k for k in missing if 'head' in k or 'classifier' in k or 'decoder' in k]

print(f"\nResults:")
print(f"  Total keys in backbone_state_dict: {len(backbone_state_dict)}")
print(f"  Missing backbone keys: {len(backbone_missing)}")
print(f"  Missing head keys (expected): {len(head_missing)}")

if len(backbone_missing) == 0:
    print("\n✅ SUCCESS! All SSL backbone weights loaded!")
else:
    print(f"\n❌ FAILED! Still missing {len(backbone_missing)} backbone keys")
    print(f"  First 5: {backbone_missing[:5]}")
