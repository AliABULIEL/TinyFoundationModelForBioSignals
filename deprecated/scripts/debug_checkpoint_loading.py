#!/usr/bin/env python3
"""Debug script to figure out why SSL checkpoint won't load."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.models.ttm_adapter import TTMAdapter

print("="*70)
print("DEBUGGING SSL CHECKPOINT LOADING")
print("="*70)

# Load SSL checkpoint
print("\n1. Loading SSL checkpoint...")
ckpt = torch.load('artifacts/butppg_ssl/best_model.pt', map_location='cpu', weights_only=False)
ssl_state = ckpt['encoder_state_dict']

# Filter to backbone keys only
ssl_backbone = {}
for k, v in ssl_state.items():
    if 'encoder.backbone' in k:
        ssl_backbone[k] = v

print(f"   SSL checkpoint has {len(ssl_backbone)} backbone keys")
print(f"   Sample SSL keys:")
for k in list(ssl_backbone.keys())[:5]:
    print(f"     {k}")

# Create fine-tuning model
print("\n2. Creating fine-tuning model...")
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

print(f"   Model has {sum(p.numel() for p in model.parameters())} total params")

# Trigger auto-adaptation
print("\n3. Triggering auto-adaptation...")
with torch.no_grad():
    dummy = torch.randn(1, 2, 1024)
    _ = model.get_encoder_output(dummy)
print(f"   Patch size after adaptation: {model.patch_size}")

# Get model keys
print("\n4. Getting model state dict keys...")
model_state = model.state_dict()
model_backbone = {k: v for k, v in model_state.items() if 'backbone' in k and 'encoder' in k}

print(f"   Model has {len(model_backbone)} backbone keys")
print(f"   Sample model keys:")
for k in list(model_backbone.keys())[:5]:
    print(f"     {k}")

# Compare keys
print("\n5. Comparing keys...")
ssl_keys = set(ssl_backbone.keys())
model_keys = set(model_backbone.keys())

matching = ssl_keys & model_keys
ssl_only = ssl_keys - model_keys
model_only = model_keys - ssl_keys

print(f"   Matching keys: {len(matching)}")
print(f"   SSL-only keys: {len(ssl_only)}")
print(f"   Model-only keys: {len(model_only)}")

if ssl_only:
    print(f"\n   First 5 SSL-only keys:")
    for k in list(ssl_only)[:5]:
        print(f"     {k}")

if model_only:
    print(f"\n   First 5 model-only keys:")
    for k in list(model_only)[:5]:
        print(f"     {k}")

# Try loading
print("\n6. Attempting to load SSL weights...")
missing, unexpected = model.load_state_dict(ssl_backbone, strict=False)

backbone_missing = [k for k in missing if 'backbone' in k and 'encoder' in k]
print(f"   Missing backbone keys: {len(backbone_missing)}")
if backbone_missing:
    print(f"   First 5 missing:")
    for k in backbone_missing[:5]:
        print(f"     {k}")

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)
