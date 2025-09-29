#!/usr/bin/env python3
"""
Test script to verify the updated TTM adapter with real TTM
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.models.ttm_adapter import create_ttm_model

print("=" * 70)
print("Testing Updated TTM Adapter with Real TTM")
print("=" * 70)

# Test configuration
config = {
    'variant': 'ibm-granite/granite-timeseries-ttm-r1',
    'task': 'classification',
    'num_classes': 2,
    'input_channels': 3,  # ECG, PPG, ABP
    'context_length': 512,
    'freeze_encoder': True,  # Like in the notebook
    'head_type': 'mlp',
    'head_config': {
        'hidden_dims': [128, 64],
        'dropout': 0.2
    }
}

print("\n1. Creating model with real TTM...")
model = create_ttm_model(config)

print("\n2. Model summary:")
model.print_parameter_summary()

print("\n3. Testing forward pass...")
batch_size = 4
x = torch.randn(batch_size, config['input_channels'], config['context_length'])

with torch.no_grad():
    output = model(x)

print(f"   Input shape: {x.shape}")
print(f"   Output shape: {output.shape}")
assert output.shape == (batch_size, config['num_classes'])
print("   ✓ Forward pass successful!")

print("\n4. Performance expectations with REAL TTM:")
if model.is_using_real_ttm():
    print("   ✅ Using pre-trained TTM foundation model")
    print("   Expected accuracy (frozen backbone): 75-85%")
    print("   Expected accuracy (fine-tuned): 85-92%")
    print("   Training time on CPU: 1-2 hours for FastTrack")
else:
    print("   ⚠️ Using fallback model")
    print("   Expected accuracy: 60-70%")

print("\n" + "=" * 70)
print("READY FOR TRAINING!")
print("=" * 70)
print("\nNext steps:")
print("1. Run FastTrack pipeline (frozen backbone, ~1-2 hours):")
print("   bash scripts/run_fasttrack.sh")
print("\n2. Or fine-tune with unfrozen decoder (better accuracy):")
print("   - Set freeze_encoder: false in configs/model.yaml")
print("   - bash scripts/run_high_accuracy.sh")

print("\n⚠️ Note: Even with real TTM, CPU training will be slow")
print("Consider using a subset of data for initial experiments")
print("=" * 70)
