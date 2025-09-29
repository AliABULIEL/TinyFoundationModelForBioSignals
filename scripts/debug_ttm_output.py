#!/usr/bin/env python3
"""
Debug script to understand TTM output shapes
"""

import torch
from tsfm_public import get_model

print("=" * 70)
print("Debugging TTM Output Shapes")
print("=" * 70)

# Load the real TTM model
model = get_model(
    "ibm-granite/granite-timeseries-ttm-r1",
    context_length=512,
    prediction_length=96,
    num_input_channels=3,
    decoder_mode="mix_channel"
)

# Test input
batch_size = 4
x = torch.randn(batch_size, 512, 3)  # [batch, time, channels]

print(f"\nInput shape: {x.shape}")

# Get backbone output
with torch.no_grad():
    # Full model output
    full_output = model(x)
    print(f"\nFull model output type: {type(full_output)}")
    if hasattr(full_output, '__dict__'):
        print(f"Full model output attributes: {full_output.__dict__.keys()}")
    
    # Try backbone only
    if hasattr(model, 'backbone'):
        backbone_output = model.backbone(x)
        print(f"\nBackbone output type: {type(backbone_output)}")
        if hasattr(backbone_output, 'shape'):
            print(f"Backbone output shape: {backbone_output.shape}")
        elif hasattr(backbone_output, '__dict__'):
            print(f"Backbone output attributes: {backbone_output.__dict__.keys()}")
            if hasattr(backbone_output, 'last_hidden_state'):
                print(f"  last_hidden_state shape: {backbone_output.last_hidden_state.shape}")
            if hasattr(backbone_output, 'encoder_output'):
                print(f"  encoder_output shape: {backbone_output.encoder_output.shape}")
    
    # Try encoder if available
    if hasattr(model, 'encoder'):
        encoder_output = model.encoder(x)
        print(f"\nEncoder output type: {type(encoder_output)}")
        if hasattr(encoder_output, 'shape'):
            print(f"Encoder output shape: {encoder_output.shape}")
        elif hasattr(encoder_output, '__dict__'):
            print(f"Encoder output attributes: {encoder_output.__dict__.keys()}")

print("\n" + "=" * 70)
print("Solution: We need to extract the right features from TTM output")
print("=" * 70)
