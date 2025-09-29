#!/usr/bin/env python3
"""
Ultra-fast sanity check with just 2 cases - runs in 10 seconds
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from src.models.ttm_adapter import create_ttm_model

print("="*60)
print("ULTRA-FAST SANITY CHECK (2 cases, 10 seconds)")
print("="*60)

# Create 2 fake "cases" of biosignal data
print("\n1. Creating 2 synthetic cases...")
X = torch.randn(2, 3, 512)  # 2 cases, 3 channels (ECG/PPG/ABP), 512 samples
y = torch.tensor([0, 1])    # Binary labels

print(f"   Data shape: {X.shape}")

# Create TTM model
print("\n2. Loading real TTM...")
model = create_ttm_model({
    'variant': 'ibm-granite/granite-timeseries-ttm-r1',
    'task': 'classification',
    'num_classes': 2,
    'input_channels': 3,
    'context_length': 512,
    'freeze_encoder': True,
    'head_type': 'linear'
})

# Test forward pass
print("\n3. Testing forward pass...")
with torch.no_grad():
    output = model(X)
print(f"   Output shape: {output.shape}")
print(f"   Output values: {output}")

# Quick training test
print("\n4. Testing one training step...")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

output = model(X)
loss = loss_fn(output, y)
loss.backward()
optimizer.step()

print(f"   Loss: {loss.item():.4f}")

print("\n" + "="*60)
print("✅ SUCCESS! TTM is working!")
print("="*60)
print("\nReady for real training. Run:")
print("python3 scripts/ttm_vitaldb.py train --mode fasttrack")
print("="*60)
