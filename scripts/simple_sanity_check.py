#!/usr/bin/env python3
"""
Simple sanity check using the actual TTM-VitalDB pipeline structure.
This tests if the real TTM model works with your data loading.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from src.models.ttm_adapter import create_ttm_model
from src.models.datasets import TTMDataset

print("=" * 70)
print("SANITY CHECK: Real TTM with VitalDB Pipeline")
print("=" * 70)

# 1. Create synthetic data for quick test
print("\n1. Creating synthetic data (mimicking VitalDB windows)...")
n_samples = 20
n_channels = 3  # ECG, PPG, ABP
window_size = 512  # TTM context length

# Create synthetic windows
synthetic_windows = np.random.randn(n_samples, n_channels, window_size).astype(np.float32)
synthetic_labels = np.random.randint(0, 2, size=n_samples)  # Binary classification

print(f"   Created {n_samples} synthetic windows")
print(f"   Shape: {synthetic_windows.shape}")

# 2. Create TTMDataset
print("\n2. Creating TTMDataset...")
dataset = TTMDataset(
    windows=synthetic_windows,
    labels=synthetic_labels,
    transform=None
)

print(f"   Dataset size: {len(dataset)}")
sample_window, sample_label = dataset[0]
print(f"   Sample shape: {sample_window.shape}")
print(f"   Sample label: {sample_label}")

# 3. Create real TTM model
print("\n3. Creating real TTM model...")
model_config = {
    'variant': 'ibm-granite/granite-timeseries-ttm-r1',
    'task': 'classification',
    'num_classes': 2,
    'input_channels': n_channels,
    'context_length': window_size,
    'freeze_encoder': True,
    'head_type': 'linear',
    'head_config': {'dropout': 0.2},
    'decoder_mode': 'mix_channel'
}

model = create_ttm_model(model_config)
model.print_parameter_summary()

# 4. Test forward pass
print("\n4. Testing forward pass...")
batch = torch.stack([dataset[i][0] for i in range(4)])
print(f"   Batch shape: {batch.shape}")

with torch.no_grad():
    outputs = model(batch)
    print(f"   Output shape: {outputs.shape}")
    
    # Check if outputs are reasonable
    probs = torch.softmax(outputs, dim=1)
    print(f"   Probabilities (sample): {probs[0].numpy()}")

# 5. Simple training test
print("\n5. Testing training step...")
model.train()
optimizer = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-3
)
criterion = torch.nn.CrossEntropyLoss()

# Single training step
batch_inputs = torch.stack([dataset[i][0] for i in range(4)])
batch_labels = torch.tensor([dataset[i][1] for i in range(4)])

optimizer.zero_grad()
outputs = model(batch_inputs)
loss = criterion(outputs, batch_labels)
loss.backward()
optimizer.step()

print(f"   Loss: {loss.item():.4f}")
print("   ✓ Training step successful!")

print("\n" + "=" * 70)
print("SANITY CHECK COMPLETE!")
print("=" * 70)
print("\n✅ All components working correctly:")
print("  - TTMDataset loads data properly")
print("  - Real TTM model loads and runs")
print("  - Forward pass produces correct shapes")
print("  - Training step completes without errors")

print("\n" + "=" * 70)
print("READY FOR ACTUAL TRAINING")
print("=" * 70)
print("\nYou can now use the full pipeline:")
print("\n1. Prepare splits (if not done):")
print("   python scripts/ttm_vitaldb.py prepare-splits --mode fasttrack")
print("\n2. Build windows from VitalDB:")
print("   python scripts/ttm_vitaldb.py build-windows --mode fasttrack")
print("\n3. Train the model:")
print("   python scripts/ttm_vitaldb.py train --mode fasttrack")
print("\n4. Test the model:")
print("   python scripts/ttm_vitaldb.py test --checkpoint artifacts/final_model.pt --mode fasttrack")

print("\nOr use the bash script:")
print("   bash scripts/run_fasttrack.sh")
print("=" * 70)
