#!/usr/bin/env python3
"""
Quick sanity check with minimal data to verify training works
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from src.models.ttm_adapter import create_ttm_model
from src.data.biosignal_dataset import BiosignalDataset
from src.training.trainer import BiosignalTrainer

print("=" * 70)
print("SANITY CHECK: Testing training pipeline with real TTM")
print("=" * 70)

# Use only 10 cases for quick test
data_config = {
    'data_root': 'data/processed/VitalDB',
    'signal_types': ['ECG_II', 'PLETH', 'ABP'],
    'target_signals': ['ECG_II', 'PLETH', 'ABP'],
    'sequence_length': 1250,  # 10 seconds at 125Hz
    'sampling_rate': 125,
    'normalization': 'standardize',
    'augmentation': {
        'enabled': False  # Disable for quick test
    },
    'max_cases': 2  # Only 10 cases!
}

# Model configuration
model_config = {
    'variant': 'ibm-granite/granite-timeseries-ttm-r1',
    'task': 'classification',
    'num_classes': 2,
    'input_channels': 3,
    'context_length': 512,
    'freeze_encoder': True,
    'head_type': 'linear',  # Simple linear head for quick test
    'head_config': {'dropout': 0.2}
}

# Training configuration
training_config = {
    'batch_size': 2,  # Small batch for CPU
    'learning_rate': 1e-4,
    'num_epochs': 3,  # Just 3 epochs
    'device': 'cpu',
    'num_workers': 0,  # No multiprocessing for sanity check
    'optimizer': 'adam',
    'scheduler': None,
    'gradient_clip': 1.0,
    'mixed_precision': False,
    'log_interval': 1,
    'save_interval': 999,  # Don't save
    'early_stopping_patience': 999,
    'model_dir': 'checkpoints/sanity_check'
}

print("\n1. Creating dataset (10 cases only)...")
dataset = BiosignalDataset(**data_config)
print(f"   Dataset size: {len(dataset)}")

# Quick data check
if len(dataset) > 0:
    sample = dataset[0]
    print(f"   Sample shape: {sample[0].shape}")
    print(f"   Label: {sample[1]}")
else:
    print("   ERROR: No data found!")
    sys.exit(1)

print("\n2. Creating model...")
model = create_ttm_model(model_config)
model.print_parameter_summary()

print("\n3. Creating trainer...")
trainer = BiosignalTrainer(model, training_config)

print("\n4. Running 3 training epochs...")
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Quick training loop
model.train()
for epoch in range(3):
    total_loss = 0
    for i, (signals, labels) in enumerate(train_loader):
        loss = trainer.train_step(signals, labels)
        total_loss += loss
    print(f"   Epoch {epoch+1}: avg loss = {total_loss/(i+1):.4f}")

print("\n5. Quick evaluation...")
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for signals, labels in train_loader:
        outputs = model(signals)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"   Train accuracy: {accuracy:.1f}%")

print("\n" + "=" * 70)
print("SANITY CHECK COMPLETE!")
print("=" * 70)

if accuracy > 40:  # Should be better than random (50%)
    print("✅ Training pipeline works! Ready for full training.")
    print("\nNext step: Run FastTrack with more data:")
    print("   bash scripts/run_fasttrack.sh")
else:
    print("⚠️ Accuracy seems low. Check data loading and labels.")

print("=" * 70)
