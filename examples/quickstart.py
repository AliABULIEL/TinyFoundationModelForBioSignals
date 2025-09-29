#!/usr/bin/env python3
"""
Quick start example for TTM × VitalDB

This script demonstrates the basic usage of the TTM model
with VitalDB data in FastTrack mode.
"""

import torch
import numpy as np
from pathlib import Path

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.ttm_adapter import create_ttm_model
from src.models.datasets import TTMDataset
from src.models.trainers import TrainerClf
from src.eval.calibration import TemperatureScaling, expected_calibration_error


def main():
    """Quick demonstration of TTM training pipeline."""
    
    print("TTM × VitalDB Quick Start Demo")
    print("=" * 50)
    
    # 1. Create synthetic data for demonstration
    print("\n1. Creating synthetic biosignal data...")
    n_samples = 100
    n_channels = 3  # ECG, PPG, ABP
    window_length = 1250  # 10 seconds at 125 Hz
    
    # Generate synthetic windows (normally loaded from VitalDB)
    X = np.random.randn(n_samples, n_channels, window_length).astype(np.float32)
    
    # Generate binary labels (normal/abnormal)
    y = np.random.randint(0, 2, n_samples)
    
    print(f"   Data shape: {X.shape}")
    print(f"   Labels: {np.bincount(y)} (class 0, class 1)")
    
    # 2. Create TTM model in FastTrack mode
    print("\n2. Initializing TTM model (frozen encoder)...")
    
    model_config = {
        'variant': 'ibm/TTM',  # HuggingFace model ID
        'task': 'classification',
        'num_classes': 2,
        'input_channels': n_channels,
        'context_length': window_length,
        
        # FastTrack settings (frozen encoder, linear head)
        'freeze_encoder': True,
        'unfreeze_last_n_blocks': 0,
        'head_type': 'linear',
        'lora': {'enabled': False}
    }
    
    model = create_ttm_model(model_config)
    model.print_parameter_summary()
    
    # 3. Create dataset and dataloader
    print("\n3. Preparing data loaders...")
    
    # Split data
    n_train = int(0.8 * n_samples)
    train_dataset = TTMDataset(
        windows=X[:n_train],
        labels=y[:n_train],
        transform=None
    )
    test_dataset = TTMDataset(
        windows=X[n_train:],
        labels=y[n_train:],
        transform=None
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False
    )
    
    # 4. Train model (FastTrack mode)
    print("\n4. Training in FastTrack mode...")
    print("   (This would normally take ~2 hours with real data)")
    
    trainer = TrainerClf(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        num_classes=2,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_amp=True,
        checkpoint_dir='artifacts/quickstart'
    )
    
    # Train for just 2 epochs for demo
    history = trainer.fit(
        num_epochs=2,
        save_best=True,
        early_stopping_patience=5
    )
    
    print(f"   Final train loss: {history['train_history'][-1]['loss']:.4f}")
    
    # 5. Evaluate and calibrate
    print("\n5. Evaluating model...")
    
    model.eval()
    device = next(model.parameters()).device
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            
            all_probs.append(probs.cpu())
            all_labels.append(labels)
    
    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    
    # Compute metrics
    predictions = (all_probs > 0.5).long()
    accuracy = (predictions == all_labels).float().mean()
    
    print(f"   Test accuracy: {accuracy:.4f}")
    
    # Calibration
    print("\n6. Applying temperature scaling calibration...")
    
    ece_before = expected_calibration_error(all_labels, all_probs)
    print(f"   ECE before calibration: {ece_before:.4f}")
    
    # Note: In practice, you'd use a separate calibration set
    # This is just for demonstration
    temp_scaler = TemperatureScaling()
    
    print("\nQuick Start Complete!")
    print("=" * 50)
    print("\nFor full training with real VitalDB data:")
    print("  bash scripts/run_fasttrack.sh")
    print("\nFor high-accuracy fine-tuning:")
    print("  bash scripts/run_high_accuracy.sh")


if __name__ == "__main__":
    main()
