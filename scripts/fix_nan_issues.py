#!/usr/bin/env python3
"""
Fix NaN issues in training by:
1. Checking data for NaN/Inf
2. Reducing learning rate
3. Adding gradient clipping
4. Checking normalization
"""

import numpy as np
import torch
import yaml
from pathlib import Path

def diagnose_data():
    """Check data for issues that could cause NaN."""
    
    print("="*60)
    print("Diagnosing Data for NaN Issues")
    print("="*60)
    
    issues = []
    
    # Check each split
    for split in ['train', 'val', 'test']:
        data_path = Path(f'artifacts/raw_windows/{split}/{split}_windows.npz')
        
        if not data_path.exists():
            print(f"\n⚠️ {split} data not found")
            continue
            
        print(f"\nChecking {split} data...")
        data = np.load(data_path)
        X = data['data']
        
        # Check for NaN
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            print(f"  ❌ Found {nan_count} NaN values!")
            issues.append(f"{split} has NaN")
        else:
            print(f"  ✅ No NaN values")
        
        # Check for Inf
        inf_count = np.isinf(X).sum()
        if inf_count > 0:
            print(f"  ❌ Found {inf_count} Inf values!")
            issues.append(f"{split} has Inf")
        else:
            print(f"  ✅ No Inf values")
        
        # Check range
        data_min, data_max = X.min(), X.max()
        data_mean, data_std = X.mean(), X.std()
        print(f"  📊 Range: [{data_min:.3f}, {data_max:.3f}]")
        print(f"  📊 Mean: {data_mean:.3f}, Std: {data_std:.3f}")
        
        if abs(data_mean) > 10 or data_std > 100:
            print(f"  ⚠️ Data may not be properly normalized!")
            issues.append(f"{split} normalization issue")
        
        # Check for constant values
        if data_std < 1e-6:
            print(f"  ❌ Data appears to be constant!")
            issues.append(f"{split} is constant")
    
    return issues

def fix_config_for_stability():
    """Update configs for more stable training."""
    
    print("\n" + "="*60)
    print("Fixing Configuration for Stability")
    print("="*60)
    
    # Fix run config
    run_config_path = Path('configs/run.yaml')
    if run_config_path.exists():
        with open(run_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Reduce learning rate
        old_lr = config.get('learning_rate', 0.001)
        config['learning_rate'] = min(old_lr, 1e-4)  # Max 1e-4
        
        # Ensure gradient clipping
        config['gradient_clip'] = 1.0
        
        # Reduce batch size if too large
        config['batch_size'] = min(config.get('batch_size', 32), 16)
        
        # Add warmup
        if 'warmup_epochs' not in config:
            config['warmup_epochs'] = 2
        
        # Save updated config
        with open(run_config_path.with_suffix('.yaml.backup'), 'w') as f:
            yaml.dump(config, f)
        
        with open(run_config_path, 'w') as f:
            yaml.dump(config, f)
        
        print(f"✅ Updated run config:")
        print(f"   Learning rate: {old_lr} → {config['learning_rate']}")
        print(f"   Gradient clip: {config['gradient_clip']}")
        print(f"   Batch size: {config['batch_size']}")

def check_labels():
    """Check if labels are properly distributed."""
    
    print("\n" + "="*60)
    print("Checking Label Distribution")
    print("="*60)
    
    for split in ['train', 'val']:
        data_path = Path(f'artifacts/raw_windows/{split}/{split}_windows.npz')
        
        if not data_path.exists():
            continue
            
        data = np.load(data_path)
        if 'labels' in data:
            labels = data['labels']
            unique, counts = np.unique(labels, return_counts=True)
            
            print(f"\n{split} labels:")
            for label, count in zip(unique, counts):
                percentage = 100 * count / len(labels)
                print(f"  Class {label}: {count} ({percentage:.1f}%)")
            
            if len(unique) == 1:
                print(f"  ⚠️ WARNING: Only one class in {split}!")
                print(f"     This will cause training issues!")

def create_stable_config():
    """Create a stable training configuration."""
    
    stable_config = """# Stable training configuration to prevent NaN

# Optimizer settings
optimizer: "adamw"
learning_rate: 5e-5  # Very small learning rate
weight_decay: 0.01

# Training settings
batch_size: 16  # Smaller batch size
num_epochs: 20
gradient_clip: 0.5  # Aggressive gradient clipping

# Learning rate schedule
scheduler: "cosine"
warmup_epochs: 2  # Warmup to prevent early instability

# Mixed precision (disable if causing issues)
use_amp: false  # Disable AMP to avoid numerical issues

# Early stopping
early_stopping_patience: 5

# Logging
log_interval: 100  # Less frequent logging
device: "cuda"

# Loss settings (for stability)
loss_type: "cross_entropy"
label_smoothing: 0.1  # Helps with overconfidence

# Regularization
dropout: 0.2
"""
    
    with open('configs/stable_run.yaml', 'w') as f:
        f.write(stable_config)
    
    print("\n✅ Created stable config: configs/stable_run.yaml")

def main():
    # Diagnose issues
    issues = diagnose_data()
    
    # Check labels
    check_labels()
    
    # Fix configuration
    fix_config_for_stability()
    
    # Create stable config
    create_stable_config()
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    if issues:
        print("\n⚠️ Found issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ No data issues found")
    
    print("\n📝 Recommendations:")
    print("1. Use the stable config: --run-yaml configs/stable_run.yaml")
    print("2. Monitor loss carefully in first epoch")
    print("3. If NaN persists, reduce learning rate further")
    print("4. Check that data is properly normalized")
    
    print("\n🚀 To train with stable settings:")
    print("python scripts/ttm_vitaldb.py train \\")
    print("    --model-yaml configs/model.yaml \\")
    print("    --run-yaml configs/stable_run.yaml \\")
    print("    --split-file configs/splits/splits_full.json \\")
    print("    --task clf \\")
    print("    --outdir artifacts/raw_windows \\")
    print("    --out artifacts/model_stable \\")
    print("    --fasttrack")

if __name__ == "__main__":
    main()
