#!/usr/bin/env python3
"""Multi-Task Fine-tuning Script for BUT-PPG.

Supports all 7 clinical tasks:
1. quality - Binary classification (0=poor, 1=good)
2. hr_estimation - Regression (heart rate in BPM)
3. motion - Multi-class classification (8 classes)
4. bp_systolic - Regression (systolic BP in mmHg)
5. bp_diastolic - Regression (diastolic BP in mmHg)
6. spo2 - Regression (SpO2 percentage)
7. glycaemia - Regression (blood glucose in mmol/l)

Usage:
    # Heart rate regression
    python scripts/finetune_butppg_multitask.py \
        --pretrained artifacts/hybrid_full_corrected/stage2_butppg_quality_ssl/best_model.pt \
        --task hr_estimation \
        --epochs 30

    # Blood pressure regression
    python scripts/finetune_butppg_multitask.py \
        --pretrained artifacts/hybrid_full_corrected/stage2_butppg_quality_ssl/best_model.pt \
        --task bp_systolic \
        --epochs 30
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm

from src.models.ttm_adapter import TTMAdapter
from src.models.channel_utils import unfreeze_last_n_blocks


# Task configuration
TASK_CONFIG = {
    'quality': {
        'type': 'classification',
        'num_classes': 2,
        'label_field': 'quality',
        'metric': 'accuracy',
        'target': 0.85,  # Target AUROC
    },
    'hr_estimation': {
        'type': 'regression',
        'output_dim': 1,
        'label_field': 'hr',
        'metric': 'mae',
        'target': 2.0,  # Target MAE (bpm)
    },
    'motion': {
        'type': 'classification',
        'num_classes': 8,
        'label_field': 'motion',
        'metric': 'accuracy',
        'target': 0.85,
    },
    'bp_systolic': {
        'type': 'regression',
        'output_dim': 1,
        'label_field': 'bp_systolic',
        'metric': 'mae',
        'target': 5.0,  # Target MAE (mmHg)
    },
    'bp_diastolic': {
        'type': 'regression',
        'output_dim': 1,
        'label_field': 'bp_diastolic',
        'metric': 'mae',
        'target': 5.0,  # Target MAE (mmHg)
    },
    'spo2': {
        'type': 'regression',
        'output_dim': 1,
        'label_field': 'spo2',
        'metric': 'mae',
        'target': 2.0,  # Target MAE (%)
    },
    'glycaemia': {
        'type': 'regression',
        'output_dim': 1,
        'label_field': 'glycaemia',
        'metric': 'mae',
        'target': 1.0,  # Target MAE (mmol/l)
    },
}


class BUTPPGMultiTaskDataset(Dataset):
    """Dataset for BUT-PPG with support for all clinical tasks."""

    def __init__(self, data_dir: Path, task: str, target_length: int = None):
        """Initialize BUT-PPG dataset.

        Args:
            data_dir: Path to directory containing window_*.npz files
            task: Task name (quality, hr_estimation, motion, etc.)
            target_length: If specified, resize signals to this length
        """
        print(f"Loading BUT-PPG data from: {data_dir}")

        if task not in TASK_CONFIG:
            raise ValueError(f"Unknown task: {task}. Available: {list(TASK_CONFIG.keys())}")

        self.task = task
        self.task_config = TASK_CONFIG[task]
        self.label_field = self.task_config['label_field']
        self.target_length = target_length

        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # Find all window files
        window_files = sorted(data_dir.glob('window_*.npz'))
        if len(window_files) == 0:
            raise FileNotFoundError(f"No window files found in {data_dir}")

        # Load all windows
        signals_list = []
        labels_list = []

        for window_file in window_files:
            data = np.load(window_file)

            # Load signal [2, 1024]
            if 'signal' not in data:
                raise KeyError(f"Expected 'signal' key in {window_file}")

            signal = data['signal']  # [2, T]

            # Load label for this task
            if self.label_field in data:
                label_val = data[self.label_field]

                # Convert 0-d numpy array to scalar
                if isinstance(label_val, np.ndarray):
                    label_val = label_val.item() if label_val.size == 1 else float(label_val)

                # Skip samples with missing labels
                if np.isnan(label_val) or label_val == -1:
                    continue

                # Convert to appropriate type
                if self.task_config['type'] == 'classification':
                    label = int(label_val)
                else:  # regression
                    label = float(label_val)
            else:
                # Skip samples without this label
                continue

            signals_list.append(signal)
            labels_list.append(label)

        if len(signals_list) == 0:
            raise ValueError(f"No valid samples found for task '{task}' in {data_dir}")

        # Stack into tensors
        self.signals = torch.from_numpy(np.stack(signals_list, axis=0)).float()  # [N, 2, T]

        if self.task_config['type'] == 'classification':
            self.labels = torch.tensor(labels_list, dtype=torch.long)  # [N]
        else:  # regression
            self.labels = torch.tensor(labels_list, dtype=torch.float32)  # [N]

        # Resize if needed
        N, C, T = self.signals.shape
        if self.target_length is not None and T != self.target_length:
            print(f"  ⚠️  Resizing signals from {T} to {self.target_length} samples")
            self.signals = F.interpolate(
                self.signals,
                size=self.target_length,
                mode='linear',
                align_corners=False
            )

        # Print dataset info
        task_type = self.task_config['type']
        if task_type == 'classification':
            unique_labels, counts = torch.unique(self.labels, return_counts=True)
            print(f"  Loaded {len(self)} samples for task '{task}' ({task_type})")
            for label, count in zip(unique_labels, counts):
                pct = (count.item() / len(self)) * 100
                print(f"    - Class {label.item()}: {count.item()} ({pct:.1f}%)")
        else:  # regression
            print(f"  Loaded {len(self)} samples for task '{task}' ({task_type})")
            print(f"    - Label mean: {self.labels.mean():.2f}")
            print(f"    - Label std: {self.labels.std():.2f}")
            print(f"    - Label range: [{self.labels.min():.2f}, {self.labels.max():.2f}]")

        print(f"    - Shape: {self.signals.shape}")

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]


def create_model(task: str, pretrained_path: str, device: str,
                 patch_size_override: int = None, context_length_override: int = None) -> nn.Module:
    """Create model for the specified task using the SAME process as SSL training.

    CRITICAL: SSL checkpoint was trained with IBM pretrained that auto-adapted.
    We must replicate this EXACT process to load weights correctly.

    Args:
        task: Task name
        pretrained_path: Path to pretrained checkpoint
        device: Device to load model on
        patch_size_override: IGNORED - detected from checkpoint
        context_length_override: IGNORED - detected from checkpoint

    Returns:
        Model ready for fine-tuning
    """
    task_config = TASK_CONFIG[task]
    task_type = task_config['type']

    if task_type == 'classification':
        num_classes = task_config['num_classes']
        output_type = 'classification'
        print(f"\n✓ Creating {task_type} model for '{task}' ({num_classes} classes)")
    else:  # regression
        num_classes = 1  # Single output for regression
        output_type = 'regression'
        print(f"\n✓ Creating {task_type} model for '{task}' (1 output)")

    # =========================================================================
    # STEP 1: Detect actual architecture from SSL checkpoint weights
    # =========================================================================
    checkpoint = torch.load(pretrained_path, map_location='cpu')

    if 'encoder_state_dict' in checkpoint:
        state_dict = checkpoint['encoder_state_dict']
        print("  ✓ Found SSL checkpoint (encoder_state_dict)")
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("  ✓ Found model checkpoint (model_state_dict)")
    else:
        state_dict = checkpoint
        print("  ✓ Using raw checkpoint as state_dict")

    # Detect d_model and actual patch_size from patcher weight
    patcher_keys = [k for k in state_dict.keys() if 'patcher.weight' in k and 'encoder' in k]
    if not patcher_keys:
        raise ValueError("Cannot find patcher.weight in checkpoint - cannot detect architecture")

    patcher_weight = state_dict[patcher_keys[0]]
    d_model = patcher_weight.shape[0]  # Output dimension
    actual_patch_size = patcher_weight.shape[1]  # Input dimension per channel

    print(f"  ✓ Detected from checkpoint weights:")
    print(f"    Key: {patcher_keys[0]}")
    print(f"    Shape: {patcher_weight.shape}")
    print(f"    → d_model: {d_model}")
    print(f"    → actual_patch_size: {actual_patch_size}")

    # Get context_length from checkpoint config or use default
    if 'config' in checkpoint:
        context_length = checkpoint['config'].get('context_length', 1024)
        print(f"    → context_length: {context_length} (from checkpoint config)")
    else:
        context_length = 1024
        print(f"    → context_length: {context_length} (default)")

    # Calculate number of patches
    num_patches = context_length // actual_patch_size
    print(f"    → num_patches: {num_patches} ({context_length}/{actual_patch_size})")

    # =========================================================================
    # STEP 2: Create fresh model matching SSL checkpoint architecture
    # =========================================================================
    # CRITICAL: SSL checkpoint already has trained weights, so we DON'T need
    # IBM pretrained anymore. Just create a model matching the SSL architecture.
    print(f"\n  ℹ️  Creating fresh model matching SSL checkpoint architecture")
    print(f"     Architecture: context={context_length}, d_model={d_model}, patch={actual_patch_size}")

    print(f"\nCreating TTMAdapter:")
    print(f"  Task: {task_type}")
    print(f"  Input channels: 2 (PPG + ECG)")
    print(f"  Context length: {context_length}")
    print(f"  Patch size: {actual_patch_size}")
    print(f"  d_model: {d_model}")
    print(f"  Loading pretrained: NO (will load SSL checkpoint instead)")

    model = TTMAdapter(
        variant='ibm-granite/granite-timeseries-ttm-r1',
        task=task_type,
        num_classes=num_classes if task_type == 'classification' else None,
        input_channels=2,
        context_length=context_length,
        patch_size=actual_patch_size,  # Use ACTUAL patch size from SSL checkpoint
        d_model=d_model,
        use_pretrained=False,  # Don't load IBM pretrained
        freeze_encoder=True  # Freeze for fine-tuning
    )

    # Move to device
    model = model.to(device)

    # =========================================================================
    # STEP 3: Load SSL checkpoint weights (backbone only)
    # =========================================================================
    print("\n📦 Loading SSL backbone encoder weights...")

    backbone_state_dict = {}
    skipped_decoder = 0
    skipped_head = 0

    for key, value in state_dict.items():
        # Skip SSL decoder (different from fine-tuning decoder)
        if 'encoder.decoder' in key or 'decoder_state' in key:
            skipped_decoder += 1
            continue
        # Skip SSL head (task-specific)
        elif 'encoder.head' in key or 'head.' in key:
            skipped_head += 1
            continue
        # Include encoder.backbone weights
        elif 'encoder.backbone' in key:
            # Add with original key
            backbone_state_dict[key] = value
            # Also add with stripped prefix (model has both)
            stripped_key = key.replace('encoder.', '', 1)
            backbone_state_dict[stripped_key] = value
        # Include plain backbone.* keys
        elif key.startswith('backbone.') and 'decoder' not in key:
            backbone_state_dict[key] = value

    print(f"  Prepared {len(backbone_state_dict)} weight entries")
    print(f"  Skipped SSL decoder: {skipped_decoder} keys")
    print(f"  Skipped SSL head: {skipped_head} keys")

    # Load backbone weights
    missing_keys, unexpected_keys = model.load_state_dict(backbone_state_dict, strict=False)

    # Verify loading success
    backbone_missing = [k for k in missing_keys if 'backbone' in k and 'encoder' in k]

    if len(backbone_missing) == 0:
        print(f"\n✅ SSL encoder weights loaded successfully!")
        print(f"  ✓ All backbone keys matched")
    else:
        print(f"\n⚠️ WARNING: SSL encoder loading FAILED:")
        print(f"  Missing backbone keys: {len(backbone_missing)}")
        print(f"  Model will train from scratch!")
        if len(backbone_missing) < 10:
            for k in backbone_missing[:5]:
                print(f"    - {k}")

    print(f"✓ Initialized new {task_type} head")

    return model


def train_epoch(model, dataloader, optimizer, scaler, task_config, device, use_amp=True):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_mae = 0.0  # For regression

    task_type = task_config['type']

    # Loss function
    if task_type == 'classification':
        criterion = nn.CrossEntropyLoss()
    else:  # regression
        criterion = nn.MSELoss()

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for signals, labels in pbar:
        signals = signals.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            outputs = model(signals)

            if task_type == 'classification':
                loss = criterion(outputs, labels)
            else:  # regression
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Compute metrics
        total_loss += loss.item() * len(signals)
        total_samples += len(signals)

        if task_type == 'classification':
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            acc = 100.0 * total_correct / total_samples
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.2f}%'})
        else:  # regression
            with torch.no_grad():
                mae = (outputs - labels).abs().mean().item()
                total_mae += mae * len(signals)
            avg_mae = total_mae / total_samples
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'mae': f'{avg_mae:.2f}'})

    avg_loss = total_loss / total_samples

    if task_type == 'classification':
        accuracy = 100.0 * total_correct / total_samples
        return avg_loss, accuracy
    else:  # regression
        avg_mae = total_mae / total_samples
        return avg_loss, avg_mae


def validate(model, dataloader, task_config, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_mae = 0.0

    task_type = task_config['type']

    # Loss function
    if task_type == 'classification':
        criterion = nn.CrossEntropyLoss()
    else:  # regression
        criterion = nn.MSELoss()

    with torch.no_grad():
        for signals, labels in dataloader:
            signals = signals.to(device)
            labels = labels.to(device)

            outputs = model(signals)

            if task_type == 'classification':
                loss = criterion(outputs, labels)
                preds = outputs.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
            else:  # regression
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels)
                mae = (outputs - labels).abs().mean().item()
                total_mae += mae * len(signals)

            total_loss += loss.item() * len(signals)
            total_samples += len(signals)

    avg_loss = total_loss / total_samples

    if task_type == 'classification':
        accuracy = 100.0 * total_correct / total_samples
        return avg_loss, accuracy
    else:  # regression
        avg_mae = total_mae / total_samples
        return avg_loss, avg_mae


def main():
    parser = argparse.ArgumentParser(description="Multi-task fine-tuning for BUT-PPG")

    # Task selection
    parser.add_argument('--task', type=str, required=True, choices=list(TASK_CONFIG.keys()),
                       help='Task to fine-tune on')

    # Model and data
    parser.add_argument('--pretrained', type=str, required=True,
                       help='Path to pretrained checkpoint')
    parser.add_argument('--data-dir', type=str,
                       default='data/processed/butppg/windows_with_labels',
                       help='Data directory')

    # Training
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')

    # Output
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: artifacts/butppg_<task>)')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)

    # Architecture overrides (to match SSL checkpoint)
    parser.add_argument('--context-length', type=int, default=None,
                       help='Override context length (should match SSL checkpoint)')
    parser.add_argument('--patch-size', type=int, default=None,
                       help='Override patch size (should match SSL checkpoint)')

    args = parser.parse_args()

    # Set output dir
    if args.output_dir is None:
        args.output_dir = f'artifacts/butppg_{args.task}'

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Print config
    task_config = TASK_CONFIG[args.task]
    print("\n" + "="*80)
    print(f"MULTI-TASK FINE-TUNING: {args.task.upper()}")
    print("="*80)
    print(f"Task type: {task_config['type']}")
    print(f"Label field: {task_config['label_field']}")
    print(f"Target metric: {task_config['metric']} ≤ {task_config['target']}")
    print(f"Pretrained: {args.pretrained}")
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("="*80)

    # Create datasets
    print("\n📊 Creating dataloaders...")
    train_dataset = BUTPPGMultiTaskDataset(
        Path(args.data_dir) / 'train',
        task=args.task,
        target_length=1024
    )
    val_dataset = BUTPPGMultiTaskDataset(
        Path(args.data_dir) / 'val',
        task=args.task,
        target_length=1024
    )
    test_dataset = BUTPPGMultiTaskDataset(
        Path(args.data_dir) / 'test',
        task=args.task,
        target_length=1024
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Create model (with architecture overrides if provided)
    model = create_model(
        args.task,
        args.pretrained,
        args.device,
        patch_size_override=args.patch_size,
        context_length_override=args.context_length
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler()

    # Training loop
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)

    best_metric = float('inf') if task_config['type'] == 'regression' else 0.0

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # Train
        train_loss, train_metric = train_epoch(
            model, train_loader, optimizer, scaler, task_config, args.device
        )

        # Validate
        val_loss, val_metric = validate(model, val_loader, task_config, args.device)

        epoch_time = time.time() - start_time

        # Print results
        metric_name = task_config['metric'].upper()
        if task_config['type'] == 'classification':
            print(f"Epoch {epoch:2d}/{args.epochs} ({epoch_time:.1f}s)")
            print(f"  Train - Loss: {train_loss:.4f}, {metric_name}: {train_metric:.2f}%")
            print(f"  Val   - Loss: {val_loss:.4f}, {metric_name}: {val_metric:.2f}%")

            # Save best model (higher accuracy is better)
            if val_metric > best_metric:
                best_metric = val_metric
                checkpoint_path = output_dir / 'best_model.pt'
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  ✓ Best model saved ({metric_name}: {val_metric:.2f}%)")
        else:  # regression
            print(f"Epoch {epoch:2d}/{args.epochs} ({epoch_time:.1f}s)")
            print(f"  Train - Loss: {train_loss:.4f}, {metric_name}: {train_metric:.2f}")
            print(f"  Val   - Loss: {val_loss:.4f}, {metric_name}: {val_metric:.2f}")

            # Save best model (lower MAE is better)
            if val_metric < best_metric:
                best_metric = val_metric
                checkpoint_path = output_dir / 'best_model.pt'
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  ✓ Best model saved ({metric_name}: {val_metric:.2f})")
        print()

    # Final evaluation
    print("="*80)
    print("FINAL EVALUATION ON TEST SET")
    print("="*80)

    # Load best model
    model.load_state_dict(torch.load(output_dir / 'best_model.pt'))
    test_loss, test_metric = validate(model, test_loader, task_config, args.device)

    metric_name = task_config['metric'].upper()
    target = task_config['target']

    if task_config['type'] == 'classification':
        print(f"Test {metric_name}: {test_metric:.2f}%")
        target_met = test_metric >= (target * 100)
    else:  # regression
        print(f"Test {metric_name}: {test_metric:.2f}")
        target_met = test_metric <= target

    print(f"Target: {target}")
    print(f"Target Met: {'✅ YES' if target_met else '❌ NO'}")
    print("="*80)

    # Save results
    results = {
        'task': args.task,
        'task_type': task_config['type'],
        'metric': metric_name.lower(),
        f'test_{metric_name.lower()}': float(test_metric),
        f'best_val_{metric_name.lower()}': float(best_metric),
        'target': target,
        'target_met': target_met,
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_dir / 'results.json'}")


if __name__ == '__main__':
    main()
