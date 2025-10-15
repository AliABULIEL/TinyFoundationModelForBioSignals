#!/usr/bin/env python3
"""Fine-tuning Script for BUT-PPG Quality Classification.

This script fine-tunes a 2-channel SSL pretrained model on 5-channel BUT-PPG data
for PPG quality classification (good vs poor).

Strategy:
1. Load 2-channel pretrained checkpoint
2. Inflate to 5 channels (ACC_X, ACC_Y, ACC_Z, PPG, ECG)
3. Staged unfreezing:
   - Stage 1 (3-5 epochs): Head-only training
   - Stage 2 (remaining epochs): Unfreeze last N encoder blocks
   - Stage 3 (optional): Full fine-tuning at very low LR
4. Monitor validation accuracy and save best model

Usage:
    # Quick test (1 epoch)
    python scripts/finetune_butppg.py \
        --pretrained artifacts/foundation_model/best_model.pt \
        --data-dir data/processed/butppg/windows \
        --pretrain-channels 2 \
        --finetune-channels 5 \
        --unfreeze-last-n 2 \
        --epochs 1 \
        --lr 2e-5 \
        --output-dir artifacts/but_ppg_finetuned

    # Full training (30 epochs with staged unfreezing)
    python scripts/finetune_butppg.py \
        --pretrained artifacts/foundation_model/best_model.pt \
        --data-dir data/processed/butppg/windows \
        --epochs 30 \
        --head-only-epochs 5 \
        --unfreeze-last-n 2 \
        --batch-size 64 \
        --output-dir artifacts/but_ppg_finetuned
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time
import warnings
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm

from src.models.channel_utils import (
    load_pretrained_with_channel_inflate,
    unfreeze_last_n_blocks
)


class BUTPPGDataset(Dataset):
    """Dataset for BUT-PPG quality classification.

    Expected data format:
    - signals: [N, 5, 1024] - 5 channels (ACC_X, ACC_Y, ACC_Z, PPG, ECG)
    - labels: [N] - binary labels (0=poor, 1=good)

    Channels:
    0: ACC_X (accelerometer X-axis)
    1: ACC_Y (accelerometer Y-axis)
    2: ACC_Z (accelerometer Z-axis)
    3: PPG (photoplethysmogram)
    4: ECG (electrocardiogram)
    """

    def __init__(self, data_file: Path, normalize: bool = True):
        """Initialize BUT-PPG dataset.

        Args:
            data_file: Path to .npz file containing 'signals' and 'labels'
            normalize: Whether to apply z-score normalization per channel
        """
        print(f"Loading BUT-PPG data from: {data_file}")
        data = np.load(data_file)

        # Support both 'signals' and 'data' keys (data prep inconsistency)
        if 'signals' in data:
            signals_array = data['signals']
        elif 'data' in data:
            signals_array = data['data']
        else:
            raise KeyError(f"Expected 'signals' or 'data' key in {data_file}, found: {list(data.keys())}")

        self.signals = torch.from_numpy(signals_array).float()
        self.labels = torch.from_numpy(data['labels']).long()     # [N]

        # Handle different axis orders: [N, T, C] vs [N, C, T]
        if len(self.signals.shape) == 3:
            # Check if we need to transpose
            if self.signals.shape[1] == 1024 and self.signals.shape[2] == 5:
                # Data is [N, T, C] but we need [N, C, T]
                print(f"  ⚠️  Transposing from [N, T, C] to [N, C, T]")
                self.signals = self.signals.transpose(1, 2)  # [N, 1024, 5] → [N, 5, 1024]

        # Validate shapes
        N, C, T = self.signals.shape
        assert C == 5, f"Expected 5 channels, got {C}"
        assert T == 1024, f"Expected 1024 timesteps, got {T} (make sure data was prepared with --window-size 1024)"
        assert len(self.labels) == N, f"Label count mismatch: {len(self.labels)} vs {N}"
        
        # Normalize if requested
        if normalize:
            # Z-score normalization per channel
            for c in range(C):
                channel_data = self.signals[:, c, :]  # [N, T]
                mean = channel_data.mean()
                std = channel_data.std()
                if std > 0:
                    self.signals[:, c, :] = (channel_data - mean) / std
        
        # Print dataset info
        n_good = (self.labels == 1).sum().item()
        n_poor = (self.labels == 0).sum().item()
        print(f"  Loaded {N} samples:")
        print(f"    - Good quality: {n_good} ({n_good/N*100:.1f}%)")
        print(f"    - Poor quality: {n_poor} ({n_poor/N*100:.1f}%)")
        print(f"    - Shape: {self.signals.shape}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]


def create_dataloaders(
    data_dir: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    use_val: bool = True
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """Create dataloaders for BUT-PPG.

    Supports two directory structures:
    1. Flat: data_dir/{train,val,test}.npz
    2. Nested: data_dir/{train,val,test}/data.npz (from prepare_all_data.py)

    Args:
        data_dir: Directory containing BUT-PPG data
        batch_size: Batch size for dataloaders
        num_workers: Number of data loading workers
        use_val: Whether to create validation loader

    Returns:
        train_loader, val_loader, test_loader
    """
    data_dir = Path(data_dir)

    # Create datasets
    print("\n" + "=" * 70)
    print("CREATING BUT-PPG DATALOADERS")
    print("=" * 70)
    print(f"Data directory: {data_dir}")

    # Support multiple directory structures
    def find_data_file(split_name: str) -> Optional[Path]:
        """Find data file for a split, supporting multiple directory structures."""
        # Try different possible paths (in order of preference)
        possible_paths = [
            # Structure 1: Flat structure with simple names
            data_dir / f'{split_name}.npz',

            # Structure 2: Nested with data.npz
            data_dir / split_name / 'data.npz',

            # Structure 3: All in train/ with _windows suffix (from prepare_all_data.py)
            data_dir / 'train' / f'{split_name}_windows.npz',

            # Structure 4: Direct _windows suffix
            data_dir / f'{split_name}_windows.npz',
        ]

        for path in possible_paths:
            if path.exists():
                return path

        return None

    train_file = find_data_file('train')
    val_file = find_data_file('val')
    test_file = find_data_file('test')

    if train_file is None:
        # Provide helpful error message
        possible_paths = [
            data_dir / 'train.npz',
            data_dir / 'train' / 'data.npz',
            data_dir / 'train' / 'train_windows.npz',
            data_dir / 'train_windows.npz',
        ]
        raise FileNotFoundError(
            f"Training data not found. Looked for:\n" +
            "\n".join(f"  - {p}" for p in possible_paths) +
            f"\n\nMake sure to run data preparation first:\n"
            f"  python scripts/prepare_all_data.py --dataset butppg --mode fasttrack\n\n"
            f"Or if you have data elsewhere, use:\n"
            f"  --data-dir /path/to/your/butppg/data"
        )
    
    # Create training loader
    print(f"\n✓ Found training data: {train_file}")
    train_dataset = BUTPPGDataset(train_file, normalize=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    # Create validation loader
    val_loader = None
    if use_val and val_file is not None:
        print(f"✓ Found validation data: {val_file}")
        val_dataset = BUTPPGDataset(val_file, normalize=True)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        print("⚠ Validation data not found, will use test set for validation")

    # Create test loader
    test_loader = None
    if test_file is not None:
        print(f"✓ Found test data: {test_file}")
        test_dataset = BUTPPGDataset(test_file, normalize=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        print("⚠ Test data not found")

    print("=" * 70)
    return train_loader, val_loader, test_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    use_amp: bool = True,
    scaler: Optional[GradScaler] = None,
    gradient_clip: float = 1.0,
    epoch: int = 0
) -> Dict[str, float]:
    """Train for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training dataloader
        criterion: Loss criterion
        optimizer: Optimizer
        device: Device to train on
        use_amp: Use automatic mixed precision
        scaler: GradScaler for AMP
        gradient_clip: Gradient clipping value
        epoch: Current epoch number
    
    Returns:
        Dictionary with training metrics
    """
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    
    for batch_idx, (signals, labels) in enumerate(pbar):
        signals = signals.to(device)
        labels = labels.to(device)
        
        # Forward pass
        with autocast(enabled=use_amp):
            logits = model(signals)
            loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            
            if gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    gradient_clip
                )
            
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    gradient_clip
                )
            
            optimizer.step()
        
        # Compute accuracy
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'acc': 100.0 * correct / total
        })
    
    return {
        'loss': total_loss / len(train_loader),
        'accuracy': 100.0 * correct / total
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    use_amp: bool = True,
    desc: str = "Val"
) -> Dict[str, float]:
    """Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate
        loader: Dataloader
        criterion: Loss criterion
        device: Device to evaluate on
        use_amp: Use automatic mixed precision
        desc: Description for progress bar
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc=f"[{desc}]", leave=False)
    
    for signals, labels in pbar:
        signals = signals.to(device)
        labels = labels.to(device)
        
        with autocast(enabled=use_amp):
            logits = model(signals)
            loss = criterion(logits, labels)
        
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({
            'loss': total_loss / len(loader),
            'acc': 100.0 * correct / total
        })
    
    # Compute additional metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Per-class accuracy
    class_0_mask = all_labels == 0
    class_1_mask = all_labels == 1
    
    class_0_acc = (all_preds[class_0_mask] == all_labels[class_0_mask]).mean() * 100 if class_0_mask.sum() > 0 else 0.0
    class_1_acc = (all_preds[class_1_mask] == all_labels[class_1_mask]).mean() * 100 if class_1_mask.sum() > 0 else 0.0
    
    return {
        'loss': total_loss / len(loader),
        'accuracy': 100.0 * correct / total,
        'class_0_acc': class_0_acc,
        'class_1_acc': class_1_acc
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    config: Dict
):
    """Save checkpoint.
    
    Args:
        path: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        metrics: Current metrics
        config: Training configuration
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config
    }
    
    torch.save(checkpoint, path)
    print(f"  ✓ Checkpoint saved: {path}")


def verify_data_structure(data_dir: Path):
    """Verify BUT-PPG data structure and contents for debugging"""
    print("\n" + "=" * 70)
    print("VERIFYING BUT-PPG DATA STRUCTURE")
    print("=" * 70)

    data_dir = Path(data_dir)
    print(f"Data directory: {data_dir}")
    print(f"Exists: {data_dir.exists()}")

    if not data_dir.exists():
        print(f"\n❌ Data directory does not exist!")
        print(f"   Create it by running:")
        print(f"     python scripts/prepare_all_data.py --dataset butppg --mode fasttrack")
        return

    print("\n📁 Directory contents:")
    has_files = False
    for item in sorted(data_dir.rglob("*")):
        if item.is_file():
            rel_path = item.relative_to(data_dir)
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"  {'📄' if item.suffix == '.npz' else '📝'} {rel_path} ({size_mb:.2f} MB)")
            has_files = True

    if not has_files:
        print(f"  (empty)")
        return

    # Check .npz files
    npz_files = list(data_dir.rglob("*.npz"))
    if npz_files:
        print(f"\n🔍 Inspecting .npz files:")
        for npz_file in sorted(npz_files):
            try:
                data = np.load(npz_file)
                rel_path = npz_file.relative_to(data_dir)
                print(f"\n  {rel_path}:")
                print(f"    Keys: {list(data.keys())}")

                # Support both 'signals' and 'data' keys
                if 'signals' in data:
                    signals = data['signals']
                elif 'data' in data:
                    signals = data['data']
                else:
                    signals = None

                if signals is not None:
                    print(f"    Signals shape: {signals.shape}")
                    if len(signals.shape) == 3:
                        N, C, T = signals.shape
                        print(f"      N={N} samples, C={C} channels, T={T} timesteps")

                        # Validate
                        status = []
                        if C == 5:
                            status.append("✅ 5 channels")
                        else:
                            status.append(f"⚠️  {C} channels (expected 5)")

                        if T == 1024:
                            status.append("✅ 1024 timesteps")
                        else:
                            status.append(f"⚠️  {T} timesteps (expected 1024)")

                        print(f"      Status: {', '.join(status)}")

                if 'labels' in data:
                    labels = data['labels']
                    unique = np.unique(labels)
                    print(f"    Labels: {len(labels)} samples, unique values: {unique}")

            except Exception as e:
                print(f"    ❌ Error loading: {e}")

    print("\n" + "=" * 70)
    print("✓ Verification complete")
    print("=" * 70)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune pretrained model on BUT-PPG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Pretrained model
    parser.add_argument(
        '--pretrained',
        type=str,
        default='artifacts/foundation_model/best_model.pt',
        help='Path to pretrained checkpoint'
    )
    
    # Data
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed/butppg/windows',
        help='Directory containing BUT-PPG data (supports both flat and nested structure)'
    )
    
    # Channel inflation
    parser.add_argument(
        '--pretrain-channels',
        type=int,
        default=2,
        help='Number of channels in pretrained model'
    )
    parser.add_argument(
        '--finetune-channels',
        type=int,
        default=5,
        help='Number of channels for fine-tuning'
    )
    
    # Training configuration
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Total number of epochs'
    )
    parser.add_argument(
        '--head-only-epochs',
        type=int,
        default=5,
        help='Number of epochs for head-only training (Stage 1)'
    )
    parser.add_argument(
        '--unfreeze-last-n',
        type=int,
        default=2,
        help='Number of last blocks to unfreeze in Stage 2'
    )
    parser.add_argument(
        '--full-finetune',
        action='store_true',
        help='Enable Stage 3: full fine-tuning at very low LR'
    )
    parser.add_argument(
        '--full-finetune-epochs',
        type=int,
        default=10,
        help='Number of epochs for full fine-tuning (Stage 3)'
    )
    
    # Optimization
    parser.add_argument(
        '--lr',
        type=float,
        default=2e-5,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.01,
        help='Weight decay'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--gradient-clip',
        type=float,
        default=1.0,
        help='Gradient clipping value'
    )
    
    # Device and performance
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--no-amp',
        action='store_true',
        help='Disable automatic mixed precision'
    )
    
    # Output
    parser.add_argument(
        '--output-dir',
        type=str,
        default='artifacts/but_ppg_finetuned',
        help='Output directory for checkpoints'
    )
    
    # Logging
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    # Debugging
    parser.add_argument(
        '--verify-data',
        action='store_true',
        help='Verify data structure and exit (useful for debugging)'
    )

    return parser.parse_args()


def main():
    """Main fine-tuning script."""
    args = parse_args()

    # Handle --verify-data early (before loading model)
    if args.verify_data:
        verify_data_structure(Path(args.data_dir))
        return 0

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print configuration
    print("\n" + "=" * 70)
    print("BUT-PPG FINE-TUNING CONFIGURATION")
    print("=" * 70)
    print(f"Pretrained model: {args.pretrained}")
    print(f"Data directory: {args.data_dir}")
    print(f"Channel inflation: {args.pretrain_channels} → {args.finetune_channels}")
    print(f"Total epochs: {args.epochs}")
    print(f"  Stage 1 (head-only): {args.head_only_epochs} epochs")
    print(f"  Stage 2 (partial unfreeze): {args.epochs - args.head_only_epochs} epochs")
    if args.full_finetune:
        print(f"  Stage 3 (full finetune): {args.full_finetune_epochs} epochs at LR={args.lr/10:.2e}")
    print(f"Learning rate: {args.lr:.2e}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"AMP: {not args.no_amp and torch.cuda.is_available()}")
    print("=" * 70)
    
    # Load pretrained model and inflate channels
    print("\n" + "=" * 70)
    print("LOADING PRETRAINED MODEL AND INFLATING CHANNELS")
    print("=" * 70)
    
    model_config = {
        'variant': 'ibm-granite/granite-timeseries-ttm-r1',
        'task': 'classification',
        'num_classes': 2,  # Binary classification: good vs poor
        'input_channels': args.finetune_channels,
        'context_length': 1024,
        'patch_size': 128,
        'head_type': 'linear',
        'freeze_encoder': False  # Will be set by load_pretrained_with_channel_inflate
    }
    
    model = load_pretrained_with_channel_inflate(
        checkpoint_path=args.pretrained,
        pretrain_channels=args.pretrain_channels,
        finetune_channels=args.finetune_channels,
        freeze_pretrained=True,  # Start with encoder frozen
        model_config=model_config,
        device=args.device
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=Path(args.data_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Use test set for validation if no validation set
    if val_loader is None and test_loader is not None:
        val_loader = test_loader
        print("\n⚠ Using test set for validation (no separate val set)")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    use_amp = not args.no_amp and torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'stage': []
    }
    
    best_val_acc = 0.0
    
    # Save training configuration
    training_config = vars(args)
    with open(output_dir / 'training_config.json', 'w') as f:
        json.dump(training_config, f, indent=2)
    
    # =========================================================================
    # STAGE 1: HEAD-ONLY TRAINING
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"STAGE 1: HEAD-ONLY TRAINING ({args.head_only_epochs} epochs)")
    print("=" * 70)
    print("Training only the classification head, encoder frozen")
    
    # Create optimizer for head-only
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    for epoch in range(args.head_only_epochs):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer,
            args.device, use_amp, scaler, args.gradient_clip, epoch
        )
        
        # Validate
        if val_loader is not None:
            val_metrics = evaluate(
                model, val_loader, criterion, args.device, use_amp, "Val"
            )
        else:
            val_metrics = {'loss': 0.0, 'accuracy': 0.0}
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{args.head_only_epochs} ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['stage'].append('stage1_head_only')
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            save_checkpoint(
                output_dir / 'best_model.pt',
                model, optimizer, epoch, val_metrics, training_config
            )
            print(f"  ✓ Best model saved (val_acc: {val_metrics['accuracy']:.2f}%)")
    
    # =========================================================================
    # STAGE 2: PARTIAL UNFREEZING
    # =========================================================================
    remaining_epochs = args.epochs - args.head_only_epochs
    
    if remaining_epochs > 0:
        print("\n" + "=" * 70)
        print(f"STAGE 2: PARTIAL UNFREEZING ({remaining_epochs} epochs)")
        print("=" * 70)
        print(f"Unfreezing last {args.unfreeze_last_n} encoder blocks")
        
        # Unfreeze last N blocks
        unfreeze_last_n_blocks(model, n=args.unfreeze_last_n, verbose=True)
        
        # Create new optimizer with unfrozen parameters
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        for epoch in range(remaining_epochs):
            epoch_start = time.time()
            
            # Train
            train_metrics = train_epoch(
                model, train_loader, criterion, optimizer,
                args.device, use_amp, scaler, args.gradient_clip,
                args.head_only_epochs + epoch
            )
            
            # Validate
            if val_loader is not None:
                val_metrics = evaluate(
                    model, val_loader, criterion, args.device, use_amp, "Val"
                )
            else:
                val_metrics = {'loss': 0.0, 'accuracy': 0.0}
            
            epoch_time = time.time() - epoch_start
            
            # Print epoch summary
            print(f"\nEpoch {args.head_only_epochs + epoch + 1}/{args.epochs} ({epoch_time:.1f}s)")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
            
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['stage'].append('stage2_partial_unfreeze')
            
            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                save_checkpoint(
                    output_dir / 'best_model.pt',
                    model, optimizer, epoch, val_metrics, training_config
                )
                print(f"  ✓ Best model saved (val_acc: {val_metrics['accuracy']:.2f}%)")
    
    # =========================================================================
    # STAGE 3: FULL FINE-TUNING (OPTIONAL)
    # =========================================================================
    if args.full_finetune and args.full_finetune_epochs > 0:
        print("\n" + "=" * 70)
        print(f"STAGE 3: FULL FINE-TUNING ({args.full_finetune_epochs} epochs)")
        print("=" * 70)
        print("Unfreezing all parameters with very low learning rate")
        
        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True
        
        # Create new optimizer with very low LR
        low_lr = args.lr / 10
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=low_lr,
            weight_decay=args.weight_decay
        )
        
        print(f"  Learning rate: {low_lr:.2e}")
        
        for epoch in range(args.full_finetune_epochs):
            epoch_start = time.time()
            
            # Train
            train_metrics = train_epoch(
                model, train_loader, criterion, optimizer,
                args.device, use_amp, scaler, args.gradient_clip,
                args.epochs + epoch
            )
            
            # Validate
            if val_loader is not None:
                val_metrics = evaluate(
                    model, val_loader, criterion, args.device, use_amp, "Val"
                )
            else:
                val_metrics = {'loss': 0.0, 'accuracy': 0.0}
            
            epoch_time = time.time() - epoch_start
            
            # Print epoch summary
            print(f"\nEpoch {args.epochs + epoch + 1}/{args.epochs + args.full_finetune_epochs} ({epoch_time:.1f}s)")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
            
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['stage'].append('stage3_full_finetune')
            
            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                save_checkpoint(
                    output_dir / 'best_model.pt',
                    model, optimizer, epoch, val_metrics, training_config
                )
                print(f"  ✓ Best model saved (val_acc: {val_metrics['accuracy']:.2f}%)")
    
    # =========================================================================
    # FINAL EVALUATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    
    # Load best model
    best_checkpoint = torch.load(output_dir / 'best_model.pt', map_location=args.device, weights_only=False)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Evaluate on test set if available
    if test_loader is not None:
        print("\nEvaluating on test set...")
        test_metrics = evaluate(
            model, test_loader, criterion, args.device, use_amp, "Test"
        )
        
        print(f"\nTest Results:")
        print(f"  Loss: {test_metrics['loss']:.4f}")
        print(f"  Accuracy: {test_metrics['accuracy']:.2f}%")
        print(f"  Class 0 (Poor) Accuracy: {test_metrics['class_0_acc']:.2f}%")
        print(f"  Class 1 (Good) Accuracy: {test_metrics['class_1_acc']:.2f}%")
        
        # Save test metrics
        with open(output_dir / 'test_metrics.json', 'w') as f:
            json.dump(test_metrics, f, indent=2)
    else:
        print("\n⚠ No test set available for final evaluation")
        test_metrics = None
    
    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save final model
    save_checkpoint(
        output_dir / 'final_model.pt',
        model, optimizer, args.epochs, val_metrics, training_config
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    if test_metrics:
        print(f"Test accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"Checkpoints saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
