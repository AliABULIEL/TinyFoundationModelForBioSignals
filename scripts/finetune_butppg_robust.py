#!/usr/bin/env python3
"""ROBUST Fine-Tuning Script for BUT-PPG with Domain Adaptation.

This script implements a production-ready fine-tuning pipeline with:
- Domain adaptation for VitalDB → BUT-PPG transfer
- AUROC metric for imbalanced data
- Early stopping to prevent overfitting
- Progressive learning rates (encoder << adapter << head)
- Comprehensive logging and checkpointing

Usage:
    # With domain adaptation (RECOMMENDED)
    python scripts/finetune_butppg_robust.py \
        --pretrained artifacts/foundation_model/best_model.pt \
        --data-dir data/processed/butppg/windows_with_labels \
        --adaptation projection \
        --epochs 30 \
        --output-dir artifacts/butppg_robust

    # Without domain adaptation (baseline comparison)
    python scripts/finetune_butppg_robust.py \
        --pretrained artifacts/foundation_model/best_model.pt \
        --data-dir data/processed/butppg/windows_with_labels \
        --no-adaptation \
        --epochs 30 \
        --output-dir artifacts/butppg_baseline

Author: Claude Code (October 2025)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time
import warnings
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from src.models.ttm_adapter import TTMAdapter
from src.models.domain_adaptation import (
    DomainProjectionAdapter,
    DomainAdversarialAdapter,
    ProgressiveFineTuner
)


class BUTPPGDataset(Dataset):
    """Dataset for BUT-PPG quality classification from windowed format."""

    def __init__(self, data_dir: Path, normalize: bool = True, target_length: int = None):
        """Initialize BUT-PPG dataset.

        Args:
            data_dir: Path to directory containing window_*.npz files
            normalize: Whether to apply z-score normalization per channel
            target_length: If specified, resize signals to this length
        """
        print(f"Loading BUT-PPG data from: {data_dir}")

        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        self.target_length = target_length

        # Find all window files
        window_files = sorted(data_dir.glob('window_*.npz'))

        if len(window_files) == 0:
            raise FileNotFoundError(
                f"No window files found in {data_dir}\n"
                f"Expected files matching pattern: window_*.npz\n"
                f"Make sure to run data preparation first:\n"
                f"  python scripts/prepare_all_data.py --dataset butppg --mode fasttrack"
            )

        # Load all windows
        signals_list = []
        labels_list = []

        for window_file in window_files:
            data = np.load(window_file)

            # Load signal [2, 1024]
            if 'signal' not in data:
                raise KeyError(f"Expected 'signal' key in {window_file}")

            signal = data['signal']  # [2, 1024]

            # Load quality label
            if 'quality' in data:
                quality = data['quality']
                label = int(quality) if not np.isnan(quality) else 0
            else:
                label = 0

            signals_list.append(signal)
            labels_list.append(label)

        # Stack into tensors
        self.signals = torch.from_numpy(np.stack(signals_list, axis=0)).float()  # [N, 2, T]
        self.labels = torch.tensor(labels_list, dtype=torch.long)  # [N]

        # Validate shapes
        N, C, T = self.signals.shape
        assert C == 2, f"Expected 2 channels, got {C}"
        assert len(self.labels) == N, f"Label count mismatch"

        # Resize if needed
        if self.target_length is not None and T != self.target_length:
            print(f"  ⚠️  Resizing signals from {T} to {self.target_length} samples")
            self.signals = F.interpolate(
                self.signals,
                size=self.target_length,
                mode='linear',
                align_corners=False
            )
            N, C, T = self.signals.shape
            print(f"  ✓ Resized to: {self.signals.shape}")

        # Normalize
        if normalize:
            for c in range(C):
                channel_data = self.signals[:, c, :]
                mean = channel_data.mean()
                std = channel_data.std()
                if std > 0:
                    self.signals[:, c, :] = (channel_data - mean) / std

        # Print statistics
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
    use_val: bool = True,
    target_length: int = None
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """Create dataloaders for BUT-PPG."""
    data_dir = Path(data_dir)

    print("\n" + "=" * 70)
    print("CREATING BUT-PPG DATALOADERS")
    print("=" * 70)
    print(f"Data directory: {data_dir}")

    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    test_dir = data_dir / 'test'

    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    # Create datasets
    print(f"\n✓ Found training directory: {train_dir}")
    train_dataset = BUTPPGDataset(train_dir, normalize=True, target_length=target_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = None
    if use_val and val_dir.exists():
        print(f"✓ Found validation directory: {val_dir}")
        val_dataset = BUTPPGDataset(val_dir, normalize=True, target_length=target_length)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        print("⚠ Validation directory not found, will use test set")

    test_loader = None
    if test_dir.exists():
        print(f"✓ Found test directory: {test_dir}")
        test_dataset = BUTPPGDataset(test_dir, normalize=True, target_length=target_length)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        print("⚠ Test directory not found")

    print("=" * 70)
    return train_loader, val_loader, test_loader


class DomainAdaptedModel(nn.Module):
    """Complete model with SSL encoder + domain adapter + task head."""

    def __init__(self, encoder: nn.Module, adapter: Optional[nn.Module], head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.adapter = adapter  # Can be None if no adaptation
        self.head = head

    def forward(self, x):
        """Forward pass through encoder → adapter → head.

        Args:
            x: Input signals [B, C, T]

        Returns:
            logits: Classification logits [B, num_classes]
        """
        # Encode: [B, C, T] → [B, P, D]
        features = self.encoder.get_encoder_output(x)

        # Adapt: [B, P, D] → [B, P, D]
        if self.adapter is not None:
            adapted = self.adapter(features)
            # Handle adversarial case (returns tuple)
            if isinstance(adapted, tuple):
                adapted = adapted[0]
        else:
            adapted = features

        # Pool patches: [B, P, D] → [B, D]
        pooled = adapted.mean(dim=1)

        # Classify: [B, D] → [B, num_classes]
        logits = self.head(pooled)
        return logits


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
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []

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
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            optimizer.step()

        # Collect predictions
        probs = F.softmax(logits, dim=1)[:, 1]  # Probability of class 1
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total_loss += loss.item()

        # Update progress bar
        pbar.set_postfix({'loss': total_loss / (batch_idx + 1)})

    # Compute metrics
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean() * 100
    auroc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
    f1 = f1_score(all_labels, all_preds)

    return {
        'loss': total_loss / len(train_loader),
        'accuracy': accuracy,
        'auroc': auroc,
        'f1': f1
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
    """Evaluate model on a dataset."""
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []

    pbar = tqdm(loader, desc=f"[{desc}]", leave=False)

    for signals, labels in pbar:
        signals = signals.to(device)
        labels = labels.to(device)

        with autocast(enabled=use_amp):
            logits = model(signals)
            loss = criterion(logits, labels)

        probs = F.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total_loss += loss.item()

    # Compute metrics
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean() * 100
    auroc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)

    # Per-class accuracy
    class_0_mask = all_labels == 0
    class_1_mask = all_labels == 1

    class_0_acc = (all_preds[class_0_mask] == all_labels[class_0_mask]).mean() * 100 if class_0_mask.sum() > 0 else 0.0
    class_1_acc = (all_preds[class_1_mask] == all_labels[class_1_mask]).mean() * 100 if class_1_mask.sum() > 0 else 0.0

    return {
        'loss': total_loss / len(loader),
        'accuracy': accuracy,
        'auroc': auroc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'class_0_acc': class_0_acc,
        'class_1_acc': class_1_acc
    }


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'max'):
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics to maximize (AUROC), 'min' for minimize (loss)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """Check if training should stop.

        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    config: Dict
):
    """Save checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config
    }

    torch.save(checkpoint, path)
    print(f"  ✓ Checkpoint saved: {path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Robust fine-tuning with domain adaptation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Pretrained model
    parser.add_argument('--pretrained', type=str, required=True,
                       help='Path to pretrained SSL checkpoint')

    # Data
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing BUT-PPG windowed data')

    # Domain adaptation
    parser.add_argument('--adaptation', type=str, choices=['projection', 'adversarial', 'none'],
                       default='projection',
                       help='Domain adaptation method')
    parser.add_argument('--no-adaptation', action='store_true',
                       help='Disable domain adaptation (baseline)')

    # Training
    parser.add_argument('--epochs', type=int, default=30,
                       help='Total number of epochs')
    parser.add_argument('--head-only-epochs', type=int, default=5,
                       help='Epochs for head-only training (Phase 1)')
    parser.add_argument('--partial-epochs', type=int, default=15,
                       help='Epochs for partial unfreezing (Phase 2)')

    # Optimization
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Base learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                       help='Gradient clipping value')

    # Early stopping
    parser.add_argument('--early-stopping', action='store_true',
                       help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')

    # Device
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--no-amp', action='store_true',
                       help='Disable automatic mixed precision')

    # Output
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for checkpoints')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    return parser.parse_args()


def main():
    """Main fine-tuning script."""
    args = parse_args()

    # Override adaptation if no-adaptation is set
    if args.no_adaptation:
        args.adaptation = 'none'

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
    print("ROBUST BUT-PPG FINE-TUNING CONFIGURATION")
    print("=" * 70)
    print(f"Pretrained model: {args.pretrained}")
    print(f"Data directory: {args.data_dir}")
    print(f"Domain adaptation: {args.adaptation}")
    print(f"Total epochs: {args.epochs}")
    print(f"  Phase 1 (head-only): {args.head_only_epochs} epochs")
    print(f"  Phase 2 (partial): {args.partial_epochs} epochs")
    print(f"  Phase 3 (full): {args.epochs - args.head_only_epochs - args.partial_epochs} epochs")
    print(f"Learning rate: {args.lr:.2e}")
    print(f"Batch size: {args.batch_size}")
    print(f"Early stopping: {'Enabled' if args.early_stopping else 'Disabled'}")
    if args.early_stopping:
        print(f"  Patience: {args.patience} epochs")
    print(f"Device: {args.device}")
    print("=" * 70)

    # Load pretrained checkpoint
    print("\n" + "=" * 70)
    print("LOADING PRETRAINED MODEL")
    print("=" * 70)

    checkpoint = torch.load(args.pretrained, map_location='cpu', weights_only=False)

    # Get encoder weights
    if 'encoder_state_dict' in checkpoint:
        encoder_state = checkpoint['encoder_state_dict']
    elif 'model_state_dict' in checkpoint:
        encoder_state = checkpoint['model_state_dict']
    else:
        encoder_state = checkpoint

    # Detect architecture from weights
    patcher_keys = [k for k in encoder_state.keys() if 'patcher.weight' in k and 'encoder' in k]
    if not patcher_keys:
        raise ValueError("Could not find patcher weights in checkpoint")

    patcher_weight = encoder_state[patcher_keys[0]]
    d_model = patcher_weight.shape[0]
    patch_size = patcher_weight.shape[1]

    context_length = 1024  # VitalDB standard

    print(f"Detected architecture:")
    print(f"  d_model: {d_model}")
    print(f"  patch_size: {patch_size}")
    print(f"  context_length: {context_length}")

    # Create SSL encoder
    print(f"\nCreating SSL encoder...")
    encoder = TTMAdapter(
        variant='ibm-granite/granite-timeseries-ttm-r1',
        task='ssl',
        input_channels=2,
        context_length=context_length,
        use_real_ttm=True
    )

    # Load SSL weights
    encoder.load_state_dict(encoder_state, strict=False)
    encoder = encoder.to(args.device)
    encoder.eval()  # Freeze encoder initially
    for param in encoder.parameters():
        param.requires_grad = False

    print(f"✓ SSL encoder loaded and frozen")

    # Create domain adapter
    if args.adaptation == 'projection':
        adapter = DomainProjectionAdapter(d_model=d_model, dropout=0.3)
        print(f"✓ Created projection adapter")
    elif args.adaptation == 'adversarial':
        adapter = DomainAdversarialAdapter(d_model=d_model, dropout=0.3)
        print(f"✓ Created adversarial adapter")
    else:
        adapter = None
        print(f"⚠️  No domain adaptation (baseline mode)")

    # Create task head
    task_head = nn.Linear(d_model, 2)  # Binary classification
    print(f"✓ Created task head ({d_model} → 2)")

    # Combine into full model
    model = DomainAdaptedModel(encoder, adapter, task_head)
    model = model.to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✓ Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=Path(args.data_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_length=context_length
    )

    if val_loader is None and test_loader is not None:
        val_loader = test_loader
        print("\n⚠ Using test set for validation")

    # Compute class weights
    print("\n⚙️  Computing class weights...")
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())
    all_labels = np.array(all_labels)

    class_counts = np.bincount(all_labels)
    total_samples = len(all_labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights = torch.FloatTensor(class_weights).to(args.device)

    print(f"  Class 0: {class_counts[0]} samples, weight: {class_weights[0]:.3f}")
    print(f"  Class 1: {class_counts[1]} samples, weight: {class_weights[1]:.3f}")

    # Setup training
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    use_amp = not args.no_amp and torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None

    # Early stopping
    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=0.001,
        mode='max'  # Maximize AUROC
    ) if args.early_stopping else None

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_auroc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auroc': [],
        'phase': []
    }

    best_val_auroc = 0.0

    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Create progressive fine-tuner
    tuner = ProgressiveFineTuner(encoder, adapter if adapter else nn.Identity(), task_head)

    # ========================================================================
    # PHASE 1: HEAD-ONLY TRAINING
    # ========================================================================
    print("\n" + "=" * 70)
    print(f"PHASE 1: HEAD-ONLY TRAINING ({args.head_only_epochs} epochs)")
    print("=" * 70)

    optimizer = tuner.get_optimizer(phase=1, base_lr=args.lr * 10, weight_decay=args.weight_decay)

    for epoch in range(args.head_only_epochs):
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer,
            args.device, use_amp, scaler, args.gradient_clip, epoch
        )

        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, criterion, args.device, use_amp, "Val")
        else:
            val_metrics = {'loss': 0.0, 'accuracy': 0.0, 'auroc': 0.5}

        print(f"\nEpoch {epoch+1}/{args.head_only_epochs}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, AUROC: {train_metrics['auroc']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, AUROC: {val_metrics['auroc']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")

        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_auroc'].append(train_metrics['auroc'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_auroc'].append(val_metrics['auroc'])
        history['phase'].append('phase1_head_only')

        # Save best model
        if val_metrics['auroc'] > best_val_auroc:
            best_val_auroc = val_metrics['auroc']
            save_checkpoint(
                output_dir / 'best_model.pt',
                model, optimizer, epoch, val_metrics, vars(args)
            )
            print(f"  🏆 Best model saved (AUROC: {val_metrics['auroc']:.4f})")

    # ========================================================================
    # PHASE 2: PARTIAL UNFREEZING
    # ========================================================================
    if args.partial_epochs > 0:
        print("\n" + "=" * 70)
        print(f"PHASE 2: PARTIAL UNFREEZING ({args.partial_epochs} epochs)")
        print("=" * 70)

        optimizer = tuner.get_optimizer(phase=2, base_lr=args.lr, weight_decay=args.weight_decay)

        for epoch in range(args.partial_epochs):
            train_metrics = train_epoch(
                model, train_loader, criterion, optimizer,
                args.device, use_amp, scaler, args.gradient_clip,
                args.head_only_epochs + epoch
            )

            if val_loader is not None:
                val_metrics = evaluate(model, val_loader, criterion, args.device, use_amp, "Val")
            else:
                val_metrics = {'loss': 0.0, 'accuracy': 0.0, 'auroc': 0.5}

            print(f"\nEpoch {args.head_only_epochs + epoch + 1}/{args.head_only_epochs + args.partial_epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, AUROC: {train_metrics['auroc']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, AUROC: {val_metrics['auroc']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")

            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['train_auroc'].append(train_metrics['auroc'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_auroc'].append(val_metrics['auroc'])
            history['phase'].append('phase2_partial')

            if val_metrics['auroc'] > best_val_auroc:
                best_val_auroc = val_metrics['auroc']
                save_checkpoint(
                    output_dir / 'best_model.pt',
                    model, optimizer, epoch, val_metrics, vars(args)
                )
                print(f"  🏆 Best model saved (AUROC: {val_metrics['auroc']:.4f})")

            # Early stopping check
            if early_stopping is not None:
                if early_stopping(val_metrics['auroc']):
                    print(f"\n⚠️  Early stopping triggered (no improvement for {args.patience} epochs)")
                    break

    # ========================================================================
    # PHASE 3: FULL FINE-TUNING
    # ========================================================================
    remaining_epochs = args.epochs - args.head_only_epochs - args.partial_epochs

    if remaining_epochs > 0 and (early_stopping is None or not early_stopping.early_stop):
        print("\n" + "=" * 70)
        print(f"PHASE 3: FULL FINE-TUNING ({remaining_epochs} epochs)")
        print("=" * 70)

        optimizer = tuner.get_optimizer(phase=3, base_lr=args.lr, weight_decay=args.weight_decay)

        for epoch in range(remaining_epochs):
            train_metrics = train_epoch(
                model, train_loader, criterion, optimizer,
                args.device, use_amp, scaler, args.gradient_clip,
                args.head_only_epochs + args.partial_epochs + epoch
            )

            if val_loader is not None:
                val_metrics = evaluate(model, val_loader, criterion, args.device, use_amp, "Val")
            else:
                val_metrics = {'loss': 0.0, 'accuracy': 0.0, 'auroc': 0.5}

            print(f"\nEpoch {args.head_only_epochs + args.partial_epochs + epoch + 1}/{args.epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, AUROC: {train_metrics['auroc']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, AUROC: {val_metrics['auroc']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")

            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['train_auroc'].append(train_metrics['auroc'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_auroc'].append(val_metrics['auroc'])
            history['phase'].append('phase3_full')

            if val_metrics['auroc'] > best_val_auroc:
                best_val_auroc = val_metrics['auroc']
                save_checkpoint(
                    output_dir / 'best_model.pt',
                    model, optimizer, epoch, val_metrics, vars(args)
                )
                print(f"  🏆 Best model saved (AUROC: {val_metrics['auroc']:.4f})")

            if early_stopping is not None:
                if early_stopping(val_metrics['auroc']):
                    print(f"\n⚠️  Early stopping triggered")
                    break

    # ========================================================================
    # FINAL EVALUATION
    # ========================================================================
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    # Load best model
    best_checkpoint = torch.load(output_dir / 'best_model.pt', map_location=args.device, weights_only=False)
    model.load_state_dict(best_checkpoint['model_state_dict'])

    # Evaluate on test set
    if test_loader is not None:
        print("\n Evaluating on test set...")
        test_metrics = evaluate(model, test_loader, criterion, args.device, use_amp, "Test")

        print(f"\n🏆 Test Results:")
        print(f"  AUROC: {test_metrics['auroc']:.4f}  ← PRIMARY METRIC")
        print(f"  Accuracy: {test_metrics['accuracy']:.2f}%")
        print(f"  F1 Score: {test_metrics['f1']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        print(f"  Class 0 (Poor) Acc: {test_metrics['class_0_acc']:.2f}%")
        print(f"  Class 1 (Good) Acc: {test_metrics['class_1_acc']:.2f}%")

        with open(output_dir / 'test_metrics.json', 'w') as f:
            json.dump(test_metrics, f, indent=2)

    # Save training history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation AUROC: {best_val_auroc:.4f}")
    if test_loader is not None:
        print(f"Test AUROC: {test_metrics['auroc']:.4f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"Checkpoints saved to: {output_dir}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
