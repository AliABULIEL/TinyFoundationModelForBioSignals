#!/usr/bin/env python3
"""Enhanced Fine-Tuning Script for BUT-PPG.

This script combines best practices from both finetune_butppg.py and finetune_butppg_robust.py
while fixing critical data loading issues.

Key Improvements:
================
✅ Uses existing BUTPPGDataset from src/data/butppg_dataset.py
✅ Loads ALL labels from NPZ files (quality, hr, bp, spo2, glycaemia, etc.)
✅ Filters invalid samples with filter_missing=True
✅ AUROC as primary metric (better for imbalanced data)
✅ Early stopping to prevent overfitting
✅ Progressive learning rates (encoder 0.1× < adapter 1× < head 1×)
✅ Optional domain adaptation (projection/adversarial/none)
✅ Multi-task support (quality, hr, blood_pressure, etc.)
✅ Comprehensive evaluation (AUROC, F1, precision, recall, per-class accuracy)

Fixes from Original Scripts:
============================
❌ OLD: Duplicate BUTPPGDataset class with flawed label loading
✅ NEW: Uses src/data/butppg_dataset.py with comprehensive label handling

❌ OLD: Missing labels default to 0 (corrupts training data)
✅ NEW: Filters out invalid samples with filter_missing=True

❌ OLD: Only loads 'quality' label
✅ NEW: Loads ALL available labels (quality, hr, bp, spo2, etc.)

❌ OLD: Global normalization across all samples (data leakage)
✅ NEW: Per-sample normalization from preprocessed windows

❌ OLD: Accuracy as primary metric (poor for imbalanced data)
✅ NEW: AUROC as primary metric

Usage:
======
    # Quality classification with projection domain adaptation (RECOMMENDED)
    python scripts/finetune_enhanced.py \
        --pretrained artifacts/foundation_model/best_model.pt \
        --data-dir data/processed/butppg/windows_with_labels \
        --task quality \
        --adaptation projection \
        --epochs 30 \
        --output-dir artifacts/butppg_enhanced

    # Heart rate regression without domain adaptation
    python scripts/finetune_enhanced.py \
        --pretrained artifacts/foundation_model/best_model.pt \
        --data-dir data/processed/butppg/windows_with_labels \
        --task hr \
        --no-adaptation \
        --epochs 30 \
        --output-dir artifacts/butppg_hr

    # Blood pressure estimation with adversarial domain adaptation
    python scripts/finetune_enhanced.py \
        --pretrained artifacts/foundation_model/best_model.pt \
        --data-dir data/processed/butppg/windows_with_labels \
        --task blood_pressure \
        --adaptation adversarial \
        --epochs 30 \
        --output-dir artifacts/butppg_bp

Author: Enhanced by Claude Code (October 2025)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress all warnings for clean output
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async CUDA

import argparse
import json
import time
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, mean_absolute_error, mean_squared_error

from src.models.ttm_adapter import TTMAdapter
from src.models.domain_adaptation import (
    DomainProjectionAdapter,
    DomainAdversarialAdapter,
    ProgressiveFineTuner
)
from src.data.butppg_dataset import BUTPPGDataset


# ============================================================================
# TASK CONFIGURATIONS
# ============================================================================

TASK_CONFIGS = {
    # Classification tasks
    'quality': {
        'type': 'classification',
        'num_classes': 2,
        'label_keys': ['quality'],
        'primary_metric': 'auroc',
        'description': 'PPG quality classification (good vs poor)',
        'target': 0.85
    },
    'motion': {
        'type': 'classification',
        'num_classes': 8,  # Updated to 8 classes (BUT-PPG full motion taxonomy)
        'label_keys': ['motion'],
        'primary_metric': 'auroc',
        'description': 'Motion state classification (8 classes)',
        'target': 0.85
    },

    # Regression tasks
    'hr': {
        'type': 'regression',
        'num_outputs': 1,
        'label_keys': ['hr'],
        'primary_metric': 'mae',
        'description': 'Heart rate estimation (bpm)',
        'target': 2.0  # MAE target in BPM
    },
    'hr_estimation': {  # Alias for hr (for compatibility)
        'type': 'regression',
        'num_outputs': 1,
        'label_keys': ['hr'],
        'primary_metric': 'mae',
        'description': 'Heart rate estimation (bpm)',
        'target': 2.0
    },
    'bp_systolic': {
        'type': 'regression',
        'num_outputs': 1,
        'label_keys': ['bp_systolic'],
        'primary_metric': 'mae',
        'description': 'Systolic blood pressure estimation (mmHg)',
        'target': 5.0  # AAMI standard
    },
    'bp_diastolic': {
        'type': 'regression',
        'num_outputs': 1,
        'label_keys': ['bp_diastolic'],
        'primary_metric': 'mae',
        'description': 'Diastolic blood pressure estimation (mmHg)',
        'target': 5.0  # AAMI standard
    },
    'blood_pressure': {
        'type': 'regression',
        'num_outputs': 2,
        'label_keys': ['bp_systolic', 'bp_diastolic'],
        'primary_metric': 'mae',
        'description': 'Blood pressure estimation (systolic, diastolic)',
        'target': 5.0
    },
    'spo2': {
        'type': 'regression',
        'num_outputs': 1,
        'label_keys': ['spo2'],
        'primary_metric': 'mae',
        'description': 'SpO2 estimation (%)',
        'target': 2.0  # Standard clinical tolerance
    },
    'glycaemia': {
        'type': 'regression',
        'num_outputs': 1,
        'label_keys': ['glycaemia'],
        'primary_metric': 'mae',
        'description': 'Blood glucose estimation (mg/dL)',
        'target': 15.0  # Clinical standard for CGM accuracy
    },
}


# ============================================================================
# DATASET WRAPPER FOR TASK-SPECIFIC LABEL EXTRACTION
# ============================================================================

class TaskSpecificDataset(torch.utils.data.Dataset):
    """Wrapper around BUTPPGDataset to extract task-specific labels.

    This wrapper:
    1. Uses BUTPPGDataset from src/data/butppg_dataset.py
    2. Extracts task-specific labels from the full label dict
    3. Handles both classification and regression tasks
    """

    def __init__(self, butppg_dataset: BUTPPGDataset, task_config: Dict):
        """Initialize task-specific dataset.

        Args:
            butppg_dataset: BUTPPGDataset instance (with return_labels=True)
            task_config: Task configuration from TASK_CONFIGS
        """
        self.dataset = butppg_dataset
        self.task_config = task_config
        self.task_type = task_config['type']
        self.label_keys = task_config['label_keys']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """Get sample with task-specific label extraction.

        Returns:
            signal: Tensor [C, T]
            label: Scalar (classification) or Tensor (regression with multiple outputs)
        """
        # BUTPPGDataset returns (signal, signal2, labels) when return_labels=True
        signal, _, labels_dict = self.dataset[idx]

        # Extract task-specific labels
        if self.task_type == 'classification':
            # Single label for classification
            label = labels_dict[self.label_keys[0]]
            label = torch.tensor(label, dtype=torch.long)
        else:  # regression
            # Single or multiple outputs
            label_values = [labels_dict[key] for key in self.label_keys]
            label = torch.tensor(label_values, dtype=torch.float32)
            if len(label_values) == 1:
                label = label.squeeze()  # Scalar for single output

        return signal, label


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class DomainAdaptedModel(nn.Module):
    """Complete model: SSL Encoder + Domain Adapter + Task Head."""

    def __init__(
        self,
        encoder: nn.Module,
        adapter: Optional[nn.Module],
        head: nn.Module,
        task_type: str = 'classification'
    ):
        super().__init__()
        self.encoder = encoder
        self.adapter = adapter  # Can be None
        self.head = head
        self.task_type = task_type

    def forward(self, x):
        """Forward pass: x → encoder → adapter → pool → head → output.

        Args:
            x: Input signals [B, C, T]

        Returns:
            output: Logits [B, num_classes] for classification
                    or predictions [B, num_outputs] for regression
        """
        # Encode: [B, C, T] → [B, P, D]
        features = self.encoder.get_encoder_output(x)

        # Adapt: [B, P, D] → [B, P, D]
        if self.adapter is not None:
            adapted = self.adapter(features)
            # Handle adversarial adapter (returns tuple)
            if isinstance(adapted, tuple):
                adapted = adapted[0]
        else:
            adapted = features

        # Pool patches: [B, P, D] → [B, D]
        pooled = adapted.mean(dim=1)

        # Task head: [B, D] → [B, output_dim]
        output = self.head(pooled)

        return output


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    task_type: str,
    use_amp: bool = True,
    scaler: Optional[GradScaler] = None,
    gradient_clip: float = 1.0,
    epoch: int = 0
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    all_preds = []
    all_probs = []  # For AUROC
    all_labels = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)

    for batch_idx, (signals, labels) in enumerate(pbar):
        signals = signals.to(device)
        labels = labels.to(device)

        # Forward pass
        with autocast(enabled=use_amp):
            outputs = model(signals)
            loss = criterion(outputs, labels)

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
        if task_type == 'classification':
            probs = F.softmax(outputs, dim=1)[:, 1]  # P(class=1)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
        else:  # regression
            preds = outputs
            all_preds.extend(preds.detach().cpu().numpy())

        all_labels.extend(labels.cpu().numpy())
        total_loss += loss.item()

        pbar.set_postfix({'loss': total_loss / (batch_idx + 1)})

    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = {'loss': total_loss / len(train_loader)}

    if task_type == 'classification':
        all_probs = np.array(all_probs)
        accuracy = (all_preds == all_labels).mean() * 100

        # AUROC (primary metric for classification)
        try:
            auroc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
        except:
            auroc = 0.5

        f1 = f1_score(all_labels, all_preds, average='binary' if len(np.unique(all_labels)) == 2 else 'weighted')

        metrics.update({
            'accuracy': accuracy,
            'auroc': auroc,
            'f1': f1
        })
    else:  # regression
        mae = mean_absolute_error(all_labels, all_preds)
        rmse = np.sqrt(mean_squared_error(all_labels, all_preds))

        metrics.update({
            'mae': mae,
            'rmse': rmse
        })

    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    task_type: str,
    num_classes: int = 2,
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
            outputs = model(signals)
            loss = criterion(outputs, labels)

        if task_type == 'classification':
            probs = F.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        else:  # regression
            preds = outputs
            all_preds.extend(preds.cpu().numpy())

        all_labels.extend(labels.cpu().numpy())
        total_loss += loss.item()

    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = {'loss': total_loss / len(loader)}

    if task_type == 'classification':
        all_probs = np.array(all_probs)
        accuracy = (all_preds == all_labels).mean() * 100

        # AUROC (primary metric)
        try:
            auroc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
        except:
            auroc = 0.5

        f1 = f1_score(all_labels, all_preds, average='binary' if num_classes == 2 else 'weighted')
        precision = precision_score(all_labels, all_preds, average='binary' if num_classes == 2 else 'weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='binary' if num_classes == 2 else 'weighted', zero_division=0)

        # Per-class accuracy
        per_class_acc = {}
        for c in range(num_classes):
            mask = all_labels == c
            if mask.sum() > 0:
                per_class_acc[f'class_{c}_acc'] = (all_preds[mask] == all_labels[mask]).mean() * 100
            else:
                per_class_acc[f'class_{c}_acc'] = 0.0

        metrics.update({
            'accuracy': accuracy,
            'auroc': auroc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            **per_class_acc
        })
    else:  # regression
        mae = mean_absolute_error(all_labels, all_preds)
        rmse = np.sqrt(mean_squared_error(all_labels, all_preds))

        # Additional regression metrics
        residuals = all_labels - all_preds
        me = residuals.mean()  # Mean error (bias)
        sde = residuals.std()  # Standard deviation of error

        metrics.update({
            'mae': mae,
            'rmse': rmse,
            'me': me,
            'sde': sde
        })

    return metrics


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
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
    print(f"  ✓ Checkpoint saved: {path.name}")


# ============================================================================
# DATA LOADING
# ============================================================================

def create_dataloaders(
    data_dir: Path,
    task_config: Dict,
    batch_size: int = 64,
    num_workers: int = 4,
    context_length: int = 1024
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """Create dataloaders using BUTPPGDataset from src/data/butppg_dataset.py.

    This function:
    1. Uses existing BUTPPGDataset (no duplicate code)
    2. Filters invalid samples with filter_missing=True
    3. Loads ALL labels from NPZ files
    4. Wraps with TaskSpecificDataset for task-specific label extraction
    """
    print("\n" + "=" * 70)
    print("CREATING DATALOADERS")
    print("=" * 70)
    print(f"Using BUTPPGDataset from src/data/butppg_dataset.py")
    print(f"Task: {task_config['description']}")
    print(f"Type: {task_config['type']}")
    print(f"Labels: {task_config['label_keys']}")

    # Check directory structure
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    test_dir = data_dir / 'test'

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training directory not found: {train_dir}\n"
            f"Expected structure:\n"
            f"  {data_dir}/\n"
            f"    ├── train/window_*.npz\n"
            f"    ├── val/window_*.npz\n"
            f"    └── test/window_*.npz"
        )

    # Create BUTPPGDataset instances
    print(f"\n✓ Creating training dataset from {train_dir}")
    train_butppg = BUTPPGDataset(
        data_dir=train_dir,
        modality=['ppg', 'ecg'],
        split='train',  # Not used in preprocessed mode, but kept for compatibility
        mode='preprocessed',
        task=task_config['label_keys'][0],  # Primary label for filtering
        return_labels=True,
        filter_missing=True,  # KEY: Filter out invalid samples
        window_sec=context_length / 125.0,  # Match SSL model context
        fs=125
    )

    train_dataset = TaskSpecificDataset(train_butppg, task_config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    # Validation
    val_loader = None
    if val_dir.exists():
        print(f"✓ Creating validation dataset from {val_dir}")
        val_butppg = BUTPPGDataset(
            data_dir=val_dir,
            modality=['ppg', 'ecg'],
            split='val',
            mode='preprocessed',
            task=task_config['label_keys'][0],
            return_labels=True,
            filter_missing=True,
            window_sec=context_length / 125.0,
            fs=125
        )
        val_dataset = TaskSpecificDataset(val_butppg, task_config)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        print("⚠ Validation directory not found")

    # Test
    test_loader = None
    if test_dir.exists():
        print(f"✓ Creating test dataset from {test_dir}")
        test_butppg = BUTPPGDataset(
            data_dir=test_dir,
            modality=['ppg', 'ecg'],
            split='test',
            mode='preprocessed',
            task=task_config['label_keys'][0],
            return_labels=True,
            filter_missing=True,
            window_sec=context_length / 125.0,
            fs=125
        )
        test_dataset = TaskSpecificDataset(test_butppg, task_config)
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


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Enhanced fine-tuning for BUT-PPG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model
    parser.add_argument('--pretrained', type=str, required=True,
                       help='Path to SSL pretrained checkpoint')

    # Data
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing BUT-PPG windowed data')
    parser.add_argument('--task', type=str, default='quality',
                       choices=list(TASK_CONFIGS.keys()),
                       help='Task to fine-tune on')

    # Domain adaptation
    parser.add_argument('--adaptation', type=str,
                       choices=['projection', 'adversarial', 'none'],
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
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--no-amp', action='store_true',
                       help='Disable automatic mixed precision')

    # Output
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for checkpoints')

    # Misc
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()

    # Override adaptation if no-adaptation is set
    if args.no_adaptation:
        args.adaptation = 'none'

    # Get task config
    task_config = TASK_CONFIGS[args.task]
    task_type = task_config['type']
    primary_metric = task_config['primary_metric']

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
    print("ENHANCED BUT-PPG FINE-TUNING")
    print("=" * 70)
    print(f"Task: {args.task} ({task_config['description']})")
    print(f"Type: {task_type}")
    print(f"Primary metric: {primary_metric.upper()}")
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

    # CRITICAL: Determine if SSL used IBM pretrained
    # SSL with patch_size=64, context=1024, d_model=192 used IBM pretrained
    # (IBM pretrained is loaded with patch=128 config, but auto-adapts to patch=64)
    ssl_used_ibm_pretrained = (
        context_length == 1024 and
        d_model == 192 and
        patch_size == 64
    )

    if ssl_used_ibm_pretrained:
        print(f"\n  ℹ️  SSL checkpoint used IBM pretrained TTM-Enhanced (946K params)")
        print(f"     Detected: context={context_length}, d_model={d_model}, patch={patch_size}")
        print(f"     Will load IBM pretrained (patch=128 config), which auto-adapts to patch={patch_size}")
        # CRITICAL: Use patch_size=128 to load IBM pretrained (same as SSL training)
        # IBM's TTM will auto-adapt from 8 patches (128) to 16 patches (64)
        model_patch_size = 128
    else:
        print(f"\n  ⚠️  SSL checkpoint used custom architecture (not IBM pretrained)")
        print(f"     Will create fresh TTM with patch={patch_size}")
        model_patch_size = patch_size

    # Create SSL encoder
    print(f"\nCreating SSL encoder...")
    encoder = TTMAdapter(
        variant='ibm-granite/granite-timeseries-ttm-r1',
        task='ssl',
        input_channels=2,
        context_length=context_length,
        patch_size=model_patch_size,  # CRITICAL: Use model_patch_size, not hardcoded!
        d_model=d_model,
        use_real_ttm=True
    )

    # Move model to device BEFORE auto-adaptation
    encoder = encoder.to(args.device)

    # CRITICAL: Trigger auto-adaptation BEFORE loading SSL weights
    # IBM TTM auto-adapts patch_size during first forward pass
    # We need this to happen BEFORE loading SSL weights so architectures match
    if ssl_used_ibm_pretrained:
        print(f"\n🔧 Triggering TTM auto-adaptation...")
        print(f"  Current model patch_size: {encoder.patch_size}")

        with torch.no_grad():
            dummy_input = torch.randn(1, 2, context_length).to(args.device)
            try:
                _ = encoder.get_encoder_output(dummy_input)
                print(f"  ✓ Auto-adaptation complete")
                print(f"  Updated model patch_size: {encoder.patch_size}")
                del dummy_input, _
                if args.device == 'cuda':
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"  ⚠️  Auto-adaptation failed: {e}")
                print(f"     Proceeding with current patch_size={encoder.patch_size}")

        # Verify patch_size matches SSL checkpoint
        print(f"\nVerifying model configuration:")
        print(f"  Model patch_size: {encoder.patch_size}")
        print(f"  SSL checkpoint patch_size: {patch_size}")

        if encoder.patch_size != patch_size:
            print(f"  ⚠️  CRITICAL: Model patch_size ({encoder.patch_size}) != SSL checkpoint ({patch_size})")
            print(f"     Weight loading may fail!")
        else:
            print(f"  ✅ Patch sizes match - ready to load SSL weights")

    # Load SSL weights
    encoder.load_state_dict(encoder_state, strict=False)
    encoder.eval()
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
    if task_type == 'classification':
        num_outputs = task_config['num_classes']
    else:
        num_outputs = task_config['num_outputs']

    task_head = nn.Linear(d_model, num_outputs)
    print(f"✓ Created task head ({d_model} → {num_outputs})")

    # Combine into full model
    model = DomainAdaptedModel(encoder, adapter, task_head, task_type)
    model = model.to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✓ Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=Path(args.data_dir),
        task_config=task_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        context_length=context_length
    )

    if val_loader is None and test_loader is not None:
        val_loader = test_loader
        print("\n⚠ Using test set for validation")

    # Setup training
    if task_type == 'classification':
        # Compute class weights for imbalanced data
        print("\n⚙️  Computing class weights...")
        all_labels = []
        for _, labels in train_loader:
            all_labels.extend(labels.numpy())
        all_labels = np.array(all_labels)

        class_counts = np.bincount(all_labels)
        total_samples = len(all_labels)
        class_weights = total_samples / (len(class_counts) * class_counts)
        class_weights = torch.FloatTensor(class_weights).to(args.device)

        for i, (count, weight) in enumerate(zip(class_counts, class_weights)):
            print(f"  Class {i}: {count} samples, weight: {weight:.3f}")

        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.MSELoss()

    use_amp = not args.no_amp and torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None

    # Early stopping
    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=0.001,
        mode='max' if primary_metric == 'auroc' else 'min'
    ) if args.early_stopping else None

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'phase': []
    }

    # Add task-specific metrics to history
    if task_type == 'classification':
        history.update({
            'train_auroc': [],
            'train_acc': [],
            'val_auroc': [],
            'val_acc': []
        })
        best_val_score = 0.0  # Maximize AUROC
    else:
        history.update({
            'train_mae': [],
            'train_rmse': [],
            'val_mae': [],
            'val_rmse': []
        })
        best_val_score = float('inf')  # Minimize MAE

    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Create progressive fine-tuner
    tuner = ProgressiveFineTuner(
        encoder,
        adapter if adapter else nn.Identity(),
        task_head
    )

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
            args.device, task_type, use_amp, scaler, args.gradient_clip, epoch
        )

        if val_loader is not None:
            val_metrics = evaluate(
                model, val_loader, criterion, args.device, task_type,
                task_config.get('num_classes', 2), use_amp, "Val"
            )
        else:
            val_metrics = train_metrics.copy()

        print(f"\nEpoch {epoch+1}/{args.head_only_epochs}")
        if task_type == 'classification':
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, AUROC: {train_metrics['auroc']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, AUROC: {val_metrics['auroc']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
        else:
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.4f}, RMSE: {train_metrics['rmse']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.4f}, RMSE: {val_metrics['rmse']:.4f}")

        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['phase'].append('phase1_head_only')

        if task_type == 'classification':
            history['train_auroc'].append(train_metrics['auroc'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_auroc'].append(val_metrics['auroc'])
            history['val_acc'].append(val_metrics['accuracy'])
            current_score = val_metrics['auroc']
        else:
            history['train_mae'].append(train_metrics['mae'])
            history['train_rmse'].append(train_metrics['rmse'])
            history['val_mae'].append(val_metrics['mae'])
            history['val_rmse'].append(val_metrics['rmse'])
            current_score = val_metrics['mae']

        # Save best model
        is_better = (current_score > best_val_score) if task_type == 'classification' else (current_score < best_val_score)
        if is_better:
            best_val_score = current_score
            save_checkpoint(
                output_dir / 'best_model.pt',
                model, optimizer, epoch, val_metrics, vars(args)
            )
            print(f"  🏆 Best model saved ({primary_metric.upper()}: {current_score:.4f})")

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
                args.device, task_type, use_amp, scaler, args.gradient_clip,
                args.head_only_epochs + epoch
            )

            if val_loader is not None:
                val_metrics = evaluate(
                    model, val_loader, criterion, args.device, task_type,
                    task_config.get('num_classes', 2), use_amp, "Val"
                )
            else:
                val_metrics = train_metrics.copy()

            print(f"\nEpoch {args.head_only_epochs + epoch + 1}/{args.head_only_epochs + args.partial_epochs}")
            if task_type == 'classification':
                print(f"  Train - Loss: {train_metrics['loss']:.4f}, AUROC: {train_metrics['auroc']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
                print(f"  Val   - Loss: {val_metrics['loss']:.4f}, AUROC: {val_metrics['auroc']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
            else:
                print(f"  Train - Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.4f}, RMSE: {train_metrics['rmse']:.4f}")
                print(f"  Val   - Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.4f}, RMSE: {val_metrics['rmse']:.4f}")

            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['phase'].append('phase2_partial')

            if task_type == 'classification':
                history['train_auroc'].append(train_metrics['auroc'])
                history['train_acc'].append(train_metrics['accuracy'])
                history['val_auroc'].append(val_metrics['auroc'])
                history['val_acc'].append(val_metrics['accuracy'])
                current_score = val_metrics['auroc']
            else:
                history['train_mae'].append(train_metrics['mae'])
                history['train_rmse'].append(train_metrics['rmse'])
                history['val_mae'].append(val_metrics['mae'])
                history['val_rmse'].append(val_metrics['rmse'])
                current_score = val_metrics['mae']

            is_better = (current_score > best_val_score) if task_type == 'classification' else (current_score < best_val_score)
            if is_better:
                best_val_score = current_score
                save_checkpoint(
                    output_dir / 'best_model.pt',
                    model, optimizer, epoch, val_metrics, vars(args)
                )
                print(f"  🏆 Best model saved ({primary_metric.upper()}: {current_score:.4f})")

            # Early stopping
            if early_stopping is not None:
                if early_stopping(current_score):
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
                args.device, task_type, use_amp, scaler, args.gradient_clip,
                args.head_only_epochs + args.partial_epochs + epoch
            )

            if val_loader is not None:
                val_metrics = evaluate(
                    model, val_loader, criterion, args.device, task_type,
                    task_config.get('num_classes', 2), use_amp, "Val"
                )
            else:
                val_metrics = train_metrics.copy()

            print(f"\nEpoch {args.head_only_epochs + args.partial_epochs + epoch + 1}/{args.epochs}")
            if task_type == 'classification':
                print(f"  Train - Loss: {train_metrics['loss']:.4f}, AUROC: {train_metrics['auroc']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
                print(f"  Val   - Loss: {val_metrics['loss']:.4f}, AUROC: {val_metrics['auroc']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
            else:
                print(f"  Train - Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.4f}, RMSE: {train_metrics['rmse']:.4f}")
                print(f"  Val   - Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.4f}, RMSE: {val_metrics['rmse']:.4f}")

            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['phase'].append('phase3_full')

            if task_type == 'classification':
                history['train_auroc'].append(train_metrics['auroc'])
                history['train_acc'].append(train_metrics['accuracy'])
                history['val_auroc'].append(val_metrics['auroc'])
                history['val_acc'].append(val_metrics['accuracy'])
                current_score = val_metrics['auroc']
            else:
                history['train_mae'].append(train_metrics['mae'])
                history['train_rmse'].append(train_metrics['rmse'])
                history['val_mae'].append(val_metrics['mae'])
                history['val_rmse'].append(val_metrics['rmse'])
                current_score = val_metrics['mae']

            is_better = (current_score > best_val_score) if task_type == 'classification' else (current_score < best_val_score)
            if is_better:
                best_val_score = current_score
                save_checkpoint(
                    output_dir / 'best_model.pt',
                    model, optimizer, epoch, val_metrics, vars(args)
                )
                print(f"  🏆 Best model saved ({primary_metric.upper()}: {current_score:.4f})")

            if early_stopping is not None:
                if early_stopping(current_score):
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
        print("\nEvaluating on test set...")
        test_metrics = evaluate(
            model, test_loader, criterion, args.device, task_type,
            task_config.get('num_classes', 2), use_amp, "Test"
        )

        print(f"\n🏆 Test Results:")
        if task_type == 'classification':
            print(f"  AUROC: {test_metrics['auroc']:.4f}  ← PRIMARY METRIC")
            print(f"  Accuracy: {test_metrics['accuracy']:.2f}%")
            print(f"  F1 Score: {test_metrics['f1']:.4f}")
            print(f"  Precision: {test_metrics['precision']:.4f}")
            print(f"  Recall: {test_metrics['recall']:.4f}")
            for key, val in test_metrics.items():
                if 'class_' in key:
                    print(f"  {key}: {val:.2f}%")
        else:
            print(f"  MAE: {test_metrics['mae']:.4f}  ← PRIMARY METRIC")
            print(f"  RMSE: {test_metrics['rmse']:.4f}")
            print(f"  ME (bias): {test_metrics['me']:.4f}")
            print(f"  SDE: {test_metrics['sde']:.4f}")

        with open(output_dir / 'test_metrics.json', 'w') as f:
            json.dump(test_metrics, f, indent=2)

    # Save training history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation {primary_metric.upper()}: {best_val_score:.4f}")
    if test_loader is not None:
        print(f"Test {primary_metric.upper()}: {test_metrics[primary_metric]:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
