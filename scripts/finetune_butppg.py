#!/usr/bin/env python3
"""Fine-tuning Script for BUT-PPG Quality Classification.

This script fine-tunes a 2-channel SSL pretrained model on 2-channel BUT-PPG data
for PPG quality classification (good vs poor).

Data Format:
- BUT-PPG data: 2 channels [PPG, ECG] in windowed format
- Each window: individual NPZ file with shape [2, 1250] (10s @ 125Hz)
- Resampled to [2, 512] for IBM TTM-Enhanced (8 patches × 64)
- No channel inflation needed (both SSL and fine-tuning use 2 channels)

Strategy:
1. Load 2-channel pretrained checkpoint from SSL pre-training
2. Add classification head for binary quality prediction
3. Staged unfreezing:
   - Stage 1 (3-5 epochs): Head-only training
   - Stage 2 (remaining epochs): Unfreeze last N encoder blocks
   - Stage 3 (optional): Full fine-tuning at very low LR
4. Monitor validation accuracy and save best model

Usage:
    # Quick test (1 epoch)
    python scripts/finetune_butppg.py \
        --pretrained artifacts/foundation_model/best_model.pt \
        --data-dir data/processed/butppg/windows_with_labels \
        --unfreeze-last-n 2 \
        --epochs 1 \
        --lr 2e-5 \
        --output-dir artifacts/but_ppg_finetuned

    # Full training (30 epochs with staged unfreezing)
    python scripts/finetune_butppg.py \
        --pretrained artifacts/foundation_model/best_model.pt \
        --data-dir data/processed/butppg/windows_with_labels \
        --epochs 30 \
        --head-only-epochs 2 \
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
from torch.optim.swa_utils import AveragedModel, SWALR  # For Stochastic Weight Averaging
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy.ndimage import gaussian_filter1d

from src.models.channel_utils import unfreeze_last_n_blocks


def load_ssl_encoder_from_checkpoint(checkpoint_path: str, device: str = 'cuda'):
    """Load the SSL encoder directly from checkpoint.

    This function recreates the encoder used in SSL training by:
    1. Loading the SSL checkpoint
    2. Detecting architecture from weights
    3. Creating a new encoder with IBM pretrained
    4. Loading the SSL-trained weights

    This ensures we use the EXACT encoder from SSL (with IBM TTM inside),
    not a new TTMAdapter that might fall back to CNN.

    Args:
        checkpoint_path: Path to SSL checkpoint
        device: Device to load on

    Returns:
        encoder: SSL-trained encoder with IBM TTM
        config: Architecture configuration
    """
    print(f"\n📦 Loading SSL encoder from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Get encoder state dict
    if 'encoder_state_dict' in checkpoint:
        encoder_state = checkpoint['encoder_state_dict']
        print("  ✓ Found encoder_state_dict")
    else:
        raise ValueError(f"Checkpoint missing encoder_state_dict. Keys: {list(checkpoint.keys())}")

    # Detect architecture from weights
    patcher_keys = [k for k in encoder_state.keys() if 'patcher.weight' in k]
    if not patcher_keys:
        raise ValueError("Cannot find patcher weights to detect architecture")

    patcher_weight = encoder_state[patcher_keys[0]]
    d_model = patcher_weight.shape[0]
    patch_size = patcher_weight.shape[1]

    print(f"  Detected from weights:")
    print(f"    d_model: {d_model}")
    print(f"    patch_size: {patch_size}")

    # Get config from checkpoint or infer
    if 'config' in checkpoint:
        config = checkpoint['config']
        context_length = config.get('context_length', 512)
    else:
        # IBM TTM-Enhanced with patch=64 uses context=512
        context_length = 512 if patch_size == 64 else 1024
        print(f"  ⚠️  No config found, inferred context_length={context_length}")

    print(f"    context_length: {context_length}")
    print(f"    num_patches: {context_length // patch_size}")

    # Import the init function from SSL script
    # This creates encoder with IBM pretrained (same as SSL training)
    import sys
    from pathlib import Path
    ssl_script_dir = Path(__file__).parent
    sys.path.insert(0, str(ssl_script_dir))

    # Import the SSL initialization function
    from continue_ssl_butppg_quality import init_ibm_pretrained

    print(f"\n  Creating encoder with IBM pretrained (matching SSL)...")
    encoder, _ = init_ibm_pretrained(
        variant='ibm-granite/granite-timeseries-ttm-r1',
        context_length=1024,  # Loading context (IBM's API)
        patch_size=patch_size,
        num_channels=2,
        device=device
    )

    print(f"  ✓ Encoder created with IBM TTM backbone")

    # Load SSL-trained weights
    print(f"\n  Loading SSL-trained weights...")
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)

    if missing:
        print(f"    ⚠️  Missing keys ({len(missing)}): {missing[:3]}...")
    if unexpected:
        print(f"    ⚠️  Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")

    if not missing and not unexpected:
        print(f"    ✓ All weights loaded successfully!")

    # Return encoder and config
    config_dict = {
        'context_length': context_length,
        'patch_size': patch_size,
        'd_model': d_model,
        'num_patches': context_length // patch_size
    }

    return encoder, config_dict


class BUTPPGDataset(Dataset):
    """Dataset for BUT-PPG quality classification.

    Expected data format (windowed):
    - Directory containing individual window files: window_*.npz
    - Each window NPZ contains:
      - 'signal': [2, 1250] - 2 channels (PPG, ECG), 10s @ 125Hz
      - 'quality': float - quality label (0=poor, 1=good)
      - Additional metadata: 'hr', 'motion', 'bp_systolic', 'bp_diastolic', etc.

    Channels:
    0: PPG (photoplethysmogram)
    1: ECG (electrocardiogram)

    Note: BUT-PPG dataset does NOT contain accelerometer data.
    Note: Signals are resampled to target_length (512 for IBM TTM-Enhanced) if specified.
    """

    def __init__(self, data_dir: Path, normalize: bool = True, target_length: int = None):
        """Initialize BUT-PPG dataset from windowed format.

        Args:
            data_dir: Path to directory containing window_*.npz files
            normalize: Whether to apply z-score normalization per channel
            target_length: If specified, resize signals to this length (for SSL model compatibility)
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

            # Load signal [2, T] where T is typically 1250 for BUT-PPG (10s @ 125Hz)
            if 'signal' not in data:
                raise KeyError(f"Expected 'signal' key in {window_file}, found: {list(data.keys())}")

            signal = data['signal']  # [2, T] - will be resampled to target_length if specified

            # Load quality label
            if 'quality' in data:
                quality = data['quality']
                # Convert to binary: assume quality is already 0/1 or convert threshold
                label = int(quality) if not np.isnan(quality) else 0
            else:
                # If no quality label, default to 0 (poor)
                label = 0

            signals_list.append(signal)
            labels_list.append(label)

        # Stack into tensors
        self.signals = torch.from_numpy(np.stack(signals_list, axis=0)).float()  # [N, 2, T]
        self.labels = torch.tensor(labels_list, dtype=torch.long)  # [N]

        # Validate shapes
        N, C, T = self.signals.shape
        assert C == 2, f"Expected 2 channels, got {C}"
        assert len(self.labels) == N, f"Label count mismatch: {len(self.labels)} vs {N}"

        # Resize if needed to match SSL model's context_length
        if self.target_length is not None and T != self.target_length:
            print(f"  ⚠️  Resizing signals from {T} to {self.target_length} samples (SSL model requirement)")
            # Use interpolation to resize
            resized_signals = torch.nn.functional.interpolate(
                self.signals,  # [N, C, T]
                size=self.target_length,
                mode='linear',
                align_corners=False
            )
            self.signals = resized_signals
            N, C, T = self.signals.shape
            print(f"  ✓ Resized to: {self.signals.shape}")

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
        print(f"  Loaded {N} samples from {len(window_files)} windows:")
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
    """Create dataloaders for BUT-PPG.

    Expected structure (windowed format from prepare_all_data.py):
    data_dir/
      ├── train/
      │   ├── window_000001.npz
      │   ├── window_000002.npz
      │   └── ...
      ├── val/
      │   └── window_*.npz
      └── test/
          └── window_*.npz

    Args:
        data_dir: Directory containing train/val/test subdirectories
        batch_size: Batch size for dataloaders
        num_workers: Number of data loading workers
        use_val: Whether to create validation loader
        target_length: If specified, resize all signals to this length (for SSL compatibility)

    Returns:
        train_loader, val_loader, test_loader
    """
    data_dir = Path(data_dir)

    # Create datasets
    print("\n" + "=" * 70)
    print("CREATING BUT-PPG DATALOADERS")
    print("=" * 70)
    print(f"Data directory: {data_dir}")

    # Find split directories
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    test_dir = data_dir / 'test'

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training directory not found: {train_dir}\n"
            f"\nExpected structure:\n"
            f"  {data_dir}/\n"
            f"    ├── train/\n"
            f"    │   └── window_*.npz\n"
            f"    ├── val/\n"
            f"    └── test/\n"
            f"\nMake sure to run data preparation first:\n"
            f"  python scripts/prepare_all_data.py --dataset butppg --mode fasttrack\n\n"
            f"Or if you have data elsewhere, use:\n"
            f"  --data-dir /path/to/your/butppg/windows_with_labels"
        )

    # Create training loader
    print(f"\n✓ Found training directory: {train_dir}")
    train_dataset = BUTPPGDataset(train_dir, normalize=True, target_length=target_length)

    # CRITICAL FIX: Use normal shuffling (NOT balanced sampling)
    # Balanced sampling creates distribution mismatch:
    #   - Training batches: 50/50 Poor/Good (from balanced sampler)
    #   - Validation: 80/20 Poor/Good (real distribution)
    # Model trained on balanced distribution but tested on imbalanced → collapse!
    # Solution: Let model see real distribution during training

    print(f"  Using normal shuffling (real distribution matching validation)")

    # Get labels for statistics
    train_labels = train_dataset.labels.numpy()
    class_sample_counts = np.bincount(train_labels)

    print(f"  Training set: Poor={class_sample_counts[0]} ({100*class_sample_counts[0]/len(train_labels):.1f}%), Good={class_sample_counts[1]} ({100*class_sample_counts[1]/len(train_labels):.1f}%)")
    print(f"  ✓ Normal shuffling (batches match real 80/20 distribution)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Normal shuffling - let model see real distribution
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop incomplete batches for stability
    )

    # Create validation loader
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
        print("⚠ Validation directory not found, will use test set for validation")

    # Create test loader
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

        # ✅ ADVANCED: Apply augmentations after epoch 5
        # Early epochs: No augmentation (let head learn basic patterns)
        # Later epochs: Quality-aware augmentation + Mixup/CutMix
        apply_augmentation = epoch >= 5

        if apply_augmentation:
            # 1. Quality-aware augmentation (50% probability)
            signals = quality_aware_augmentation(signals, labels, augmentation_prob=0.5)

            # 2. Randomly choose between Mixup, CutMix, or none
            aug_choice = np.random.rand()

            if aug_choice < 0.3:  # 30% Mixup
                signals, labels_a, labels_b, lam = mixup_data(signals, labels, alpha=0.2)
                # Forward pass with mixup
                with autocast(enabled=use_amp):
                    logits = model(signals)
                    loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
            elif aug_choice < 0.5:  # 20% CutMix
                signals, labels_a, labels_b, lam = cutmix_data(signals, labels, alpha=1.0)
                # Forward pass with cutmix
                with autocast(enabled=use_amp):
                    logits = model(signals)
                    loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
            else:  # 50% No mixup/cutmix (just quality-aware aug)
                with autocast(enabled=use_amp):
                    logits = model(signals)
                    loss = criterion(logits, labels)
        else:
            # No augmentation in early epochs
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
    all_probs = []  # CRITICAL: Collect probabilities for AUROC

    pbar = tqdm(loader, desc=f"[{desc}]", leave=False)

    with torch.no_grad():  # Add no_grad for efficiency
        for signals, labels in pbar:
            signals = signals.to(device)
            labels = labels.to(device)

            with autocast(enabled=use_amp):
                logits = model(signals)
                loss = criterion(logits, labels)

            # Get predictions and probabilities
            preds = logits.argmax(dim=1)
            probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of Good class (class 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            pbar.set_postfix({
                'loss': total_loss / len(loader),
                'acc': 100.0 * correct / total
            })

    # Compute additional metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # CRITICAL: Compute AUROC (primary metric for imbalanced data)
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        # Handle case where only one class present in batch
        auroc = 0.0

    # Per-class accuracy (return as fraction 0-1, not percentage)
    class_0_mask = all_labels == 0
    class_1_mask = all_labels == 1

    class_0_acc = (all_preds[class_0_mask] == all_labels[class_0_mask]).mean() if class_0_mask.sum() > 0 else 0.0
    class_1_acc = (all_preds[class_1_mask] == all_labels[class_1_mask]).mean() if class_1_mask.sum() > 0 else 0.0

    return {
        'loss': total_loss / len(loader),
        'accuracy': 100.0 * correct / total,
        'auroc': auroc,  # CRITICAL: Primary metric for saving best model
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

    # Check for train/val/test subdirectories
    print("\n📁 Expected structure:")
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        if split_dir.exists():
            window_files = list(split_dir.glob('window_*.npz'))
            print(f"  ✅ {split}/ ({len(window_files)} windows)")
        else:
            print(f"  ❌ {split}/ (not found)")

    # Sample and inspect a few window files
    print("\n🔍 Inspecting sample window files:")
    sample_count = 0
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue

        window_files = sorted(split_dir.glob('window_*.npz'))
        if not window_files:
            print(f"\n  ⚠️  {split}/: No window files found")
            continue

        # Inspect first window file
        sample_file = window_files[0]
        try:
            data = np.load(sample_file)
            rel_path = sample_file.relative_to(data_dir)
            print(f"\n  {rel_path}:")
            print(f"    Keys: {list(data.keys())}")

            # Check signal
            if 'signal' in data:
                signal = data['signal']
                print(f"    Signal shape: {signal.shape}")

                if len(signal.shape) == 2:
                    C, T = signal.shape
                    status = []
                    if C == 2:
                        status.append("✅ 2 channels (PPG, ECG)")
                    else:
                        status.append(f"⚠️  {C} channels (expected 2)")

                    if T == 1250:
                        status.append("✅ 1250 timesteps (10s @ 125Hz)")
                    elif T == 512:
                        status.append("✅ 512 timesteps (already resampled for TTM)")
                    else:
                        status.append(f"⚠️  {T} timesteps (expected 1250 or 512)")

                    print(f"      Status: {', '.join(status)}")
            else:
                print(f"    ⚠️  No 'signal' key found")

            # Check labels
            if 'quality' in data:
                quality = data['quality']
                print(f"    Quality label: {quality}")
            else:
                print(f"    ⚠️  No 'quality' key found")

            sample_count += 1

        except Exception as e:
            print(f"    ❌ Error loading: {e}")

    if sample_count == 0:
        print("\n  ❌ No valid window files found!")

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
        default='data/processed/butppg/windows_with_labels',
        help='Directory containing BUT-PPG windowed data (train/val/test subdirectories)'
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
        default=10,
        help='Number of epochs for head-only training (Stage 1) - CRITICAL: increased to 10 for proper head initialization'
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
    print(f"Channels: 2 (PPG + ECG)")
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

    # Load pretrained SSL encoder (CRITICAL FIX: Use encoder from SSL directly)
    print("\n" + "=" * 70)
    print("LOADING SSL ENCODER")
    print("=" * 70)
    print("Using Option 3: Load complete SSL encoder with IBM TTM inside")
    print("This avoids fallback CNN and preserves all SSL features!")

    # Use our new function to load SSL encoder
    try:
        ssl_encoder, ssl_config = load_ssl_encoder_from_checkpoint(
            checkpoint_path=args.pretrained,
            device=args.device
        )

        context_length = ssl_config['context_length']
        patch_size = ssl_config['patch_size']
        d_model = ssl_config['d_model']
        num_patches = ssl_config['num_patches']

        print(f"\n✅ SSL encoder loaded successfully!")
        print(f"  Context length: {context_length}")
        print(f"  Patch size: {patch_size}")
        print(f"  d_model: {d_model}")
        print(f"  Num patches: {num_patches}")
        print(f"  Total encoder params: {sum(p.numel() for p in ssl_encoder.parameters()):,}")

    except Exception as e:
        print(f"\n❌ FATAL: Failed to load SSL encoder: {e}")
        import traceback
        traceback.print_exc()
        print("\nCannot proceed without SSL encoder - exiting")
        import sys
        sys.exit(1)

    # Freeze SSL encoder for Stage 1
    for param in ssl_encoder.parameters():
        param.requires_grad = False
    print(f"  ✓ Encoder frozen (will train head only in Stage 1)")

    # ========================================================================
    # ADVANCED TECHNIQUES: 10 High-Impact Modules and Functions
    # ========================================================================

    # TECHNIQUE 1: ATTENTION POOLING (⭐⭐⭐⭐⭐)
    class AttentionPooling(nn.Module):
        """Learned attention-based pooling over patches.

        Replaces simple mean pooling with learned attention weights,
        allowing the model to focus on the most discriminative patches.
        """

        def __init__(self, d_model: int):
            super().__init__()
            self.attention = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.Tanh(),
                nn.Linear(d_model // 2, 1)
            )

        def forward(self, x):
            """
            Args:
                x: [batch, num_patches, d_model]
            Returns:
                pooled: [batch, d_model]
            """
            attn_weights = F.softmax(self.attention(x), dim=1)  # [B, P, 1]
            pooled = (x * attn_weights).sum(dim=1)  # [B, D]
            return pooled

    # TECHNIQUE 2: MULTI-SCALE FEATURE FUSION (⭐⭐⭐⭐)
    class MultiScaleClassificationHead(nn.Module):
        """Classification head with multi-scale feature fusion.

        Extracts features from multiple encoder depths and fuses them
        for richer representation learning.
        """

        def __init__(self, encoder, d_model: int, num_classes: int):
            super().__init__()
            self.encoder = encoder
            self.d_model = d_model
            self.num_classes = num_classes

            # Attention pooling for each scale
            self.pooling = AttentionPooling(d_model)

            # Feature fusion: combine multi-scale features
            # Total features: d_model (final) + d_model (middle) + d_model (early) = 3 * d_model
            self.fusion = nn.Sequential(
                nn.Linear(3 * d_model, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Dropout(0.3)
            )

            # Classification head
            self.classifier = nn.Linear(d_model, num_classes)

            # Store config for compatibility
            self.context_length = getattr(encoder, 'context_length', None)
            self.patch_size = getattr(encoder, 'patch_size', None)

        def forward(self, x):
            """Forward pass with multi-scale feature extraction."""
            # Get final encoder output: [B, C, T] → [B, P, D]
            final_features = self.encoder.get_encoder_output(x)

            # Extract intermediate features if available
            # For IBM TTM: try to get features from different transformer blocks
            if hasattr(self.encoder, 'backbone') and hasattr(self.encoder.backbone, 'encoder'):
                try:
                    # Get encoder blocks
                    blocks = self.encoder.backbone.encoder.layers
                    num_blocks = len(blocks)

                    # Forward through early blocks (1/3)
                    early_idx = num_blocks // 3
                    x_early = x
                    for i in range(early_idx):
                        x_early = blocks[i](x_early)
                    early_features = x_early  # [B, P, D]

                    # Forward through middle blocks (2/3)
                    middle_idx = 2 * num_blocks // 3
                    x_middle = early_features
                    for i in range(early_idx, middle_idx):
                        x_middle = blocks[i](x_middle)
                    middle_features = x_middle  # [B, P, D]

                except (AttributeError, IndexError):
                    # Fallback: use final features for all scales
                    early_features = final_features
                    middle_features = final_features
            else:
                # Fallback: use final features for all scales
                early_features = final_features
                middle_features = final_features

            # Pool each scale with attention
            early_pooled = self.pooling(early_features)    # [B, D]
            middle_pooled = self.pooling(middle_features)  # [B, D]
            final_pooled = self.pooling(final_features)    # [B, D]

            # Concatenate multi-scale features
            multi_scale = torch.cat([early_pooled, middle_pooled, final_pooled], dim=1)  # [B, 3*D]

            # Fuse features
            fused = self.fusion(multi_scale)  # [B, D]

            # Classify
            logits = self.classifier(fused)  # [B, num_classes]

            return logits

        def get_encoder_output(self, x):
            """Get encoder features (for analysis)."""
            return self.encoder.get_encoder_output(x)

    # TECHNIQUE 3 & 4: MIXUP AND CUTMIX AUGMENTATION (⭐⭐⭐⭐⭐)
    def mixup_data(x, y, alpha=0.2):
        """Mixup augmentation: interpolate samples and labels.

        Args:
            x: Input signals [B, C, T]
            y: Labels [B]
            alpha: Beta distribution parameter (0.2 recommended)

        Returns:
            mixed_x: Mixed signals [B, C, T]
            y_a, y_b: Original labels for both samples
            lam: Mixing coefficient
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam

    def cutmix_data(x, y, alpha=1.0):
        """CutMix augmentation: replace time segments between samples.

        Args:
            x: Input signals [B, C, T]
            y: Labels [B]
            alpha: Beta distribution parameter (1.0 recommended)

        Returns:
            mixed_x: Mixed signals [B, C, T]
            y_a, y_b: Original labels for both samples
            lam: Mixing coefficient (proportion of original sample)
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        # Get time dimension
        time_length = x.size(2)

        # Calculate cut length
        cut_length = int(time_length * (1 - lam))

        # Random start position
        start_pos = np.random.randint(0, time_length - cut_length + 1)

        # Create mixed sample
        mixed_x = x.clone()
        mixed_x[:, :, start_pos:start_pos + cut_length] = x[index, :, start_pos:start_pos + cut_length]

        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam

    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        """Loss function for mixup/cutmix."""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    # TECHNIQUE 9: QUALITY-AWARE AUGMENTATION (⭐⭐⭐⭐⭐)
    def quality_aware_augmentation(signals, labels, augmentation_prob=0.5):
        """Apply different augmentations based on quality label.

        Poor quality (0): Apply "improvements" (smoothing, denoising)
        Good quality (1): Apply "degradations" (noise, dropouts)

        This helps the model learn robust quality discrimination.

        Args:
            signals: [B, C, T] tensor
            labels: [B] tensor (0=Poor, 1=Good)
            augmentation_prob: Probability of applying augmentation

        Returns:
            augmented_signals: [B, C, T] tensor
        """
        augmented = signals.clone()

        for i in range(signals.size(0)):
            if np.random.rand() > augmentation_prob:
                continue

            label = labels[i].item()

            if label == 0:  # Poor quality → apply "improvements"
                # Smoothing (simulate quality improvement)
                signal = signals[i].cpu().numpy()  # [C, T]
                for c in range(signal.shape[0]):
                    signal[c] = gaussian_filter1d(signal[c], sigma=2.0)
                augmented[i] = torch.from_numpy(signal).to(signals.device)

            else:  # Good quality → apply "degradations"
                # Add noise (simulate quality degradation)
                noise_level = 0.05
                noise = torch.randn_like(signals[i]) * noise_level
                augmented[i] = signals[i] + noise

                # Random dropout (simulate signal loss)
                if np.random.rand() < 0.5:
                    dropout_length = int(signals.size(2) * 0.1)  # 10% dropout
                    start = np.random.randint(0, signals.size(2) - dropout_length)
                    augmented[i, :, start:start + dropout_length] = 0

        return augmented

    # TECHNIQUE 5: DISCRIMINATIVE LEARNING RATES (⭐⭐⭐⭐⭐)
    def get_discriminative_params(model, base_lr, head_multiplier=50):
        """Get parameter groups with layer-wise learning rate decay.

        Args:
            model: Model with encoder and head/classifier
            base_lr: Base learning rate for early encoder layers
            head_multiplier: LR multiplier for head (e.g., 50x)

        Returns:
            List of parameter groups for optimizer
        """
        param_groups = []

        # Head/Classifier: highest LR (50x)
        if hasattr(model, 'classifier'):
            param_groups.append({
                'params': model.classifier.parameters(),
                'lr': base_lr * head_multiplier,
                'name': 'classifier'
            })
        elif hasattr(model, 'head'):
            param_groups.append({
                'params': model.head.parameters(),
                'lr': base_lr * head_multiplier,
                'name': 'head'
            })

        # Fusion layer: high LR (25x)
        if hasattr(model, 'fusion'):
            param_groups.append({
                'params': model.fusion.parameters(),
                'lr': base_lr * 25,
                'name': 'fusion'
            })

        # Pooling: high LR (25x)
        if hasattr(model, 'pooling'):
            param_groups.append({
                'params': model.pooling.parameters(),
                'lr': base_lr * 25,
                'name': 'pooling'
            })

        # Encoder blocks with layer-wise decay
        if hasattr(model, 'encoder'):
            encoder = model.encoder
            if hasattr(encoder, 'backbone') and hasattr(encoder.backbone, 'encoder'):
                blocks = encoder.backbone.encoder.layers
                num_blocks = len(blocks)

                # Last 1/3 of blocks: 5x LR
                for i in range(2 * num_blocks // 3, num_blocks):
                    param_groups.append({
                        'params': blocks[i].parameters(),
                        'lr': base_lr * 5,
                        'name': f'encoder_late_block_{i}'
                    })

                # Middle 1/3 of blocks: 2x LR
                for i in range(num_blocks // 3, 2 * num_blocks // 3):
                    param_groups.append({
                        'params': blocks[i].parameters(),
                        'lr': base_lr * 2,
                        'name': f'encoder_middle_block_{i}'
                    })

                # First 1/3 of blocks: 1x LR (base)
                for i in range(num_blocks // 3):
                    param_groups.append({
                        'params': blocks[i].parameters(),
                        'lr': base_lr,
                        'name': f'encoder_early_block_{i}'
                    })

        return param_groups

    # TECHNIQUE 6: FOCAL LOSS (⭐⭐⭐⭐)
    class FocalLoss(nn.Module):
        """Focal Loss for handling class imbalance.

        Focuses training on hard-to-classify examples.
        FL(pt) = -alpha * (1-pt)^gamma * log(pt)

        Args:
            alpha: Class weights [weight_class_0, weight_class_1]
            gamma: Focusing parameter (1.0 recommended for moderate focus)
        """

        def __init__(self, alpha=None, gamma=1.0):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma

        def forward(self, inputs, targets):
            """
            Args:
                inputs: [B, num_classes] logits
                targets: [B] class indices
            """
            ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss
            return focal_loss.mean()

    # TECHNIQUE 8: TEST-TIME AUGMENTATION (⭐⭐⭐⭐)
    def tta_predict(model, x, num_augmentations=5):
        """Test-Time Augmentation: average predictions over augmented versions.

        Args:
            model: Trained model
            x: Input signals [B, C, T]
            num_augmentations: Number of augmented versions to average

        Returns:
            avg_probs: [B, num_classes] averaged probabilities
        """
        model.eval()
        all_probs = []

        with torch.no_grad():
            # Original prediction
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs)

            # Augmented predictions
            for _ in range(num_augmentations - 1):
                # Random augmentation: add small noise
                noise = torch.randn_like(x) * 0.02
                x_aug = x + noise

                # Random time shift
                shift = np.random.randint(-10, 10)
                if shift > 0:
                    x_aug = torch.cat([x_aug[:, :, shift:], x_aug[:, :, :shift]], dim=2)
                elif shift < 0:
                    x_aug = torch.cat([x_aug[:, :, shift:], x_aug[:, :, :shift]], dim=2)

                logits_aug = model(x_aug)
                probs_aug = F.softmax(logits_aug, dim=1)
                all_probs.append(probs_aug)

        # Average all predictions
        avg_probs = torch.stack(all_probs).mean(dim=0)
        return avg_probs

    def evaluate_with_tta(model, loader, device, num_augmentations=5):
        """Evaluate with test-time augmentation.

        Returns:
            Dictionary with accuracy, AUROC, and per-class metrics
        """
        model.eval()

        all_labels = []
        all_probs = []

        pbar = tqdm(loader, desc="TTA Eval")
        for signals, labels in pbar:
            signals = signals.to(device)
            labels = labels.to(device)

            # Get TTA predictions
            probs = tta_predict(model, signals, num_augmentations)

            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Compute metrics
        preds = (all_probs > 0.5).astype(int)
        accuracy = 100.0 * (preds == all_labels).sum() / len(all_labels)

        try:
            auroc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auroc = 0.0

        # Per-class accuracy
        class_0_mask = all_labels == 0
        class_1_mask = all_labels == 1
        class_0_acc = (preds[class_0_mask] == all_labels[class_0_mask]).mean() if class_0_mask.sum() > 0 else 0.0
        class_1_acc = (preds[class_1_mask] == all_labels[class_1_mask]).mean() if class_1_mask.sum() > 0 else 0.0

        return {
            'accuracy': accuracy,
            'auroc': auroc,
            'class_0_acc': class_0_acc,
            'class_1_acc': class_1_acc
        }

    # ========================================================================

    # Create classification wrapper around SSL encoder
    print(f"\n📦 Creating classification wrapper...")

    class ClassificationWrapper(nn.Module):
        """Wrapper that adds classification head to SSL encoder."""

        def __init__(self, encoder, d_model: int, num_classes: int):
            super().__init__()
            self.encoder = encoder
            self.head = nn.Linear(d_model, num_classes)

            # Store config for compatibility
            self.context_length = getattr(encoder, 'context_length', None)
            self.patch_size = getattr(encoder, 'patch_size', None)
            self.num_classes = num_classes

        def forward(self, x):
            """Forward pass: encoder → pool → classify."""
            # Get encoder features: [B, C, T] → [B, P, D]
            features = self.encoder.get_encoder_output(x)
            # Pool across patches: [B, P, D] → [B, D]
            features_pooled = features.mean(dim=1)
            # Classify: [B, D] → [B, num_classes]
            logits = self.head(features_pooled)
            return logits

        def get_encoder_output(self, x):
            """Get encoder features (for analysis)."""
            return self.encoder.get_encoder_output(x)

    # ✅ ADVANCED: Use MultiScaleClassificationHead with Attention Pooling
    # This replaces simple ClassificationWrapper for better feature extraction
    model = MultiScaleClassificationHead(ssl_encoder, d_model=d_model, num_classes=2)
    model = model.to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  ✓ Multi-Scale Classification Head created (with Attention Pooling)")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} (head + fusion + pooling)")
    print(f"  Frozen parameters: {total_params - trainable_params:,} (encoder)")
    print(f"  Features: Early + Middle + Late encoder layers (3-scale fusion)")

    # Weight loading complete - encoder already has SSL weights from our load function!

    # Create dataloaders with target_length matching SSL model
    print(f"\n📊 Creating dataloaders (resizing windows to {context_length} samples if needed)...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=Path(args.data_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_length=context_length  # Resize BUT-PPG (1250) to match SSL model (512 for TTM-Enhanced)
    )
    
    # Use test set for validation if no validation set
    if val_loader is None and test_loader is not None:
        val_loader = test_loader
        print("\n⚠ Using test set for validation (no separate val set)")

    # Calculate class weights to handle imbalance
    print("\n⚙️  Computing class weights for imbalanced data...")
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())
    all_labels = np.array(all_labels)

    class_counts = np.bincount(all_labels)
    total_samples = len(all_labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights = torch.FloatTensor(class_weights).to(args.device)

    print(f"  Class 0 (Poor): {class_counts[0]} samples, weight: {class_weights[0]:.3f}")
    print(f"  Class 1 (Good): {class_counts[1]} samples, weight: {class_weights[1]:.3f}")
    print(f"  ✓ Using weighted loss to handle {class_counts[0]/class_counts[1]:.1f}:1 imbalance")

    # CRITICAL FIX: Use CORRECT inverse frequency class weights
    # Previous attempt used [1.0, 2.0] which UNDER-weighted the minority class!
    # Good class (21.4%) is MINORITY but was getting LESS emphasis than Poor (78.6%)
    # This caused model to ignore Good class → collapse to predicting only Poor
    #
    # New strategy: Inverse frequency weighting
    #   - Poor (majority 78.6%): weight = 1.0 (baseline)
    #   - Good (minority 21.4%): weight = ratio (upweight by ~3.67x)

    poor_count = class_counts[0]
    good_count = class_counts[1]
    imbalance_ratio = poor_count / good_count

    print(f"\n  Class imbalance: {imbalance_ratio:.2f}:1 (Poor:Good)")
    print(f"  ✅ ADVANCED: Using Focal Loss (gamma=1.0) for hard example mining")
    print(f"  Previous issues with gamma=4.0 caused collapse → now using moderate gamma=1.0")

    # Calculate proper inverse frequency weights
    # Poor (majority): normal weight (1.0)
    # Good (minority): upweight by ratio to balance contribution
    class_weights = torch.tensor([
        1.0,              # Poor (majority class) - baseline weight
        imbalance_ratio   # Good (minority class) - upweight by inverse frequency
    ], dtype=torch.float32).to(args.device)

    print(f"  Class weights: Poor={class_weights[0]:.3f}, Good={class_weights[1]:.3f}")
    print(f"  Focal gamma: 1.0 (moderate focusing on hard examples)")

    # ✅ ADVANCED: Focal Loss with gamma=1.0 (moderate, not aggressive 4.0)
    criterion = FocalLoss(alpha=class_weights, gamma=1.0)
    use_amp = not args.no_amp and torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auroc': [],  # CRITICAL: Track AUROC as primary metric
        'stage': []
    }

    # CRITICAL: Use AUROC as primary metric for saving best model
    # Accuracy is misleading with imbalanced data (80/20 Poor/Good)
    best_auroc = 0.0
    
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
    
    # ✅ ADVANCED: Use discriminative learning rates for head components
    # Different LR for classifier, fusion, and pooling layers
    print(f"  Stage 1 duration: {args.head_only_epochs} epochs")
    print(f"  ✅ Using discriminative learning rates for head components:")
    print(f"     - Classifier: {args.lr * 50:.2e} (50x base LR)")
    print(f"     - Fusion: {args.lr * 25:.2e} (25x base LR)")
    print(f"     - Pooling: {args.lr * 25:.2e} (25x base LR)")
    print(f"  Encoder learning rate: N/A (frozen)")

    # Get parameter groups with discriminative LRs
    param_groups = get_discriminative_params(model, base_lr=args.lr, head_multiplier=50)

    # Fallback if get_discriminative_params returns empty (head-only training)
    if len(param_groups) == 0:
        param_groups = [{'params': [p for p in model.parameters() if p.requires_grad], 'lr': args.lr * 50}]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # ✅ ADVANCED: Cosine Annealing with Warm Restarts (replaces ReduceLROnPlateau)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # Restart every 10 epochs
        T_mult=1,  # Keep same cycle length
        eta_min=args.lr * 0.1  # Min LR = 10% of base
    )
    print(f"  ✅ Using CosineAnnealingWarmRestarts (T_0=10, eta_min={args.lr * 0.1:.2e})")
    
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
        
        # Print epoch summary with AUROC and per-class accuracy
        print(f"\nEpoch {epoch+1}/{args.head_only_epochs} ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%, AUROC: {val_metrics['auroc']:.3f}")
        print(f"          Poor: {val_metrics['class_0_acc']*100:.1f}%, Good: {val_metrics['class_1_acc']*100:.1f}%")

        # Warn if class collapse detected
        if val_metrics['class_1_acc'] < 0.10:  # Class 1 (Good) < 10%
            print(f"  ⚠️  WARNING: Class 1 accuracy very low ({val_metrics['class_1_acc']*100:.1f}%) - model collapsing to majority class!")
        elif val_metrics['class_0_acc'] < 0.10:  # Class 0 (Poor) < 10%
            print(f"  ⚠️  WARNING: Class 0 accuracy very low ({val_metrics['class_0_acc']*100:.1f}%) - model collapsing to minority class!")

        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_auroc'].append(val_metrics['auroc'])
        history['stage'].append('stage1_head_only')

        # CRITICAL: Save best model based on AUROC, not accuracy
        if val_metrics['auroc'] > best_auroc:
            best_auroc = val_metrics['auroc']
            save_checkpoint(
                output_dir / 'best_model.pt',
                model, optimizer, epoch, val_metrics, training_config
            )
            print(f"  ✓ Best model saved (AUROC: {val_metrics['auroc']:.3f})")

        # Step learning rate scheduler
        scheduler.step()

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
        # Our model is a MultiScaleClassificationHead, so unfreeze on model.encoder
        unfreeze_last_n_blocks(model.encoder, n=args.unfreeze_last_n, verbose=True)

        # ✅ ADVANCED: Use discriminative learning rates with layer-wise decay
        print(f"  ✅ Using discriminative learning rates (layer-wise decay):")
        print(f"     - Classifier: {args.lr * 50:.2e} (50x base LR)")
        print(f"     - Fusion: {args.lr * 25:.2e} (25x base LR)")
        print(f"     - Pooling: {args.lr * 25:.2e} (25x base LR)")
        print(f"     - Encoder (late blocks): {args.lr * 5:.2e} (5x base LR)")
        print(f"     - Encoder (middle blocks): {args.lr * 2:.2e} (2x base LR)")
        print(f"     - Encoder (early blocks): {args.lr:.2e} (1x base LR)")

        # Get parameter groups with discriminative LRs
        param_groups = get_discriminative_params(model, base_lr=args.lr, head_multiplier=50)

        # Fallback to simple differential LR if needed
        if len(param_groups) == 0:
            param_groups = [
                {'params': model.classifier.parameters(), 'lr': args.lr * 50},
                {'params': model.fusion.parameters(), 'lr': args.lr * 25},
                {'params': model.pooling.parameters(), 'lr': args.lr * 25},
                {'params': model.encoder.parameters(), 'lr': args.lr}
            ]

        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

        # ✅ ADVANCED: Cosine Annealing with Warm Restarts (replaces ReduceLROnPlateau)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=1,  # Keep same cycle length
            eta_min=args.lr * 0.1  # Min LR = 10% of base
        )
        print(f"  ✓ Using CosineAnnealingWarmRestarts (T_0=10, eta_min={args.lr * 0.1:.2e})")

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
            
            # Print epoch summary with AUROC and per-class accuracy
            print(f"\nEpoch {args.head_only_epochs + epoch + 1}/{args.epochs} ({epoch_time:.1f}s)")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%, AUROC: {val_metrics['auroc']:.3f}")
            print(f"          Poor: {val_metrics['class_0_acc']*100:.1f}%, Good: {val_metrics['class_1_acc']*100:.1f}%")

            # Warn if class collapse detected
            if val_metrics['class_1_acc'] < 0.10:
                print(f"  ⚠️  WARNING: Class 1 accuracy very low ({val_metrics['class_1_acc']*100:.1f}%) - model collapsing to majority class!")
            elif val_metrics['class_0_acc'] < 0.10:
                print(f"  ⚠️  WARNING: Class 0 accuracy very low ({val_metrics['class_0_acc']*100:.1f}%) - model collapsing to minority class!")

            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_auroc'].append(val_metrics['auroc'])
            history['stage'].append('stage2_partial_unfreeze')

            # CRITICAL: Save best model based on AUROC, not accuracy
            if val_metrics['auroc'] > best_auroc:
                best_auroc = val_metrics['auroc']
                save_checkpoint(
                    output_dir / 'best_model.pt',
                    model, optimizer, epoch, val_metrics, training_config
                )
                print(f"  ✓ Best model saved (AUROC: {val_metrics['auroc']:.3f})")

            # Step learning rate scheduler
            scheduler.step()

    # =========================================================================
    # STAGE 3: FULL FINE-TUNING (OPTIONAL) with SWA
    # =========================================================================
    if args.full_finetune and args.full_finetune_epochs > 0:
        print("\n" + "=" * 70)
        print(f"STAGE 3: FULL FINE-TUNING ({args.full_finetune_epochs} epochs)")
        print("=" * 70)
        print("Unfreezing all parameters with SWA (Stochastic Weight Averaging)")

        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True

        # ✅ ADVANCED: Use discriminative learning rates even in Stage 3
        print(f"  ✅ Using discriminative learning rates with SWA:")
        print(f"     - Classifier: {args.lr * 10:.2e} (10x base LR)")
        print(f"     - Fusion: {args.lr * 5:.2e} (5x base LR)")
        print(f"     - Pooling: {args.lr * 5:.2e} (5x base LR)")
        print(f"     - Encoder (all blocks): {args.lr * 0.1:.2e} (0.1x base LR)")

        # Get parameter groups with lower multipliers for Stage 3
        param_groups = [
            {'params': model.classifier.parameters(), 'lr': args.lr * 10},
            {'params': model.fusion.parameters(), 'lr': args.lr * 5},
            {'params': model.pooling.parameters(), 'lr': args.lr * 5},
            {'params': model.encoder.parameters(), 'lr': args.lr * 0.1}
        ]

        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

        # ✅ ADVANCED: Setup SWA (Stochastic Weight Averaging)
        # Start averaging weights from epoch 25 (assuming 30 total epochs)
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=args.lr * 0.1)
        swa_start = max(0, args.full_finetune_epochs - 5)  # Last 5 epochs

        print(f"  ✅ SWA enabled: will average weights from epoch {swa_start + 1} to {args.full_finetune_epochs}")
        print(f"  SWA learning rate: {args.lr * 0.1:.2e}")
        
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
            
            # Print epoch summary with AUROC and per-class accuracy
            print(f"\nEpoch {args.epochs + epoch + 1}/{args.epochs + args.full_finetune_epochs} ({epoch_time:.1f}s)")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%, AUROC: {val_metrics['auroc']:.3f}")
            print(f"          Poor: {val_metrics['class_0_acc']*100:.1f}%, Good: {val_metrics['class_1_acc']*100:.1f}%")

            # Warn if class collapse detected
            if val_metrics['class_1_acc'] < 0.10:
                print(f"  ⚠️  WARNING: Class 1 accuracy very low ({val_metrics['class_1_acc']*100:.1f}%) - model collapsing to majority class!")
            elif val_metrics['class_0_acc'] < 0.10:
                print(f"  ⚠️  WARNING: Class 0 accuracy very low ({val_metrics['class_0_acc']*100:.1f}%) - model collapsing to minority class!")

            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_auroc'].append(val_metrics['auroc'])
            history['stage'].append('stage3_full_finetune')

            # ✅ ADVANCED: Update SWA model if in SWA phase
            if epoch >= swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()
                print(f"  ✓ SWA model updated (epoch {epoch + 1}/{args.full_finetune_epochs})")

            # CRITICAL: Save best model based on AUROC, not accuracy
            if val_metrics['auroc'] > best_auroc:
                best_auroc = val_metrics['auroc']
                save_checkpoint(
                    output_dir / 'best_model.pt',
                    model, optimizer, epoch, val_metrics, training_config
                )
                print(f"  ✓ Best model saved (AUROC: {val_metrics['auroc']:.3f})")

        # ✅ ADVANCED: Finalize SWA model (update batch norm statistics)
        print(f"\n  ✅ Finalizing SWA model...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=args.device)

        # Save SWA model separately
        swa_checkpoint = {
            'model_state_dict': swa_model.module.state_dict(),
            'training_config': training_config
        }
        torch.save(swa_checkpoint, output_dir / 'swa_model.pt')
        print(f"  ✓ SWA model saved to {output_dir / 'swa_model.pt'}")

        # Use SWA model for final evaluation
        model = swa_model.module

    # =========================================================================
    # FINAL EVALUATION with TTA
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL EVALUATION WITH TEST-TIME AUGMENTATION (TTA)")
    print("=" * 70)

    # Load best model (if not using SWA)
    if not (args.full_finetune and args.full_finetune_epochs > 0):
        best_checkpoint = torch.load(output_dir / 'best_model.pt', map_location=args.device, weights_only=False)
        model.load_state_dict(best_checkpoint['model_state_dict'])

    # Evaluate on test set if available
    if test_loader is not None:
        print("\n✅ Evaluating with Test-Time Augmentation (5 augmentations per sample)...")
        test_metrics_tta = evaluate_with_tta(
            model, test_loader, args.device, num_augmentations=5
        )

        print(f"\nTest Results (with TTA):")
        print(f"  Accuracy: {test_metrics_tta['accuracy']:.2f}%")
        print(f"  AUROC: {test_metrics_tta['auroc']:.3f}")
        print(f"  Class 0 (Poor) Accuracy: {test_metrics_tta['class_0_acc']*100:.2f}%")
        print(f"  Class 1 (Good) Accuracy: {test_metrics_tta['class_1_acc']*100:.2f}%")

        # Save test metrics
        with open(output_dir / 'test_metrics.json', 'w') as f:
            json.dump(test_metrics_tta, f, indent=2)
        test_metrics = test_metrics_tta
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
    print(f"Best validation AUROC: {best_auroc:.3f}")
    if test_metrics:
        print(f"\nFinal Test Results:")
        print(f"  Accuracy: {test_metrics['accuracy']:.2f}%")
        print(f"  AUROC: {test_metrics['auroc']:.3f}")
        print(f"  Poor (class 0): {test_metrics['class_0_acc']*100:.1f}%")
        print(f"  Good (class 1): {test_metrics['class_1_acc']*100:.1f}%")
    print(f"\nCheckpoints saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
