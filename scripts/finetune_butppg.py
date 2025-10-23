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
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
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
    
    # Per-class accuracy (return as fraction 0-1, not percentage)
    class_0_mask = all_labels == 0
    class_1_mask = all_labels == 1

    class_0_acc = (all_preds[class_0_mask] == all_labels[class_0_mask]).mean() if class_0_mask.sum() > 0 else 0.0
    class_1_acc = (all_preds[class_1_mask] == all_labels[class_1_mask]).mean() if class_1_mask.sum() > 0 else 0.0
    
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

    model = ClassificationWrapper(ssl_encoder, d_model=d_model, num_classes=2)
    model = model.to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  ✓ Classification wrapper created")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} (head only)")
    print(f"  Frozen parameters: {total_params - trainable_params:,} (encoder)")

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

    # Focal Loss for severe class imbalance (addresses collapse to majority class)
    class FocalLoss(nn.Module):
        """Focal Loss - focuses on hard examples and minority class.

        Args:
            alpha: Weight for minority class (0-1). Higher = more focus on minority.
            gamma: Focusing parameter (0-5). Higher = more focus on hard examples.
                   gamma=0 → standard cross-entropy
                   gamma=2 → recommended for most cases
                   gamma=5 → aggressive focusing

        References:
            Lin et al. "Focal Loss for Dense Object Detection" (2017)
        """
        def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, inputs, targets):
            # Standard cross-entropy
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')

            # Probability of correct class
            pt = torch.exp(-ce_loss)

            # Focal weight: (1 - pt)^gamma
            # When pt is high (easy example) → weight ≈ 0 (ignore)
            # When pt is low (hard example) → weight ≈ 1 (focus)
            focal_weight = (1 - pt) ** self.gamma

            # Alpha weighting for class balance
            # Apply higher weight to minority class
            alpha_weight = self.alpha * (targets == 1).float() + (1 - self.alpha) * (targets == 0).float()

            # Combine
            focal_loss = alpha_weight * focal_weight * ce_loss

            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                return focal_loss

    # Choose loss function based on severity of imbalance
    imbalance_ratio = class_counts[0] / class_counts[1]

    if imbalance_ratio > 3.0:
        # Severe imbalance (>3:1) - use Focal Loss
        print(f"\n⚠️  Severe class imbalance detected ({imbalance_ratio:.1f}:1)")
        print(f"  Using AGGRESSIVE Focal Loss (alpha=0.85, gamma=4.0) to prevent collapse")
        criterion = FocalLoss(alpha=0.85, gamma=4.0)  # Much more aggressive!
    else:
        # Moderate imbalance - standard weighted cross-entropy
        print(f"\n  Using weighted cross-entropy for {imbalance_ratio:.1f}:1 imbalance")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
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

        # Warn if class collapse detected
        if val_metrics['class_1_acc'] < 0.10:  # Class 1 (Good) < 10%
            print(f"  ⚠️  WARNING: Class 1 accuracy very low ({val_metrics['class_1_acc']*100:.1f}%) - model collapsing to majority class!")
            print(f"      Class 0 (Poor): {val_metrics['class_0_acc']*100:.1f}%, Class 1 (Good): {val_metrics['class_1_acc']*100:.1f}%")
        
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
        # Our model is a ClassificationWrapper, so unfreeze on model.encoder
        unfreeze_last_n_blocks(model.encoder, n=args.unfreeze_last_n, verbose=True)
        
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

            # Warn if class collapse detected
            if val_metrics['class_1_acc'] < 0.10:
                print(f"  ⚠️  WARNING: Class 1 accuracy very low ({val_metrics['class_1_acc']*100:.1f}%) - model collapsing to majority class!")
                print(f"      Class 0 (Poor): {val_metrics['class_0_acc']*100:.1f}%, Class 1 (Good): {val_metrics['class_1_acc']*100:.1f}%")
            
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

            # Warn if class collapse detected
            if val_metrics['class_1_acc'] < 0.10:
                print(f"  ⚠️  WARNING: Class 1 accuracy very low ({val_metrics['class_1_acc']*100:.1f}%) - model collapsing to majority class!")
                print(f"      Class 0 (Poor): {val_metrics['class_0_acc']*100:.1f}%, Class 1 (Good): {val_metrics['class_1_acc']*100:.1f}%")
            
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
        print(f"  Class 0 (Poor) Accuracy: {test_metrics['class_0_acc']*100:.2f}%")
        print(f"  Class 1 (Good) Accuracy: {test_metrics['class_1_acc']*100:.2f}%")
        
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
