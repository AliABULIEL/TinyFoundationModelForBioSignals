#!/usr/bin/env python3
"""Fine-tuning Script for BUT-PPG Multi-Task Learning.

This script fine-tunes either an SSL pretrained model OR IBM TTM baseline on 2-channel
BUT-PPG data for multiple clinical tasks:

Tasks (7 total):
- quality: Signal quality classification (2 classes: Poor/Good)
- motion: Motion artifact classification (8 classes)
- hr_estimation: Heart rate regression (bpm)
- bp_systolic: Systolic blood pressure regression (mmHg)
- bp_diastolic: Diastolic blood pressure regression (mmHg)
- spo2: SpO2 regression (%)
- glycaemia: Blood glucose regression (mg/dL)

Data Format:
- BUT-PPG data: 2 channels [PPG, ECG] in windowed format
- Each window: individual NPZ file with shape [2, 1250] (10s @ 125Hz)
- Resampled to [2, 512] for IBM TTM-Enhanced (8 patches × 64)
- No channel inflation needed (both SSL and fine-tuning use 2 channels)

Strategy:
1. Load encoder: either SSL checkpoint OR IBM TTM baseline
2. Add task-specific head (classification or regression) with multi-scale features
3. Staged unfreezing:
   - Stage 1 (10 epochs): Head-only training with augmentations
   - Stage 2 (remaining epochs): Unfreeze last N encoder blocks
   - Stage 3 (optional): Full fine-tuning with SWA
4. Monitor validation metric (AUROC for classification, MAE for regression)

Usage:
    # Quality classification (default):
    python scripts/finetune_butppg.py \
        --pretrained artifacts/SSL_IMPROVED/best_model.pt \
        --task quality \
        --data-dir data/processed/butppg/windows_with_labels \
        --output-dir artifacts/FINETUNE_QUALITY \
        --epochs 30 \
        --device cuda

    # Heart rate regression:
    python scripts/finetune_butppg.py \
        --pretrained artifacts/SSL_IMPROVED/best_model.pt \
        --task hr_estimation \
        --data-dir data/processed/butppg/windows_with_labels \
        --output-dir artifacts/FINETUNE_HR \
        --epochs 30 \
        --device cuda

    # IBM baseline (no SSL):
    python scripts/finetune_butppg.py \
        --skip-ssl \
        --task quality \
        --data-dir data/processed/butppg/windows_with_labels \
        --output-dir artifacts/IBM_BASELINE_QUALITY \
        --epochs 40 \
        --device cuda
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


# ============================================================================
# TASK CONFIGURATIONS
# ============================================================================

TASK_CONFIGS = {
    # Classification tasks
    'quality': {
        'type': 'classification',
        'num_classes': 2,
        'label_key': 'quality',
        'metric': 'auroc',
        'description': 'Signal quality (Poor/Good)'
    },
    'motion': {
        'type': 'classification',
        'num_classes': 8,
        'label_key': 'motion',
        'metric': 'auroc',
        'description': 'Motion artifact (8 classes)'
    },

    # Regression tasks
    'hr_estimation': {
        'type': 'regression',
        'label_key': 'hr',
        'metric': 'mae',
        'description': 'Heart rate (bpm)'
    },
    'bp_systolic': {
        'type': 'regression',
        'label_key': 'bp_sys',
        'metric': 'mae',
        'description': 'Systolic BP (mmHg)'
    },
    'bp_diastolic': {
        'type': 'regression',
        'label_key': 'bp_dia',
        'metric': 'mae',
        'description': 'Diastolic BP (mmHg)'
    },
    'spo2': {
        'type': 'regression',
        'label_key': 'spo2',
        'metric': 'mae',
        'description': 'SpO2 (%)'
    },
    'glycaemia': {
        'type': 'regression',
        'label_key': 'glucose',
        'metric': 'mae',
        'description': 'Blood glucose (mg/dL)'
    }
}


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

    # Get encoder state dict - handle both SSL and fine-tuned checkpoints
    if 'encoder_state_dict' in checkpoint:
        # SSL checkpoint (from continue_ssl_butppg_quality.py)
        encoder_state = checkpoint['encoder_state_dict']
        print("  ✓ Found encoder_state_dict (SSL checkpoint)")
    elif 'model_state_dict' in checkpoint:
        # Fine-tuned checkpoint (from finetune_butppg.py)
        # Extract encoder weights from full model
        print("  ✓ Found model_state_dict (fine-tuned checkpoint)")
        print("  → Extracting encoder weights from full model...")

        full_model_state = checkpoint['model_state_dict']
        encoder_state = {}

        # Extract keys starting with 'encoder.'
        for key, value in full_model_state.items():
            if key.startswith('encoder.'):
                # Remove 'encoder.' prefix to match SSL checkpoint format
                new_key = key[len('encoder.'):]
                encoder_state[new_key] = value

        if len(encoder_state) == 0:
            raise ValueError(
                f"No encoder weights found in model_state_dict. "
                f"Keys: {list(full_model_state.keys())[:5]}..."
            )

        print(f"  ✓ Extracted {len(encoder_state)} encoder parameters")
    else:
        raise ValueError(
            f"Checkpoint missing encoder weights. "
            f"Expected 'encoder_state_dict' (SSL) or 'model_state_dict' (fine-tuned). "
            f"Found keys: {list(checkpoint.keys())}"
        )

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

    def __init__(
        self,
        data_dir: Path,
        task: str = 'quality',
        normalize: bool = True,
        target_length: int = None
    ):
        """Initialize BUT-PPG dataset from windowed format.

        Args:
            data_dir: Path to directory containing window_*.npz files
            task: Task name (quality, motion, hr_estimation, etc.)
            normalize: Whether to apply z-score normalization per channel
            target_length: If specified, resize signals to this length (for SSL model compatibility)
        """
        if task not in TASK_CONFIGS:
            raise ValueError(f"Unknown task: {task}. Available: {list(TASK_CONFIGS.keys())}")

        self.task = task
        self.task_config = TASK_CONFIGS[task]
        self.task_type = self.task_config['type']
        self.label_key = self.task_config['label_key']

        print(f"Loading BUT-PPG data for task: {task} ({self.task_config['description']})")
        print(f"  Task type: {self.task_type}")
        print(f"  Label key: {self.label_key}")

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
        skipped = 0

        for window_file in window_files:
            data = np.load(window_file)

            # Load signal [2, T] where T is typically 1250 for BUT-PPG (10s @ 125Hz)
            if 'signal' not in data:
                raise KeyError(f"Expected 'signal' key in {window_file}, found: {list(data.keys())}")

            signal = data['signal']  # [2, T] - will be resampled to target_length if specified

            # Load task-specific label
            if self.label_key not in data:
                # Skip samples without the required label
                skipped += 1
                continue

            label_value = data[self.label_key]

            # Skip NaN labels
            if np.isnan(label_value).any() if hasattr(label_value, '__iter__') else np.isnan(label_value):
                skipped += 1
                continue

            # Convert label based on task type
            if self.task_type == 'classification':
                label = int(label_value)
            else:  # regression
                label = float(label_value)

            signals_list.append(signal)
            labels_list.append(label)

        if len(signals_list) == 0:
            raise ValueError(
                f"No valid samples found for task '{task}' (label key: '{self.label_key}').\n"
                f"Checked {len(window_files)} files, all were missing the label or had NaN values."
            )

        # Stack into tensors
        self.signals = torch.from_numpy(np.stack(signals_list, axis=0)).float()  # [N, 2, T]

        if self.task_type == 'classification':
            self.labels = torch.tensor(labels_list, dtype=torch.long)  # [N]
        else:  # regression
            self.labels = torch.tensor(labels_list, dtype=torch.float32)  # [N]

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
        print(f"  Loaded {N} samples from {len(window_files)} windows (skipped {skipped} without {self.label_key}):")

        if self.task_type == 'classification':
            # Print class distribution
            for class_idx in range(self.task_config['num_classes']):
                n_class = (self.labels == class_idx).sum().item()
                print(f"    - Class {class_idx}: {n_class} ({n_class/N*100:.1f}%)")
        else:
            # Print regression statistics
            labels_np = self.labels.numpy()
            print(f"    - Mean: {labels_np.mean():.2f}")
            print(f"    - Std: {labels_np.std():.2f}")
            print(f"    - Min: {labels_np.min():.2f}")
            print(f"    - Max: {labels_np.max():.2f}")

        print(f"    - Shape: {self.signals.shape}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]


def create_dataloaders(
    data_dir: Path,
    task: str = 'quality',
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
        task: Task name (quality, motion, hr_estimation, etc.)
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
    train_dataset = BUTPPGDataset(train_dir, task=task, normalize=True, target_length=target_length)

    # CRITICAL FIX: Use normal shuffling (NOT balanced sampling)
    # Balanced sampling creates distribution mismatch:
    #   - Training batches: 50/50 Poor/Good (from balanced sampler)
    #   - Validation: 80/20 Poor/Good (real distribution)
    # Model trained on balanced distribution but tested on imbalanced → collapse!
    # Solution: Let model see real distribution during training

    print(f"  Using normal shuffling (real distribution matching validation)")

    # Get labels for statistics (only for classification tasks)
    task_type = TASK_CONFIGS[task]['type']
    if task_type == 'classification':
        train_labels = train_dataset.labels.numpy()
        class_sample_counts = np.bincount(train_labels)

        # Print distribution based on number of classes
        num_classes = TASK_CONFIGS[task]['num_classes']
        class_dist_str = ", ".join([
            f"Class {i}={class_sample_counts[i] if i < len(class_sample_counts) else 0} ({100*(class_sample_counts[i] if i < len(class_sample_counts) else 0)/len(train_labels):.1f}%)"
            for i in range(num_classes)
        ])
        print(f"  Training set: {class_dist_str}")
        print(f"  ✓ Normal shuffling (batches match real distribution)")

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
        val_dataset = BUTPPGDataset(val_dir, task=task, normalize=True, target_length=target_length)
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
        test_dataset = BUTPPGDataset(test_dir, task=task, normalize=True, target_length=target_length)
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


def quality_aware_augmentation(signals, labels, augmentation_prob=0.3):
    """Apply quality-aware augmentations to help model learn discriminative features.

    Strategy:
    - For Poor quality (label=0): Apply "improvements" (smoothing, denoising)
    - For Good quality (label=1): Apply "degradations" (noise, artifacts)

    This helps the model learn what features differentiate quality levels.

    Args:
        signals: [batch, channels, length] input signals
        labels: [batch] quality labels (0=Poor, 1=Good)
        augmentation_prob: probability of applying each augmentation type

    Returns:
        augmented signals [batch, channels, length]
    """
    aug_signals = signals.clone()
    batch_size = signals.size(0)

    for i in range(batch_size):
        label = labels[i].item()

        if label == 0:  # Poor quality - try to "improve" it
            # Augmentation 1: Smooth signal (reduce noise)
            if np.random.rand() < augmentation_prob:
                for c in range(aug_signals.size(1)):
                    signal_np = aug_signals[i, c].cpu().numpy()
                    smoothed = gaussian_filter1d(signal_np, sigma=2.0)
                    aug_signals[i, c] = torch.from_numpy(smoothed).to(signals.device)

            # Augmentation 2: Remove baseline wander
            if np.random.rand() < augmentation_prob:
                for c in range(aug_signals.size(1)):
                    signal_np = aug_signals[i, c].cpu().numpy()
                    baseline = gaussian_filter1d(signal_np, sigma=50.0)
                    corrected = signal_np - baseline
                    aug_signals[i, c] = torch.from_numpy(corrected).to(signals.device)

        else:  # Good quality (label == 1) - try to "degrade" it
            # Augmentation 1: Add Gaussian noise
            if np.random.rand() < augmentation_prob:
                noise_level = np.random.uniform(0.05, 0.15)
                noise = torch.randn_like(aug_signals[i]) * noise_level
                aug_signals[i] = aug_signals[i] + noise

            # Augmentation 2: Add baseline wander
            if np.random.rand() < augmentation_prob:
                length = aug_signals.size(2)
                t = torch.linspace(0, 2 * np.pi, length).to(signals.device)
                frequency = np.random.uniform(0.3, 0.7)
                amplitude = np.random.uniform(0.1, 0.3)
                baseline_wander = amplitude * torch.sin(frequency * t)
                for c in range(aug_signals.size(1)):
                    aug_signals[i, c] = aug_signals[i, c] + baseline_wander

            # Augmentation 3: Add motion artifact (sudden spike)
            if np.random.rand() < augmentation_prob * 0.5:  # Less frequent
                artifact_pos = np.random.randint(50, aug_signals.size(2) - 50)
                artifact_length = np.random.randint(5, 20)
                artifact_amplitude = np.random.uniform(0.5, 1.5)
                for c in range(aug_signals.size(1)):
                    aug_signals[i, c, artifact_pos:artifact_pos+artifact_length] += artifact_amplitude

    return aug_signals


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    task_type: str = 'classification',
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
        task_type: 'classification' or 'regression'
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
        
        # Compute metrics based on task type
        if task_type == 'classification':
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        else:  # regression
            # For regression, track MAE instead of accuracy
            mae = torch.abs(logits - labels).mean().item()
            if batch_idx == 0:
                total_mae = mae
            else:
                total_mae = (total_mae * batch_idx + mae) / (batch_idx + 1)

        total_loss += loss.item()

        # Update progress bar
        if task_type == 'classification':
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100.0 * correct / total if total > 0 else 0.0
            })
        else:  # regression
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'mae': total_mae
            })

    if task_type == 'classification':
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': 100.0 * correct / total if total > 0 else 0.0
        }
    else:  # regression
        return {
            'loss': total_loss / len(train_loader),
            'mae': total_mae
        }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    task_type: str = 'classification',
    num_classes: int = 2,
    use_amp: bool = True,
    desc: str = "Val"
) -> Dict[str, float]:
    """Evaluate model on a dataset.

    Args:
        model: Model to evaluate
        loader: Dataloader
        criterion: Loss criterion
        device: Device to evaluate on
        task_type: 'classification' or 'regression'
        num_classes: Number of classes (for classification only)
        use_amp: Use automatic mixed precision
        desc: Description for progress bar

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    # Classification-specific
    if task_type == 'classification':
        correct = 0
        total = 0
        all_probs = []  # Collect probabilities for AUROC

    pbar = tqdm(loader, desc=f"[{desc}]", leave=False)

    with torch.no_grad():
        for signals, labels in pbar:
            signals = signals.to(device)
            labels = labels.to(device)

            with autocast(enabled=use_amp):
                outputs = model(signals)
                loss = criterion(outputs, labels)

            total_loss += loss.item()

            if task_type == 'classification':
                # Get predictions and probabilities
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # For binary classification, get probability of positive class
                if num_classes == 2:
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    all_probs.extend(probs.cpu().numpy())

                pbar.set_postfix({
                    'loss': total_loss / (len(all_preds) / labels.size(0)),
                    'acc': 100.0 * correct / total
                })
            else:  # regression
                # For regression, outputs are predictions directly
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Compute running MAE
                mae = np.abs(np.array(all_preds) - np.array(all_labels)).mean()
                pbar.set_postfix({
                    'loss': total_loss / (len(all_preds) / labels.size(0)),
                    'mae': mae
                })

    # Convert to numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute final metrics based on task type
    if task_type == 'classification':
        # Compute AUROC for binary classification
        if num_classes == 2 and len(all_probs) > 0:
            all_probs = np.array(all_probs)
            try:
                auroc = roc_auc_score(all_labels, all_probs)
            except ValueError:
                auroc = 0.0
        else:
            auroc = 0.0

        # Per-class accuracy (return as fraction 0-1, not percentage)
        class_accuracies = {}
        for class_idx in range(num_classes):
            class_mask = all_labels == class_idx
            if class_mask.sum() > 0:
                class_acc = (all_preds[class_mask] == all_labels[class_mask]).mean()
                class_accuracies[f'class_{class_idx}_acc'] = class_acc
            else:
                class_accuracies[f'class_{class_idx}_acc'] = 0.0

        return {
            'loss': total_loss / len(loader),
            'accuracy': 100.0 * correct / total if total > 0 else 0.0,
            'auroc': auroc,
            **class_accuracies
        }
    else:  # regression
        # Compute regression metrics
        mae = np.abs(all_preds - all_labels).mean()
        rmse = np.sqrt(((all_preds - all_labels) ** 2).mean())

        return {
            'loss': total_loss / len(loader),
            'mae': mae,
            'rmse': rmse
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

    # Task selection
    parser.add_argument(
        '--task',
        type=str,
        default='quality',
        choices=list(TASK_CONFIGS.keys()),
        help='Task to fine-tune on (default: quality)'
    )

    # Debugging
    parser.add_argument(
        '--verify-data',
        action='store_true',
        help='Verify data structure and exit (useful for debugging)'
    )

    # SSL and baseline options
    parser.add_argument(
        '--skip-ssl',
        action='store_true',
        help='Skip SSL checkpoint loading and use IBM TTM baseline directly (for testing IBM baseline performance)'
    )
    parser.add_argument(
        '--use-ibm-pretrained',
        action='store_true',
        default=True,
        help='Use IBM pretrained TTM weights when initializing baseline (default: True)'
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

    # Get task configuration
    task_config = TASK_CONFIGS[args.task]
    task_type = task_config['type']
    task_description = task_config['description']

    # Print configuration
    print("\n" + "=" * 70)
    print("BUT-PPG MULTI-TASK FINE-TUNING CONFIGURATION")
    print("=" * 70)

    print(f"Task: {args.task} ({task_description})")
    print(f"  Type: {task_type}")
    if task_type == 'classification':
        print(f"  Classes: {task_config['num_classes']}")
        print(f"  Metric: {task_config['metric'].upper()}")
    else:
        print(f"  Metric: {task_config['metric'].upper()} (lower is better)")

    if args.skip_ssl:
        print("\nMode: IBM TTM BASELINE (No SSL)")
        print("Pretrained: IBM Granite TTM-r1" if args.use_ibm_pretrained else "Random initialization")
    else:
        print(f"\nMode: SSL FINE-TUNING")
        print(f"Pretrained model: {args.pretrained}")

    print(f"\nData directory: {args.data_dir}")
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

    # Load encoder: either SSL checkpoint or IBM TTM baseline
    print("\n" + "=" * 70)
    if args.skip_ssl:
        print("INITIALIZING IBM TTM BASELINE (NO SSL)")
    else:
        print("LOADING SSL ENCODER")
    print("=" * 70)

    if args.skip_ssl:
        # Initialize encoder directly from IBM TTM pretrained weights
        from src.models.ttm_encoder import create_ttm_encoder

        print("\n✓ Skipping SSL checkpoint loading")
        print("✓ Initializing encoder directly from IBM TTM pretrained weights\n")

        try:
            ssl_encoder = create_ttm_encoder(
                context_length=512,
                patch_size=64,
                d_model=192,
                num_input_channels=2,
                num_layers=3,
                use_positional_encoding=True,
                use_real_ttm=True,
                pretrained_variant='ibm-granite/granite-timeseries-ttm-r1' if args.use_ibm_pretrained else None
            )

            # Set config values
            context_length = 512
            patch_size = 64
            d_model = 192
            num_patches = context_length // patch_size

            print(f"✅ IBM TTM encoder initialized:")
            print(f"  Context length: {context_length}")
            print(f"  Patch size: {patch_size}")
            print(f"  d_model: {d_model}")
            print(f"  Num patches: {num_patches}")
            print(f"  Total encoder params: {sum(p.numel() for p in ssl_encoder.parameters()):,}")
            print(f"  Using pretrained: {args.use_ibm_pretrained}")

        except Exception as e:
            print(f"\n❌ FATAL: Failed to initialize IBM TTM encoder: {e}")
            import traceback
            traceback.print_exc()
            print("\nCannot proceed without encoder - exiting")
            import sys
            sys.exit(1)

    else:
        # Load SSL pre-trained encoder
        print("Using Option 3: Load complete SSL encoder with IBM TTM inside")
        print("This avoids fallback CNN and preserves all SSL features!")

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

    # TECHNIQUE 2.5: MULTI-SCALE REGRESSION HEAD (⭐⭐⭐⭐⭐)
    class MultiScaleRegressionHead(nn.Module):
        """Regression head with multi-scale feature fusion.

        Similar to MultiScaleClassificationHead but for continuous targets.
        Extracts features from multiple encoder depths and fuses them.
        """

        def __init__(self, encoder, d_model: int):
            super().__init__()
            self.encoder = encoder
            self.d_model = d_model

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

            # Regression head (single output)
            self.regressor = nn.Linear(d_model, 1)

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
                    # Get encoder blocks (use mixers for TTM)
                    if hasattr(self.encoder.backbone.encoder, 'mixers'):
                        # TTM architecture uses mixers
                        blocks = self.encoder.backbone.encoder.mixers
                    elif hasattr(self.encoder.backbone.encoder, 'layers'):
                        # Transformer architecture uses layers
                        blocks = self.encoder.backbone.encoder.layers
                    else:
                        # Fallback
                        raise AttributeError("No blocks found")

                    num_blocks = len(blocks)

                    # Get patch embeddings
                    x_patches = self.encoder.backbone.patcher(x)  # [B, P, D]

                    # Forward through early blocks (1/3)
                    early_idx = max(1, num_blocks // 3)
                    x_early = x_patches
                    for i in range(early_idx):
                        x_early = blocks[i](x_early)
                    early_features = x_early  # [B, P, D]

                    # Forward through middle blocks (2/3)
                    middle_idx = max(early_idx + 1, 2 * num_blocks // 3)
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

            # Regress (returns [B, 1], squeeze to [B] for compatibility)
            output = self.regressor(fused).squeeze(-1)  # [B]

            return output

        def get_encoder_output(self, x):
            """Get encoder features (for analysis)."""
            return self.encoder.get_encoder_output(x)

    # NOTE: All augmentation functions (mixup_data, cutmix_data, mixup_criterion,
    # quality_aware_augmentation) are now defined at module level (before train_epoch)

    # TECHNIQUE 5: DISCRIMINATIVE LEARNING RATES (⭐⭐⭐⭐⭐)
    def get_discriminative_params(model, base_lr, head_multiplier=50):
        """Create parameter groups with discriminative learning rates.

        Strategy:
        - Head components (classifier, fusion, pooling): 50x base LR
        - Last encoder block (mixers.2): 5x base LR
        - Middle encoder blocks (mixers.1): 2x base LR
        - Early encoder blocks (mixers.0): 1x base LR

        Args:
            model: The fine-tuning model
            base_lr: Base learning rate
            head_multiplier: Multiplier for head learning rate

        Returns:
            List of parameter groups for optimizer
        """
        param_groups = []

        print("\n" + "="*70)
        print("DISCRIMINATIVE LEARNING RATES")
        print("="*70)

        # ==================================================================
        # GROUP 1: HEAD COMPONENTS (highest LR)
        # ==================================================================
        head_params = []
        head_keywords = ['classifier', 'fusion', 'pooling', 'attention']

        for name, param in model.named_parameters():
            if param.requires_grad and any(keyword in name for keyword in head_keywords):
                head_params.append(param)

        if head_params:
            param_groups.append({
                'params': head_params,
                'lr': base_lr * head_multiplier,
                'name': 'head_components'
            })
            print(f"  Head components: {len(head_params)} params, LR = {base_lr * head_multiplier:.2e}")

        # ==================================================================
        # GROUP 2: ENCODER BLOCKS (layer-wise learning rates)
        # ==================================================================
        encoder_params = {
            'late': [],      # mixers.2 - highest encoder LR
            'middle': [],    # mixers.1 - medium encoder LR
            'early': [],     # mixers.0 - lowest encoder LR
            'other': []      # patcher, etc. - medium LR
        }

        for name, param in model.named_parameters():
            # Only process encoder parameters that are trainable
            if not param.requires_grad:
                continue

            # Skip if already in head group
            if any(keyword in name for keyword in head_keywords):
                continue

            # Must be an encoder parameter
            if 'encoder' not in name:
                continue

            # Categorize by mixer block index
            if 'mixers.2' in name:
                encoder_params['late'].append(param)
            elif 'mixers.1' in name:
                encoder_params['middle'].append(param)
            elif 'mixers.0' in name:
                encoder_params['early'].append(param)
            else:
                # Other encoder params (patcher, normalization, etc.)
                encoder_params['other'].append(param)

        # Add encoder parameter groups with different learning rates
        if encoder_params['late']:
            param_groups.append({
                'params': encoder_params['late'],
                'lr': base_lr * 5,
                'name': 'encoder_late'
            })
            print(f"  Encoder (late):   {len(encoder_params['late'])} params, LR = {base_lr * 5:.2e}")

        if encoder_params['middle']:
            param_groups.append({
                'params': encoder_params['middle'],
                'lr': base_lr * 2,
                'name': 'encoder_middle'
            })
            print(f"  Encoder (middle): {len(encoder_params['middle'])} params, LR = {base_lr * 2:.2e}")

        if encoder_params['early']:
            param_groups.append({
                'params': encoder_params['early'],
                'lr': base_lr * 1,
                'name': 'encoder_early'
            })
            print(f"  Encoder (early):  {len(encoder_params['early'])} params, LR = {base_lr * 1:.2e}")

        if encoder_params['other']:
            param_groups.append({
                'params': encoder_params['other'],
                'lr': base_lr * 2,
                'name': 'encoder_other'
            })
            print(f"  Encoder (other):  {len(encoder_params['other'])} params, LR = {base_lr * 2:.2e}")

        print("="*70 + "\n")

        # Sanity check: make sure we have at least one parameter group
        if not param_groups:
            raise ValueError("No trainable parameters found! Check model.requires_grad settings.")

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

    # ✅ ADVANCED: Use Multi-Scale Head with Attention Pooling
    # Choose classification or regression head based on task type
    if task_type == 'classification':
        num_classes = task_config['num_classes']
        model = MultiScaleClassificationHead(ssl_encoder, d_model=d_model, num_classes=num_classes)
        print(f"  ✓ Multi-Scale Classification Head created (with Attention Pooling)")
        print(f"  Number of classes: {num_classes}")
    else:  # regression
        model = MultiScaleRegressionHead(ssl_encoder, d_model=d_model)
        print(f"  ✓ Multi-Scale Regression Head created (with Attention Pooling)")
        print(f"  Output: Continuous value ({task_description})")

    model = model.to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} (head + fusion + pooling)")
    print(f"  Frozen parameters: {total_params - trainable_params:,} (encoder)")
    print(f"  Features: Early + Middle + Late encoder layers (3-scale fusion)")

    # Weight loading complete - encoder already has SSL weights from our load function!

    # Create dataloaders with target_length matching SSL model
    print(f"\n📊 Creating dataloaders (resizing windows to {context_length} samples if needed)...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=Path(args.data_dir),
        task=args.task,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_length=context_length  # Resize BUT-PPG (1250) to match SSL model (512 for TTM-Enhanced)
    )
    
    # Use test set for validation if no validation set
    if val_loader is None and test_loader is not None:
        val_loader = test_loader
        print("\n⚠ Using test set for validation (no separate val set)")

    # Setup loss function based on task type
    print("\n⚙️  Setting up loss function...")

    if task_type == 'classification':
        # Calculate class weights to handle imbalance
        print("  Computing class weights for imbalanced data...")
        all_labels = []
        for _, labels in train_loader:
            all_labels.extend(labels.numpy())
        all_labels = np.array(all_labels)

        class_counts = np.bincount(all_labels)
        num_classes = task_config['num_classes']

        # CRITICAL FIX: Use CORRECT inverse frequency class weights
        # For binary: [1.0, ratio] where ratio = majority_count / minority_count
        # For multi-class: inverse frequency for each class
        total_samples = len(all_labels)
        class_weights = total_samples / (num_classes * class_counts)
        class_weights = torch.FloatTensor(class_weights).to(args.device)

        # Print class distribution
        for i in range(num_classes):
            print(f"  Class {i}: {class_counts[i]} samples, weight: {class_weights[i]:.3f}")

        if num_classes == 2:
            imbalance_ratio = class_counts[0] / class_counts[1]
            print(f"  ✓ Using weighted loss to handle {imbalance_ratio:.1f}:1 imbalance")
            print(f"  ✅ ADVANCED: Using Focal Loss (gamma=1.0) for hard example mining")

        # Use Focal Loss for classification (moderate gamma to avoid collapse)
        criterion = FocalLoss(alpha=class_weights, gamma=1.0)
        print(f"  Loss: Focal Loss (gamma=1.0, weighted)")

    else:  # regression
        # For regression, use MAE loss (more robust than MSE for outliers)
        criterion = nn.L1Loss()  # MAE
        print(f"  Loss: MAE (L1 Loss) - robust to outliers")
        print(f"  Metric: MAE (Mean Absolute Error) - lower is better")
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
            args.device, task_type, use_amp, scaler, args.gradient_clip, epoch
        )

        # Validate
        if val_loader is not None:
            val_metrics = evaluate(
                model, val_loader, criterion, args.device, task_type,
                task_config.get('num_classes', 2), use_amp, "Val"
            )
        else:
            val_metrics = {'loss': 0.0, 'accuracy': 0.0}
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary (different for classification vs regression)
        print(f"\nEpoch {epoch+1}/{args.head_only_epochs} ({epoch_time:.1f}s)")

        if task_type == 'classification':
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%, AUROC: {val_metrics.get('auroc', 0.0):.3f}")

            # Print per-class accuracy for classification
            for i in range(task_config.get('num_classes', 2)):
                class_acc_key = f'class_{i}_acc'
                if class_acc_key in val_metrics:
                    print(f"          Class {i}: {val_metrics[class_acc_key]*100:.1f}%", end='')
                    if i < task_config.get('num_classes', 2) - 1:
                        print(", ", end='')
            print()  # New line

            # Warn if class collapse detected
            for i in range(task_config.get('num_classes', 2)):
                class_acc_key = f'class_{i}_acc'
                if class_acc_key in val_metrics and val_metrics[class_acc_key] < 0.10:
                    print(f"  ⚠️  WARNING: Class {i} accuracy very low ({val_metrics[class_acc_key]*100:.1f}%) - model collapsing!")
        else:  # regression
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics.get('mae', 0.0):.2f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics.get('mae', 0.0):.2f}, RMSE: {val_metrics.get('rmse', 0.0):.2f}")

        # Update history
        history['train_loss'].append(train_metrics['loss'])
        if task_type == 'classification':
            history['train_acc'].append(train_metrics.get('accuracy', 0.0))
            history['val_acc'].append(val_metrics.get('accuracy', 0.0))
            history['val_auroc'].append(val_metrics.get('auroc', 0.0))
        else:
            history['train_acc'].append(train_metrics.get('mae', 0.0))
            history['val_acc'].append(val_metrics.get('mae', 0.0))
            history['val_auroc'].append(val_metrics.get('rmse', 0.0))
        history['val_loss'].append(val_metrics['loss'])
        history['stage'].append('stage1_head_only')

        # Save best model based on primary metric
        primary_metric = task_config['metric']
        if task_type == 'classification':
            # For classification, higher AUROC is better
            if val_metrics.get('auroc', 0.0) > best_auroc:
                best_auroc = val_metrics['auroc']
                save_checkpoint(
                    output_dir / 'best_model.pt',
                    model, optimizer, epoch, val_metrics, training_config
                )
                print(f"  ✓ Best model saved (AUROC: {val_metrics['auroc']:.3f})")
        else:  # regression
            # For regression, lower MAE is better
            current_mae = val_metrics.get('mae', float('inf'))
            if best_auroc == 0.0:  # First epoch
                best_auroc = current_mae
                save_checkpoint(
                    output_dir / 'best_model.pt',
                    model, optimizer, epoch, val_metrics, training_config
                )
                print(f"  ✓ Best model saved (MAE: {current_mae:.3f})")
            elif current_mae < best_auroc:
                best_auroc = current_mae
                save_checkpoint(
                    output_dir / 'best_model.pt',
                    model, optimizer, epoch, val_metrics, training_config
                )
                print(f"  ✓ Best model saved (MAE: {current_mae:.3f})")

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
                args.device, task_type, use_amp, scaler, args.gradient_clip,
                args.head_only_epochs + epoch
            )

            # Validate
            if val_loader is not None:
                val_metrics = evaluate(
                    model, val_loader, criterion, args.device, task_type,
                    task_config.get('num_classes', 2), use_amp, "Val"
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
                args.device, task_type, use_amp, scaler, args.gradient_clip,
                args.epochs + epoch
            )

            # Validate
            if val_loader is not None:
                val_metrics = evaluate(
                    model, val_loader, criterion, args.device, task_type,
                    task_config.get('num_classes', 2), use_amp, "Val"
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
        if task_type == 'classification':
            print("\n✅ Evaluating with Test-Time Augmentation (5 augmentations per sample)...")
            test_metrics_tta = evaluate_with_tta(
                model, test_loader, args.device, num_augmentations=5
            )

            print(f"\nTest Results (with TTA):")
            print(f"  Accuracy: {test_metrics_tta['accuracy']:.2f}%")
            print(f"  AUROC: {test_metrics_tta['auroc']:.3f}")
            for i in range(task_config.get('num_classes', 2)):
                class_acc_key = f'class_{i}_acc'
                if class_acc_key in test_metrics_tta:
                    print(f"  Class {i} Accuracy: {test_metrics_tta[class_acc_key]*100:.2f}%")
        else:  # regression
            print("\n✅ Evaluating on test set...")
            test_metrics_tta = evaluate(
                model, test_loader, criterion, args.device, task_type,
                task_config.get('num_classes', 2), use_amp, "Test"
            )

            print(f"\nTest Results:")
            print(f"  MAE: {test_metrics_tta['mae']:.3f}")
            print(f"  RMSE: {test_metrics_tta['rmse']:.3f}")

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

    if task_type == 'classification':
        print(f"Best validation AUROC: {best_auroc:.3f}")
        if test_metrics:
            print(f"\nFinal Test Results:")
            print(f"  Accuracy: {test_metrics.get('accuracy', 0.0):.2f}%")
            print(f"  AUROC: {test_metrics.get('auroc', 0.0):.3f}")
            for i in range(task_config.get('num_classes', 2)):
                class_acc_key = f'class_{i}_acc'
                if class_acc_key in test_metrics:
                    print(f"  Class {i}: {test_metrics[class_acc_key]*100:.1f}%")
    else:  # regression
        print(f"Best validation MAE: {best_auroc:.3f}")
        if test_metrics:
            print(f"\nFinal Test Results:")
            print(f"  MAE: {test_metrics.get('mae', 0.0):.3f}")
            print(f"  RMSE: {test_metrics.get('rmse', 0.0):.3f}")

    print(f"\nCheckpoints saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
