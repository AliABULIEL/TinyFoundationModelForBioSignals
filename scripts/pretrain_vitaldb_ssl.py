#!/usr/bin/env python3
"""
SSL Pre-training Script for VitalDB Foundation Model
====================================================

This script performs Self-Supervised Learning (SSL) pre-training using:
- Masked Signal Modeling (MSM) with 40% masking
- Multi-Resolution STFT loss for spectral preservation
- VitalDB biosignals (PPG + ECG, 2 channels)
- TTM (Tiny Time Mixers) encoder architecture
- Lightweight reconstruction decoder

Training Strategy (from research):
- Input: VitalDB windows [N, 2, 1250] (2 channels, 10s @ 125Hz)
- Masking: 40% of 1-second patches (10 patches total)
- Loss: MSM + 0.3 × Multi-Resolution STFT
- Target: ~500K windows, 100-300 epochs
- Output: Foundation model checkpoint for downstream fine-tuning

Usage:
    # Basic usage (uses configs/ssl_pretrain.yaml)
    python scripts/pretrain_vitaldb_ssl.py
    
    # Specify custom config
    python scripts/pretrain_vitaldb_ssl.py --config configs/ssl_pretrain.yaml
    
    # FastTrack mode (fewer epochs, smaller data)
    python scripts/pretrain_vitaldb_ssl.py --mode fasttrack --epochs 10
    
    # Resume from checkpoint
    python scripts/pretrain_vitaldb_ssl.py --resume artifacts/foundation_model/checkpoint_epoch_50.pt
    
    # Colab-friendly (force CPU if needed)
    python scripts/pretrain_vitaldb_ssl.py --device cpu

Author: Senior ML & SW Engineer  
Date: October 2025
Aligned with: prepare_all_data.py data pipeline
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import SSL components
from src.ssl.masking import random_masking, block_masking
from src.ssl.objectives import MaskedSignalModeling, MultiResolutionSTFT
from src.ssl.pretrainer import SSLTrainer

# Import model components
from src.models.ttm_adapter import TTMAdapter, create_ttm_model
from src.models.decoders import ReconstructionHead1D

# Import utilities
from src.utils.seed import set_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ssl_pretraining.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VitalDBSSLDataset(Dataset):
    """
    Dataset for SSL pre-training on VitalDB.
    
    Loads preprocessed windows from prepare_all_data.py output.
    Handles both:
    1. Single file format: split/split_windows.npz
    2. Separated modality format: split/ppg/*.npz + split/ecg/*.npz
    
    Args:
        data_file_or_dir: Path to .npz file OR directory with modality folders
        transform_to_channels_first: Convert [N,T,C] to [N,C,T] if needed
    """
    
    def __init__(
        self,
        data_file_or_dir: str,
        transform_to_channels_first: bool = True
    ):
        """Load preprocessed VitalDB windows."""
        path = Path(data_file_or_dir)

        # Check if it's a directory
        if path.is_dir():
            logger.info(f"Detected directory format: {path}")

            # Check for NEW paired format (case_*.npz files)
            case_files = list(path.glob('case_*.npz'))
            if case_files:
                logger.info(f"  Found {len(case_files)} paired case files (NEW format)")
                windows = self._load_paired_cases(path)
            else:
                # OLD format: separated modalities (ppg/, ecg/)
                logger.info("  Looking for separated modality folders (ppg/, ecg/)...")
                windows = self._load_separated_modalities(path)
        elif path.exists() and path.suffix == '.npz':
            logger.info(f"Loading single file: {path}")
            windows = self._load_single_file(path)
        else:
            raise FileNotFoundError(f"Data not found: {path}")
        
        logger.info(f"  Loaded shape: {windows.shape}")
        
        # Handle shape: need [N, C, T] for SSL
        if windows.ndim == 2:
            # [N, T] → [N, 1, T]
            windows = windows[:, np.newaxis, :]
            logger.info(f"  Expanded to: {windows.shape}")
        elif windows.ndim == 3:
            # Check if [N, T, C] or [N, C, T]
            if transform_to_channels_first and windows.shape[2] < windows.shape[1]:
                # Likely [N, T, C] → transpose to [N, C, T]
                windows = np.transpose(windows, (0, 2, 1))
                logger.info(f"  Transposed to: {windows.shape} [channels-first]")
        
        # Verify shape
        if windows.ndim != 3:
            raise ValueError(
                f"Expected 3D array [N, C, T] after shape handling, got {windows.ndim}D: {windows.shape}\n"
                f"This suggests an issue with data loading or shape normalization."
            )
        
        N, C, T = windows.shape
        expected_T = 1250  # 10s @ 125Hz
        
        if T != expected_T:
            logger.warning(f"Unexpected time dimension: {T} (expected {expected_T})")
            logger.warning("This may cause issues with patch size configuration")
        
        # Convert to float32
        self.windows = windows.astype(np.float32)
        
        # Statistics
        self.mean = np.mean(self.windows)
        self.std = np.std(self.windows)
        
        logger.info(f"  Final shape: {self.windows.shape}")
        logger.info(f"  Channels: {C}, Time steps: {T}")
        logger.info(f"  Windows: {N}")
        logger.info(f"  Mean: {self.mean:.4f}, Std: {self.std:.4f}")
        logger.info(f"  Min: {self.windows.min():.4f}, Max: {self.windows.max():.4f}")
        
        # Check for NaN/Inf
        has_nan = np.any(np.isnan(self.windows))
        has_inf = np.any(np.isinf(self.windows))
        
        if has_nan or has_inf:
            logger.error(f"  ⚠️  Data quality issues: NaN={has_nan}, Inf={has_inf}")
            raise ValueError("Data contains NaN or Inf values!")
    
    def _load_paired_cases(self, directory: Path) -> np.ndarray:
        """
        Load and concatenate data from paired case files (NEW format from rebuild_vitaldb_paired.py).

        Expected structure:
          directory/case_00001_windows.npz
          directory/case_00002_windows.npz
          ...

        Each file contains:
          data: [N, 2, 1024] where channel 0=PPG, channel 1=ECG

        Returns:
          Combined array [Total_N, 2, 1024]
        """
        # Find all case files (excluding stats files)
        case_files = sorted([
            f for f in directory.glob('case_*.npz')
            if 'stats' not in f.name.lower()
        ])

        logger.info(f"  Found {len(case_files)} paired case files")

        if len(case_files) == 0:
            raise ValueError(f"No case_*.npz files found in {directory}")

        # Load all case data
        all_windows = []
        for f in case_files:
            data = np.load(f)
            if 'data' in data:
                case_windows = data['data']  # Shape: [N, 2, 1024]
                all_windows.append(case_windows)
                logger.info(f"    Loaded {f.name}: shape={case_windows.shape}")
            else:
                logger.warning(f"    Skipping {f.name}: no 'data' key found")

        if len(all_windows) == 0:
            raise ValueError(f"No valid data found in case files in {directory}")

        # Concatenate all cases
        combined = np.concatenate(all_windows, axis=0)
        logger.info(f"  Combined shape: {combined.shape} [N, 2, T]")

        return combined

    def _load_single_file(self, filepath: Path) -> np.ndarray:
        """Load data from single .npz file."""
        # Check if this is a stats file (should not be loaded as data)
        if 'stats' in filepath.name.lower():
            raise ValueError(
                f"Attempting to load stats file as data: {filepath}\n"
                f"Stats files (train_stats.npz) should not be used as input data."
            )
        
        data = np.load(filepath, allow_pickle=True)
        if 'data' in data:
            return data['data']
        elif 'windows' in data:
            return data['windows']
        elif 'signals' in data:
            return data['signals']
        else:
            keys = [k for k in data.keys() if not k.startswith('_')]
            if keys:
                logger.warning(f"Using key '{keys[0]}' from {filepath.name}")
                result = data[keys[0]]
                # Validate it's actually signal data (should be 2D or 3D)
                if isinstance(result, np.ndarray) and result.ndim >= 2:
                    return result
                else:
                    raise ValueError(
                        f"Key '{keys[0]}' in {filepath.name} is not signal data. "
                        f"Got type={type(result)}, shape={getattr(result, 'shape', 'N/A')}"
                    )
            raise ValueError(f"No data arrays found in {filepath}")
    
    def _load_separated_modalities(self, directory: Path) -> np.ndarray:
        """
        Load and merge data from separated modality folders.
        
        Expected structure:
          directory/ppg/*.npz
          directory/ecg/*.npz
        
        Returns:
          Combined array [N, 2, T] where channel 0=PPG, channel 1=ECG
        """
        ppg_dir = directory / 'ppg'
        ecg_dir = directory / 'ecg'
        
        if not ppg_dir.exists() or not ecg_dir.exists():
            raise FileNotFoundError(
                f"Expected modality folders not found:\n"
                f"  PPG: {ppg_dir} (exists={ppg_dir.exists()})\n"
                f"  ECG: {ecg_dir} (exists={ecg_dir.exists()})"
            )
        
        # Load PPG files
        ppg_files = sorted([f for f in ppg_dir.glob('*.npz') if 'stats' not in f.name.lower()])
        logger.info(f"  Found {len(ppg_files)} PPG files")
        
        if len(ppg_files) == 0:
            raise ValueError(f"No .npz data files found in {ppg_dir}")
        
        # Load ECG files
        ecg_files = sorted([f for f in ecg_dir.glob('*.npz') if 'stats' not in f.name.lower()])
        logger.info(f"  Found {len(ecg_files)} ECG files")
        
        if len(ecg_files) == 0:
            raise ValueError(f"No .npz files found in {ecg_dir}")
        
        # Load all PPG data
        ppg_arrays = []
        for f in ppg_files:
            data = self._load_single_file(f)
            logger.info(f"    Loaded {f.name}: shape={data.shape}")
            ppg_arrays.append(data)
        
        ppg_data = np.concatenate(ppg_arrays, axis=0)
        logger.info(f"  PPG combined shape: {ppg_data.shape}")
        
        # Load all ECG data
        ecg_arrays = []
        for f in ecg_files:
            data = self._load_single_file(f)
            logger.info(f"    Loaded {f.name}: shape={data.shape}")
            ecg_arrays.append(data)
        
        ecg_data = np.concatenate(ecg_arrays, axis=0)
        logger.info(f"  ECG combined shape: {ecg_data.shape}")
        
        # Verify they have same number of windows
        if ppg_data.shape[0] != ecg_data.shape[0]:
            logger.warning(
                f"PPG and ECG have different numbers of windows: "
                f"PPG={ppg_data.shape[0]}, ECG={ecg_data.shape[0]}"
            )
            # Use minimum length
            min_len = min(ppg_data.shape[0], ecg_data.shape[0])
            ppg_data = ppg_data[:min_len]
            ecg_data = ecg_data[:min_len]
            logger.info(f"  Using first {min_len} windows from each modality")
        
        # Normalize shapes to [N, T]
        def normalize_to_2d(data: np.ndarray, modality: str) -> np.ndarray:
            """Ensure data is 2D [N, T]."""
            logger.info(f"  Normalizing {modality} from shape {data.shape}...")
            
            if data.ndim == 2:
                # Already [N, T], perfect
                return data
            
            elif data.ndim == 3:
                # Could be [N, C, T] or [N, T, C]
                if data.shape[1] == 1:
                    # [N, 1, T] -> [N, T]
                    result = data.squeeze(1)
                    logger.info(f"    Squeezed axis 1: {result.shape}")
                    return result
                elif data.shape[2] == 1:
                    # [N, T, 1] -> [N, T]
                    result = data.squeeze(2)
                    logger.info(f"    Squeezed axis 2: {result.shape}")
                    return result
                else:
                    # Ambiguous, assume channels-last [N, T, C] if C is small
                    if data.shape[2] < data.shape[1]:
                        # [N, T, C] -> take first channel -> [N, T]
                        logger.warning(
                            f"    Ambiguous 3D shape {data.shape}, assuming [N,T,C] format. "
                            f"Taking first channel."
                        )
                        return data[:, :, 0]
                    else:
                        # [N, C, T] -> take first channel -> [N, T]
                        logger.warning(
                            f"    Ambiguous 3D shape {data.shape}, assuming [N,C,T] format. "
                            f"Taking first channel."
                        )
                        return data[:, 0, :]
            
            elif data.ndim == 4:
                # [N, C1, C2, T] or similar - need to flatten
                logger.warning(f"    4D data {data.shape}, reshaping to [N, -1]")
                return data.reshape(data.shape[0], -1)
            
            else:
                raise ValueError(
                    f"Cannot normalize {modality} data with shape {data.shape} to [N, T]"
                )
        
        ppg_data = normalize_to_2d(ppg_data, "PPG")
        ecg_data = normalize_to_2d(ecg_data, "ECG")
        
        logger.info(f"  Normalized PPG shape: {ppg_data.shape}")
        logger.info(f"  Normalized ECG shape: {ecg_data.shape}")
        
        # Stack into [N, 2, T]
        combined = np.stack([ppg_data, ecg_data], axis=1)
        logger.info(f"  Combined shape: {combined.shape} [N, 2, T]")
        
        return combined
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get single window.
        
        Returns:
            Tensor of shape [C, T] for SSL training
        """
        return torch.from_numpy(self.windows[idx])


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def find_preprocessed_data(
    base_dir: str = 'data/processed/vitaldb/windows',
    mode: str = 'fasttrack'
) -> Dict[str, Path]:
    """
    Find preprocessed VitalDB data from prepare_all_data.py output.
    
    Handles two formats:
    1. Single file: split/split_windows.npz
    2. Separated modalities: split/ppg/*.npz + split/ecg/*.npz
    
    Args:
        base_dir: Base directory for processed data
        mode: 'fasttrack' or 'full'
    
    Returns:
        Dictionary with paths to train/val/test data files OR directories
    """
    base_path = Path(base_dir)
    
    # Try different possible locations
    possible_bases = [
        base_path,
        Path('data/processed/vitaldb/windows'),
        Path('artifacts/raw_windows'),
        Path('data/vitaldb_windows')
    ]
    
    data_files = {}
    
    for base in possible_bases:
        # Look for split directories
        for split in ['train', 'val', 'test']:
            split_dir = base / split
            
            # Check for single .npz file
            split_file = split_dir / f'{split}_windows.npz'
            if split_file.exists():
                data_files[split] = split_file
                continue
            
            # Check for modality-separated structure (ppg/, ecg/)
            ppg_dir = split_dir / 'ppg'
            ecg_dir = split_dir / 'ecg'
            
            if ppg_dir.exists() and ecg_dir.exists():
                # Found modality-separated structure
                # Return the parent directory (e.g., train/) not the file
                data_files[split] = split_dir
                logger.info(f"  Found {split} with separated modalities: {split_dir}")
        
        # If we found train data, we're in the right place
        if 'train' in data_files:
            logger.info(f"Found preprocessed data at: {base}")
            break
    
    if 'train' not in data_files:
        logger.error(f"Could not find training data. Searched:")
        for base in possible_bases:
            logger.error(f"  - {base / 'train' / 'train_windows.npz'}")
            logger.error(f"  - {base / 'train' / 'ppg'} + {base / 'train' / 'ecg'}")
        raise FileNotFoundError(
            "Preprocessed training data not found. "
            "Please run: python scripts/prepare_all_data.py --dataset vitaldb"
        )
    
    return data_files


def create_ssl_dataloaders(
    data_files: Dict[str, Path],
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create DataLoaders for SSL training.
    
    Args:
        data_files: Dictionary with paths to train/val data
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster GPU transfer
    
    Returns:
        (train_loader, val_loader) tuple
    """
    # Ensure proper types
    batch_size = int(batch_size)
    num_workers = int(num_workers)
    
    logger.info("\n" + "=" * 70)
    logger.info("Creating SSL DataLoaders")
    logger.info("=" * 70)
    
    # Create training dataset
    train_dataset = VitalDBSSLDataset(
        data_file_or_dir=str(data_files['train']),
        transform_to_channels_first=True
    )
    
    # Adjust batch size if needed
    dataset_size = len(train_dataset)
    if batch_size > dataset_size:
        logger.warning(
            f"Batch size ({batch_size}) > dataset size ({dataset_size}). "
            f"Reducing batch size to {dataset_size}"
        )
        batch_size = dataset_size
    
    # Only drop last batch if dataset is large enough
    drop_last = dataset_size > batch_size
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Important for SSL
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last  # Only drop if we have multiple batches
    )
    
    logger.info(f"\n✓ Train loader created:")
    logger.info(f"  Samples: {len(train_dataset)}")
    logger.info(f"  Batches: {len(train_loader)}")
    logger.info(f"  Batch size: {batch_size}")
    
    # Create validation dataset if available
    val_loader = None
    if 'val' in data_files:
        val_dataset = VitalDBSSLDataset(
            data_file_or_dir=str(data_files['val']),
            transform_to_channels_first=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
        
        logger.info(f"\n✓ Validation loader created:")
        logger.info(f"  Samples: {len(val_dataset)}")
        logger.info(f"  Batches: {len(val_loader)}")
    
    elif 'test' in data_files:
        # Use test set as validation if no val set
        logger.info("\nNo validation set found, using test set for validation")
        val_dataset = VitalDBSSLDataset(
            data_file_or_dir=str(data_files['test']),
            transform_to_channels_first=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
        
        logger.info(f"\n✓ Validation loader created (using test set):")
        logger.info(f"  Samples: {len(val_dataset)}")
        logger.info(f"  Batches: {len(val_loader)}")
    else:
        logger.warning("\n⚠️  No validation data found. Training without validation.")
    
    return train_loader, val_loader


def create_ssl_model(
    config: Dict,
    device: str = 'cuda'
) -> Tuple[nn.Module, nn.Module]:
    """
    Create encoder and decoder for SSL pre-training.
    
    Args:
        config: SSL configuration dictionary
        device: Device to place models on
    
    Returns:
        (encoder, decoder) tuple
    """
    logger.info("\n" + "=" * 70)
    logger.info("Creating SSL Model (Encoder + Decoder)")
    logger.info("=" * 70)
    
    # Extract configuration
    model_cfg = config['model']
    ssl_cfg = config['ssl']
    
    input_channels = model_cfg['input_channels']
    context_length = model_cfg['context_length']
    patch_size = ssl_cfg['patch_size']
    d_model = model_cfg.get('d_model', 192)
    
    logger.info(f"\nModel Configuration:")
    logger.info(f"  Input channels: {input_channels}")
    logger.info(f"  Context length: {context_length}")
    logger.info(f"  Patch size: {patch_size}")
    logger.info(f"  Hidden dim (d_model): {d_model}")
    logger.info(f"  Number of patches: {context_length // patch_size}")
    
    # Create encoder (TTM)
    logger.info(f"\n1. Creating TTM Encoder...")
    
    encoder = TTMAdapter(
        variant=model_cfg['encoder'],
        task='ssl',  # Important: SSL task mode
        input_channels=input_channels,
        context_length=context_length,
        patch_size=patch_size,
        freeze_encoder=False,  # Train from scratch or fine-tune
        use_real_ttm=True,  # Always use real TTM (will load from config if context_length != 1024)
        decoder_mode='mix_channel'
    )
    
    encoder = encoder.to(device)
    
    logger.info(f"✓ Encoder created:")
    logger.info(f"  Using real TTM: {encoder.is_using_real_ttm()}")
    logger.info(f"  Parameters: {sum(p.numel() for p in encoder.parameters()):,}")

    # 🔧 CRITICAL FIX: Determine actual patch_size BEFORE creating decoder
    # TTM may output different patch counts than config expects
    # logger.info(f"\n1b. Determining actual encoder patch_size...")
    # logger.info(f"  Config patch_size: {patch_size}")
    # logger.info(f"  Config num_patches: {context_length // patch_size}")

    # Run a dummy forward pass to get actual encoder output dimensions
    with torch.no_grad():
        dummy_input = torch.randn(1, input_channels, context_length).to(device)
        # logger.info(f"  Running dummy forward pass with input shape: {list(dummy_input.shape)}")

        try:
            dummy_output = encoder.get_encoder_output(dummy_input)
            actual_num_patches = dummy_output.size(1)
            actual_patch_size = context_length // actual_num_patches

            # logger.info(f"  ✅ Encoder outputs {actual_num_patches} patches")
            # logger.info(f"  ✅ Actual patch_size: {actual_patch_size}")

            # Update patch_size to match encoder's actual output
            if actual_patch_size != patch_size:
                logger.warning(f"  Config patch_size ({patch_size}) != actual ({actual_patch_size}), using actual={actual_patch_size}")
                patch_size = actual_patch_size

            del dummy_input, dummy_output
            torch.cuda.empty_cache() if device == 'cuda' else None

        except Exception as e:
            logger.error(f"  ❌ Failed to determine patch_size: {e}")
            logger.warning(f"  Using config patch_size={patch_size} (may cause issues)")

    # Create decoder (Lightweight reconstruction head)
    logger.info(f"\n2. Creating Reconstruction Decoder...")
    decoder_cfg = model_cfg.get('decoder', {})
    n_channels = decoder_cfg.get('n_channels', input_channels)

    # ✅ Get actual d_model from encoder dynamically - NO HARD-CODED VALUES!
    # Query the encoder's actual output dimension to ensure decoder compatibility
    try:
        if hasattr(encoder, 'backbone') and hasattr(encoder.backbone, 'config'):
            actual_d_model = encoder.backbone.config.d_model
            logger.info(f"  Got d_model={actual_d_model} from TTM encoder config")
        elif hasattr(encoder, 'backbone') and hasattr(encoder.backbone, 'd_model'):
            actual_d_model = encoder.backbone.d_model
            logger.info(f"  Got d_model={actual_d_model} from TTM encoder attribute")
        else:
            # Fallback: calculate from patch_size based on TTM formula
            # TTM-Enhanced (patch_size=128) -> d_model=192
            # TTM formula: d_model = 64 * (128 // patch_size) for patch_size >= 32
            if patch_size >= 32:
                actual_d_model = max(64, 64 * (128 // patch_size))
            else:
                actual_d_model = 384
            logger.info(f"  Calculated d_model={actual_d_model} from patch_size={patch_size}")
    except Exception as e:
        logger.warning(f"  Could not get d_model from encoder: {e}")
        # Safe default for TTM-Enhanced (most common case)
        actual_d_model = 192
        logger.warning(f"  Using safe default d_model={actual_d_model}")

    # logger.info(f"  ✅ Using d_model={actual_d_model} for decoder (matches encoder output)")
    # logger.info(f"  ✅ Using patch_size={patch_size} for decoder (matches encoder output)")

    decoder = ReconstructionHead1D(
        d_model=actual_d_model,  # Now dynamically matches encoder!
        patch_size=patch_size,   # Now matches encoder's actual patch_size!
        n_channels=n_channels
    )

    decoder = decoder.to(device)
    
    logger.info(f"✓ Decoder created:")
    logger.info(f"  Type: ReconstructionHead1D (lightweight)")
    logger.info(f"  Output channels: {n_channels}")
    logger.info(f"  Parameters: {sum(p.numel() for p in decoder.parameters()):,}")
    
    # Total parameters
    total_params = sum(p.numel() for p in encoder.parameters()) + \
                   sum(p.numel() for p in decoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad) + \
                       sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    
    logger.info(f"\n✓ Model Summary:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Trainable %: {trainable_params/total_params*100:.1f}%")
    
    return encoder, decoder


def create_ssl_optimizer(
    encoder: nn.Module,
    decoder: nn.Module,
    config: Dict
) -> torch.optim.Optimizer:
    """
    Create optimizer for SSL training.
    
    Args:
        encoder: Encoder model
        decoder: Decoder model
        config: Training configuration
    
    Returns:
        Optimizer
    """
    train_cfg = config['training']
    
    # Get all trainable parameters
    params = list(encoder.parameters()) + list(decoder.parameters())
    params = [p for p in params if p.requires_grad]
    
    # Create optimizer
    optimizer_type = train_cfg.get('optimizer', 'adamw').lower()
    lr = float(train_cfg['lr'])  # Ensure lr is float, not string
    weight_decay = float(train_cfg.get('weight_decay', 0.01))  # Ensure float
    
    if optimizer_type == 'adamw':
        betas = train_cfg.get('betas', [0.9, 0.999])
        # Ensure betas are floats
        betas = [float(b) for b in betas]
        eps = float(train_cfg.get('eps', 1e-8))
        
        optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'adam':
        betas = train_cfg.get('betas', [0.9, 0.999])
        # Ensure betas are floats
        betas = [float(b) for b in betas]
        eps = float(train_cfg.get('eps', 1e-8))
        
        optimizer = torch.optim.Adam(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    logger.info(f"\n✓ Optimizer created:")
    logger.info(f"  Type: {optimizer_type}")
    logger.info(f"  Learning rate: {lr}")
    logger.info(f"  Weight decay: {weight_decay}")
    logger.info(f"  Trainable params: {len(params)}")
    
    return optimizer


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict,
    num_training_steps: int
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer
        config: Training configuration
        num_training_steps: Total number of training steps
    
    Returns:
        LR scheduler or None
    """
    train_cfg = config['training']
    schedule_type = train_cfg.get('schedule', 'cosine').lower()
    
    if schedule_type == 'none':
        return None
    
    warmup_epochs = int(train_cfg.get('warmup_epochs', 10))
    min_lr = float(train_cfg.get('min_lr', 1e-6))
    
    if schedule_type == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps,
            eta_min=min_lr
        )
        
        logger.info(f"\n✓ LR Scheduler created:")
        logger.info(f"  Type: Cosine Annealing")
        logger.info(f"  Min LR: {min_lr}")
        logger.info(f"  Warmup epochs: {warmup_epochs}")
        
        return scheduler
    
    return None


def run_ssl_pretraining(
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    encoder: nn.Module,
    decoder: nn.Module,
    config: Dict,
    output_dir: str = 'artifacts/foundation_model',
    resume_from: Optional[str] = None,
    device: str = 'cuda'
) -> Dict:
    """
    Run SSL pre-training loop.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        encoder: Encoder model
        decoder: Decoder model
        config: SSL configuration
        output_dir: Directory to save checkpoints
        resume_from: Path to checkpoint to resume from
        device: Device to train on
    
    Returns:
        Training history dictionary
    """
    logger.info("\n" + "=" * 70)
    logger.info("Starting SSL Pre-training")
    logger.info("=" * 70)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create optimizer
    optimizer = create_ssl_optimizer(encoder, decoder, config)
    
    # Create SSL losses
    ssl_cfg = config['ssl']
    train_cfg = config['training']

    # Ensure proper types for SSL configuration
    # 🔧 CRITICAL: Use encoder's actual patch_size (already determined in create_ssl_model)
    config_patch_size = int(ssl_cfg['patch_size'])
    actual_patch_size = encoder.patch_size if hasattr(encoder, 'patch_size') else config_patch_size

    # if actual_patch_size != config_patch_size:
    #     logger.info(f"\n⚠️  MSM Loss: Using actual patch_size={actual_patch_size} (config={config_patch_size})")

    patch_size = actual_patch_size
    mask_ratio = float(ssl_cfg['mask_ratio'])

    # MSM loss with correct patch_size
    msm_criterion = MaskedSignalModeling(
        patch_size=patch_size
    )

    # logger.info(f"\n✓ MSM Loss created with patch_size={patch_size}")
    
    # STFT loss (optional)
    stft_criterion = None
    stft_weight = 0.0
    if ssl_cfg.get('stft', {}).get('enabled', True):
        stft_cfg = ssl_cfg['stft']
        # Ensure proper types
        n_ffts = [int(x) for x in stft_cfg['n_ffts']]
        hop_lengths = [int(x) for x in stft_cfg['hop_lengths']]
        stft_weight = float(stft_cfg['loss_weight'])
        
        stft_criterion = MultiResolutionSTFT(
            n_ffts=n_ffts,
            hop_lengths=hop_lengths,
            weight=1.0,  # Will be scaled by stft_weight in trainer
            use_spectral_convergence=False
        )
        logger.info(f"\n✓ STFT Loss enabled:")
        logger.info(f"  FFT sizes: {n_ffts}")
        logger.info(f"  Weight: {stft_weight}")
    
    # Create masking function
    mask_type = ssl_cfg.get('mask_type', 'random')
    if mask_type == 'random':
        mask_fn = random_masking
    elif mask_type == 'block':
        mask_fn = block_masking
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")
    
    logger.info(f"\n✓ Masking configuration:")
    logger.info(f"  Type: {mask_type}")
    logger.info(f"  Ratio: {mask_ratio}")
    logger.info(f"  Patch size: {patch_size}")
    
    # Ensure proper types for trainer parameters
    use_amp = bool(train_cfg.get('amp', True))
    gradient_clip = float(train_cfg.get('gradient_clip', 1.0))
    
    # Create SSL trainer
    trainer = SSLTrainer(
        encoder=encoder,
        decoder=decoder,
        optimizer=optimizer,
        msm_criterion=msm_criterion,
        stft_criterion=stft_criterion,
        mask_fn=mask_fn,
        device=device,
        use_amp=use_amp,
        gradient_clip=gradient_clip,
        stft_weight=stft_weight
    )
    
    # Resume from checkpoint if specified
    if resume_from:
        logger.info(f"\nResuming from checkpoint: {resume_from}")
        trainer.load_checkpoint(resume_from)
    
    # Run training
    num_epochs = int(train_cfg.get('epochs', 100))
    batch_size = int(train_cfg.get('batch_size', 128))
    log_interval = int(train_cfg.get('log_freq', 100))
    save_interval = int(config.get('checkpoint', {}).get('save_freq', 10))
    
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Training Configuration:")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {train_cfg['lr']}")
    logger.info(f"  Device: {device}")
    logger.info(f"  AMP: {use_amp}")
    logger.info(f"  Gradient clip: {gradient_clip}")
    logger.info(f"={'=' * 70}\n")
    
    # Start training
    start_time = time.time()
    
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        save_dir=str(output_path),
        mask_ratio=mask_ratio,
        log_interval=log_interval,
        save_interval=save_interval
    )
    
    elapsed_time = time.time() - start_time
    
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Training Complete!")
    logger.info(f"  Total time: {elapsed_time/3600:.2f} hours")
    logger.info(f"  Time per epoch: {elapsed_time/num_epochs:.1f} seconds")
    logger.info(f"  Best val loss: {trainer.best_val_loss:.4f}")
    logger.info(f"  Checkpoints saved to: {output_path}")
    logger.info(f"={'=' * 70}\n")
    
    return history


def verify_gpu_setup(device: str) -> str:
    """
    Verify GPU setup and return appropriate device.
    
    Args:
        device: Requested device ('cuda', 'cpu', or 'auto')
    
    Returns:
        Actual device to use
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = 'cpu'
        else:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"\n✓ GPU Available:")
            logger.info(f"  Device: {gpu_name}")
            logger.info(f"  Memory: {gpu_memory:.1f} GB")
            
            # Test GPU
            try:
                test_tensor = torch.randn(1, 2, 1250).cuda()
                logger.info(f"  Test allocation: OK")
                del test_tensor
                torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"  GPU test failed: {e}")
                logger.warning("  Falling back to CPU")
                device = 'cpu'
    
    logger.info(f"\n✓ Using device: {device.upper()}")
    
    return device


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SSL Pre-training for VitalDB Foundation Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default config
  python scripts/pretrain_vitaldb_ssl.py
  
  # Specify custom config
  python scripts/pretrain_vitaldb_ssl.py --config configs/ssl_pretrain.yaml
  
  # FastTrack mode (10 epochs, smaller data)
  python scripts/pretrain_vitaldb_ssl.py --mode fasttrack --epochs 10
  
  # Resume from checkpoint
  python scripts/pretrain_vitaldb_ssl.py --resume artifacts/foundation_model/checkpoint_epoch_50.pt
  
  # Force CPU (for testing)
  python scripts/pretrain_vitaldb_ssl.py --device cpu
  
  # Custom output directory
  python scripts/pretrain_vitaldb_ssl.py --output my_foundation_model

Data Requirements:
  This script expects preprocessed data from prepare_all_data.py:
    - data/processed/vitaldb/windows/train/train_windows.npz
    - data/processed/vitaldb/windows/val/val_windows.npz (optional)
  
  Run first: python scripts/prepare_all_data.py --dataset vitaldb
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/ssl_pretrain.yaml',
        help='Path to SSL configuration file'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed/vitaldb/windows',
        help='Directory containing preprocessed VitalDB data'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='artifacts/foundation_model',
        help='Output directory for checkpoints and logs'
    )
    
    parser.add_argument(
        '--mode',
        choices=['fasttrack', 'full'],
        default='fasttrack',
        help='Training mode (affects data location)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (overrides config)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use for training'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=2,  # Reduced default for Colab compatibility
        help='Number of data loading workers'
    )
    
    args = parser.parse_args()
    
    # Print header
    logger.info("\n" + "=" * 70)
    logger.info("SSL PRE-TRAINING FOR VITALDB FOUNDATION MODEL")
    logger.info("=" * 70)
    logger.info(f"Project root: {project_root}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Mode: {args.mode}")
    
    # Set random seed
    set_seed(args.seed)
    logger.info(f"Random seed: {args.seed}")
    
    # Verify GPU setup
    device = verify_gpu_setup(args.device)
    
    # Load configuration
    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        logger.info("Please create configs/ssl_pretrain.yaml or specify --config")
        return 1
    
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.epochs is not None:
        config['training']['epochs'] = int(args.epochs)
        logger.info(f"Overriding epochs: {args.epochs}")
    
    if args.batch_size is not None:
        config['training']['batch_size'] = int(args.batch_size)
        logger.info(f"Overriding batch size: {args.batch_size}")
    
    if args.lr is not None:
        config['training']['lr'] = float(args.lr)
        logger.info(f"Overriding learning rate: {args.lr}")
    
    try:
        # Step 1: Find preprocessed data
        logger.info("\n" + "=" * 70)
        logger.info("Step 1: Loading Preprocessed Data")
        logger.info("=" * 70)
        
        data_files = find_preprocessed_data(
            base_dir=args.data_dir,
            mode=args.mode
        )
        
        # Step 2: Create dataloaders
        train_loader, val_loader = create_ssl_dataloaders(
            data_files=data_files,
            batch_size=config['training']['batch_size'],
            num_workers=args.num_workers,
            pin_memory=(device == 'cuda')
        )
        
        # Step 3: Create model
        logger.info("\n" + "=" * 70)
        logger.info("Step 2: Creating SSL Model")
        logger.info("=" * 70)
        
        encoder, decoder = create_ssl_model(
            config=config,
            device=device
        )
        
        # Step 4: Run training
        logger.info("\n" + "=" * 70)
        logger.info("Step 3: Running SSL Pre-training")
        logger.info("=" * 70)
        
        history = run_ssl_pretraining(
            train_loader=train_loader,
            val_loader=val_loader,
            encoder=encoder,
            decoder=decoder,
            config=config,
            output_dir=args.output,
            resume_from=args.resume,
            device=device
        )
        
        # Success!
        logger.info("\n" + "=" * 70)
        logger.info("✅ SSL PRE-TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)
        logger.info(f"\nFoundation model saved to: {args.output}")
        logger.info(f"Best checkpoint: {args.output}/best_model.pt")
        logger.info(f"Training history: {args.output}/training_history.json")
        logger.info(f"\nNext steps:")
        logger.info(f"  1. Fine-tune on BUT-PPG for signal quality task")
        logger.info(f"  2. Use foundation model for downstream tasks")
        logger.info(f"  3. Evaluate on held-out test set")
        
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ SSL pre-training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
