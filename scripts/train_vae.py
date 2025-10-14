#!/usr/bin/env python3
"""
Complete TTM × VitalDB pipeline with VAE support.
Includes unsupervised VAE pretraining and supervised fine-tuning.
"""

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.splits import make_patient_level_splits
from src.data.vitaldb_loader import list_cases, load_channel, get_available_case_sets
from src.data.windows import (
    make_windows, compute_normalization_stats, normalize_windows, NormalizationStats
)
from src.data.filters import apply_bandpass_filter
from src.data.detect import find_ppg_peaks, find_ecg_rpeaks
from src.data.quality import compute_sqi
from src.models.ttm_adapter import create_ttm_model
from src.models.vae_adapter import VAEAdapter
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_vae_unsupervised(model, train_loader, val_loader, config, device, save_dir):
    """
    Train VAE in unsupervised mode (reconstruction + KL).
    
    Args:
        model: VAEAdapter model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        device: Device to train on
        save_dir: Directory to save checkpoints
    """
    logger.info("Starting VAE unsupervised pretraining...")
    
    # Ensure VAE mode
    if not hasattr(model, 'vae'):
        raise ValueError("Model must be a VAEAdapter for unsupervised training")
    
    # Optimizer for VAE only
    optimizer = torch.optim.Adam(model.vae.parameters(), lr=config.get('learning_rate', 1e-3))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # KL annealing setup
    kl_annealing = config.get('kl_annealing', False)
    if kl_annealing:
        min_kl = config.get('min_kl_weight', 0.01)
        max_kl = config.get('max_kl_weight', 1.0)
        annealing_epochs = config.get('annealing_epochs', 10)
    
    best_val_loss = float('inf')
    num_epochs = config.get('vae_pretrain_epochs', 50)
    
    for epoch in range(num_epochs):
        model.train()
        train_losses = {'total': 0, 'recon': 0, 'kl': 0}
        
        # KL weight for this epoch
        if kl_annealing:
            kl_weight = min(max_kl, min_kl + (max_kl - min_kl) * (epoch / annealing_epochs))
            model.vae.beta = kl_weight
            logger.info(f"Epoch {epoch+1}: KL weight = {kl_weight:.3f}")
        
        # Training loop
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(device)
            
            # Get VAE losses
            total_loss, recon_loss, kl_loss = model.get_vae_loss(data)
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Update statistics
            train_losses['total'] += total_loss.item()
            train_losses['recon'] += recon_loss.item()
            train_losses['kl'] += kl_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'recon': f"{recon_loss.item():.4f}",
                'kl': f"{kl_loss.item():.4f}"
            })
        
        # Average training losses
        for key in train_losses:
            train_losses[key] /= len(train_loader)
        
        # Validation
        model.eval()
        val_losses = {'total': 0, 'recon': 0, 'kl': 0}
        
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                total_loss, recon_loss, kl_loss = model.get_vae_loss(data)
                
                val_losses['total'] += total_loss.item()
                val_losses['recon'] += recon_loss.item()
                val_losses['kl'] += kl_loss.item()
        
        # Average validation losses
        for key in val_losses:
            val_losses[key] /= len(val_loader)
        
        # Log epoch results
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"  Train - Total: {train_losses['total']:.4f}, "
                   f"Recon: {train_losses['recon']:.4f}, KL: {train_losses['kl']:.4f}")
        logger.info(f"  Val - Total: {val_losses['total']:.4f}, "
                   f"Recon: {val_losses['recon']:.4f}, KL: {val_losses['kl']:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_losses['total'])
        
        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            save_path = Path(save_dir) / 'vae_pretrained.pt'
            torch.save(checkpoint, save_path)
            logger.info(f"  ✓ Saved best VAE model (val_loss={best_val_loss:.4f})")
    
    logger.info(f"VAE pretraining complete! Best val_loss: {best_val_loss:.4f}")
    return best_val_loss


def train_vae_supervised(model, train_loader, val_loader, config, device, save_dir):
    """
    Fine-tune VAE with supervised task (classification/regression).
    
    Args:
        model: VAEAdapter model with task head
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        device: Device to train on
        save_dir: Directory to save checkpoints
    """
    logger.info("Starting VAE supervised fine-tuning...")
    
    # Task setup
    task = config.get('task', 'classification')
    if task == 'classification':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    
    # Optimizer - train whole model or just head based on config
    if config.get('freeze_encoder', False):
        # Freeze VAE encoder
        for param in model.vae.encoder.parameters():
            param.requires_grad = False
        parameters = [p for p in model.parameters() if p.requires_grad]
        logger.info("  Encoder frozen, training head only")
    else:
        parameters = model.parameters()
        logger.info("  Training full model (encoder + head)")
    
    optimizer = torch.optim.Adam(parameters, lr=config.get('learning_rate', 1e-4))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training settings
    num_epochs = config.get('num_epochs', 30)
    vae_loss_weight = config.get('vae_loss_weight', 0.5)
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        train_losses = {'total': 0, 'task': 0, 'vae': 0}
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            # Task loss
            if task == 'classification':
                task_loss = criterion(output, target)
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
            else:
                task_loss = criterion(output.squeeze(), target.float())
            
            # VAE loss (optional during fine-tuning)
            if vae_loss_weight > 0:
                vae_total, _, _ = model.get_vae_loss(data)
                total_loss = task_loss + vae_loss_weight * vae_total
            else:
                total_loss = task_loss
                vae_total = torch.tensor(0.0)
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Update statistics
            train_losses['total'] += total_loss.item()
            train_losses['task'] += task_loss.item()
            train_losses['vae'] += vae_total.item() if vae_loss_weight > 0 else 0
            
            # Update progress bar
            pbar_dict = {
                'loss': f"{total_loss.item():.4f}",
                'task': f"{task_loss.item():.4f}"
            }
            if task == 'classification' and train_total > 0:
                pbar_dict['acc'] = f"{100.*train_correct/train_total:.2f}%"
            pbar.set_postfix(pbar_dict)
        
        # Average training losses
        for key in train_losses:
            train_losses[key] /= len(train_loader)
        
        if task == 'classification':
            train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_losses = {'total': 0, 'task': 0}
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                output = model(data)
                
                if task == 'classification':
                    val_loss = criterion(output, target)
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
                else:
                    val_loss = criterion(output.squeeze(), target.float())
                
                val_losses['task'] += val_loss.item()
        
        # Average validation losses
        val_losses['task'] /= len(val_loader)
        
        if task == 'classification':
            val_acc = 100. * val_correct / val_total
        
        # Log epoch results
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        if task == 'classification':
            logger.info(f"  Train - Loss: {train_losses['task']:.4f}, Acc: {train_acc:.2f}%")
            logger.info(f"  Val - Loss: {val_losses['task']:.4f}, Acc: {val_acc:.2f}%")
        else:
            logger.info(f"  Train - Loss: {train_losses['task']:.4f}")
            logger.info(f"  Val - Loss: {val_losses['task']:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_losses['task'])
        
        # Save best model
        save_best = False
        if task == 'classification' and val_acc > best_val_acc:
            best_val_acc = val_acc
            save_best = True
        elif val_losses['task'] < best_val_loss:
            best_val_loss = val_losses['task']
            save_best = True
        
        if save_best:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_losses['task'],
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            if task == 'classification':
                checkpoint['val_acc'] = val_acc
                checkpoint['train_acc'] = train_acc
            
            save_path = Path(save_dir) / 'vae_finetuned.pt'
            torch.save(checkpoint, save_path)
            
            if task == 'classification':
                logger.info(f"  ✓ Saved best model (val_acc={val_acc:.2f}%)")
            else:
                logger.info(f"  ✓ Saved best model (val_loss={val_losses['task']:.4f})")
    
    logger.info("Fine-tuning complete!")
    if task == 'classification':
        logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    else:
        logger.info(f"Best validation loss: {best_val_loss:.4f}")


def train_vae_command(args):
    """Main VAE training command."""
    logger.info("Training VAE model...")
    
    # Load configurations
    model_config = load_config(args.model_yaml)
    run_config = load_config(args.run_yaml)
    
    # Set seed
    set_seed(run_config.get('seed', 42))
    
    # Device
    device = torch.device(run_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    logger.info(f"Using device: {device}")
    
    # Load data
    train_file = Path(args.outdir) / 'train' / 'train_windows.npz'
    val_file = Path(args.outdir) / 'val' / 'val_windows.npz'
    
    if not train_file.exists():
        train_file = Path(args.outdir) / 'train_windows.npz'
    if not val_file.exists():
        val_file = Path(args.outdir) / 'val_windows.npz'
    
    train_data = np.load(train_file)
    val_data = np.load(val_file) if val_file.exists() else None
    
    logger.info(f"Loaded training data: {train_data['data'].shape}")
    if val_data:
        logger.info(f"Loaded validation data: {val_data['data'].shape}")
    
    # Create datasets
    from torch.utils.data import TensorDataset, DataLoader
    
    X_train = torch.from_numpy(train_data['data']).float()
    y_train = torch.from_numpy(train_data['labels']).long()
    train_dataset = TensorDataset(X_train, y_train)
    
    if val_data is not None:
        X_val = torch.from_numpy(val_data['data']).float()
        y_val = torch.from_numpy(val_data['labels']).long()
        val_dataset = TensorDataset(X_val, y_val)
    else:
        # Split training data
        split_idx = int(0.9 * len(train_dataset))
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [split_idx, len(train_dataset) - split_idx]
        )
    
    # Create data loaders
    batch_size = run_config.get('batch_size', 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Configure model
    model_config['input_channels'] = train_data['data'].shape[2] if train_data['data'].ndim == 3 else 1
    model_config['context_length'] = train_data['data'].shape[1]
    
    # Create model
    model = create_ttm_model(model_config)
    model = model.to(device)
    
    logger.info(f"Created VAE model: latent_dim={model_config.get('latent_dim', 64)}, "
               f"beta={model_config.get('beta', 1.0)}")
    
    # Create save directory
    save_dir = Path(args.out)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Stage 1: Unsupervised pretraining (if not loading pretrained)
    if not args.load_pretrained and not model_config.get('skip_pretraining', False):
        train_vae_unsupervised(model, train_loader, val_loader, run_config, device, save_dir)
    elif args.load_pretrained:
        # Load pretrained VAE
        checkpoint = torch.load(args.load_pretrained, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded pretrained VAE from {args.load_pretrained}")
    
    # Stage 2: Supervised fine-tuning (if task specified)
    if model_config.get('task') in ['classification', 'regression']:
        # Optionally freeze encoder after pretraining
        if run_config.get('freeze_after_pretrain', True):
            model_config['freeze_encoder'] = True
            # Recreate model with frozen encoder
            model = create_ttm_model(model_config)
            model = model.to(device)
            
            # Load pretrained weights
            pretrained_path = save_dir / 'vae_pretrained.pt'
            if pretrained_path.exists():
                checkpoint = torch.load(pretrained_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                logger.info("Loaded pretrained VAE weights for fine-tuning")
        
        train_vae_supervised(model, train_loader, val_loader, run_config, device, save_dir)
    
    logger.info(f"Training complete! Models saved to {save_dir}")


def main():
    """Main entry point with VAE support."""
    parser = argparse.ArgumentParser(description="VAE Training Pipeline")
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s:%(message)s'
    )
    
    # Training arguments
    parser.add_argument('--model-yaml', type=str, required=True, help='Model config (VAE)')
    parser.add_argument('--run-yaml', type=str, required=True, help='Run config')
    parser.add_argument('--outdir', type=str, required=True, help='Data directory')
    parser.add_argument('--out', type=str, required=True, help='Output directory')
    parser.add_argument('--load-pretrained', type=str, help='Path to pretrained VAE')
    
    args = parser.parse_args()
    
    train_vae_command(args)


if __name__ == "__main__":
    main()
