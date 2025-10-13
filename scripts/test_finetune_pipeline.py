#!/usr/bin/env python3
"""Quick smoke test for the complete SSL → Fine-tuning pipeline.

This script tests:
1. Mock BUT-PPG data creation
2. Fine-tuning with channel inflation (2→5 channels)
3. Staged unfreezing strategy

Usage:
    python scripts/test_finetune_pipeline.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import subprocess
import shutil
import torch
import numpy as np


def run_command(cmd: list, description: str):
    """Run a command and handle errors."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n❌ FAILED: {description}")
        sys.exit(1)
    
    print(f"\n✓ SUCCESS: {description}")
    return result


def create_mock_ssl_checkpoint(output_path: Path):
    """Create a mock SSL pretrained checkpoint for testing.
    
    This creates a minimal checkpoint that mimics the structure of a
    real SSL pretrained model with 2 channels.
    """
    print(f"\n{'='*70}")
    print("CREATING MOCK SSL CHECKPOINT")
    print(f"{'='*70}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create mock encoder state dict
    # This mimics TTM encoder structure
    encoder_state = {
        'backbone.encoder.layer.0.attention.attention.query.weight': torch.randn(192, 192),
        'backbone.encoder.layer.0.attention.attention.key.weight': torch.randn(192, 192),
        'backbone.encoder.layer.0.attention.attention.value.weight': torch.randn(192, 192),
    }
    
    # Create mock decoder state dict
    decoder_state = {
        'decoder.linear.weight': torch.randn(2500, 1920),  # [C*T, P*D]
        'decoder.linear.bias': torch.randn(2500),
    }
    
    checkpoint = {
        'epoch': 100,
        'encoder_state_dict': encoder_state,
        'decoder_state_dict': decoder_state,
        'best_val_loss': 0.123,
        'metrics': {'loss': 0.123, 'msm_loss': 0.100, 'stft_loss': 0.023},
        'config': {
            'input_channels': 2,
            'context_length': 1250,
            'patch_size': 125,
            'gradient_clip': 1.0,
            'stft_weight': 0.3,
            'use_amp': True
        }
    }
    
    torch.save(checkpoint, output_path)
    print(f"✓ Mock checkpoint saved: {output_path}")
    print(f"  - Input channels: 2")
    print(f"  - Encoder layers: {len([k for k in encoder_state.keys() if 'encoder' in k])}")
    print(f"  - Decoder params: {len(decoder_state)}")


def main():
    """Run complete pipeline smoke test."""
    print("\n" + "="*70)
    print("FINE-TUNING PIPELINE SMOKE TEST")
    print("="*70)
    print("\nThis will test:")
    print("1. Mock data creation (BUT-PPG, 5 channels)")
    print("2. Mock SSL checkpoint (2 channels)")
    print("3. Channel inflation (2→5)")
    print("4. Staged fine-tuning (1 epoch each stage)")
    print("="*70)
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data' / 'but_ppg_test'
    checkpoint_dir = project_root / 'artifacts' / 'ssl_test'
    output_dir = project_root / 'artifacts' / 'butppg_test'
    
    # Clean up previous test artifacts
    if data_dir.exists():
        shutil.rmtree(data_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    try:
        # Step 1: Create mock BUT-PPG data
        run_command(
            [
                'python', 'scripts/create_mock_butppg_data.py',
                '--output', str(data_dir),
                '--samples', '30',  # Small dataset for testing
            ],
            "Step 1: Creating mock BUT-PPG data"
        )
        
        # Verify data was created
        assert (data_dir / 'train.npz').exists(), "train.npz not found"
        assert (data_dir / 'val.npz').exists(), "val.npz not found"
        assert (data_dir / 'test.npz').exists(), "test.npz not found"
        print("\n✓ Data files verified")
        
        # Load and check data format
        train_data = np.load(data_dir / 'train.npz')
        assert 'signals' in train_data, "signals not in train.npz"
        assert 'labels' in train_data, "labels not in train.npz"
        signals = train_data['signals']
        labels = train_data['labels']
        print(f"  - Train signals shape: {signals.shape}")
        print(f"  - Train labels shape: {labels.shape}")
        assert signals.shape[1] == 5, f"Expected 5 channels, got {signals.shape[1]}"
        assert signals.shape[2] == 1250, f"Expected 1250 timesteps, got {signals.shape[2]}"
        
        # Step 2: Create mock SSL checkpoint
        checkpoint_path = checkpoint_dir / 'best_model.pt'
        create_mock_ssl_checkpoint(checkpoint_path)
        
        # Step 3: Run fine-tuning with staged unfreezing
        run_command(
            [
                'python', 'scripts/finetune_butppg.py',
                '--pretrained', str(checkpoint_path),
                '--data-dir', str(data_dir),
                '--pretrain-channels', '2',
                '--finetune-channels', '5',
                '--epochs', '3',  # Quick test: 1 epoch stage1, 2 epochs stage2
                '--head-only-epochs', '1',
                '--unfreeze-last-n', '1',
                '--lr', '2e-5',
                '--batch-size', '4',
                '--output-dir', str(output_dir),
                '--num-workers', '0',  # Avoid multiprocessing issues
            ],
            "Step 3: Fine-tuning with channel inflation"
        )
        
        # Step 4: Verify outputs
        print(f"\n{'='*70}")
        print("VERIFYING OUTPUTS")
        print(f"{'='*70}")
        
        assert (output_dir / 'best_model.pt').exists(), "best_model.pt not found"
        assert (output_dir / 'final_model.pt').exists(), "final_model.pt not found"
        assert (output_dir / 'training_history.json').exists(), "training_history.json not found"
        assert (output_dir / 'training_config.json').exists(), "training_config.json not found"
        
        # Load and check checkpoint
        checkpoint = torch.load(output_dir / 'best_model.pt', map_location='cpu')
        assert 'model_state_dict' in checkpoint, "model_state_dict not in checkpoint"
        assert 'metrics' in checkpoint, "metrics not in checkpoint"
        
        print(f"\n✓ All output files created:")
        print(f"  - best_model.pt")
        print(f"  - final_model.pt")
        print(f"  - training_history.json")
        print(f"  - training_config.json")
        
        # Load training history
        import json
        with open(output_dir / 'training_history.json', 'r') as f:
            history = json.load(f)
        
        print(f"\nTraining history:")
        print(f"  - Epochs completed: {len(history['train_loss'])}")
        print(f"  - Final train accuracy: {history['train_acc'][-1]:.2f}%")
        print(f"  - Final val accuracy: {history['val_acc'][-1]:.2f}%")
        print(f"  - Stages: {set(history['stage'])}")
        
        # Check that we have both stages
        assert 'stage1_head_only' in history['stage'], "Stage 1 not executed"
        assert 'stage2_partial_unfreeze' in history['stage'], "Stage 2 not executed"
        
        print(f"\n{'='*70}")
        print("✅ ALL TESTS PASSED")
        print(f"{'='*70}")
        print("\nThe fine-tuning pipeline is working correctly!")
        print(f"\nTest artifacts saved to:")
        print(f"  - Data: {data_dir}")
        print(f"  - Checkpoints: {output_dir}")
        print(f"\nYou can inspect the outputs or clean them up with:")
        print(f"  rm -rf {data_dir}")
        print(f"  rm -rf {output_dir}")
        print(f"{'='*70}")
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ TEST FAILED")
        print(f"{'='*70}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
