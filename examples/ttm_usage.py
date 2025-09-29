"""Example usage of TTM adapter with different configurations."""

import torch
from src.models import create_ttm_model, TTM_AVAILABLE


def example_frozen_fm():
    """Example: Use TTM as frozen foundation model (default)."""
    if not TTM_AVAILABLE:
        print("TTM not available. Install with: pip install tinytimemixers")
        return
    
    print("=" * 60)
    print("Example 1: Frozen Foundation Model (Default)")
    print("=" * 60)
    
    # Configuration for frozen FM
    config = {
        'variant': 'ttm-512-96',
        'task': 'classification',
        'num_classes': 2,
        'head_type': 'linear',
        'freeze_encoder': True,  # Frozen encoder
        'input_channels': 1,
        'context_length': 96
    }
    
    model = create_ttm_model(config)
    model.print_parameter_summary()
    
    # Create dummy input
    batch_size = 8
    x = torch.randn(batch_size, 96, 1)
    
    # Forward pass
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Check trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")


def example_partial_unfreeze():
    """Example: Partial unfreezing of last N blocks."""
    if not TTM_AVAILABLE:
        return
    
    print("\n" + "=" * 60)
    print("Example 2: Partial Unfreeze (Last 2 Blocks)")
    print("=" * 60)
    
    config = {
        'variant': 'ttm-512-96',
        'task': 'classification',
        'num_classes': 2,
        'head_type': 'mlp',
        'head_config': {'hidden_dims': [256], 'dropout': 0.2},
        'freeze_encoder': True,
        'unfreeze_last_n_blocks': 2,  # Unfreeze last 2 blocks
        'input_channels': 1,
        'context_length': 96
    }
    
    model = create_ttm_model(config)
    model.print_parameter_summary()
    
    # Check which parameters are trainable
    encoder_trainable = sum(
        p.numel() for p in model.encoder.parameters() if p.requires_grad
    )
    print(f"Encoder trainable parameters: {encoder_trainable:,}")


def example_lora_adaptation():
    """Example: LoRA adaptation for efficient fine-tuning."""
    if not TTM_AVAILABLE:
        return
    
    print("\n" + "=" * 60)
    print("Example 3: LoRA Adaptation")
    print("=" * 60)
    
    config = {
        'variant': 'ttm-512-96',
        'task': 'regression',
        'out_features': 1,
        'head_type': 'linear',
        'freeze_encoder': True,
        'lora': {
            'enabled': True,
            'r': 8,
            'alpha': 16,
            'dropout': 0.1,
            'target_modules': ['mixer', 'mlp']
        },
        'input_channels': 3,  # Multi-channel input
        'context_length': 96
    }
    
    model = create_ttm_model(config)
    
    # LoRA summary is printed automatically
    
    # Test forward pass
    x = torch.randn(4, 96, 3)
    output = model(x)
    print(f"\nMulti-channel input shape: {x.shape}")
    print(f"Regression output shape: {output.shape}")


def example_multimodal_input():
    """Example: Multi-modal input (ECG + PPG + ABP)."""
    if not TTM_AVAILABLE:
        return
    
    print("\n" + "=" * 60)
    print("Example 4: Multi-Modal Input")
    print("=" * 60)
    
    config = {
        'variant': 'ttm-1024-96',  # Larger model
        'task': 'classification',
        'num_classes': 5,  # Multi-class
        'head_type': 'mlp',
        'head_config': {
            'hidden_dims': [512, 256],
            'dropout': 0.2,
            'use_batch_norm': True
        },
        'freeze_encoder': False,  # Full fine-tuning
        'input_channels': 3,  # ECG + PPG + ABP
        'context_length': 96
    }
    
    model = create_ttm_model(config)
    
    # Simulate multi-modal input
    batch_size = 16
    seq_len = 96
    
    # Stack channels: [ECG, PPG, ABP]
    ecg = torch.randn(batch_size, seq_len, 1)
    ppg = torch.randn(batch_size, seq_len, 1)
    abp = torch.randn(batch_size, seq_len, 1)
    
    x = torch.cat([ecg, ppg, abp], dim=-1)  # [B, T, 3]
    
    output, features = model(x, return_features=True)
    print(f"Multi-modal input shape: {x.shape}")
    print(f"Features shape: {features.shape if features is not None else 'None'}")
    print(f"Output shape: {output.shape}")
    
    model.print_parameter_summary()


def example_training_loop():
    """Example: Simple training loop with LoRA."""
    if not TTM_AVAILABLE:
        return
    
    print("\n" + "=" * 60)
    print("Example 5: Training Loop with LoRA")
    print("=" * 60)
    
    # Create model with LoRA
    config = {
        'variant': 'ttm-512-96',
        'task': 'classification',
        'num_classes': 2,
        'freeze_encoder': True,
        'lora': {
            'enabled': True,
            'r': 4,
            'alpha': 8
        },
        'input_channels': 1,
        'context_length': 96
    }
    
    model = create_ttm_model(config)
    
    # Create optimizer with only trainable parameters
    trainable_params = model.get_trainable_params()
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
    
    # Loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Dummy training step
    model.train()
    for i in range(3):
        # Dummy batch
        x = torch.randn(8, 96, 1)
        y = torch.randint(0, 2, (8,))
        
        # Forward pass
        logits = model(x)
        loss = criterion(logits, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients
        encoder_grads = sum(
            1 for p in model.encoder.parameters() 
            if p.grad is not None and p.grad.sum() != 0
        )
        head_grads = sum(
            1 for p in model.head.parameters() 
            if p.grad is not None and p.grad.sum() != 0
        )
        
        print(f"Step {i+1}: Loss={loss.item():.4f}, "
              f"Encoder grads={encoder_grads}, Head grads={head_grads}")
        
        # Update weights
        optimizer.step()


if __name__ == "__main__":
    # Run examples
    example_frozen_fm()
    example_partial_unfreeze()
    example_lora_adaptation()
    example_multimodal_input()
    example_training_loop()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
