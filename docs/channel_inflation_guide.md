# Channel Inflation Usage Guide

## Overview

Channel inflation utilities enable transfer learning from SSL pretraining (2 channels: PPG+ECG) to downstream fine-tuning (5 channels: ACC_X, ACC_Y, ACC_Z, PPG, ECG).

## Quick Start

```python
from src.models.channel_utils import (
    load_pretrained_with_channel_inflate,
    unfreeze_last_n_blocks
)

# Load pretrained model and inflate channels
model_config = {
    'variant': 'ibm-granite/granite-timeseries-ttm-r1',
    'task': 'classification',
    'num_classes': 2,
    'input_channels': 5,
    'context_length': 1250,
    'patch_size': 125
}

model = load_pretrained_with_channel_inflate(
    checkpoint_path='checkpoints/ssl_pretrained.pt',
    pretrain_channels=2,
    finetune_channels=5,
    freeze_pretrained=True,
    model_config=model_config
)

# Train, then progressively unfreeze
unfreeze_last_n_blocks(model, n=2)
```

## Key Functions

### 1. load_pretrained_with_channel_inflate()

Load 2-ch pretrained model and inflate to 5-ch for fine-tuning.

**Parameters:**
- `checkpoint_path`: Path to pretrained checkpoint
- `pretrain_channels`: Original channel count (default: 2)
- `finetune_channels`: Target channel count (default: 5)
- `freeze_pretrained`: Freeze pretrained weights (default: True)
- `model_config`: Model configuration dict

### 2. unfreeze_last_n_blocks()

Progressively unfreeze transformer blocks.

**Parameters:**
- `model`: Model to unfreeze
- `n`: Number of blocks to unfreeze from end
- `verbose`: Print unfreezing info

### 3. verify_channel_inflation()

Verify weight transfer correctness.

**Parameters:**
- `model_2ch`: Original 2-channel model
- `model_5ch`: Inflated 5-channel model
- `verbose`: Print verification details

## Channel Inflation Strategy

### Weight Transfer:

1. **Direct Copy** (Exact match):
   - Transformer blocks
   - Layer normalization
   - Position embeddings

2. **Inflated** (Channel-dependent):
   - Input projection
   - Channel embeddings

### New Channel Initialization:

```
[0] PPG    ← Copy from pretrained
[1] ECG    ← Copy from pretrained
[2] ACC_X  ← Mean(PPG, ECG) + noise
[3] ACC_Y  ← Mean(PPG, ECG) + noise
[4] ACC_Z  ← Mean(PPG, ECG) + noise
```

## Training Strategies

### Strategy 1: Frozen Encoder (Recommended)

```python
model = load_pretrained_with_channel_inflate(
    ..., freeze_pretrained=True
)
# Train only: new channel weights + task head
```

**Use case**: Limited data, avoid catastrophic forgetting

### Strategy 2: Progressive Unfreezing

```python
model = load_pretrained_with_channel_inflate(..., freeze_pretrained=True)
train(model, epochs=5)

unfreeze_last_n_blocks(model, n=2)
train(model, epochs=5, lr=5e-4)

unfreeze_last_n_blocks(model, n=4)
train(model, epochs=5, lr=1e-4)
```

**Use case**: Medium datasets, balanced adaptation

### Strategy 3: Full Fine-tuning

```python
model = load_pretrained_with_channel_inflate(..., freeze_pretrained=True)
train(model, epochs=5)  # Warm up

for param in model.parameters():
    param.requires_grad = True
train(model, epochs=10, lr=1e-5)
```

**Use case**: Large datasets, significant domain shift

## Complete Example

```python
import torch
from src.models.channel_utils import (
    load_pretrained_with_channel_inflate,
    unfreeze_last_n_blocks
)

# 1. Load and inflate
model = load_pretrained_with_channel_inflate(
    checkpoint_path='checkpoints/vitaldb_ssl.pt',
    pretrain_channels=2,
    finetune_channels=5,
    freeze_pretrained=True,
    model_config={
        'task': 'classification',
        'num_classes': 2,
        'input_channels': 5,
        'context_length': 1250
    }
)

# 2. Setup training
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-3
)

# 3. Training with progressive unfreezing
for epoch in range(20):
    if epoch == 5:
        unfreeze_last_n_blocks(model, n=2)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=5e-4
        )
    
    # Train epoch...
    model.train()
    for x, y in train_loader:  # x: [B, 5, 1250]
        logits = model(x)
        loss = criterion(logits, y)
        # Backprop...
```

## Troubleshooting

**Shape mismatch errors:**
- Ensure `model_config['input_channels']` equals `finetune_channels`

**No parameters inflated:**
- Check checkpoint has channel-dependent layers
- Use `get_channel_inflation_report()` to inspect

**All parameters trainable:**
- Verify `freeze_pretrained=True`
- Check model has 'backbone' or 'encoder' attribute

## References

- Transfer Learning: [doi.org/10.1145/3459637](https://doi.org/10.1145/3459637)
- Progressive Unfreezing: Howard & Ruder (ULMFiT), 2018
- Channel Inflation: Carreira & Zisserman (I3D), 2017
