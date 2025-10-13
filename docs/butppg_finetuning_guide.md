# BUT-PPG Fine-tuning Guide

## Overview

This guide covers fine-tuning a 2-channel SSL pretrained model on 5-channel BUT-PPG data for PPG quality classification.

## Quick Start

### 1. Generate Test Data (if needed)
```bash
python scripts/generate_butppg_test_data.py \
    --output-dir data/but_ppg \
    --train-samples 200 \
    --val-samples 50 \
    --test-samples 100
```

### 2. Run Fine-tuning
```bash
# Quick test (1 epoch)
python scripts/finetune_butppg.py \
    --pretrained artifacts/foundation_model/best_model.pt \
    --data-dir data/but_ppg \
    --pretrain-channels 2 \
    --finetune-channels 5 \
    --unfreeze-last-n 2 \
    --epochs 1 \
    --lr 2e-5 \
    --output-dir artifacts/but_ppg_finetuned

# Full training (30 epochs)
python scripts/finetune_butppg.py \
    --pretrained artifacts/foundation_model/best_model.pt \
    --data-dir data/but_ppg \
    --pretrain-channels 2 \
    --finetune-channels 5 \
    --unfreeze-last-n 2 \
    --epochs 30 \
    --lr 2e-5 \
    --output-dir artifacts/but_ppg_finetuned
```

### 3. Test Complete Pipeline
```bash
# Run end-to-end test (SSL pretraining → fine-tuning)
chmod +x scripts/test_full_pipeline.sh
./scripts/test_full_pipeline.sh
```

---

## Channel Inflation

The fine-tuning script automatically handles channel inflation from 2→5 channels:

### Pretrained Channels (VitalDB SSL)
- Channel 0: PPG
- Channel 1: ECG

### Fine-tuned Channels (BUT-PPG)
- Channel 0: ACC_X (initialized from mean of PPG+ECG)
- Channel 1: ACC_Y (initialized from mean of PPG+ECG)
- Channel 2: ACC_Z (initialized from mean of PPG+ECG)
- Channel 3: PPG (copied from pretrained)
- Channel 4: ECG (copied from pretrained)

**Strategy:**
- Directly transfer PPG and ECG weights from pretrained model
- Initialize ACC channels from mean of PPG+ECG with small noise
- Keep pretrained weights frozen initially

---

## Staged Training Strategy

### Stage 1: Head-Only Training (Default: 5 epochs)
**Goal:** Adapt classification head to new task without disturbing pretrained features

**Configuration:**
- Encoder: **Frozen** ❄️
- Classification head: **Trainable** ✓
- Learning rate: `--lr` (default: 2e-5)

**What happens:**
- Only the classification head learns
- Pretrained encoder preserves learned representations
- New channel weights adapt to BUT-PPG data

### Stage 2: Partial Unfreezing (Remaining epochs)
**Goal:** Fine-tune last N encoder blocks for task-specific adaptation

**Configuration:**
- Encoder (last N blocks): **Trainable** ✓
- Encoder (other blocks): **Frozen** ❄️
- Classification head: **Trainable** ✓
- Learning rate: `--lr` (default: 2e-5)

**What happens:**
- Last N encoder blocks adapt to task specifics
- Earlier blocks preserve general representations
- Prevents catastrophic forgetting

### Stage 3: Full Fine-tuning (Optional)
**Goal:** Fine-grained adaptation at very low learning rate

**Configuration:**
- All parameters: **Trainable** ✓
- Learning rate: `--lr / 10` (10x lower)

**Enable with:**
```bash
python scripts/finetune_butppg.py \
    ... \
    --full-finetune \
    --full-finetune-epochs 10
```

---

## Command Line Arguments

### Required Arguments
- `--pretrained`: Path to 2-channel pretrained checkpoint
- `--data-dir`: Directory containing BUT-PPG data files

### Channel Configuration
- `--pretrain-channels`: Channels in pretrained model (default: 2)
- `--finetune-channels`: Channels for fine-tuning (default: 5)

### Training Configuration
- `--epochs`: Total training epochs (default: 30)
- `--head-only-epochs`: Epochs for Stage 1 (default: 5)
- `--unfreeze-last-n`: Blocks to unfreeze in Stage 2 (default: 2)
- `--full-finetune`: Enable Stage 3 (flag)
- `--full-finetune-epochs`: Epochs for Stage 3 (default: 10)

### Optimization
- `--lr`: Learning rate (default: 2e-5)
- `--weight-decay`: Weight decay (default: 0.01)
- `--batch-size`: Batch size (default: 32)
- `--gradient-clip`: Gradient clipping (default: 1.0)

### System
- `--device`: Device (cuda/cpu, default: auto-detect)
- `--num-workers`: Data loading workers (default: 4)
- `--no-amp`: Disable automatic mixed precision

### Output
- `--output-dir`: Output directory (default: artifacts/but_ppg_finetuned)
- `--seed`: Random seed (default: 42)

---

## Data Format

### Expected Directory Structure
```
data/but_ppg/
├── train.npz     # Training data
├── val.npz       # Validation data (optional)
└── test.npz      # Test data (optional)
```

### File Format (.npz)
Each `.npz` file must contain:
- `signals`: Array of shape `[N, 5, 1250]`
  - N: Number of samples
  - 5: Channels (ACC_X, ACC_Y, ACC_Z, PPG, ECG)
  - 1250: Time steps (10 seconds @ 125 Hz)
- `labels`: Array of shape `[N]`
  - 0: Poor quality
  - 1: Good quality

### Creating Custom Data
```python
import numpy as np

# Create synthetic data
signals = np.random.randn(100, 5, 1250).astype(np.float32)
labels = np.random.randint(0, 2, 100).astype(np.int64)

# Save
np.savez('data/but_ppg/train.npz', signals=signals, labels=labels)
```

---

## Output Files

After training, the output directory contains:

```
artifacts/but_ppg_finetuned/
├── best_model.pt              # Best checkpoint (highest val accuracy)
├── final_model.pt             # Final checkpoint
├── training_config.json       # Training configuration
├── training_history.json      # Training curves
└── test_metrics.json          # Test set results
```

### Loading Trained Model
```python
import torch
from src.models.ttm_adapter import TTMAdapter

# Load checkpoint
checkpoint = torch.load('artifacts/but_ppg_finetuned/best_model.pt')

# Create model
model_config = checkpoint['config']
model = TTMAdapter(**model_config)
model.load_state_dict(checkpoint['model_state_dict'])

# Use for inference
model.eval()
with torch.no_grad():
    logits = model(signals)  # [B, 2]
    predictions = logits.argmax(dim=1)  # [B]
```

---

## Training Monitoring

### During Training
The script prints per-epoch metrics:
```
Epoch 1/30 (45.2s)
  Train - Loss: 0.543, Acc: 72.5%
  Val   - Loss: 0.512, Acc: 75.2%
  ✓ Best model saved (val_acc: 75.2%)
```

### Training History
Inspect `training_history.json`:
```json
{
  "train_loss": [0.543, 0.489, ...],
  "train_acc": [72.5, 76.2, ...],
  "val_loss": [0.512, 0.478, ...],
  "val_acc": [75.2, 78.1, ...],
  "stage": ["stage1_head_only", "stage1_head_only", "stage2_partial_unfreeze", ...]
}
```

### Test Metrics
Inspect `test_metrics.json`:
```json
{
  "loss": 0.456,
  "accuracy": 81.3,
  "class_0_acc": 79.5,
  "class_1_acc": 83.2
}
```

---

## Expected Performance

### Baseline (Random Initialization)
- Validation accuracy: 60-70%
- Converges slowly (30+ epochs)

### With SSL Pretraining
- Validation accuracy: 75-85%
- Converges faster (10-15 epochs)
- Better per-class balance

### Performance Factors
- **Dataset size**: More data → better performance
- **Quality ratio**: Balanced classes → better performance
- **Pretraining quality**: Better SSL → better fine-tuning
- **Unfreezing strategy**: Staged → prevents overfitting

---

## Troubleshooting

### Issue: Out of Memory
**Solution:**
```bash
# Reduce batch size
--batch-size 16

# Disable AMP
--no-amp

# Use CPU
--device cpu
```

### Issue: Poor Performance
**Checklist:**
1. ✓ Verify data quality and balance
2. ✓ Check pretrained model is loaded correctly
3. ✓ Ensure learning rate is appropriate (try 1e-5 to 1e-4)
4. ✓ Increase training epochs
5. ✓ Try different unfreezing strategies

### Issue: Overfitting
**Solutions:**
- Use more training data
- Reduce `--unfreeze-last-n` (freeze more layers)
- Increase `--weight-decay`
- Keep encoder frozen longer (`--head-only-epochs 10`)

### Issue: Data Loading Error
**Check:**
- Files exist: `train.npz`, `val.npz`, `test.npz`
- Correct shapes: `signals [N, 5, 1250]`, `labels [N]`
- Data types: `signals=float32`, `labels=int64`

---

## Advanced Usage

### Custom Unfreezing Schedule
```bash
# Conservative: unfreeze only 1 block
python scripts/finetune_butppg.py ... --unfreeze-last-n 1

# Aggressive: unfreeze 4 blocks
python scripts/finetune_butppg.py ... --unfreeze-last-n 4
```

### Three-Stage Training
```bash
python scripts/finetune_butppg.py \
    --pretrained artifacts/foundation_model/best_model.pt \
    --data-dir data/but_ppg \
    --epochs 30 \
    --head-only-epochs 5 \
    --unfreeze-last-n 2 \
    --full-finetune \
    --full-finetune-epochs 10 \
    --lr 2e-5 \
    --output-dir artifacts/but_ppg_finetuned
```

### Learning Rate Tuning
```bash
# Lower LR for more stable training
--lr 1e-5

# Higher LR for faster convergence
--lr 5e-5
```

---

## References

- **Progressive Unfreezing**: Howard & Ruder (ULMFiT), 2018
- **Channel Inflation**: Carreira & Zisserman (I3D), 2017
- **Transfer Learning**: Yosinski et al., 2014

---

## Complete Pipeline Example

```bash
# 1. Generate test data
python scripts/generate_butppg_test_data.py \
    --output-dir data/but_ppg \
    --train-samples 500 \
    --val-samples 100 \
    --test-samples 200

# 2. SSL pretraining (if not done)
python scripts/pretrain_vitaldb_ssl.py \
    --config configs/ssl_pretrain.yaml \
    --data-dir data/vitaldb_windows \
    --output-dir artifacts/foundation_model \
    --epochs 100 \
    --batch-size 128

# 3. Fine-tuning on BUT-PPG
python scripts/finetune_butppg.py \
    --pretrained artifacts/foundation_model/best_model.pt \
    --data-dir data/but_ppg \
    --pretrain-channels 2 \
    --finetune-channels 5 \
    --unfreeze-last-n 2 \
    --epochs 30 \
    --lr 2e-5 \
    --output-dir artifacts/but_ppg_finetuned

# 4. Evaluate results
python -c "
import json
with open('artifacts/but_ppg_finetuned/test_metrics.json') as f:
    metrics = json.load(f)
    print(f'Test Accuracy: {metrics[\"accuracy\"]:.2f}%')
    print(f'Class 0 Acc: {metrics[\"class_0_acc\"]:.2f}%')
    print(f'Class 1 Acc: {metrics[\"class_1_acc\"]:.2f}%')
"
```

---

**For questions or issues, refer to the main project documentation.**
