# BUT-PPG Fine-Tuning Guide

## Overview

Fine-tune a 2-channel SSL pretrained model on 5-channel BUT-PPG data for PPG quality classification using **channel inflation** and **staged unfreezing**.

---

## Quick Start

### 1. Create Mock Data (for testing)

```bash
python scripts/create_mock_butppg_data.py \
    --output data/but_ppg \
    --samples 100
```

### 2. Run Fine-Tuning

```bash
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

### 3. Run Complete Pipeline Test

```bash
python scripts/test_finetune_pipeline.py
```

---

## Channel Inflation Strategy

### Pretrained Model (2 channels):
```
[0] PPG  ← Pretrained on VitalDB
[1] ECG  ← Pretrained on VitalDB
```

### Fine-tuned Model (5 channels):
```
[0] ACC_X ← Initialized: Mean(PPG, ECG) + noise
[1] ACC_Y ← Initialized: Mean(PPG, ECG) + noise
[2] ACC_Z ← Initialized: Mean(PPG, ECG) + noise
[3] PPG   ← **Transferred directly** from pretrained
[4] ECG   ← **Transferred directly** from pretrained
```

**Key Point**: PPG and ECG channels get their weights from the pretrained model, while ACC channels are newly initialized and learned during fine-tuning.

---

## Staged Unfreezing Strategy

The fine-tuning uses **3 optional stages** to prevent catastrophic forgetting:

### Stage 1: Head-Only Training (Epochs 1-5)
- **Frozen**: Entire encoder (all transformer blocks)
- **Trainable**: Classification head only
- **Learning Rate**: `--lr` (e.g., 2e-5)
- **Purpose**: Adapt the head to the new task without disrupting pretrained features

```
┌─────────────┐
│  Encoder    │  ❄️  FROZEN
│ (Transformer│
│   Blocks)   │
└─────────────┘
       ↓
┌─────────────┐
│    Head     │  🔥 TRAINABLE
│ (Classifier)│
└─────────────┘
```

### Stage 2: Partial Unfreezing (Epochs 6-30)
- **Frozen**: Early encoder blocks
- **Trainable**: Last N transformer blocks + classification head
- **Learning Rate**: Same as Stage 1 (`--lr`)
- **Purpose**: Fine-tune top layers for task-specific features

```
┌─────────────┐
│  Block 1-10 │  ❄️  FROZEN
│  (Encoder)  │
└─────────────┘
       ↓
┌─────────────┐
│  Block 11-12│  🔥 TRAINABLE
│ (Last N)    │
└─────────────┘
       ↓
┌─────────────┐
│    Head     │  🔥 TRAINABLE
└─────────────┘
```

### Stage 3: Full Fine-Tuning (Optional)
- **Frozen**: Nothing
- **Trainable**: All parameters
- **Learning Rate**: `--lr / 10` (e.g., 2e-6)
- **Purpose**: Full adaptation with very low LR to avoid forgetting

Enabled with: `--full-finetune --full-finetune-epochs 10`

```
┌─────────────┐
│  All Blocks │  🔥 TRAINABLE
│  (Encoder)  │
└─────────────┘
       ↓
┌─────────────┐
│    Head     │  🔥 TRAINABLE
└─────────────┘
```

---

## Command Line Arguments

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--pretrained` | Path to SSL checkpoint | `artifacts/foundation_model/best_model.pt` |
| `--data-dir` | Directory with BUT-PPG data | `data/but_ppg` |

### Channel Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--pretrain-channels` | 2 | Channels in pretrained model (PPG + ECG) |
| `--finetune-channels` | 5 | Channels for fine-tuning (ACC + PPG + ECG) |

### Training Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 30 | Total training epochs |
| `--head-only-epochs` | 5 | Stage 1 duration |
| `--unfreeze-last-n` | 2 | Blocks to unfreeze in Stage 2 |
| `--full-finetune` | False | Enable Stage 3 |
| `--full-finetune-epochs` | 10 | Stage 3 duration |

### Optimization

| Argument | Default | Description |
|----------|---------|-------------|
| `--lr` | 2e-5 | Learning rate |
| `--weight-decay` | 0.01 | Weight decay for AdamW |
| `--batch-size` | 32 | Batch size |
| `--gradient-clip` | 1.0 | Gradient clipping value |

### Device & Performance

| Argument | Default | Description |
|----------|---------|-------------|
| `--device` | auto | `cuda` or `cpu` |
| `--num-workers` | 4 | Data loading workers |
| `--no-amp` | False | Disable automatic mixed precision |

### Output

| Argument | Default | Description |
|----------|---------|-------------|
| `--output-dir` | `artifacts/but_ppg_finetuned` | Checkpoint directory |

---

## Data Format

The script expects `.npz` files in `--data-dir`:

### Required Files:
```
data/but_ppg/
├── train.npz          # Training data (required)
├── val.npz            # Validation data (optional, uses test if missing)
└── test.npz           # Test data (optional)
```

### .npz File Structure:
```python
{
    'signals': np.ndarray,  # Shape: [N, 5, 1250]
                            # N = number of samples
                            # 5 = channels (ACC_X, ACC_Y, ACC_Z, PPG, ECG)
                            # 1250 = timesteps (10s @ 125Hz)
    
    'labels': np.ndarray    # Shape: [N]
                            # 0 = poor quality
                            # 1 = good quality
}
```

---

## Output Files

After training, `--output-dir` contains:

```
artifacts/but_ppg_finetuned/
├── best_model.pt              # Best checkpoint (highest val accuracy)
├── final_model.pt             # Final checkpoint (last epoch)
├── training_config.json       # All hyperparameters
├── training_history.json      # Loss/accuracy curves per epoch
└── test_metrics.json          # Final test set evaluation
```

### Checkpoint Contents

```python
checkpoint = {
    'epoch': 29,
    'model_state_dict': {...},      # Full model weights
    'optimizer_state_dict': {...},
    'metrics': {
        'loss': 0.234,
        'accuracy': 87.5,
        'class_0_acc': 85.0,
        'class_1_acc': 90.0
    },
    'config': {...}                 # Training config
}
```

---

## Usage Examples

### Quick Test (1 epoch, mock data)

```bash
# Create mock data
python scripts/create_mock_butppg_data.py \
    --output data/but_ppg_test \
    --samples 30

# Quick fine-tune test
python scripts/finetune_butppg.py \
    --pretrained artifacts/foundation_model/best_model.pt \
    --data-dir data/but_ppg_test \
    --pretrain-channels 2 \
    --finetune-channels 5 \
    --epochs 1 \
    --lr 2e-5 \
    --batch-size 4 \
    --output-dir artifacts/butppg_test
```

### Full Training (30 epochs, 2-stage)

```bash
python scripts/finetune_butppg.py \
    --pretrained artifacts/foundation_model/best_model.pt \
    --data-dir data/but_ppg \
    --pretrain-channels 2 \
    --finetune-channels 5 \
    --epochs 30 \
    --head-only-epochs 5 \
    --unfreeze-last-n 2 \
    --lr 2e-5 \
    --batch-size 32 \
    --output-dir artifacts/but_ppg_finetuned
```

### Full Training with Stage 3 (40 epochs, 3-stage)

```bash
python scripts/finetune_butppg.py \
    --pretrained artifacts/foundation_model/best_model.pt \
    --data-dir data/but_ppg \
    --pretrain-channels 2 \
    --finetune-channels 5 \
    --epochs 30 \
    --head-only-epochs 5 \
    --unfreeze-last-n 2 \
    --full-finetune \
    --full-finetune-epochs 10 \
    --lr 2e-5 \
    --output-dir artifacts/but_ppg_finetuned_full
```

### CPU Training (smaller batch size)

```bash
python scripts/finetune_butppg.py \
    --pretrained artifacts/foundation_model/best_model.pt \
    --data-dir data/but_ppg \
    --device cpu \
    --batch-size 16 \
    --epochs 10 \
    --no-amp \
    --output-dir artifacts/butppg_cpu
```

---

## Monitoring Training

### Training Output

```
Epoch 5/30 (12.3s)
  Train - Loss: 0.345, Acc: 85.20%
  Val   - Loss: 0.312, Acc: 87.50%
  ✓ Best model saved (val_acc: 87.50%)
```

### Training History

```json
{
  "train_loss": [0.543, 0.412, 0.345, ...],
  "train_acc": [75.2, 81.3, 85.2, ...],
  "val_loss": [0.489, 0.367, 0.312, ...],
  "val_acc": [78.5, 83.1, 87.5, ...],
  "stage": ["stage1_head_only", "stage1_head_only", ...]
}
```

### Test Metrics

```json
{
  "loss": 0.298,
  "accuracy": 88.75,
  "class_0_acc": 86.5,
  "class_1_acc": 91.0
}
```

---

## Expected Performance

### Training Time
- **FastTrack (100 samples)**: ~5 minutes (30 epochs, GPU)
- **Full Dataset (10k samples)**: ~2 hours (30 epochs, GPU)

### Memory Usage
- **Batch size 32**: ~6GB GPU memory
- **Batch size 16**: ~3GB GPU memory
- **CPU mode**: ~4GB RAM

### Convergence
- **Stage 1 (head-only)**: Val accuracy ~70-75% by epoch 5
- **Stage 2 (partial unfreeze)**: Val accuracy ~80-90% by epoch 30
- **Stage 3 (full finetune)**: Val accuracy ~85-92% by epoch 40

---

## Troubleshooting

### "Pretrained checkpoint not found"
→ Run SSL pretraining first: `python scripts/pretrain_vitaldb_ssl.py ...`

### "Training data not found"
→ Create mock data: `python scripts/create_mock_butppg_data.py ...`

### Out of Memory (OOM)
```bash
# Reduce batch size
--batch-size 16

# Disable AMP
--no-amp

# Use CPU
--device cpu
```

### Poor Performance (< 70% accuracy)
- Check data quality and class balance
- Increase `--head-only-epochs` (try 10)
- Increase `--unfreeze-last-n` (try 4)
- Try Stage 3 full fine-tuning

### Model Not Improving After Stage 1
- This is expected! Stage 2 should improve performance
- If still no improvement, try:
  - Higher learning rate: `--lr 5e-5`
  - Unfreeze more blocks: `--unfreeze-last-n 4`
  - Enable Stage 3: `--full-finetune`

---

## Complete Pipeline

### End-to-End Example (from scratch)

```bash
# 1. Prepare VitalDB data
python scripts/ttm_vitaldb.py prepare-splits --mode fasttrack --output data
python scripts/ttm_vitaldb.py build-windows --split train --outdir data/vitaldb_windows
python scripts/ttm_vitaldb.py build-windows --split val --outdir data/vitaldb_windows

# 2. SSL Pretraining (2 channels: PPG + ECG)
python scripts/pretrain_vitaldb_ssl.py \
    --config configs/ssl_pretrain.yaml \
    --data-dir data/vitaldb_windows \
    --channels PPG ECG \
    --output-dir artifacts/foundation_model \
    --mask-ratio 0.4 \
    --epochs 100 \
    --batch-size 128

# 3. Create or prepare BUT-PPG data (5 channels)
python scripts/create_mock_butppg_data.py \
    --output data/but_ppg \
    --samples 1000

# 4. Fine-tune on BUT-PPG (channel inflation: 2→5)
python scripts/finetune_butppg.py \
    --pretrained artifacts/foundation_model/best_model.pt \
    --data-dir data/but_ppg \
    --pretrain-channels 2 \
    --finetune-channels 5 \
    --unfreeze-last-n 2 \
    --epochs 30 \
    --lr 2e-5 \
    --output-dir artifacts/but_ppg_finetuned

# 5. Evaluate
python scripts/evaluate_task.py \
    --checkpoint artifacts/but_ppg_finetuned/best_model.pt \
    --data-dir data/but_ppg \
    --split test
```

---

## References

- **Transfer Learning**: [doi.org/10.1145/3459637](https://doi.org/10.1145/3459637)
- **Channel Inflation**: Carreira & Zisserman (I3D), 2017
- **Progressive Unfreezing**: Howard & Ruder (ULMFiT), 2018
- **MAE**: He et al., Masked Autoencoders Are Scalable Vision Learners, 2022

---

## See Also

- [SSL Pretraining Guide](../docs/ssl_pretraining_guide.md)
- [Channel Inflation Guide](../docs/channel_inflation_guide.md)
- [TTM VitalDB Usage](scripts/TTM_VITALDB_USAGE.md)
