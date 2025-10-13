# SSL Pretraining Guide

## Overview

Self-supervised learning (SSL) pretraining on VitalDB biosignals using masked signal modeling.

## Quick Start

```bash
# Full training (100 epochs)
python scripts/pretrain_vitaldb_ssl.py \
    --config configs/ssl_pretrain.yaml \
    --data-dir data/vitaldb_windows \
    --output-dir artifacts/foundation_model \
    --epochs 100 \
    --batch-size 128

# Quick test (5 epochs, CPU, small data)
python scripts/pretrain_vitaldb_ssl.py \
    --config configs/ssl_pretrain.yaml \
    --data-dir data/vitaldb_windows \
    --output-dir artifacts/ssl_test \
    --epochs 5 \
    --batch-size 32 \
    --fast \
    --device cpu
```

## SSL Strategy

### Masked Signal Modeling (MSM)

1. **Input**: 2-channel biosignals (PPG + ECG), 10s windows (1250 samples @ 125Hz)
2. **Masking**: Randomly mask 40% of 1-second patches
3. **Encoding**: TTM encoder processes masked signal → latent representations
4. **Decoding**: Lightweight decoder reconstructs original signal
5. **Loss**: MSM (reconstruction on masked patches) + Multi-Resolution STFT

### Architecture

```
Input Signal [B, 2, 1250]
         ↓
    Masking (40%)
         ↓
Masked Signal [B, 2, 1250]  ← 40% patches zeroed
         ↓
  TTM Encoder
         ↓
 Latents [B, 10, 192]       ← 10 patches, 192-d
         ↓
Reconstruction Decoder
         ↓
Reconstructed [B, 2, 1250]
         ↓
    Compute Losses
    - MSM: (reconstructed - original) on masked patches
    - STFT: Multi-resolution spectral loss
```

## Command Line Arguments

### Required Arguments

- `--data-dir`: Directory containing VitalDB window files
  - Must contain `train_windows.npz` and `val_windows.npz`
  - Created by `scripts/ttm_vitaldb.py build-windows`

### Configuration

- `--config`: Path to SSL config YAML (default: `configs/ssl_pretrain.yaml`)
- `--output-dir`: Output directory for checkpoints (default: `artifacts/foundation_model`)

### Data Options

- `--channels`: Channels to use (default: `PPG ECG`)
- `--train-file`: Training data filename (default: `train_windows.npz`)
- `--val-file`: Validation data filename (default: `val_windows.npz`)

### Training Hyperparameters

- `--mask-ratio`: Masking ratio (default: 0.4)
- `--mask-type`: Masking strategy - `random` or `block` (default: `random`)
- `--epochs`: Number of epochs (default: 100)
- `--batch-size`: Batch size (default: 128)
- `--lr`: Learning rate (default: 5e-4)
- `--weight-decay`: Weight decay (default: 0.01)
- `--warmup-epochs`: Warmup epochs (default: 10)

### Model Configuration

- `--context-length`: Sequence length (default: 1250 for 10s @ 125Hz)
- `--patch-size`: Patch size in samples (default: 125 for 1s)

### Training Options

- `--device`: Device (`cuda` or `cpu`)
- `--num-workers`: Data loading workers (default: 4)
- `--no-amp`: Disable automatic mixed precision
- `--gradient-clip`: Gradient clipping value (default: 1.0)
- `--stft-weight`: Weight for STFT loss (default: 0.3)

### Logging and Checkpointing

- `--log-interval`: Log every N batches (default: 50)
- `--save-interval`: Save checkpoint every N epochs (default: 10)

### Testing

- `--fast`: Fast mode - use small data subset for testing
- `--seed`: Random seed (default: 42)

## Output Files

After training, the output directory contains:

```
artifacts/foundation_model/
├── best_model.pt              # Best checkpoint (lowest val loss)
├── last_model.pt              # Final checkpoint
├── checkpoint_epoch_10.pt     # Periodic checkpoints
├── checkpoint_epoch_20.pt
├── ...
├── training_history.json      # Loss curves
└── training_config.json       # Training configuration
```

### Checkpoint Contents

```python
checkpoint = {
    'epoch': 99,
    'encoder_state_dict': {...},      # TTM encoder weights
    'decoder_state_dict': {...},      # Reconstruction decoder
    'optimizer_state_dict': {...},
    'best_val_loss': 0.123,
    'metrics': {...},
    'config': {...}
}
```

## Training Pipeline

### 1. Data Preparation

First, create VitalDB windows:

```bash
# Prepare splits
python scripts/ttm_vitaldb.py prepare-splits \
    --mode fasttrack \
    --case-set bis \
    --output data

# Build windows
python scripts/ttm_vitaldb.py build-windows \
    --channels-yaml configs/channels.yaml \
    --windows-yaml configs/windows.yaml \
    --split-file data/splits_fasttrack.json \
    --split train \
    --outdir data/vitaldb_windows

python scripts/ttm_vitaldb.py build-windows \
    --split val \
    ...
```

### 2. SSL Pretraining

```bash
python scripts/pretrain_vitaldb_ssl.py \
    --config configs/ssl_pretrain.yaml \
    --data-dir data/vitaldb_windows \
    --output-dir artifacts/foundation_model \
    --epochs 100 \
    --batch-size 128
```

### 3. Fine-tuning (Next Step)

Use the pretrained encoder for downstream tasks:

```python
from src.models.channel_utils import load_pretrained_with_channel_inflate

# Load pretrained model and inflate channels
model = load_pretrained_with_channel_inflate(
    checkpoint_path='artifacts/foundation_model/best_model.pt',
    pretrain_channels=2,
    finetune_channels=5,
    model_config={...}
)

# Fine-tune on downstream task
train(model, downstream_data)
```

## Loss Functions

### Masked Signal Modeling (MSM)

- **Objective**: Reconstruct masked signal patches
- **Formula**: MSE on masked regions only
- **Weight**: 1.0 (baseline)

```python
loss_msm = MSE(reconstructed[masked], original[masked])
```

### Multi-Resolution STFT

- **Objective**: Preserve spectral characteristics
- **Resolutions**: FFT sizes [512, 1024, 2048]
- **Formula**: L1 loss on log-magnitude spectrograms
- **Weight**: 0.3 (configurable)

```python
loss_stft = L1(log(STFT(reconstructed)), log(STFT(original)))
```

### Combined Loss

```python
total_loss = loss_msm + 0.3 * loss_stft
```

## Training Schedule

### Learning Rate Schedule

1. **Warmup** (10 epochs): Linear increase from 0 to peak LR
2. **Cosine Decay** (90 epochs): Smooth decrease to ~0

```python
if epoch < warmup_epochs:
    lr = peak_lr * (epoch / warmup_epochs)
else:
    lr = 0.5 * peak_lr * (1 + cos(π * progress))
```

### Example Schedule (100 epochs, lr=5e-4)

```
Epochs  1-10:  0 → 5e-4     (warmup)
Epochs 11-50:  5e-4 → 2.5e-4 (cosine)
Epochs 51-90:  2.5e-4 → 5e-5 (cosine)
Epochs 91-100: 5e-5 → 0     (cosine)
```

## Monitoring Training

### Training Logs

```
Epoch 1/100 (45.2s)
  Train - Loss: 0.342, MSM: 0.287, STFT: 0.183
  Val   - Loss: 0.298, MSM: 0.251, STFT: 0.157
  ✓ Best model saved (val_loss: 0.298)

Epoch 2/100 (44.8s)
  Train - Loss: 0.276, MSM: 0.231, STFT: 0.150
  Val   - Loss: 0.262, MSM: 0.221, STFT: 0.137
  ✓ Best model saved (val_loss: 0.262)
```

### Interpreting Metrics

- **Train/Val Loss**: Lower is better
- **MSM Loss**: Reconstruction quality on masked patches
- **STFT Loss**: Spectral similarity
- **Val << Train**: Good generalization
- **Val >> Train**: Overfitting (increase regularization)

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
--batch-size 64

# Disable AMP (uses less memory)
--no-amp

# Use CPU
--device cpu
```

### Slow Training

```bash
# Enable AMP (2-3x speedup on GPU)
# (enabled by default on GPU)

# Increase workers
--num-workers 8

# Larger batch size (if memory allows)
--batch-size 256
```

### Poor Reconstruction

- Increase training epochs
- Adjust mask ratio (try 0.3 or 0.5)
- Tune STFT weight (try 0.5 or 1.0)
- Check data quality

### Data Not Found

```bash
# Verify data exists
ls -lh data/vitaldb_windows/

# Should contain:
# train_windows.npz
# val_windows.npz (optional)
```

If missing, run window building:
```bash
python scripts/ttm_vitaldb.py build-windows ...
```

## Advanced Usage

### Custom Masking

```bash
# Block masking (contiguous spans)
--mask-type block

# Higher mask ratio (harder task)
--mask-ratio 0.6

# Lower mask ratio (easier task)
--mask-ratio 0.2
```

### Resume Training

The script doesn't support automatic resume yet. To continue:

1. Load checkpoint manually in code
2. Create new trainer with loaded states
3. Continue from saved epoch

### Multi-GPU Training

Not currently supported. Future enhancement.

## Expected Performance

### Training Time

- **FastTrack (70 cases)**: ~3 hours (100 epochs, GPU)
- **Full VitalDB (6000+ cases)**: ~2-3 days (100 epochs, GPU)

### Memory Usage

- **Batch size 128**: ~8GB GPU memory
- **Batch size 64**: ~4GB GPU memory
- **CPU mode**: ~4GB RAM

### Convergence

- **Good**: Val loss < 0.15 by epoch 100
- **Excellent**: Val loss < 0.10 by epoch 100

## References

- **MAE**: He et al., 2022 - Masked Autoencoders Are Scalable Vision Learners
- **MR-STFT**: Yamamoto et al., 2020 - Parallel WaveGAN
- **bioFAME**: ICLR 2024 - Foundation models for biosignals
