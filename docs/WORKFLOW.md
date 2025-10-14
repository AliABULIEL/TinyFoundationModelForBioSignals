# TTM Biosignal Foundation Model Workflow

## Overview

This repository implements a **foundation model for biosignals** using TTM (Tiny Time Mixer) with self-supervised learning (SSL) pretraining on VitalDB data, followed by fine-tuning on downstream tasks like BUT-PPG quality classification.

**Pipeline**: VitalDB SSL Pretrain (2ch: PPG+ECG) → Save Foundation Checkpoint → Channel Inflation (2→5ch) → BUT-PPG Fine-tune

---

## End-to-End Workflow

### Phase 1: Data Preparation

Preprocess raw VitalDB data into windowed format:

```bash
# 1. Prepare splits
python scripts/ttm_vitaldb.py prepare-splits \
    --mode fasttrack \
    --case-set bis \
    --output data

# 2. Build training windows
python scripts/ttm_vitaldb.py build-windows \
    --channels-yaml configs/channels.yaml \
    --windows-yaml configs/windows.yaml \
    --split-file data/splits_fasttrack.json \
    --split train \
    --outdir data/vitaldb_windows

# 3. Build validation windows
python scripts/ttm_vitaldb.py build-windows \
    --channels-yaml configs/channels.yaml \
    --windows-yaml configs/windows.yaml \
    --split-file data/splits_fasttrack.json \
    --split val \
    --outdir data/vitaldb_windows
```

**Output**: `data/vitaldb_windows/{train,val}_windows.npz`

---

### Phase 2: SSL Pretraining (Foundation Model)

Train encoder on 2-channel biosignals (PPG + ECG) using masked signal modeling:

```bash
python scripts/pretrain_vitaldb_ssl.py \
    --config configs/ssl_pretrain.yaml \
    --data-dir data/vitaldb_windows \
    --channels PPG ECG \
    --output-dir artifacts/foundation_model \
    --mask-ratio 0.4 \
    --epochs 100 \
    --batch-size 128
```

**What Happens**:
- Input: [B, 2, 1250] (2 channels, 10s @ 125Hz)
- Mask 40% of patches randomly
- Encoder (TTM): [B, 2, 1250] → [B, 10, 192] (10 patches, 192-d latents)
- Decoder: [B, 10, 192] → [B, 2, 1250] (reconstruct original)
- Loss: MSM (masked signal modeling) + Multi-Resolution STFT

**Success Criteria**:
- Val loss < 0.15 after 100 epochs
- No NaNs in gradients/outputs
- Checkpoints saved successfully

**Output**: `artifacts/foundation_model/best_model.pt`

---

### Phase 3: Downstream Fine-tuning (BUT-PPG)

Fine-tune on 5-channel data (ACC_X, ACC_Y, ACC_Z, PPG, ECG) with channel inflation:

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

**What Happens**:
- Load 2-ch pretrained encoder
- Inflate to 5 channels (PPG/ECG transferred, ACC initialized)
- Stage 1 (5 epochs): Train head only, encoder frozen
- Stage 2 (25 epochs): Unfreeze last 2 blocks + head
- Binary classification: good vs poor PPG quality

**Success Criteria**:
- Val accuracy > 80% after 30 epochs
- Best checkpoint saved
- Test metrics computed

**Output**: `artifacts/but_ppg_finetuned/best_model.pt`

---

## Quick 5-Minute Smoke Test (CPU)

Test the SSL pipeline on a small subset of real data:

```bash
python scripts/smoke_realdata_5min.py \
    --data-dir data/vitaldb_windows \
    --max-windows 64
```

**What Happens**:
- Loads 64 real VitalDB windows (deterministic sampling)
- Trains TTM encoder for 1 epoch on CPU
- Validates shapes and checks for NaNs
- Saves checkpoint to `artifacts/smoke/best_model.pt`

**Success Criteria**:
- Runtime: ~5 minutes on CPU
- No NaNs/Infs detected
- Loss decreases during training
- Shapes: Input [8, 2, 1250] → Tokens [8, 10, ~192] → Output [8, 2, 1250]

**Important**: This requires **REAL preprocessed VitalDB data**. No synthetic data fallback is provided.

---

## Expected Artifacts

After running the complete pipeline:

```
artifacts/
├── foundation_model/
│   ├── best_model.pt              # SSL pretrained encoder (2ch)
│   ├── training_history.json      # Loss curves
│   └── training_config.json       # Hyperparameters
│
├── but_ppg_finetuned/
│   ├── best_model.pt              # Fine-tuned model (5ch)
│   ├── training_history.json
│   ├── training_config.json
│   └── test_metrics.json          # Final test results
│
└── smoke/
    ├── best_model.pt              # Quick smoke test checkpoint
    └── smoke_metrics.json         # Runtime and metrics
```

---

## Configuration Files

### `configs/ssl_pretrain.yaml`
SSL pretraining configuration:
- Mask ratio: 0.4
- STFT loss weight: 0.3
- Batch size: 128
- Learning rate: 5e-4
- Optimizer: AdamW

### `configs/windows.yaml`
Window generation parameters:
- Window length: 10.0 seconds
- Sampling rate: 125 Hz
- Timesteps: 1250
- Min cardiac cycles: 3

### `configs/channels.yaml`
Channel definitions for VitalDB:
- PPG: PLETH track
- ECG: ECG_II track
- Filtering: Chebyshev II (0.5-10 Hz)

---

## Model Architecture

### TTM Encoder
```
Input: [B, C=2, T=1250]
  ↓ Patchify (patch_size=125)
Patches: [B, C=2, P=10, patch_size=125]
  ↓ Linear projection
Tokens: [B, P=10, D=192]
  ↓ Transformer blocks (12 layers)
Latents: [B, P=10, D=192]
```

### Reconstruction Decoder (SSL)
```
Latents: [B, P=10, D=192]
  ↓ Linear layer
Output: [B, C=2, T=1250]
```

### Classification Head (Fine-tuning)
```
Latents: [B, P=10, D=192]
  ↓ Global average pooling
Pooled: [B, D=192]
  ↓ Linear classifier
Logits: [B, num_classes=2]
```

---

## Running on Google Colab (GPU)

### 1. Setup Environment

```python
# Install dependencies
!pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install tsfm_public  # IBM TTM
!pip install pyyaml tqdm numpy scipy

# Clone repository
!git clone https://github.com/your-repo/TinyFoundationModelForBioSignals.git
%cd TinyFoundationModelForBioSignals
```

### 2. Mount Google Drive (for data)

```python
from google.colab import drive
drive.mount('/content/drive')

# Point to your preprocessed VitalDB data in Drive
DATA_DIR = '/content/drive/MyDrive/vitaldb_windows'
```

### 3. Run Quick Smoke Test (GPU)

```python
# Modify smoke script to use GPU
!python scripts/smoke_realdata_5min.py \
    --data-dir {DATA_DIR} \
    --max-windows 64
```

### 4. Run Full SSL Pretraining

```python
!python scripts/pretrain_vitaldb_ssl.py \
    --config configs/ssl_pretrain.yaml \
    --data-dir {DATA_DIR} \
    --output-dir artifacts/foundation_model \
    --epochs 1 \
    --batch-size 32  # Adjust for GPU memory
```

**Note**: 
- TTM weights will download from HuggingFace on first use (~50MB)
- Expected GPU memory: ~6GB for batch_size=32
- Full pretraining (100 epochs): ~2-3 hours on GPU

---

## Troubleshooting

### "Training data not found"
**Problem**: `data/vitaldb_windows/train_windows.npz` doesn't exist

**Solution**: Run data preprocessing pipeline first:
```bash
python scripts/ttm_vitaldb.py prepare-splits --output data
python scripts/ttm_vitaldb.py build-windows --split train --outdir data/vitaldb_windows
```

### "TTM model download fails"
**Problem**: HuggingFace model not accessible

**Solution**: 
1. Check internet connection
2. Try: `huggingface-cli login` (if authentication required)
3. Verify model ID: `ibm-granite/granite-timeseries-ttm-r1`

### "Out of memory" during training
**Problem**: GPU/CPU runs out of memory

**Solution**:
- Reduce batch size: `--batch-size 16` or `--batch-size 8`
- For CPU: Disable AMP with `--no-amp`
- Reduce max-windows in smoke test: `--max-windows 32`

### "NaNs detected in gradients"
**Problem**: Numerical instability

**Solution**:
- Lower learning rate: `--lr 1e-4`
- Enable gradient clipping (already default: 1.0)
- Check data quality (remove corrupted windows)

### "Shape mismatch" errors
**Problem**: Incorrect dimensions

**Solution**: Verify configuration:
- Context length: 1250 (10s @ 125Hz)
- Patch size: 125 (1s patches)
- Input channels: 2 for SSL, 5 for BUT-PPG
- Check data shape: `[N, C, 1250]`

---

## Performance Expectations

### SSL Pretraining (VitalDB)
- **Training time**: 
  - FastTrack (70 cases): ~3 hours (100 epochs, GPU)
  - Full VitalDB (6000+ cases): ~2-3 days (100 epochs, GPU)
- **Val loss**: < 0.15 (good), < 0.10 (excellent)
- **Memory**: ~8GB GPU for batch_size=128

### Fine-tuning (BUT-PPG)
- **Training time**: ~30 minutes (30 epochs, GPU)
- **Accuracy**: 80-90% (val set)
- **Memory**: ~6GB GPU for batch_size=32

### Smoke Test (CPU)
- **Training time**: ~5 minutes (1 epoch, 64 windows)
- **Memory**: ~4GB RAM
- **Purpose**: Quick sanity check, not for benchmarking

---

## Next Steps

1. **Prepare Data**: Preprocess VitalDB into windowed format
2. **Run Smoke Test**: Verify pipeline works end-to-end
3. **SSL Pretrain**: Train foundation model on full VitalDB
4. **Fine-tune**: Adapt to BUT-PPG or other downstream tasks
5. **Evaluate**: Compare with baselines, analyze transfer learning benefits

---

## References

- **TTM**: IBM Granite Timeseries (HuggingFace)
- **MAE**: He et al., Masked Autoencoders Are Scalable Vision Learners (2022)
- **bioFAME**: Foundation models for biosignals (ICLR 2024)
- **VitalDB**: Open biosignal database (https://vitaldb.net/)
- **BUT-PPG**: PPG quality assessment dataset

---

## Support

For issues:
1. Check this workflow documentation
2. Verify data format and paths
3. Review error messages carefully
4. Ensure all dependencies are installed

**No synthetic data generators are provided**. All scripts require real preprocessed data.
