# 🚀 SSL Pre-training Quick Start Guide

## TL;DR - Run This:
```bash
# 1. Prepare VitalDB data (10 minutes)
python scripts/prepare_all_data.py --dataset vitaldb --mode fasttrack

# 2. Run SSL pre-training (30 minutes on GPU)
python scripts/pretrain_vitaldb_ssl.py --mode fasttrack --epochs 10

# Done! Foundation model at: artifacts/foundation_model/best_model.pt
```

---

## 📁 File Locations

### **Created/Modified Files:**
```
scripts/
├── pretrain_vitaldb_ssl.py  ← NEW! Main SSL pre-training script
├── prepare_all_data.py       ← EXISTING (creates preprocessed data)
└── ttm_vitaldb.py             ← EXISTING (supervised training)

configs/
├── ssl_pretrain.yaml          ← FIXED (LR: 5e-4 → 1e-4)
├── windows.yaml               ← UPDATED (added supervised overlap)
└── channels.yaml              ← OK (no changes needed)
```

### **Expected Data Structure:**
```
data/processed/vitaldb/windows/
├── train/
│   ├── train_windows.npz      ← [N, T, C] or [N, C, T]
│   └── train_stats.npz        ← Normalization statistics
├── val/
│   └── val_windows.npz        ← (optional)
└── test/
    └── test_windows.npz       ← (optional, used as val if no val)
```

---

## ⚙️ Command-Line Options

### **Basic Usage:**
```bash
python scripts/pretrain_vitaldb_ssl.py
```

### **Custom Configuration:**
```bash
# Specify epochs
python scripts/pretrain_vitaldb_ssl.py --epochs 50

# Specify batch size
python scripts/pretrain_vitaldb_ssl.py --batch-size 64

# Specify learning rate
python scripts/pretrain_vitaldb_ssl.py --lr 1e-4

# Custom output directory
python scripts/pretrain_vitaldb_ssl.py --output my_foundation

# Resume from checkpoint
python scripts/pretrain_vitaldb_ssl.py --resume artifacts/foundation_model/checkpoint_epoch_50.pt

# Force CPU (testing)
python scripts/pretrain_vitaldb_ssl.py --device cpu

# Combine options
python scripts/pretrain_vitaldb_ssl.py --epochs 100 --batch-size 128 --lr 1e-4
```

---

## 🎯 What Gets Trained

### **SSL Training Loop:**
```
Input: VitalDB windows [N, 2, 1250]
       ↓
Apply 40% masking on 10 patches
       ↓
TTM Encoder [B,2,1250] → [B,10,192]
       ↓
Decoder [B,10,192] → [B,2,1250]
       ↓
Loss = MSM + 0.3×STFT
       ↓
Update encoder + decoder weights
```

### **Loss Components:**
- **MSM (Masked Signal Modeling):** Reconstruct masked patches
  - Weight: 1.0 (70-75% of total loss)
  - Metric: MSE on masked regions only

- **STFT (Multi-Resolution Spectral):** Preserve frequencies
  - Weight: 0.3 (25-30% of total loss)
  - Windows: [512, 1024, 2048] samples
  - Metric: L1 on log-magnitude spectrograms

---

## 📊 Monitoring Training

### **Good Training Signs ✅:**
```
Epoch   Train Loss   Val Loss   MSM      STFT
1       0.0856       0.0892     0.0621   0.0235   ✅
10      0.0412       0.0438     0.0298   0.0114   ✅
50      0.0187       0.0201     0.0135   0.0052   ✅
100     0.0103       0.0115     0.0074   0.0029   ✅

✅ Loss decreases smoothly
✅ Val loss tracks train loss (gap < 15%)
✅ MSM ~70%, STFT ~30% of total loss
```

### **Warning Signs 🚨:**
```
Epoch   Train Loss   Val Loss   Issue
15      0.0891       -          🚨 Loss spike (LR too high)
50      0.0187       0.0325     🚨 Val diverging (overfitting)
40-60   0.0231±0.003 -          🚨 Plateau (LR too low)
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Data file not found" | Run `prepare_all_data.py` first |
| "CUDA out of memory" | Use `--batch-size 64` |
| "Loss is NaN" | Check data quality, reduce LR |
| "Training too slow" | Check GPU usage, reduce `--num-workers` |
| "No improvement" | Train longer or increase LR |

---

## 📈 Expected Timeline

### **FastTrack Mode:**
- Data prep: ~10 minutes (70 cases, 60s each)
- Training: ~30 minutes (10 epochs, GPU)
- Total: **~40 minutes**

### **Full Mode:**
- Data prep: ~1-2 hours (all cases, 300s each)
- Training: ~12-24 hours (100 epochs, GPU)
- Total: **~13-26 hours**

---

## 🎯 Output Files

After training, you'll have:
```
artifacts/foundation_model/
├── best_model.pt              ← Best checkpoint (use this!)
├── last_model.pt              ← Final checkpoint
├── checkpoint_epoch_10.pt     ← Periodic checkpoints
├── checkpoint_epoch_20.pt
├── ...
└── training_history.json      ← Loss curves, metrics
```

### **Load Foundation Model:**
```python
import torch
checkpoint = torch.load('artifacts/foundation_model/best_model.pt')

# Access encoder
encoder_state = checkpoint['encoder_state_dict']

# Access training info
best_epoch = checkpoint['epoch']
best_loss = checkpoint['best_val_loss']
```

---

## 🔄 Complete Workflow

### **Phase 1: Data Preparation**
```bash
python scripts/prepare_all_data.py \
  --dataset vitaldb \
  --mode fasttrack
```
**Output:** Preprocessed windows in `data/processed/vitaldb/windows/`

### **Phase 2: SSL Pre-training** ← YOU ARE HERE
```bash
python scripts/pretrain_vitaldb_ssl.py \
  --mode fasttrack \
  --epochs 10
```
**Output:** Foundation model in `artifacts/foundation_model/`

### **Phase 3: Fine-tuning** (Next step)
```bash
python scripts/finetune_butppg.py \
  --checkpoint artifacts/foundation_model/best_model.pt \
  --task signal_quality
```
**Output:** Task-specific model

---

## 🎓 Key Configuration

### **`configs/ssl_pretrain.yaml`**
```yaml
ssl:
  mask_ratio: 0.4           # 40% masking ✅
  patch_size: 125           # 1-second patches ✅
  
  stft:
    enabled: true
    n_ffts: [512, 1024, 2048]  # Multi-resolution ✅
    loss_weight: 0.3              # STFT weight ✅

model:
  input_channels: 2         # PPG + ECG ✅
  context_length: 1250      # 10s @ 125Hz ✅
  patch_size: 125           # Matches SSL ✅
  d_model: 192              # TTM hidden dim ✅

training:
  lr: 1e-4                  # FIXED ✅
  batch_size: 128           # Good for GPU ✅
  epochs: 100               # Full training ✅
  gradient_clip: 1.0        # Stability ✅
  amp: true                 # Mixed precision ✅
```

---

## 💡 Tips & Best Practices

### **1. Start with FastTrack**
```bash
# Test the full pipeline quickly
python scripts/pretrain_vitaldb_ssl.py --mode fasttrack --epochs 5
```

### **2. Monitor GPU Usage**
```bash
# In another terminal
watch -n 1 nvidia-smi
```

### **3. Save Log Files**
```bash
# Redirect output to file
python scripts/pretrain_vitaldb_ssl.py 2>&1 | tee ssl_training.log
```

### **4. Use TensorBoard** (if configured)
```bash
tensorboard --logdir artifacts/foundation_model/logs
```

### **5. Verify Data First**
```python
import numpy as np
data = np.load('data/processed/vitaldb/windows/train/train_windows.npz')
print(f"Shape: {data['data'].shape}")
print(f"Mean: {data['data'].mean():.4f}")
print(f"Std: {data['data'].std():.4f}")
```

---

## ✅ Pre-flight Checklist

Before running SSL pre-training:

- [ ] Data prepared: `data/processed/vitaldb/windows/train/train_windows.npz` exists
- [ ] GPU available: `nvidia-smi` shows GPU
- [ ] Config correct: `configs/ssl_pretrain.yaml` has `lr: 1e-4`
- [ ] Output dir: `artifacts/foundation_model/` will be created
- [ ] Enough disk space: ~500 MB for checkpoints

---

## 🎉 Success Indicators

After training completes, you should see:
```
✅ SSL PRE-TRAINING COMPLETED SUCCESSFULLY!
✅ Foundation model saved to: artifacts/foundation_model
✅ Best checkpoint: artifacts/foundation_model/best_model.pt
✅ Training history: artifacts/foundation_model/training_history.json

Next steps:
  1. Fine-tune on BUT-PPG for signal quality task
  2. Use foundation model for downstream tasks
  3. Evaluate on held-out test set
```

---

## 🚀 You're Ready!

Everything is set up. Just run:
```bash
python scripts/pretrain_vitaldb_ssl.py
```

Good luck! 🎯
