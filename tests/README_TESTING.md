# Research Pipeline Testing Guide

## Overview

This directory contains phase-by-phase tests for your TTM biosignal foundation model research pipeline.

```
Phase 0: Environment Setup           (1 min)
    ↓
Phase 1: Data Preparation           (1 min)
    ↓
Phase 2: SSL Pretraining Smoke      (5 min)
    ↓
Phase 3: Fine-tuning Smoke          (2 min)
```

**Total test time: ~10 minutes**

---

## Quick Start

### Option 1: Run All Tests (Recommended)

```bash
# Run complete pipeline test
python tests/run_all_phases.py

# Quick mode (smaller datasets, faster)
python tests/run_all_phases.py --quick

# Skip specific phases
python tests/run_all_phases.py --skip-phase 0 --skip-phase 1
```

### Option 2: Run Individual Phases

```bash
# Phase 0: Check environment
python tests/test_phase0_setup.py

# Phase 1: Verify data is ready
python tests/test_phase1_data_prep.py

# Phase 2: Test SSL pretraining loop (5 min)
python tests/test_phase2_ssl_smoke.py

# Phase 3: Test fine-tuning with channel inflation (2 min)
python tests/test_phase3_finetune_smoke.py
```

---

## Phase Details

### Phase 0: Environment Setup

**Purpose:** Verify all dependencies and project structure

**What it checks:**
- ✓ PyTorch installed and working
- ✓ IBM TTM (`tsfm_public`) accessible
- ✓ Biosignal libraries (NeuroKit2, WFDB, VitalDB)
- ✓ Project directories exist
- ✓ Critical code files present

**Run:**
```bash
python tests/test_phase0_setup.py
```

**Expected output:**
```
✓ PASS | PyTorch
✓ PASS | IBM TTM
✓ PASS | Biosignal libs
✓ PASS | Project structure
✓ PASS | Critical files

Result: 5/5 checks passed
🎉 Environment is ready!
```

**If it fails:** Install missing dependencies
```bash
pip install torch numpy scipy
pip install neurokit2 wfdb vitaldb
pip install git+https://github.com/ibm-granite/granite-tsfm.git
```

---

### Phase 1: Data Preparation

**Purpose:** Verify VitalDB windows are preprocessed correctly

**What it checks:**
- ✓ Train/val/test `.npz` files exist
- ✓ Shape: [N, 2, 1250] (2 channels, 10s @ 125Hz)
- ✓ No NaNs or Infs
- ✓ Data is normalized (mean≈0, std≈1)
- ✓ PyTorch DataLoader works
- ✓ Window count (target: 400K-500K for train)

**Run:**
```bash
python tests/test_phase1_data_prep.py

# Or specify custom data directory
python tests/test_phase1_data_prep.py --data-dir data/vitaldb_windows
```

**Expected output:**
```
✓ train | 450,000 windows | VALID
✓ val   | 50,000 windows  | VALID
✓ test  | 50,000 windows  | VALID

Window count analysis:
  Train: 450,000 windows
  ✓ Excellent! Article recommends ~500K windows

🎉 Data is ready for SSL pretraining!
```

**If data is missing:** Run data preparation
```bash
# Step 1: Create splits
python scripts/ttm_vitaldb.py prepare-splits \
    --mode fasttrack \
    --output configs/splits

# Step 2: Build windows
python scripts/build_windows_quiet.py train
python scripts/build_windows_quiet.py val
python scripts/build_windows_quiet.py test
```

---

### Phase 2: SSL Pretraining Smoke Test

**Purpose:** Test SSL training loop on real data (5 minutes)

**What it checks:**
- ✓ Real VitalDB data loads correctly
- ✓ IBM TTM weights load
- ✓ SSL masking works (40% mask ratio)
- ✓ Encoder output shape correct: [B, 10, 192]
- ✓ Decoder reconstructs: [B, 2, 1250]
- ✓ MSM + STFT losses compute correctly
- ✓ Training completes 1 epoch without errors
- ✓ Loss is finite (no NaN/Inf)
- ✓ Checkpoint saves successfully

**Run:**
```bash
# Default (64 windows, ~5 min on CPU)
python tests/test_phase2_ssl_smoke.py

# Custom options
python tests/test_phase2_ssl_smoke.py \
    --data-dir data/vitaldb_windows \
    --max-windows 64 \
    --batch-size 8 \
    --mask-ratio 0.4
```

**Expected output:**
```
[1/5] LOADING DATA
  Using: 64 windows
  Train: 51 windows
  Val:   13 windows

[2/5] BUILDING MODEL
  ✓ Model loaded (IBM TTM)

[3/5] SETUP TRAINING
  ✓ Optimizer ready
  ✓ Loss functions ready

[4/5] SHAPE VALIDATION
  Input: torch.Size([8, 2, 1250])
  Latents: torch.Size([8, 10, 192])
  Reconstructed: torch.Size([8, 2, 1250])
  ✓ All shapes correct

[5/5] TRAINING 1 EPOCH
  Epoch completed in 45.2s

  Train:
    Loss:      0.4523
    MSM Loss:  0.3821
    STFT Loss: 0.0702

  Validation:
    Loss:      0.4658
    MSM Loss:  0.3956
    STFT Loss: 0.0702

✓ Checkpoint saved

Result: 6/6 checks passed
Runtime: 45.2s
🎉 SSL pretraining pipeline works!
```

**What this means:**
- ✅ Your SSL implementation is correct
- ✅ IBM TTM integrates properly
- ✅ Ready for full 100-epoch pretraining

---

### Phase 3: Fine-tuning Smoke Test

**Purpose:** Test channel inflation (2→5) and fine-tuning

**What it checks:**
- ✓ Mock 5-channel BUT-PPG data created
- ✓ 2-channel checkpoint loads (if available)
- ✓ Channel inflation works: 2 → 5 channels
- ✓ Classification head attached
- ✓ Training completes 1 epoch
- ✓ Accuracy > random (50%)
- ✓ Checkpoint saves

**Run:**
```bash
# Without pretrained (random init)
python tests/test_phase3_finetune_smoke.py

# With pretrained checkpoint from Phase 2
python tests/test_phase3_finetune_smoke.py \
    --pretrained artifacts/smoke_ssl/checkpoint.pt

# Custom options
python tests/test_phase3_finetune_smoke.py \
    --n-samples 128 \
    --batch-size 16 \
    --lr 2e-5
```

**Expected output:**
```
[1/5] CREATING MOCK BUT-PPG DATA
  Generating 5-channel signals (ACC_X, ACC_Y, ACC_Z, PPG, ECG)...
  Signals shape: torch.Size([128, 5, 1250])
  Train: 102 samples
  Val:   26 samples

[2/5] BUILDING MODEL WITH CHANNEL INFLATION
  Loading pretrained checkpoint: artifacts/smoke_ssl/checkpoint.pt
  Inflating channels: 2 → 5
  ✓ Loaded pretrained encoder weights
  ✓ Channel inflation successful

[3/5] SETUP TRAINING
  ✓ Optimizer ready

[4/5] SHAPE VALIDATION
  Input: torch.Size([16, 5, 1250]) (5 channels)
  Output: torch.Size([16, 2]) (2 classes)
  ✓ All shapes correct

[5/5] TRAINING 1 EPOCH
  Epoch completed in 12.3s

  Train:
    Loss:     0.6523
    Accuracy: 62.75%

  Validation:
    Loss:     0.6789
    Accuracy: 57.69%

✓ Checkpoint saved

Result: 7/7 checks passed
Runtime: 12.3s
🎉 Fine-tuning pipeline works!
```

**What this means:**
- ✅ Channel inflation works correctly
- ✅ Transfer learning pipeline ready
- ✅ Can use 2-ch pretrained → 5-ch fine-tuned

---

## Troubleshooting

### "Module not found" errors

```bash
# Make sure you're in the project root
cd /path/to/TinyFoundationModelForBioSignals

# Install all dependencies
pip install -r requirements.txt

# Install IBM TTM
pip install git+https://github.com/ibm-granite/granite-tsfm.git
```

### "Training data not found"

```bash
# Run data preparation first
python tests/test_phase1_data_prep.py

# If files are missing, build windows:
python scripts/build_windows_quiet.py train
```

### "Out of memory" errors

```bash
# Reduce batch size or max windows
python tests/test_phase2_ssl_smoke.py \
    --max-windows 32 \
    --batch-size 4
```

### "Loss is NaN"

This usually means:
1. Learning rate too high → Try `--lr 1e-5`
2. Data not normalized → Check Phase 1 data validation
3. Gradient explosion → Check gradient clipping is enabled

---

## After All Tests Pass

### 1. Run Full SSL Pretraining (VitalDB)

```bash
python scripts/pretrain_vitaldb_ssl.py \
    --config configs/ssl_pretrain.yaml \
    --data-dir data/vitaldb_windows \
    --output-dir artifacts/foundation_model \
    --epochs 100 \
    --batch-size 128
```

**Expected time:** 12-24 hours on V100 GPU (depends on data size)

### 2. Fine-tune on BUT-PPG

```bash
python scripts/finetune_butppg.py \
    --pretrained artifacts/foundation_model/best_model.pt \
    --data-dir data/but_ppg \
    --pretrain-channels 2 \
    --finetune-channels 5 \
    --epochs 50 \
    --lr 2e-5 \
    --output-dir artifacts/but_ppg_finetuned
```

**Expected time:** 1-3 hours on V100 GPU

### 3. Evaluate Results

```bash
python scripts/evaluate_transfer_learning.py \
    --checkpoint artifacts/but_ppg_finetuned/best_model.pt \
    --data-dir data/but_ppg \
    --output-dir artifacts/evaluation
```

---

## Test File Reference

| File | Purpose | Runtime | Real Data? |
|------|---------|---------|------------|
| `test_phase0_setup.py` | Environment check | 1 min | No |
| `test_phase1_data_prep.py` | Data validation | 1 min | Yes |
| `test_phase2_ssl_smoke.py` | SSL training test | 5 min | Yes |
| `test_phase3_finetune_smoke.py` | Fine-tuning test | 2 min | No (mock) |
| `run_all_phases.py` | Run all tests | 10 min | Mixed |

---

## FAQ

**Q: Do I need GPU for smoke tests?**  
A: No, all smoke tests work on CPU. They'll be slower (5-10 min) but functional.

**Q: How much data do I need?**  
A: Minimum 10K windows for testing. Article recommends 400K-500K for full training.

**Q: Can I skip phases?**  
A: Yes! Use `--skip-phase` with `run_all_phases.py`

**Q: Why mock data in Phase 3?**  
A: BUT-PPG requires manual download. Phase 3 tests the code with synthetic data. Use real data for actual fine-tuning.

**Q: What if Phase 2 is slow?**  
A: Use `--max-windows 32` to test with less data. Smoke test just validates the pipeline works.

---

## Contact

If tests fail unexpectedly, check:
1. All dependencies installed (`pip list`)
2. Data files exist (`ls data/vitaldb_windows/`)
3. Enough disk space (logs, checkpoints)
4. Python version ≥3.8

For research questions, refer to the article specifications in the main documentation.

---

**Last updated:** October 2025
