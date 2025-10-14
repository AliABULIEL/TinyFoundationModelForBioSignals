# TTM Biosignal Research Pipeline - Quick Reference

## 🚀 Complete Workflow

### 1. Run All Tests (~10 minutes)
```bash
python tests/run_all_phases.py
```

### 2. If tests pass → Run Full Pipeline

```bash
# Step 1: SSL Pretraining on VitalDB (12-24 hours)
python scripts/pretrain_vitaldb_ssl.py \
    --config configs/ssl_pretrain.yaml \
    --data-dir data/vitaldb_windows \
    --epochs 100

# Step 2: Fine-tune on BUT-PPG (1-3 hours)
python scripts/finetune_butppg.py \
    --pretrained artifacts/foundation_model/best_model.pt \
    --data-dir data/but_ppg \
    --pretrain-channels 2 \
    --finetune-channels 5 \
    --epochs 50

# Step 3: Evaluate
python scripts/evaluate_transfer_learning.py \
    --checkpoint artifacts/but_ppg_finetuned/best_model.pt \
    --data-dir data/but_ppg
```

---

## 📊 Individual Test Commands

```bash
# Phase 0: Environment (1 min)
python tests/test_phase0_setup.py

# Phase 1: Data Validation (1 min)
python tests/test_phase1_data_prep.py

# Phase 2: SSL Smoke Test (5 min)
python tests/test_phase2_ssl_smoke.py

# Phase 3: Fine-tuning Smoke Test (2 min)
python tests/test_phase3_finetune_smoke.py
```

---

## 🔧 Data Preparation (if needed)

```bash
# Create splits
python scripts/ttm_vitaldb.py prepare-splits \
    --mode fasttrack \
    --output configs/splits

# Build windows
python scripts/build_windows_quiet.py train
python scripts/build_windows_quiet.py val
python scripts/build_windows_quiet.py test

# Verify
python tests/test_phase1_data_prep.py
```

---

## ✅ Success Criteria

### Tests Should Show:
- ✓ All 5/5 environment checks pass
- ✓ VitalDB: 400K-500K train windows (2 channels, 1250 timesteps)
- ✓ SSL: Loss decreases, no NaN/Inf
- ✓ Fine-tune: Channel inflation 2→5 works, accuracy >50%

### Training Should Show:
- **SSL (100 epochs):** Val loss plateaus around 0.3-0.5
- **Fine-tune (50 epochs):** Val accuracy reaches target (≥88% for quality)

---

## 📁 Key Files

### Tests
```
tests/
├── run_all_phases.py              # Run everything
├── test_phase0_setup.py           # Check environment  
├── test_phase1_data_prep.py       # Validate data
├── test_phase2_ssl_smoke.py       # Test SSL (5 min)
├── test_phase3_finetune_smoke.py  # Test fine-tuning (2 min)
└── README_TESTING.md              # Full testing guide
```

### Scripts
```
scripts/
├── pretrain_vitaldb_ssl.py        # Full SSL pretraining
├── finetune_butppg.py             # Fine-tuning with channel inflation
├── build_windows_quiet.py         # Build VitalDB windows
└── ttm_vitaldb.py                 # VitalDB data pipeline
```

### Configs
```
configs/
├── ssl_pretrain.yaml              # SSL hyperparameters
├── windows.yaml                   # Window specs (10s, 125Hz)
└── channels.yaml                  # Channel definitions (2-ch / 5-ch)
```

---

## 🐛 Quick Fixes

### "Module not found"
```bash
pip install -r requirements.txt
pip install git+https://github.com/ibm-granite/granite-tsfm.git
```

### "Data not found"
```bash
python tests/test_phase1_data_prep.py  # Check status
python scripts/build_windows_quiet.py train  # Build if missing
```

### "Out of memory"
```bash
# Reduce batch size in configs/ssl_pretrain.yaml
batch_size: 64  # Instead of 128

# Or use gradient accumulation
python scripts/pretrain_vitaldb_ssl.py \
    --batch-size 64 \
    --accumulate-grad-batches 2
```

### "Loss is NaN"
```bash
# Reduce learning rate
python scripts/pretrain_vitaldb_ssl.py \
    --lr 1e-5  # Instead of 5e-4
```

---

## 📈 Expected Results (from Article)

| Task | Metric | Target | Baseline |
|------|--------|--------|----------|
| BUT-PPG Quality | AUROC | ≥0.88 | 0.74-0.76 |
| VitalDB Hypotension | AUROC | ≥0.91 | 0.88 |
| VitalDB MAP | MAE | ≤5.0 mmHg | 3.8 mmHg |

---

## ⏱️ Time Estimates

| Task | CPU | GPU (V100) |
|------|-----|-----------|
| All smoke tests | 10 min | 5 min |
| SSL pretraining (100 epochs) | N/A | 12-24 hrs |
| Fine-tuning (50 epochs) | N/A | 1-3 hrs |
| Evaluation | 5 min | 2 min |

---

## 💾 Disk Space

| Component | Size |
|-----------|------|
| VitalDB windows (train) | ~15-30 GB |
| SSL checkpoints | ~500 MB each |
| Fine-tuned models | ~200 MB each |
| Total (full pipeline) | ~20-40 GB |

---

## 🎯 Research Goals Checklist

- [ ] Phase 0: Environment ready
- [ ] Phase 1: Data prepared (~500K windows)
- [ ] Phase 2: SSL smoke test passes
- [ ] Phase 3: Fine-tuning smoke test passes
- [ ] SSL pretraining completes (100 epochs)
- [ ] Fine-tuning completes (50 epochs)
- [ ] Evaluation shows target performance
- [ ] Results documented

---

## 📚 Documentation

- **Full testing guide:** `tests/README_TESTING.md`
- **Article comparison:** Check artifact from previous analysis
- **Code documentation:** Docstrings in source files
- **Workflow guide:** `docs/WORKFLOW.md`

---

**Quick question? Run:** `python tests/run_all_phases.py --help`
