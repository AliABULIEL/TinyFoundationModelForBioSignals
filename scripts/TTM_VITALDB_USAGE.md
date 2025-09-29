# TTM VitalDB Pipeline - Usage Guide

## Fixed and Ready to Use!

The `ttm_vitaldb.py` script has been completely refactored with correct imports and proper implementation.

## Commands

### 1. Prepare Splits
Create train/val/test splits from VitalDB cases.

```bash
python scripts/ttm_vitaldb.py prepare-splits \
    --mode fasttrack \
    --case-set bis \
    --output data \
    --seed 42
```

**Options:**
- `--mode`: `fasttrack` (70 cases) or `full` (all cases)
- `--case-set`: VitalDB case set (bis, desflurane, sevoflurane, remifentanil, propofol, tiva)
- `--output`: Output directory for splits JSON
- `--seed`: Random seed for reproducibility

**Output:** `data/splits_fasttrack.json` or `data/splits_full.json`

---

### 2. Build Windows
Process VitalDB data into windowed format.

```bash
# Build train windows
python scripts/ttm_vitaldb.py build-windows \
    --channels-yaml configs/channels.yaml \
    --windows-yaml configs/windows.yaml \
    --split-file data/splits_fasttrack.json \
    --split train \
    --outdir data

# Build test windows  
python scripts/ttm_vitaldb.py build-windows \
    --channels-yaml configs/channels.yaml \
    --windows-yaml configs/windows.yaml \
    --split-file data/splits_fasttrack.json \
    --split test \
    --outdir data
```

**Options:**
- `--channels-yaml`: Channel configuration (signal types, filters)
- `--windows-yaml`: Window configuration (size, stride, min cycles)
- `--split-file`: Path to splits JSON from step 1
- `--split`: Which split to process (train/val/test)
- `--channel`: Specific channel to process (optional)
- `--duration-sec`: Seconds to load per case (default: 60)
- `--min-sqi`: Minimum signal quality (default: 0.5)
- `--outdir`: Output directory

**Output:** 
- `data/train_windows.npz`
- `data/test_windows.npz`
- `data/train_stats.npz` (normalization statistics)

---

### 3. Train Model
Train the TTM model on preprocessed windows.

```bash
python scripts/ttm_vitaldb.py train \
    --model-yaml configs/model.yaml \
    --run-yaml configs/run.yaml \
    --split-file data/splits_fasttrack.json \
    --outdir data \
    --out checkpoints \
    --fasttrack
```

**Options:**
- `--model-yaml`: Model configuration (architecture, task)
- `--run-yaml`: Training configuration (epochs, batch size, LR)
- `--split-file`: Path to splits JSON
- `--outdir`: Data directory (where windows are stored)
- `--out`: Output directory for checkpoints
- `--fasttrack`: Use FastTrack mode (frozen encoder)

**Output:**
- `checkpoints/model.pt` (best model checkpoint)
- `checkpoints/train_metrics.json` (training history)

---

### 4. Test Model
Evaluate trained model on test set.

```bash
python scripts/ttm_vitaldb.py test \
    --ckpt checkpoints/model.pt \
    --model-yaml configs/model.yaml \
    --run-yaml configs/run.yaml \
    --split-file data/splits_fasttrack.json \
    --outdir data \
    --out results
```

**Options:**
- `--ckpt`: Path to model checkpoint
- `--model-yaml`: Model configuration
- `--run-yaml`: Run configuration
- `--split-file`: Path to splits JSON
- `--outdir`: Data directory
- `--out`: Output directory for results

**Output:**
- `results/test_results.json` (metrics: accuracy, precision, recall, F1, AUC)

---

### 5. Inspect (Optional)
Inspect data files or model checkpoints.

```bash
# Inspect data
python scripts/ttm_vitaldb.py inspect --data data/train_windows.npz

# Inspect model
python scripts/ttm_vitaldb.py inspect --model checkpoints/model.pt

# Inspect both
python scripts/ttm_vitaldb.py inspect \
    --data data/train_windows.npz \
    --model checkpoints/model.pt
```

---

## Complete Pipeline Example

### FastTrack Mode (3 hours, 70 cases)

```bash
# 1. Prepare splits
python scripts/ttm_vitaldb.py prepare-splits \
    --mode fasttrack \
    --case-set bis \
    --output data

# 2. Build train windows
python scripts/ttm_vitaldb.py build-windows \
    --channels-yaml configs/channels.yaml \
    --windows-yaml configs/windows.yaml \
    --split-file data/splits_fasttrack.json \
    --split train \
    --outdir data

# 3. Build test windows
python scripts/ttm_vitaldb.py build-windows \
    --channels-yaml configs/channels.yaml \
    --windows-yaml configs/windows.yaml \
    --split-file data/splits_fasttrack.json \
    --split test \
    --outdir data

# 4. Train model
python scripts/ttm_vitaldb.py train \
    --model-yaml configs/model.yaml \
    --run-yaml configs/run.yaml \
    --split-file data/splits_fasttrack.json \
    --outdir data \
    --out checkpoints \
    --fasttrack

# 5. Test model
python scripts/ttm_vitaldb.py test \
    --ckpt checkpoints/model.pt \
    --model-yaml configs/model.yaml \
    --run-yaml configs/run.yaml \
    --split-file data/splits_fasttrack.json \
    --outdir data \
    --out results
```

---

## Key Fixes Applied

### Import Corrections
✅ `make_patient_level_splits` (not `create_patient_splits`)
✅ Direct functions from `vitaldb_loader` (no `VitalDBLoader` class)
✅ `make_windows`, `compute_normalization_stats` (no `WindowBuilder`)
✅ `find_ppg_peaks`, `find_ecg_rpeaks` (signal-specific)
✅ `compute_sqi` only (no `compute_ssqi`)
✅ `TensorDataset` from PyTorch (no `TTMDataset`)

### Implementation Improvements
- Proper handling of VitalDB case sets
- Correct window creation with peak validation
- Normalization statistics saved and reused
- Simplified dataset creation with TensorDataset
- Better error handling and logging
- Progress bars for long operations

---

## Configuration Files

Ensure these exist:

### `configs/channels.yaml`
```yaml
ppg:
  vitaldb_track: PLETH
  type: ppg
  filter:
    type: cheby2
    order: 4
    lowcut: 0.5
    highcut: 10.0
```

### `configs/windows.yaml`
```yaml
window_length_sec: 10.0
stride_sec: 10.0
min_cycles: 3
normalize_method: zscore
```

### `configs/model.yaml`
```yaml
variant: ibm-granite/granite-timeseries-ttm-r1
task: classification
num_classes: 2
freeze_encoder: true
head_type: linear
```

### `configs/run.yaml`
```yaml
num_epochs: 10
batch_size: 32
learning_rate: 0.001
weight_decay: 0.0001
use_amp: false
grad_clip: 1.0
patience: 10
num_workers: 0
device: cpu
seed: 42
```

---

## Troubleshooting

**"Split not found in splits file"**
→ Run `prepare-splits` first

**"Training data not found"**
→ Run `build-windows` for train split first

**"No valid windows created"**
→ Check `--min-sqi` threshold or `--duration-sec`

**"Model loading failed"**
→ First run downloads TTM from HuggingFace (~50MB)

---

## Quick Test (5-10 minutes)

For a quick end-to-end test, use the dedicated test script:
```bash
python scripts/quick_e2e_test.py
```

This tests the entire pipeline with 2 cases in 5-10 minutes.
