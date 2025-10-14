# Quick Start Guide: Build VitalDB Windows

## 🎯 **Three Ways to Build Windows (All Warnings Disabled)**

---

## **Option 1: Simple One-Liner (Recommended)**

```bash
PYTHONWARNINGS=ignore python3 scripts/ttm_vitaldb.py build-windows \
    --channels-yaml configs/channels.yaml \
    --windows-yaml configs/windows.yaml \
    --split-file configs/splits/splits_fallback.json \
    --split train \
    --outdir data/vitaldb_windows \
    --multiprocess \
    --num-workers 4
```

For validation windows:
```bash
PYTHONWARNINGS=ignore python3 scripts/ttm_vitaldb.py build-windows \
    --channels-yaml configs/channels.yaml \
    --windows-yaml configs/windows.yaml \
    --split-file configs/splits/splits_fallback.json \
    --split val \
    --outdir data/vitaldb_windows \
    --multiprocess \
    --num-workers 4
```

---

## **Option 2: Using Bash Script**

```bash
# Make executable
chmod +x scripts/run_build_windows.sh

# Build train windows
bash scripts/run_build_windows.sh

# Build val windows
bash scripts/run_build_windows.sh val
```

---

## **Option 3: Using Python Wrapper (Cleanest)**

```bash
# Build train windows
python3 scripts/build_windows_quiet.py train

# Build val windows
python3 scripts/build_windows_quiet.py val

# Build test windows
python3 scripts/build_windows_quiet.py test
```

---

## 📝 **What Each Argument Does**

| Argument | File | Purpose |
|----------|------|---------|
| `--channels-yaml` | `configs/channels.yaml` | Channel definitions (PPG, ECG tracks) |
| `--windows-yaml` | `configs/windows.yaml` | Window parameters (10s, 125Hz) |
| `--split-file` | `configs/splits/splits_fallback.json` | Train/val/test case IDs |
| `--split` | `train` / `val` / `test` | Which split to build |
| `--outdir` | `data/vitaldb_windows` | Where to save `.npz` files |
| `--multiprocess` | (flag) | Enable parallel processing |
| `--num-workers` | `4` | Number of CPU cores to use |

---

## 🔇 **How Warnings Are Disabled**

All three options suppress warnings differently:

### Option 1 (Environment variable):
```bash
PYTHONWARNINGS=ignore python3 script.py
```

### Option 2 (In bash script):
```bash
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL=3
```

### Option 3 (In Python code):
```python
import warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
```

---

## 📦 **Expected Output**

After successful execution:

```
data/vitaldb_windows/
├── train_windows.npz     # Shape: [N, 2, 1250]
├── val_windows.npz       # Shape: [M, 2, 1250]
└── test_windows.npz      # Shape: [K, 2, 1250]
```

Where:
- **N, M, K**: Number of windows per split
- **2**: Number of channels (PPG, ECG)
- **1250**: Samples per window (10s × 125Hz)

---

## ⚡ **Complete Pipeline**

```bash
# Step 1: Create splits (if not already done)
PYTHONWARNINGS=ignore python3 scripts/ttm_vitaldb.py prepare-splits \
    --mode fasttrack \
    --case-set bis \
    --output configs/splits

# Step 2: Build train windows
python3 scripts/build_windows_quiet.py train

# Step 3: Build val windows
python3 scripts/build_windows_quiet.py val

# Step 4: Verify outputs
ls -lh data/vitaldb_windows/
```

---

## 🐛 **Troubleshooting**

### Problem: "splits_fallback.json not found"
**Solution:**
```bash
# Create splits first
PYTHONWARNINGS=ignore python3 scripts/ttm_vitaldb.py prepare-splits \
    --mode fasttrack \
    --output configs/splits
```

### Problem: "No VitalDB data found"
**Solution:** Check that VitalDB vital files exist in the expected location:
```bash
# Default location
ls ~/vitaldb/vital_files/
```

Or specify custom path in `configs/channels.yaml`

### Problem: Still seeing warnings
**Solution:** Use Option 3 (Python wrapper) which suppresses warnings in code

### Problem: Process killed / Out of memory
**Solution:** Reduce workers:
```bash
# Use fewer cores
python3 scripts/build_windows_quiet.py train --num-workers 2

# Or disable multiprocessing
PYTHONWARNINGS=ignore python3 scripts/ttm_vitaldb.py build-windows \
    ... (all args) \
    --num-workers 1  # Remove --multiprocess flag
```

---

## 💡 **Pro Tips**

1. **Check progress**: The script shows a progress bar for each case being processed

2. **Verify output**:
```python
import numpy as np
data = np.load('data/vitaldb_windows/train_windows.npz')
print(f"Shape: {data['signals'].shape}")  # Should be [N, 2, 1250]
```

3. **Speed up**: Use more workers if you have many CPU cores:
```bash
python3 scripts/build_windows_quiet.py train --num-workers 8
```

4. **FastTrack mode**: For quick testing, use fasttrack splits (70 cases):
```bash
# In prepare-splits step
--mode fasttrack  # Instead of 'full'
```

---

## ✅ **Next Steps**

After building windows:

1. **Run smoke test**:
```bash
python3 scripts/smoke_realdata_5min.py \
    --data-dir data/vitaldb_windows \
    --max-windows 64
```

2. **Start SSL pretraining**:
```bash
python3 scripts/pretrain_vitaldb_ssl.py \
    --config configs/ssl_pretrain.yaml \
    --data-dir data/vitaldb_windows \
    --output-dir artifacts/foundation_model \
    --epochs 100
```

---

## 📚 **Documentation**

- Full workflow: `docs/WORKFLOW.md`
- QA audit: `QA_AUDIT_REPORT.md`
- Configuration: `configs/windows.yaml`, `configs/channels.yaml`
