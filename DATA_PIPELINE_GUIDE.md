# 📊 Complete Data Pipeline Guide: VitalDB & BUT-PPG

## 🎯 Overview: What Data Do You Have?

### VitalDB Dataset
- ✅ **PPG** (Photoplethysmography) - Track: `PLETH`
- ✅ **ECG** (Electrocardiogram) - Track: `ECG_II`
- ✅ **500+ surgical cases** from ICU patients
- ✅ **Multi-modal:** Can use both PPG + ECG together

### BUT-PPG Dataset
- ✅ **PPG** (Photoplethysmography) only
- ❌ **NO ECG** (it's a wearable PPG-only dataset)
- ⚠️ **Your data:** You need to provide this (not included)
- 📁 **Format:** .npy, .mat, .csv, .hdf5 files

### Your Expectation vs Reality

**Your expectation:** "I expect to have ppg ecg for both datasets"

**Reality:**
```
VitalDB:   ✅ PPG + ECG (multi-modal available)
BUT-PPG:   ✅ PPG only (❌ no ECG - it's from wearables)
```

**Solution for Multi-modal:**
- Use **VitalDB** for multi-modal pre-training (PPG + ECG)
- Use **BUT-PPG** for single-modal fine-tuning (PPG only)
- OR find another dataset with PPG + ECG for fine-tuning

---

## 📋 Complete Data Pipeline Overview

```
┌────────────────────────────────────────────────────────────┐
│                    VITALDB PIPELINE                        │
└────────────────────────────────────────────────────────────┘

Stage 1: Discover Cases
    ↓
Stage 2: Download & Cache Signals (PLETH + ECG_II)
    ↓
Stage 3: Preprocess & Filter (0.5-8 Hz)
    ↓
Stage 4: Create Windows (10s @ 125Hz)
    ↓
Stage 5: Quality Control (min 3 cycles)
    ↓
Stage 6: Create Train/Val/Test Splits
    ↓
Stage 7: Pre-training (SSL with pairs)


┌────────────────────────────────────────────────────────────┐
│                   BUT-PPG PIPELINE                         │
└────────────────────────────────────────────────────────────┐

Stage 1: Prepare Data Files (.npy/.mat/.csv)
    ↓
Stage 2: Load & Validate
    ↓
Stage 3: Preprocess & Filter (0.5-8 Hz) ← SAME as VitalDB
    ↓
Stage 4: Create Windows (10s @ 125Hz) ← SAME
    ↓
Stage 5: Quality Control ← SAME
    ↓
Stage 6: Extract Clinical Labels (if available)
    ↓
Stage 7: Create Train/Val/Test Splits
    ↓
Stage 8: Fine-tuning (Transfer Learning)
```

---

## 🔧 VitalDB Data Pipeline (Detailed)

### Stage 1: Discover Available Cases

**What it does:** Find VitalDB cases that have PPG (PLETH) and/or ECG data

**Command:**
```bash
cd /Users/aliab/Desktop/TinyFoundationModelForBioSignals

# Discover cases with PPG
python -c "
from src.data.vitaldb_loader import find_cases_with_track, get_available_case_sets

# Method 1: Find cases with specific track
ppg_cases = find_cases_with_track('PLETH')
print(f'Found {len(ppg_cases)} cases with PPG')

# Method 2: Use pre-defined sets
case_sets = get_available_case_sets()
print(f'Available case sets: {list(case_sets.keys())}')
print(f'BIS cases: {len(case_sets[\"bis\"])} cases')
"
```

**Output:**
```
Found 500+ cases with PPG
Available case sets: ['bis', 'desflurane', 'sevoflurane', ...]
BIS cases: 500 cases
```

### Stage 2: Create Train/Val/Test Splits

**What it does:** Divide cases into train/val/test sets

**Command:**
```bash
# FastTrack mode (70 cases for quick testing)
python scripts/ttm_vitaldb.py prepare-splits \
    --output data \
    --mode fasttrack \
    --seed 42

# Full mode (all available cases)
python scripts/ttm_vitaldb.py prepare-splits \
    --output data \
    --mode full \
    --seed 42
```

**Output:**
```
data/
├── splits_fasttrack.json    # 50 train, 10 val, 10 test
└── splits_full.json          # 350 train, 75 val, 75 test (approx)
```

**File format:**
```json
{
  "train": ["1", "2", "3", ...],
  "val": ["100", "101", ...],
  "test": ["200", "201", ...]
}
```

### Stage 3: Download & Cache Signals

**What it does:** Download PPG and ECG from VitalDB, apply basic filtering, and cache

**Command:**
```bash
# Quick test (5 minutes of data from 5 cases)
python scripts/smoke_realdata_5min.py

# OR build full windows (processes all cases in splits)
python scripts/ttm_vitaldb.py build-windows \
    --splits data/splits_fasttrack.json \
    --output data/cache \
    --modality ppg,ecg \
    --duration 300 \
    --workers 4
```

**What happens:**
1. Loads PLETH (PPG) and ECG_II tracks from VitalDB
2. Handles sampling rate issues (100 Hz → 125 Hz)
3. Applies bandpass filter (0.5-8.0 Hz Chebyshev)
4. Removes NaN values and artifacts
5. Saves cleaned signals to cache

**Output:**
```
data/cache/
├── case_1_clean/
│   ├── PLETH.npz      # PPG: [T, 1] at 125 Hz
│   └── ECG_II.npz     # ECG: [T, 1] at 125 Hz
├── case_2_clean/
│   ├── PLETH.npz
│   └── ECG_II.npz
└── ...
```

### Stage 4: Create Dataset & Verify

**What it does:** Load data using PyTorch Dataset, create windows, apply QC

**Command:**
```bash
# Test loading with VitalDB dataset
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').absolute()))

from src.data.vitaldb_dataset import VitalDBDataset

# Create dataset
dataset = VitalDBDataset(
    cache_dir='data/cache',
    splits_file='data/splits_fasttrack.json',
    split='train',
    modality='ppg',  # or 'ecg' or 'ppg,ecg'
    window_sec=10.0,
    target_fs=125.0
)

print(f'✓ Dataset created')
print(f'  Split: train')
print(f'  Cases: {len(dataset.case_ids)}')
print(f'  Modality: ppg')

# Load sample
seg1, seg2 = dataset[0]
print(f'✓ Sample loaded')
print(f'  seg1 shape: {seg1.shape}')
print(f'  seg2 shape: {seg2.shape}')
print(f'  Mean: {seg1.mean():.3f}, Std: {seg1.std():.3f}')
"
```

**Expected output:**
```
✓ Dataset created
  Split: train
  Cases: 50
  Modality: ppg
✓ Sample loaded
  seg1 shape: torch.Size([1, 1250])
  seg2 shape: torch.Size([1, 1250])
  Mean: 0.012, Std: 0.998
```

### Stage 5: Pre-training (SSL)

**What it does:** Train foundation model using self-supervised learning

**Command:**
```bash
# Start pre-training
python scripts/ttm_vitaldb.py \
    --mode full \
    --config configs/ssl_pretrain.yaml \
    --splits data/splits_fasttrack.json \
    --cache-dir data/cache \
    --output checkpoints/vitaldb_pretrain \
    --epochs 100 \
    --batch-size 128 \
    --modality ppg,ecg  # Both PPG and ECG

# Monitor training
tensorboard --logdir checkpoints/vitaldb_pretrain/logs
```

**What happens:**
1. Loads cached signals from `data/cache/`
2. Creates windows (10s, 1250 samples)
3. Applies quality control (min 3 cardiac cycles)
4. Creates same-patient pairs for contrastive learning
5. Trains encoder with masked modeling
6. Saves best checkpoint

**Training logs:**
```
Epoch 1/100: loss=2.341, val_loss=2.156
Epoch 2/100: loss=2.012, val_loss=1.987
...
Epoch 100/100: loss=0.456, val_loss=0.523
✓ Best model saved: checkpoints/vitaldb_pretrain/best.ckpt
```

---

## 🔧 BUT-PPG Data Pipeline (Detailed)

### Stage 1: Prepare Your Data

**What you need:** BUT-PPG database files in one of these formats:

**Option A: NPY format (recommended)**
```bash
data/but_ppg/
├── subject_001.npy       # Signal: np.array([n_samples,])
├── subject_001.json      # Metadata: {"fs": 100}
├── subject_002.npy
├── subject_002.json
└── ...
```

**Option B: MAT format**
```bash
data/but_ppg/
├── subject_001.mat       # Contains: 'ppg', 'fs'
├── subject_002.mat
└── ...
```

**Option C: CSV format**
```bash
data/but_ppg/
├── subject_001.csv       # Columns: ppg, timestamp
├── subject_002.csv
└── ...
```

**If you don't have BUT-PPG data yet:**
```bash
# Download from: https://peterhcharlton.github.io/RRest/ppg_datasets.html
# Or use placeholder data for testing:
python -c "
import numpy as np
from pathlib import Path
import json

# Create test data
data_dir = Path('data/but_ppg')
data_dir.mkdir(exist_ok=True, parents=True)

for i in range(1, 11):
    # Generate 60s synthetic PPG at 100 Hz
    fs = 100
    duration = 60
    t = np.linspace(0, duration, int(fs * duration))
    ppg = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(len(t))
    
    # Save
    np.save(data_dir / f'subject_{i:03d}.npy', ppg)
    with open(data_dir / f'subject_{i:03d}.json', 'w') as f:
        json.dump({'fs': fs}, f)

print(f'✓ Created 10 test subjects in data/but_ppg/')
"
```

### Stage 2: Verify Data Loading

**What it does:** Test that BUT-PPG loader can read your files

**Command:**
```bash
python -c "
from src.data.butppg_loader import BUTPPGLoader

# Initialize loader
loader = BUTPPGLoader(
    data_dir='data/but_ppg',
    fs=125.0,  # Target frequency
    apply_windowing=False
)

print(f'✓ Loader initialized')
print(f'  Subjects found: {len(loader.get_subject_list())}')
print(f'  Subjects: {loader.get_subject_list()[:5]}...')

# Test loading
subject_id = loader.get_subject_list()[0]
result = loader.load_subject(subject_id, return_windows=False)

if result:
    signal, metadata = result
    print(f'✓ Loaded subject: {subject_id}')
    print(f'  Signal shape: {signal.shape}')
    print(f'  Sampling rate: {metadata[\"fs\"]} Hz')
    print(f'  Duration: {signal.shape[0] / metadata[\"fs\"]:.1f}s')
else:
    print(f'✗ Failed to load {subject_id}')
"
```

**Expected output:**
```
✓ Loader initialized
  Subjects found: 10
  Subjects: ['subject_001', 'subject_002', 'subject_003', ...]
✓ Loaded subject: subject_001
  Signal shape: (6000, 1)
  Sampling rate: 125.0 Hz
  Duration: 48.0s
```

### Stage 3: Extract Clinical Labels (Optional)

**What it does:** Extract labels from metadata for supervised learning

**If you have a metadata CSV:**
```bash
# Create metadata file
cat > data/but_ppg/metadata.csv << EOF
subject_id,age,sex,bmi,condition
subject_001,45,M,24.5,healthy
subject_002,52,F,28.3,disease
subject_003,38,M,22.1,healthy
EOF

# Extract labels
python -c "
from src.data.clinical_labels import ClinicalLabelExtractor

extractor = ClinicalLabelExtractor(
    metadata_path='data/but_ppg/metadata.csv'
)

# Extract labels for subjects
labels = extractor.extract_batch_labels(['subject_001', 'subject_002'])
print('✓ Extracted labels:')
for subject_id, subject_labels in labels.items():
    print(f'  {subject_id}: {subject_labels}')
"
```

### Stage 4: Create Dataset & Verify Preprocessing

**What it does:** Create PyTorch dataset with unified preprocessing

**Command:**
```bash
python -c "
from src.data.butppg_dataset import BUTPPGDataset

# Create dataset
dataset = BUTPPGDataset(
    data_dir='data/but_ppg',
    split='train',
    modality='ppg',
    window_sec=10.0,
    hop_sec=5.0,
    train_ratio=0.7,
    val_ratio=0.15,
    enable_qc=True
)

print(f'✓ BUT-PPG Dataset created')
print(f'  Split: train')
print(f'  Subjects: {len(dataset.subjects)}')
print(f'  Total segments: {len(dataset)}')
print(f'  Window size: {dataset.segment_length} samples')

# Load sample
seg1, seg2 = dataset[0]
print(f'✓ Sample loaded')
print(f'  seg1 shape: {seg1.shape}')
print(f'  seg2 shape: {seg2.shape}')
print(f'  Mean: {seg1.mean():.3f}, Std: {seg1.std():.3f}')

# Verify preprocessing matches VitalDB
print(f'✓ Preprocessing config:')
print(f'  Filter: {dataset.preprocessing_config[\"filter_type\"]} {dataset.preprocessing_config[\"filter_order\"]}th order')
print(f'  Band: {dataset.preprocessing_config[\"filter_band\"]} Hz')
print(f'  Sampling: {dataset.preprocessing_config[\"target_fs\"]} Hz')
print(f'  Window: {dataset.preprocessing_config[\"window_sec\"]}s')
"
```

**Expected output:**
```
✓ BUT-PPG Dataset created
  Split: train
  Subjects: 7
  Total segments: 140
  Window size: 1250 samples
✓ Sample loaded
  seg1 shape: torch.Size([1, 1250])
  seg2 shape: torch.Size([1, 1250])
  Mean: -0.003, Std: 1.012
✓ Preprocessing config:
  Filter: cheby2 4th order
  Band: [0.5, 8.0] Hz
  Sampling: 125 Hz
  Window: 10.0s
```

### Stage 5: Fine-tuning

**What it does:** Fine-tune pre-trained model on BUT-PPG

**Command:**
```bash
# Fine-tune with pre-trained weights
python scripts/finetune_butppg.py \
    --pretrained checkpoints/vitaldb_pretrain/best.ckpt \
    --data-dir data/but_ppg \
    --task classification \
    --num-classes 2 \
    --output checkpoints/butppg_finetune \
    --epochs 50 \
    --batch-size 64 \
    --lr 1e-4

# Evaluate on test set
python scripts/evaluate_butppg.py \
    --model checkpoints/butppg_finetune/best.ckpt \
    --data-dir data/but_ppg \
    --split test \
    --output results/butppg_eval.json
```

---

## 🔄 Multi-Modal Pipeline (PPG + ECG)

### For VitalDB (Both Modalities Available)

**Configure for multi-modal:**
```yaml
# configs/ssl_pretrain.yaml
data:
  channels: [PPG, ECG]  # Both modalities
  fs: 125
  window_sec: 10.0
  n_channels: 2  # Two input channels

model:
  input_channels: 2  # PPG + ECG
  context_length: 1250
```

**Load multi-modal data:**
```python
from src.data.vitaldb_dataset import VitalDBDataset

# Multi-modal dataset
dataset = VitalDBDataset(
    cache_dir='data/cache',
    split='train',
    modality='ppg,ecg',  # Both!
    window_sec=10.0
)

# Returns stacked channels
seg1, seg2 = dataset[0]
# seg1.shape = [2, 1250]  # 2 channels (PPG + ECG)
```

### For BUT-PPG (PPG Only)

**Option 1: Use PPG only (current setup)**
```python
dataset = BUTPPGDataset(
    data_dir='data/but_ppg',
    modality='ppg',  # PPG only
    ...
)
```

**Option 2: Find ECG-enabled wearable dataset**
- MIMIC-III waveforms (has PPG + ECG)
- PhysioNet datasets
- Your own collected data

---

## 📊 Complete End-to-End Pipeline Script

Here's a master script to run everything:

```bash
#!/bin/bash
# complete_pipeline.sh

set -e  # Exit on error

echo "=========================================="
echo "  COMPLETE DATA PIPELINE"
echo "=========================================="

# ============================================
# PART 1: VITALDB PREPARATION
# ============================================

echo ""
echo "[1/8] Creating VitalDB splits..."
python scripts/ttm_vitaldb.py prepare-splits \
    --output data \
    --mode fasttrack \
    --seed 42

echo ""
echo "[2/8] Caching VitalDB signals (this may take a while)..."
python scripts/smoke_realdata_5min.py

echo ""
echo "[3/8] Verifying VitalDB dataset..."
python -c "
from src.data.vitaldb_dataset import VitalDBDataset
dataset = VitalDBDataset(
    cache_dir='data/cache',
    split='train',
    modality='ppg'
)
print(f'✓ VitalDB ready: {len(dataset.case_ids)} cases')
"

echo ""
echo "[4/8] Pre-training on VitalDB..."
python scripts/ttm_vitaldb.py \
    --mode full \
    --config configs/ssl_pretrain.yaml \
    --splits data/splits_fasttrack.json \
    --output checkpoints/vitaldb_pretrain \
    --epochs 100 \
    --batch-size 128

# ============================================
# PART 2: BUT-PPG PREPARATION
# ============================================

echo ""
echo "[5/8] Preparing BUT-PPG data..."
# Assuming you have data in data/but_ppg/

echo ""
echo "[6/8] Verifying BUT-PPG dataset..."
python -c "
from src.data.butppg_dataset import BUTPPGDataset
dataset = BUTPPGDataset(
    data_dir='data/but_ppg',
    split='train',
    modality='ppg'
)
print(f'✓ BUT-PPG ready: {len(dataset.subjects)} subjects')
"

echo ""
echo "[7/8] Fine-tuning on BUT-PPG..."
python scripts/finetune_butppg.py \
    --pretrained checkpoints/vitaldb_pretrain/best.ckpt \
    --data-dir data/but_ppg \
    --task classification \
    --output checkpoints/butppg_finetune \
    --epochs 50

echo ""
echo "[8/8] Evaluating on test set..."
python scripts/evaluate_butppg.py \
    --model checkpoints/butppg_finetune/best.ckpt \
    --data-dir data/but_ppg \
    --split test \
    --output results/butppg_eval.json

echo ""
echo "=========================================="
echo "  ✓ PIPELINE COMPLETE!"
echo "=========================================="
echo "Results saved to:"
echo "  - checkpoints/vitaldb_pretrain/"
echo "  - checkpoints/butppg_finetune/"
echo "  - results/butppg_eval.json"
```

**Make it executable and run:**
```bash
chmod +x complete_pipeline.sh
./complete_pipeline.sh
```

---

## 📁 Expected Directory Structure After Pipeline

```
TinyFoundationModelForBioSignals/
├── data/
│   ├── cache/                      # VitalDB cached signals
│   │   ├── case_1_clean/
│   │   │   ├── PLETH.npz          # PPG at 125 Hz
│   │   │   └── ECG_II.npz         # ECG at 125 Hz
│   │   ├── case_2_clean/
│   │   └── ...                     # 500+ cases
│   ├── but_ppg/                    # BUT-PPG data
│   │   ├── subject_001.npy
│   │   ├── subject_001.json
│   │   ├── metadata.csv            # Optional labels
│   │   └── ...
│   ├── splits_fasttrack.json       # Train/val/test splits
│   └── splits_full.json
├── checkpoints/
│   ├── vitaldb_pretrain/
│   │   ├── best.ckpt              # Pre-trained model
│   │   ├── logs/                   # TensorBoard logs
│   │   └── config.yaml
│   └── butppg_finetune/
│       ├── best.ckpt              # Fine-tuned model
│       └── logs/
└── results/
    ├── butppg_eval.json            # Test results
    └── comparison.pdf              # Baseline comparison
```

---

## 🎯 Quick Start Commands

### Just Test Everything Works:
```bash
# 1. Test VitalDB loading
python scripts/smoke_realdata_5min.py

# 2. Test BUT-PPG loading
python tests/test_butppg_fix.py

# 3. Run all tests
pytest tests/test_butppg_* -v
```

### Run Full Pipeline:
```bash
# Create and run the complete pipeline
cat > run_pipeline.sh << 'EOF'
#!/bin/bash
# Stage 1: VitalDB
python scripts/ttm_vitaldb.py prepare-splits --mode fasttrack
python scripts/smoke_realdata_5min.py
python scripts/ttm_vitaldb.py --mode full --epochs 100

# Stage 2: BUT-PPG
python scripts/finetune_butppg.py \
    --pretrained checkpoints/vitaldb_pretrain/best.ckpt
python scripts/evaluate_butppg.py --split test
EOF

chmod +x run_pipeline.sh
./run_pipeline.sh
```

---

## ❓ FAQ

### Q: Can I use only PPG without ECG?
**A:** Yes! Just set `modality='ppg'` in configs/datasets.

### Q: How do I add ECG to BUT-PPG?
**A:** BUT-PPG typically doesn't have ECG. Use a different dataset like MIMIC-III or collect your own.

### Q: How long does pre-training take?
**A:** On GPU: 4-12 hours for fasttrack, 1-3 days for full.

### Q: Can I resume interrupted training?
**A:** Yes! Add `--resume checkpoints/vitaldb_pretrain/last.ckpt`

### Q: How much GPU memory needed?
**A:** ~8GB for batch_size=128, ~4GB for batch_size=64

---

## 📚 Key Files Reference

**Data Loading:**
- `src/data/vitaldb_loader.py` - Raw VitalDB access
- `src/data/vitaldb_dataset.py` - PyTorch dataset
- `src/data/butppg_loader.py` - Raw BUT-PPG access
- `src/data/butppg_dataset.py` - PyTorch dataset

**Preprocessing:**
- `src/data/filters.py` - Bandpass filters
- `src/data/windows.py` - Windowing logic
- `src/data/quality.py` - Quality control

**Scripts:**
- `scripts/ttm_vitaldb.py` - VitalDB pipeline
- `scripts/finetune_butppg.py` - Fine-tuning
- `scripts/evaluate_butppg.py` - Evaluation

**Configs:**
- `configs/ssl_pretrain.yaml` - Pre-training settings
- `configs/channels.yaml` - Channel specifications

---

## ✅ Pipeline Checklist

Before running the full pipeline, verify:

- [ ] VitalDB access working (`python scripts/smoke_realdata_5min.py`)
- [ ] BUT-PPG data prepared in `data/but_ppg/`
- [ ] Tests passing (`pytest tests/test_butppg_* -v`)
- [ ] GPU available (`nvidia-smi` or check MPS on Mac)
- [ ] Enough disk space (~10GB for cache, ~5GB for checkpoints)
- [ ] TensorBoard installed (`pip install tensorboard`)

---

This guide covers the complete end-to-end pipeline for both datasets!
