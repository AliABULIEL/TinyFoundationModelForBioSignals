# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Tiny Foundation Model for Biosignals** - A PyTorch-based foundation model for multi-modal biosignal analysis using IBM's Tiny Time Mixers (TTM) architecture. The project implements self-supervised pre-training on VitalDB (PPG/ECG) and downstream fine-tuning on clinical tasks including hypotension prediction, blood pressure estimation, and signal quality assessment.

**Key Innovation:** Multi-modal biosignal learning with patch-based masked signal modeling (MSM) and multi-resolution STFT loss, achieving SOTA performance on multiple clinical benchmarks.

## Architecture Overview

### Core Components

1. **Self-Supervised Learning (src/ssl/)**
   - `masking.py`: Patch-based masking strategies (random, block) for MSM
   - `objectives.py`: MSM loss + Multi-Resolution STFT loss
   - `pretrainer.py`: SSL training loop with automatic mixed precision
   - Key: 40% masking ratio, 125-sample patches (1s @ 125Hz)

2. **Model Architecture (src/models/)**
   - `ttm_adapter.py`: Wrapper for IBM TTM with biosignal-specific adaptations
   - `decoders.py`: Lightweight reconstruction heads for SSL and task-specific heads
   - `heads.py`: Task-specific heads (classification, regression)
   - `lora.py`: LoRA adapters for parameter-efficient fine-tuning
   - `trainers.py`: Training utilities with best model saving and early stopping

3. **Data Pipeline (src/data/)**
   - **VitalDB**: `vitaldb_dataset.py`, `vitaldb_loader.py` - Multi-modal PPG+ECG loading
   - **BUT-PPG**: `butppg_dataset.py`, `butppg_loader.py` - Signal quality dataset
   - **Preprocessing**: `windows.py` (windowing), `filters.py` (signal processing)
   - **Quality Control**: `quality.py` (SQI computation), `detect.py` (PPG cycle detection)
   - **Subject-level splits**: `configs/splits/splits_full.json` prevents data leakage

4. **Tasks (src/tasks/)**
   - `base.py`: Abstract task interface with evaluation and benchmarking
   - `hypotension.py`: MAP < 65 mmHg prediction (5/10/15 min windows)
   - `blood_pressure.py`: BP regression with AAMI compliance metrics
   - `ppg_quality_butppg.py`: Signal quality classification
   - `clinical_tasks.py`: Additional clinical prediction tasks

5. **Evaluation (src/eval/)**
   - `metrics.py`: Classification (AUROC, AUPRC, F1) and regression (MAE, RMSE, CCC) metrics
   - `calibration.py`: Temperature scaling, Platt scaling, ECE/MCE metrics
   - Note: Missing confidence intervals and AAMI compliance (see AUDIT_REPORT.md)

### Data Flow

```
Raw Data (VitalDB API / BUT-PPG files)
  ↓
Preprocessing (windowing, filtering, quality checks)
  ↓
Subject-level splits (train/val/test)
  ↓
SSL Pre-training (MSM + STFT loss on VitalDB)
  ↓
Foundation Model Checkpoint
  ↓
Fine-tuning (task-specific heads with LoRA)
  ↓
Evaluation (subject-level metrics, benchmark comparison)
```

## Common Development Commands

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_ssl_masking.py -v

# Run specific test function
pytest tests/test_datasets.py::test_vitaldb_dataset -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run phased tests (setup → data prep → SSL → fine-tuning)
python tests/run_all_phases.py

# Run SSL-specific tests
python tests/run_ssl_tests.py

# Quick fix verification
python tests/quick_fix_verification.py
```

### Training

```bash
# SSL Pre-training on VitalDB
python scripts/pretrain_vitaldb_ssl.py \
  --config configs/ssl_pretrain.yaml \
  --mode fasttrack \
  --epochs 50

# Resume from checkpoint
python scripts/pretrain_vitaldb_ssl.py \
  --resume artifacts/foundation_model/checkpoint_epoch_50.pt

# Fine-tune on BUT-PPG signal quality task
python scripts/finetune_butppg.py \
  --pretrained artifacts/foundation_model/best_model.pt \
  --config configs/finetune_butppg.yaml

# Quick smoke test (5 minutes)
python scripts/smoke_realdata_5min.py
```

### Data Preparation

```bash
# Download BUT-PPG dataset
python scripts/download_but_ppg.py --output data/butppg

# Build preprocessed windows (VitalDB)
python scripts/prepare_vitaldb_fixes.py

# Build BUT-PPG windows
python scripts/build_butppg_windows.py

# Verify data pipeline
python scripts/test_dataloader_creation.py
```

### Debugging

```bash
# Check environment setup
python scripts/env_check.py

# Debug VitalDB loading
python scripts/debug_vitaldb.py

# Test with real VitalDB data
python scripts/test_real_vitaldb.py
```

## Configuration Files

### Key Configs
- `configs/ssl_pretrain.yaml`: SSL pre-training hyperparameters
  - Model: TTM encoder variant, d_model, patch_size
  - SSL: mask_ratio (0.4), mask_type (random/block)
  - Training: epochs, batch_size, lr, optimizer
  - STFT: n_ffts, hop_lengths, loss_weight

- `configs/splits/splits_full.json`: Subject-level train/val/test splits
  - **CRITICAL**: Always use for evaluation to prevent data leakage
  - Format: `{"train": [case_ids], "val": [case_ids], "test": [case_ids]}`

- `configs/model.yaml`: Model architecture configuration
- `configs/channels.yaml`: Channel/modality configuration
- `configs/windows.yaml`: Windowing parameters

## Critical Implementation Details

### 1. Patch Size Synchronization
**CRITICAL FIX (Oct 2025)**: The TTM encoder and SSL masking must use matching patch sizes.

```python
# ✓ CORRECT: Get patch_size from encoder config
if hasattr(encoder, 'backbone') and hasattr(encoder.backbone, 'config'):
    patch_size = encoder.backbone.config.patch_length

# ✗ WRONG: Hard-coded patch_size mismatch
patch_size = 125  # May not match encoder!
```

**Why:** TTM adapts patch_size based on context_length. Using mismatched values causes dimension errors during reconstruction. See `scripts/pretrain_vitaldb_ssl.py:595` for dynamic d_model calculation.

### 2. Subject-Level Evaluation
**Always use subject-level splits** from `configs/splits/splits_full.json` to prevent data leakage. Never evaluate at the segment level for test sets.

```python
# ✓ CORRECT: Load splits for subject-level evaluation
with open('configs/splits/splits_full.json') as f:
    splits = json.load(f)
test_cases = splits['test']

# ✗ WRONG: Random segment splits can leak patient data
```

### 3. Multi-Modal Channel Ordering
**Consistent channel ordering is critical:**
- Channel 0: PPG (PLETH)
- Channel 1: ECG (ECG_II)

Both `VitalDBDataset` and `BUTPPGDataset` follow this convention. Do not change ordering without updating all downstream components.

### 4. Best Model Saving
Trainers in `src/models/trainers.py` automatically save:
- `best_model.pt`: Best model based on validation metric
- `last_checkpoint.pt`: Most recent epoch
- `metrics.json`: Training history

Always use `best_model.pt` for evaluation, not `last_checkpoint.pt`.

### 5. Missing Features (from AUDIT_REPORT.md)
The evaluation pipeline is **60% complete** and missing:
- ❌ Confidence intervals (bootstrap CI) for all metrics
- ❌ AAMI compliance metrics (ME ≤ 5 mmHg, SDE ≤ 8 mmHg) for BP regression
- ❌ Heart rate estimation task implementation
- ❌ Automated benchmark report generation

**Do not claim publication-ready evaluation results** until these are implemented.

## Code Style & Conventions

### Naming Conventions
- **Classes**: PascalCase (`VitalDBDataset`, `TTMAdapter`)
- **Functions/Methods**: snake_case (`load_channel`, `random_masking`)
- **Constants**: UPPER_SNAKE_CASE (`SUPPORTED_MODALITIES`, `TRACK_MAPPING`)
- **Private methods**: Leading underscore (`_preprocess_signal`, `_load_single_file`)

### Docstrings
Use Google-style docstrings with:
```python
def function(arg1: Type, arg2: Type) -> ReturnType:
    """Brief description.

    Longer description if needed.

    Args:
        arg1: Description
        arg2: Description

    Returns:
        Description of return value

    Example:
        >>> result = function(val1, val2)
        >>> result.shape
        (16, 2, 1250)
    """
```

### Error Handling
- Use informative error messages with context
- Log warnings for recoverable issues
- Raise `ValueError` for invalid arguments, `FileNotFoundError` for missing data

## Common Pitfalls

1. **Patch Size Mismatch**: Always query encoder's actual patch_size, never hard-code
2. **Data Leakage**: Use subject-level splits, never random segment splits for evaluation
3. **NaN Handling**: Always check for NaN/Inf after signal processing
4. **Channel Ordering**: Maintain PPG=0, ECG=1 convention throughout
5. **Tensor Shapes**: VitalDB/BUT-PPG use [N, C, T] (channels-first) format
6. **Device Placement**: Always move models and data to same device before forward pass
7. **Stats Files**: Never load `*_stats.npz` as training data (these are metadata)

## Project Structure

```
TinyFoundationModelForBioSignals/
├── configs/              # Configuration files
│   ├── ssl_pretrain.yaml
│   ├── model.yaml
│   └── splits/          # Subject-level splits
├── src/
│   ├── data/            # Data loaders and preprocessing
│   ├── models/          # Model architectures and adapters
│   ├── ssl/             # Self-supervised learning components
│   ├── tasks/           # Downstream task definitions
│   ├── eval/            # Evaluation metrics and benchmarking
│   └── utils/           # Utilities (logging, paths, seeding)
├── scripts/             # Training and evaluation scripts
├── tests/               # Unit and integration tests
├── artifacts/           # Model checkpoints and outputs
└── data/               # Raw and processed data
```

## Dependencies

**Core:**
- PyTorch ≥2.0.0
- tsfm[notebooks] ≥0.1.0 (IBM TTM)
- scipy ≥1.9.0
- numpy <2.0.0 (compatibility requirement)

**Biosignal:**
- vitaldb ≥1.3.0
- neurokit2 ≥0.2.0

**ML/Eval:**
- scikit-learn ≥1.2.0
- matplotlib ≥3.6.0
- seaborn ≥0.12.0

**Dev Tools (optional):**
```bash
pip install -e .[dev]  # Installs pytest, ruff, black, mypy
```

## Important Files to Read Before Major Changes

1. **SSL Architecture**: `src/ssl/masking.py`, `src/ssl/objectives.py`, `src/ssl/pretrainer.py`
2. **Model Adapters**: `src/models/ttm_adapter.py` - Critical for understanding TTM integration
3. **Data Loading**: `src/data/vitaldb_dataset.py`, `src/data/butppg_dataset.py`
4. **Training Loops**: `src/models/trainers.py` - TrainerClf, TrainerReg
5. **Evaluation**: `src/eval/AUDIT_REPORT.md` - Current limitations and missing features
6. **Recent Fixes**: Check recent commits for critical fixes (e.g., patch_size synchronization)

## Recent Critical Fixes (October 2025)

1. **Patch Size Synchronization** (commit 1ac9549): Recreate MSM mask after TTM adapts patch_size
2. **Subject-Level Validation** (commit 053425b): Fixed train/val mask mismatch in SSL
3. **Best Model Saving** (trainers.py): Always save best_model.pt, not just when save_best=True
4. **Channel Ordering** (butppg_loader.py): Maintain consistent PPG=0, ECG=1 ordering
5. **Stats File Validation** (pretrain_vitaldb_ssl.py): Prevent loading metadata as training data

Always check `git log` before starting work to understand recent changes.
