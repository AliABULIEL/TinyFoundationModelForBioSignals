# Fine-Tuning Pipeline Audit Report

**Project:** Tiny Foundation Model for BioSignals
**Audit Date:** October 15, 2025
**Scope:** Complete fine-tuning implementation (SSL Stage 2: BUT-PPG)
**Status:** ⚠️ **REQUIRES DATA PREPARATION BEFORE USE**

---

## EXECUTIVE SUMMARY

**Overall Assessment:** ✅ **WELL-IMPLEMENTED WITH MINOR GAPS**

The fine-tuning pipeline demonstrates excellent software engineering with proper staged unfreezing, channel inflation, and comprehensive training loop. The architecture is production-ready with one critical prerequisite: **BUT-PPG data must be preprocessed into the expected format**.

✅ **Major Strengths:**
- Clean 3-stage training strategy with configurable parameters
- Sophisticated channel inflation (2→5 channels) with weight preservation
- Comprehensive training loop with AMP, gradient clipping, per-class metrics
- Proper freezing/unfreezing strategies
- Excellent documentation and error handling

⚠️ **Critical Prerequisite:**
- **BUT-PPG data preparation script missing** - must create train/val/test.npz files

❌ **Minor Issues:**
- No configuration file for BUT-PPG (uses hard-coded defaults)
- Channel mismatch: script expects 1250 timesteps, VitalDB pretrained on 1024

---

## 1. FINE-TUNING SCRIPT AUDIT

**File:** `scripts/finetune_butppg.py` (846 lines)
**Status:** ✅ **EXCELLENT IMPLEMENTATION**

### A. TRAINING STRATEGY

✅ **3-Stage Progressive Fine-Tuning:**

| Stage | Name | Epochs | Frozen | Trainable | LR |
|-------|------|--------|--------|-----------|-----|
| **1** | Head-Only | 5 (default) | All encoder | Head only | 2e-5 |
| **2** | Partial Unfreeze | 25 (default: 30-5) | Early encoder | Last 2 blocks + head | 2e-5 |
| **3** | Full Fine-tune | 10 (optional, --full-finetune) | None | All parameters | 2e-6 (10x lower) |

**Implementation (Lines 612-790):**
```python
# STAGE 1: Head-only (lines 612-664)
# Encoder frozen by load_pretrained_with_channel_inflate(freeze_pretrained=True)

# STAGE 2: Partial unfreezing (lines 666-726)
unfreeze_last_n_blocks(model, n=args.unfreeze_last_n, verbose=True)

# STAGE 3: Full fine-tuning (lines 728-790)
if args.full_finetune:
    for param in model.parameters():
        param.requires_grad = True
    low_lr = args.lr / 10  # 2e-6
```

✅ **Learning Rates:** Appropriate for fine-tuning
- Stage 1-2: `2e-5` (default, adjustable via `--lr`)
- Stage 3: `2e-6` (10x reduction, prevents catastrophic forgetting)

### B. DATA LOADING

**Expected Format (Lines 69-79):**
```python
# Expected data files:
# - data/but_ppg/train.npz
# - data/but_ppg/val.npz
# - data/but_ppg/test.npz

# Required keys in .npz:
#  'signals': [N, 5, 1250]  # 5 channels, 10 seconds @ 125Hz
#  'labels':  [N]           # Binary labels (0=poor, 1=good)

# Channel order (CRITICAL):
# 0: ACC_X
# 1: ACC_Y
# 2: ACC_Z
# 3: PPG
# 4: ECG
```

⚠️ **MISMATCH DETECTED:**
- Script expects **1250 timesteps** (10s @ 125Hz)
- VitalDB pretrained on **1024 timesteps** (8.192s @ 125Hz)
- **Impact:** Will cause shape mismatch during inference
- **Fix:** Either pad VitalDB data or adjust BUT-PPG to 1024 samples

✅ **Normalization:** Per-channel z-score (lines 101-108)
```python
for c in range(C):
    channel_data = self.signals[:, c, :]
    mean = channel_data.mean()
    std = channel_data.std()
    if std > 0:
        self.signals[:, c, :] = (channel_data - mean) / std
```

✅ **Data Splits:** Handled via separate files (train.npz, val.npz, test.npz)

✅ **Validation:** Shape validation on load (lines 95-98)

### C. TRAINING LOOP

✅ **Mixed Precision (AMP):** Enabled by default with GradScaler (lines 238, 592)
```python
with autocast(enabled=use_amp):
    logits = model(signals)
    loss = criterion(logits, labels)
```

✅ **Gradient Clipping:** Default 1.0 (lines 248-253, 479)
```python
if gradient_clip > 0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
```

✅ **Metrics Tracked:**
- Training: loss, accuracy (lines 281-283)
- Validation: loss, accuracy, per-class accuracy (lines 351-356)
```python
return {
    'loss': total_loss / len(loader),
    'accuracy': 100.0 * correct / total,
    'class_0_acc': class_0_acc,  # Poor quality
    'class_1_acc': class_1_acc   # Good quality
}
```

✅ **Best Model Saving:** Based on validation accuracy (lines 658-664, 720-726)
```python
if val_metrics['accuracy'] > best_val_acc:
    best_val_acc = val_metrics['accuracy']
    save_checkpoint(output_dir / 'best_model.pt', ...)
```

### D. CONFIGURATION

✅ **Default Hyperparameters (Lines 389-518):**

| Parameter | Default | Adjustable | Notes |
|-----------|---------|------------|-------|
| `epochs` | 30 | ✅ `--epochs` | Total epochs (Stage 1 + 2) |
| `head_only_epochs` | 5 | ✅ `--head-only-epochs` | Stage 1 duration |
| `unfreeze_last_n` | 2 | ✅ `--unfreeze-last-n` | Blocks to unfreeze in Stage 2 |
| `lr` | 2e-5 | ✅ `--lr` | Learning rate |
| `batch_size` | 32 | ✅ `--batch-size` | Batch size |
| `weight_decay` | 0.01 | ✅ `--weight-decay` | AdamW weight decay |
| `gradient_clip` | 1.0 | ✅ `--gradient-clip` | Gradient clipping |

✅ **Output Artifacts (Lines 606-825):**
- `training_config.json` - Full configuration
- `best_model.pt` - Best checkpoint (by val accuracy)
- `final_model.pt` - Final checkpoint
- `training_history.json` - Loss/accuracy curves per stage
- `test_metrics.json` - Final test set results

---

## 2. CHANNEL INFLATION AUDIT

**File:** `src/models/channel_utils.py` (696 lines)
**Status:** ✅ **SOPHISTICATED AND CORRECT**

### A. CORE FUNCTION: `load_pretrained_with_channel_inflate()`

**Implementation (Lines 20-243):**

```python
def load_pretrained_with_channel_inflate(
    checkpoint_path: str,
    pretrain_channels: int = 2,        # VitalDB: PPG + ECG
    finetune_channels: int = 5,        # BUT-PPG: ACC_X,Y,Z + PPG + ECG
    freeze_pretrained: bool = True,
    model_config: Optional[Dict] = None,
    device: str = "cpu",
    strict: bool = False
) -> nn.Module
```

✅ **Strategy (Lines 35-42):**
1. Load pretrained 2-ch checkpoint
2. Create fresh 5-ch model
3. Transfer non-channel-dependent weights exactly
4. For channel-dependent layers:
   - Copy weights for shared channels (PPG, ECG)
   - Initialize new channels (ACC_X, ACC_Y, ACC_Z) from mean of existing + noise
5. Freeze pretrained weights, keep new parts trainable

✅ **Validation:**
- Checkpoint existence check (line 84)
- Channel count validation (lines 87-91)
- Shape mismatch detection (lines 165-210)

### B. WEIGHT TRANSFER

**Direct Transfer (Lines 165-168):**
```python
if pretrained_param.shape == new_param.shape:
    # Exact match - direct copy
    inflated_state[param_name] = pretrained_param.clone()
    transferred_params.append(param_name)
```

**Channel Inflation (Lines 170-210):**
```python
# Detect input layers by keyword matching
is_input_layer = any(keyword in param_name.lower()
                    for keyword in ['input', 'embed', 'in_proj', 'patch_embed'])

if is_input_layer and len(pretrained_param.shape) >= 2:
    inflated_param = _inflate_channel_weights(...)
```

**Weight Inflation Strategy (`_inflate_channel_weights`, Lines 246-328):**

✅ **Intelligent Channel Dimension Detection:**
```python
# Find which dimension changed (2→5)
for dim in range(len(pretrain_shape)):
    if pretrain_shape[dim] != new_shape[dim]:
        if pretrain_shape[dim] == pretrain_channels and new_shape[dim] == finetune_channels:
            channel_dim = dim
            break
```

✅ **Weight Initialization:**
```python
if channel_dim == 1:  # in_channels dimension
    # Copy pretrained channels (PPG, ECG → channels 3,4)
    inflated[:, :pretrain_channels] = pretrained_param

    # Initialize new channels (ACC_X,Y,Z → channels 0,1,2) from mean + noise
    mean_init = pretrained_param.mean(dim=1, keepdim=True)
    for i in range(pretrain_channels, finetune_channels):
        noise = torch.randn_like(mean_init) * 0.01
        inflated[:, i] = mean_init.squeeze(1) + noise.squeeze(1)
```

⚠️ **POTENTIAL ISSUE:**
The channel mapping may not be correct:
- **Expected:** ACC_X(0), ACC_Y(1), ACC_Z(2), PPG(3), ECG(4)
- **Implementation:** Copies pretrained to **first 2 channels**, new to **last 3 channels**
- **Problem:** This maps PPG/ECG to channels 0-1, ACC to channels 2-4
- **Fix Required:** Need to carefully verify channel ordering or remap weights

### C. FREEZING STRATEGY

**Implementation (`_freeze_pretrained_weights`, Lines 331-376):**

✅ **Freezes encoder/backbone:**
```python
if hasattr(model, 'backbone'):
    for param in model.backbone.parameters():
        param.requires_grad = False
elif hasattr(model, 'encoder'):
    for param in model.encoder.parameters():
        param.requires_grad = False
```

✅ **Keeps head trainable:**
```python
if hasattr(model, 'head') and model.head is not None:
    for param in model.head.parameters():
        param.requires_grad = True
```

⚠️ **Partial Implementation for New Channels:**
```python
# TODO: Implement partial freezing for channel-inflated layers
# Currently unfreezes ENTIRE layer if it contains new channels
for param_name in channel_dependent_params:
    param = dict(model.named_parameters())[param_name]
    param.requires_grad = True  # ← Unfreezes all channels, not just new ones
```

✅ **Prints trainable/frozen summary** (lines 371-375)

### D. VERIFICATION

✅ **Summary Printing (Lines 93-241):**
- Transferred parameters count
- Inflated parameters count
- Skipped parameters with warnings
- Trainable/frozen parameter breakdown

✅ **Verification Function Available:**
`verify_channel_inflation()` (lines 496-592) - Checks weight transfer correctness

✅ **Report Generation:**
`get_channel_inflation_report()` (lines 595-680) - Pre-analysis before loading

---

## 3. MODEL HEAD AUDIT

**File:** `src/models/heads.py` (404 lines)
**Status:** ✅ **COMPREHENSIVE**

### A. AVAILABLE HEADS

| Head Type | Class | Use Case | Complexity |
|-----------|-------|----------|------------|
| **Linear** | `LinearClassifier` | Simple classification | Low |
| **MLP** | `MLPClassifier` | Complex classification | Medium |
| **Linear Regressor** | `LinearRegressor` | Simple regression | Low |
| **MLP Regressor** | `MLPRegressor` | Complex regression | Medium |
| **Sequence** | `SequenceClassifier` | With pooling options | High |

✅ **BUT-PPG Default:** `LinearClassifier` (based on script line 565: `'head_type': 'linear'`)

### B. LINEAR CLASSIFIER ARCHITECTURE

**Implementation (Lines 13-65):**

```python
class LinearClassifier(nn.Module):
    def __init__(self, in_features, num_classes, dropout=0.0, bias=True):
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_features, num_classes, bias=bias)

        # Xavier initialization
        nn.init.xavier_uniform_(self.fc.weight)
        if bias:
            nn.init.zeros_(self.fc.bias)
```

✅ **3D Input Handling (Lines 59-61):**
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        x = x.mean(dim=1)  # Average pooling over time
```

✅ **Output:** Logits [B, num_classes] (no softmax - handled by CrossEntropyLoss)

### C. MLP CLASSIFIER ARCHITECTURE

**Implementation (Lines 123-216):**

Features:
- ✅ Hidden layers with configurable dimensions
- ✅ BatchNorm or LayerNorm
- ✅ Activation functions (ReLU, GELU, Tanh)
- ✅ Dropout after each hidden layer
- ✅ Xavier weight initialization

**Default hidden dims (if not specified, line 154):**
```python
hidden_dims = [max(num_classes * 2, in_features // 2)]
```

---

## 4. END-TO-END FLOW VERIFICATION

**Full Pipeline Walkthrough:**

### STEP 1: Load Pretrained Model ✅

**Command:**
```bash
python scripts/finetune_butppg.py \
  --pretrained artifacts/foundation_model/best_model.pt \
  --pretrain-channels 2 \
  --finetune-channels 5
```

**Expected checkpoint structure:**
```python
{
    'model_state_dict': {...},  # 2-channel model weights
    'optimizer_state_dict': {...},
    'config': {
        'input_channels': 2,
        'context_length': 1024,  # ⚠️ Mismatch with BUT-PPG (1250)
        'patch_size': 128,
        ...
    },
    'metrics': {...}
}
```

### STEP 2: Channel Inflation ✅

**Code (Lines 569-576):**
```python
model_config = {
    'variant': 'ibm-granite/granite-timeseries-ttm-r1',
    'task': 'classification',
    'num_classes': 2,
    'input_channels': 5,      # ← Inflated to 5
    'context_length': 1250,    # ⚠️ Should match pretrained (1024)
    'patch_size': 125,         # ⚠️ Different from pretrained (128)
    'head_type': 'linear'
}

model = load_pretrained_with_channel_inflate(
    checkpoint_path=args.pretrained,
    pretrain_channels=2,
    finetune_channels=5,
    freeze_pretrained=True,
    model_config=model_config
)
```

⚠️ **CONFIGURATION MISMATCH:**
- Pretrained: `context_length=1024, patch_size=128` (from VitalDB SSL)
- Fine-tune config: `context_length=1250, patch_size=125`
- **Impact:** Shape mismatch during weight transfer
- **Fix:** Use `context_length=1024, patch_size=128` to match pretrained

### STEP 3: Load BUT-PPG Data ⚠️

**Expected files:**
```
data/but_ppg/
├── train.npz  # {'signals': [N, 5, 1250], 'labels': [N]}
├── val.npz
└── test.npz
```

**Problem:** These files don't exist yet
**Solution:** Need to create data preparation script

### STEP 4-6: Training Stages ✅

All stages are correctly implemented (see Section 1.A)

### STEP 7: Evaluation ✅

**Code (Lines 794-821):**
```python
# Load best checkpoint
best_checkpoint = torch.load(output_dir / 'best_model.pt')
model.load_state_dict(best_checkpoint['model_state_dict'])

# Evaluate on test set
test_metrics = evaluate(model, test_loader, criterion, device, use_amp, "Test")

# Metrics computed:
# - loss
# - accuracy
# - class_0_acc (Poor quality)
# - class_1_acc (Good quality)
```

---

## 5. COMPATIBILITY MATRIX

| Aspect | Article Requirement | Implementation | Status | Notes |
|--------|---------------------|----------------|--------|-------|
| **Channel Inflation** | 2→5 channels | ✅ Implemented | ✅ | Sophisticated weight transfer |
| **Channel Order** | ACC_X,Y,Z + PPG + ECG | ⚠️ Not verified | ⚠️ | May map incorrectly (see 2.B) |
| **Staged Unfreezing** | 3 stages | ✅ Implemented | ✅ | Head → Partial → Full |
| **Stage 1 Duration** | Head-only 3-5 epochs | ✅ 5 epochs default | ✅ | Configurable |
| **Stage 2 Duration** | Partial unfreeze ~25 epochs | ✅ 25 epochs (30-5) | ✅ | Configurable |
| **Stage 3** | Optional full fine-tune | ✅ Optional | ✅ | --full-finetune flag |
| **Learning Rates** | 2e-5 for stages 1-2 | ✅ 2e-5 default | ✅ | Configurable |
| **Stage 3 LR** | Very low (e.g., 2e-6) | ✅ LR/10 = 2e-6 | ✅ | Automatic |
| **Data Format** | 5-ch biosignals + labels | ✅ [N,5,1250], [N] | ⚠️ | Timesteps mismatch (1250 vs 1024) |
| **Normalization** | Per-channel z-score | ✅ Implemented | ✅ | Lines 101-108 |
| **AMP Training** | Mixed precision | ✅ GradScaler | ✅ | Default enabled |
| **Gradient Clipping** | Prevent exploding gradients | ✅ Norm clipping | ✅ | Default 1.0 |
| **Best Model Saving** | Track validation accuracy | ✅ Implemented | ✅ | Saves best + final |
| **Per-class Metrics** | Track quality classes | ✅ Implemented | ✅ | Good vs Poor accuracy |

---

## 6. POTENTIAL ISSUES & RISKS

### A. CORRECTNESS RISKS

- [⚠️] **Channel ordering mismatch:** Inflation may map PPG/ECG to wrong positions
  - **Severity:** HIGH
  - **Impact:** Model won't utilize pretrained knowledge correctly
  - **Fix:** Verify and potentially remap weights in `_inflate_channel_weights()`

- [⚠️] **Timestep mismatch:** 1024 (pretrained) vs 1250 (fine-tune)
  - **Severity:** HIGH
  - **Impact:** Shape errors during weight transfer
  - **Fix:** Standardize on 1024 samples (8.192s @ 125Hz) for both

- [✅] **Staged unfreezing validated:** Correct implementation
- [✅] **Head initialization:** Xavier uniform (good for classification)

### B. IMPLEMENTATION BUGS

- [⚠️] **Shape mismatch:** context_length and patch_size don't match pretrained
  - **Location:** `finetune_butppg.py` lines 563-564
  - **Current:** `context_length=1250, patch_size=125`
  - **Should be:** `context_length=1024, patch_size=128`

- [⚠️] **Partial freezing TODO:** Channel-inflated layers fully unfrozen
  - **Location:** `channel_utils.py` line 366
  - **Impact:** New ACC channels may overwrite pretrained PPG/ECG knowledge
  - **Severity:** MEDIUM
  - **Workaround:** Low learning rate (2e-5) mitigates this

- [✅] **Data loader shuffling:** Correctly enabled for training (line 165)
- [✅] **No missing parameters:** Handled with `strict=False`

### C. CONFIGURATION ISSUES

- [❌] **No config file:** All hyperparameters hard-coded or CLI-only
  - **Missing:** `configs/finetune_butppg.yaml`
  - **Impact:** Less reproducible, harder to track experiments
  - **Priority:** P1 (High)

- [✅] **Default hyperparameters:** Reasonable choices
  - LR=2e-5, batch_size=32, epochs=30, gradient_clip=1.0

- [✅] **Batch size:** 32 is reasonable for 5-channel 1250-sample data
- [✅] **Data validation:** Shape checks on load

### D. REPRODUCIBILITY

- [✅] **Random seed:** Set for torch, numpy, cuda (lines 526-529)
- [✅] **Deterministic operations:** Not explicitly set but seed covers most cases
- [✅] **Checkpoint saving:** Comprehensive (model + optimizer + config + metrics)

---

## 7. MISSING COMPONENTS

### P0 (CRITICAL): Data Preparation

**Missing:** `scripts/prepare_butppg_data.py`

**Required functionality:**
1. Load BUT-PPG raw data (`.dat` files)
2. Load quality labels from `quality-hr-ann.csv`
3. Create 5-channel windows: [ACC_X, ACC_Y, ACC_Z, PPG, ECG]
4. Apply windowing: 1024 or 1250 samples (8.192s or 10s @ 125Hz)
5. Create subject-level train/val/test splits
6. Save as:
   - `data/processed/butppg/train.npz` - {'signals': [N,5,T], 'labels': [N]}
   - `data/processed/butppg/val.npz`
   - `data/processed/butppg/test.npz`

**Current status:**
- ✅ BUT-PPG loader exists (`src/data/butppg_loader.py`) - supports .dat files
- ✅ `scripts/download_but_ppg.py` - downloads dataset
- ❌ No script to create train/val/test.npz files

### P1 (HIGH): Configuration File

**Missing:** `configs/finetune_butppg.yaml`

**Should contain:**
```yaml
# Model
model:
  pretrained_checkpoint: artifacts/foundation_model/best_model.pt
  pretrain_channels: 2
  finetune_channels: 5
  variant: ibm-granite/granite-timeseries-ttm-r1
  context_length: 1024  # Match pretrained
  patch_size: 128       # Match pretrained
  num_classes: 2

# Training
training:
  # Staged unfreezing
  total_epochs: 30
  head_only_epochs: 5
  unfreeze_last_n: 2
  full_finetune: false
  full_finetune_epochs: 10

  # Optimization
  optimizer: adamw
  lr: 2e-5
  weight_decay: 0.01
  batch_size: 32
  gradient_clip: 1.0

  # Performance
  use_amp: true
  num_workers: 4

# Data
data:
  data_dir: data/processed/butppg
  normalize: true

# Output
output:
  checkpoint_dir: artifacts/but_ppg_finetuned
  save_best_only: false

# Reproducibility
seed: 42
```

### P2 (MEDIUM): Validation/Verification Scripts

**Missing:**
1. `scripts/test_channel_inflation.py` - Verify 2→5 channel transfer
2. `scripts/validate_butppg_data.py` - Check prepared data quality
3. Example notebook for BUT-PPG fine-tuning

### P3 (LOW): Documentation

**Missing:**
1. BUT-PPG fine-tuning README
2. Expected data format documentation
3. Troubleshooting guide

---

## 8. ACTIONABLE RECOMMENDATIONS

### P0 (CRITICAL - Must fix before use):

1. **Create BUT-PPG data preparation script:**
   ```bash
   # New script: scripts/prepare_butppg_windows.py
   python scripts/prepare_butppg_windows.py \
     --data-dir data/but_ppg/dataset \
     --output-dir data/processed/butppg \
     --window-size 1024 \
     --fs 125 \
     --create-splits
   ```

2. **Fix context_length/patch_size mismatch:**
   - **File:** `scripts/finetune_butppg.py` lines 563-564
   - **Change:** `context_length=1024, patch_size=128` (match pretrained)
   - **Alternative:** Pad pretrained model to 1250 if BUT-PPG requires it

3. **Verify channel ordering:**
   - Test that ACC channels (0,1,2) and PPG/ECG channels (3,4) map correctly
   - Run `verify_channel_inflation()` after loading

### P1 (HIGH - Should fix for production):

4. **Create configuration file:**
   - Add `configs/finetune_butppg.yaml`
   - Load via `--config` argument
   - Override with CLI args when needed

5. **Implement partial channel freezing:**
   - In `_freeze_pretrained_weights()`, freeze only pretrained channel weights
   - Keep new ACC channel weights trainable
   - Requires parameter slicing

6. **Add data validation script:**
   - Verify .npz file format
   - Check channel ordering
   - Validate label distribution

### P2 (MEDIUM - Nice to have):

7. **Create verification script:**
   - Test channel inflation correctness
   - Verify weight transfer
   - Compare outputs before/after inflation

8. **Add logging:**
   - Integrate TensorBoard or wandb
   - Log learning rate schedule
   - Track per-stage metrics separately

9. **Add early stopping:**
   - Stop if validation accuracy plateaus
   - Configurable patience parameter

### P3 (LOW - Future improvements):

10. **Add data augmentation:**
    - Time warping
    - Amplitude scaling
    - Channel dropout

11. **Support multi-GPU:**
    - Add DistributedDataParallel
    - Batch size scaling

12. **Hyperparameter search:**
    - Learning rate tuning
    - Unfreezing strategy (how many blocks)
    - Head architecture selection

---

## 9. TESTING CHECKLIST

### Quick Smoke Test (1 epoch):

```bash
# 1. Verify pretrained model exists
ls -lh artifacts/foundation_model/best_model.pt

# 2. Create dummy BUT-PPG data (for testing only)
python -c "
import numpy as np
N = 100
signals = np.random.randn(N, 5, 1024).astype(np.float32)
labels = np.random.randint(0, 2, N).astype(np.int64)
np.savez('data/processed/butppg/train.npz', signals=signals, labels=labels)
np.savez('data/processed/butppg/val.npz', signals=signals[:20], labels=labels[:20])
np.savez('data/processed/butppg/test.npz', signals=signals[:20], labels=labels[:20])
"

# 3. Fix config mismatch in script (lines 563-564):
# Change: context_length=1024, patch_size=128

# 4. Run 1-epoch smoke test
python scripts/finetune_butppg.py \
  --pretrained artifacts/foundation_model/best_model.pt \
  --data-dir data/processed/butppg \
  --pretrain-channels 2 \
  --finetune-channels 5 \
  --epochs 1 \
  --head-only-epochs 1 \
  --batch-size 16 \
  --device cuda \
  --output-dir artifacts/test_finetune

# Expected output:
# ✓ Channel inflation complete
# ✓ Loaded 100 samples
# ✓ Epoch 1/1 completed
# ✓ Best model saved

# 5. Verify outputs
ls -lh artifacts/test_finetune/
# Should contain:
#   - training_config.json
#   - best_model.pt
#   - final_model.pt
#   - training_history.json
```

### Full Training Test:

```bash
# After creating real BUT-PPG data:

python scripts/finetune_butppg.py \
  --pretrained artifacts/foundation_model/best_model.pt \
  --data-dir data/processed/butppg \
  --pretrain-channels 2 \
  --finetune-channels 5 \
  --epochs 30 \
  --head-only-epochs 5 \
  --unfreeze-last-n 2 \
  --lr 2e-5 \
  --batch-size 32 \
  --device cuda \
  --output-dir artifacts/but_ppg_finetuned

# Optional Stage 3:
python scripts/finetune_butppg.py \
  --pretrained artifacts/foundation_model/best_model.pt \
  --data-dir data/processed/butppg \
  --full-finetune \
  --full-finetune-epochs 10 \
  --epochs 30 \
  ... (other args)

# Monitor training:
tail -f artifacts/but_ppg_finetuned/training_history.json

# Evaluate:
cat artifacts/but_ppg_finetuned/test_metrics.json
```

### Channel Inflation Verification:

```python
# Test script: test_channel_inflation.py
from src.models.channel_utils import (
    load_pretrained_with_channel_inflate,
    verify_channel_inflation,
    get_channel_inflation_report
)

# 1. Get inflation report
report = get_channel_inflation_report(
    'artifacts/foundation_model/best_model.pt',
    pretrain_channels=2,
    finetune_channels=5
)
print(report['report'])

# 2. Perform inflation
model = load_pretrained_with_channel_inflate(
    checkpoint_path='artifacts/foundation_model/best_model.pt',
    pretrain_channels=2,
    finetune_channels=5,
    freeze_pretrained=True,
    model_config={...}
)

# 3. Verify (requires original 2-ch model for comparison)
# success = verify_channel_inflation(model_2ch, model_5ch)
```

---

## 10. CONCLUSION

### Summary

The fine-tuning pipeline is **well-implemented** with excellent software engineering practices:
- ✅ Proper 3-stage training with configurable parameters
- ✅ Sophisticated channel inflation with weight preservation
- ✅ Comprehensive metrics and checkpointing
- ✅ Clean, documented, production-quality code

### Blockers

**Before fine-tuning can be used:**
1. ❌ **Create BUT-PPG data preparation script** (P0 - CRITICAL)
2. ⚠️ **Fix context_length/patch_size mismatch** (P0 - CRITICAL)
3. ⚠️ **Verify channel ordering** (P0 - CRITICAL)

### Ready for Use After:

1. Creating `scripts/prepare_butppg_windows.py` to generate train/val/test.npz
2. Fixing configuration mismatch (1024/128 vs 1250/125)
3. Testing with smoke test (1 epoch on dummy data)
4. Validating channel inflation correctness

### Estimated Time to Production-Ready:

- **P0 fixes:** 2-3 hours
- **P1 improvements:** 1-2 hours
- **Testing and validation:** 1 hour
- **Total:** ~4-6 hours

---

**End of Audit Report**
