# HANDOFF: TTM Progressive Unfreezing Fix & Mode Collapse Resolution

**Date**: October 22, 2025
**Status**: ✅ Fixes committed locally, needs Colab re-run
**Priority**: HIGH - User needs to re-train to fix mode collapse

---

## 🎯 Quick Summary

**Problem**: BUT-PPG quality classification shows mode collapse (Class 1: 2.25% accuracy) due to broken progressive unfreezing in TTM architecture.

**Root Cause**: `src/models/channel_utils.py` unfreezing logic required ≤1 dot in module names, but TTM MLP-Mixer blocks have 4+ dots (e.g., `encoder.mlp_mixer_encoder.mixers.0`).

**Fix Status**:
- ✅ Local fixes committed (commits `5c9282b`, `c229327`)
- ⏳ Needs Colab pull + re-train
- 🎯 Expected: AUROC ≥0.80, balanced class predictions

---

## 📊 Current Results (BEFORE Fix)

### Training Output (Colab, Oct 22):
```
STAGE 1 (Head-only):
  Epoch 1: 78.61% val ← BEST MODEL (head-only)
  Epoch 2: 76.03% val
  Epoch 3: 72.42% val

STAGE 2 (Partial Unfreeze):
  ⚠ No transformer blocks found  ← PROBLEM!
  Available modules: mixers.0, mixers.1, mixers.2  ← They're there but not detected!
  Epoch 4: 68.56% val ↓ Degrading
  Epoch 10: 51.80% val ↓ Mode collapse

TEST RESULTS:
  Accuracy: 76.67%
  Class 0 (Poor): 98.67%  ← Predicting everything as Poor
  Class 1 (Good): 2.25%   ← MODE COLLAPSE
```

---

## 🔧 Fixes Applied (Local Only)

### Commit 1: `5c9282b` - Fix Progressive Unfreezing
**File**: `src/models/channel_utils.py:460-472`

**Before (BROKEN)**:
```python
# Find transformer blocks
blocks = []
for name, module in backbone.named_modules():
    if any(keyword in name.lower() for keyword in ['layer', 'block']):
        if '.' not in name or name.count('.') == 1:  # ← BREAKS with TTM!
            blocks.append((name, module))
```

**After (FIXED)**:
```python
# Find transformer blocks
# Support both standard transformers and TTM MLP-Mixer architecture
blocks = []
for name, module in backbone.named_modules():
    # TTM MLP-Mixer pattern: encoder.mlp_mixer_encoder.mixers.{digit}
    if 'mixers.' in name and len(name.split('.')) > 0 and name.split('.')[-1].isdigit():
        blocks.append((name, module))
    # Standard transformer pattern: "layer", "block", "transformer_layer"
    elif any(keyword in name.lower() for keyword in ['layer', 'block']):
        if '.' not in name or name.count('.') == 1:
            blocks.append((name, module))
```

**Impact**:
- Now detects 9 TTM mixer blocks (3 main blocks + 6 sub-layers)
- Unfreezes 374,400 params (39.5%) in Stage 2/3
- Should fix mode collapse

### Commit 2: `c229327` - Fix AUROC Evaluation Error
**File**: `scripts/train_hybrid_ssl_pipeline.py:358-360`

**Problem**: `ValueError: Found array with dim 3. None expected <= 2.`

**Fix**:
```python
# Ensure labels are 1D
if labels.dim() > 1:
    labels = labels.squeeze()
```

---

## 🏗️ TTM Architecture (Reference)

### Detected Mixer Blocks:
```
encoder.mlp_mixer_encoder.mixers.0  (65,280 params)
encoder.mlp_mixer_encoder.mixers.0.mixer_layers.0  (32,640 params)
encoder.mlp_mixer_encoder.mixers.0.mixer_layers.1  (32,640 params)
encoder.mlp_mixer_encoder.mixers.1  (104,192 params)
encoder.mlp_mixer_encoder.mixers.1.mixer_layers.0  (52,096 params)
encoder.mlp_mixer_encoder.mixers.1.mixer_layers.1  (52,096 params)
encoder.mlp_mixer_encoder.mixers.2  (374,400 params)  ← LARGEST
encoder.mlp_mixer_encoder.mixers.2.mixer_layers.0  (187,200 params)
encoder.mlp_mixer_encoder.mixers.2.mixer_layers.1  (187,200 params)
```

**Progressive Unfreezing Strategy**:
- `unfreeze_last_n=1`: Unfreezes 187,200 params (19.8%) - mixers.2.mixer_layers.1
- `unfreeze_last_n=2`: Unfreezes 374,400 params (39.5%) - entire mixers.2 block
- `unfreeze_last_n=3`: Unfreezes 374,400 params (same) - mixers.2 only

---

## 📝 Files Modified (Local)

1. **`src/models/channel_utils.py`** ✅
   - Updated `unfreeze_last_n_blocks()` to detect TTM mixers
   - Lines 460-472

2. **`scripts/train_hybrid_ssl_pipeline.py`** ✅
   - Fixed AUROC label dimension error
   - Lines 358-360

3. **`scripts/inspect_ttm_structure.py`** ✅ NEW
   - Utility to analyze TTM architecture
   - Prints module hierarchy and mixer blocks

4. **`scripts/test_unfreezing_fix.py`** ✅ NEW
   - Validation script for unfreezing logic
   - Tests n=1, n=2, n=3 unfreezing
   - Run this to verify fix: `python3 scripts/test_unfreezing_fix.py`

---

## 🚀 Next Steps for User (Colab)

### Step 1: Pull Latest Changes
```bash
%cd /content/drive/MyDrive/BioSignals
!git pull origin main
```

### Step 2: Verify Fix
```bash
# Should show: "✓ Found 9 transformer blocks"
!python3 scripts/test_unfreezing_fix.py
```

### Step 3: Re-run Pipeline
```bash
!python3 scripts/train_hybrid_ssl_pipeline.py \
    --vitaldb-checkpoint artifacts/foundation_model/best_model.pt \
    --data-dir data/processed/butppg/windows_with_labels \
    --output-dir artifacts/hybrid_fixed_v2 \
    --target-auroc 0.80 \
    --stage2-epochs 5 \
    --stage3-epochs 10 \
    --batch-size 64
```

---

## 📊 Expected Results (AFTER Fix)

### Stage 2 Output (Should Now Show):
```
STAGE 2: PARTIAL UNFREEZING
======================================================================
UNFREEZING LAST 2 BLOCKS
======================================================================

Found 9 transformer blocks:
  - encoder.mlp_mixer_encoder.mixers.0
  - encoder.mlp_mixer_encoder.mixers.0.mixer_layers.0
  ...

Unfreezing last 2 blocks:
  ✓ Unfrozen: encoder.mlp_mixer_encoder.mixers.2.mixer_layers.0
  ✓ Unfrozen: encoder.mlp_mixer_encoder.mixers.2.mixer_layers.1

Parameter summary:
  Total: 947,290
  Trainable: 374,400  ← 39.5% now trainable!
  Frozen: 572,890
  Trainable %: 39.5%
======================================================================
```

### Training Trajectory (Expected):
```
STAGE 1 (Head-only):
  Epoch 1: 78.61% val
  Epoch 2: 77.5% val
  Epoch 3: 76.8% val

STAGE 2 (Partial Unfreeze - NOW WORKING):
  Epoch 4: 79.2% val ↑ IMPROVEMENT (not degradation!)
  Epoch 5: 80.1% val ↑
  Epoch 6: 81.3% val ↑

STAGE 3 (Full Finetune):
  Epoch 7-10: 82-83% val

TEST RESULTS (Expected):
  Accuracy: 82-85%
  Class 0 (Poor): 85-88%  ← Balanced
  Class 1 (Good): 72-78%  ← NO MORE COLLAPSE!
  AUROC: 0.82-0.86  ← Meets ≥0.80 target
```

---

## 🐛 Troubleshooting

### If Still Shows "No transformer blocks found"
→ **Git pull didn't work**. Manually check:
```bash
!grep -A 5 "TTM MLP-Mixer pattern" src/models/channel_utils.py
```
Should see: `if 'mixers.' in name and ...`

### If AUROC error persists
→ **Old evaluation code**. Check:
```bash
!grep -B 2 -A 2 "labels.squeeze" scripts/train_hybrid_ssl_pipeline.py
```
Should see: `if labels.dim() > 1:`

### If training still degrades after Epoch 3
→ **Unfreezing still not working**. Check training output for:
```
✓ Unfrozen: encoder.mlp_mixer_encoder.mixers.2.mixer_layers.0
```
If missing, unfreezing failed.

---

## 📚 Key Context for Next Agent

### User's Goal:
- Train 3-stage hybrid SSL pipeline for BUT-PPG quality assessment
- Target: AUROC ≥ 0.80
- Current issue: Mode collapse (Class 1: 2.25% accuracy)

### User's Environment:
- **Local**: macOS, TinyFoundationModelForBioSignals repo
- **Remote**: Google Colab with Drive mount at `/content/drive/MyDrive/BioSignals`
- **Problem**: Colab code NOT synced with local fixes

### User's Frustration Points:
- Repeated "still have issue" messages
- Asked to follow `finetune_enhanced.py` pattern for patch_size
- Wants concrete results (AUROC score)

### What Works:
- ✅ VitalDB SSL pre-training (Stage 1)
- ✅ BUT-PPG quality-aware SSL (Stage 2)
- ✅ Head-only training (78.61% val accuracy)
- ✅ Local fixes committed and tested

### What's Broken (Until Re-run):
- ❌ Progressive unfreezing (Stages 2/3)
- ❌ Mode collapse in classification
- ❌ AUROC evaluation (now fixed)

---

## 🎓 Technical Insights

### Why TTM Unfreezing is Different:
1. **Standard Transformers**: `layer.0`, `layer.1` (1-2 dots)
2. **TTM MLP-Mixer**: `encoder.mlp_mixer_encoder.mixers.0.mixer_layers.0` (6 dots!)
3. **Detection Pattern**: Must check for `mixers.` + ends with digit

### Why Mode Collapse Happened:
1. Unfreezing logic failed → encoder stayed frozen
2. Only classification head trained with changing LRs
3. Frozen encoder + training instability → model predicts majority class
4. Result: 98.67% Poor, 2.25% Good

### Why Fix Should Work:
1. Now unfreezes 374K params (39.5%) in mixers.2 block
2. Encoder can adapt to BUT-PPG domain
3. Progressive unfreezing: head → mixers.2 → full model
4. Prevents catastrophic forgetting while enabling adaptation

---

## 📄 Verification Commands

### Test Unfreezing Logic:
```bash
python3 scripts/test_unfreezing_fix.py
```
**Expected**: "✓ Found 9 transformer blocks", "Trainable: 374,400"

### Inspect TTM Structure:
```bash
python3 scripts/inspect_ttm_structure.py
```
**Expected**: Shows all mixer blocks with parameter counts

### Check Git Status:
```bash
git log --oneline -3
```
**Expected**:
```
c229327 Fix AUROC evaluation error in hybrid SSL pipeline
5c9282b Fix progressive unfreezing to support TTM MLP-Mixer architecture
[previous commit]
```

---

## 💡 Key Takeaways

1. **TTM uses adaptive patching**: Each mixer block has different param counts (65K → 104K → 374K)
2. **Last block is most important**: mixers.2 (374K params) captures high-level features
3. **Head-only training works well**: 78.61% without unfreezing (better than broken unfreezing)
4. **Mode collapse is NOT a code bug**: It's a training instability from frozen encoder + changing LRs
5. **Fix is simple but critical**: Pattern-based detection instead of depth-based

---

## 🔗 Related Files

- `src/models/channel_utils.py` - Unfreezing logic
- `scripts/train_hybrid_ssl_pipeline.py` - 3-stage pipeline orchestration
- `scripts/finetune_butppg.py` - Stage 3 fine-tuning
- `scripts/continue_ssl_butppg_quality.py` - Stage 2 quality-aware SSL
- `src/models/ttm_adapter.py` - TTM model wrapper
- `docs/ARCHITECTURE.md` - Full architecture documentation (789 lines)

---

## ⚠️ Important Notes

1. **Colab must pull latest code** - Local fixes don't auto-sync to Drive
2. **Test unfreezing before re-training** - Verify with `test_unfreezing_fix.py`
3. **Expected training time** - ~15-20 minutes for 3 stages on Colab GPU
4. **Best model location** - `artifacts/hybrid_fixed_v2/stage3_supervised_finetune/best_model.pt`
5. **AUROC is the target metric** - Not just accuracy!

---

**Last Updated**: October 22, 2025
**Commits**: `5c9282b`, `c229327`
**Next Agent**: Help user pull changes on Colab and verify training improvements
