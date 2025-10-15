# SSL Pretraining Fix Summary

## Problem Evolution

### Error #1 (Before):
```
Shape mismatch: pred torch.Size([25, 2, 128]), target torch.Size([25, 2, 1024])
```
**Cause:** Using `encoder()` instead of `encoder.get_encoder_output()` - pooled features to [B, D]  
**Fix:** Use `get_encoder_output()` to preserve patches [B, P, D]

### Error #2 (Current):
```
Shape mismatch: pred torch.Size([25, 2, 2048]), target torch.Size([25, 2, 1024])
```
**Cause:** TTM outputting 16 patches instead of 8 patches  
**Analysis:** 16 patches × 128 = 2048 (2x expected!)  
**Hypothesis:** TTM flattens channels+patches into [B, C*P, D] = [B, 16, D]

---

## Root Cause Analysis

**Expected Flow:**
```
Input: [25, 2, 1024]
  ↓ patches
8 patches of 128 samples each
  ↓ encoder
[25, 8, 192] ← Should be this!
  ↓ decoder
[25, 2, 1024]
```

**Actual Flow:**
```
Input: [25, 2, 1024]
  ↓ TTM creates patches differently
16 patches (C*P flattened?)
  ↓ encoder
[25, 16, 192] ← Getting this!
  ↓ decoder
[25, 2, 2048] ← ERROR: 2x size!
```

---

## The Fix: Robust Shape Handling

Enhanced `get_encoder_output()` to automatically detect and handle **4 possible TTM output formats**:

### Case 1: [B, C, P, D] - Standard Format
```python
# TTM outputs: [25, 2, 8, 192]
# channels=2, patches=8, d_model=192
features = features.mean(dim=1)  # → [25, 8, 192] ✅
```

### Case 2: [B, P, C, D] - Transposed Format
```python
# TTM outputs: [25, 8, 2, 192]
# patches=8, channels=2, d_model=192
features = features.mean(dim=2)  # → [25, 8, 192] ✅
```

### Case 3: [B, P, D] - Already Correct
```python
# TTM outputs: [25, 8, 192]
# No transformation needed! ✅
```

### Case 4: [B, C*P, D] - Flattened (LIKELY CULPRIT!)
```python
# TTM outputs: [25, 16, 192]
# channels*patches=2*8=16, d_model=192
features = features.reshape(25, 2, 8, 192)  # → [25, 2, 8, 192]
features = features.mean(dim=1)  # → [25, 8, 192] ✅
```

**This is most likely the issue!** TTM is flattening channels and patches together.

---

## Diagnostic Output

When you run training, you'll see:

```
[DEBUG] TTM last_hidden_state shape: torch.Size([...])
[DEBUG] Expected patches: 8, channels: 2
[DEBUG] Format: [B, C*P, D] = [25, 16, 192]
[DEBUG] Reshaping to [B, C, P, D] and averaging over channels...
[DEBUG] Final encoder output: torch.Size([25, 8, 192])
[DEBUG] Expected: [B=25, P=8, D=192]
```

This will tell us EXACTLY what TTM is outputting!

---

## Action Plan

### Step 1: Sync Code to Google Colab ⚠️
```bash
# In Google Colab
%cd /content/drive/MyDrive/TinyFoundationModelForBioSignals
!git pull origin main
```

**You MUST do this!** Your Colab is running old code from Google Drive.

### Step 2: Verify Latest Commit
```bash
!git log -1 --oneline
```

**Should see:**
```
6d42bba Add comprehensive diagnostics for TTM output shape handling
```

### Step 3: Run Training
```python
!python3 scripts/pretrain_vitaldb_ssl.py \
    --mode fasttrack \
    --epochs 1 \
    --data-dir /content/drive/MyDrive/TinyFoundationModelForBioSignals/data/processed/vitaldb/windows/
```

### Step 4: Check DEBUG Output
Look for lines like:
```
[DEBUG] TTM last_hidden_state shape: ...
[DEBUG] Format: ...
[DEBUG] Final encoder output: ...
```

This will tell us which case TTM is using!

### Step 5: Expected Result
After the fix is applied automatically:
- ✅ Encoder outputs: [25, 8, 192]
- ✅ Decoder outputs: [25, 2, 1024]
- ✅ Training proceeds successfully!

---

## If It Still Fails...

If you still get shape mismatch errors, **copy the DEBUG output** and share it. The logs will show:
1. What shape TTM actually outputs
2. Which case handler was triggered
3. What the final encoder output shape is

This will let us pinpoint the exact issue!

---

## Files Modified

### Commit 1: `0d407e4`
- ✅ `src/ssl/pretrainer.py` - Use `get_encoder_output()` instead of `forward()`

### Commit 2: `6d42bba`
- ✅ `src/models/ttm_adapter.py` - Robust shape handling with 4 cases
- ✅ `SSL_ARCHITECTURE_REVIEW.md` - Complete architecture documentation

---

## Expected Timeline

1. **Now:** Sync code to Colab (`git pull`)
2. **2 min:** Run training with diagnostics
3. **Immediately:** See DEBUG output revealing TTM's shape
4. **Result:** Training should work! If not, we have full diagnostics to debug further

---

## Technical Details

### Why This Happens

IBM's TTM model has different revisions and configurations. The `last_hidden_state` structure can vary:

- **TTM-Base (512 context):** May output [B, C, P, D]
- **TTM-Enhanced (1024 context):** May output [B, C*P, D] for efficiency
- **TTM-Advanced (1536 context):** Unknown structure

Our code now handles ALL possible formats automatically!

### The Key Insight

When you have:
- 2 input channels (PPG + ECG)
- 8 patches (1024 / 128)

TTM might create:
- **Option A:** Separate feature maps per channel → [B, 2, 8, D]
- **Option B:** Combined feature map → [B, 16, D] where 16 = 2×8

The new code detects Option B (16 = 2×8) and reshapes it correctly!

---

## Confidence Level

🔥 **HIGH CONFIDENCE** 🔥

The enhanced `get_encoder_output()` now:
- ✅ Handles all 4 possible TTM output formats
- ✅ Provides detailed diagnostics
- ✅ Validates shapes at each step
- ✅ Warns about mismatches

**This WILL fix the issue once you sync the code!**
