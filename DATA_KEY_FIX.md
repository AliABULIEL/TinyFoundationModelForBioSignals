# Data Key Inconsistency Fix

**Date:** October 16, 2025
**Status:** ✅ FIXED

---

## Issue

The BUT-PPG data preparation scripts create `.npz` files with the key `'data'`, but the fine-tuning script expected the key `'signals'`. This caused data loading to fail with:

```
KeyError: 'signals'
```

---

## Root Cause

**Data Preparation** (`prepare_all_data.py`):
```python
np.savez(output_file, data=signals, labels=labels)  # Uses 'data' key
```

**Fine-Tuning Script** (`finetune_butppg.py`):
```python
self.signals = torch.from_numpy(data['signals']).float()  # Expected 'signals' key
```

---

## Solution

Updated both scripts to support **both** key names (`'signals'` and `'data'`):

### 1. Fine-Tuning Script (`scripts/finetune_butppg.py`)

**BUTPPGDataset class** (lines 90-96):
```python
# Support both 'signals' and 'data' keys (data prep inconsistency)
if 'signals' in data:
    signals_array = data['signals']
elif 'data' in data:
    signals_array = data['data']
else:
    raise KeyError(f"Expected 'signals' or 'data' key in {data_file}, found: {list(data.keys())}")

self.signals = torch.from_numpy(signals_array).float()  # [N, 5, 1024]
```

**verify_data_structure function** (lines 482-488):
```python
# Support both 'signals' and 'data' keys
if 'signals' in data:
    signals = data['signals']
elif 'data' in data:
    signals = data['data']
else:
    signals = None
```

### 2. Verification Script (`scripts/check_butppg_data.py`)

**check_data function** (lines 30-42):
```python
# Support both 'signals' and 'data' keys
if 'signals' in data:
    signals = data['signals']
elif 'data' in data:
    signals = data['data']
else:
    print(f"  ❌ Missing signal data key (expected 'signals' or 'data')")
    return False

if 'labels' not in data:
    print(f"  ❌ Missing 'labels' key")
    return False
```

---

## Verification

### Test with 'data' key files:

```bash
$ python3 scripts/check_butppg_data.py --data-dir data/processed/butppg/windows

======================================================================
BUT-PPG DATA COMPATIBILITY CHECK
======================================================================

Found 3 .npz files:
  - train/test_windows.npz
  - train/train_windows.npz
  - train/val_windows.npz

Checking: data/processed/butppg/windows/train/test_windows.npz
  ✓ File loads successfully
  Keys: ['data', 'labels']
  Signals shape: (50, 5, 1024)
    N=50 samples, C=5 channels, T=1024 timesteps
  ✅ All checks passed!

Checking: data/processed/butppg/windows/train/train_windows.npz
  ✓ File loads successfully
  Keys: ['data', 'labels']
  Signals shape: (100, 5, 1024)
    N=100 samples, C=5 channels, T=1024 timesteps
  ✅ All checks passed!

Checking: data/processed/butppg/windows/train/val_windows.npz
  ✓ File loads successfully
  Keys: ['data', 'labels']
  Signals shape: (50, 5, 1024)
    N=50 samples, C=5 channels, T=1024 timesteps
  ✅ All checks passed!

======================================================================
SUMMARY
======================================================================
Passed: 3/3 ✅
```

### Test with --verify-data option:

```bash
$ python3 scripts/finetune_butppg.py --verify-data

======================================================================
VERIFYING BUT-PPG DATA STRUCTURE
======================================================================
Data directory: data/processed/butppg/windows
Exists: True

📁 Directory contents:
  📄 train/test_windows.npz (0.98 MB)
  📄 train/train_windows.npz (1.95 MB)
  📄 train/val_windows.npz (0.98 MB)

🔍 Inspecting .npz files:

  train/test_windows.npz:
    Keys: ['data', 'labels']
    Signals shape: (50, 5, 1024)
      N=50 samples, C=5 channels, T=1024 timesteps
      Status: ✅ 5 channels, ✅ 1024 timesteps
    Labels: 50 samples, unique values: [0 1]

  train/train_windows.npz:
    Keys: ['data', 'labels']
    Signals shape: (100, 5, 1024)
      N=100 samples, C=5 channels, T=1024 timesteps
      Status: ✅ 5 channels, ✅ 1024 timesteps
    Labels: 100 samples, unique values: [0 1]

  train/val_windows.npz:
    Keys: ['data', 'labels']
    Signals shape: (50, 5, 1024)
      N=50 samples, C=5 channels, T=1024 timesteps
      Status: ✅ 5 channels, ✅ 1024 timesteps
    Labels: 50 samples, unique values: [0 1]

======================================================================
✓ Verification complete
======================================================================
```

---

## Backward Compatibility

✅ Scripts now support **both** formats:

### Format 1: Original (with 'signals' key)
```python
np.savez('train.npz', signals=signals, labels=labels)
```

### Format 2: Current (with 'data' key)
```python
np.savez('train.npz', data=signals, labels=labels)
```

Both formats will load correctly!

---

## Files Modified

1. **scripts/finetune_butppg.py**
   - Line 90-96: `BUTPPGDataset.__init__()` - Flexible key detection
   - Line 482-488: `verify_data_structure()` - Support both keys
   - Line 666-669: Added early check for `--verify-data` flag

2. **scripts/check_butppg_data.py**
   - Line 30-42: `check_data()` - Flexible key detection

---

## Impact

✅ **No breaking changes** - Old files with 'signals' key still work
✅ **New files supported** - Files with 'data' key now work
✅ **Better error messages** - Clear indication of which keys were found
✅ **Verification tools** - Both scripts can validate data format

---

## Testing Checklist

- [x] Files with 'data' key load correctly
- [x] Files with 'signals' key still work (backward compatible)
- [x] Verification script detects both formats
- [x] Fine-tuning script loads both formats
- [x] --verify-data option works without loading model
- [x] Error messages are clear when wrong keys present

---

## Next Steps for Users

**On Colab:**

After BUT-PPG data preparation completes, you should now be able to run fine-tuning directly:

```bash
# Verify data format
python3 scripts/check_butppg_data.py --data-dir data/processed/butppg/windows

# Or use built-in verification
python3 scripts/finetune_butppg.py --verify-data

# If all looks good, start fine-tuning
python3 scripts/finetune_butppg.py \
  --pretrained artifacts/foundation_model/best_model.pt \
  --epochs 1 \
  --head-only-epochs 1 \
  --batch-size 32 \
  --output-dir artifacts/butppg_test
```

**Locally:**

The fix is already in place and tested with sample data. Both key formats are now supported.

---

## Related Documentation

- `FINETUNE_SCRIPT_FIX.md` - Complete fine-tuning script fixes
- `CHANNEL_INFLATION_FIX.md` - Channel inflation mapping fixes
- `DOWNSTREAM_EVALUATION_GUIDE.md` - Evaluation system documentation

---

**Status:** ✅ Issue resolved, scripts support both data formats
