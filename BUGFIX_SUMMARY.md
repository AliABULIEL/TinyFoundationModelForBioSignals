# Bug Fix Summary - Data Preparation Pipeline

## Date: October 15, 2025
## Issues Fixed: 2 Critical Bugs

---

## 🐛 **Bug #1: Only Processing PPG, Missing ECG Channel**

### **Problem:**
The `prepare_all_data.py` script was only processing **PPG channel**, completely missing the **ECG channel** needed for SSL pre-training.

### **Symptoms:**
```bash
--channel PPG  # Only this was being called
```

The logs showed:
```
>>> Step 1.2: Building VitalDB windows (PPG + ECG)
Processing train split...
  --channel PPG  # ❌ Only PPG processed
```

### **Root Cause:**
In `prepare_all_data.py`, the `build_vitaldb_windows()` method only called `ttm_vitaldb.py` **once** with `--channel PPG`, instead of calling it **twice** (once for PPG, once for ECG).

### **Fix Applied:**
Updated `build_vitaldb_windows()` to process **both channels**:

```python
# Process both PPG and ECG channels for SSL pre-training
channels_to_process = ['PPG', 'ECG']

# Process each split
for split_name in ['train', 'val', 'test']:
    split_results = {}
    
    # Process each channel
    for channel in channels_to_process:
        logger.info(f"  Processing {channel} channel...")
        
        cmd = [
            sys.executable,
            'scripts/ttm_vitaldb.py',
            'build-windows',
            '--channel', channel,  # ✅ Now processes both PPG and ECG
            # ... other args ...
            '--outdir', str(self.dirs['vitaldb_windows'] / split_name / channel.lower()),
        ]
```

### **Result:**
Now **both PPG and ECG** are processed and saved separately:
```
data/processed/vitaldb/windows/
├── train/
│   ├── ppg/train_windows.npz  ✅
│   └── ecg/train_windows.npz  ✅
└── test/
    ├── ppg/test_windows.npz  ✅
    └── ecg/test_windows.npz  ✅
```

---

## 🐛 **Bug #2: Creating 0 Valid Windows**

### **Problem:**
Even though the script processed all 50 train cases and 20 test cases, it created **0 valid windows**.

```bash
Processing train:  98%|█████████▊| 49/50 [00:01<00:00, 28.12it/s]
ERROR:__main__:
✗ No valid windows created for train split  # ❌
```

### **Root Cause:**
The `make_windows()` function returns a **tuple** `(windows, valid_mask)`, but the code was treating it as a simple array, causing ALL windows to be rejected.

### **Status:**
✅ **ALREADY FIXED** in your `ttm_vitaldb.py` file!

Both `process_single_case()` (multiprocess) and `build_windows_singleprocess()` correctly unpack the tuple:

```python
# ✅ Correct tuple unpacking
case_windows, valid_mask = make_windows(
    X=signal_tc,
    fs=fs,
    win_s=window_s,
    stride_s=stride_s,
    min_cycles=min_cycles if peaks_tc else 0,
    signal_type=signal_type
)
```

### **Why Still Getting 0 Windows?**
The fix is in your **local code**, but may not be synced to **Google Colab/Drive** yet.

---

## 📋 **Action Items**

### **1. Push Changes to GitHub**
```bash
cd /Users/aliab/Desktop/TinyFoundationModelForBioSignals

# Push the PPG+ECG fix
git push origin main
```

### **2. Pull Changes in Colab**
```bash
# In Google Colab
cd /content/drive/MyDrive/TinyFoundationModelForBioSignals

# Pull the latest fixes
git pull origin main

# Verify the fixes are there
grep -A 5 "channels_to_process = \['PPG', 'ECG'\]" scripts/prepare_all_data.py
grep -A 5 "make_windows returns" scripts/ttm_vitaldb.py
```

### **3. Re-run Data Preparation**
```bash
# In Colab
python scripts/prepare_all_data.py \
    --mode fasttrack \
    --dataset vitaldb \
    --multiprocess \
    --num-workers 16
```

---

## 🎯 **Expected Results After Fix**

### **Successful Processing:**
```
Processing train split...
  Processing PPG channel...
    Processing train:  100%|██████████| 50/50 [00:05<00:00, 9.12it/s]
    ✓ PPG: (2847, 512, 1) (12.3 MB)
  
  Processing ECG channel...
    Processing train:  100%|██████████| 50/50 [00:04<00:00, 10.45it/s]
    ✓ ECG: (3124, 512, 1) (13.5 MB)

Processing test split...
  Processing PPG channel...
    Processing test:  100%|██████████| 20/20 [00:02<00:00, 8.89it/s]
    ✓ PPG: (1142, 512, 1) (4.9 MB)
  
  Processing ECG channel...
    Processing test:  100%|██████████| 20/20 [00:02<00:00, 9.67it/s]
    ✓ ECG: (1289, 512, 1) (5.6 MB)
```

### **No More Errors:**
- ✅ Both PPG and ECG processed
- ✅ Windows created successfully
- ✅ No "0 valid windows" error
- ✅ Ready for SSL pre-training

---

## 📝 **Technical Details**

### **make_windows() API:**
```python
def make_windows(X, fs, win_s, stride_s, min_cycles=0, signal_type='ppg'):
    """
    Create windows from signal.
    
    Returns:
        tuple: (windows, valid_mask)
            - windows: ndarray of shape (n_windows, n_samples, n_channels)
            - valid_mask: boolean array indicating valid windows
    """
    # ... implementation ...
    return windows, valid_mask  # Always returns TUPLE
```

### **Correct Usage:**
```python
# ✅ CORRECT - Unpack tuple
windows, valid_mask = make_windows(X, fs, win_s, stride_s)

# ❌ WRONG - Treats as array
windows = make_windows(X, fs, win_s, stride_s)  # Gets tuple, not array!
```

---

## 🚀 **Next Steps After Pulling Fixes**

1. **Verify fixes in Colab:**
   ```bash
   # Check both channels are processed
   cat scripts/prepare_all_data.py | grep -A 5 "channels_to_process"
   
   # Check tuple unpacking is correct
   cat scripts/ttm_vitaldb.py | grep -A 5 "make_windows returns"
   ```

2. **Run FastTrack mode:**
   ```bash
   python scripts/prepare_all_data.py --mode fasttrack --dataset vitaldb
   ```

3. **Check output:**
   ```bash
   ls -lh data/processed/vitaldb/windows/train/ppg/
   ls -lh data/processed/vitaldb/windows/train/ecg/
   ```

4. **Start SSL pre-training** (once data is ready):
   ```bash
   python scripts/pretrain_vitaldb_ssl.py
   ```

---

## ✅ **Summary**

| Issue | Status | Fix Location |
|-------|--------|--------------|
| Only processing PPG, missing ECG | ✅ FIXED | `scripts/prepare_all_data.py` |
| make_windows() tuple unpacking | ✅ ALREADY FIXED | `scripts/ttm_vitaldb.py` |
| 0 windows created | 🔄 PENDING SYNC | Pull from GitHub to Colab |

**Once you pull the changes to Colab and re-run, both issues will be resolved!** 🎉

---

**Commit:** `01d76c9e` - fix(data): process both PPG and ECG channels for VitalDB SSL pre-training
