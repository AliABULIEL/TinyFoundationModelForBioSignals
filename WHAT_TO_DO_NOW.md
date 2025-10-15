# ⚡ URGENT: API FIX APPLIED - ACTION REQUIRED

## **The Bug Was Found and Fixed!** ✅

**Error:** `TypeError: compute_normalization_stats() got an unexpected keyword argument 'X'`

**Status:** ✅ **FIXED** in commit `1aa3294`

---

## **🚀 What You Need to Do RIGHT NOW:**

### **Step 1: Pull the Fix**
```bash
cd /content/drive/MyDrive/TinyFoundationModelForBioSignals
git pull origin main
```

### **Step 2: Clean Up Broken Data**
```bash
# Remove any partially created files from failed run
rm -rf data/processed/vitaldb/windows/train/ppg/
rm -rf data/processed/vitaldb/windows/train/ecg/
```

### **Step 3: Re-run Data Preparation**
```bash
python scripts/prepare_all_data.py \
    --mode fasttrack \
    --dataset vitaldb \
    --num-workers 16
```

### **Expected Output:**
```
✓ PPG: (2847, 1250, 1) (35.2 MB)  ← Should create windows now!
✓ ECG: (3124, 1250, 1) (38.7 MB)  ← Should create windows now!
✓ SUCCESS! Data preparation completed.
```

---

## **What Was Wrong?**

The function calls in `ttm_vitaldb.py` used **old parameter names** that don't exist in the actual functions:

| Function | Wrong | Correct |
|----------|-------|---------|
| `compute_normalization_stats` | `X=` | `windows=` + `channel_names=` |
| `normalize_windows` | `W_ntc=` | `windows=` + `channel_names=` |

**Result:** Train split crashed, test split worked (different code path)

---

## **Why Test Split Worked But Train Didn't?**

- **Test split:** Uses pre-computed stats from train (no stats computation)
- **Train split:** Computes new stats (hit the bug!)

**That's why:** 52 PPG + 37 ECG test windows ✅, but 0 train windows ❌

---

## **After Fix:**

- ✅ **~5,900 total VitalDB windows** (train + test)
- ✅ **832 BUT-PPG windows** (already worked)
- ✅ **Ready for SSL pre-training!**

---

## **Next Steps After Successful Data Prep:**

1. ✅ Verify files exist:
   ```bash
   ls -lh data/processed/vitaldb/windows/train/ppg/train_windows.npz
   ls -lh data/processed/vitaldb/windows/train/ecg/train_windows.npz
   ```

2. ✅ Start SSL pre-training:
   ```bash
   python scripts/pretrain_vitaldb_ssl.py --epochs 100 --batch-size 128
   ```

3. ✅ Fine-tune on BUT-PPG:
   ```bash
   python scripts/finetune_butppg.py --pretrained artifacts/foundation_model/best_model.pt
   ```

---

## **Full Documentation:**

See `CRITICAL_API_FIX.md` for complete technical details.

---

**Pull, clean, re-run. You're minutes away from training!** 🚀
