# Quick End-to-End Pipeline Test

## Overview
This test runs the complete TTM × VitalDB pipeline with minimal data (2 cases) to verify everything works correctly. Designed to complete in **5-10 minutes on CPU**.

## What It Tests

### 1. Data Loading & Preprocessing
- ✅ VitalDB API connection
- ✅ Signal loading (PPG/PLETH)
- ✅ Bandpass filtering (Chebyshev Type-II)
- ✅ Peak detection (Elgendi algorithm)
- ✅ Quality assessment (SQI computation)

### 2. Window Building
- ✅ 10-second window creation
- ✅ Minimum 3 cycles per window
- ✅ Z-score normalization
- ✅ Train/test statistics computation

### 3. Model Training
- ✅ TTM model loading (real IBM model)
- ✅ Frozen encoder + trainable head
- ✅ 2 epochs of training
- ✅ Loss computation and backpropagation
- ✅ Checkpoint saving

### 4. Evaluation
- ✅ Inference on test set
- ✅ Metrics computation (Acc, Prec, Recall, F1, AUC)
- ✅ Results saving

## Usage

### Option 1: Shell Script (Recommended)
```bash
cd /Users/aliab/Desktop/TinyFoundationModelForBioSignals
bash scripts/run_quick_test.sh
```

### Option 2: Direct Python
```bash
cd /Users/aliab/Desktop/TinyFoundationModelForBioSignals
python3 scripts/quick_e2e_test.py
```

## Output

The test creates a timestamped directory with all outputs:
```
test_run_YYYYMMDD_HHMMSS/
├── splits.json              # Train/test case IDs
├── train_windows.npz        # Preprocessed training windows
├── test_windows.npz         # Preprocessed test windows
├── model.pt                 # Trained model checkpoint
├── test_results.json        # Evaluation metrics
└── pipeline_test.log        # Detailed logs
```

## Expected Runtime

| Step | Time | Description |
|------|------|-------------|
| 1. Splits | 5-10s | Fetch case IDs |
| 2. Windows | 2-3 min | Load, filter, detect, QC |
| 3. Train | 2-4 min | 2 epochs, CPU |
| 4. Test | 30-60s | Inference + metrics |
| **Total** | **5-8 min** | End-to-end |

## Configuration

The test uses minimal configurations optimized for speed:

**Data**:
- 2 VitalDB cases (1 train, 1 test)
- 30 seconds of signal per case
- 10-second windows with 10-second stride

**Model**:
- Real IBM TTM (805K params)
- Frozen encoder
- Linear classification head (trainable)

**Training**:
- 2 epochs only
- Batch size: 4
- Learning rate: 1e-3
- CPU only (no AMP)

## Success Criteria

The test passes if:
1. ✅ Both train and test windows are created
2. ✅ Model trains without errors
3. ✅ Loss decreases during training
4. ✅ Test metrics are computed
5. ✅ All files are saved correctly

## Troubleshooting

### "Failed to load signal"
- VitalDB API might be down
- Try running again in a few minutes
- Check internet connection

### "No valid windows created"
- Signal quality might be too low
- The test will automatically try another case
- This is normal behavior

### "Model loading failed"
- First run downloads TTM from HuggingFace (~50MB)
- Requires internet connection
- Downloads are cached for future runs

### Import errors
```bash
pip install -r requirements.txt
```

## Next Steps

After successful test:

1. **Run FastTrack mode** (50 train + 20 test cases, ~3 hours):
   ```bash
   python scripts/ttm_vitaldb.py prepare-splits --mode fasttrack
   python scripts/ttm_vitaldb.py build-windows --mode fasttrack
   python scripts/ttm_vitaldb.py train --mode fasttrack
   ```

2. **Run full training** (all cases, higher accuracy):
   ```bash
   bash scripts/run_high_accuracy.sh
   ```

3. **Customize configuration**:
   - Edit `configs/model.yaml` for model settings
   - Edit `configs/run.yaml` for training settings
   - Edit `configs/channels.yaml` for signal processing

## Technical Details

### Signal Processing
- **PPG**: Chebyshev Type-II filter (0.5-10 Hz, order 4)
- **Sampling**: Resampled to 25 Hz
- **Peak Detection**: Elgendi algorithm via NeuroKit2
- **Quality**: SQI threshold = 0.5

### Normalization
- Method: Z-score using train statistics
- Applied per-sample across time dimension
- Train mean/std computed and frozen

### Model Architecture
```
Input: [batch, channels, timesteps]
  ↓
TTM Encoder (frozen, 805K params)
  ↓
[batch, 3, 8, 192] → mean pool → [batch, 192]
  ↓
Linear Head (trainable)
  ↓
Output: [batch, num_classes]
```

## Logs

Detailed logs saved to `test_run_*/pipeline_test.log`:
```bash
# View logs
cat test_run_*/pipeline_test.log

# Follow logs in real-time
tail -f test_run_*/pipeline_test.log
```

## Verification

After running, verify outputs:
```bash
# Check results
cat test_run_*/test_results.json | python -m json.tool

# Check window shapes
python3 -c "
import numpy as np
train = np.load('test_run_*/train_windows.npz')
test = np.load('test_run_*/test_windows.npz')
print(f'Train: {train[\"windows\"].shape}')
print(f'Test: {test[\"windows\"].shape}')
"

# Check model
python3 -c "
import torch
ckpt = torch.load('test_run_*/model.pt', map_location='cpu')
print(f'Epoch: {ckpt[\"epoch\"]}')
print(f'Best val loss: {ckpt[\"best_val_loss\"]:.4f}')
"
```
