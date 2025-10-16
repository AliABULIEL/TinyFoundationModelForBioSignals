# BUT-PPG Dataset Pipeline

**Complete pipeline for downloading, processing, and preparing the BUT-PPG v2.0.0 dataset for downstream evaluation.**

---

## Overview

The BUT-PPG (Brno University of Technology - Photoplethysmography) dataset is a PhysioNet database containing multi-modal biosignal recordings for clinical task evaluation. This pipeline automates the complete process from raw data download to evaluation-ready format.

**Dataset Details:**
- **Source**: PhysioNet (https://physionet.org/content/butppg/2.0.0/)
- **Subjects**: 50 (IDs: 100-149)
- **Total Recordings**: ~3,888 recordings
- **Signals**: PPG, ECG, 3-axis Accelerometer (ACC_X, ACC_Y, ACC_Z)
- **Sampling Rates**: Variable (resampled to 125Hz)
- **Tasks**: Signal Quality, Heart Rate Estimation, Motion Classification

---

## Quick Start

### One-Command Setup

```bash
bash scripts/setup_butppg_complete.sh
```

This will:
1. Install dependencies
2. Download raw BUT-PPG data from PhysioNet
3. Process clinical data for all 3 tasks
4. Verify data structure

**Expected Runtime**: 30-60 minutes (depending on network speed)

---

## Installation

### Required Dependencies

```bash
pip install wfdb pandas scipy numpy tqdm
```

**Package Descriptions:**
- `wfdb`: PhysioNet WFDB format reader/writer
- `pandas`: CSV annotation loading
- `scipy`: Signal processing (filtering, resampling)
- `numpy`: Numerical operations
- `tqdm`: Progress bars

---

## Pipeline Components

### 1. Dataset Downloader (`download_butppg_dataset.py`)

Downloads raw BUT-PPG v2.0.0 dataset from PhysioNet.

**Usage:**

```bash
python scripts/download_butppg_dataset.py \
    --output-dir data/but_ppg/raw \
    --method zip \
    --skip-if-exists
```

**Arguments:**
- `--output-dir`: Output directory for raw data (default: `data/but_ppg/raw`)
- `--method`: Download method ('zip' - downloads complete 86.7MB archive)
- `--skip-if-exists`: Skip downloading if ZIP file already exists
- `--keep-zip`: Keep ZIP file after extraction (default: delete after extraction)

**What it Downloads:**
- Complete dataset ZIP (86.7 MB)
- 3,888 recordings from 50 subjects
- Each recording has separate files for PPG, ECG, ACC signals
- Annotation files:
  - `quality-hr-ann.csv` - Quality labels + reference heart rates (combined)
  - `subject-info.csv` - Subject demographics + motion type labels

**Output Structure:**

```
data/but_ppg/raw/
├── but-ppg-an-annotated-photoplethysmography-dataset-2.0.0/
│   ├── 100001_PPG.dat, 100001_PPG.hea
│   ├── 100001_ECG.dat, 100001_ECG.hea
│   ├── 100001_ACC.dat, 100001_ACC.hea  (from 112001 onwards)
│   ├── 100002_PPG.dat, 100002_PPG.hea
│   └── ... (3,888 recordings)
├── annotations/
│   ├── quality-hr-ann.csv
│   └── subject-info.csv
└── README.txt  (dataset info)
```

**Note:** Record IDs are 6-digit format (e.g., 100001, 112001), where:
- First 3 digits: Subject identifier
- Last 3 digits: Measurement number for that subject

---

### 2. Clinical Data Processor (`process_butppg_clinical.py`)

Processes raw signals into task-specific datasets with proper preprocessing.

**Usage:**

```bash
python scripts/process_butppg_clinical.py \
    --raw-dir data/but_ppg/raw/but-ppg-an-annotated-photoplethysmography-dataset-2.0.0 \
    --annotations-dir data/but_ppg/raw/annotations \
    --output-dir data/processed/butppg \
    --target-fs 125 \
    --window-size 1024 \
    --tasks quality,heart_rate,motion
```

**Arguments:**
- `--raw-dir`: Directory with raw BUT-PPG recordings (contains `*_PPG.dat` files)
- `--annotations-dir`: Directory with annotation CSV files
- `--output-dir`: Output directory for processed data
- `--target-fs`: Target sampling frequency in Hz (default: 125)
- `--window-size`: Window size in samples (default: 1024)
- `--tasks`: Tasks to process (comma-separated)

**Signal Preprocessing:**

1. **Bandpass Filtering** (signal-specific):
   - PPG: 0.5-8.0 Hz (captures cardiac frequency range)
   - ECG: 0.5-40.0 Hz (captures QRS complex)
   - ACC: 0.5-20.0 Hz (captures motion artifacts)

2. **Resampling**:
   - All signals resampled to 125 Hz (uniform sampling rate)

3. **Window Extraction**:
   - Fixed-size windows of 1024 samples (8.192 seconds @ 125Hz)
   - Zero-padding if too short

4. **Normalization**:
   - Z-score normalization: `(x - mean) / (std + 1e-8)`

5. **Channel Mapping**:
   - Channel 0: ACC_X
   - Channel 1: ACC_Y
   - Channel 2: ACC_Z
   - Channel 3: PPG
   - Channel 4: ECG

**Output Format:**

Each task generates 3 files (train/val/test) with:
- `signals`: `[N, 5, 1024]` numpy array (5 channels, 1024 timesteps)
- `labels`: `[N]` numpy array (task-specific)

**Data Splitting:**
- **Subject-level split**: 70% train, 15% val, 15% test
- **Seed**: 42 (reproducible)
- **Critical**: Subject-level splitting prevents data leakage (samples from same subject don't appear in multiple splits)

---

### 3. Complete Pipeline Script (`setup_butppg_complete.sh`)

Orchestrates the complete pipeline from download to verification.

**Usage:**

```bash
bash scripts/setup_butppg_complete.sh
```

**Pipeline Steps:**
1. Install dependencies (`pip install wfdb pandas scipy numpy tqdm`)
2. Download raw data (`download_butppg_dataset.py`)
3. Process clinical data (`process_butppg_clinical.py`)
4. Verify data structure

**Final Data Structure:**

```
data/processed/butppg/
├── quality/
│   ├── train.npz  # [N, 5, 1024] signals, [N] binary labels
│   ├── val.npz
│   └── test.npz
├── heart_rate/
│   ├── train.npz  # [N, 5, 1024] signals, [N] HR (bpm) labels
│   ├── val.npz
│   └── test.npz
└── motion/
    ├── train.npz  # [N, 5, 1024] signals, [N] 8-class labels
    ├── val.npz
    └── test.npz
```

---

## Clinical Tasks

### 1. Signal Quality Classification

**Task**: Binary classification of PPG signal quality
- **Classes**: 0 = Poor quality, 1 = Good quality
- **Labels**: Expert consensus from `PPGQualityLabels.csv`
- **Evaluation**: AUROC, AUPRC, Accuracy, F1-score
- **Clinical Relevance**: Quality assessment for automated monitoring systems

**Expected Performance:**
- Baseline: ~75-80% accuracy
- SOTA: ~90-95% accuracy

---

### 2. Heart Rate Estimation

**Task**: Regression of heart rate from PPG/ECG signals
- **Target**: Heart rate in beats per minute (BPM)
- **Labels**: Reference values from `HRReference.csv`
- **Evaluation**: MAE, RMSE, Pearson correlation
- **Clinical Relevance**: Continuous heart rate monitoring

**Expected Performance:**
- Baseline: MAE ~5-8 BPM
- SOTA: MAE ~2-4 BPM

---

### 3. Motion Classification

**Task**: 8-class classification of subject motion type
- **Classes**:
  0. Sitting
  1. Standing
  2. Walking (slow)
  3. Walking (normal)
  4. Walking (fast)
  5. Running
  6. Cycling
  7. Hand movement
- **Labels**: Motion annotations from `MotionLabels.csv`
- **Evaluation**: Accuracy, macro-F1, confusion matrix
- **Clinical Relevance**: Context-aware signal interpretation

**Expected Performance:**
- Baseline: ~60-70% accuracy
- SOTA: ~85-90% accuracy

---

## Integration with Downstream Evaluation

After running the pipeline, use the processed data for downstream evaluation:

```bash
python scripts/run_downstream_evaluation.py \
    --butppg-checkpoint artifacts/but_ppg_finetuned/best_model.pt \
    --butppg-data data/processed/butppg \
    --output-dir artifacts/butppg_evaluation \
    --batch-size 32
```

**Requirements:**
1. Fine-tuned model checkpoint (from `finetune_butppg.py`)
2. Processed BUT-PPG data (from this pipeline)
3. Model checkpoint must match input format `[N, 5, 1024]`

---

## Troubleshooting

### Issue: Download Fails with 404 Error

**Cause**: Network issues or PhysioNet unavailable

**Solution**:
```bash
# Method 1: Retry with skip-if-exists
python scripts/download_butppg_dataset.py \
    --output-dir data/but_ppg/raw \
    --skip-if-exists

# Method 2: Manual download with wget
wget -O data/but_ppg/raw/but-ppg-2.0.0.zip \
  https://physionet.org/static/published-projects/butppg/but-ppg-an-annotated-photoplethysmography-dataset-2.0.0.zip

# Then extract manually
cd data/but_ppg/raw
unzip but-ppg-2.0.0.zip
```

### Issue: Missing Annotation Files

**Cause**: Extraction failed or incomplete download

**Solution**:
1. Check if ZIP file is complete (should be 86.7 MB)
2. Re-extract: `unzip -o data/but_ppg/raw/but-ppg-2.0.0.zip`
3. Annotation files should be in extracted directory:
   - `quality-hr-ann.csv`
   - `subject-info.csv`
4. Processor will copy them to `data/but_ppg/raw/annotations/`

### Issue: No Samples Collected for Task

**Cause**: Label column name mismatch or missing annotation files

**Solution**:
1. Check annotation files exist:
   ```bash
   ls -lh data/but_ppg/raw/annotations/
   # Should show: quality-hr-ann.csv, subject-info.csv
   ```

2. Check CSV structure:
   ```bash
   head -n 3 data/but_ppg/raw/annotations/quality-hr-ann.csv
   ```

3. The processor automatically detects common column name variations:
   - ID columns: `id`, `record_id`, `signal_id`, `signalid`
   - Quality: `quality`, `ppg_quality`, `ppgquality`
   - HR: `hr`, `heart_rate`, `heartrate`, `reference_hr`
   - Motion: `motion`, `motion_type`, `motiontype`, `activity`

4. If columns still not detected, check processor output for:
   ```
   ✓ Loaded quality-hr-ann.csv: X entries
     Columns: ['actual', 'column', 'names']
   ```

### Issue: Processing Takes Too Long

**Cause**: Large number of recordings (3,888 total)

**Solution**:
1. Processing all 3,888 recordings takes ~10-15 minutes
2. To test with subset, manually filter recordings:
   ```bash
   # Process only recordings from subject 100 (first 3 digits)
   cd data/but_ppg/raw/but-ppg-an-annotated-photoplethysmography-dataset-2.0.0
   mkdir -p ../test_subset
   cp 100*_PPG.* ../test_subset/
   cp 100*_ECG.* ../test_subset/
   cp 100*_ACC.* ../test_subset/

   # Process subset
   python scripts/process_butppg_clinical.py \
       --raw-dir data/but_ppg/raw/test_subset \
       --annotations-dir data/but_ppg/raw/annotations \
       --output-dir data/processed/butppg_test
   ```

### Issue: Dimension Mismatch During Evaluation

**Cause**: Model trained with different input shape

**Solution**:
- Ensure fine-tuning used same `--window-size` and `--target-fs`
- Check model config: `input_channels=5`, `context_length=1024`

---

## Data Format Verification

### Quick Check Script

```python
import numpy as np

# Load processed data
data = np.load('data/processed/butppg/quality/train.npz')
signals = data['signals']
labels = data['labels']

print(f"Signals shape: {signals.shape}")  # Expected: [N, 5, 1024]
print(f"Labels shape: {labels.shape}")    # Expected: [N]
print(f"Signal range: [{signals.min():.2f}, {signals.max():.2f}]")
print(f"Label values: {np.unique(labels)}")
```

**Expected Output:**
```
Signals shape: (1234, 5, 1024)
Labels shape: (1234,)
Signal range: [-5.23, 4.87]
Label values: [0 1]
```

---

## Citation

If you use the BUT-PPG dataset, please cite:

```bibtex
@article{nemcova2021monitoring,
  title={Monitoring of heart rate, blood oxygen saturation, and blood pressure using a smartphone},
  author={Nemcova, Andrea and Jordanova, Iren and Varecka, Martin and Smisek, Radovan and Marsanova, Lucie and Smital, Lukas and Vitek, Martin},
  journal={Biomedical Signal Processing and Control},
  volume={59},
  pages={101928},
  year={2020},
  publisher={Elsevier}
}
```

**PhysioNet Citation:**
```
Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000).
PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals.
Circulation [Online]. 101 (23), pp. e215–e220.
```

---

## Additional Resources

- **BUT-PPG Dataset**: https://physionet.org/content/butppg/2.0.0/
- **WFDB Documentation**: https://wfdb.readthedocs.io/
- **Signal Processing Guide**: See `src/data/filters.py` for filter implementations
- **Evaluation Metrics**: See `src/eval/metrics.py` for metric definitions

---

## Support

For issues with:
- **Pipeline scripts**: Open GitHub issue in repository
- **PhysioNet access**: Contact PhysioNet support
- **Data format questions**: See `src/data/butppg_dataset.py` implementation

---

**Last Updated**: October 2025
