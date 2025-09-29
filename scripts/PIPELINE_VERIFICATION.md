# TTM VitalDB Pipeline - Complete Verification ✅

## Script: ttm_vitaldb_fixed.py

### ✅ All Commands Implemented

1. **prepare-splits** ✅
   - Creates train/val/test splits from VitalDB cases
   - Supports fasttrack (70 cases) and full modes
   - Handles different case sets (bis, desflurane, etc.)
   - Saves to JSON format

2. **build-windows** ✅ 
   - Loads VitalDB signals with proper track names
   - Applies evidence-based filtering
   - Performs quality checks (SQI)
   - Creates normalized windows
   - Saves train stats for consistent normalization

3. **train** ✅
   - Loads preprocessed windows
   - Creates TTM model with proper dimensions
   - Supports FastTrack (frozen encoder) mode
   - Trains with configurable parameters
   - Saves best model checkpoint

4. **test** ✅ (NOW COMPLETE)
   - Loads test data and model checkpoint
   - Runs evaluation on test set
   - Calculates metrics (accuracy, F1, AUC, etc.)
   - Includes calibration metrics (ECE)
   - Saves results to JSON

5. **inspect** ✅
   - Inspects data files (.npz)
   - Inspects model checkpoints
   - Shows shapes and keys

### ✅ Critical Fixes Applied

1. **Configuration Handling** ✅
   ```python
   # Handles nested 'channels:' structure
   if 'channels' in channels_config:
       channels_dict = channels_config['channels']
   ```

2. **Filter Type Mapping** ✅
   ```python
   filter_type_map = {
       'butterworth': 'butter',
       'chebyshev2': 'cheby2',
       'cheby2': 'cheby2',
       'butter': 'butter'
   }
   ```

3. **Parameter Name Flexibility** ✅
   ```python
   # Handles both naming conventions
   lowcut = filt.get('lowcut', filt.get('low_freq', 0.5))
   highcut = filt.get('highcut', filt.get('high_freq', 10))
   ```

4. **VitalDB Track Fallback** ✅
   ```python
   track_mapping = {
       'PPG': 'PLETH',
       'ECG': 'ECG_II',
       'ABP': 'ABP',
       'EEG': 'EEG1'
   }
   vitaldb_track = ch_config.get('vitaldb_track', 
                                  track_mapping.get(channel_name.upper(), 'PLETH'))
   ```

5. **Robust Data Path Handling** ✅
   ```python
   # Tries multiple locations
   train_file = Path(args.outdir) / 'train_windows.npz'
   if not train_file.exists():
       train_file = Path(args.outdir) / 'train' / 'train_windows.npz'
   ```

### ✅ Features Working

- VitalDB case loading with alternating signal fix
- Bandpass filtering (Butterworth & Chebyshev II)
- PPG peak detection with NeuroKit2 fallback
- Signal quality assessment (SQI)
- Window creation with cardiac cycle validation
- Z-score normalization with saved statistics
- TTM model creation with proper dimensions
- Classification and regression support
- Calibration metrics (ECE)
- Comprehensive error handling and logging

### 📋 Complete Pipeline Commands

```bash
# Step 1: Create splits
python scripts/ttm_vitaldb_fixed.py prepare-splits \
    --mode fasttrack \
    --output artifacts

# Step 2: Build train windows
python scripts/ttm_vitaldb_fixed.py build-windows \
    --channels-yaml configs/channels.yaml \
    --windows-yaml configs/windows.yaml \
    --split-file artifacts/splits_fasttrack.json \
    --split train \
    --outdir artifacts/raw_windows/train \
    --channel PPG \
    --duration-sec 60 \
    --min-sqi 0.5

# Step 3: Build test windows
python scripts/ttm_vitaldb_fixed.py build-windows \
    --channels-yaml configs/channels.yaml \
    --windows-yaml configs/windows.yaml \
    --split-file artifacts/splits_fasttrack.json \
    --split test \
    --outdir artifacts/raw_windows/test \
    --channel PPG \
    --duration-sec 60 \
    --min-sqi 0.5

# Step 4: Train model
python scripts/ttm_vitaldb_fixed.py train \
    --model-yaml configs/model.yaml \
    --run-yaml configs/run.yaml \
    --split-file artifacts/splits_fasttrack.json \
    --outdir artifacts/raw_windows \
    --out artifacts/checkpoints \
    --fasttrack

# Step 5: Test model
python scripts/ttm_vitaldb_fixed.py test \
    --ckpt artifacts/checkpoints/best_model.pt \
    --model-yaml configs/model.yaml \
    --run-yaml configs/run.yaml \
    --split-file artifacts/splits_fasttrack.json \
    --outdir artifacts/raw_windows \
    --out artifacts/results

# Optional: Inspect results
python scripts/ttm_vitaldb_fixed.py inspect \
    --data artifacts/raw_windows/train/train_windows.npz \
    --model artifacts/checkpoints/best_model.pt
```

### 🎯 Expected Outputs

1. **Splits**: `artifacts/splits_fasttrack.json`
2. **Train Windows**: `artifacts/raw_windows/train/train_windows.npz`
3. **Test Windows**: `artifacts/raw_windows/test/test_windows.npz` 
4. **Train Stats**: `artifacts/raw_windows/train/train_stats.npz`
5. **Model Checkpoint**: `artifacts/checkpoints/best_model.pt`
6. **Test Results**: `artifacts/results/test_results.json`

### ✅ Verification Complete

The script `ttm_vitaldb_fixed.py` now contains:
- All 5 commands fully implemented
- All critical fixes for configuration handling
- Complete error handling and logging
- Support for the entire pipeline from data prep to evaluation

**The pipeline is ready to run end-to-end!** 🚀
