#!/usr/bin/env python3
"""
Quick End-to-End Pipeline Test (5-10 minutes, CPU only)
Tests the entire flow with 2 VitalDB cases:
1. Prepare splits
2. Build windows (load, filter, detect, QC)
3. Train model (2 epochs)
4. Test/evaluate
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import torch
import yaml
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.data.splits import create_patient_splits
from src.data.vitaldb_loader import VitalDBLoader, load_channel
from src.data.windows import WindowBuilder, WindowConfig
from src.data.filters import apply_bandpass_filter
from src.data.detect import detect_peaks
from src.data.quality import compute_sqi
from src.models.datasets import TTMDataset
from src.models.ttm_adapter import create_ttm_model
from src.models.trainers import TrainerClf
from src.eval.metrics import compute_classification_metrics
from src.utils.seed import set_seed

# Setup
set_seed(42)
device = torch.device('cpu')  # Force CPU
print("="*70)
print("QUICK END-TO-END PIPELINE TEST (2 CASES, CPU ONLY)")
print("="*70)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Device: {device}")
print()

# Create test directory
test_dir = Path(__file__).parent.parent / f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
test_dir.mkdir(exist_ok=True)
print(f"Test directory: {test_dir}")
print()

# Setup logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(test_dir / 'pipeline_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

#=============================================================================
# STEP 1: PREPARE SPLITS (2 cases)
#=============================================================================
print("="*70)
print("STEP 1: PREPARE SPLITS")
print("="*70)

try:
    # Get 2 test cases from VitalDB
    loader = VitalDBLoader()
    logger.info("Fetching available VitalDB cases...")
    
    # Try to get high-quality cases (BIS > 70)
    all_cases = loader.get_available_case_sets()
    if 'bis' in all_cases and len(all_cases['bis']) >= 2:
        test_cases = sorted(list(all_cases['bis']))[:2]
        logger.info(f"Using BIS cases: {test_cases}")
    else:
        # Fallback: just use first 2 available cases
        all_ids = loader.get_available_cases()
        test_cases = all_ids[:2]
        logger.info(f"Using first available cases: {test_cases}")
    
    # Create minimal splits (1 train, 1 test)
    splits = {
        'train': [test_cases[0]],
        'val': [],  # Skip validation for speed
        'test': [test_cases[1]]
    }
    
    splits_file = test_dir / 'splits.json'
    with open(splits_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"✓ Splits created: {splits}")
    print(f"✓ Saved to: {splits_file}")
    print()
    
except Exception as e:
    logger.error(f"Failed to prepare splits: {e}")
    raise

#=============================================================================
# STEP 2: BUILD WINDOWS (Load → Filter → Detect → QC)
#=============================================================================
print("="*70)
print("STEP 2: BUILD WINDOWS FROM VITALDB")
print("="*70)

# Window configuration (minimal for speed)
window_config = WindowConfig(
    window_length_sec=10,
    stride_sec=10,
    min_cycles=3,
    fs_target={'ppg': 25, 'ecg': 125, 'abp': 100},
    normalize_method='zscore'
)

# Channel configuration
channels_config = {
    'ppg': {
        'vitaldb_track': 'PLETH',
        'filter': {'type': 'cheby2', 'order': 4, 'lowcut': 0.5, 'highcut': 10},
        'fs_original': 100
    }
}

windows_data = {'train': [], 'test': []}
train_stats = {'mean': None, 'std': None}

for split_name, case_ids in splits.items():
    if not case_ids:
        continue
    
    print(f"\nProcessing {split_name} split ({len(case_ids)} case)...")
    
    for case_id in case_ids:
        print(f"  Case {case_id}:")
        
        try:
            # 1. Load signal
            print(f"    - Loading PPG signal...")
            signal, fs = load_channel(
                case_id=case_id,
                track_name='PLETH',
                duration_sec=30  # Just 30 seconds for speed
            )
            
            if signal is None or len(signal) < 100:
                print(f"    ✗ Failed to load signal")
                continue
            
            print(f"      Loaded {len(signal)} samples at {fs} Hz")
            
            # 2. Filter
            print(f"    - Applying bandpass filter...")
            filtered = apply_bandpass_filter(
                signal, fs,
                lowcut=0.5,
                highcut=10,
                filter_type='cheby2',
                order=4
            )
            
            # 3. Detect peaks
            print(f"    - Detecting peaks...")
            peaks = detect_peaks(filtered, fs, modality='ppg')
            print(f"      Found {len(peaks)} peaks")
            
            # 4. Quality check
            print(f"    - Computing SQI...")
            sqi = compute_sqi(filtered, fs, peaks=peaks, modality='ppg')
            print(f"      SQI: {sqi:.3f}")
            
            if sqi < 0.5:
                print(f"    ✗ Low quality (SQI < 0.5)")
                continue
            
            # 5. Create windows
            print(f"    - Creating windows...")
            builder = WindowBuilder(window_config)
            case_windows = builder.create_windows(
                signals={'ppg': filtered},
                fs={'ppg': fs},
                peaks={'ppg': peaks}
            )
            
            if case_windows is None or len(case_windows) == 0:
                print(f"    ✗ No valid windows created")
                continue
            
            print(f"      Created {len(case_windows)} windows")
            
            # 6. Compute/apply normalization
            if split_name == 'train':
                # Compute train statistics
                all_windows = np.concatenate(case_windows, axis=0)
                train_stats['mean'] = np.mean(all_windows)
                train_stats['std'] = np.std(all_windows)
                print(f"      Train stats: mean={train_stats['mean']:.3f}, std={train_stats['std']:.3f}")
            
            # Apply normalization
            normalized_windows = []
            for w in case_windows:
                w_norm = (w - train_stats['mean']) / (train_stats['std'] + 1e-8)
                normalized_windows.append(w_norm)
            
            # Store windows
            windows_data[split_name].extend(normalized_windows)
            print(f"    ✓ Processed successfully")
            
        except Exception as e:
            print(f"    ✗ Error processing case {case_id}: {e}")
            logger.exception(f"Error processing case {case_id}")
            continue

# Save windows
for split_name, windows_list in windows_data.items():
    if windows_list:
        windows_array = np.array(windows_list)
        out_file = test_dir / f'{split_name}_windows.npz'
        np.savez_compressed(
            out_file,
            windows=windows_array,
            labels=np.random.randint(0, 2, len(windows_array))  # Dummy labels
        )
        print(f"\n✓ Saved {split_name}: {windows_array.shape} to {out_file}")

if not windows_data['train'] or not windows_data['test']:
    raise RuntimeError("Failed to create windows for train or test split")

print()

#=============================================================================
# STEP 3: TRAIN MODEL (2 epochs only)
#=============================================================================
print("="*70)
print("STEP 3: TRAIN MODEL (2 EPOCHS)")
print("="*70)

# Create datasets
train_data = np.load(test_dir / 'train_windows.npz')
test_data = np.load(test_dir / 'test_windows.npz')

train_dataset = TTMDataset(
    windows=train_data['windows'],
    labels=train_data['labels'],
    task='classification'
)

test_dataset = TTMDataset(
    windows=test_data['windows'],
    labels=test_data['labels'],
    task='classification'
)

print(f"Train dataset: {len(train_dataset)} samples")
print(f"Test dataset: {len(test_dataset)} samples")
print()

# Create model
print("Creating TTM model...")
model_config = {
    'variant': 'ibm-granite/granite-timeseries-ttm-r1',
    'task': 'classification',
    'num_classes': 2,
    'input_channels': 1,
    'context_length': train_data['windows'].shape[-1],
    'freeze_encoder': True,
    'head_type': 'linear'
}

model = create_ttm_model(model_config)
model = model.to(device)

print(f"Model created: {sum(p.numel() for p in model.parameters())} parameters")
print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")
print()

# Create trainer
train_config = {
    'num_epochs': 2,  # Just 2 epochs for speed
    'batch_size': 4,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'use_amp': False,  # No AMP on CPU
    'grad_clip': 1.0,
    'patience': 10,
    'num_workers': 0,
    'device': 'cpu'
}

trainer = TrainerClf(
    model=model,
    train_dataset=train_dataset,
    val_dataset=None,  # Skip validation for speed
    config=train_config,
    save_dir=test_dir
)

# Train
print("Training...")
train_metrics = trainer.train()

print("\nTraining completed!")
print(f"Final loss: {train_metrics['train_loss'][-1]:.4f}")
print()

#=============================================================================
# STEP 4: TEST/EVALUATE
#=============================================================================
print("="*70)
print("STEP 4: TEST AND EVALUATE")
print("="*70)

# Load best model
checkpoint_path = test_dir / 'model.pt'
if checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded checkpoint from {checkpoint_path}")
else:
    print("⚠ No checkpoint found, using current model state")

model.eval()

# Evaluate on test set
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

all_preds = []
all_labels = []
all_probs = []

print("\nRunning inference on test set...")
with torch.no_grad():
    for batch in test_loader:
        X, y = batch
        X, y = X.to(device), y.to(device)
        
        outputs = model(X)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())

all_probs = np.concatenate(all_probs, axis=0)
all_preds = np.concatenate(all_preds, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# Compute metrics
metrics = compute_classification_metrics(
    y_true=all_labels,
    y_pred=all_preds,
    y_prob=all_probs[:, 1]
)

print("\nTest Results:")
print(f"  Accuracy:  {metrics['accuracy']:.3f}")
print(f"  Precision: {metrics['precision']:.3f}")
print(f"  Recall:    {metrics['recall']:.3f}")
print(f"  F1 Score:  {metrics['f1']:.3f}")
print(f"  AUC-ROC:   {metrics['auroc']:.3f}")
print()

# Save results
results = {
    'test_metrics': metrics,
    'train_metrics': train_metrics,
    'model_config': model_config,
    'train_config': train_config,
    'splits': splits
}

results_file = test_dir / 'test_results.json'
with open(results_file, 'w') as f:
    # Convert numpy types to native Python types for JSON
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    json.dump(results, f, indent=2, default=convert)

print(f"✓ Results saved to: {results_file}")
print()

#=============================================================================
# SUMMARY
#=============================================================================
print("="*70)
print("TEST COMPLETED SUCCESSFULLY!")
print("="*70)
print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nAll outputs saved to: {test_dir}")
print("\nGenerated files:")
print(f"  - {splits_file.name}")
print(f"  - train_windows.npz")
print(f"  - test_windows.npz")
print(f"  - model.pt")
print(f"  - {results_file.name}")
print(f"  - pipeline_test.log")
print()
print("="*70)
print("✅ PIPELINE VERIFICATION COMPLETE!")
print("="*70)
print("\nNext steps:")
print("  1. Review results: cat", test_dir / "test_results.json")
print("  2. Check logs: cat", test_dir / "pipeline_test.log")
print("  3. Run full training: python scripts/ttm_vitaldb.py train --mode fasttrack")
print()
