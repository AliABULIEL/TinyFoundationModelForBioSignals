#!/usr/bin/env python3
"""
Quick E2E Test - TTM Compatible Version
Matches TTM's architecture requirements:
- context_length = 512 samples
- input_channels = 3 (ECG, PPG, ABP)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

from src.data.splits import make_patient_level_splits
from src.data.vitaldb_loader import load_channel, get_available_case_sets
from src.data.windows import make_windows, compute_normalization_stats, normalize_windows
from src.data.filters import apply_bandpass_filter
from src.data.detect import find_ppg_peaks, find_ecg_rpeaks
from src.data.quality import compute_sqi
from src.models.ttm_adapter import create_ttm_model
from src.models.trainers import TrainerClf
from src.eval.metrics import compute_classification_metrics
from src.utils.seed import set_seed

set_seed(42)
device = torch.device('cpu')

print("="*70)
print("TTM-COMPATIBLE E2E TEST")
print("="*70)
print("TTM Requirements:")
print("  - context_length: 512 samples")
print("  - input_channels: 3 (multi-channel)")
print("  - prediction_length: 96 (for decoder)")
print("="*70)
print()

# Create test directory
test_dir = Path(__file__).parent.parent / f"test_ttm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
test_dir.mkdir(exist_ok=True)

# Setup logging
import logging
file_handler = logging.FileHandler(test_dir / 'pipeline_test.log')
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])
logger = logging.getLogger(__name__)

#=============================================================================
# STEP 1: PREPARE SPLITS
#=============================================================================
print("STEP 1: PREPARE SPLITS")
print("-"*70)

case_sets = get_available_case_sets()
test_case_ids = sorted(list(case_sets['bis']))[:2]
cases = [{'case_id': str(cid), 'subject_id': str(cid)} for cid in test_case_ids]

splits = make_patient_level_splits(cases, ratios=(0.5, 0.5), seed=42)
simple_splits = {
    'train': [c['case_id'] for c in splits['train']],
    'test': [c['case_id'] for c in splits['test']]
}

print(f"✓ Using cases: {simple_splits}")
print()

#=============================================================================
# STEP 2: BUILD TTM-COMPATIBLE WINDOWS
#=============================================================================
print("STEP 2: BUILD TTM-COMPATIBLE WINDOWS")
print("-"*70)
print("Target: [N, 512, 3] windows")
print()

# TTM requirements
TTM_CONTEXT_LENGTH = 512  # Required by pre-trained model
TTM_NUM_CHANNELS = 3      # Multi-channel input
TARGET_FS = 50.0          # After NaN fix, VitalDB PPG is ~50Hz
WIN_DURATION = TTM_CONTEXT_LENGTH / TARGET_FS  # 10.24 seconds

windows_data = {'train': [], 'test': []}
labels_data = {'train': [], 'test': []}
train_stats = None

# Channel names to load
CHANNELS = ['PLETH', 'ECG_II', 'ABP']  # PPG, ECG, Arterial BP

for split_name, case_ids in simple_splits.items():
    print(f"\nProcessing {split_name} split...")
    
    for case_id in case_ids:
        print(f"  Case {case_id}:")
        
        try:
            # Load multiple channels
            all_channels = []
            valid_channels = []
            
            for ch_name in CHANNELS:
                try:
                    print(f"    - Loading {ch_name}...")
                    signal, fs = load_channel(
                        case_id=case_id,
                        channel=ch_name,
                        duration_sec=60,
                        auto_fix_alternating=True
                    )
                    
                    if signal is None or len(signal) < 100:
                        print(f"      ⚠ Failed to load {ch_name}, will pad")
                        continue
                    
                    # Apply appropriate filter
                    if 'PLETH' in ch_name or 'PPG' in ch_name:
                        filtered = apply_bandpass_filter(signal, fs, 0.5, 10, 'cheby2', 4)
                    elif 'ECG' in ch_name:
                        filtered = apply_bandpass_filter(signal, fs, 0.5, 40, 'butter', 4)
                    elif 'ABP' in ch_name:
                        filtered = apply_bandpass_filter(signal, fs, 0.5, 20, 'butter', 4)
                    else:
                        filtered = signal
                    
                    all_channels.append(filtered)
                    valid_channels.append(ch_name)
                    print(f"      ✓ Loaded {len(filtered)} samples at {fs} Hz")
                    
                except Exception as e:
                    print(f"      ✗ Error loading {ch_name}: {e}")
                    continue
            
            # Must have at least one channel
            if len(all_channels) == 0:
                print(f"    ✗ No valid channels loaded")
                continue
            
            # Pad to 3 channels if needed
            while len(all_channels) < TTM_NUM_CHANNELS:
                print(f"    ⚠ Padding with zeros (only {len(all_channels)} channels available)")
                all_channels.append(np.zeros_like(all_channels[0]))
                valid_channels.append('PADDED')
            
            # Stack channels: [time, channels]
            min_len = min(len(ch) for ch in all_channels)
            multichannel_signal = np.stack([ch[:min_len] for ch in all_channels], axis=1)
            
            print(f"    - Stacked shape: {multichannel_signal.shape}")
            
            # Detect peaks on first valid channel for validation
            primary_signal = all_channels[0]
            if 'PPG' in valid_channels[0] or 'PLETH' in valid_channels[0]:
                peaks = find_ppg_peaks(primary_signal, fs)
            elif 'ECG' in valid_channels[0]:
                peaks, _ = find_ecg_rpeaks(primary_signal, fs)
            else:
                peaks = np.array([])
            
            print(f"    - Found {len(peaks)} peaks in {valid_channels[0]}")
            
            if len(peaks) < 10:
                print(f"    ✗ Not enough peaks")
                continue
            
            # Create TTM-compatible windows
            case_windows = make_windows(
                X_tc=multichannel_signal,
                fs=fs,
                win_s=WIN_DURATION,  # 10.24s to get exactly 512 samples
                stride_s=WIN_DURATION,
                min_cycles=1,  # Lenient for testing
                peaks_tc={0: peaks}  # Validate on first channel
            )
            
            if case_windows is None or len(case_windows) == 0:
                print(f"    ✗ No valid windows created")
                continue
            
            print(f"    - Created {len(case_windows)} windows of shape {case_windows.shape}")
            
            # Verify shape matches TTM requirements
            n_windows, time_steps, n_channels = case_windows.shape
            
            if time_steps != TTM_CONTEXT_LENGTH:
                print(f"    ⚠ Window length {time_steps} != {TTM_CONTEXT_LENGTH}, will resample")
                # Simple resampling to match TTM
                resampled_windows = []
                for w in case_windows:
                    # Interpolate each channel to 512 samples
                    resampled = np.zeros((TTM_CONTEXT_LENGTH, n_channels))
                    for ch in range(n_channels):
                        x_old = np.linspace(0, 1, time_steps)
                        x_new = np.linspace(0, 1, TTM_CONTEXT_LENGTH)
                        resampled[:, ch] = np.interp(x_new, x_old, w[:, ch])
                    resampled_windows.append(resampled)
                case_windows = np.array(resampled_windows)
                print(f"    ✓ Resampled to {case_windows.shape}")
            
            # Compute/apply normalization
            if split_name == 'train' and train_stats is None:
                train_stats = compute_normalization_stats(
                    X=case_windows,
                    method='zscore',
                    axis=(0, 1)
                )
                print(f"    ✓ Computed train stats")
            
            if train_stats is not None:
                normalized_windows = normalize_windows(
                    W_ntc=case_windows,
                    stats=train_stats,
                    baseline_correction=False,
                    per_channel=False
                )
            else:
                normalized_windows = case_windows
            
            # Store windows with labels
            for w in normalized_windows:
                windows_data[split_name].append(w)
                labels_data[split_name].append(np.random.randint(0, 2))
            
            print(f"    ✓ Stored {len(normalized_windows)} windows")
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            logger.exception(f"Error processing case {case_id}")
            continue

# Save windows
print("\nSaving windows...")
for split_name in ['train', 'test']:
    if windows_data[split_name]:
        windows_array = np.array(windows_data[split_name])
        labels_array = np.array(labels_data[split_name])
        
        out_file = test_dir / f'{split_name}_windows.npz'
        np.savez_compressed(out_file, data=windows_array, labels=labels_array)
        print(f"✓ Saved {split_name}: {windows_array.shape}")

if not windows_data['train'] or not windows_data['test']:
    raise RuntimeError("Failed to create windows")

print()

#=============================================================================
# STEP 3: TRAIN WITH REAL TTM
#=============================================================================
print("STEP 3: TRAIN MODEL WITH REAL TTM")
print("-"*70)

# Load windows
train_npz = np.load(test_dir / 'train_windows.npz')
test_npz = np.load(test_dir / 'test_windows.npz')

X_train = torch.from_numpy(train_npz['data']).float()
y_train = torch.from_numpy(train_npz['labels']).long()
X_test = torch.from_numpy(test_npz['data']).float()
y_test = torch.from_numpy(test_npz['labels']).long()

print(f"Train data: {X_train.shape}")
print(f"Test data: {X_test.shape}")

# Verify TTM compatibility
assert X_train.shape[1] == TTM_CONTEXT_LENGTH, f"Wrong context length: {X_train.shape[1]} != {TTM_CONTEXT_LENGTH}"
assert X_train.shape[2] == TTM_NUM_CHANNELS, f"Wrong num channels: {X_train.shape[2]} != {TTM_NUM_CHANNELS}"

print("✓ Data shape matches TTM requirements!")
print()

# Create datasets
from torch.utils.data import TensorDataset
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Create TTM model with matching config
model_config = {
    'variant': 'ibm-granite/granite-timeseries-ttm-r1',
    'task': 'classification',
    'num_classes': 2,
    'input_channels': TTM_NUM_CHANNELS,
    'context_length': TTM_CONTEXT_LENGTH,
    'prediction_length': 96,  # TTM decoder requirement
    'freeze_encoder': True,
    'head_type': 'linear',
    'use_real_ttm': True,
    'decoder_mode': 'mix_channel'
}

print("Creating TTM model with config:")
for k, v in model_config.items():
    print(f"  {k}: {v}")
print()

model = create_ttm_model(model_config)
model = model.to(device)

print(f"Model: {sum(p.numel() for p in model.parameters())} total params")
print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)} params")
print(f"Using real TTM: {model.is_using_real_ttm()}")
print()

if not model.is_using_real_ttm():
    print("⚠️ WARNING: TTM failed to load! Using fallback CNN.")
    print("This means the pre-trained foundation model is NOT being used!")
    print()

# Create data loader
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=4, shuffle=True, num_workers=0
)

# Create trainer
trainer = TrainerClf(
    model=model,
    train_loader=train_loader,
    val_loader=None,
    num_classes=2,
    device='cpu',
    use_amp=False,
    gradient_clip=1.0,
    checkpoint_dir=str(test_dir)
)

# Train
print("Training for 2 epochs...")
train_metrics = trainer.fit(num_epochs=2, save_best=True, early_stopping_patience=10)

if train_metrics['train_history']:
    final_loss = train_metrics['train_history'][-1]['loss']
    print(f"✓ Training completed! Final loss: {final_loss:.4f}")

print()

#=============================================================================
# STEP 4: TEST
#=============================================================================
print("STEP 4: TEST AND EVALUATE")
print("-"*70)

# Load best model
checkpoint_path = test_dir / 'model.pt'
if checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded checkpoint")

model.eval()

# Evaluate
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=4, shuffle=False, num_workers=0
)

all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for X, y in test_loader:
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

metrics = compute_classification_metrics(
    y_true=all_labels,
    y_pred=all_preds,
    y_prob=all_probs[:, 1] if all_probs.shape[1] > 1 else all_probs[:, 0]
)

print("\nTest Results:")
print(f"  Accuracy:  {metrics['accuracy']:.3f}")
print(f"  Precision: {metrics['precision']:.3f}")
print(f"  Recall:    {metrics['recall']:.3f}")
print(f"  F1 Score:  {metrics['f1']:.3f}")
print(f"  AUC-ROC:   {metrics['auroc']:.3f}")
print()

print("="*70)
print("✅ TTM-COMPATIBLE TEST COMPLETED!")
print("="*70)
print(f"\nUsed real TTM: {model.is_using_real_ttm()}")
print(f"Results saved to: {test_dir}")
print()
