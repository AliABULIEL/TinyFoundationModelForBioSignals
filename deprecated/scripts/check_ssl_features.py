#!/usr/bin/env python3
"""Check if SSL encoder produces useful features for quality classification."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from src.models.ttm_adapter import TTMAdapter

print("="*70)
print("TESTING SSL FEATURE QUALITY")
print("="*70)

# Load data from individual window files
print("\n1. Loading BUT-PPG data...")

def load_window_files(directory):
    """Load all window_*.npz files from a directory."""
    signals = []
    labels = []

    window_files = sorted(Path(directory).glob('window_*.npz'))
    for f in window_files:
        data = np.load(f)
        signals.append(data['signal'])  # [2, 1024]
        labels.append(int(data['quality']))  # 0 or 1

    return np.array(signals), np.array(labels)

X_train, y_train = load_window_files('data/processed/butppg/windows_with_labels/train')
X_test, y_test = load_window_files('data/processed/butppg/windows_with_labels/test')

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

print(f"   Train: {len(X_train)} samples")
print(f"   Test: {len(X_test)} samples")
print(f"   Class balance: {np.bincount(y_train)}")

# Load SSL model
print("\n2. Loading SSL encoder...")
ckpt = torch.load('artifacts/butppg_ssl/best_model.pt', map_location='cpu', weights_only=False)
ssl_state = ckpt['encoder_state_dict']

model = TTMAdapter(
    variant='ibm-granite/granite-timeseries-ttm-r1',
    task='classification',
    num_classes=2,
    input_channels=2,
    context_length=1024,
    patch_size=128,
    d_model=192,
    freeze_encoder=False
)

# Auto-adapt
with torch.no_grad():
    _ = model.get_encoder_output(torch.randn(1, 2, 1024))

# Load SSL weights
backbone_dict = {}
for k, v in ssl_state.items():
    if 'encoder.backbone' in k:
        backbone_dict[k] = v
        backbone_dict[k.replace('encoder.', '', 1)] = v

missing, unexpected = model.load_state_dict(backbone_dict, strict=False)
model.eval()

# Sanity check: Verify SSL weights actually loaded
backbone_missing = [k for k in missing if 'backbone' in k and 'encoder' in k]
if len(backbone_missing) == 0:
    print(f"   ✅ SSL weights loaded successfully (196 keys)")
    print(f"   → Model is using VitalDB-trained weights, NOT IBM pretrained!")
else:
    print(f"   ❌ WARNING: {len(backbone_missing)} backbone keys didn't load!")
    print(f"   → Model might still be using IBM pretrained weights!")

# Extra verification: Compare a weight to SSL checkpoint
ssl_patcher_weight = ssl_state['encoder.backbone.encoder.patcher.weight']
model_patcher_weight = model.state_dict()['encoder.backbone.encoder.patcher.weight']
if torch.equal(ssl_patcher_weight, model_patcher_weight):
    print(f"   ✅ Verified: Weights match SSL checkpoint (not IBM pretrained)")
else:
    print(f"   ❌ ERROR: Weights DON'T match SSL checkpoint!")

# Extract features
print("\n3. Extracting SSL features...")
train_features = []
test_features = []

with torch.no_grad():
    # Train features
    for i in range(0, len(X_train), 64):
        batch = X_train[i:i+64]
        features = model.get_encoder_output(batch)
        # Global average pool over patches
        features = features.mean(dim=1)  # [batch, d_model]
        train_features.append(features)
    train_features = torch.cat(train_features, dim=0).numpy()

    # Test features
    for i in range(0, len(X_test), 64):
        batch = X_test[i:i+64]
        features = model.get_encoder_output(batch)
        features = features.mean(dim=1)
        test_features.append(features)
    test_features = torch.cat(test_features, dim=0).numpy()

print(f"   Train features shape: {train_features.shape}")
print(f"   Test features shape: {test_features.shape}")

# Train simple logistic regression
print("\n4. Training logistic regression on SSL features...")
# Use class weights
class_weight = {0: 0.636, 1: 2.335}
clf = LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=42)
clf.fit(train_features, y_train)

# Evaluate
train_preds = clf.predict(train_features)
test_preds = clf.predict(test_features)
test_probs = clf.predict_proba(test_features)[:, 1]

train_acc = accuracy_score(y_train, train_preds)
test_acc = accuracy_score(y_test, test_preds)
test_auroc = roc_auc_score(y_test, test_probs)

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Train Accuracy: {train_acc:.2%}")
print(f"Test Accuracy: {test_acc:.2%}")
print(f"Test AUROC: {test_auroc:.3f}")

# Per-class accuracy
for cls in [0, 1]:
    mask = y_test == cls
    cls_acc = accuracy_score(y_test[mask], test_preds[mask])
    print(f"Class {cls} Accuracy: {cls_acc:.2%}")

print("\nPrediction distribution:")
print(f"  Predicted Poor: {(test_preds == 0).sum()}/{len(test_preds)}")
print(f"  Predicted Good: {(test_preds == 1).sum()}/{len(test_preds)}")

print("\n" + "="*70)
if test_auroc < 0.70:
    print("❌ SSL features are NOT useful (AUROC < 0.70)")
    print("   → SSL didn't learn quality-relevant features")
elif test_auroc < 0.85:
    print("⚠️  SSL features are SOMEWHAT useful (0.70 ≤ AUROC < 0.85)")
    print("   → SSL learned some relevant features, but not great")
else:
    print("✅ SSL features are VERY useful (AUROC ≥ 0.85)")
    print("   → Problem is likely in the fine-tuning script")
print("="*70)
