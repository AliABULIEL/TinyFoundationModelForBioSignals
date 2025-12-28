#!/usr/bin/env python3
"""Comprehensive diagnostic to understand why SSL features don't transfer."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from src.models.ttm_adapter import TTMAdapter

print("="*80)
print("COMPREHENSIVE SSL DIAGNOSIS")
print("="*80)

# Load BUT-PPG data
def load_window_files(directory):
    signals, labels = [], []
    for f in sorted(Path(directory).glob('window_*.npz')):
        data = np.load(f)
        signals.append(data['signal'])
        labels.append(int(data['quality']))
    return np.array(signals), np.array(labels)

print("\n1. Loading BUT-PPG data...")
X_train, y_train = load_window_files('data/processed/butppg/windows_with_labels/train')
X_test, y_test = load_window_files('data/processed/butppg/windows_with_labels/test')

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

print(f"   Train: {len(X_train)} samples, {np.bincount(y_train)} class distribution")
print(f"   Test: {len(X_test)} samples, {np.bincount(y_test)} class distribution")

# Load SSL checkpoint
print("\n2. Loading SSL checkpoint...")
ckpt = torch.load('artifacts/butppg_ssl/best_model.pt', map_location='cpu', weights_only=False)
ssl_state = ckpt['encoder_state_dict']
ssl_config = ckpt.get('config', {})

print(f"   SSL training config:")
print(f"     Epochs: {ckpt.get('epoch', 'unknown')}")
print(f"     Best val loss: {ckpt.get('best_val_loss', 'unknown')}")
print(f"     Metrics: {ckpt.get('metrics', {})}")

# Create 3 models for comparison
print("\n3. Creating 3 models for comparison...")

# Model 1: IBM pretrained (baseline)
print("\n   Model 1: IBM Pretrained (no SSL)")
model_ibm = TTMAdapter(
    variant='ibm-granite/granite-timeseries-ttm-r1',
    task='classification',
    num_classes=2,
    input_channels=2,
    context_length=1024,
    patch_size=128,
    d_model=192,
    freeze_encoder=False
)
with torch.no_grad():
    _ = model_ibm.get_encoder_output(torch.randn(1, 2, 1024))
model_ibm.eval()

# Model 2: SSL weights loaded
print("\n   Model 2: SSL Weights (VitalDB-adapted)")
model_ssl = TTMAdapter(
    variant='ibm-granite/granite-timeseries-ttm-r1',
    task='classification',
    num_classes=2,
    input_channels=2,
    context_length=1024,
    patch_size=128,
    d_model=192,
    freeze_encoder=False
)
with torch.no_grad():
    _ = model_ssl.get_encoder_output(torch.randn(1, 2, 1024))

# Load SSL weights
backbone_dict = {}
for k, v in ssl_state.items():
    if 'encoder.backbone' in k:
        backbone_dict[k] = v
        backbone_dict[k.replace('encoder.', '', 1)] = v

missing, unexpected = model_ssl.load_state_dict(backbone_dict, strict=False)
backbone_missing = [k for k in missing if 'backbone' in k and 'encoder' in k]

print(f"   SSL weight loading:")
print(f"     Attempted: {len(backbone_dict)} keys")
print(f"     Missing backbone: {len(backbone_missing)}")

# Verify weights actually changed
ibm_patcher = model_ibm.state_dict()['encoder.backbone.encoder.patcher.weight']
ssl_patcher = model_ssl.state_dict()['encoder.backbone.encoder.patcher.weight']
ssl_ckpt_patcher = ssl_state['encoder.backbone.encoder.patcher.weight']

if torch.equal(ibm_patcher, ssl_patcher):
    print(f"     ❌ ERROR: SSL model weights = IBM weights (SSL didn't load!)")
elif torch.equal(ssl_patcher, ssl_ckpt_patcher):
    print(f"     ✅ VERIFIED: SSL model weights match checkpoint")
else:
    print(f"     ⚠️  WARNING: SSL weights partially loaded?")

model_ssl.eval()

# Model 3: Random initialization
print("\n   Model 3: Random Init (control)")
model_random = TTMAdapter(
    variant='ibm-granite/granite-timeseries-ttm-r1',
    task='classification',
    num_classes=2,
    input_channels=2,
    context_length=1024,
    patch_size=128,
    d_model=192,
    freeze_encoder=False
)
with torch.no_grad():
    _ = model_random.get_encoder_output(torch.randn(1, 2, 1024))

# Randomize all backbone weights
for name, param in model_random.named_parameters():
    if 'backbone' in name and 'encoder' in name:
        param.data = torch.randn_like(param.data) * 0.02

model_random.eval()

# Extract features from all 3 models
print("\n4. Extracting features from all 3 models...")

def extract_features(model, X):
    features = []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            batch = X[i:i+64]
            feats = model.get_encoder_output(batch)
            feats = feats.mean(dim=1)  # Global avg pool
            features.append(feats)
    return torch.cat(features, dim=0).numpy()

train_feat_ibm = extract_features(model_ibm, X_train)
train_feat_ssl = extract_features(model_ssl, X_train)
train_feat_random = extract_features(model_random, X_train)

test_feat_ibm = extract_features(model_ibm, X_test)
test_feat_ssl = extract_features(model_ssl, X_test)
test_feat_random = extract_features(model_random, X_test)

print(f"   Feature shapes: {train_feat_ibm.shape}")

# Train logistic regression on each feature set
print("\n5. Training classifiers on each feature set...")

class_weight = {0: 0.636, 1: 2.335}
results = {}

for name, train_f, test_f in [
    ('IBM Pretrained', train_feat_ibm, test_feat_ibm),
    ('SSL (VitalDB)', train_feat_ssl, test_feat_ssl),
    ('Random Init', train_feat_random, test_feat_random)
]:
    clf = LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=42)
    clf.fit(train_f, y_train)

    test_preds = clf.predict(test_f)
    test_probs = clf.predict_proba(test_f)[:, 1]

    acc = accuracy_score(y_test, test_preds)
    auroc = roc_auc_score(y_test, test_probs)

    # Per-class accuracy
    cls0_acc = accuracy_score(y_test[y_test==0], test_preds[y_test==0])
    cls1_acc = accuracy_score(y_test[y_test==1], test_preds[y_test==1])

    results[name] = {
        'accuracy': acc,
        'auroc': auroc,
        'class_0_acc': cls0_acc,
        'class_1_acc': cls1_acc
    }

print("\n" + "="*80)
print("RESULTS COMPARISON")
print("="*80)
print(f"{'Model':<20} {'AUROC':<10} {'Accuracy':<10} {'Poor Acc':<10} {'Good Acc':<10}")
print("-"*80)
for name, res in results.items():
    print(f"{name:<20} {res['auroc']:<10.3f} {res['accuracy']:<10.2%} {res['class_0_acc']:<10.2%} {res['class_1_acc']:<10.2%}")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

ssl_auroc = results['SSL (VitalDB)']['auroc']
ibm_auroc = results['IBM Pretrained']['auroc']
random_auroc = results['Random Init']['auroc']

if ssl_auroc < 0.55:
    print("❌ SSL features are WORSE than random!")
    print("   → Likely bug: SSL weights didn't load correctly")
elif ssl_auroc < ibm_auroc - 0.05:
    print("❌ SSL features are WORSE than IBM pretrained!")
    print("   → SSL training on VitalDB hurt performance")
    print("   → Masked reconstruction may not be right objective")
elif ssl_auroc < 0.70:
    print("⚠️  SSL features are weak (AUROC < 0.70)")
    print("   → SSL learned something, but not quality-relevant")
    print("   → VitalDB → BUT-PPG domain gap is too large")
elif ssl_auroc < ibm_auroc + 0.05:
    print("⚠️  SSL features are NO BETTER than IBM pretrained")
    print("   → SSL on VitalDB didn't add value for BUT-PPG quality")
    print("   → Domain gap: Hospital ICU ≠ Smartphone PPG")
else:
    print("✅ SSL features ARE better than IBM!")
    print(f"   → AUROC improvement: {ssl_auroc - ibm_auroc:.3f}")
    print("   → But still may not be good enough for deployment")

print("\nKey Findings:")
print(f"  1. SSL AUROC: {ssl_auroc:.3f} (vs IBM: {ibm_auroc:.3f}, Random: {random_auroc:.3f})")
print(f"  2. SSL improved over IBM: {ssl_auroc > ibm_auroc}")
print(f"  3. SSL improvement: {(ssl_auroc - ibm_auroc)*100:.1f} percentage points")

if ssl_auroc < 0.70:
    print("\n⚠️  RESEARCH IMPLICATION:")
    print("  Masked signal reconstruction on VitalDB does NOT create a")
    print("  foundation model that transfers well to BUT-PPG quality assessment.")
    print("\n  Possible reasons:")
    print("    - Domain gap: Hospital ICU PPG ≠ Smartphone PPG")
    print("    - Task mismatch: Reconstruction ≠ Quality assessment")
    print("    - Need different SSL objective (contrastive, supervised, etc.)")

print("="*80)
