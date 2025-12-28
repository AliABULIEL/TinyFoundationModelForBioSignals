#!/usr/bin/env python3
"""Test if SSL features are discriminative for quality classification.

This script performs three critical tests:
1. Linear Probe: Train simple logistic regression on SSL features
2. Feature Visualization: PCA to see if classes are separable
3. Feature Statistics: Quantify difference between quality classes

If SSL features are good (>65% accuracy), problem is in fine-tuning.
If SSL features are weak (<55% accuracy), SSL training failed.

Usage:
    python scripts/test_ssl_features.py \
        --ssl-checkpoint artifacts/FINAL_RUN/stage2_butppg_quality_ssl/best_model.pt \
        --data-dir data/processed/butppg/windows_with_labels \
        --device cuda
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import argparse


def load_ssl_encoder(checkpoint_path: str, device: str = 'cuda'):
    """Load SSL encoder from checkpoint."""
    print(f"\n📦 Loading SSL encoder from: {checkpoint_path}")

    # Import the function from SSL script
    from continue_ssl_butppg_quality import init_ibm_pretrained

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Get encoder state
    if 'encoder_state_dict' in checkpoint:
        encoder_state = checkpoint['encoder_state_dict']
        print("  ✓ Found encoder_state_dict")
    else:
        raise ValueError(f"No encoder_state_dict in checkpoint. Keys: {list(checkpoint.keys())}")

    # Detect architecture
    patcher_keys = [k for k in encoder_state.keys() if 'patcher.weight' in k]
    if not patcher_keys:
        raise ValueError("Cannot find patcher weights")

    patcher_weight = encoder_state[patcher_keys[0]]
    d_model = patcher_weight.shape[0]
    patch_size = patcher_weight.shape[1]

    # Get context length
    config = checkpoint.get('config', {})
    context_length = config.get('context_length', 512 if patch_size == 64 else 1024)

    print(f"  Detected: context={context_length}, patch={patch_size}, d_model={d_model}")

    # Create encoder
    encoder, _ = init_ibm_pretrained(
        variant='ibm-granite/granite-timeseries-ttm-r1',
        context_length=1024,  # Loading context
        patch_size=patch_size,
        num_channels=2,
        device=device
    )

    # Load weights
    encoder.load_state_dict(encoder_state, strict=False)
    encoder.eval()

    print(f"  ✓ Encoder loaded: {sum(p.numel() for p in encoder.parameters()):,} params")

    return encoder, context_length, d_model


def extract_features(encoder, loader, context_length: int, device: str = 'cuda'):
    """Extract SSL features from data."""
    features, labels = [], []

    print(f"\nExtracting features (target length={context_length})...")

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device)

            # Resample to SSL context length if needed
            if x.shape[-1] != context_length:
                x = F.interpolate(
                    x,
                    size=context_length,
                    mode='linear',
                    align_corners=False
                )

            # Get features: [B, P, D]
            feat = encoder.get_encoder_output(x)

            # Pool: [B, P, D] → [B, D]
            feat = feat.mean(dim=1)

            features.append(feat.cpu().numpy())
            labels.append(y.numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1} batches...", end='\r')

    features = np.vstack(features)
    labels = np.concatenate(labels)

    print(f"\n  ✓ Extracted: {features.shape} features, {labels.shape} labels")
    print(f"  Class distribution: {np.bincount(labels)}")

    return features, labels


def test_linear_probe(X_train, y_train, X_test, y_test):
    """Test 1: Linear probe with logistic regression."""
    print("\n" + "="*80)
    print("TEST 1: LINEAR PROBE (Logistic Regression)")
    print("="*80)
    print("This tests if SSL features can be linearly separated.")
    print("Good SSL should achieve >65% accuracy with simple linear model.\n")

    # Train logistic regression
    clf = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
        verbose=0
    )

    print("Training logistic regression...")
    clf.fit(X_train, y_train)

    # Evaluate
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # Metrics
    auroc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n📊 Results:")
    print(f"  Train accuracy: {train_acc:.1%}")
    print(f"  Test accuracy:  {test_acc:.1%}")
    print(f"  Test AUROC:     {auroc:.3f}")

    print(f"\n📊 Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Poor    Good")
    print(f"Actual Poor   {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"       Good   {cm[1,0]:4d}    {cm[1,1]:4d}")

    # Per-class accuracy
    poor_acc = cm[0,0] / cm[0].sum() if cm[0].sum() > 0 else 0
    good_acc = cm[1,1] / cm[1].sum() if cm[1].sum() > 0 else 0

    print(f"\n📊 Per-Class Accuracy:")
    print(f"  Poor (Class 0): {poor_acc:.1%}")
    print(f"  Good (Class 1): {good_acc:.1%}")

    print(f"\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Poor', 'Good'], digits=3))

    return test_acc, auroc, poor_acc, good_acc


def visualize_features(X_train, y_train, X_test, y_test, output_path='ssl_features_pca.png'):
    """Test 2: Visualize feature separability with PCA."""
    print("\n" + "="*80)
    print("TEST 2: FEATURE VISUALIZATION (PCA)")
    print("="*80)
    print("Projecting features to 2D to visualize class separation.\n")

    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print(f"Explained variance: {pca.explained_variance_ratio_[0]:.1%} + {pca.explained_variance_ratio_[1]:.1%} = {pca.explained_variance_ratio_.sum():.1%}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Train
    axes[0].scatter(X_train_pca[y_train==0, 0], X_train_pca[y_train==0, 1],
                    alpha=0.5, label='Poor Quality', s=20, c='red')
    axes[0].scatter(X_train_pca[y_train==1, 0], X_train_pca[y_train==1, 1],
                    alpha=0.5, label='Good Quality', s=20, c='green')
    axes[0].set_title('Train Features (PCA)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Test
    axes[1].scatter(X_test_pca[y_test==0, 0], X_test_pca[y_test==0, 1],
                    alpha=0.5, label='Poor Quality', s=20, c='red')
    axes[1].scatter(X_test_pca[y_test==1, 0], X_test_pca[y_test==1, 1],
                    alpha=0.5, label='Good Quality', s=20, c='green')
    axes[1].set_title('Test Features (PCA)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved visualization: {output_path}")

    # Calculate separation metric (distance between class centroids)
    centroid_poor = X_train_pca[y_train==0].mean(axis=0)
    centroid_good = X_train_pca[y_train==1].mean(axis=0)
    separation = np.linalg.norm(centroid_poor - centroid_good)

    print(f"  Centroid separation (Euclidean): {separation:.3f}")
    print(f"    (Higher = better separation, >5 is good)")


def analyze_feature_statistics(X_train, y_train):
    """Test 3: Statistical analysis of feature discriminability."""
    print("\n" + "="*80)
    print("TEST 3: FEATURE STATISTICS")
    print("="*80)
    print("Analyzing statistical differences between quality classes.\n")

    features_poor = X_train[y_train == 0]
    features_good = X_train[y_train == 1]

    # Mean difference
    mean_poor = features_poor.mean(axis=0)
    mean_good = features_good.mean(axis=0)
    mean_diff = np.abs(mean_poor - mean_good)

    # Std comparison
    std_poor = features_poor.std(axis=0)
    std_good = features_good.std(axis=0)

    # Cohen's d (effect size)
    pooled_std = np.sqrt((std_poor**2 + std_good**2) / 2)
    cohens_d = mean_diff / (pooled_std + 1e-8)

    print(f"📊 Feature Statistics:")
    print(f"  Mean absolute difference: {mean_diff.mean():.4f}")
    print(f"    (Higher = more discriminative, >0.1 is good)")

    print(f"\n  Std ratio (Poor/Good): {(std_poor / (std_good + 1e-8)).mean():.4f}")
    print(f"    (Close to 1.0 = similar variance)")

    print(f"\n  Cohen's d (effect size): {cohens_d.mean():.4f}")
    print(f"    (>0.5 = medium, >0.8 = large effect)")

    # Top discriminative features
    top_k = 10
    top_indices = np.argsort(mean_diff)[-top_k:][::-1]

    print(f"\n  Top {top_k} most discriminative features:")
    for i, idx in enumerate(top_indices, 1):
        print(f"    {i}. Feature {idx}: diff={mean_diff[idx]:.4f}, d={cohens_d[idx]:.4f}")

    return mean_diff.mean(), cohens_d.mean()


def diagnose_results(test_acc, auroc, poor_acc, good_acc, mean_diff, effect_size):
    """Provide diagnosis based on all test results."""
    print("\n" + "="*80)
    print("DIAGNOSIS & RECOMMENDATIONS")
    print("="*80)

    print(f"\n📊 Summary:")
    print(f"  Linear probe accuracy:  {test_acc:.1%}")
    print(f"  AUROC:                  {auroc:.3f}")
    print(f"  Poor class recall:      {poor_acc:.1%}")
    print(f"  Good class recall:      {good_acc:.1%}")
    print(f"  Feature mean diff:      {mean_diff:.4f}")
    print(f"  Cohen's d:              {effect_size:.4f}")

    print(f"\n🔍 Diagnosis:")

    if test_acc > 0.65 and auroc > 0.7:
        print("  ✅ SSL features are GOOD (>65% accuracy, >0.7 AUROC)")
        print("     Features contain strong quality-discriminative information.")
        print()
        print("  🎯 Recommendation: FIX FINE-TUNING")
        print("     Problem is NOT in SSL - it's in the fine-tuning setup.")
        print()
        print("  Next steps:")
        print("     1. Remove Focal Loss (too aggressive)")
        print("     2. Use balanced CrossEntropy with moderate weights")
        print("     3. Add balanced sampling to training loop")
        print("     4. Use differential learning rates (head vs encoder)")
        print("     5. Reduce head-only epochs (2-3 instead of 5)")
        print()
        print("  Expected after fix: 70-75% accuracy, 0.75-0.80 AUROC")

    elif test_acc > 0.55 and auroc > 0.6:
        print("  ⚠️  SSL features are MARGINAL (55-65% accuracy)")
        print("     Features capture some quality info but discriminability is weak.")
        print()
        print("  🎯 Recommendation: IMPROVE SSL + FIX FINE-TUNING")
        print("     SSL needs improvement, but fine-tuning also needs fixing.")
        print()
        print("  Next steps:")
        print("     1. Retrain SSL with stronger contrastive loss")
        print("     2. Use harder negative mining (lower temperature)")
        print("     3. Add more aggressive data augmentation")
        print("     4. Ensure quality labels are accurate")
        print("     5. Fix fine-tuning loss (remove Focal, use balanced CE)")
        print()
        print("  Expected after fix: 65-70% accuracy, 0.70-0.75 AUROC")

    else:
        print("  ❌ SSL features are NOT USEFUL (<55% accuracy)")
        print("     SSL training FAILED to learn quality-discriminative features.")
        print()
        print("  🎯 Recommendation: RETRAIN SSL FROM SCRATCH")
        print("     Current SSL checkpoint is not useful - need fundamental redesign.")
        print()
        print("  Root cause analysis:")
        if effect_size < 0.3:
            print("     • Effect size too small - classes are barely different")
            print("       → SSL learned to make all embeddings similar")
        if poor_acc < 0.3 and good_acc < 0.3:
            print("     • Both classes have low recall")
            print("       → Features are noisy/random")
        if poor_acc > 0.7 and good_acc < 0.3:
            print("     • Model biased to Poor class")
            print("       → Quality labels might be wrong")
        elif poor_acc < 0.3 and good_acc > 0.7:
            print("     • Model biased to Good class")
            print("       → Contrastive loss might have collapsed")
        print()
        print("  Next steps:")
        print("     1. Check SSL contrastive loss (should be ~0.3-0.5, not 0.001)")
        print("     2. Verify quality labels are correct")
        print("     3. Use harder negatives (temperature=0.05)")
        print("     4. Increase contrastive weight (2.0 vs reconstruction 0.1)")
        print("     5. Add quality-aware sampling")
        print()
        print("  Expected after SSL retraining: 70-80% accuracy, 0.75-0.85 AUROC")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Test SSL feature discriminability')
    parser.add_argument('--ssl-checkpoint', type=str, required=True,
                       help='Path to SSL checkpoint')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to BUT-PPG data directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for feature extraction')
    parser.add_argument('--output', type=str, default='ssl_features_pca.png',
                       help='Output path for PCA visualization')

    args = parser.parse_args()

    print("="*80)
    print("SSL FEATURE DIAGNOSTIC TEST")
    print("="*80)
    print("This will test if SSL-learned features are discriminative for quality.")
    print(f"SSL checkpoint: {args.ssl_checkpoint}")
    print(f"Data directory: {args.data_dir}")
    print(f"Device: {args.device}")

    # Load encoder
    encoder, context_length, d_model = load_ssl_encoder(args.ssl_checkpoint, args.device)

    # Load data
    from finetune_butppg import create_dataloaders

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=Path(args.data_dir),
        batch_size=args.batch_size,
        num_workers=4,
        target_length=context_length
    )

    # Extract features
    print(f"\n{'='*80}")
    print("EXTRACTING FEATURES")
    print("="*80)

    X_train, y_train = extract_features(encoder, train_loader, context_length, args.device)
    X_test, y_test = extract_features(encoder, test_loader, context_length, args.device)

    # Run tests
    test_acc, auroc, poor_acc, good_acc = test_linear_probe(X_train, y_train, X_test, y_test)

    visualize_features(X_train, y_train, X_test, y_test, args.output)

    mean_diff, effect_size = analyze_feature_statistics(X_train, y_train)

    # Diagnosis
    diagnose_results(test_acc, auroc, poor_acc, good_acc, mean_diff, effect_size)

    print("\n✓ Diagnostic complete!")


if __name__ == '__main__':
    main()
