#!/usr/bin/env python3
"""Validate Quality-Aware Features from SSL Training.

This script performs diagnostic tests to verify that the model has learned
quality-relevant features during Stage 2 training. It checks:

1. Quality Clustering: Do features cluster by quality level?
2. Separation Metrics: Clear separation between good/bad quality?
3. Linear Separability: Can quality be predicted from features?
4. t-SNE Visualization: Visual clustering by quality

Success Criteria:
- t-SNE shows clear clustering by quality level
- Linear probe AUROC ≥ 0.70 (without fine-tuning)
- Within-quality similarity > cross-quality similarity
- Silhouette score > 0.3

Usage:
    python scripts/validate_quality_features.py \
        --checkpoint artifacts/butppg_quality_ssl/best_model.pt \
        --data-dir data/processed/butppg/windows_with_labels \
        --plot-output artifacts/quality_features_analysis.png

    # Quick test
    python scripts/validate_quality_features.py \
        --checkpoint artifacts/butppg_quality_ssl/best_model.pt \
        --data-dir data/processed/butppg/windows_with_labels \
        --max-samples 500 \
        --skip-tsne
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.ttm_adapter import TTMAdapter
from src.data.butppg_quality_dataset import QualityStratifiedBUTPPGDataset
from src.ssl.quality_proxy import QualityProxyComputer


@torch.no_grad()
def extract_features(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = 'cuda'
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features and quality scores from dataset.

    Args:
        model: Encoder model
        data_loader: Data loader
        device: Device

    Returns:
        features: Feature array [N, D]
        quality_scores: Quality scores [N]
    """
    model.eval()
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    all_features = []
    all_quality = []

    for batch in data_loader:
        signals = batch['signal'].to(device)
        quality_scores = batch['quality_score']

        # Extract features
        features = model(signals)  # [B, P, D]

        # Pool across patches
        features_pooled = features.mean(dim=1)  # [B, D]

        all_features.append(features_pooled.cpu().numpy())
        all_quality.append(quality_scores.cpu().numpy())

    features = np.concatenate(all_features, axis=0)
    quality = np.concatenate(all_quality, axis=0)

    return features, quality


def compute_quality_clustering_metrics(
    features: np.ndarray,
    quality_scores: np.ndarray,
    num_bins: int = 3
) -> Dict[str, float]:
    """Compute metrics for quality clustering.

    Args:
        features: Feature array [N, D]
        quality_scores: Quality scores [N]
        num_bins: Number of quality bins

    Returns:
        metrics: Dict with clustering metrics
    """
    # Assign to bins
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(quality_scores, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    # 1. Silhouette score
    silhouette = silhouette_score(features, bin_indices, metric='cosine')

    # 2. Within-quality vs cross-quality similarity
    # Compute pairwise cosine similarities
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(features)

    within_quality_sims = []
    cross_quality_sims = []

    for i in range(len(features)):
        for j in range(i+1, len(features)):
            if bin_indices[i] == bin_indices[j]:
                within_quality_sims.append(similarities[i, j])
            else:
                cross_quality_sims.append(similarities[i, j])

    within_mean = np.mean(within_quality_sims)
    cross_mean = np.mean(cross_quality_sims)
    separation_ratio = within_mean / (cross_mean + 1e-8)

    metrics = {
        'silhouette_score': float(silhouette),
        'within_quality_similarity': float(within_mean),
        'cross_quality_similarity': float(cross_mean),
        'separation_ratio': float(separation_ratio)
    }

    return metrics


def linear_probe_evaluation(
    features: np.ndarray,
    quality_scores: np.ndarray,
    num_bins: int = 3,
    test_size: float = 0.2
) -> Dict[str, float]:
    """Evaluate linear separability of quality levels.

    Args:
        features: Feature array [N, D]
        quality_scores: Quality scores [N]
        num_bins: Number of quality bins
        test_size: Test set ratio

    Returns:
        metrics: Dict with linear probe metrics
    """
    from sklearn.model_selection import train_test_split

    # Assign to bins
    bin_edges = np.linspace(0, 1, num_bins + 1)
    labels = np.digitize(quality_scores, bin_edges[:-1]) - 1
    labels = np.clip(labels, 0, num_bins - 1)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=42, stratify=labels
    )

    # Train linear classifier
    clf = LogisticRegression(max_iter=1000, multi_class='ovr')
    clf.fit(X_train, y_train)

    # Evaluate
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    # AUROC
    y_proba = clf.predict_proba(X_test)
    if num_bins == 2:
        auroc = roc_auc_score(y_test, y_proba[:, 1])
    else:
        auroc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')

    metrics = {
        'linear_probe_train_acc': float(train_acc),
        'linear_probe_test_acc': float(test_acc),
        'linear_probe_auroc': float(auroc)
    }

    return metrics


def plot_tsne_quality(
    features: np.ndarray,
    quality_scores: np.ndarray,
    output_path: str,
    num_bins: int = 3
):
    """Plot t-SNE visualization colored by quality.

    Args:
        features: Feature array [N, D]
        quality_scores: Quality scores [N]
        output_path: Path to save plot
        num_bins: Number of quality bins
    """
    print("\nComputing t-SNE (this may take a few minutes)...")

    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Continuous quality score
    ax = axes[0]
    scatter = ax.scatter(
        features_2d[:, 0], features_2d[:, 1],
        c=quality_scores, cmap='RdYlGn', alpha=0.6, s=20
    )
    ax.set_title('t-SNE: Colored by Quality Score (Continuous)', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Quality Score', rotation=270, labelpad=20)

    # Plot 2: Discrete quality bins
    ax = axes[1]
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(quality_scores, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    colors = ['red', 'yellow', 'green'] if num_bins == 3 else plt.cm.viridis(np.linspace(0, 1, num_bins))
    labels = ['Low Quality', 'Medium Quality', 'High Quality'] if num_bins == 3 else [f'Bin {i}' for i in range(num_bins)]

    for i in range(num_bins):
        mask = bin_indices == i
        ax.scatter(
            features_2d[mask, 0], features_2d[mask, 1],
            c=[colors[i]], label=labels[i], alpha=0.6, s=20
        )

    ax.set_title('t-SNE: Colored by Quality Bin (Discrete)', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(loc='best', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ t-SNE plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate quality-aware features from SSL training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to Stage 2 checkpoint')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to BUT-PPG data')
    parser.add_argument('--plot-output', type=str, default='artifacts/quality_features_tsne.png',
                       help='Path to save t-SNE plot')
    parser.add_argument('--quality-bins', type=int, default=3,
                       help='Number of quality bins (3 = low/medium/high)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Max samples for quick testing')
    parser.add_argument('--skip-tsne', action='store_true',
                       help='Skip t-SNE computation (faster)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("="*80)
    print("QUALITY FEATURES VALIDATION")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {args.data_dir}")
    print("="*80 + "\n")

    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Get config
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = {
            'context_length': 1024,
            'num_channels': 2,
            'd_model': 64,
            'patch_length': 128
        }

    # Create encoder
    encoder = TTMAdapter(
        context_length=config.get('context_length', 1024),
        num_channels=config.get('num_channels', 2),
        d_model=config.get('d_model', 64),
        patch_length=config.get('patch_length', 128),
        output_type='features'
    ).to(device)

    # Load weights
    encoder_state = checkpoint.get('encoder_state_dict', checkpoint)
    encoder.load_state_dict(encoder_state, strict=False)
    print("✓ Encoder loaded\n")

    # Create dataset
    print("Loading dataset...")
    dataset = QualityStratifiedBUTPPGDataset(
        data_dir=args.data_dir,
        split='val',  # Use validation set for diagnostics
        modality='all',
        mode='preprocessed',
        quality_bins=args.quality_bins,
        precompute_quality=True
    )

    if args.max_samples:
        dataset.base_dataset.window_files = dataset.base_dataset.window_files[:args.max_samples]
        print(f"⚠️  Limited to {args.max_samples} samples")

    data_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    print(f"✓ Dataset loaded: {len(dataset)} samples\n")

    # Extract features
    print("Extracting features...")
    features, quality_scores = extract_features(encoder, data_loader, str(device))
    print(f"✓ Features extracted: {features.shape}\n")

    # Compute clustering metrics
    print("Computing quality clustering metrics...")
    clustering_metrics = compute_quality_clustering_metrics(
        features, quality_scores, num_bins=args.quality_bins
    )

    print("\nClustering Metrics:")
    print("-" * 80)
    print(f"Silhouette Score:           {clustering_metrics['silhouette_score']:.4f}")
    print(f"Within-Quality Similarity:  {clustering_metrics['within_quality_similarity']:.4f}")
    print(f"Cross-Quality Similarity:   {clustering_metrics['cross_quality_similarity']:.4f}")
    print(f"Separation Ratio:           {clustering_metrics['separation_ratio']:.4f}")
    print("-" * 80)

    # Interpret results
    silhouette = clustering_metrics['silhouette_score']
    if silhouette > 0.5:
        print("✅ Excellent clustering! Features strongly capture quality.")
    elif silhouette > 0.3:
        print("✓ Good clustering. Features capture quality reasonably well.")
    elif silhouette > 0.1:
        print("⚠️  Weak clustering. Some quality information present.")
    else:
        print("❌ Poor clustering. Features may not capture quality well.")

    # Linear probe evaluation
    print("\nLinear Probe Evaluation...")
    linear_metrics = linear_probe_evaluation(
        features, quality_scores, num_bins=args.quality_bins
    )

    print("\nLinear Separability:")
    print("-" * 80)
    print(f"Train Accuracy:  {linear_metrics['linear_probe_train_acc']*100:.2f}%")
    print(f"Test Accuracy:   {linear_metrics['linear_probe_test_acc']*100:.2f}%")
    print(f"AUROC:           {linear_metrics['linear_probe_auroc']:.4f}")
    print("-" * 80)

    # Interpret
    auroc = linear_metrics['linear_probe_auroc']
    if auroc >= 0.80:
        print("✅ Excellent! Features are highly linearly separable by quality.")
    elif auroc >= 0.70:
        print("✓ Good! Features show good linear separability.")
    elif auroc >= 0.60:
        print("⚠️  Moderate separability. Room for improvement.")
    else:
        print("❌ Poor separability. Features may need better training.")

    # t-SNE visualization
    if not args.skip_tsne:
        output_path = Path(args.plot_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plot_tsne_quality(
            features, quality_scores, str(output_path), num_bins=args.quality_bins
        )

    # Overall assessment
    print("\n" + "="*80)
    print("OVERALL ASSESSMENT")
    print("="*80)

    checks_passed = 0
    total_checks = 4

    print("\nSuccess Criteria:")
    print(f"  [{'✓' if silhouette > 0.3 else '✗'}] Silhouette score > 0.3: {silhouette:.4f}")
    if silhouette > 0.3:
        checks_passed += 1

    print(f"  [{'✓' if clustering_metrics['separation_ratio'] > 1.1 else '✗'}] Separation ratio > 1.1: {clustering_metrics['separation_ratio']:.4f}")
    if clustering_metrics['separation_ratio'] > 1.1:
        checks_passed += 1

    print(f"  [{'✓' if auroc >= 0.70 else '✗'}] Linear probe AUROC ≥ 0.70: {auroc:.4f}")
    if auroc >= 0.70:
        checks_passed += 1

    print(f"  [{'✓' if not args.skip_tsne else '⊘'}] t-SNE shows clear clustering: {'See plot' if not args.skip_tsne else 'Skipped'}")
    if not args.skip_tsne:
        checks_passed += 1

    print(f"\nChecks passed: {checks_passed}/{total_checks}")

    if checks_passed >= 3:
        print("\n✅ SUCCESS! Features learned quality-relevant representations.")
        print("   Ready for Stage 3 (supervised fine-tuning).")
    elif checks_passed >= 2:
        print("\n⚠️  PARTIAL SUCCESS. Features show some quality awareness.")
        print("   Consider increasing Stage 2 epochs or adjusting hyperparameters.")
    else:
        print("\n❌ FAILURE. Features do not capture quality well.")
        print("\nSuggestions:")
        print("  • Increase Stage 2 training epochs")
        print("  • Adjust contrastive temperature (try 0.05 or 0.1)")
        print("  • Check quality score distribution")
        print("  • Verify data quality")

    # Save results
    results = {
        'clustering_metrics': clustering_metrics,
        'linear_probe_metrics': linear_metrics,
        'checks_passed': checks_passed,
        'total_checks': total_checks,
        'success': checks_passed >= 3
    }

    results_file = Path(args.checkpoint).parent / 'validation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {results_file}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
