#!/usr/bin/env python3
"""Complete Hybrid SSL Pipeline for High AUROC on BUT-PPG Quality Assessment.

This script orchestrates the full 3-stage SSL approach:
1. Stage 1: VitalDB MAE (existing checkpoint - general biosignal features)
2. Stage 2: BUT-PPG Quality-Aware SSL (NEW - domain adaptation + quality learning)
3. Stage 3: BUT-PPG Supervised Fine-tuning (task-specific refinement)

Target Performance:
- AUROC ≥ 0.85 (minimum acceptable)
- AUROC ≥ 0.88 (article benchmark target)

Baseline Comparison:
- IBM Pretrained: 0.622 AUROC
- VitalDB SSL only: 0.597 AUROC (❌ worse than baseline!)
- Random Init: 0.508 AUROC

Expected Improvement:
- Hybrid SSL (3 stages): 0.850 AUROC (✅ +23pp over IBM baseline)

Usage:
    # Full pipeline (recommended)
    python scripts/train_hybrid_ssl_pipeline.py \
        --vitaldb-checkpoint artifacts/foundation_model/best_model.pt \
        --data-dir data/processed/butppg/windows_with_labels \
        --output-dir artifacts/hybrid_pipeline \
        --target-auroc 0.85 \
        --stage2-epochs 50 \
        --stage3-epochs 30

    # Quick test (30 minutes)
    python scripts/train_hybrid_ssl_pipeline.py \
        --vitaldb-checkpoint artifacts/foundation_model/best_model.pt \
        --data-dir data/processed/butppg/windows_with_labels \
        --output-dir artifacts/hybrid_pipeline_test \
        --target-auroc 0.80 \
        --stage2-epochs 5 \
        --stage3-epochs 5 \
        --max-samples 1000

    # Skip Stage 2 (direct fine-tuning from VitalDB)
    python scripts/train_hybrid_ssl_pipeline.py \
        --vitaldb-checkpoint artifacts/foundation_model/best_model.pt \
        --data-dir data/processed/butppg/windows_with_labels \
        --output-dir artifacts/direct_finetune \
        --skip-stage2

    # Evaluate existing Stage 3 checkpoint
    python scripts/train_hybrid_ssl_pipeline.py \
        --stage3-checkpoint artifacts/hybrid_pipeline/stage3_finetuned/best_model.pt \
        --data-dir data/processed/butppg/windows_with_labels \
        --evaluate-only
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

import argparse
import json
import subprocess
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support

from src.models.ttm_adapter import TTMAdapter
from src.models.heads import MLPClassifier
from src.data.butppg_dataset import BUTPPGDataset
from src.utils.seed import set_seed


def run_stage2_quality_ssl(
    data_dir: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 5e-5,
    max_samples: Optional[int] = None,
    vitaldb_checkpoint: Optional[str] = None,
    use_ibm_pretrained: bool = False,
    ibm_variant: str = 'ibm-granite/granite-timeseries-ttm-r1',
    ibm_context_length: int = 1024,
    ibm_patch_size: int = 128
) -> str:
    """Run Stage 2: Quality-aware SSL on BUT-PPG.

    Args:
        data_dir: Path to BUT-PPG data
        output_dir: Output directory
        epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        max_samples: Max samples for testing
        vitaldb_checkpoint: Path to VitalDB SSL checkpoint (optional)
        use_ibm_pretrained: Whether to use IBM pretrained TTM
        ibm_variant: IBM TTM variant to use
        ibm_context_length: Context length for IBM TTM
        ibm_patch_size: Patch size for IBM TTM

    Returns:
        checkpoint_path: Path to best Stage 2 checkpoint
    """
    print("\n" + "="*80)
    print("[Stage 2/3] BUT-PPG Quality-Aware SSL")
    print("="*80)
    print("Objective: Bridge domain gap and learn quality-relevant features")
    print("Method: Contrastive learning with quality-based positive/negative pairs")
    print("="*80 + "\n")

    stage2_dir = Path(output_dir) / 'stage2_butppg_quality_ssl'
    stage2_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        'python3', 'scripts/continue_ssl_butppg_quality.py',
        '--data-dir', data_dir,
        '--output-dir', str(stage2_dir),
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
        '--lr', str(lr),
        '--contrastive-weight', '1.0',
        '--reconstruction-weight', '0.3',
        '--temperature', '0.07',
        '--balanced-sampling'
    ]

    # Add either VitalDB checkpoint or IBM pretrained config
    if use_ibm_pretrained:
        print(f"  Initializing from IBM pretrained TTM ({ibm_variant})")
        cmd.extend([
            '--use-ibm-pretrained',
            '--ibm-variant', ibm_variant,
            '--ibm-context-length', str(ibm_context_length),
            '--ibm-patch-size', str(ibm_patch_size)
        ])
    elif vitaldb_checkpoint:
        print(f"  Initializing from VitalDB checkpoint: {vitaldb_checkpoint}")
        cmd.extend(['--vitaldb-checkpoint', vitaldb_checkpoint])
    else:
        raise ValueError("Must provide either vitaldb_checkpoint or use_ibm_pretrained=True")

    if max_samples:
        cmd.extend(['--max-samples', str(max_samples)])

    # Run
    print(f"Running command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=True)

    checkpoint_path = str(stage2_dir / 'best_model.pt')

    print("\n✓ Stage 2 complete!")
    print(f"  Checkpoint: {checkpoint_path}")

    return checkpoint_path


def run_stage3_supervised_finetune(
    init_checkpoint: str,
    data_dir: str,
    output_dir: str,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-4,
    max_samples: Optional[int] = None
) -> str:
    """Run Stage 3: Supervised fine-tuning on BUT-PPG quality labels.

    Args:
        init_checkpoint: Initialization checkpoint (from Stage 2)
        data_dir: Path to BUT-PPG data
        output_dir: Output directory
        epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        max_samples: Max samples for testing

    Returns:
        checkpoint_path: Path to best Stage 3 checkpoint
    """
    print("\n" + "="*80)
    print("[Stage 3/3] BUT-PPG Supervised Fine-tuning")
    print("="*80)
    print("Objective: Fine-tune with quality labels for classification")
    print("Method: Supervised training with cross-entropy loss")
    print("="*80 + "\n")

    stage3_dir = Path(output_dir) / 'stage3_supervised_finetune'
    stage3_dir.mkdir(parents=True, exist_ok=True)

    # Build command - use existing fine-tuning script
    cmd = [
        'python3', 'scripts/finetune_butppg.py',
        '--pretrained', init_checkpoint,
        '--data-dir', data_dir,
        '--output-dir', str(stage3_dir),
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
        '--lr', str(lr),
        '--head-only-epochs', '3',  # Stage 1: head-only warmup
        '--unfreeze-last-n', '2',   # Stage 2: progressive unfreezing
        '--full-finetune',          # Stage 3: full fine-tuning enabled
        '--full-finetune-epochs', str(max(5, epochs - 3))  # Remaining epochs for full FT
    ]

    # Note: max_samples not supported by finetune_butppg.py
    # Dataset will use all available data

    # Run
    print(f"Running command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=True)

    checkpoint_path = str(stage3_dir / 'best_model.pt')

    print("\n✓ Stage 3 complete!")
    print(f"  Checkpoint: {checkpoint_path}")

    return checkpoint_path


@torch.no_grad()
def evaluate_quality_classification(
    checkpoint_path: str,
    data_dir: str,
    device: str = 'cuda'
) -> Dict[str, float]:
    """Evaluate quality classification performance.

    Args:
        checkpoint_path: Path to trained model checkpoint
        data_dir: Path to BUT-PPG data
        device: Device

    Returns:
        results: Dict with evaluation metrics
    """
    print("\n" + "="*80)
    print("[Evaluation] Testing on BUT-PPG Quality Assessment")
    print("="*80 + "\n")

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

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

    # Create model
    encoder = TTMAdapter(
        context_length=config.get('context_length', 1024),
        num_channels=config.get('num_channels', 2),
        d_model=config.get('d_model', 64),
        patch_length=config.get('patch_length', 128),
        output_type='features'
    ).to(device)

    # Classification head
    num_classes = config.get('num_classes', 3)  # poor/medium/good
    classifier = MLPClassifier(
        in_features=config.get('d_model', 64),
        num_classes=num_classes,
        hidden_dims=[128],  # Single hidden layer with 128 units
        dropout=0.1
    ).to(device)

    # Load weights
    if 'encoder_state_dict' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
    else:
        # Try loading full model
        model_state = checkpoint.get('model_state_dict', checkpoint)
        encoder_state = {k.replace('encoder.', ''): v for k, v in model_state.items() if k.startswith('encoder.')}
        classifier_state = {k.replace('classifier.', ''): v for k, v in model_state.items() if k.startswith('classifier.')}
        encoder.load_state_dict(encoder_state, strict=False)
        classifier.load_state_dict(classifier_state, strict=False)

    encoder.eval()
    classifier.eval()

    # Create test dataset
    test_dataset = BUTPPGDataset(
        data_dir=data_dir,
        split='test',
        modality='all',
        mode='preprocessed',
        return_labels=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4
    )

    print(f"Test set: {len(test_dataset)} samples")

    # Evaluate
    all_probs = []
    all_labels = []

    for batch in test_loader:
        if isinstance(batch, tuple):
            signals, labels = batch
        else:
            signals = batch['signal']
            labels = batch['label']

        signals = signals.to(device)

        # Forward pass
        features = encoder(signals)  # [B, P, D]
        features_pooled = features.mean(dim=1)  # [B, D]
        logits = classifier(features_pooled)  # [B, num_classes]

        probs = torch.softmax(logits, dim=1)

        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)  # [N, num_classes]
    all_labels = np.concatenate(all_labels, axis=0)  # [N]

    # Compute metrics
    predictions = np.argmax(all_probs, axis=1)

    # AUROC (one-vs-rest for multi-class)
    if num_classes == 2:
        auroc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        auroc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')

    # Accuracy
    accuracy = accuracy_score(all_labels, predictions)

    # Per-class accuracy
    class_accuracies = {}
    for cls in range(num_classes):
        mask = all_labels == cls
        if mask.sum() > 0:
            class_acc = accuracy_score(all_labels[mask], predictions[mask])
            class_accuracies[f'class_{cls}_acc'] = class_acc

    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, predictions, average='macro', zero_division=0
    )

    results = {
        'auroc': float(auroc),
        'accuracy': float(accuracy * 100),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        **class_accuracies
    }

    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"AUROC:      {results['auroc']:.4f}")
    print(f"Accuracy:   {results['accuracy']:.2f}%")
    print(f"Precision:  {results['precision']:.4f}")
    print(f"Recall:     {results['recall']:.4f}")
    print(f"F1 Score:   {results['f1']:.4f}")

    if class_accuracies:
        print("\nPer-class Accuracy:")
        for cls_name, cls_acc in class_accuracies.items():
            print(f"  {cls_name}: {cls_acc*100:.2f}%")

    print("="*80)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid SSL Pipeline for BUT-PPG Quality Assessment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Paths
    parser.add_argument('--vitaldb-checkpoint', type=str, default=None,
                       help='Path to VitalDB SSL checkpoint (Stage 1) - if not provided, uses IBM pretrained TTM')
    parser.add_argument('--stage2-checkpoint', type=str,
                       help='Path to existing Stage 2 checkpoint (skip Stage 2 if provided)')
    parser.add_argument('--stage3-checkpoint', type=str,
                       help='Path to existing Stage 3 checkpoint (evaluate only)')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to BUT-PPG data')
    parser.add_argument('--output-dir', type=str, default='artifacts/hybrid_pipeline',
                       help='Output directory')

    # IBM Pretrained TTM options
    parser.add_argument('--use-ibm-pretrained', action='store_true', default=False,
                       help='Use IBM pretrained TTM instead of VitalDB checkpoint')
    parser.add_argument('--ibm-variant', type=str, default='ibm-granite/granite-timeseries-ttm-r1',
                       help='IBM TTM variant to use')
    parser.add_argument('--ibm-context-length', type=int, default=1024,
                       help='Context length for IBM TTM (512, 1024, or 1536)')
    parser.add_argument('--ibm-patch-size', type=int, default=128,
                       help='Patch size for IBM TTM (64 or 128)')

    # Pipeline control
    parser.add_argument('--skip-stage2', action='store_true',
                       help='Skip Stage 2 (direct fine-tuning from VitalDB)')
    parser.add_argument('--evaluate-only', action='store_true',
                       help='Only evaluate existing checkpoint (requires --stage3-checkpoint)')

    # Training parameters
    parser.add_argument('--stage2-epochs', type=int, default=50,
                       help='Epochs for Stage 2')
    parser.add_argument('--stage3-epochs', type=int, default=30,
                       help='Epochs for Stage 3')
    parser.add_argument('--stage2-batch-size', type=int, default=128,
                       help='Batch size for Stage 2')
    parser.add_argument('--stage3-batch-size', type=int, default=64,
                       help='Batch size for Stage 3')
    parser.add_argument('--stage2-lr', type=float, default=5e-5,
                       help='Learning rate for Stage 2')
    parser.add_argument('--stage3-lr', type=float, default=1e-4,
                       help='Learning rate for Stage 3')

    # Target and testing
    parser.add_argument('--target-auroc', type=float, default=0.85,
                       help='Target AUROC to achieve')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Max samples for quick testing')

    # Other
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("HYBRID SSL PIPELINE FOR BUT-PPG QUALITY ASSESSMENT")
    print("="*80)
    print(f"Target AUROC: ≥{args.target_auroc}")
    print(f"Output: {args.output_dir}")
    print("="*80)

    # Evaluate-only mode
    if args.evaluate_only:
        if not args.stage3_checkpoint:
            raise ValueError("--stage3-checkpoint required for --evaluate-only mode")

        results = evaluate_quality_classification(
            args.stage3_checkpoint, args.data_dir, args.device
        )

        # Check target
        if results['auroc'] >= args.target_auroc:
            print(f"\n✅ SUCCESS! Achieved target AUROC ≥ {args.target_auroc}")
        else:
            print(f"\n⚠️  Below target (AUROC: {results['auroc']:.4f} < {args.target_auroc})")

        return

    # Stage 1: Initialize encoder (VitalDB SSL checkpoint OR IBM pretrained TTM)
    print("\n[Stage 1/3] Foundation Model Initialization")

    # Determine whether to use IBM pretrained or VitalDB checkpoint
    use_ibm = args.use_ibm_pretrained or (args.vitaldb_checkpoint is None)

    if use_ibm:
        print("  Source: IBM Pretrained TTM")
        print(f"  Variant: {args.ibm_variant}")
        print(f"  Context: {args.ibm_context_length}, Patch: {args.ibm_patch_size}")

        # Map to pretrained variant name
        if args.ibm_context_length == 512 and args.ibm_patch_size == 64:
            variant_name = "TTM-Base"
        elif args.ibm_context_length == 1024 and args.ibm_patch_size == 128:
            variant_name = "TTM-Enhanced"
        elif args.ibm_context_length == 1536 and args.ibm_patch_size == 128:
            variant_name = "TTM-Advanced"
        else:
            variant_name = "Custom"

        print(f"  Pretrained: {variant_name}")
        stage1_checkpoint = None  # Will initialize from scratch in Stage 2
        stage1_config = {
            'use_ibm_pretrained': True,
            'ibm_variant': args.ibm_variant,
            'context_length': args.ibm_context_length,
            'patch_size': args.ibm_patch_size,
            'variant_name': variant_name
        }
    else:
        print("  Source: VitalDB SSL Checkpoint")
        print(f"  Checkpoint: {args.vitaldb_checkpoint}")
        stage1_checkpoint = args.vitaldb_checkpoint
        stage1_config = {
            'use_ibm_pretrained': False
        }

    # Stage 2: Quality-aware SSL on BUT-PPG
    if args.skip_stage2:
        print("\n[Stage 2/3] BUT-PPG Quality-Aware SSL")
        print("  Status: ⊘ Skipped (using Stage 1 checkpoint directly)")
        stage2_checkpoint = stage1_checkpoint
    elif args.stage2_checkpoint:
        print("\n[Stage 2/3] BUT-PPG Quality-Aware SSL")
        print("  Status: ✓ Using existing checkpoint")
        print(f"  Checkpoint: {args.stage2_checkpoint}")
        stage2_checkpoint = args.stage2_checkpoint
    else:
        # Pass either checkpoint path or IBM config
        if use_ibm:
            stage2_checkpoint = run_stage2_quality_ssl(
                vitaldb_checkpoint=None,  # No checkpoint
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                epochs=args.stage2_epochs,
                batch_size=args.stage2_batch_size,
                lr=args.stage2_lr,
                max_samples=args.max_samples,
                use_ibm_pretrained=True,
                ibm_variant=args.ibm_variant,
                ibm_context_length=args.ibm_context_length,
                ibm_patch_size=args.ibm_patch_size
            )
        else:
            stage2_checkpoint = run_stage2_quality_ssl(
                vitaldb_checkpoint=stage1_checkpoint,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                epochs=args.stage2_epochs,
                batch_size=args.stage2_batch_size,
                lr=args.stage2_lr,
                max_samples=args.max_samples
            )

    # Stage 3: Supervised fine-tuning
    stage3_checkpoint = run_stage3_supervised_finetune(
        init_checkpoint=stage2_checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.stage3_epochs,
        batch_size=args.stage3_batch_size,
        lr=args.stage3_lr,
        max_samples=args.max_samples
    )

    # Evaluation
    results = evaluate_quality_classification(
        stage3_checkpoint, args.data_dir, args.device
    )

    # Final summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"Final AUROC: {results['auroc']:.4f}")
    print(f"Final Accuracy: {results['accuracy']:.2f}%")
    print(f"Target AUROC: {args.target_auroc}")

    if results['auroc'] >= args.target_auroc:
        print(f"\n✅ SUCCESS! Achieved target AUROC ≥ {args.target_auroc}")
        improvement = (results['auroc'] - 0.622) * 100  # vs IBM baseline
        print(f"Improvement over IBM baseline: +{improvement:.1f}pp")
    else:
        print(f"\n⚠️  Below target (AUROC: {results['auroc']:.4f} < {args.target_auroc})")
        print("\nSuggestions for improvement:")
        print("  • Increase Stage 2 epochs (try 100)")
        print("  • Adjust contrastive temperature (try 0.05 or 0.1)")
        print("  • Use balanced quality sampling")
        print("  • Increase Stage 3 training data")

    # Save results
    results_file = output_dir / 'final_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'results': results,
            'target_auroc': args.target_auroc,
            'success': results['auroc'] >= args.target_auroc,
            'args': vars(args)
        }, f, indent=2)

    print(f"\n✓ Results saved to: {results_file}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
