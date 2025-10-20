#!/usr/bin/env python3
"""
Master Downstream Evaluation Script

Runs all 6 downstream tasks (3 VitalDB + 3 BUT-PPG) and generates
comprehensive benchmark comparison report.

Usage:
    # Evaluate VitalDB tasks only
    python scripts/run_downstream_evaluation.py \
        --vitaldb-checkpoint artifacts/vitaldb_finetuned/best_model.pt \
        --vitaldb-data data/processed/vitaldb \
        --output-dir artifacts/downstream_evaluation

    # Evaluate BUT-PPG tasks only
    python scripts/run_downstream_evaluation.py \
        --butppg-checkpoint artifacts/butppg_finetuned/best_model.pt \
        --butppg-data data/processed/butppg \
        --output-dir artifacts/downstream_evaluation

    # Evaluate both
    python scripts/run_downstream_evaluation.py \
        --vitaldb-checkpoint artifacts/vitaldb_finetuned/best_model.pt \
        --butppg-checkpoint artifacts/butppg_finetuned/best_model.pt \
        --vitaldb-data data/processed/vitaldb \
        --butppg-data data/processed/butppg \
        --output-dir artifacts/downstream_evaluation

This will:
1. Load fine-tuned checkpoints for each dataset
2. Run all applicable downstream tasks
3. Compute all metrics with confidence intervals
4. Compare against article benchmarks
5. Generate comprehensive report with tables and plots
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from torch.utils.data import DataLoader, TensorDataset

from src.eval.tasks.vitaldb_tasks import run_all_vitaldb_tasks
from src.eval.tasks.butppg_tasks import run_all_butppg_tasks
from src.eval.reports.benchmark_comparison import BenchmarkComparator
from src.models.ttm_adapter import create_ttm_model


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run downstream task evaluation and benchmark comparison'
    )

    # Checkpoints
    parser.add_argument('--vitaldb-checkpoint', type=str,
                       help='Path to VitalDB fine-tuned checkpoint')
    parser.add_argument('--butppg-checkpoint', type=str,
                       help='Path to BUT-PPG fine-tuned checkpoint')

    # Data directories
    parser.add_argument('--vitaldb-data', type=str,
                       help='Path to VitalDB processed data directory')
    parser.add_argument('--butppg-data', type=str,
                       help='Path to BUT-PPG processed data directory')

    # Output
    parser.add_argument('--output-dir', default='artifacts/downstream_evaluation',
                       help='Output directory for results')

    # Evaluation options
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--no-ci', action='store_true',
                       help='Skip confidence interval computation (faster)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run on')

    # Model info (for report)
    parser.add_argument('--model-name', default='TTM Foundation Model',
                       help='Model name for report')
    parser.add_argument('--model-params', default='~1M',
                       help='Number of parameters for report')

    return parser.parse_args()


def load_model(checkpoint_path: str, device: str) -> nn.Module:
    """
    Load a fine-tuned model from checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load on

    Returns:
        Loaded model in eval mode
    """
    print(f"\nLoading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract config and state dict
    if isinstance(checkpoint, dict):
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            raise ValueError("Checkpoint missing 'config' key")

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            raise ValueError("Checkpoint missing model state dict")
    else:
        raise ValueError("Checkpoint format not recognized")

    # Infer missing config fields from state dict
    # The fine-tuning script saves the argparse config, not the full model config!
    # We need to build a proper model config by inspecting the state dict

    print("  Inspecting checkpoint config and state dict...")

    # Check if this is a model config or training config (argparse)
    is_training_config = 'pretrained' in config or 'data_dir' in config

    if is_training_config:
        print("  ⚠️  Checkpoint contains training config, not model config")
        print("  Building model config from state dict...")

        # Build model config from state dict
        new_config = {}

        # Infer input_channels from embedding layer
        for key in state_dict.keys():
            if 'backbone.input_projection.weight' in key or 'input_embedding' in key:
                weight_shape = state_dict[key].shape
                # Shape should be [d_model, input_channels, patch_size]
                if len(weight_shape) >= 2:
                    new_config['input_channels'] = weight_shape[1]
                    print(f"    Inferred input_channels={new_config['input_channels']} from {key}: {weight_shape}")
                break

        # Infer context_length and patch_size from positional encoding
        for key in state_dict.keys():
            if 'position' in key.lower():
                pos_shape = state_dict[key].shape
                # Position encoding shape gives us num_patches
                if len(pos_shape) >= 1:
                    num_patches = pos_shape[0]
                    print(f"    Found num_patches={num_patches} from {key}")
                    break

        # Infer num_classes from head
        # Look for the actual classification/regression head, not internal TTM layers
        head_key = None
        for key in state_dict.keys():
            # Look for these patterns (from fine-tuning script)
            if key in ['head.fc.weight', 'head.weight', 'classifier.weight']:
                head_key = key
                break
            # Fallback: head.something.weight but NOT encoder.head (TTM internal)
            if 'head.' in key and 'weight' in key and 'encoder.head' not in key:
                head_key = key
                break

        if head_key:
            head_shape = state_dict[head_key].shape
            if len(head_shape) >= 1:
                new_config['num_classes'] = head_shape[0]
                new_config['task'] = 'classification'
                print(f"    Inferred num_classes={new_config['num_classes']} from {head_key}: {head_shape}")
        else:
            # Default to binary classification if no head found
            new_config['num_classes'] = 2
            new_config['task'] = 'classification'
            print(f"    ⚠️  Could not find classification head, defaulting to num_classes=2")

        # Set defaults
        new_config['variant'] = 'ibm-granite/granite-timeseries-ttm-r1'
        new_config['head_type'] = 'linear'

        # Try to get context_length and patch_size from original config if available
        if 'finetune_channels' in config:
            new_config['input_channels'] = config['finetune_channels']
        if 'input_channels' in config and 'input_channels' not in new_config:
            new_config['input_channels'] = config['input_channels']

        # Use reasonable defaults based on typical fine-tuning setup
        if 'context_length' not in new_config:
            new_config['context_length'] = 1024  # Standard for BUT-PPG
        if 'patch_size' not in new_config:
            new_config['patch_size'] = 128  # Standard for 1024 context

        config = new_config
        print(f"    Built model config: {config}")

    else:
        # This is already a model config, just fill in missing fields
        if 'num_classes' not in config or config['num_classes'] is None:
            # Look for classification head in state dict
            for key in state_dict.keys():
                if 'head' in key and 'weight' in key:
                    head_shape = state_dict[key].shape
                    if len(head_shape) >= 1:
                        config['num_classes'] = head_shape[0]
                        print(f"    Inferred num_classes={config['num_classes']} from head shape: {head_shape}")
                        break

        # Ensure required fields are set
        if 'variant' not in config:
            config['variant'] = 'ibm-granite/granite-timeseries-ttm-r1'
        if 'head_type' not in config:
            config['head_type'] = 'linear'
        if 'task' not in config:
            config['task'] = 'classification'

    # Create model
    model = create_ttm_model(config)
    model.load_state_dict(state_dict, strict=False)  # Use strict=False in case of minor mismatches
    model.to(device)
    model.eval()

    print(f"  ✓ Model loaded: {model.__class__.__name__}")
    print(f"  ✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


def load_vitaldb_data(data_dir: str, batch_size: int) -> Dict[str, DataLoader]:
    """
    Load VitalDB test data for all tasks

    Expected structure:
        data_dir/
            hypotension/test.npz  # {'signals': [N,C,T], 'labels': [N]}
            blood_pressure/test.npz

    Args:
        data_dir: Directory containing processed VitalDB data
        batch_size: Batch size for DataLoader

    Returns:
        Dict with keys 'hypotension', 'bp_estimation' containing DataLoaders
    """
    data_dir = Path(data_dir)
    loaders = {}

    print("\nLoading VitalDB test data...")

    # Hypotension task
    hypo_path = data_dir / 'hypotension' / 'test.npz'
    if hypo_path.exists():
        data = np.load(hypo_path)
        signals = torch.from_numpy(data['signals']).float()
        labels = torch.from_numpy(data['labels']).long()

        # Check for subject IDs
        if 'subject_ids' in data:
            subject_ids = torch.from_numpy(data['subject_ids'])
            dataset = TensorDataset(signals, labels, subject_ids)
        else:
            dataset = TensorDataset(signals, labels)

        loaders['hypotension'] = DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )
        print(f"  ✓ Hypotension: {len(dataset)} samples")
    else:
        print(f"  ⚠ Hypotension data not found: {hypo_path}")

    # Blood pressure task
    bp_path = data_dir / 'blood_pressure' / 'test.npz'
    if bp_path.exists():
        data = np.load(bp_path)
        signals = torch.from_numpy(data['signals']).float()
        labels = torch.from_numpy(data['labels']).float()

        if 'subject_ids' in data:
            subject_ids = torch.from_numpy(data['subject_ids'])
            dataset = TensorDataset(signals, labels, subject_ids)
        else:
            dataset = TensorDataset(signals, labels)

        loaders['bp_estimation'] = DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )
        print(f"  ✓ Blood Pressure: {len(dataset)} samples")
    else:
        print(f"  ⚠ Blood Pressure data not found: {bp_path}")

    return loaders


def load_butppg_data(data_dir: str, batch_size: int) -> Dict[str, DataLoader]:
    """
    Load BUT-PPG test data for all tasks

    Supports two structures:
    1. Legacy: data_dir/quality/test.npz, data_dir/heart_rate/test.npz
    2. Unified: data_dir/windows/test/test_windows.npz (with all clinical labels)

    Args:
        data_dir: Directory containing processed BUT-PPG data
        batch_size: Batch size for DataLoader

    Returns:
        Dict with keys 'quality', 'hr_estimation', 'motion' containing DataLoaders
    """
    data_dir = Path(data_dir)
    loaders = {}

    print("\nLoading BUT-PPG test data...")

    # Try unified format first (from prepare_all_data.py)
    unified_paths = [
        data_dir / 'windows' / 'test' / 'test_windows.npz',
        data_dir / 'test' / 'test_windows.npz',
        data_dir / 'test_windows.npz',
    ]

    unified_data = None
    for path in unified_paths:
        if path.exists():
            print(f"  ✓ Found unified data: {path}")
            unified_data = np.load(path)
            break

    if unified_data is not None:
        # Load from unified format (all labels in one file)
        print("  Loading from unified format with clinical labels...")

        # Get signals (handle both 'data' and 'signals' keys)
        if 'data' in unified_data:
            signals_array = unified_data['data']
        elif 'signals' in unified_data:
            signals_array = unified_data['signals']
        else:
            raise KeyError(f"No 'data' or 'signals' key found in {path}")

        signals = torch.from_numpy(signals_array).float()

        # Handle shape: [N, T, C] → [N, C, T] if needed
        if signals.shape[1] == 1024 and signals.shape[2] == 5:
            print(f"    Transposing from [N, T, C] to [N, C, T]")
            signals = signals.transpose(1, 2)  # [N, 1024, 5] → [N, 5, 1024]

        # Quality task
        if 'quality' in unified_data:
            quality_labels = torch.from_numpy(unified_data['quality']).long()
            # Filter out missing values (-1)
            valid_mask = quality_labels != -1
            if valid_mask.sum() > 0:
                dataset = TensorDataset(signals[valid_mask], quality_labels[valid_mask])
                loaders['quality'] = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                print(f"    ✓ Quality: {len(dataset)} samples")
        elif 'labels' in unified_data:
            # Fallback to 'labels' key (backwards compatibility)
            quality_labels = torch.from_numpy(unified_data['labels']).long()
            dataset = TensorDataset(signals, quality_labels)
            loaders['quality'] = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            print(f"    ✓ Quality (from 'labels'): {len(dataset)} samples")

        # Heart rate task
        if 'hr' in unified_data:
            hr_labels = torch.from_numpy(unified_data['hr']).float()
            valid_mask = hr_labels != -1
            if valid_mask.sum() > 0:
                dataset = TensorDataset(signals[valid_mask], hr_labels[valid_mask])
                loaders['hr_estimation'] = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                print(f"    ✓ Heart Rate: {len(dataset)} samples")

        # Motion task
        if 'motion' in unified_data:
            motion_labels = torch.from_numpy(unified_data['motion']).long()
            valid_mask = motion_labels != -1
            if valid_mask.sum() > 0:
                dataset = TensorDataset(signals[valid_mask], motion_labels[valid_mask])
                loaders['motion'] = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                print(f"    ✓ Motion: {len(dataset)} samples")
            else:
                print(f"    ⚠️  Motion: All labels are -1 (dataset issue)")

        # Blood pressure tasks (bonus!)
        if 'bp_systolic' in unified_data and 'bp_diastolic' in unified_data:
            bp_sys = torch.from_numpy(unified_data['bp_systolic']).float()
            bp_dia = torch.from_numpy(unified_data['bp_diastolic']).float()
            valid_mask = (bp_sys != -1) & (bp_dia != -1)
            if valid_mask.sum() > 0:
                bp_labels = torch.stack([bp_sys[valid_mask], bp_dia[valid_mask]], dim=1)
                dataset = TensorDataset(signals[valid_mask], bp_labels)
                loaders['bp_estimation'] = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                print(f"    ✓ Blood Pressure: {len(dataset)} samples")

        return loaders

    # Fallback to legacy format (separate directories per task)
    print("  Unified format not found, trying legacy format...")

    # Quality task (legacy)
    quality_path = data_dir / 'quality' / 'test.npz'
    if quality_path.exists():
        data = np.load(quality_path)
        signals = torch.from_numpy(data['signals']).float()
        labels = torch.from_numpy(data['labels']).long()

        if 'subject_ids' in data:
            subject_ids = torch.from_numpy(data['subject_ids'])
            dataset = TensorDataset(signals, labels, subject_ids)
        else:
            dataset = TensorDataset(signals, labels)

        loaders['quality'] = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        print(f"  ✓ Quality: {len(dataset)} samples")
    else:
        print(f"  ⚠ Quality data not found: {quality_path}")

    # Heart rate task (legacy)
    hr_path = data_dir / 'heart_rate' / 'test.npz'
    if hr_path.exists():
        data = np.load(hr_path)
        signals = torch.from_numpy(data['signals']).float()
        labels = torch.from_numpy(data['labels']).float()

        if 'subject_ids' in data:
            subject_ids = torch.from_numpy(data['subject_ids'])
            dataset = TensorDataset(signals, labels, subject_ids)
        else:
            dataset = TensorDataset(signals, labels)

        loaders['hr_estimation'] = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        print(f"  ✓ Heart Rate: {len(dataset)} samples")
    else:
        print(f"  ⚠ Heart Rate data not found: {hr_path}")

    # Motion task (legacy)
    motion_path = data_dir / 'motion' / 'test.npz'
    if motion_path.exists():
        data = np.load(motion_path)
        signals = torch.from_numpy(data['signals']).float()
        labels = torch.from_numpy(data['labels']).long()

        if 'subject_ids' in data:
            subject_ids = torch.from_numpy(data['subject_ids'])
            dataset = TensorDataset(signals, labels, subject_ids)
        else:
            dataset = TensorDataset(signals, labels)

        loaders['motion'] = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        print(f"  ✓ Motion: {len(dataset)} samples")
    else:
        print(f"  ⚠ Motion data not found: {motion_path}")

    return loaders


def main():
    args = parse_args()

    # Validate inputs
    if not args.vitaldb_checkpoint and not args.butppg_checkpoint:
        print("Error: Must specify at least one checkpoint (--vitaldb-checkpoint or --butppg-checkpoint)")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("DOWNSTREAM TASK EVALUATION & BENCHMARK COMPARISON")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Compute CI: {not args.no_ci}")

    all_results = {}

    # ==========================
    # VITALDB EVALUATION
    # ==========================
    if args.vitaldb_checkpoint and args.vitaldb_data:
        print("\n" + "="*80)
        print("EVALUATING VITALDB TASKS")
        print("="*80)

        # Load model
        vitaldb_model = load_model(args.vitaldb_checkpoint, args.device)

        # Load data
        vitaldb_loaders = load_vitaldb_data(args.vitaldb_data, args.batch_size)

        # Run evaluation
        if 'hypotension' in vitaldb_loaders and 'bp_estimation' in vitaldb_loaders:
            vitaldb_results = run_all_vitaldb_tasks(
                model=vitaldb_model,
                hypotension_loader=vitaldb_loaders['hypotension'],
                bp_loader=vitaldb_loaders['bp_estimation'],
                device=args.device,
                compute_ci=not args.no_ci
            )
            all_results['vitaldb'] = vitaldb_results
        else:
            print("⚠ Warning: Missing VitalDB data, skipping evaluation")
            all_results['vitaldb'] = None

    # ==========================
    # BUT-PPG EVALUATION
    # ==========================
    if args.butppg_checkpoint and args.butppg_data:
        print("\n" + "="*80)
        print("EVALUATING BUT-PPG TASKS")
        print("="*80)

        # Load model
        butppg_model = load_model(args.butppg_checkpoint, args.device)

        # Load data
        butppg_loaders = load_butppg_data(args.butppg_data, args.batch_size)

        # Run evaluation
        if 'quality' in butppg_loaders:
            butppg_results = run_all_butppg_tasks(
                quality_model=butppg_model,
                hr_model=butppg_model if 'hr_estimation' in butppg_loaders else None,
                motion_model=butppg_model if 'motion' in butppg_loaders else None,
                quality_loader=butppg_loaders['quality'],
                hr_loader=butppg_loaders.get('hr_estimation', None),
                motion_loader=butppg_loaders.get('motion', None),
                device=args.device,
                compute_ci=not args.no_ci
            )
            all_results['butppg'] = butppg_results
        else:
            print("⚠ Warning: Missing BUT-PPG data, skipping evaluation")
            all_results['butppg'] = None

    # ==========================
    # GENERATE REPORT
    # ==========================
    if all_results.get('vitaldb') or all_results.get('butppg'):
        print("\n" + "="*80)
        print("GENERATING BENCHMARK COMPARISON REPORT")
        print("="*80)

        comparator = BenchmarkComparator(output_dir)

        model_info = {
            'name': args.model_name,
            'parameters': args.model_params,
            'architecture': 'IBM Granite TTM'
        }

        # Generate plots
        if all_results.get('vitaldb') and all_results.get('butppg'):
            comparator.generate_comparison_plots(
                all_results['vitaldb'],
                all_results['butppg']
            )
            comparator.generate_full_report(
                all_results['vitaldb'],
                all_results['butppg'],
                model_info
            )
        elif all_results.get('vitaldb'):
            # VitalDB only - generate partial report
            vitaldb_md = comparator.generate_vitaldb_comparison_table(all_results['vitaldb'])
            with open(output_dir / 'vitaldb_comparison.md', 'w') as f:
                f.write(vitaldb_md)
            print(f"✓ Saved: {output_dir / 'vitaldb_comparison.md'}")
        elif all_results.get('butppg'):
            # BUT-PPG only - generate partial report
            butppg_md = comparator.generate_butppg_comparison_table(all_results['butppg'])
            with open(output_dir / 'butppg_comparison.md', 'w') as f:
                f.write(butppg_md)
            print(f"✓ Saved: {output_dir / 'butppg_comparison.md'}")

        # Save JSON results
        with open(output_dir / 'all_results.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"✓ Saved: {output_dir / 'all_results.json'}")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")

    if (output_dir / 'benchmark_report.html').exists():
        print(f"  📊 HTML Report: {output_dir / 'benchmark_report.html'}")
    if (output_dir / 'benchmark_comparison.png').exists():
        print(f"  📈 Plots: {output_dir / 'benchmark_comparison.png'}")
    if (output_dir / 'all_results.json').exists():
        print(f"  📄 JSON: {output_dir / 'all_results.json'}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if all_results.get('vitaldb'):
        vitaldb = all_results['vitaldb']
        print("\nVitalDB Tasks:")
        if 'hypotension' in vitaldb:
            hypo = vitaldb['hypotension']
            print(f"  Hypotension: AUROC={hypo['auroc']:.3f} {'✅ PASS' if hypo['target_met'] else '❌ FAIL'}")
        if 'bp_estimation' in vitaldb:
            bp = vitaldb['bp_estimation']
            print(f"  BP Estimation: MAE={bp['mae']:.2f} mmHg {'✅ PASS' if bp['mae_target_met'] else '❌ FAIL'}")
            print(f"  AAMI Compliance: {'✅ PASS' if bp['aami_compliant'] else '❌ FAIL'}")

    if all_results.get('butppg'):
        butppg = all_results['butppg']
        print("\nBUT-PPG Tasks:")
        if 'quality' in butppg:
            qual = butppg['quality']
            print(f"  Quality: AUROC={qual['auroc']:.3f} {'✅ PASS' if qual['target_met'] else '❌ FAIL'}")
        if 'hr_estimation' in butppg:
            hr = butppg['hr_estimation']
            print(f"  Heart Rate: MAE={hr['mae']:.2f} bpm {'✅ PASS' if hr['target_met'] else '❌ FAIL'}")
        if 'motion' in butppg:
            motion = butppg['motion']
            print(f"  Motion: Accuracy={motion['accuracy']:.3f} {'✅ PASS' if motion['target_met'] else '❌ FAIL'}")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
