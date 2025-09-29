#!/usr/bin/env python
"""Evaluate trained TTM model on VitalDB downstream tasks.

Usage:
    python scripts/evaluate_task.py --task hypotension_5min --checkpoint path/to/model.pt
    python scripts/evaluate_task.py --task blood_pressure_both --checkpoint path/to/model.pt
    python scripts/evaluate_task.py --list-tasks
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tasks import get_task, list_tasks, get_task_info
from src.benchmarks.tracker import BenchmarkTracker, load_vitaldb_clinical_data
from src.models.ttm_adapter import create_ttm_model
from src.utils.io import load_yaml
from src.data.vitaldb_loader import load_channel


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate TTM model on VitalDB downstream tasks'
    )
    
    # Task selection
    parser.add_argument(
        '--task',
        type=str,
        help='Task name (e.g., hypotension_5min, blood_pressure_both)'
    )
    parser.add_argument(
        '--list-tasks',
        action='store_true',
        help='List all available tasks'
    )
    parser.add_argument(
        '--task-info',
        type=str,
        help='Show information about a specific task'
    )
    
    # Model
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--model-config',
        type=str,
        default='configs/model.yaml',
        help='Model configuration file'
    )
    
    # Data
    parser.add_argument(
        '--case-ids',
        type=str,
        nargs='+',
        help='Specific case IDs to evaluate (optional)'
    )
    parser.add_argument(
        '--split-file',
        type=str,
        help='JSON file with train/val/test splits'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Which split to evaluate'
    )
    parser.add_argument(
        '--max-cases',
        type=int,
        default=None,
        help='Maximum number of cases to evaluate'
    )
    
    # Output
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/evaluations',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--compare-benchmarks',
        action='store_true',
        default=True,
        help='Compare results to published benchmarks'
    )
    parser.add_argument(
        '--generate-report',
        action='store_true',
        default=True,
        help='Generate HTML benchmark report'
    )
    
    # Device
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for evaluation'
    )
    
    return parser.parse_args()


def load_model(checkpoint_path: str, model_config_path: str, device: str):
    """Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        model_config_path: Path to model config
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    # Load config
    config = load_yaml(model_config_path)
    
    # Create model
    model = create_ttm_model(config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"✓ Loaded model from {checkpoint_path}")
    model.print_parameter_summary()
    
    return model


def evaluate_task(
    task,
    model,
    case_ids: list,
    clinical_df: pd.DataFrame,
    device: str,
    max_cases: int = None
) -> dict:
    """Evaluate model on a specific task.
    
    Args:
        task: Task instance
        model: Trained model
        case_ids: List of case IDs to evaluate
        clinical_df: Clinical data DataFrame
        device: Device for inference
        max_cases: Maximum number of cases to evaluate
        
    Returns:
        Dictionary with predictions, targets, and metadata
    """
    if max_cases:
        case_ids = case_ids[:max_cases]
    
    all_predictions = []
    all_targets = []
    valid_cases = []
    
    print(f"\nEvaluating {task.config.name} on {len(case_ids)} cases...")
    
    for case_id in tqdm(case_ids, desc="Evaluating cases"):
        try:
            # Load signals
            signals = {}
            for channel in task.config.required_channels:
                signal, fs = load_channel(
                    case_id,
                    channel,
                    duration_sec=task.config.window_size_s
                )
                signals[channel] = signal
            
            # Validate signals
            valid, reasons = task.validate_signals(signals)
            if not valid:
                continue
            
            # Get clinical data
            clinical_data = clinical_df[clinical_df['caseid'] == int(case_id)].iloc[0]
            
            # Generate labels
            labels = task.generate_labels(case_id, signals, clinical_data)
            
            # Prepare input for model
            # Convert signals to tensor [batch, channels, time]
            signal_array = np.stack([signals[ch] for ch in task.config.required_channels])
            x = torch.tensor(signal_array, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Model inference
            with torch.no_grad():
                predictions = model(x)
            
            predictions = predictions.cpu().numpy()
            
            all_predictions.append(predictions)
            
            if isinstance(labels, dict):
                # For tasks with multiple outputs
                all_targets.append(labels)
            else:
                all_targets.append(labels)
            
            valid_cases.append(case_id)
            
        except Exception as e:
            print(f"Error processing case {case_id}: {e}")
            continue
    
    print(f"✓ Successfully evaluated {len(valid_cases)}/{len(case_ids)} cases")
    
    # Aggregate results
    predictions_array = np.concatenate(all_predictions, axis=0)
    
    # Convert targets to array
    if isinstance(all_targets[0], dict):
        # Handle dict targets (e.g., blood pressure with SBP/DBP)
        targets_array = {
            key: np.array([t[key] for t in all_targets])
            for key in all_targets[0].keys()
            if isinstance(all_targets[0][key], (int, float, np.ndarray))
        }
    else:
        targets_array = np.array(all_targets)
    
    results = {
        'predictions': predictions_array,
        'targets': targets_array,
        'case_ids': valid_cases,
        'n_patients': len(valid_cases)
    }
    
    return results


def main():
    args = parse_args()
    
    # List tasks
    if args.list_tasks:
        print("\n" + "="*60)
        print("AVAILABLE VITALDB DOWNSTREAM TASKS")
        print("="*60)
        tasks = list_tasks()
        for task_name in tasks:
            print(f"  - {task_name}")
        print("="*60 + "\n")
        return
    
    # Show task info
    if args.task_info:
        info = get_task_info(args.task_info)
        print(f"\n{'='*60}")
        print(f"TASK INFO: {info['name']}")
        print(f"{'='*60}")
        print(f"Type: {info['type']}")
        print(f"Required channels: {info['required_channels']}")
        if 'num_classes' in info:
            print(f"Number of classes: {info['num_classes']}")
        if 'target_dim' in info:
            print(f"Target dimension: {info['target_dim']}")
        if 'clinical_threshold' in info:
            print(f"Clinical threshold: {info['clinical_threshold']}")
        
        print(f"\nPublished Benchmarks:")
        for bench in info['benchmarks']:
            print(f"\n  {bench['paper']} ({bench['year']})")
            print(f"  Metrics: {bench['metrics']}")
            if bench['notes']:
                print(f"  Notes: {bench['notes']}")
        print(f"{'='*60}\n")
        return
    
    # Validate required arguments
    if not args.task:
        print("Error: --task required")
        return
    if not args.checkpoint:
        print("Error: --checkpoint required")
        return
    
    # Load task
    task = get_task(args.task)
    print(f"\n✓ Loaded task: {task.config.name}")
    
    # Load model
    model = load_model(args.checkpoint, args.model_config, args.device)
    
    # Load clinical data
    print("\nLoading VitalDB clinical data...")
    clinical_df = load_vitaldb_clinical_data()
    print(f"✓ Loaded {len(clinical_df)} cases")
    
    # Get case IDs to evaluate
    if args.case_ids:
        case_ids = args.case_ids
    elif args.split_file:
        import json
        with open(args.split_file) as f:
            splits = json.load(f)
        case_ids = splits[args.split]
    else:
        # Use all available cases
        case_ids = clinical_df['caseid'].astype(str).tolist()
    
    print(f"\nEvaluating on {len(case_ids)} cases from {args.split} split")
    
    # Evaluate
    results = evaluate_task(
        task,
        model,
        case_ids,
        clinical_df,
        args.device,
        max_cases=args.max_cases
    )
    
    # Compute metrics
    print("\nComputing evaluation metrics...")
    
    # Handle different target formats
    if isinstance(results['targets'], dict):
        # Multiple targets (e.g., SBP and DBP)
        targets = np.stack([
            results['targets'][k] for k in sorted(results['targets'].keys())
            if isinstance(results['targets'][k], np.ndarray)
        ], axis=1)
    else:
        targets = results['targets']
    
    metrics = task.evaluate(
        results['predictions'],
        targets,
        return_detailed=True
    )
    
    metrics['n_patients'] = results['n_patients']
    
    # Initialize benchmark tracker
    tracker = BenchmarkTracker(output_dir=args.output_dir)
    
    # Log results
    tracker.log_result(
        task_name=task.config.name,
        metrics=metrics,
        model_name="TTM-VitalDB",
        n_patients=results['n_patients'],
        split=args.split
    )
    
    # Compare to benchmarks
    if args.compare_benchmarks:
        print("\nComparing to published benchmarks...")
        comparison = tracker.compare_to_benchmarks(task, metrics)
        
        # Plot comparison for key metrics
        if 'auroc' in metrics:
            tracker.plot_comparison(
                task.config.name,
                'auroc',
                comparison
            )
        if 'mae' in metrics:
            tracker.plot_comparison(
                task.config.name,
                'mae',
                comparison
            )
    
    # Generate report
    if args.generate_report:
        print("\nGenerating benchmark report...")
        tracker.generate_report()
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Task: {task.config.name}")
    print(f"Cases evaluated: {results['n_patients']}")
    print(f"Results saved to: {args.output_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
