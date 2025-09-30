#!/usr/bin/env python3
"""Quick start example for VitalDB downstream task evaluation.

This script demonstrates the complete workflow:
1. Load a trained TTM model
2. Evaluate on multiple downstream tasks
3. Compare results to published benchmarks
4. Generate evaluation report

Usage:
    python3 examples/quick_start_tasks.py

Requirements:
    - Trained TTM model checkpoint
    - VitalDB access (or mock data)
    - ~10 minutes runtime for 50 cases
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tasks import get_task, list_tasks
from src.benchmarks import BenchmarkTracker
from src.models.ttm_adapter import create_ttm_model
from src.utils.io import load_yaml


def main():
    print("=" * 80)
    print("VitalDB Downstream Tasks - Quick Start Example")
    print("=" * 80)
    
    # Step 1: List available tasks
    print("\n📋 Step 1: Available Tasks")
    print("-" * 80)
    tasks = list_tasks()
    for i, task_name in enumerate(tasks, 1):
        print(f"  {i}. {task_name}")
    
    # Step 2: Select tasks to evaluate
    print("\n🎯 Step 2: Selecting Tasks for Evaluation")
    print("-" * 80)
    
    selected_tasks = [
        'hypotension_5min',
        'blood_pressure_both',
        'signal_quality'
    ]
    
    print(f"Selected {len(selected_tasks)} tasks:")
    for task_name in selected_tasks:
        print(f"  ✓ {task_name}")
    
    # Step 3: Load model (or create mock model)
    print("\n🤖 Step 3: Loading Model")
    print("-" * 80)
    
    # For demonstration, create a simple model
    # In real usage, load your trained checkpoint
    config = {
        'task': 'classification',
        'num_classes': 2,
        'input_channels': 3,
        'context_length': 1250,
        'freeze_encoder': True,
        'head_type': 'linear',
        'variant': 'ibm-granite/granite-timeseries-ttm-r1'
    }
    
    print("Creating TTM model (for demonstration)...")
    model = create_ttm_model(config)
    model.eval()
    print("✓ Model loaded")
    model.print_parameter_summary()
    
    # Step 4: Initialize benchmark tracker
    print("\n📊 Step 4: Initializing Benchmark Tracker")
    print("-" * 80)
    
    tracker = BenchmarkTracker(output_dir='results/quick_start')
    print("✓ Tracker initialized")
    
    # Step 5: Evaluate each task
    print("\n🔬 Step 5: Evaluating Tasks")
    print("-" * 80)
    
    for task_name in selected_tasks:
        print(f"\n  Evaluating: {task_name}")
        print("  " + "-" * 76)
        
        # Load task
        task = get_task(task_name)
        
        # Generate synthetic data for demonstration
        # In real usage, load actual VitalDB data
        n_samples = 100
        
        if task.config.task_type == 'classification':
            # Binary classification
            predictions = np.random.randn(n_samples, task.config.num_classes)
            targets = np.random.randint(0, task.config.num_classes, n_samples)
        else:
            # Regression
            predictions = np.random.uniform(80, 140, (n_samples, task.config.target_dim))
            targets = predictions + np.random.normal(0, 5, (n_samples, task.config.target_dim))
        
        # Evaluate
        metrics = task.evaluate(predictions, targets)
        metrics['n_patients'] = n_samples
        
        # Log results
        tracker.log_result(
            task_name=task.config.name,
            metrics=metrics,
            model_name="TTM-QuickStart",
            n_patients=n_samples,
            split='test'
        )
        
        # Compare to benchmarks
        print("\n  Comparing to benchmarks...")
        comparison = tracker.compare_to_benchmarks(task, metrics)
        
        # Show key metrics
        if task.config.task_type == 'classification':
            if 'auroc' in metrics:
                print(f"  ✓ AUROC: {metrics['auroc']:.3f}")
            if 'auprc' in metrics:
                print(f"  ✓ AUPRC: {metrics['auprc']:.3f}")
        else:
            if 'mae' in metrics:
                print(f"  ✓ MAE: {metrics['mae']:.3f}")
            if 'pearson_r' in metrics:
                print(f"  ✓ Correlation: {metrics['pearson_r']:.3f}")
    
    # Step 6: Generate comprehensive report
    print("\n📑 Step 6: Generating Report")
    print("-" * 80)
    
    tracker.generate_report()
    
    # Display summary
    summary = tracker.get_summary_table()
    print("\nSummary Table:")
    print(summary.to_string(index=False))
    
    # Step 7: Results saved
    print("\n✅ Step 7: Complete")
    print("=" * 80)
    print("\nResults saved to: results/quick_start/")
    print("  - Individual task results (JSON)")
    print("  - Benchmark comparisons (CSV)")
    print("  - HTML report (benchmark_report.html)")
    print("\nNext steps:")
    print("  1. Review the HTML report in your browser")
    print("  2. Compare your results to published benchmarks")
    print("  3. Identify best-performing tasks for fine-tuning")
    print("  4. Run evaluation on real VitalDB data")
    print("=" * 80)


def demonstrate_detailed_evaluation():
    """Demonstrate detailed evaluation with all features."""
    print("\n" + "=" * 80)
    print("DETAILED EVALUATION EXAMPLE")
    print("=" * 80)
    
    # Example: Hypotension prediction with full pipeline
    print("\n🎯 Task: Hypotension Prediction (5-minute window)")
    print("-" * 80)
    
    task = get_task('hypotension_5min')
    
    # Show task configuration
    print("\nTask Configuration:")
    print(f"  Name: {task.config.name}")
    print(f"  Type: {task.config.task_type}")
    print(f"  Classes: {task.config.num_classes}")
    print(f"  Required channels: {task.config.required_channels}")
    print(f"  Sampling rate: {task.config.sampling_rate} Hz")
    print(f"  Clinical threshold: {task.config.clinical_threshold} mmHg")
    
    # Show benchmarks
    print("\nPublished Benchmarks:")
    for i, benchmark in enumerate(task.benchmarks, 1):
        print(f"\n  {i}. {benchmark.paper} ({benchmark.year})")
        print(f"     Dataset: {benchmark.dataset}")
        print(f"     Patients: {benchmark.n_patients}")
        print(f"     Metrics: {benchmark.metrics}")
        print(f"     Notes: {benchmark.notes}")
    
    # Simulate evaluation
    print("\n" + "-" * 80)
    print("Running evaluation...")
    
    # Generate synthetic predictions
    n_samples = 500
    predictions = np.random.randn(n_samples, 2)
    targets = np.random.randint(0, 2, n_samples)
    
    # Evaluate with detailed metrics
    metrics = task.evaluate(predictions, targets, return_detailed=True)
    
    print("\nEvaluation Results:")
    print(f"  Samples evaluated: {n_samples}")
    print(f"  AUROC: {metrics.get('auroc', 0):.3f}")
    print(f"  AUPRC: {metrics.get('auprc', 0):.3f}")
    print(f"  Accuracy: {metrics.get('accuracy', 0):.3f}")
    print(f"  F1 Score: {metrics.get('f1', 0):.3f}")
    
    # Compare to benchmarks
    print("\nBenchmark Comparison:")
    comparison = task.compare_to_benchmarks({'auroc': metrics.get('auroc', 0), 
                                             'auprc': metrics.get('auprc', 0)})
    print(comparison.to_string(index=False))
    
    print("\n" + "=" * 80)


def demonstrate_blood_pressure_evaluation():
    """Demonstrate blood pressure evaluation with AAMI compliance."""
    print("\n" + "=" * 80)
    print("BLOOD PRESSURE ESTIMATION EXAMPLE")
    print("=" * 80)
    
    task = get_task('blood_pressure_both')
    
    print("\n🩺 Task: Blood Pressure Estimation (SBP/DBP)")
    print("-" * 80)
    
    # Show clinical standards
    print("\nClinical Standards:")
    print("  AAMI Compliance:")
    print("    - Mean Error (ME) ≤ 5 mmHg")
    print("    - Standard Deviation (SD) ≤ 8 mmHg")
    print("  BHS Grade A:")
    print("    - 60% within 5 mmHg")
    print("    - 85% within 10 mmHg")
    print("    - 95% within 15 mmHg")
    
    # Simulate realistic BP predictions
    n_samples = 200
    
    # Ground truth: SBP ~120, DBP ~80
    sbp_true = np.random.normal(120, 15, n_samples)
    dbp_true = np.random.normal(80, 10, n_samples)
    targets = np.stack([sbp_true, dbp_true], axis=1)
    
    # Predictions with realistic error
    sbp_pred = sbp_true + np.random.normal(0, 4, n_samples)  # ~4 mmHg error
    dbp_pred = dbp_true + np.random.normal(0, 3, n_samples)  # ~3 mmHg error
    predictions = np.stack([sbp_pred, dbp_pred], axis=1)
    
    # Evaluate
    metrics = task.evaluate(predictions, targets)
    
    print("\nEvaluation Results:")
    print(f"  Samples: {n_samples}")
    print(f"\n  Systolic Blood Pressure (SBP):")
    print(f"    MAE: {metrics.get('sbp_mae', 0):.2f} mmHg")
    print(f"    ME:  {metrics.get('sbp_me', 0):.2f} mmHg")
    print(f"    SD:  {metrics.get('sbp_sd', 0):.2f} mmHg")
    print(f"    AAMI Compliant: {'✓' if metrics.get('sbp_aami_compliant', False) else '✗'}")
    print(f"    BHS Grade: {metrics.get('sbp_bhs_grade', 'N/A')}")
    
    print(f"\n  Diastolic Blood Pressure (DBP):")
    print(f"    MAE: {metrics.get('dbp_mae', 0):.2f} mmHg")
    print(f"    ME:  {metrics.get('dbp_me', 0):.2f} mmHg")
    print(f"    SD:  {metrics.get('dbp_sd', 0):.2f} mmHg")
    print(f"    AAMI Compliant: {'✓' if metrics.get('dbp_aami_compliant', False) else '✗'}")
    print(f"    BHS Grade: {metrics.get('dbp_bhs_grade', 'N/A')}")
    
    # Compare to benchmarks
    print("\nBenchmark Comparison:")
    comparison = task.compare_to_benchmarks(metrics)
    print(comparison[['Paper', 'Year', 'sbp_mae', 'dbp_mae']].to_string(index=False))
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Run quick start
    main()
    
    # Run detailed examples
    print("\n\n")
    demonstrate_detailed_evaluation()
    
    print("\n\n")
    demonstrate_blood_pressure_evaluation()
    
    print("\n\n🎉 Quick start complete!")
    print("\nFor real evaluation, use:")
    print("  python3 scripts/evaluate_task.py --task <task_name> --checkpoint <model.pt>")
