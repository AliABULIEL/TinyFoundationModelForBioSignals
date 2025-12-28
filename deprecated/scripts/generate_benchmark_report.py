#!/usr/bin/env python3
"""
Generate Benchmark Report from Evaluation Results
==================================================

This script generates comprehensive benchmark comparison reports from
evaluation results. Can be run standalone or integrated into evaluation pipelines.

Usage:
    # Generate from multi-task results directory
    python scripts/generate_benchmark_report.py \\
        --results-dir artifacts/all_tasks_FINAL \\
        --output-dir artifacts/reports \\
        --model-name "TTM-SSL Foundation Model" \\
        --experiment-name "Hybrid SSL Multi-Task"

    # Generate from individual task results
    python scripts/generate_benchmark_report.py \\
        --task-results artifacts/task_quality/metrics.json \\
                      artifacts/task_hr/metrics.json \\
        --output-dir artifacts/reports

    # Generate from JSON summary
    python scripts/generate_benchmark_report.py \\
        --from-json artifacts/evaluation_summary.json \\
        --output-dir artifacts/reports

Author: Foundation Model Team
Date: October 2025
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.eval.report_generator import BenchmarkReportGenerator


# BUT-PPG benchmark definitions (from butppg_tasks.py)
BUTPPG_BENCHMARKS = {
    'quality': {
        'task_name': 'Signal Quality Classification',
        'metric': 'AUROC',
        'target': 0.88,
        'baseline': 0.74,
        'sota': 0.88,
        'sota_paper': 'Article Target'
    },
    'hr_estimation': {
        'task_name': 'Heart Rate Estimation',
        'metric': 'MAE',
        'target': 2.0,  # bpm
        'baseline': 3.5,
        'sota': 1.5,
        'sota_paper': 'Clinical Standard'
    },
    'motion': {
        'task_name': 'Motion Classification',
        'metric': 'Accuracy',
        'target': 0.85,
        'baseline': 0.70,
        'sota': 0.85,
        'sota_paper': 'Article Target'
    },
    'bp_systolic': {
        'task_name': 'Systolic Blood Pressure Estimation',
        'metric': 'MAE',
        'target': 5.0,  # mmHg (AAMI standard)
        'baseline': 8.0,
        'sota': 5.0,
        'sota_paper': 'AAMI/ISO Standard (2013)'
    },
    'bp_diastolic': {
        'task_name': 'Diastolic Blood Pressure Estimation',
        'metric': 'MAE',
        'target': 5.0,  # mmHg (AAMI standard)
        'baseline': 6.0,
        'sota': 5.0,
        'sota_paper': 'AAMI/ISO Standard (2013)'
    },
    'spo2': {
        'task_name': 'SpO2 Estimation',
        'metric': 'MAE',
        'target': 2.0,  # percentage
        'baseline': 3.0,
        'sota': 2.0,
        'sota_paper': 'Clinical Standard'
    },
    'glycaemia': {
        'task_name': 'Glycaemia Estimation',
        'metric': 'MAE',
        'target': 1.0,  # mmol/l
        'baseline': 1.5,
        'sota': 1.0,
        'sota_paper': 'Research Target'
    }
}


def load_task_results_from_dir(results_dir: Path) -> Dict[str, Dict]:
    """Load results from multi-task directory structure.

    Expected structure:
        results_dir/
            task_quality/
                metrics.json
                final_results.json
            task_hr_estimation/
                metrics.json
                final_results.json
            ...

    Returns:
        Dictionary mapping task names to result dictionaries
    """
    results = {}

    # Look for task_* subdirectories
    for task_dir in results_dir.glob("task_*"):
        if not task_dir.is_dir():
            continue

        task_name = task_dir.name.replace("task_", "")

        # Try to load results
        metrics_file = task_dir / "metrics.json"
        final_results_file = task_dir / "final_results.json"
        results_file = task_dir / "results.json"

        if final_results_file.exists():
            with open(final_results_file) as f:
                task_data = json.load(f)
        elif metrics_file.exists():
            with open(metrics_file) as f:
                task_data = json.load(f)
        elif results_file.exists():
            with open(results_file) as f:
                task_data = json.load(f)
        else:
            print(f"⚠️  No results found for {task_name}")
            continue

        results[task_name] = task_data

    return results


def load_task_results_from_files(file_paths: List[Path]) -> Dict[str, Dict]:
    """Load results from individual JSON files.

    Returns:
        Dictionary mapping task names to result dictionaries
    """
    results = {}

    for file_path in file_paths:
        with open(file_path) as f:
            task_data = json.load(f)

        # Infer task name from file path or data
        if 'task_name' in task_data:
            task_name = task_data['task_name']
        else:
            # Extract from path: .../task_quality/metrics.json → quality
            parts = file_path.parts
            for part in parts:
                if part.startswith('task_'):
                    task_name = part.replace('task_', '')
                    break
            else:
                task_name = file_path.stem

        results[task_name] = task_data

    return results


def extract_metrics(task_data: Dict, task_name: str) -> Dict:
    """Extract relevant metrics from task result data.

    Handles various JSON formats from different evaluation scripts.

    Returns:
        Dictionary with standardized metric fields
    """
    metrics = {}

    # Try different JSON structures
    if 'test_results' in task_data:
        # Format from run_downstream_evaluation.py
        test_results = task_data['test_results']
        metrics['value'] = test_results.get('primary_metric', test_results.get('auroc', test_results.get('mae', 0.0)))
        metrics['ci_lower'] = test_results.get('ci_lower')
        metrics['ci_upper'] = test_results.get('ci_upper')

        # Secondary metrics
        metrics['secondary'] = {k: v for k, v in test_results.items()
                               if k not in ['primary_metric', 'ci_lower', 'ci_upper']}

    elif 'metrics' in task_data:
        # Format from finetune_enhanced.py
        test_metrics = task_data['metrics'].get('test', {})
        metrics['value'] = test_metrics.get('auroc', test_metrics.get('mae', 0.0))
        metrics['secondary'] = test_metrics

    else:
        # Direct format
        metrics['value'] = task_data.get('auroc', task_data.get('mae', task_data.get('accuracy', 0.0)))
        metrics['secondary'] = {k: v for k, v in task_data.items() if isinstance(v, (int, float))}

    # Extract metadata
    metrics['num_samples'] = task_data.get('num_test_samples', task_data.get('test_size'))
    metrics['num_subjects'] = task_data.get('num_test_subjects')
    metrics['training_time'] = task_data.get('training_time', task_data.get('total_training_time'))

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark comparison report from evaluation results"
    )

    # Input sources (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--results-dir',
        type=str,
        help='Directory containing task_* subdirectories with results'
    )
    input_group.add_argument(
        '--task-results',
        type=str,
        nargs='+',
        help='Individual task result JSON files'
    )
    input_group.add_argument(
        '--from-json',
        type=str,
        help='Pre-aggregated evaluation summary JSON'
    )

    # Output configuration
    parser.add_argument(
        '--output-dir',
        type=str,
        default='artifacts/reports',
        help='Directory to save reports (default: artifacts/reports)'
    )

    # Report metadata
    parser.add_argument(
        '--model-name',
        type=str,
        default='TTM-SSL Foundation Model',
        help='Name of the model'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='Multi-Task Evaluation',
        help='Name of the experiment'
    )
    parser.add_argument(
        '--author',
        type=str,
        default='Foundation Model Team',
        help='Report author'
    )
    parser.add_argument(
        '--description',
        type=str,
        default='',
        help='Experiment description'
    )

    # Report formats
    parser.add_argument(
        '--formats',
        type=str,
        nargs='+',
        default=['markdown', 'latex', 'json', 'html'],
        choices=['markdown', 'latex', 'json', 'html'],
        help='Report formats to generate (default: all)'
    )

    args = parser.parse_args()

    # Load task results
    print(f"\n{'='*70}")
    print("LOADING EVALUATION RESULTS")
    print(f"{'='*70}\n")

    if args.results_dir:
        results_dir = Path(args.results_dir)
        print(f"Loading from directory: {results_dir}")
        task_results = load_task_results_from_dir(results_dir)
    elif args.task_results:
        print(f"Loading from {len(args.task_results)} files")
        task_files = [Path(f) for f in args.task_results]
        task_results = load_task_results_from_files(task_files)
    else:  # from-json
        print(f"Loading from JSON: {args.from_json}")
        with open(args.from_json) as f:
            summary = json.load(f)
        task_results = summary.get('results', summary)

    print(f"✅ Loaded results for {len(task_results)} tasks:")
    for task_name in task_results.keys():
        print(f"   - {task_name}")

    # Create report generator
    print(f"\n{'='*70}")
    print("CREATING BENCHMARK REPORT")
    print(f"{'='*70}\n")

    generator = BenchmarkReportGenerator(
        model_name=args.model_name,
        experiment_name=args.experiment_name,
        output_dir=args.output_dir,
        author=args.author,
        description=args.description
    )

    # Add results to generator
    for task_name, task_data in task_results.items():
        # Get benchmark for this task
        if task_name in BUTPPG_BENCHMARKS:
            benchmark = BUTPPG_BENCHMARKS[task_name]
        else:
            print(f"⚠️  No benchmark found for task: {task_name}, skipping...")
            continue

        # Extract metrics
        metrics = extract_metrics(task_data, task_name)

        # Add to generator
        generator.add_task_result(
            task_name=task_name,
            task_description=benchmark['task_name'],
            metric=benchmark['metric'],
            value=metrics['value'],
            ci_lower=metrics.get('ci_lower'),
            ci_upper=metrics.get('ci_upper'),
            target=benchmark['target'],
            baseline=benchmark['baseline'],
            sota=benchmark.get('sota'),
            sota_paper=benchmark.get('sota_paper'),
            secondary_metrics=metrics.get('secondary', {}),
            num_samples=metrics.get('num_samples'),
            num_subjects=metrics.get('num_subjects'),
            training_time=metrics.get('training_time')
        )

        print(f"✅ Added {task_name}: {benchmark['metric']} = {metrics['value']:.4f}")

    # Generate reports
    print(f"\n{'='*70}")
    print("GENERATING REPORTS")
    print(f"{'='*70}\n")

    reports = {}
    if 'markdown' in args.formats:
        reports['markdown'] = generator.generate_markdown_report()
    if 'latex' in args.formats:
        reports['latex'] = generator.generate_latex_table()
    if 'json' in args.formats:
        reports['json'] = generator.generate_json_summary()
    if 'html' in args.formats:
        reports['html'] = generator.generate_html_report()

    # Summary
    print(f"\n{'='*70}")
    print("✅ BENCHMARK REPORT GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nGenerated {len(reports)} reports:")
    for format_name, file_path in reports.items():
        print(f"  [{format_name.upper()}] {file_path}")
    print(f"\n{'='*70}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
