"""Comprehensive evaluation framework for downstream tasks.

Handles evaluation on VitalDB and BUT-PPG tasks with benchmark comparisons.
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import (
    classification_metrics,
    regression_metrics,
    auroc,
    auprc,
    mae,
    rmse
)
from .benchmarks import (
    get_benchmark,
    get_target_metric,
    get_sota,
    get_baseline,
    categorize_performance,
    format_benchmark_table
)


class DownstreamEvaluator:
    """Evaluator for downstream tasks."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        task_type: str = "classification",  # or "regression"
        dataset_name: str = "vitaldb",  # or "butppg"
        task_name: str = "hypotension",
        save_dir: Optional[Path] = None
    ):
        """Initialize evaluator.
        
        Args:
            model: Trained model
            device: Device to run on
            task_type: 'classification' or 'regression'
            dataset_name: 'vitaldb' or 'butppg'
            task_name: Specific task name (e.g., 'hypotension', 'quality')
            save_dir: Directory to save results
        """
        self.model = model
        self.device = device
        self.task_type = task_type
        self.dataset_name = dataset_name
        self.task_name = task_name
        self.save_dir = Path(save_dir) if save_dir else Path("results")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Store results
        self.results = {
            "dataset": dataset_name,
            "task": task_name,
            "task_type": task_type,
            "predictions": [],
            "labels": [],
            "subject_ids": [],
            "metrics": {},
            "per_subject_metrics": {},
            "benchmarks": {},
            "timestamp": None
        }
    
    def evaluate(
        self,
        dataloader: DataLoader,
        split: str = "test",
        return_predictions: bool = True,
        verbose: bool = True
    ) -> Dict:
        """Evaluate model on a dataset.
        
        Args:
            dataloader: DataLoader for evaluation
            split: 'val' or 'test'
            return_predictions: Whether to return predictions
            verbose: Whether to print progress
            
        Returns:
            Dictionary of results
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []  # For classification
        all_subject_ids = []
        
        start_time = time.time()
        
        with torch.no_grad():
            iterator = tqdm(dataloader, desc=f"Evaluating {split}") if verbose else dataloader
            
            for batch in iterator:
                # Get batch data
                signals = batch['signals'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Get subject IDs if available
                if 'subject_id' in batch:
                    subject_ids = batch['subject_id']
                else:
                    subject_ids = ['unknown'] * len(signals)
                
                # Forward pass
                outputs = self.model(signals)
                
                if self.task_type == "classification":
                    # Get probabilities
                    if outputs.shape[-1] == 1:
                        # Binary classification
                        probs = torch.sigmoid(outputs).squeeze(-1)
                        preds = (probs > 0.5).long()
                    else:
                        # Multi-class
                        probs = torch.softmax(outputs, dim=-1)
                        preds = torch.argmax(probs, dim=-1)
                    
                    all_probs.append(probs.cpu().numpy())
                    all_preds.append(preds.cpu().numpy())
                
                else:  # regression
                    preds = outputs.squeeze(-1)
                    all_preds.append(preds.cpu().numpy())
                
                all_labels.append(labels.cpu().numpy())
                all_subject_ids.extend(subject_ids)
        
        # Concatenate all batches
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        if self.task_type == "classification":
            all_probs = np.concatenate(all_probs)
        
        eval_time = time.time() - start_time
        
        # Compute metrics
        if self.task_type == "classification":
            metrics = classification_metrics(
                all_labels,
                all_preds,
                all_probs
            )
        else:
            metrics = regression_metrics(all_labels, all_preds)
        
        metrics['eval_time'] = eval_time
        metrics['num_samples'] = len(all_labels)
        
        # Store results
        self.results['predictions'] = all_preds if return_predictions else []
        self.results['labels'] = all_labels if return_predictions else []
        self.results['subject_ids'] = all_subject_ids if return_predictions else []
        self.results['metrics'] = metrics
        self.results['split'] = split
        self.results['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        if self.task_type == "classification":
            self.results['probabilities'] = all_probs if return_predictions else []
        
        # Compute per-subject metrics
        if 'unknown' not in all_subject_ids:
            per_subject = self._compute_per_subject_metrics(
                all_labels,
                all_preds,
                all_probs if self.task_type == "classification" else None,
                all_subject_ids
            )
            self.results['per_subject_metrics'] = per_subject
        
        # Compare with benchmarks
        benchmark_comparison = self._compare_with_benchmarks(metrics)
        self.results['benchmarks'] = benchmark_comparison
        
        # Print results if verbose
        if verbose:
            self._print_results(metrics, benchmark_comparison)
        
        return self.results
    
    def _compute_per_subject_metrics(
        self,
        labels: np.ndarray,
        preds: np.ndarray,
        probs: Optional[np.ndarray],
        subject_ids: List[str]
    ) -> Dict:
        """Compute per-subject metrics.
        
        Args:
            labels: All labels
            preds: All predictions
            probs: All probabilities (for classification)
            subject_ids: Subject IDs
            
        Returns:
            Dictionary of per-subject metrics
        """
        unique_subjects = list(set(subject_ids))
        per_subject = {}
        
        for subject in unique_subjects:
            # Get indices for this subject
            indices = [i for i, sid in enumerate(subject_ids) if sid == subject]
            
            if len(indices) == 0:
                continue
            
            subj_labels = labels[indices]
            subj_preds = preds[indices]
            
            # Compute metrics
            if self.task_type == "classification":
                subj_probs = probs[indices] if probs is not None else None
                
                try:
                    metrics = classification_metrics(
                        subj_labels,
                        subj_preds,
                        subj_probs
                    )
                except:
                    # May fail if only one class present
                    metrics = {
                        'accuracy': float((subj_preds == subj_labels).mean()),
                        'num_samples': len(subj_labels)
                    }
            else:
                metrics = regression_metrics(subj_labels, subj_preds)
            
            metrics['num_samples'] = len(indices)
            per_subject[subject] = metrics
        
        # Compute statistics across subjects
        if len(per_subject) > 0:
            # Get primary metric
            if self.task_type == "classification":
                primary_metric = 'auroc' if 'auroc' in list(per_subject.values())[0] else 'accuracy'
            else:
                primary_metric = 'mae'
            
            # Collect values
            values = [m[primary_metric] for m in per_subject.values() if primary_metric in m]
            
            if len(values) > 0:
                per_subject['_summary'] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values)),
                    'num_subjects': len(values),
                    'metric': primary_metric
                }
        
        return per_subject
    
    def _compare_with_benchmarks(self, metrics: Dict) -> Dict:
        """Compare results with benchmarks.
        
        Args:
            metrics: Computed metrics
            
        Returns:
            Benchmark comparison dictionary
        """
        comparison = {
            'target': None,
            'sota': None,
            'baseline': None,
            'performance_category': None,
            'benchmarks_table': None
        }
        
        try:
            # Get benchmarks
            target = get_target_metric(self.dataset_name, self.task_name)
            sota = get_sota(self.dataset_name, self.task_name)
            baseline = get_baseline(self.dataset_name, self.task_name)
            
            # Get the metric to compare
            metric_name = target.metric.lower()
            
            if metric_name in metrics:
                value = metrics[metric_name]
                
                # Determine if higher is better
                higher_is_better = metric_name in ['auroc', 'auprc', 'accuracy', 'f1', 'precision', 'recall', 'r2']
                
                # Categorize performance
                category = categorize_performance(
                    self.dataset_name,
                    self.task_name,
                    metric_name,
                    value,
                    higher_is_better
                )
                
                comparison['target'] = {
                    'metric': target.metric,
                    'value': target.value,
                    'model': target.model,
                    'achieved': value,
                    'difference': value - target.value if higher_is_better else target.value - value,
                    'percentage': ((value / target.value - 1) * 100) if target.value > 0 else 0
                }
                
                comparison['sota'] = {
                    'metric': sota.metric,
                    'value': sota.value,
                    'model': sota.model,
                    'achieved': value,
                    'difference': value - sota.value if higher_is_better else sota.value - value,
                    'percentage': ((value / sota.value - 1) * 100) if sota.value > 0 else 0
                }
                
                comparison['baseline'] = {
                    'metric': baseline.metric,
                    'value': baseline.value,
                    'model': baseline.model,
                    'achieved': value,
                    'difference': value - baseline.value if higher_is_better else baseline.value - value,
                    'percentage': ((value / baseline.value - 1) * 100) if baseline.value > 0 else 0
                }
                
                comparison['performance_category'] = category
                comparison['benchmarks_table'] = format_benchmark_table(self.dataset_name, self.task_name)
        
        except Exception as e:
            print(f"Warning: Could not compare with benchmarks: {e}")
        
        return comparison
    
    def _print_results(self, metrics: Dict, benchmark_comparison: Dict):
        """Print evaluation results.
        
        Args:
            metrics: Computed metrics
            benchmark_comparison: Benchmark comparison
        """
        print("\n" + "="*80)
        print(f"EVALUATION RESULTS - {self.dataset_name.upper()} - {self.task_name.upper()}")
        print("="*80)
        
        # Print metrics
        print("\n📊 METRICS:")
        print("-" * 80)
        
        if self.task_type == "classification":
            print(f"  AUROC:      {metrics.get('auroc', 0):.4f}")
            print(f"  AUPRC:      {metrics.get('auprc', 0):.4f}")
            print(f"  Accuracy:   {metrics.get('accuracy', 0):.4f}")
            print(f"  Precision:  {metrics.get('precision', 0):.4f}")
            print(f"  Recall:     {metrics.get('recall', 0):.4f}")
            print(f"  F1 Score:   {metrics.get('f1', 0):.4f}")
        else:
            print(f"  MAE:        {metrics.get('mae', 0):.4f}")
            print(f"  RMSE:       {metrics.get('rmse', 0):.4f}")
            print(f"  R²:         {metrics.get('r2', 0):.4f}")
            print(f"  CCC:        {metrics.get('ccc', 0):.4f}")
            print(f"  Pearson r:  {metrics.get('pearson_r', 0):.4f}")
        
        print(f"\n  Samples:    {metrics.get('num_samples', 0)}")
        print(f"  Time:       {metrics.get('eval_time', 0):.2f}s")
        
        # Print benchmark comparison
        if benchmark_comparison['performance_category']:
            print("\n" + "="*80)
            print("🎯 BENCHMARK COMPARISON")
            print("="*80)
            
            cat = benchmark_comparison['performance_category']
            print(f"\nPerformance Category: {cat}")
            
            if benchmark_comparison['target']:
                t = benchmark_comparison['target']
                print(f"\n📌 TARGET ({t['model']}):")
                print(f"   Target:   {t['value']:.4f}")
                print(f"   Achieved: {t['achieved']:.4f}")
                print(f"   Diff:     {t['difference']:+.4f} ({t['percentage']:+.1f}%)")
            
            if benchmark_comparison['sota']:
                s = benchmark_comparison['sota']
                print(f"\n🏆 SOTA ({s['model']}):")
                print(f"   SOTA:     {s['value']:.4f}")
                print(f"   Achieved: {s['achieved']:.4f}")
                print(f"   Diff:     {s['difference']:+.4f} ({s['percentage']:+.1f}%)")
            
            if benchmark_comparison['baseline']:
                b = benchmark_comparison['baseline']
                print(f"\n📊 BASELINE ({b['model']}):")
                print(f"   Baseline: {b['value']:.4f}")
                print(f"   Achieved: {b['achieved']:.4f}")
                print(f"   Diff:     {b['difference']:+.4f} ({b['percentage']:+.1f}%)")
            
            # Print full benchmark table
            if benchmark_comparison['benchmarks_table']:
                print("\n" + benchmark_comparison['benchmarks_table'])
        
        # Print per-subject summary if available
        if '_summary' in self.results.get('per_subject_metrics', {}):
            summary = self.results['per_subject_metrics']['_summary']
            print("\n" + "="*80)
            print("👥 PER-SUBJECT ANALYSIS")
            print("="*80)
            print(f"\nMetric: {summary['metric'].upper()}")
            print(f"  Mean:      {summary['mean']:.4f}")
            print(f"  Std:       {summary['std']:.4f}")
            print(f"  Min:       {summary['min']:.4f}")
            print(f"  Max:       {summary['max']:.4f}")
            print(f"  Median:    {summary['median']:.4f}")
            print(f"  Subjects:  {summary['num_subjects']}")
        
        print("\n" + "="*80 + "\n")
    
    def save_results(self, filename: Optional[str] = None):
        """Save results to JSON file.
        
        Args:
            filename: Optional custom filename
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.dataset_name}_{self.task_name}_{timestamp}.json"
        
        filepath = self.save_dir / filename
        
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = {}
        for key, value in self.results.items():
            if isinstance(value, np.ndarray):
                results_serializable[key] = value.tolist()
            elif isinstance(value, dict):
                results_serializable[key] = self._make_serializable(value)
            else:
                results_serializable[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"\n✅ Results saved to: {filepath}")
    
    def _make_serializable(self, obj):
        """Recursively convert numpy arrays to lists."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj


def evaluate_model_on_task(
    model: nn.Module,
    dataloader: DataLoader,
    task_config: Dict,
    device: torch.device,
    save_dir: Optional[Path] = None,
    verbose: bool = True
) -> Dict:
    """Convenience function to evaluate a model on a downstream task.
    
    Args:
        model: Trained model
        dataloader: Test dataloader
        task_config: Task configuration dict with 'dataset', 'task', 'task_type'
        device: Device to run on
        save_dir: Directory to save results
        verbose: Whether to print results
        
    Returns:
        Dictionary of results
    """
    evaluator = DownstreamEvaluator(
        model=model,
        device=device,
        task_type=task_config['task_type'],
        dataset_name=task_config['dataset'],
        task_name=task_config['task'],
        save_dir=save_dir
    )
    
    results = evaluator.evaluate(
        dataloader=dataloader,
        split='test',
        return_predictions=True,
        verbose=verbose
    )
    
    if save_dir:
        evaluator.save_results()
    
    return results
