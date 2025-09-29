"""Benchmark tracking and comparison system for VitalDB tasks."""

from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class BenchmarkTracker:
    """Track and compare results against published benchmarks."""
    
    def __init__(self, output_dir: str = "benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = []
        self.comparisons = []
    
    def log_result(
        self,
        task_name: str,
        metrics: Dict[str, float],
        model_name: str = "TTM-VitalDB",
        n_patients: int = 0,
        split: str = "test",
        notes: str = ""
    ):
        """Log evaluation results for a task.
        
        Args:
            task_name: Name of the downstream task
            metrics: Dictionary of evaluation metrics
            model_name: Name of the model
            n_patients: Number of patients in evaluation
            split: Data split (train/val/test)
            notes: Additional notes
        """
        result = {
            'timestamp': datetime.now().isoformat(),
            'task_name': task_name,
            'model_name': model_name,
            'n_patients': n_patients,
            'split': split,
            'metrics': metrics,
            'notes': notes
        }
        
        self.results.append(result)
        
        # Save to file
        result_file = self.output_dir / f"{task_name}_{split}_results.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"RESULTS: {task_name} ({split})")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Patients: {n_patients}")
        print(f"\nMetrics:")
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric_name:30s}: {value:.4f}")
            else:
                print(f"  {metric_name:30s}: {value}")
        print(f"{'='*60}\n")
    
    def compare_to_benchmarks(
        self,
        task,
        results: Dict[str, float]
    ) -> pd.DataFrame:
        """Compare results to published benchmarks.
        
        Args:
            task: Task object with benchmarks
            results: Dictionary of evaluation results
            
        Returns:
            DataFrame with comparison
        """
        comparison = task.compare_to_benchmarks(results)
        
        self.comparisons.append({
            'task_name': task.config.name,
            'timestamp': datetime.now().isoformat(),
            'comparison': comparison
        })
        
        # Save comparison
        comparison_file = self.output_dir / f"{task.config.name}_comparison.csv"
        comparison.to_csv(comparison_file, index=False)
        
        # Print comparison
        print(f"\n{'='*80}")
        print(f"BENCHMARK COMPARISON: {task.config.name}")
        print(f"{'='*80}")
        print(comparison.to_string(index=False))
        print(f"{'='*80}\n")
        
        return comparison
    
    def generate_report(
        self,
        output_file: Optional[str] = None
    ):
        """Generate comprehensive benchmark report.
        
        Args:
            output_file: Output HTML file path
        """
        if output_file is None:
            output_file = self.output_dir / "benchmark_report.html"
        
        html = self._generate_html_report()
        
        with open(output_file, 'w') as f:
            f.write(html)
        
        print(f"Report saved to: {output_file}")
    
    def _generate_html_report(self) -> str:
        """Generate HTML report with all results and comparisons."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>VitalDB Benchmark Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #2c3e50; }
                h2 { color: #34495e; margin-top: 40px; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #3498db; color: white; }
                tr:nth-child(even) { background-color: #f2f2f2; }
                .metric { font-weight: bold; color: #2980b9; }
                .benchmark { color: #27ae60; }
                .warning { color: #e74c3c; }
                .timestamp { color: #7f8c8d; font-size: 0.9em; }
            </style>
        </head>
        <body>
            <h1>🏥 VitalDB Downstream Tasks - Benchmark Report</h1>
            <p class="timestamp">Generated: {timestamp}</p>
        """.format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # Add summary
        html += "<h2>Summary</h2>"
        html += f"<p>Total tasks evaluated: {len(self.results)}</p>"
        html += f"<p>Total comparisons: {len(self.comparisons)}</p>"
        
        # Add results for each task
        for result in self.results:
            html += f"""
            <h2>{result['task_name']} ({result['split']})</h2>
            <p><strong>Model:</strong> {result['model_name']}</p>
            <p><strong>Patients:</strong> {result['n_patients']}</p>
            <p><strong>Date:</strong> {result['timestamp']}</p>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
            """
            
            for metric, value in result['metrics'].items():
                if isinstance(value, (int, float)):
                    html += f"<tr><td>{metric}</td><td class='metric'>{value:.4f}</td></tr>"
                else:
                    html += f"<tr><td>{metric}</td><td>{value}</td></tr>"
            
            html += "</table>"
            
            if result['notes']:
                html += f"<p><em>Notes: {result['notes']}</em></p>"
        
        # Add comparisons
        html += "<h2>Benchmark Comparisons</h2>"
        for comp in self.comparisons:
            html += f"<h3>{comp['task_name']}</h3>"
            html += comp['comparison'].to_html(index=False, classes='table')
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def plot_comparison(
        self,
        task_name: str,
        metric: str,
        comparison_df: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """Plot comparison of results vs benchmarks.
        
        Args:
            task_name: Task name
            metric: Metric to plot
            comparison_df: Comparison DataFrame
            save_path: Path to save plot
        """
        if metric not in comparison_df.columns:
            print(f"Metric {metric} not found in comparison")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract data
        papers = comparison_df['Paper'].values
        values = comparison_df[metric].values
        
        # Create bar plot
        colors = ['#e74c3c' if paper == 'This Work' else '#3498db' 
                  for paper in papers]
        bars = ax.barh(papers, values, color=colors)
        
        ax.set_xlabel(metric, fontsize=12)
        ax.set_title(f'{task_name}: {metric}', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / f"{task_name}_{metric}_comparison.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved to: {save_path}")
    
    def get_summary_table(self) -> pd.DataFrame:
        """Get summary table of all results.
        
        Returns:
            DataFrame with summary of all tasks
        """
        summary_data = []
        
        for result in self.results:
            row = {
                'Task': result['task_name'],
                'Model': result['model_name'],
                'Split': result['split'],
                'N_Patients': result['n_patients'],
                'Date': result['timestamp'][:10]
            }
            
            # Add key metrics
            metrics = result['metrics']
            if 'auroc' in metrics:
                row['AUROC'] = f"{metrics['auroc']:.3f}"
            if 'auprc' in metrics:
                row['AUPRC'] = f"{metrics['auprc']:.3f}"
            if 'mae' in metrics:
                row['MAE'] = f"{metrics['mae']:.3f}"
            if 'accuracy' in metrics:
                row['Accuracy'] = f"{metrics['accuracy']:.3f}"
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)


def load_vitaldb_clinical_data() -> pd.DataFrame:
    """Load VitalDB clinical parameters.
    
    Returns:
        DataFrame with 73 clinical parameters for all cases
    """
    df = pd.read_csv('https://api.vitaldb.net/cases')
    return df


def load_vitaldb_lab_data() -> pd.DataFrame:
    """Load VitalDB laboratory data.
    
    Returns:
        DataFrame with lab results (90 days pre/post surgery)
    """
    df = pd.read_csv('https://api.vitaldb.net/labs')
    return df
