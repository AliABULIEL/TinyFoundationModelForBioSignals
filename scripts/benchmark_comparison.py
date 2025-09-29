#!/usr/bin/env python
"""Compare results across multiple downstream tasks and generate aggregate report.

Usage:
    python scripts/benchmark_comparison.py --results-dir results/
    python scripts/benchmark_comparison.py --results-dir results/ --format html
"""

import argparse
import sys
from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarks.tracker import BenchmarkTracker


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare results across multiple downstream tasks'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        required=True,
        help='Directory containing task evaluation results'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='table',
        choices=['table', 'html', 'csv', 'all'],
        help='Output format'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='benchmark_comparison',
        help='Output filename (without extension)'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate comparison plots'
    )
    
    return parser.parse_args()


def load_all_results(results_dir: Path) -> list:
    """Load all result JSON files from directory.
    
    Args:
        results_dir: Directory containing result files
        
    Returns:
        List of result dictionaries
    """
    all_results = []
    
    for result_file in results_dir.rglob("*_results.json"):
        try:
            with open(result_file) as f:
                result = json.load(f)
            all_results.append(result)
        except Exception as e:
            print(f"Warning: Could not load {result_file}: {e}")
    
    return all_results


def create_summary_table(results: list) -> pd.DataFrame:
    """Create summary table from all results.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Summary DataFrame
    """
    summary_data = []
    
    for result in results:
        row = {
            'Task': result['task_name'],
            'Split': result['split'],
            'N_Patients': result['n_patients'],
            'Date': result['timestamp'][:10]
        }
        
        metrics = result['metrics']
        
        # Classification metrics
        if 'auroc' in metrics:
            row['AUROC'] = f"{metrics['auroc']:.3f}"
        if 'auprc' in metrics:
            row['AUPRC'] = f"{metrics['auprc']:.3f}"
        if 'accuracy' in metrics:
            row['Accuracy'] = f"{metrics['accuracy']:.3f}"
        if 'f1' in metrics:
            row['F1'] = f"{metrics['f1']:.3f}"
        
        # Regression metrics
        if 'mae' in metrics:
            row['MAE'] = f"{metrics['mae']:.3f}"
        if 'rmse' in metrics:
            row['RMSE'] = f"{metrics['rmse']:.3f}"
        if 'pearson_r' in metrics:
            row['Corr'] = f"{metrics['pearson_r']:.3f}"
        if 'percentage_error' in metrics:
            row['PE%'] = f"{metrics['percentage_error']:.1f}"
        
        # AAMI compliance (blood pressure)
        if 'aami_compliant' in metrics:
            row['AAMI'] = '✓' if metrics['aami_compliant'] else '✗'
        if 'bhs_grade' in metrics:
            row['BHS'] = metrics['bhs_grade']
        
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    
    # Sort by task name
    if len(df) > 0:
        df = df.sort_values('Task')
    
    return df


def create_benchmark_comparison_plot(
    summary_df: pd.DataFrame,
    metric: str,
    output_file: Path
):
    """Create comparison plot for a specific metric.
    
    Args:
        summary_df: Summary DataFrame
        metric: Metric to plot
        output_file: Output file path
    """
    if metric not in summary_df.columns:
        print(f"Metric {metric} not found in results")
        return
    
    # Extract numeric values
    values = []
    tasks = []
    for idx, row in summary_df.iterrows():
        if metric in row and pd.notna(row[metric]):
            try:
                val = float(row[metric])
                values.append(val)
                tasks.append(row['Task'])
            except:
                pass
    
    if not values:
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = sns.color_palette("viridis", len(values))
    bars = ax.barh(tasks, values, color=colors)
    
    ax.set_xlabel(metric, fontsize=12, fontweight='bold')
    ax.set_title(f'Performance Comparison: {metric}', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, values):
        width = bar.get_width()
        ax.text(
            width, bar.get_y() + bar.get_height()/2,
            f' {val:.3f}',
            ha='left', va='center', fontweight='bold'
        )
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Plot saved: {output_file}")


def generate_html_report(
    summary_df: pd.DataFrame,
    results: list,
    output_file: Path
):
    """Generate HTML report with all results.
    
    Args:
        summary_df: Summary DataFrame
        results: List of result dictionaries
        output_file: Output HTML file path
    """
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>VitalDB Downstream Tasks - Benchmark Report</title>
        <style>
            body {
                font-family: 'Segoe UI', Arial, sans-serif;
                margin: 40px;
                background-color: #f5f5f5;
            }
            .container {
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1 {
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }
            h2 {
                color: #34495e;
                margin-top: 40px;
                border-bottom: 2px solid #e0e0e0;
                padding-bottom: 8px;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            th, td {
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }
            th {
                background-color: #3498db;
                color: white;
                font-weight: bold;
            }
            tr:nth-child(even) {
                background-color: #f8f9fa;
            }
            tr:hover {
                background-color: #e8f4f8;
            }
            .metric-good {
                color: #27ae60;
                font-weight: bold;
            }
            .metric-warning {
                color: #f39c12;
                font-weight: bold;
            }
            .metric-poor {
                color: #e74c3c;
                font-weight: bold;
            }
            .summary-box {
                background-color: #ecf0f1;
                padding: 20px;
                border-radius: 5px;
                margin: 20px 0;
            }
            .task-section {
                margin: 30px 0;
                padding: 20px;
                border-left: 4px solid #3498db;
                background-color: #f8f9fa;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏥 VitalDB Downstream Tasks - Benchmark Report</h1>
            <div class="summary-box">
                <p><strong>Generated:</strong> {timestamp}</p>
                <p><strong>Total Tasks:</strong> {n_tasks}</p>
                <p><strong>Total Evaluations:</strong> {n_evals}</p>
            </div>
    """.format(
        timestamp=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        n_tasks=len(summary_df['Task'].unique()) if len(summary_df) > 0 else 0,
        n_evals=len(results)
    )
    
    # Summary table
    html += "<h2>Summary</h2>"
    html += summary_df.to_html(index=False, escape=False, classes='table')
    
    # Detailed results for each task
    html += "<h2>Detailed Results</h2>"
    
    for result in results:
        task_name = result['task_name']
        html += f"""
        <div class="task-section">
            <h3>{task_name}</h3>
            <p><strong>Split:</strong> {result['split']}</p>
            <p><strong>Patients:</strong> {result['n_patients']}</p>
            <p><strong>Date:</strong> {result['timestamp'][:10]}</p>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
        """
        
        for metric, value in result['metrics'].items():
            if isinstance(value, (int, float)):
                # Color code based on metric type and value
                css_class = ""
                if 'auroc' in metric.lower() or 'accuracy' in metric.lower():
                    if value >= 0.90:
                        css_class = "metric-good"
                    elif value >= 0.80:
                        css_class = "metric-warning"
                    else:
                        css_class = "metric-poor"
                
                html += f"<tr><td>{metric}</td><td class='{css_class}'>{value:.4f}</td></tr>"
        
        html += """
            </table>
        </div>
        """
    
    html += """
        </div>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"✓ HTML report saved: {output_file}")


def main():
    args = parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    # Load all results
    print(f"Loading results from {results_dir}...")
    results = load_all_results(results_dir)
    
    if not results:
        print("No results found!")
        return
    
    print(f"✓ Loaded {len(results)} result files")
    
    # Create summary table
    summary_df = create_summary_table(results)
    
    # Output results
    output_base = Path(args.output)
    
    if args.format in ['table', 'all']:
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        print(summary_df.to_string(index=False))
        print("="*80 + "\n")
    
    if args.format in ['csv', 'all']:
        csv_file = output_base.with_suffix('.csv')
        summary_df.to_csv(csv_file, index=False)
        print(f"✓ CSV saved: {csv_file}")
    
    if args.format in ['html', 'all']:
        html_file = output_base.with_suffix('.html')
        generate_html_report(summary_df, results, html_file)
    
    # Generate plots
    if args.plot:
        plot_dir = Path('plots')
        plot_dir.mkdir(exist_ok=True)
        
        # Plot key metrics
        metrics_to_plot = ['AUROC', 'AUPRC', 'MAE', 'RMSE', 'Accuracy']
        for metric in metrics_to_plot:
            if metric in summary_df.columns:
                plot_file = plot_dir / f"comparison_{metric.lower()}.png"
                create_benchmark_comparison_plot(summary_df, metric, plot_file)


if __name__ == "__main__":
    main()
