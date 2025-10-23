"""Automated Benchmark Report Generation

Generates comprehensive benchmark comparison reports in multiple formats:
- Markdown tables with performance indicators
- LaTeX tables for publications
- HTML reports with visualizations
- JSON summaries for programmatic access

Usage:
    from src.eval.report_generator import BenchmarkReportGenerator

    # Create generator
    generator = BenchmarkReportGenerator(
        model_name="TTM-SSL Foundation Model",
        experiment_name="Hybrid SSL Pipeline",
        output_dir="artifacts/reports"
    )

    # Add task results
    generator.add_task_result(
        task_name="quality",
        metric="AUROC",
        value=0.85,
        ci_lower=0.82,
        ci_upper=0.88,
        target=0.88,
        baseline=0.74,
        sota=0.88
    )

    # Generate reports
    generator.generate_all_reports()
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import json

import numpy as np


@dataclass
class TaskResult:
    """Single task evaluation result with benchmark comparison."""
    task_name: str
    task_description: str
    metric: str
    value: float
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    std: Optional[float] = None

    # Benchmarks
    target: Optional[float] = None
    baseline: Optional[float] = None
    sota: Optional[float] = None
    sota_paper: Optional[str] = None

    # Additional metrics
    secondary_metrics: Dict[str, float] = field(default_factory=dict)

    # Metadata
    num_samples: Optional[int] = None
    num_subjects: Optional[int] = None
    training_time: Optional[float] = None

    def performance_category(self) -> str:
        """Categorize performance relative to target."""
        if self.target is None:
            return "N/A"

        if self.value >= self.target:
            return "✅ EXCEEDS"
        elif self.value >= self.target * 0.95:  # Within 5% of target
            return "✓ MEETS"
        elif self.value >= self.target * 0.90:  # Within 10% of target
            return "⚠️ CLOSE"
        else:
            return "❌ BELOW"

    def vs_baseline(self) -> str:
        """Compare to baseline."""
        if self.baseline is None:
            return "N/A"

        # For metrics where lower is better (MAE, RMSE, etc.)
        if 'MAE' in self.metric or 'RMSE' in self.metric or 'MSE' in self.metric:
            improvement = ((self.baseline - self.value) / self.baseline) * 100
        else:  # For metrics where higher is better (AUROC, accuracy, etc.)
            improvement = ((self.value - self.baseline) / self.baseline) * 100

        if improvement > 0:
            return f"+{improvement:.1f}%"
        else:
            return f"{improvement:.1f}%"


class BenchmarkReportGenerator:
    """Automated benchmark report generator.

    Generates comprehensive reports comparing model performance to benchmarks.
    Supports multiple output formats: Markdown, LaTeX, HTML, JSON.

    Args:
        model_name: Name of the model being evaluated
        experiment_name: Name of the experiment/run
        output_dir: Directory to save reports
        author: Report author name
        description: Optional experiment description

    Example:
        >>> generator = BenchmarkReportGenerator(
        ...     model_name="TTM-SSL Foundation Model",
        ...     experiment_name="Hybrid SSL Pipeline",
        ...     output_dir="artifacts/reports"
        ... )
        >>> generator.add_task_result(
        ...     task_name="quality",
        ...     metric="AUROC",
        ...     value=0.85,
        ...     target=0.88
        ... )
        >>> generator.generate_markdown_report()
    """

    def __init__(
        self,
        model_name: str,
        experiment_name: str,
        output_dir: Union[str, Path],
        author: str = "Foundation Model Team",
        description: str = ""
    ):
        self.model_name = model_name
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.author = author
        self.description = description
        self.timestamp = datetime.now()

        # Storage for results
        self.results: Dict[str, TaskResult] = {}

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def add_task_result(
        self,
        task_name: str,
        task_description: str,
        metric: str,
        value: float,
        ci_lower: Optional[float] = None,
        ci_upper: Optional[float] = None,
        std: Optional[float] = None,
        target: Optional[float] = None,
        baseline: Optional[float] = None,
        sota: Optional[float] = None,
        sota_paper: Optional[str] = None,
        secondary_metrics: Optional[Dict[str, float]] = None,
        num_samples: Optional[int] = None,
        num_subjects: Optional[int] = None,
        training_time: Optional[float] = None
    ):
        """Add a task result to the report.

        Args:
            task_name: Short task identifier (e.g., 'quality', 'hr_estimation')
            task_description: Human-readable description
            metric: Primary metric name (e.g., 'AUROC', 'MAE')
            value: Measured metric value
            ci_lower: Lower bound of 95% confidence interval
            ci_upper: Upper bound of 95% confidence interval
            std: Standard deviation
            target: Target performance from literature
            baseline: Baseline performance
            sota: State-of-the-art performance
            sota_paper: Reference for SOTA
            secondary_metrics: Dict of additional metrics
            num_samples: Number of test samples
            num_subjects: Number of test subjects
            training_time: Training time in seconds
        """
        result = TaskResult(
            task_name=task_name,
            task_description=task_description,
            metric=metric,
            value=value,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            std=std,
            target=target,
            baseline=baseline,
            sota=sota,
            sota_paper=sota_paper,
            secondary_metrics=secondary_metrics or {},
            num_samples=num_samples,
            num_subjects=num_subjects,
            training_time=training_time
        )

        self.results[task_name] = result

    def generate_markdown_report(self, filename: str = "benchmark_report.md") -> Path:
        """Generate a Markdown report with benchmark comparison.

        Returns:
            Path to generated report
        """
        output_path = self.output_dir / filename

        with open(output_path, 'w') as f:
            # Header
            f.write(f"# Benchmark Evaluation Report\n\n")
            f.write(f"**Model:** {self.model_name}  \n")
            f.write(f"**Experiment:** {self.experiment_name}  \n")
            f.write(f"**Date:** {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}  \n")
            f.write(f"**Author:** {self.author}  \n")

            if self.description:
                f.write(f"\n**Description:** {self.description}\n")

            f.write(f"\n---\n\n")

            # Summary statistics
            f.write(f"## Summary\n\n")
            f.write(f"- **Total Tasks:** {len(self.results)}\n")

            exceeds = sum(1 for r in self.results.values() if "EXCEEDS" in r.performance_category())
            meets = sum(1 for r in self.results.values() if "MEETS" in r.performance_category())
            close = sum(1 for r in self.results.values() if "CLOSE" in r.performance_category())
            below = sum(1 for r in self.results.values() if "BELOW" in r.performance_category())

            f.write(f"- **Exceeds Target:** {exceeds}/{len(self.results)}\n")
            f.write(f"- **Meets Target:** {meets}/{len(self.results)}\n")
            f.write(f"- **Close to Target:** {close}/{len(self.results)}\n")
            f.write(f"- **Below Target:** {below}/{len(self.results)}\n")

            f.write(f"\n---\n\n")

            # Main results table
            f.write(f"## Results\n\n")
            f.write("| Task | Metric | Our Model | Target | Baseline | Status | vs Baseline |\n")
            f.write("|------|--------|-----------|--------|----------|--------|-------------|\n")

            for task_name, result in self.results.items():
                # Format value with CI
                if result.ci_lower is not None and result.ci_upper is not None:
                    value_str = f"{result.value:.3f} [{result.ci_lower:.3f}, {result.ci_upper:.3f}]"
                elif result.std is not None:
                    value_str = f"{result.value:.3f} ± {result.std:.3f}"
                else:
                    value_str = f"{result.value:.3f}"

                # Format target
                target_str = f"{result.target:.3f}" if result.target is not None else "N/A"

                # Format baseline
                baseline_str = f"{result.baseline:.3f}" if result.baseline is not None else "N/A"

                # Status
                status = result.performance_category()

                # vs Baseline
                vs_base = result.vs_baseline()

                f.write(f"| {result.task_description} | {result.metric} | {value_str} | "
                       f"{target_str} | {baseline_str} | {status} | {vs_base} |\n")

            f.write(f"\n---\n\n")

            # Detailed results per task
            f.write(f"## Detailed Results\n\n")

            for task_name, result in self.results.items():
                f.write(f"### {result.task_description}\n\n")
                f.write(f"**Primary Metric:** {result.metric} = **{result.value:.4f}**")

                if result.ci_lower is not None and result.ci_upper is not None:
                    f.write(f" (95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}])")
                elif result.std is not None:
                    f.write(f" (±{result.std:.4f})")

                f.write(f"\n\n")

                # Benchmark comparison
                f.write(f"**Benchmark Comparison:**\n")
                if result.target is not None:
                    f.write(f"- Target: {result.target:.4f}\n")
                if result.baseline is not None:
                    f.write(f"- Baseline: {result.baseline:.4f} ({result.vs_baseline()} improvement)\n")
                if result.sota is not None:
                    f.write(f"- SOTA: {result.sota:.4f}")
                    if result.sota_paper:
                        f.write(f" ({result.sota_paper})")
                    f.write(f"\n")

                f.write(f"- **Status:** {result.performance_category()}\n\n")

                # Secondary metrics
                if result.secondary_metrics:
                    f.write(f"**Secondary Metrics:**\n")
                    for metric_name, metric_value in result.secondary_metrics.items():
                        f.write(f"- {metric_name}: {metric_value:.4f}\n")
                    f.write(f"\n")

                # Metadata
                if result.num_samples or result.num_subjects:
                    f.write(f"**Dataset:**\n")
                    if result.num_samples:
                        f.write(f"- Test samples: {result.num_samples}\n")
                    if result.num_subjects:
                        f.write(f"- Test subjects: {result.num_subjects}\n")
                    f.write(f"\n")

                if result.training_time:
                    hours = result.training_time / 3600
                    f.write(f"**Training Time:** {hours:.2f} hours\n\n")

                f.write(f"---\n\n")

            # Footer
            f.write(f"\n*Report generated by BenchmarkReportGenerator on {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}*\n")

        print(f"✅ Markdown report saved: {output_path}")
        return output_path

    def generate_latex_table(self, filename: str = "benchmark_table.tex") -> Path:
        """Generate a LaTeX table suitable for publications.

        Returns:
            Path to generated LaTeX file
        """
        output_path = self.output_dir / filename

        with open(output_path, 'w') as f:
            f.write("% Benchmark Comparison Table\n")
            f.write(f"% Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write(f"\\caption{{{self.experiment_name} - Benchmark Comparison}}\n")
            f.write("\\label{tab:benchmark_results}\n")
            f.write("\\begin{tabular}{llcccc}\n")
            f.write("\\toprule\n")
            f.write("Task & Metric & Our Model & Target & Baseline & Status \\\\\n")
            f.write("\\midrule\n")

            for task_name, result in self.results.items():
                # Escape special characters for LaTeX
                task_desc = result.task_description.replace("&", "\\&")

                # Format value with CI
                if result.ci_lower is not None and result.ci_upper is not None:
                    value_str = f"{result.value:.3f} \\scriptsize{{[{result.ci_lower:.3f}, {result.ci_upper:.3f}]}}"
                elif result.std is not None:
                    value_str = f"{result.value:.3f} $\\pm$ {result.std:.3f}"
                else:
                    value_str = f"{result.value:.3f}"

                # Format target and baseline
                target_str = f"{result.target:.3f}" if result.target is not None else "---"
                baseline_str = f"{result.baseline:.3f}" if result.baseline is not None else "---"

                # Status symbol
                category = result.performance_category()
                if "EXCEEDS" in category:
                    status_str = "\\textcolor{green}{$\\checkmark\\checkmark$}"
                elif "MEETS" in category:
                    status_str = "\\textcolor{green}{$\\checkmark$}"
                elif "CLOSE" in category:
                    status_str = "\\textcolor{orange}{$\\sim$}"
                else:
                    status_str = "\\textcolor{red}{$\\times$}"

                f.write(f"{task_desc} & {result.metric} & {value_str} & {target_str} & "
                       f"{baseline_str} & {status_str} \\\\\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

        print(f"✅ LaTeX table saved: {output_path}")
        return output_path

    def generate_json_summary(self, filename: str = "benchmark_results.json") -> Path:
        """Generate JSON summary for programmatic access.

        Returns:
            Path to generated JSON file
        """
        output_path = self.output_dir / filename

        summary = {
            "metadata": {
                "model_name": self.model_name,
                "experiment_name": self.experiment_name,
                "timestamp": self.timestamp.isoformat(),
                "author": self.author,
                "description": self.description
            },
            "summary_statistics": {
                "total_tasks": len(self.results),
                "exceeds_target": sum(1 for r in self.results.values() if "EXCEEDS" in r.performance_category()),
                "meets_target": sum(1 for r in self.results.values() if "MEETS" in r.performance_category()),
                "close_to_target": sum(1 for r in self.results.values() if "CLOSE" in r.performance_category()),
                "below_target": sum(1 for r in self.results.values() if "BELOW" in r.performance_category())
            },
            "results": {}
        }

        for task_name, result in self.results.items():
            summary["results"][task_name] = {
                "task_description": result.task_description,
                "metric": result.metric,
                "value": float(result.value),
                "ci_lower": float(result.ci_lower) if result.ci_lower is not None else None,
                "ci_upper": float(result.ci_upper) if result.ci_upper is not None else None,
                "std": float(result.std) if result.std is not None else None,
                "target": float(result.target) if result.target is not None else None,
                "baseline": float(result.baseline) if result.baseline is not None else None,
                "sota": float(result.sota) if result.sota is not None else None,
                "sota_paper": result.sota_paper,
                "performance_category": result.performance_category(),
                "vs_baseline": result.vs_baseline(),
                "secondary_metrics": result.secondary_metrics,
                "num_samples": result.num_samples,
                "num_subjects": result.num_subjects,
                "training_time": result.training_time
            }

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"✅ JSON summary saved: {output_path}")
        return output_path

    def generate_html_report(self, filename: str = "benchmark_report.html") -> Path:
        """Generate an interactive HTML report.

        Returns:
            Path to generated HTML file
        """
        output_path = self.output_dir / filename

        with open(output_path, 'w') as f:
            f.write("<!DOCTYPE html>\n")
            f.write("<html lang='en'>\n")
            f.write("<head>\n")
            f.write("  <meta charset='UTF-8'>\n")
            f.write("  <meta name='viewport' content='width=device-width, initial-scale=1.0'>\n")
            f.write(f"  <title>{self.experiment_name} - Benchmark Report</title>\n")
            f.write("  <style>\n")
            f.write("""
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; margin: 40px; background: #f5f5f5; }
    .container { max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
    h2 { color: #34495e; margin-top: 30px; }
    .metadata { background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }
    .metadata p { margin: 5px 0; }
    .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
    .summary-card { background: #fff; border: 2px solid #ddd; padding: 20px; border-radius: 5px; text-align: center; }
    .summary-card h3 { margin: 0 0 10px 0; font-size: 14px; color: #7f8c8d; }
    .summary-card .value { font-size: 32px; font-weight: bold; color: #2c3e50; }
    table { width: 100%; border-collapse: collapse; margin: 20px 0; }
    th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
    th { background: #34495e; color: white; font-weight: 600; }
    tr:hover { background: #f8f9fa; }
    .status-exceeds { color: #27ae60; font-weight: bold; }
    .status-meets { color: #16a085; font-weight: bold; }
    .status-close { color: #f39c12; font-weight: bold; }
    .status-below { color: #e74c3c; font-weight: bold; }
    .task-detail { background: #f8f9fa; padding: 20px; margin: 20px 0; border-left: 4px solid #3498db; border-radius: 3px; }
    .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 10px 0; }
    .metric-item { background: white; padding: 10px; border-radius: 3px; border: 1px solid #ddd; }
    .metric-item strong { display: block; color: #7f8c8d; font-size: 12px; margin-bottom: 5px; }
    .footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #7f8c8d; font-size: 14px; }
""")
            f.write("  </style>\n")
            f.write("</head>\n")
            f.write("<body>\n")
            f.write("  <div class='container'>\n")

            # Header
            f.write(f"    <h1>{self.experiment_name}</h1>\n")
            f.write("    <div class='metadata'>\n")
            f.write(f"      <p><strong>Model:</strong> {self.model_name}</p>\n")
            f.write(f"      <p><strong>Date:</strong> {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>\n")
            f.write(f"      <p><strong>Author:</strong> {self.author}</p>\n")
            if self.description:
                f.write(f"      <p><strong>Description:</strong> {self.description}</p>\n")
            f.write("    </div>\n\n")

            # Summary cards
            exceeds = sum(1 for r in self.results.values() if "EXCEEDS" in r.performance_category())
            meets = sum(1 for r in self.results.values() if "MEETS" in r.performance_category())
            close = sum(1 for r in self.results.values() if "CLOSE" in r.performance_category())
            below = sum(1 for r in self.results.values() if "BELOW" in r.performance_category())

            f.write("    <div class='summary'>\n")
            f.write("      <div class='summary-card'>\n")
            f.write("        <h3>Total Tasks</h3>\n")
            f.write(f"        <div class='value'>{len(self.results)}</div>\n")
            f.write("      </div>\n")
            f.write("      <div class='summary-card'>\n")
            f.write("        <h3>Exceeds Target</h3>\n")
            f.write(f"        <div class='value' style='color: #27ae60;'>{exceeds}</div>\n")
            f.write("      </div>\n")
            f.write("      <div class='summary-card'>\n")
            f.write("        <h3>Meets Target</h3>\n")
            f.write(f"        <div class='value' style='color: #16a085;'>{meets}</div>\n")
            f.write("      </div>\n")
            f.write("      <div class='summary-card'>\n")
            f.write("        <h3>Close to Target</h3>\n")
            f.write(f"        <div class='value' style='color: #f39c12;'>{close}</div>\n")
            f.write("      </div>\n")
            f.write("      <div class='summary-card'>\n")
            f.write("        <h3>Below Target</h3>\n")
            f.write(f"        <div class='value' style='color: #e74c3c;'>{below}</div>\n")
            f.write("      </div>\n")
            f.write("    </div>\n\n")

            # Results table
            f.write("    <h2>Results Overview</h2>\n")
            f.write("    <table>\n")
            f.write("      <tr>\n")
            f.write("        <th>Task</th><th>Metric</th><th>Our Model</th><th>Target</th><th>Baseline</th><th>Status</th><th>vs Baseline</th>\n")
            f.write("      </tr>\n")

            for task_name, result in self.results.items():
                # Format value with CI
                if result.ci_lower is not None and result.ci_upper is not None:
                    value_str = f"{result.value:.3f} <small>[{result.ci_lower:.3f}, {result.ci_upper:.3f}]</small>"
                elif result.std is not None:
                    value_str = f"{result.value:.3f} ± {result.std:.3f}"
                else:
                    value_str = f"{result.value:.3f}"

                target_str = f"{result.target:.3f}" if result.target is not None else "N/A"
                baseline_str = f"{result.baseline:.3f}" if result.baseline is not None else "N/A"

                category = result.performance_category()
                if "EXCEEDS" in category:
                    status_class = "status-exceeds"
                elif "MEETS" in category:
                    status_class = "status-meets"
                elif "CLOSE" in category:
                    status_class = "status-close"
                else:
                    status_class = "status-below"

                f.write("      <tr>\n")
                f.write(f"        <td>{result.task_description}</td>\n")
                f.write(f"        <td>{result.metric}</td>\n")
                f.write(f"        <td>{value_str}</td>\n")
                f.write(f"        <td>{target_str}</td>\n")
                f.write(f"        <td>{baseline_str}</td>\n")
                f.write(f"        <td class='{status_class}'>{category}</td>\n")
                f.write(f"        <td>{result.vs_baseline()}</td>\n")
                f.write("      </tr>\n")

            f.write("    </table>\n\n")

            # Detailed results
            f.write("    <h2>Detailed Results</h2>\n")
            for task_name, result in self.results.items():
                f.write("    <div class='task-detail'>\n")
                f.write(f"      <h3>{result.task_description}</h3>\n")
                f.write("      <div class='metric-grid'>\n")

                # Primary metric
                f.write("        <div class='metric-item'>\n")
                f.write(f"          <strong>{result.metric}</strong>\n")
                f.write(f"          <div style='font-size: 20px; font-weight: bold;'>{result.value:.4f}</div>\n")
                if result.ci_lower and result.ci_upper:
                    f.write(f"          <small>[{result.ci_lower:.4f}, {result.ci_upper:.4f}]</small>\n")
                f.write("        </div>\n")

                # Secondary metrics
                for metric_name, metric_value in result.secondary_metrics.items():
                    f.write("        <div class='metric-item'>\n")
                    f.write(f"          <strong>{metric_name}</strong>\n")
                    f.write(f"          <div style='font-size: 16px;'>{metric_value:.4f}</div>\n")
                    f.write("        </div>\n")

                f.write("      </div>\n")
                f.write("    </div>\n")

            # Footer
            f.write("    <div class='footer'>\n")
            f.write(f"      <p>Report generated on {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>\n")
            f.write("    </div>\n")
            f.write("  </div>\n")
            f.write("</body>\n")
            f.write("</html>\n")

        print(f"✅ HTML report saved: {output_path}")
        return output_path

    def generate_all_reports(self) -> Dict[str, Path]:
        """Generate all report formats.

        Returns:
            Dictionary mapping format name to file path
        """
        print(f"\n{'='*70}")
        print(f"GENERATING BENCHMARK REPORTS")
        print(f"{'='*70}")
        print(f"Model: {self.model_name}")
        print(f"Experiment: {self.experiment_name}")
        print(f"Tasks: {len(self.results)}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*70}\n")

        reports = {
            "markdown": self.generate_markdown_report(),
            "latex": self.generate_latex_table(),
            "json": self.generate_json_summary(),
            "html": self.generate_html_report()
        }

        print(f"\n{'='*70}")
        print(f"✅ ALL REPORTS GENERATED SUCCESSFULLY")
        print(f"{'='*70}\n")

        return reports
