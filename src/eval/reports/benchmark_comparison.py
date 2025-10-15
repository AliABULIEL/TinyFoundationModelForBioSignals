"""
Benchmark Comparison Report Generator

Generates comprehensive comparison tables and visualizations
comparing model performance against article benchmarks.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class BenchmarkComparator:
    """Generate benchmark comparison reports"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
        sns.set_palette("husl")

    def generate_vitaldb_comparison_table(self, results: Dict) -> str:
        """Generate markdown table comparing VitalDB results to benchmarks"""

        table = """
# VitalDB Benchmark Comparison

| Task | Metric | Your Model | 95% CI | Target | SOTA | Status | Gap to SOTA |
|------|--------|------------|--------|--------|------|--------|-------------|
"""

        # Hypotension
        hypo = results['hypotension']
        status = "✅ PASS" if hypo['target_met'] else "❌ FAIL"
        ci_str = ""
        if 'auroc_ci_lower' in hypo:
            ci_str = f"[{hypo['auroc_ci_lower']:.3f}, {hypo['auroc_ci_upper']:.3f}]"

        table += f"| Hypotension (10-min) | AUROC | {hypo['auroc']:.3f} | {ci_str} | {hypo['target']:.3f} | {hypo['sota']:.3f} | {status} | {hypo['sota_gap']:+.3f} |\n"
        table += f"| | AUPRC | {hypo['auprc']:.3f} | - | - | - | - | - |\n"
        table += f"| | Sensitivity | {hypo['sensitivity']:.3f} | - | - | - | - | - |\n"
        table += f"| | Specificity | {hypo['specificity']:.3f} | - | - | - | - | - |\n"
        table += f"| | Samples | {hypo['n_samples']} ({hypo['n_positive']}+/{hypo['n_negative']}-) | - | - | - | - | - |\n"

        # BP Estimation
        bp = results['bp_estimation']
        mae_status = "✅ PASS" if bp['mae_target_met'] else "❌ FAIL"
        aami_status = "✅ PASS" if bp['aami_compliant'] else "❌ FAIL"

        bp_ci_str = ""
        if 'mae_ci_lower' in bp:
            bp_ci_str = f"[{bp['mae_ci_lower']:.2f}, {bp['mae_ci_upper']:.2f}]"

        table += f"| BP Estimation (MAP) | MAE (mmHg) | {bp['mae']:.2f} | {bp_ci_str} | {bp['mae_target']:.2f} | {bp['mae_sota']:.2f} | {mae_status} | {bp['mae_sota_gap']:+.2f} |\n"
        table += f"| | RMSE (mmHg) | {bp['rmse']:.2f} | - | - | - | - | - |\n"
        table += f"| | R² | {bp['r2']:.3f} | - | - | - | - | - |\n"
        table += f"| | Samples | {bp['n_samples']} | - | - | - | - | - |\n"
        table += f"| BP AAMI Compliance | ME (mmHg) | {bp['me']:.2f} | - | 5.0 | - | {'✅' if bp['me_compliant'] else '❌'} | - |\n"
        table += f"| | SDE (mmHg) | {bp['sde']:.2f} | - | 8.0 | - | {'✅' if bp['sde_compliant'] else '❌'} | - |\n"
        table += f"| | AAMI Pass | {'YES' if bp['aami_compliant'] else 'NO'} | - | - | - | {aami_status} | - |\n"

        table += "\n## Summary\n"
        tasks_passed = sum([hypo['target_met'], bp['mae_target_met'], bp['aami_compliant']])
        table += f"- Tasks Passed: {tasks_passed}/3\n"
        table += f"- Average Gap to SOTA: {(hypo['sota_gap'] + bp['mae_sota_gap'])/2:.3f}\n"

        return table

    def generate_butppg_comparison_table(self, results: Dict) -> str:
        """Generate markdown table comparing BUT-PPG results to benchmarks"""

        table = """
# BUT-PPG Benchmark Comparison

| Task | Metric | Your Model | 95% CI | Target | Baseline | Status | Improvement |
|------|--------|------------|--------|--------|----------|--------|-------------|
"""

        # Quality
        qual = results['quality']
        status = "✅ PASS" if qual['target_met'] else "❌ FAIL"

        qual_ci_str = ""
        if 'auroc_ci_lower' in qual:
            qual_ci_str = f"[{qual['auroc_ci_lower']:.3f}, {qual['auroc_ci_upper']:.3f}]"

        table += f"| Signal Quality | AUROC | {qual['auroc']:.3f} | {qual_ci_str} | {qual['target']:.3f} | {qual['baseline']:.3f} (traditional) | {status} | +{qual['improvement_over_baseline']:.3f} |\n"
        table += f"| | | | | | {qual['dl_baseline']:.3f} (DL) | | +{qual['improvement_over_dl']:.3f} |\n"
        table += f"| | Accuracy | {qual['accuracy']:.3f} | - | - | - | - | - |\n"
        table += f"| | F1 | {qual['f1']:.3f} | - | - | - | - | - |\n"
        table += f"| | Sensitivity | {qual['sensitivity']:.3f} | - | - | - | - | - |\n"
        table += f"| | Specificity | {qual['specificity']:.3f} | - | - | - | - | - |\n"
        table += f"| | Samples | {qual['n_samples']} ({qual['n_positive']}+/{qual['n_negative']}-) | - | - | - | - | - |\n"

        # HR (if present)
        if 'hr_estimation' in results:
            hr = results['hr_estimation']
            hr_status = "✅ PASS" if hr['target_met'] else "❌ FAIL"

            hr_ci_str = ""
            if 'mae_ci_lower' in hr:
                hr_ci_str = f"[{hr['mae_ci_lower']:.2f}, {hr['mae_ci_upper']:.2f}]"

            table += f"| Heart Rate | MAE (bpm) | {hr['mae']:.2f} | {hr_ci_str} | {hr['target']:.2f} | {hr['baseline']:.2f} (traditional) | {hr_status} | +{hr['vs_baseline']:.2f} |\n"
            table += f"| | | | | {hr['human_expert']:.2f} (human) | | | {hr['vs_human_expert']:+.2f} |\n"
            table += f"| | RMSE (bpm) | {hr['rmse']:.2f} | - | - | - | - | - |\n"
            table += f"| | MAPE (%) | {hr['mape']:.1f} | - | - | - | - | - |\n"
            table += f"| | Within 5 bpm | {hr['within_5bpm']:.1f}% | - | - | - | - | - |\n"
            table += f"| | Samples | {hr['n_samples']} | - | - | - | - | - |\n"

        # Motion (if present)
        if 'motion' in results:
            motion = results['motion']
            motion_status = "✅ PASS" if motion['target_met'] else "❌ FAIL"

            motion_ci_str = ""
            if 'accuracy_ci_lower' in motion:
                motion_ci_str = f"[{motion['accuracy_ci_lower']:.3f}, {motion['accuracy_ci_upper']:.3f}]"

            table += f"| Motion (8-class) | Accuracy | {motion['accuracy']:.3f} | {motion_ci_str} | {motion['target']:.3f} | {motion['baseline']:.3f} | {motion_status} | +{motion['improvement_over_baseline']:.3f} |\n"
            table += f"| | F1 (macro) | {motion['f1_macro']:.3f} | - | - | - | - | - |\n"
            table += f"| | F1 (weighted) | {motion['f1_weighted']:.3f} | - | - | - | - | - |\n"
            table += f"| | Samples | {motion['n_samples']} | - | - | - | - | - |\n"

        table += "\n## Summary\n"
        tasks_passed = [qual['target_met']]
        if 'hr_estimation' in results:
            tasks_passed.append(results['hr_estimation']['target_met'])
        if 'motion' in results:
            tasks_passed.append(results['motion']['target_met'])

        table += f"- Tasks Passed: {sum(tasks_passed)}/{len(tasks_passed)}\n"

        return table

    def generate_comparison_plots(
        self,
        vitaldb_results: Dict,
        butppg_results: Dict
    ):
        """Generate visualization comparing results to benchmarks"""

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: VitalDB AUROC comparison
        ax = axes[0, 0]
        tasks = ['Hypotension\n(10-min)']
        your_scores = [vitaldb_results['hypotension']['auroc']]
        targets = [vitaldb_results['hypotension']['target']]
        sotas = [vitaldb_results['hypotension']['sota']]

        # Add error bars if CI available
        yerr = None
        if 'auroc_ci_lower' in vitaldb_results['hypotension']:
            ci_low = vitaldb_results['hypotension']['auroc_ci_lower']
            ci_high = vitaldb_results['hypotension']['auroc_ci_upper']
            yerr = [[your_scores[0] - ci_low], [ci_high - your_scores[0]]]

        x = np.arange(len(tasks))
        width = 0.25

        ax.bar(x - width, your_scores, width, label='Your Model', color='#2ecc71', yerr=yerr, capsize=5)
        ax.bar(x, targets, width, label='Target', color='#f39c12')
        ax.bar(x + width, sotas, width, label='SOTA', color='#3498db')

        ax.set_ylabel('AUROC', fontsize=12, fontweight='bold')
        ax.set_title('VitalDB: Hypotension Prediction', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, fontsize=10)
        ax.legend(fontsize=10)
        ax.set_ylim([0.85, 1.0])
        ax.grid(axis='y', alpha=0.3)

        # Plot 2: VitalDB MAE comparison
        ax = axes[0, 1]
        tasks = ['BP (MAP)']
        your_scores = [vitaldb_results['bp_estimation']['mae']]
        targets = [vitaldb_results['bp_estimation']['mae_target']]
        sotas = [vitaldb_results['bp_estimation']['mae_sota']]

        # Add error bars if CI available
        yerr = None
        if 'mae_ci_lower' in vitaldb_results['bp_estimation']:
            ci_low = vitaldb_results['bp_estimation']['mae_ci_lower']
            ci_high = vitaldb_results['bp_estimation']['mae_ci_upper']
            yerr = [[your_scores[0] - ci_low], [ci_high - your_scores[0]]]

        x = np.arange(len(tasks))
        ax.bar(x - width, your_scores, width, label='Your Model', color='#2ecc71', yerr=yerr, capsize=5)
        ax.bar(x, targets, width, label='Target', color='#f39c12')
        ax.bar(x + width, sotas, width, label='SOTA', color='#3498db')

        ax.set_ylabel('MAE (mmHg)', fontsize=12, fontweight='bold')
        ax.set_title('VitalDB: Blood Pressure Estimation', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        # Plot 3: BUT-PPG AUROC comparison
        ax = axes[1, 0]
        tasks = ['Signal\nQuality']
        your_scores = [butppg_results['quality']['auroc']]
        targets = [butppg_results['quality']['target']]
        baselines = [butppg_results['quality']['dl_baseline']]

        # Add error bars if CI available
        yerr = None
        if 'auroc_ci_lower' in butppg_results['quality']:
            ci_low = butppg_results['quality']['auroc_ci_lower']
            ci_high = butppg_results['quality']['auroc_ci_upper']
            yerr = [[your_scores[0] - ci_low], [ci_high - your_scores[0]]]

        x = np.arange(len(tasks))
        ax.bar(x - width, your_scores, width, label='Your Model', color='#2ecc71', yerr=yerr, capsize=5)
        ax.bar(x, targets, width, label='Target', color='#f39c12')
        ax.bar(x + width, baselines, width, label='DL Baseline', color='#e74c3c')

        ax.set_ylabel('AUROC', fontsize=12, fontweight='bold')
        ax.set_title('BUT-PPG: Quality Classification', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, fontsize=10)
        ax.legend(fontsize=10)
        ax.set_ylim([0.70, 1.0])
        ax.grid(axis='y', alpha=0.3)

        # Plot 4: BUT-PPG HR MAE comparison (if available)
        ax = axes[1, 1]
        if 'hr_estimation' in butppg_results:
            tasks = ['Heart Rate']
            your_scores = [butppg_results['hr_estimation']['mae']]
            targets = [butppg_results['hr_estimation']['target']]
            human = [butppg_results['hr_estimation']['human_expert']]

            # Add error bars if CI available
            yerr = None
            if 'mae_ci_lower' in butppg_results['hr_estimation']:
                ci_low = butppg_results['hr_estimation']['mae_ci_lower']
                ci_high = butppg_results['hr_estimation']['mae_ci_upper']
                yerr = [[your_scores[0] - ci_low], [ci_high - your_scores[0]]]

            x = np.arange(len(tasks))
            ax.bar(x - width, your_scores, width, label='Your Model', color='#2ecc71', yerr=yerr, capsize=5)
            ax.bar(x, targets, width, label='Target', color='#f39c12')
            ax.bar(x + width, human, width, label='Human Expert', color='#9b59b6')

            ax.set_ylabel('MAE (bpm)', fontsize=12, fontweight='bold')
            ax.set_title('BUT-PPG: HR Estimation', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(tasks, fontsize=10)
            ax.legend(fontsize=10)
            ax.grid(axis='y', alpha=0.3)
        else:
            # Placeholder if HR not available
            ax.text(0.5, 0.5, 'HR Estimation\nNot Evaluated',
                   ha='center', va='center', fontsize=14, color='gray')
            ax.axis('off')

        plt.tight_layout()
        plot_path = self.output_dir / 'benchmark_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved: {plot_path}")

    def generate_full_report(
        self,
        vitaldb_results: Dict,
        butppg_results: Dict,
        model_info: Dict
    ):
        """Generate comprehensive HTML benchmark report"""

        vitaldb_md = self.generate_vitaldb_comparison_table(vitaldb_results)
        butppg_md = self.generate_butppg_comparison_table(butppg_results)

        report = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Benchmark Comparison Report</title>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 40px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-top: 40px;
        }}
        h3 {{
            color: #34495e;
            margin-top: 30px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .pass {{
            color: #27ae60;
            font-weight: bold;
        }}
        .fail {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .metric {{
            font-weight: bold;
            color: #2980b9;
        }}
        .summary-box {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
            border-left: 4px solid #3498db;
        }}
        .summary-box h3 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        img {{
            max-width: 100%;
            height: auto;
            margin: 20px 0;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 0.9em;
            margin-top: 10px;
        }}
        .model-info {{
            background-color: #fff;
            border: 1px solid #ddd;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .model-info p {{
            margin: 5px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Foundation Model Benchmark Comparison Report</h1>

        <div class="model-info">
            <h3>Model Information</h3>
            <p><strong>Model:</strong> {model_info.get('name', 'TTM Foundation Model')}</p>
            <p><strong>Parameters:</strong> {model_info.get('parameters', 'N/A')}</p>
            <p><strong>Architecture:</strong> {model_info.get('architecture', 'IBM Granite TTM')}</p>
            <p class="timestamp"><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="summary-box">
            <h3>Executive Summary</h3>
            <p>This report compares your fine-tuned model's performance against targets and SOTA results from the article across 6 downstream tasks.</p>
        </div>

        <h2>VitalDB Tasks</h2>
        <pre style="white-space: pre-wrap; font-family: inherit;">{vitaldb_md}</pre>

        <h2>BUT-PPG Tasks</h2>
        <pre style="white-space: pre-wrap; font-family: inherit;">{butppg_md}</pre>

        <h2>Visual Comparison</h2>
        <img src="benchmark_comparison.png" alt="Benchmark Comparison Charts">

        <div class="summary-box">
            <h3>Notes</h3>
            <ul>
                <li><strong>Subject-level evaluation:</strong> All metrics computed at subject level to prevent data leakage</li>
                <li><strong>Confidence intervals:</strong> 95% CIs computed using bootstrap resampling (1000 iterations)</li>
                <li><strong>AAMI compliance:</strong> Per-subject errors aggregated according to AAMI standard</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""

        html_path = self.output_dir / 'benchmark_report.html'
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"✓ Saved: {html_path}")

        # Also save markdown versions
        with open(self.output_dir / 'vitaldb_comparison.md', 'w') as f:
            f.write(vitaldb_md)

        with open(self.output_dir / 'butppg_comparison.md', 'w') as f:
            f.write(butppg_md)

        print(f"✓ Saved: {self.output_dir / 'vitaldb_comparison.md'}")
        print(f"✓ Saved: {self.output_dir / 'butppg_comparison.md'}")
