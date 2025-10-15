"""Benchmark definitions from TTM Foundation Model articles.

Contains SOTA results and target metrics for VitalDB and BUT-PPG tasks.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    task: str
    dataset: str
    metric: str
    value: float
    std: Optional[float] = None
    model: str = ""
    paper: str = ""
    year: int = 0
    notes: str = ""


# ============================================================================
# VitalDB Benchmarks
# ============================================================================

VITALDB_HYPOTENSION_BENCHMARKS = [
    BenchmarkResult(
        task="Hypotension Prediction (10-min)",
        dataset="VitalDB",
        metric="AUROC",
        value=0.934,
        model="SAFDNet (Multi-modal: ABP+ECG+PPG+CO2)",
        paper="ArXiv",
        year=2024,
        notes="Current SOTA - Multi-modal approach"
    ),
    BenchmarkResult(
        task="Hypotension Prediction (10-min)",
        dataset="VitalDB",
        metric="AUROC",
        value=0.91,
        model="Target for TTM (Multi-modal)",
        paper="Article Recommendation",
        year=2025,
        notes="Target performance for TTM foundation model"
    ),
    BenchmarkResult(
        task="Hypotension Prediction (10-min)",
        dataset="VitalDB",
        metric="AUROC",
        value=0.88,
        std=0.02,
        model="ABP-only baseline",
        paper="Baseline",
        year=2024,
        notes="Single modality baseline"
    ),
]

VITALDB_BP_BENCHMARKS = [
    BenchmarkResult(
        task="Blood Pressure (MAP)",
        dataset="VitalDB",
        metric="MAE",
        value=3.8,
        std=5.7,
        model="AnesthNet (calibrated, tested on LaribDB)",
        paper="Nature",
        year=2025,
        notes="Current SOTA with personalized calibration"
    ),
    BenchmarkResult(
        task="Blood Pressure (MAP)",
        dataset="VitalDB",
        metric="MAE",
        value=5.0,
        model="Target for TTM (calibration-free)",
        paper="Article Recommendation",
        year=2025,
        notes="Target for calibration-free approach"
    ),
    BenchmarkResult(
        task="Blood Pressure (MAP) - AAMI",
        dataset="VitalDB",
        metric="ME",
        value=5.0,
        model="AAMI Standard (Mean Error)",
        paper="AAMI Standard",
        year=2024,
        notes="Medical device standard - ME ≤ 5 mmHg"
    ),
    BenchmarkResult(
        task="Blood Pressure (MAP) - AAMI",
        dataset="VitalDB",
        metric="SDE",
        value=8.0,
        model="AAMI Standard (Standard Deviation)",
        paper="AAMI Standard",
        year=2024,
        notes="Medical device standard - SDE ≤ 8 mmHg"
    ),
]

# ============================================================================
# BUT-PPG Benchmarks
# ============================================================================

BUTPPG_QUALITY_BENCHMARKS = [
    BenchmarkResult(
        task="PPG Quality Assessment",
        dataset="BUT-PPG",
        metric="AUROC",
        value=0.88,
        model="Target for TTM",
        paper="Article Recommendation",
        year=2025,
        notes="Target performance after VitalDB pretraining"
    ),
    BenchmarkResult(
        task="PPG Quality Assessment",
        dataset="BUT-PPG",
        metric="AUROC",
        value=0.85,
        std=0.03,
        model="Deep Learning Baseline",
        paper="Baseline",
        year=2023,
        notes="Typical DL performance range: 0.85-0.90"
    ),
    BenchmarkResult(
        task="PPG Quality Assessment",
        dataset="BUT-PPG",
        metric="AUROC",
        value=0.758,
        model="STD-width SQI (traditional)",
        paper="MDPI",
        year=2023,
        notes="Traditional signal quality index baseline"
    ),
    BenchmarkResult(
        task="PPG Quality Assessment",
        dataset="BUT-PPG",
        metric="AUROC",
        value=0.74,
        std=0.02,
        model="Traditional SQI methods",
        paper="Baseline Range",
        year=2023,
        notes="Traditional methods range: 0.74-0.76"
    ),
]

BUTPPG_HR_BENCHMARKS = [
    BenchmarkResult(
        task="Heart Rate Estimation",
        dataset="BUT-PPG",
        metric="MAE",
        value=1.5,
        std=0.5,
        model="Human Expert Consensus",
        paper="BioMed",
        year=2021,
        notes="Human baseline: 1.5-2.0 bpm"
    ),
    BenchmarkResult(
        task="Heart Rate Estimation",
        dataset="BUT-PPG",
        metric="MAE",
        value=2.0,
        model="Target for TTM",
        paper="Article Recommendation",
        year=2025,
        notes="Target to match human expert performance"
    ),
]

# ============================================================================
# Benchmark Collections
# ============================================================================

VITALDB_BENCHMARKS = {
    "hypotension": VITALDB_HYPOTENSION_BENCHMARKS,
    "blood_pressure": VITALDB_BP_BENCHMARKS,
}

BUTPPG_BENCHMARKS = {
    "quality": BUTPPG_QUALITY_BENCHMARKS,
    "heart_rate": BUTPPG_HR_BENCHMARKS,
}

ALL_BENCHMARKS = {
    "vitaldb": VITALDB_BENCHMARKS,
    "butppg": BUTPPG_BENCHMARKS,
}


# ============================================================================
# Helper Functions
# ============================================================================

def get_benchmark(dataset: str, task: str) -> List[BenchmarkResult]:
    """Get benchmark results for a specific task.
    
    Args:
        dataset: 'vitaldb' or 'butppg'
        task: Task name (e.g., 'hypotension', 'quality')
        
    Returns:
        List of benchmark results
    """
    if dataset not in ALL_BENCHMARKS:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    if task not in ALL_BENCHMARKS[dataset]:
        raise ValueError(f"Unknown task '{task}' for dataset '{dataset}'")
    
    return ALL_BENCHMARKS[dataset][task]


def get_target_metric(dataset: str, task: str) -> BenchmarkResult:
    """Get the target metric for a task.
    
    Args:
        dataset: 'vitaldb' or 'butppg'
        task: Task name
        
    Returns:
        Target benchmark result
    """
    benchmarks = get_benchmark(dataset, task)
    
    # Find the target (TTM goal)
    for bench in benchmarks:
        if "Target for TTM" in bench.model or "Article Recommendation" in bench.paper:
            return bench
    
    # If no explicit target, return SOTA
    return benchmarks[0]


def get_sota(dataset: str, task: str) -> BenchmarkResult:
    """Get the current SOTA result for a task.
    
    Args:
        dataset: 'vitaldb' or 'butppg'
        task: Task name
        
    Returns:
        SOTA benchmark result
    """
    benchmarks = get_benchmark(dataset, task)
    
    # Find the SOTA (highest value, excluding targets and standards)
    sota = None
    for bench in benchmarks:
        if "SOTA" in bench.notes or bench.year >= 2024:
            if "Target" not in bench.model and "Standard" not in bench.model:
                if sota is None or bench.value > sota.value:
                    sota = bench
    
    # If no SOTA found, return first
    return sota if sota else benchmarks[0]


def get_baseline(dataset: str, task: str) -> BenchmarkResult:
    """Get the baseline result for a task.
    
    Args:
        dataset: 'vitaldb' or 'butppg'
        task: Task name
        
    Returns:
        Baseline benchmark result
    """
    benchmarks = get_benchmark(dataset, task)
    
    # Find the baseline (lowest performing or explicitly labeled baseline)
    baseline = None
    for bench in benchmarks:
        if "Baseline" in bench.model or "Baseline" in bench.paper or "traditional" in bench.model.lower():
            if baseline is None or bench.value < baseline.value:
                baseline = bench
    
    # If no baseline found, return last
    return baseline if baseline else benchmarks[-1]


def format_benchmark_table(dataset: str, task: str) -> str:
    """Format benchmark results as a table.
    
    Args:
        dataset: 'vitaldb' or 'butppg'
        task: Task name
        
    Returns:
        Formatted table string
    """
    benchmarks = get_benchmark(dataset, task)
    
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"{benchmarks[0].task} - {dataset.upper()} Benchmarks")
    lines.append(f"{'='*80}")
    lines.append(f"{'Model':<40} {'Metric':<10} {'Value':<15} {'Year':<6}")
    lines.append(f"{'-'*80}")
    
    for bench in benchmarks:
        value_str = f"{bench.value:.3f}"
        if bench.std is not None:
            value_str += f" ± {bench.std:.3f}"
        
        lines.append(f"{bench.model:<40} {bench.metric:<10} {value_str:<15} {bench.year:<6}")
        if bench.notes:
            lines.append(f"  └─ {bench.notes}")
    
    lines.append(f"{'='*80}\n")
    
    return "\n".join(lines)


def print_all_benchmarks():
    """Print all benchmark tables."""
    print("\n" + "="*80)
    print("TTM FOUNDATION MODEL - BENCHMARK TARGETS")
    print("="*80)
    
    print("\n" + "🏥 VITALDB TASKS ".ljust(80, "="))
    for task_name in VITALDB_BENCHMARKS.keys():
        print(format_benchmark_table("vitaldb", task_name))
    
    print("\n" + "📱 BUT-PPG TASKS ".ljust(80, "="))
    for task_name in BUTPPG_BENCHMARKS.keys():
        print(format_benchmark_table("butppg", task_name))


# ============================================================================
# Performance Categories
# ============================================================================

def categorize_performance(
    dataset: str,
    task: str,
    metric: str,
    value: float,
    higher_is_better: bool = True
) -> str:
    """Categorize performance relative to benchmarks.
    
    Args:
        dataset: 'vitaldb' or 'butppg'
        task: Task name
        metric: Metric name
        value: Achieved value
        higher_is_better: Whether higher values are better
        
    Returns:
        Performance category: 'SOTA', 'Target', 'Good', 'Baseline', 'Below Baseline'
    """
    target = get_target_metric(dataset, task)
    sota = get_sota(dataset, task)
    baseline = get_baseline(dataset, task)
    
    if higher_is_better:
        if value >= sota.value:
            return "🏆 SOTA"
        elif value >= target.value:
            return "✅ Target Achieved"
        elif value >= baseline.value * 1.1:  # 10% above baseline
            return "✓ Good"
        elif value >= baseline.value:
            return "→ Baseline"
        else:
            return "⚠ Below Baseline"
    else:
        if value <= sota.value:
            return "🏆 SOTA"
        elif value <= target.value:
            return "✅ Target Achieved"
        elif value <= baseline.value * 0.9:  # 10% below baseline (better)
            return "✓ Good"
        elif value <= baseline.value:
            return "→ Baseline"
        else:
            return "⚠ Below Baseline"


if __name__ == "__main__":
    print_all_benchmarks()
