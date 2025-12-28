"""Evaluation metrics and utilities."""

from .metrics import (
    # Classification metrics
    accuracy,
    auroc,
    auprc,
    f1,
    precision,
    recall,
    classification_metrics,
    # Regression metrics
    mae,
    rmse,
    mse,
    ccc,
    pearson_r,
    r2,
    regression_metrics,
    # Utilities
    MetricTracker
)

from .benchmarks import (
    BenchmarkResult,
    VITALDB_BENCHMARKS,
    BUTPPG_BENCHMARKS,
    ALL_BENCHMARKS,
    get_benchmark,
    get_target_metric,
    get_sota,
    get_baseline,
    categorize_performance,
    format_benchmark_table,
    print_all_benchmarks
)

from .evaluator import (
    DownstreamEvaluator,
    evaluate_model_on_task
)

from .visualization import (
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    plot_regression_scatter,
    plot_bland_altman,
    plot_benchmark_comparison,
    plot_per_subject_distribution,
    create_evaluation_report
)

__all__ = [
    # Classification
    'accuracy',
    'auroc',
    'auprc',
    'f1',
    'precision',
    'recall',
    'classification_metrics',
    # Regression
    'mae',
    'rmse',
    'mse',
    'ccc',
    'pearson_r',
    'r2',
    'regression_metrics',
    # Utilities
    'MetricTracker',
    # Benchmarks
    'BenchmarkResult',
    'VITALDB_BENCHMARKS',
    'BUTPPG_BENCHMARKS',
    'ALL_BENCHMARKS',
    'get_benchmark',
    'get_target_metric',
    'get_sota',
    'get_baseline',
    'categorize_performance',
    'format_benchmark_table',
    'print_all_benchmarks',
    # Evaluator
    'DownstreamEvaluator',
    'evaluate_model_on_task',
    # Visualization
    'plot_roc_curve',
    'plot_precision_recall_curve',
    'plot_confusion_matrix',
    'plot_regression_scatter',
    'plot_bland_altman',
    'plot_benchmark_comparison',
    'plot_per_subject_distribution',
    'create_evaluation_report'
]
