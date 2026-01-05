"""Evaluation module for TTM-HAR."""

from src.evaluation.metrics import (
    compute_metrics,
    accuracy,
    balanced_accuracy,
    macro_f1,
    weighted_f1,
    per_class_metrics,
    confusion_matrix,
    classification_report,
)

from src.evaluation.splitters import (
    SubjectSplitter,
    HoldoutSubjectSplitter,
    KFoldSubjectSplitter,
    LeaveOneSubjectOut,
)

from src.evaluation.evaluator import (
    Evaluator,
    ModelEvaluator,
    evaluate_model,
)

from src.evaluation.analysis import (
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_training_curves,
    analyze_predictions,
    compute_subject_level_metrics,
)

__all__ = [
    # Metrics
    "compute_metrics",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "per_class_metrics",
    "confusion_matrix",
    "classification_report",
    # Splitters
    "SubjectSplitter",
    "HoldoutSubjectSplitter",
    "KFoldSubjectSplitter",
    "LeaveOneSubjectOut",
    # Evaluator
    "Evaluator",
    "ModelEvaluator",
    "evaluate_model",
    # Analysis
    "plot_confusion_matrix",
    "plot_per_class_metrics",
    "plot_training_curves",
    "analyze_predictions",
    "compute_subject_level_metrics",
]
