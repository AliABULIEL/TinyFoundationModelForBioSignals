"""Downstream task evaluators for VitalDB and BUT-PPG datasets."""

from .vitaldb_tasks import (
    HypotensionPredictor,
    BPEstimator,
    run_all_vitaldb_tasks,
    VITALDB_BENCHMARKS
)
from .butppg_tasks import (
    QualityClassifier,
    HREstimator,
    MotionClassifier,
    run_all_butppg_tasks,
    BUTPPG_BENCHMARKS
)

__all__ = [
    'HypotensionPredictor',
    'BPEstimator',
    'run_all_vitaldb_tasks',
    'VITALDB_BENCHMARKS',
    'QualityClassifier',
    'HREstimator',
    'MotionClassifier',
    'run_all_butppg_tasks',
    'BUTPPG_BENCHMARKS',
]
