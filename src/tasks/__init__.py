"""VitalDB Downstream Tasks Registry.

Provides centralized access to all implemented downstream tasks.
"""

from .base import BaseTask, ClassificationTask, RegressionTask, TaskConfig, Benchmark
from .hypotension import HypotensionPredictionTask
from .blood_pressure import BloodPressureEstimationTask
from .clinical_tasks import (
    CardiacOutputTask,
    MortalityPredictionTask,
    ICUAdmissionTask,
    AKIPredictionTask,
    AnesthesiaDepthTask,
    SignalQualityTask
)


# Task registry
TASK_REGISTRY = {
    # Time-series prediction tasks
    'hypotension_5min': lambda: HypotensionPredictionTask(prediction_window_min=5),
    'hypotension_10min': lambda: HypotensionPredictionTask(prediction_window_min=10),
    'hypotension_15min': lambda: HypotensionPredictionTask(prediction_window_min=15),
    
    # Regression tasks
    'blood_pressure_sbp': lambda: BloodPressureEstimationTask(target='sbp'),
    'blood_pressure_dbp': lambda: BloodPressureEstimationTask(target='dbp'),
    'blood_pressure_both': lambda: BloodPressureEstimationTask(target='both'),
    'cardiac_output': lambda: CardiacOutputTask(),
    'anesthesia_depth': lambda: AnesthesiaDepthTask(),
    
    # Classification tasks
    'mortality_30day': lambda: MortalityPredictionTask(prediction_window_days=30),
    'mortality_90day': lambda: MortalityPredictionTask(prediction_window_days=90),
    'icu_admission': lambda: ICUAdmissionTask(),
    'aki_prediction': lambda: AKIPredictionTask(),
    'signal_quality': lambda: SignalQualityTask(),
}


def get_task(task_name: str) -> BaseTask:
    """Get task by name from registry.
    
    Args:
        task_name: Name of the task
        
    Returns:
        Task instance
        
    Raises:
        ValueError: If task not found
    """
    if task_name not in TASK_REGISTRY:
        available_tasks = ', '.join(TASK_REGISTRY.keys())
        raise ValueError(
            f"Task '{task_name}' not found. "
            f"Available tasks: {available_tasks}"
        )
    
    return TASK_REGISTRY[task_name]()


def list_tasks() -> list:
    """List all available tasks.
    
    Returns:
        List of task names
    """
    return list(TASK_REGISTRY.keys())


def get_task_info(task_name: str) -> dict:
    """Get information about a task.
    
    Args:
        task_name: Name of the task
        
    Returns:
        Dictionary with task information
    """
    task = get_task(task_name)
    
    info = {
        'name': task.config.name,
        'type': task.config.task_type,
        'required_channels': task.config.required_channels,
        'benchmarks': [
            {
                'paper': b.paper,
                'year': b.year,
                'metrics': b.metrics,
                'notes': b.notes
            }
            for b in task.benchmarks
        ]
    }
    
    if task.config.num_classes:
        info['num_classes'] = task.config.num_classes
    if task.config.target_dim:
        info['target_dim'] = task.config.target_dim
    if task.config.clinical_threshold:
        info['clinical_threshold'] = task.config.clinical_threshold
    
    return info


__all__ = [
    # Base classes
    'BaseTask',
    'ClassificationTask',
    'RegressionTask',
    'TaskConfig',
    'Benchmark',
    
    # Task implementations
    'HypotensionPredictionTask',
    'BloodPressureEstimationTask',
    'CardiacOutputTask',
    'MortalityPredictionTask',
    'ICUAdmissionTask',
    'AKIPredictionTask',
    'AnesthesiaDepthTask',
    'SignalQualityTask',
    
    # Registry functions
    'get_task',
    'list_tasks',
    'get_task_info',
    'TASK_REGISTRY',
]
