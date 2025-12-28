"""Label-Window Alignment Module.

Handles mapping between case-level clinical labels and temporal window segments.
Supports various alignment strategies and temporal label interpolation.

Author: Senior Data Engineering Team
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

import numpy as np
import pandas as pd

from .clinical_labels import ClinicalLabels, ClinicalLabelExtractor, LabelType

logger = logging.getLogger(__name__)


@dataclass
class WindowLabel:
    """Label information for a single window."""
    window_idx: int
    case_id: str
    start_time: float  # seconds
    end_time: float  # seconds
    labels: Dict[str, Any]
    temporal_values: Optional[Dict[str, float]] = None  # Time-varying labels within window
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'window_idx': self.window_idx,
            'case_id': self.case_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'labels': self.labels,
            'temporal_values': self.temporal_values,
            'metadata': self.metadata
        }


class LabelWindowAligner:
    """Aligns clinical labels with temporal windows.
    
    Handles:
    - Case-level label propagation to all windows
    - Temporal label interpolation
    - Missing data strategies
    - Multi-task label organization
    
    Example:
        >>> aligner = LabelWindowAligner(
        ...     label_extractor=extractor,
        ...     window_duration=10.0
        ... )
        >>> window_labels = aligner.align_case_windows(
        ...     case_id="1",
        ...     n_windows=100,
        ...     label_names=["mortality", "age"]
        ... )
    """
    
    def __init__(
        self,
        label_extractor: ClinicalLabelExtractor,
        window_duration: float = 10.0,
        propagation_strategy: str = "replicate"  # replicate, time_decay, interpolate
    ):
        """Initialize label-window aligner.
        
        Args:
            label_extractor: Clinical label extractor instance
            window_duration: Duration of each window in seconds
            propagation_strategy: How to propagate case-level labels to windows
        """
        self.label_extractor = label_extractor
        self.window_duration = window_duration
        self.propagation_strategy = propagation_strategy
        
        logger.info(f"LabelWindowAligner initialized (window_duration={window_duration}s, strategy={propagation_strategy})")
    
    def align_case_windows(
        self,
        case_id: Union[str, int],
        n_windows: int,
        label_names: Optional[List[str]] = None,
        start_time: float = 0.0,
        temporal_labels: Optional[pd.DataFrame] = None
    ) -> List[WindowLabel]:
        """Align labels with windows for a specific case.
        
        Args:
            case_id: Case identifier
            n_windows: Number of windows in the case
            label_names: Specific labels to align
            start_time: Start time of first window in seconds
            temporal_labels: Optional time-varying labels (timestamp, values)
            
        Returns:
            List of WindowLabel objects, one per window
        """
        # Extract case-level labels
        case_labels = self.label_extractor.extract_case_labels(case_id, label_names)
        
        if case_labels is None:
            logger.warning(f"No labels found for case {case_id}")
            return []
        
        # Create window labels
        window_labels = []
        
        for window_idx in range(n_windows):
            win_start = start_time + (window_idx * self.window_duration)
            win_end = win_start + self.window_duration
            
            # Propagate case-level labels
            propagated_labels = self._propagate_labels(
                case_labels.labels,
                window_idx,
                n_windows
            )
            
            # Extract temporal values if available
            temporal_values = None
            if temporal_labels is not None:
                temporal_values = self._extract_temporal_values(
                    temporal_labels,
                    win_start,
                    win_end
                )
            
            window_label = WindowLabel(
                window_idx=window_idx,
                case_id=str(case_id),
                start_time=win_start,
                end_time=win_end,
                labels=propagated_labels,
                temporal_values=temporal_values,
                metadata={
                    'n_windows': n_windows,
                    'relative_position': window_idx / max(n_windows - 1, 1)
                }
            )
            
            window_labels.append(window_label)
        
        logger.debug(f"Aligned {len(window_labels)} windows for case {case_id}")
        return window_labels
    
    def _propagate_labels(
        self,
        case_labels: Dict[str, Any],
        window_idx: int,
        n_windows: int
    ) -> Dict[str, Any]:
        """Propagate case-level labels to a specific window.
        
        Args:
            case_labels: Case-level labels
            window_idx: Window index
            n_windows: Total number of windows
            
        Returns:
            Labels for the window
        """
        propagated = {}
        
        if self.propagation_strategy == "replicate":
            # Simple replication - same label for all windows
            propagated = case_labels.copy()
            
        elif self.propagation_strategy == "time_decay":
            # Apply temporal decay for certain labels
            relative_pos = window_idx / max(n_windows - 1, 1)
            
            for label_name, value in case_labels.items():
                config = self.label_extractor.label_configs.get(label_name)
                
                # Only apply decay to continuous outcomes
                if config and config.label_type == LabelType.CONTINUOUS:
                    # Example: mortality risk increases with time
                    if 'mortality' in label_name.lower() or 'death' in label_name.lower():
                        # Early windows have lower risk
                        decay_factor = 0.5 + 0.5 * relative_pos
                        propagated[label_name] = value * decay_factor
                    else:
                        propagated[label_name] = value
                else:
                    propagated[label_name] = value
        
        else:
            # Default to replication
            propagated = case_labels.copy()
        
        return propagated
    
    def _extract_temporal_values(
        self,
        temporal_labels: pd.DataFrame,
        start_time: float,
        end_time: float
    ) -> Dict[str, float]:
        """Extract temporal label values within a window.
        
        Args:
            temporal_labels: DataFrame with timestamp column and value columns
            start_time: Window start time
            end_time: Window end time
            
        Returns:
            Dictionary of temporal values (mean, std, min, max per label)
        """
        if 'timestamp' not in temporal_labels.columns:
            logger.warning("Temporal labels missing 'timestamp' column")
            return {}
        
        # Filter to window timerange
        window_data = temporal_labels[
            (temporal_labels['timestamp'] >= start_time) &
            (temporal_labels['timestamp'] < end_time)
        ]
        
        if window_data.empty:
            return {}
        
        # Compute statistics for each temporal label
        temporal_values = {}
        
        for column in window_data.columns:
            if column == 'timestamp':
                continue
            
            values = window_data[column].dropna()
            if len(values) > 0:
                temporal_values[f"{column}_mean"] = float(values.mean())
                temporal_values[f"{column}_std"] = float(values.std())
                temporal_values[f"{column}_min"] = float(values.min())
                temporal_values[f"{column}_max"] = float(values.max())
        
        return temporal_values
    
    def align_batch_windows(
        self,
        case_window_mapping: Dict[Union[str, int], int],
        label_names: Optional[List[str]] = None,
        temporal_labels_dict: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Dict[str, List[WindowLabel]]:
        """Align labels for multiple cases efficiently.
        
        Args:
            case_window_mapping: Dictionary mapping case_id -> n_windows
            label_names: Specific labels to align
            temporal_labels_dict: Optional dict mapping case_id -> temporal labels
            
        Returns:
            Dictionary mapping case_id -> list of WindowLabel objects
        """
        results = {}
        
        for case_id, n_windows in case_window_mapping.items():
            temporal_labels = None
            if temporal_labels_dict and case_id in temporal_labels_dict:
                temporal_labels = temporal_labels_dict[case_id]
            
            window_labels = self.align_case_windows(
                case_id=case_id,
                n_windows=n_windows,
                label_names=label_names,
                temporal_labels=temporal_labels
            )
            
            if window_labels:
                results[str(case_id)] = window_labels
        
        total_windows = sum(len(labels) for labels in results.values())
        logger.info(f"Aligned {total_windows} windows across {len(results)} cases")
        
        return results
    
    def create_window_label_arrays(
        self,
        window_labels: List[WindowLabel],
        label_names: List[str],
        return_masks: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Convert window labels to numpy arrays for training.
        
        Args:
            window_labels: List of WindowLabel objects
            label_names: Ordered list of label names to extract
            return_masks: Return binary mask for missing values
            
        Returns:
            label_array: Array of shape [n_windows, n_labels]
            mask_array: Binary mask [n_windows, n_labels] (1=valid, 0=missing)
        """
        n_windows = len(window_labels)
        n_labels = len(label_names)
        
        label_array = np.full((n_windows, n_labels), np.nan, dtype=np.float32)
        mask_array = np.zeros((n_windows, n_labels), dtype=np.bool_)
        
        for i, window_label in enumerate(window_labels):
            for j, label_name in enumerate(label_names):
                if label_name in window_label.labels:
                    value = window_label.labels[label_name]
                    
                    # Convert to float
                    try:
                        label_array[i, j] = float(value)
                        mask_array[i, j] = True
                    except (ValueError, TypeError):
                        logger.debug(f"Could not convert {label_name}={value} to float for window {i}")
        
        if return_masks:
            return label_array, mask_array
        return label_array, None
    
    def aggregate_window_predictions(
        self,
        predictions: np.ndarray,
        case_ids: List[str],
        aggregation: str = "mean"
    ) -> Dict[str, float]:
        """Aggregate window-level predictions to case-level.
        
        Args:
            predictions: Array of shape [n_windows, n_classes or 1]
            case_ids: Case ID for each window
            aggregation: Aggregation method (mean, max, median, last)
            
        Returns:
            Dictionary mapping case_id -> aggregated prediction
        """
        case_preds = {}
        unique_cases = np.unique(case_ids)
        
        for case_id in unique_cases:
            case_mask = np.array(case_ids) == case_id
            case_predictions = predictions[case_mask]
            
            if aggregation == "mean":
                agg_pred = np.mean(case_predictions, axis=0)
            elif aggregation == "max":
                agg_pred = np.max(case_predictions, axis=0)
            elif aggregation == "median":
                agg_pred = np.median(case_predictions, axis=0)
            elif aggregation == "last":
                agg_pred = case_predictions[-1]
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation}")
            
            # Convert to scalar if single output
            if isinstance(agg_pred, np.ndarray):
                if agg_pred.size == 1:
                    agg_pred = float(agg_pred.item())
                elif agg_pred.ndim == 0:
                    agg_pred = float(agg_pred.item())
            elif isinstance(agg_pred, (np.floating, np.float64, np.float32)):
                agg_pred = float(agg_pred)
            
            case_preds[case_id] = agg_pred
        
        return case_preds
    
    def create_multi_task_labels(
        self,
        window_labels: List[WindowLabel],
        task_configs: Dict[str, List[str]]
    ) -> Dict[str, np.ndarray]:
        """Create label arrays for multi-task learning.
        
        Args:
            window_labels: List of WindowLabel objects
            task_configs: Dictionary mapping task_name -> list of label names
            
        Returns:
            Dictionary mapping task_name -> label array
        """
        task_arrays = {}
        
        for task_name, label_names in task_configs.items():
            label_array, _ = self.create_window_label_arrays(
                window_labels,
                label_names,
                return_masks=False
            )
            task_arrays[task_name] = label_array
        
        logger.info(f"Created {len(task_arrays)} task-specific label arrays")
        return task_arrays
    
    def save_aligned_labels(
        self,
        window_labels: List[WindowLabel],
        output_path: Union[str, Path],
        format: str = "npz"
    ) -> None:
        """Save aligned window labels to disk.
        
        Args:
            window_labels: List of WindowLabel objects
            output_path: Output file path
            format: File format (npz, json, parquet)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "npz":
            # Convert to arrays
            data = {
                'window_indices': np.array([w.window_idx for w in window_labels]),
                'case_ids': np.array([w.case_id for w in window_labels]),
                'start_times': np.array([w.start_time for w in window_labels]),
                'end_times': np.array([w.end_time for w in window_labels]),
            }
            
            # Extract unique labels
            all_label_names = set()
            for w in window_labels:
                all_label_names.update(w.labels.keys())
            
            # Create label arrays
            for label_name in sorted(all_label_names):
                label_values = []
                for w in window_labels:
                    value = w.labels.get(label_name, np.nan)
                    label_values.append(value)
                data[f"label_{label_name}"] = np.array(label_values)
            
            np.savez_compressed(output_path, **data)
            logger.info(f"✓ Saved {len(window_labels)} window labels to {output_path}")
            
        elif format == "json":
            import json
            
            # Convert window labels to JSON-serializable format
            def convert_to_serializable(obj):
                """Convert numpy types to Python native types."""
                if isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                elif isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj
            
            serializable_data = [convert_to_serializable(w.to_dict()) for w in window_labels]
            
            with open(output_path, 'w') as f:
                json.dump(serializable_data, f, indent=2)
            logger.info(f"✓ Saved window labels to {output_path}")
            
        elif format == "parquet":
            # Convert to DataFrame
            rows = []
            for w in window_labels:
                row = {
                    'window_idx': w.window_idx,
                    'case_id': w.case_id,
                    'start_time': w.start_time,
                    'end_time': w.end_time,
                    **w.labels
                }
                rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_parquet(output_path, index=False)
            logger.info(f"✓ Saved window labels to {output_path}")
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def load_aligned_labels(
        self,
        input_path: Union[str, Path],
        format: str = "npz"
    ) -> List[WindowLabel]:
        """Load aligned window labels from disk.
        
        Args:
            input_path: Input file path
            format: File format (npz, json, parquet)
            
        Returns:
            List of WindowLabel objects
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Label file not found: {input_path}")
        
        window_labels = []
        
        if format == "npz":
            data = np.load(input_path, allow_pickle=True)
            
            n_windows = len(data['window_indices'])
            
            # Extract label names (keys starting with "label_")
            label_names = [k.replace('label_', '') for k in data.keys() if k.startswith('label_')]
            
            for i in range(n_windows):
                labels = {}
                for label_name in label_names:
                    value = data[f'label_{label_name}'][i]
                    # Handle both numeric and string types
                    try:
                        if np.isscalar(value) and np.isnan(value):
                            continue
                    except (TypeError, ValueError):
                        # Not a numeric type, include it
                        pass
                    labels[label_name] = value
                
                window_label = WindowLabel(
                    window_idx=int(data['window_indices'][i]),
                    case_id=str(data['case_ids'][i]),
                    start_time=float(data['start_times'][i]),
                    end_time=float(data['end_times'][i]),
                    labels=labels
                )
                window_labels.append(window_label)
            
            logger.info(f"✓ Loaded {len(window_labels)} window labels from {input_path}")
            
        elif format == "json":
            import json
            with open(input_path, 'r') as f:
                data_list = json.load(f)
            
            for data in data_list:
                window_label = WindowLabel(**data)
                window_labels.append(window_label)
            
            logger.info(f"✓ Loaded {len(window_labels)} window labels from {input_path}")
            
        elif format == "parquet":
            df = pd.read_parquet(input_path)
            
            for _, row in df.iterrows():
                labels = {k: v for k, v in row.items() 
                         if k not in ['window_idx', 'case_id', 'start_time', 'end_time']}
                
                window_label = WindowLabel(
                    window_idx=int(row['window_idx']),
                    case_id=str(row['case_id']),
                    start_time=float(row['start_time']),
                    end_time=float(row['end_time']),
                    labels=labels
                )
                window_labels.append(window_label)
            
            logger.info(f"✓ Loaded {len(window_labels)} window labels from {input_path}")
        
        return window_labels
