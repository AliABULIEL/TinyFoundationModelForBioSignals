"""Clinical Label Extraction Module.

Handles extraction and preprocessing of clinical labels from VitalDB and other sources.
Supports both case-level and temporal labels with proper validation.

Author: Senior Data Engineering Team
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LabelType(Enum):
    """Types of clinical labels."""
    BINARY = "binary"
    CATEGORICAL = "categorical"
    CONTINUOUS = "continuous"
    TEMPORAL = "temporal"


@dataclass
class LabelConfig:
    """Configuration for a clinical label."""
    name: str
    column_name: str
    label_type: LabelType
    description: str = ""
    missing_value_strategy: str = "drop"  # drop, fill, mean, median
    fill_value: Optional[Any] = None
    valid_range: Optional[Tuple[float, float]] = None
    categories: Optional[List[Any]] = None
    
    def validate(self) -> bool:
        """Validate label configuration."""
        if self.label_type == LabelType.CATEGORICAL and not self.categories:
            logger.warning(f"Categorical label {self.name} has no categories defined")
            return False
        if self.valid_range and len(self.valid_range) != 2:
            logger.error(f"Invalid range for {self.name}: {self.valid_range}")
            return False
        return True


@dataclass
class ClinicalLabels:
    """Container for clinical labels with metadata."""
    case_id: str
    labels: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    temporal_labels: Optional[pd.DataFrame] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'case_id': self.case_id,
            'labels': self.labels,
            'metadata': self.metadata
        }


class ClinicalLabelExtractor:
    """Extracts and preprocesses clinical labels from various sources.
    
    Features:
    - Case-level label extraction from CSV
    - Temporal label extraction from time-series data
    - Missing value handling
    - Label validation and quality checks
    - Multiple imputation strategies
    
    Example:
        >>> extractor = ClinicalLabelExtractor(
        ...     metadata_path="data/cache/vitaldb_cases.csv"
        ... )
        >>> extractor.add_label_config(
        ...     name="mortality",
        ...     column_name="death_inhosp",
        ...     label_type=LabelType.BINARY
        ... )
        >>> labels = extractor.extract_case_labels(case_id="1")
    """
    
    # Standard VitalDB label configurations
    VITALDB_STANDARD_LABELS = {
        'mortality': LabelConfig(
            name='mortality',
            column_name='death_inhosp',
            label_type=LabelType.BINARY,
            description='In-hospital mortality',
            missing_value_strategy='fill',
            fill_value=0
        ),
        'icu_los': LabelConfig(
            name='icu_los',
            column_name='icu_days',
            label_type=LabelType.CONTINUOUS,
            description='ICU length of stay in days',
            missing_value_strategy='fill',
            fill_value=0,
            valid_range=(0, 365)
        ),
        'age': LabelConfig(
            name='age',
            column_name='age',
            label_type=LabelType.CONTINUOUS,
            description='Patient age in years',
            missing_value_strategy='mean',
            valid_range=(0, 120)
        ),
        'sex': LabelConfig(
            name='sex',
            column_name='sex',
            label_type=LabelType.CATEGORICAL,
            description='Patient sex',
            categories=['M', 'F'],
            missing_value_strategy='drop'
        ),
        'bmi': LabelConfig(
            name='bmi',
            column_name='bmi',
            label_type=LabelType.CONTINUOUS,
            description='Body Mass Index',
            missing_value_strategy='mean',
            valid_range=(10, 60)
        ),
        'asa': LabelConfig(
            name='asa',
            column_name='asa',
            label_type=LabelType.CATEGORICAL,
            description='ASA Physical Status Classification',
            categories=[1, 2, 3, 4, 5],
            missing_value_strategy='median'
        ),
        'emergency_op': LabelConfig(
            name='emergency_op',
            column_name='emop',
            label_type=LabelType.BINARY,
            description='Emergency operation',
            missing_value_strategy='fill',
            fill_value=0
        )
    }
    
    def __init__(
        self,
        metadata_path: Optional[Union[str, Path]] = None,
        case_id_column: str = 'caseid',
        auto_load_standard: bool = True
    ):
        """Initialize clinical label extractor.
        
        Args:
            metadata_path: Path to clinical metadata CSV
            case_id_column: Column name for case IDs
            auto_load_standard: Automatically load standard VitalDB labels
        """
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.case_id_column = case_id_column
        self.metadata_df: Optional[pd.DataFrame] = None
        self.label_configs: Dict[str, LabelConfig] = {}
        
        # Load metadata if provided
        if self.metadata_path and self.metadata_path.exists():
            self._load_metadata()
        
        # Load standard labels
        if auto_load_standard:
            self._load_standard_labels()
        
        logger.info(f"ClinicalLabelExtractor initialized with {len(self.label_configs)} labels")
    
    def _load_metadata(self) -> None:
        """Load metadata CSV file."""
        try:
            self.metadata_df = pd.read_csv(self.metadata_path)
            logger.info(f"✓ Loaded metadata: {len(self.metadata_df)} cases, {len(self.metadata_df.columns)} columns")
            
            # Validate case ID column
            if self.case_id_column not in self.metadata_df.columns:
                raise ValueError(f"Case ID column '{self.case_id_column}' not found in metadata")
            
            # Convert case IDs to strings for consistency
            self.metadata_df[self.case_id_column] = self.metadata_df[self.case_id_column].astype(str)
            
        except Exception as e:
            logger.error(f"Failed to load metadata from {self.metadata_path}: {e}")
            raise
    
    def _load_standard_labels(self) -> None:
        """Load standard VitalDB label configurations."""
        for name, config in self.VITALDB_STANDARD_LABELS.items():
            if config.validate():
                self.label_configs[name] = config
        logger.info(f"Loaded {len(self.label_configs)} standard label configurations")
    
    def add_label_config(
        self,
        name: str,
        column_name: str,
        label_type: Union[LabelType, str],
        description: str = "",
        **kwargs
    ) -> None:
        """Add a custom label configuration.
        
        Args:
            name: Label identifier
            column_name: Column name in metadata
            label_type: Type of label (binary, categorical, continuous, temporal)
            description: Label description
            **kwargs: Additional configuration parameters
        """
        if isinstance(label_type, str):
            label_type = LabelType(label_type)
        
        config = LabelConfig(
            name=name,
            column_name=column_name,
            label_type=label_type,
            description=description,
            **kwargs
        )
        
        if config.validate():
            self.label_configs[name] = config
            logger.info(f"Added label config: {name} ({label_type.value})")
        else:
            logger.error(f"Invalid label configuration for {name}")
    
    def extract_case_labels(
        self,
        case_id: Union[str, int],
        label_names: Optional[List[str]] = None,
        return_metadata: bool = True
    ) -> Optional[ClinicalLabels]:
        """Extract labels for a specific case.
        
        Args:
            case_id: Case identifier
            label_names: Specific labels to extract (None = all)
            return_metadata: Include additional metadata
            
        Returns:
            ClinicalLabels object or None if case not found
        """
        if self.metadata_df is None:
            logger.error("No metadata loaded. Call load_metadata() first.")
            return None
        
        case_id = str(case_id)
        case_data = self.metadata_df[self.metadata_df[self.case_id_column] == case_id]
        
        if case_data.empty:
            logger.warning(f"Case {case_id} not found in metadata")
            return None
        
        # Use first row if multiple matches
        case_row = case_data.iloc[0]
        
        # Determine which labels to extract
        if label_names is None:
            label_names = list(self.label_configs.keys())
        
        # Extract labels
        labels = {}
        metadata = {'case_id': case_id}
        
        for label_name in label_names:
            if label_name not in self.label_configs:
                logger.warning(f"Label config not found: {label_name}")
                continue
            
            config = self.label_configs[label_name]
            
            # Check if column exists
            if config.column_name not in case_row.index:
                logger.debug(f"Column {config.column_name} not found for case {case_id}")
                continue
            
            # Extract and process value
            value = case_row[config.column_name]
            processed_value = self._process_value(value, config)
            
            if processed_value is not None:
                labels[label_name] = processed_value
            else:
                logger.debug(f"Label {label_name} is None for case {case_id} after processing")
        
        # Add metadata if requested
        if return_metadata:
            metadata.update({
                'n_labels': len(labels),
                'available_labels': list(labels.keys())
            })
        
        return ClinicalLabels(
            case_id=case_id,
            labels=labels,
            metadata=metadata
        )
    
    def _process_value(self, value: Any, config: LabelConfig) -> Optional[Any]:
        """Process a label value according to its configuration.
        
        Args:
            value: Raw value from metadata
            config: Label configuration
            
        Returns:
            Processed value or None
        """
        # Handle missing values
        if pd.isna(value):
            return self._handle_missing(value, config)
        
        # Validate range for continuous values
        if config.label_type == LabelType.CONTINUOUS and config.valid_range:
            min_val, max_val = config.valid_range
            if not (min_val <= float(value) <= max_val):
                logger.warning(f"Value {value} outside valid range {config.valid_range} for {config.name}")
                return None
        
        # Validate categories
        if config.label_type == LabelType.CATEGORICAL and config.categories:
            if value not in config.categories:
                logger.warning(f"Value {value} not in valid categories for {config.name}")
                return None
        
        # Type conversion
        if config.label_type == LabelType.BINARY:
            return int(value)
        elif config.label_type == LabelType.CONTINUOUS:
            return float(value)
        elif config.label_type == LabelType.CATEGORICAL:
            return value
        
        return value
    
    def _handle_missing(self, value: Any, config: LabelConfig) -> Optional[Any]:
        """Handle missing values according to strategy.
        
        Args:
            value: Missing value
            config: Label configuration
            
        Returns:
            Imputed value or None
        """
        strategy = config.missing_value_strategy
        
        if strategy == 'drop':
            return None
        elif strategy == 'fill' and config.fill_value is not None:
            return config.fill_value
        elif strategy in ['mean', 'median'] and self.metadata_df is not None:
            # Compute from metadata
            column_data = self.metadata_df[config.column_name].dropna()
            if len(column_data) > 0:
                if strategy == 'mean':
                    return float(column_data.mean())
                else:  # median
                    return float(column_data.median())
        
        return None
    
    def extract_batch_labels(
        self,
        case_ids: List[Union[str, int]],
        label_names: Optional[List[str]] = None
    ) -> Dict[str, ClinicalLabels]:
        """Extract labels for multiple cases efficiently.
        
        Args:
            case_ids: List of case identifiers
            label_names: Specific labels to extract
            
        Returns:
            Dictionary mapping case_id -> ClinicalLabels
        """
        results = {}
        
        for case_id in case_ids:
            labels = self.extract_case_labels(case_id, label_names)
            if labels:
                results[str(case_id)] = labels
        
        logger.info(f"Extracted labels for {len(results)}/{len(case_ids)} cases")
        return results
    
    def get_label_statistics(self, label_name: str) -> Dict[str, Any]:
        """Compute statistics for a specific label.
        
        Args:
            label_name: Name of label
            
        Returns:
            Dictionary with statistics
        """
        if self.metadata_df is None or label_name not in self.label_configs:
            return {}
        
        config = self.label_configs[label_name]
        column_data = self.metadata_df[config.column_name]
        
        stats = {
            'name': label_name,
            'type': config.label_type.value,
            'n_total': len(column_data),
            'n_missing': column_data.isna().sum(),
            'missing_rate': column_data.isna().mean()
        }
        
        # Type-specific statistics
        if config.label_type == LabelType.CONTINUOUS:
            valid_data = column_data.dropna()
            stats.update({
                'mean': float(valid_data.mean()) if len(valid_data) > 0 else None,
                'std': float(valid_data.std()) if len(valid_data) > 0 else None,
                'min': float(valid_data.min()) if len(valid_data) > 0 else None,
                'max': float(valid_data.max()) if len(valid_data) > 0 else None,
                'median': float(valid_data.median()) if len(valid_data) > 0 else None
            })
        elif config.label_type in [LabelType.BINARY, LabelType.CATEGORICAL]:
            value_counts = column_data.value_counts()
            stats['value_counts'] = value_counts.to_dict()
            stats['unique_values'] = len(value_counts)
        
        return stats
    
    def generate_label_report(self) -> pd.DataFrame:
        """Generate comprehensive report of all labels.
        
        Returns:
            DataFrame with label statistics
        """
        report_data = []
        
        for label_name in self.label_configs.keys():
            stats = self.get_label_statistics(label_name)
            report_data.append(stats)
        
        return pd.DataFrame(report_data)
    
    def validate_labels(self, case_ids: List[Union[str, int]]) -> Dict[str, Any]:
        """Validate labels for a set of cases.
        
        Args:
            case_ids: List of case identifiers
            
        Returns:
            Validation report
        """
        validation_results = {
            'n_cases': len(case_ids),
            'n_valid': 0,
            'n_invalid': 0,
            'missing_cases': [],
            'incomplete_labels': {}
        }
        
        for case_id in case_ids:
            labels = self.extract_case_labels(case_id)
            
            if labels is None:
                validation_results['missing_cases'].append(str(case_id))
                validation_results['n_invalid'] += 1
            else:
                validation_results['n_valid'] += 1
                
                # Check for incomplete labels
                expected_labels = set(self.label_configs.keys())
                actual_labels = set(labels.labels.keys())
                missing_labels = expected_labels - actual_labels
                
                if missing_labels:
                    validation_results['incomplete_labels'][str(case_id)] = list(missing_labels)
        
        logger.info(f"Validation: {validation_results['n_valid']}/{validation_results['n_cases']} valid cases")
        return validation_results
