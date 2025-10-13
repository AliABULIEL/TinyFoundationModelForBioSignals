"""Data processing modules for VitalDB signals and BUT PPG."""

from .detect import find_ecg_rpeaks, find_ppg_peaks
from .filters import (
    design_abp_filter,
    design_ecg_filter,
    design_eeg_filter,
    design_ppg_filter,
    filter_abp,
    filter_ecg,
    filter_eeg,
    filter_ppg,
    freqz_response,
)
from .quality import (
    ecg_sqi,
    hard_artifacts,
    ppg_abp_corr,
    ppg_ssqi,
    template_corr,
    window_accept,
)
from .splits import (
    create_cv_folds,
    create_leave_one_subject_out_splits,
    create_temporal_splits,
    get_split_statistics,
    load_splits,
    make_patient_level_splits,
    save_splits,
    stratified_patient_split,
    verify_no_subject_leakage,
)
from .sync import align_streams, resample_to_fs
from .vitaldb_loader import list_cases, load_channel
from .windows import (
    NormalizationStats,
    aggregate_window_predictions,
    compute_normalization_stats,
    create_sliding_windows,
    make_windows,
    normalize_windows,
    validate_cardiac_cycles,
)

# BUT PPG imports
from .butppg_loader import BUTPPGLoader, find_butppg_cases, load_butppg_signal
from .butppg_dataset import BUTPPGDataset, create_butppg_dataloaders
from .dataset_compatibility import DatasetCompatibilityValidator, validate_datasets

# VitalDB SSL imports
from .vitaldb_dataset import VitalDBDataset, create_vitaldb_dataloaders
from .manifests import build_manifest, verify_manifest_integrity, extract_subject_id, hash_subject_to_split

__all__ = [
    # VitalDB Loader
    'list_cases',
    'load_channel',
    # Sync
    'resample_to_fs',
    'align_streams',
    # Filters
    'design_ppg_filter',
    'design_ecg_filter',
    'design_abp_filter',
    'design_eeg_filter',
    'filter_ppg',
    'filter_ecg',
    'filter_abp',
    'filter_eeg',
    'freqz_response',
    # Detect
    'find_ecg_rpeaks',
    'find_ppg_peaks',
    # Quality
    'ecg_sqi',
    'template_corr',
    'ppg_ssqi',
    'ppg_abp_corr',
    'hard_artifacts',
    'window_accept',
    # Windows
    'make_windows',
    'compute_normalization_stats',
    'normalize_windows',
    'validate_cardiac_cycles',
    'NormalizationStats',
    'create_sliding_windows',
    'aggregate_window_predictions',
    # Splits
    'make_patient_level_splits',
    'verify_no_subject_leakage',
    'stratified_patient_split',
    'create_cv_folds',
    'save_splits',
    'load_splits',
    'get_split_statistics',
    'create_temporal_splits',
    'create_leave_one_subject_out_splits',
    # BUT PPG - NEW
    'BUTPPGLoader',
    'BUTPPGDataset',
    'create_butppg_dataloaders',
    'find_butppg_cases',
    'load_butppg_signal',
    'DatasetCompatibilityValidator',
    'validate_datasets',
    # VitalDB SSL - NEW
    'VitalDBDataset',
    'create_vitaldb_dataloaders',
    'build_manifest',
    'verify_manifest_integrity',
    'extract_subject_id',
    'hash_subject_to_split',
]
