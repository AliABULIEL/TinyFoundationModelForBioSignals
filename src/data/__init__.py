"""Data processing modules for VitalDB signals and BUT PPG."""

# Core utilities - these have no circular dependencies
try:
    from .detect import find_ecg_rpeaks, find_ppg_peaks
    from .filters import filter_ppg, filter_ecg, design_ppg_filter, design_ecg_filter
    from .quality import compute_sqi, ecg_sqi, ppg_ssqi
    from .sync import resample_to_fs, align_streams
    from .windows import (
        make_windows, 
        validate_cardiac_cycles,
        compute_normalization_stats,
        normalize_windows,
        NormalizationStats
    )
    from .splits import (
        make_patient_level_splits,
        verify_no_subject_leakage,
        save_splits,
        load_splits
    )
    from .vitaldb_loader import list_cases, load_channel
except ImportError as e:
    import warnings
    warnings.warn(f"Some core utilities could not be imported: {e}")

# Dataset classes - import these DIRECTLY to avoid circular imports
# Example: from src.data.vitaldb_dataset import VitalDBDataset
# DO NOT: from src.data import VitalDBDataset (may cause circular import)

__all__ = [
    'find_ecg_rpeaks', 'find_ppg_peaks',
    'filter_ppg', 'filter_ecg',
    'compute_sqi', 'ecg_sqi', 'ppg_ssqi',
    'resample_to_fs', 'align_streams',
    'make_windows', 'validate_cardiac_cycles',
    'list_cases', 'load_channel',
]

# Note: Import datasets directly:
# from src.data.vitaldb_dataset import VitalDBDataset
# from src.data.butppg_dataset import BUTPPGDataset
# from src.data.butppg_loader import BUTPPGLoader
