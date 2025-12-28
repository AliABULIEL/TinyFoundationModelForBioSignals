#!/usr/bin/env python3
"""
BUT-PPG Clinical Data Processor

Processes raw BUT-PPG data and extracts clinical labels for:
1. Signal Quality (binary classification)
2. Heart Rate (regression)
3. Motion Type (8-class classification)

Usage:
    python scripts/process_butppg_clinical.py \
        --raw-dir data/but_ppg/raw/but-ppg-an-annotated-photoplethysmography-dataset-2.0.0 \
        --annotations-dir data/but_ppg/raw/annotations \
        --output-dir data/processed/butppg \
        --target-fs 125 \
        --window-size 1024
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, sosfilt, resample_poly
from tqdm import tqdm
from typing import Dict, Tuple, List, Optional


class BUTPPGProcessor:
    """Process BUT-PPG raw data into task-specific datasets"""

    # Channel order for 5-channel input
    CHANNEL_ORDER = ['ACC_X', 'ACC_Y', 'ACC_Z', 'PPG', 'ECG']

    # Motion classes (from BUT-PPG subject-info.csv)
    MOTION_CLASSES = {
        'sitting': 0,
        'standing': 1,
        'walking_slow': 2,
        'walking': 3,  # Normal walking
        'walking_fast': 4,
        'running': 5,
        'cycling': 6,
        'hand_movement': 7,
        'hand': 7,  # Alias
        'sit': 0,  # Aliases
        'stand': 1
    }

    def __init__(
        self,
        raw_dir: Path,
        annotations_dir: Path,
        target_fs: int = 125,
        window_size: int = 1024
    ):
        self.raw_dir = Path(raw_dir)
        self.annotations_dir = Path(annotations_dir)
        self.target_fs = target_fs
        self.window_size = window_size

        # Load annotations
        print("\n" + "="*80)
        print("LOADING ANNOTATIONS")
        print("="*80)
        self.quality_hr_df = self._load_quality_hr_annotations()
        self.subject_info_df = self._load_subject_info()

    def _load_quality_hr_annotations(self) -> pd.DataFrame:
        """Load quality and HR annotations from quality-hr-ann.csv"""
        ann_file = self.annotations_dir / 'quality-hr-ann.csv'

        if not ann_file.exists():
            print(f"❌ Quality/HR annotations not found: {ann_file}")
            return pd.DataFrame()

        df = pd.read_csv(ann_file)
        print(f"✓ Loaded quality-hr-ann.csv: {len(df)} entries")
        print(f"  Columns: {list(df.columns)}")

        # Preview first few rows
        print(f"  Sample data:")
        print(df.head(3).to_string(index=False))

        return df

    def _load_subject_info(self) -> pd.DataFrame:
        """Load subject information (including motion) from subject-info.csv"""
        ann_file = self.annotations_dir / 'subject-info.csv'

        if not ann_file.exists():
            print(f"❌ Subject info not found: {ann_file}")
            return pd.DataFrame()

        df = pd.read_csv(ann_file)
        print(f"\n✓ Loaded subject-info.csv: {len(df)} entries")
        print(f"  Columns: {list(df.columns)}")

        # Preview first few rows
        print(f"  Sample data:")
        print(df.head(3).to_string(index=False))

        return df

    def bandpass_filter(
        self,
        signal: np.ndarray,
        lowcut: float,
        highcut: float,
        fs: int
    ) -> np.ndarray:
        """Apply bandpass filter"""
        sos = butter(4, [lowcut, highcut], btype='band', fs=fs, output='sos')
        filtered = sosfilt(sos, signal)
        return filtered

    def preprocess_signal(
        self,
        signal: np.ndarray,
        original_fs: int,
        signal_type: str
    ) -> np.ndarray:
        """
        Preprocess a single signal

        Args:
            signal: Raw signal
            original_fs: Original sampling frequency
            signal_type: 'PPG', 'ECG', or 'ACC'

        Returns:
            Preprocessed signal at target_fs with length window_size
        """
        # Apply appropriate bandpass filter
        if signal_type == 'PPG':
            filtered = self.bandpass_filter(signal, 0.5, 8.0, original_fs)
        elif signal_type == 'ECG':
            filtered = self.bandpass_filter(signal, 0.5, 40.0, original_fs)
        elif signal_type == 'ACC':
            filtered = self.bandpass_filter(signal, 0.5, 20.0, original_fs)
        else:
            filtered = signal

        # Resample to target frequency
        if original_fs != self.target_fs:
            resampled = resample_poly(
                filtered,
                up=self.target_fs,
                down=original_fs
            )
        else:
            resampled = filtered

        # Ensure correct length (window_size)
        if len(resampled) > self.window_size:
            resampled = resampled[:self.window_size]
        elif len(resampled) < self.window_size:
            # Zero-pad if too short
            resampled = np.pad(
                resampled,
                (0, self.window_size - len(resampled)),
                mode='constant'
            )

        # Z-score normalization
        normalized = (resampled - resampled.mean()) / (resampled.std() + 1e-8)

        return normalized

    def load_recording(self, record_id: str) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """
        Load and preprocess a single BUT-PPG recording

        Args:
            record_id: 6-digit record ID (e.g., "100001")

        Returns:
            signals: [5, 1024] array (ACC_X, ACC_Y, ACC_Z, PPG, ECG)
            metadata: Dict with sampling rates and record info
        """
        try:
            # BUT-PPG has files in subdirectories:
            # data/but_ppg/dataset/100001/100001_PPG.dat

            signals_5ch = np.zeros((5, self.window_size))
            metadata = {'record_id': record_id, 'fs': {}}

            # Record directory
            record_dir = self.raw_dir / record_id

            # Load PPG signal
            ppg_path = record_dir / f"{record_id}_PPG"
            if ppg_path.with_suffix('.hea').exists():
                record_ppg = wfdb.rdrecord(str(ppg_path))
                ppg_signal = record_ppg.p_signal[:, 0]  # Single channel
                ppg_fs = record_ppg.fs
                signals_5ch[3] = self.preprocess_signal(ppg_signal, ppg_fs, 'PPG')
                metadata['fs']['PPG'] = ppg_fs

            # Load ECG signal
            ecg_path = record_dir / f"{record_id}_ECG"
            if ecg_path.with_suffix('.hea').exists():
                record_ecg = wfdb.rdrecord(str(ecg_path))
                ecg_signal = record_ecg.p_signal[:, 0]  # Single channel
                ecg_fs = record_ecg.fs
                signals_5ch[4] = self.preprocess_signal(ecg_signal, ecg_fs, 'ECG')
                metadata['fs']['ECG'] = ecg_fs

            # Load ACC signal (3-axis)
            acc_path = record_dir / f"{record_id}_ACC"
            if acc_path.with_suffix('.hea').exists():
                record_acc = wfdb.rdrecord(str(acc_path))
                acc_fs = record_acc.fs

                # ACC has 3 channels (X, Y, Z)
                if record_acc.p_signal.shape[1] >= 3:
                    for i in range(3):
                        acc_signal = record_acc.p_signal[:, i]
                        signals_5ch[i] = self.preprocess_signal(acc_signal, acc_fs, 'ACC')

                metadata['fs']['ACC'] = acc_fs

            return signals_5ch, metadata

        except Exception as e:
            print(f"  ⚠️  Error loading {record_id}: {e}")
            return None, None

    def get_quality_label(self, record_id: str) -> Optional[int]:
        """Get quality label for a recording (binary: 0=poor, 1=good)"""
        if self.quality_hr_df.empty:
            return None

        # quality-hr-ann.csv format (need to check actual column names)
        # Typically: "id" or "record_id", "quality" or "Quality"

        # Try different possible column names
        id_col = None
        quality_col = None

        for col in self.quality_hr_df.columns:
            if col.lower() in ['id', 'record_id', 'signal_id', 'signalid']:
                id_col = col
            if col.lower() in ['quality', 'ppg_quality', 'ppgquality']:
                quality_col = col

        if id_col is None or quality_col is None:
            return None

        # Match record
        row = self.quality_hr_df[self.quality_hr_df[id_col].astype(str) == str(record_id)]

        if row.empty:
            return None

        quality = row[quality_col].iloc[0]

        # Convert to binary (1=good, 0=poor)
        if isinstance(quality, str):
            return 1 if quality.lower() in ['good', 'high', '1', 'yes'] else 0
        else:
            return int(quality)

    def get_hr_label(self, record_id: str) -> Optional[float]:
        """Get heart rate label for a recording (BPM)"""
        if self.quality_hr_df.empty:
            return None

        # Try different possible column names
        id_col = None
        hr_col = None

        for col in self.quality_hr_df.columns:
            if col.lower() in ['id', 'record_id', 'signal_id', 'signalid']:
                id_col = col
            if col.lower() in ['hr', 'heart_rate', 'heartrate', 'reference_hr']:
                hr_col = col

        if id_col is None or hr_col is None:
            return None

        # Match record
        row = self.quality_hr_df[self.quality_hr_df[id_col].astype(str) == str(record_id)]

        if row.empty:
            return None

        hr = row[hr_col].iloc[0]
        return float(hr)

    def get_motion_label(self, record_id: str) -> Optional[int]:
        """Get motion label for a recording (8-class)"""
        if self.subject_info_df.empty:
            return None

        # Try different possible column names
        id_col = None
        motion_col = None

        for col in self.subject_info_df.columns:
            if col.lower() in ['id', 'record_id', 'signal_id', 'signalid']:
                id_col = col
            if col.lower() in ['motion', 'motion_type', 'motiontype', 'activity']:
                motion_col = col

        if id_col is None or motion_col is None:
            return None

        # Match record
        row = self.subject_info_df[self.subject_info_df[id_col].astype(str) == str(record_id)]

        if row.empty:
            return None

        motion_str = str(row[motion_col].iloc[0]).lower().strip()

        # Map motion string to class index
        return self.MOTION_CLASSES.get(motion_str, None)

    def get_bp_label(self, record_id: str) -> Optional[Tuple[float, float]]:
        """
        Get blood pressure label for a recording

        Returns:
            Tuple of (systolic, diastolic) in mmHg, or None if not available
        """
        if self.subject_info_df.empty:
            return None

        # Try different possible column names
        id_col = None
        bp_col = None

        for col in self.subject_info_df.columns:
            if col.lower() in ['id', 'record_id', 'signal_id', 'signalid']:
                id_col = col
            if 'blood' in col.lower() and 'pressure' in col.lower():
                bp_col = col

        if id_col is None or bp_col is None:
            return None

        # Match record
        row = self.subject_info_df[self.subject_info_df[id_col].astype(str) == str(record_id)]

        if row.empty:
            return None

        bp_value = row[bp_col].iloc[0]

        # Handle NaN or empty values
        if pd.isna(bp_value) or str(bp_value).strip() == '':
            return None

        # Parse BP format (typically "120/80" or similar)
        bp_str = str(bp_value).strip()
        if '/' in bp_str:
            try:
                parts = bp_str.split('/')
                systolic = float(parts[0].strip())
                diastolic = float(parts[1].strip())
                return (systolic, diastolic)
            except (ValueError, IndexError):
                return None

        # If single value, assume systolic only
        try:
            return (float(bp_str), None)
        except ValueError:
            return None

    def get_spo2_label(self, record_id: str) -> Optional[float]:
        """
        Get SpO2 (oxygen saturation) label for a recording

        Returns:
            SpO2 percentage (0-100), or None if not available
        """
        if self.subject_info_df.empty:
            return None

        # Try different possible column names
        id_col = None
        spo2_col = None

        for col in self.subject_info_df.columns:
            if col.lower() in ['id', 'record_id', 'signal_id', 'signalid']:
                id_col = col
            if 'spo2' in col.lower() or ('oxygen' in col.lower() and 'sat' in col.lower()):
                spo2_col = col

        if id_col is None or spo2_col is None:
            return None

        # Match record
        row = self.subject_info_df[self.subject_info_df[id_col].astype(str) == str(record_id)]

        if row.empty:
            return None

        spo2_value = row[spo2_col].iloc[0]

        # Handle NaN or empty values
        if pd.isna(spo2_value) or str(spo2_value).strip() == '':
            return None

        try:
            spo2 = float(spo2_value)
            # Sanity check: SpO2 should be 0-100%
            if 0 <= spo2 <= 100:
                return spo2
        except ValueError:
            pass

        return None

    def get_glycaemia_label(self, record_id: str) -> Optional[float]:
        """
        Get glycaemia (blood glucose) label for a recording

        Returns:
            Blood glucose in mmol/l, or None if not available
        """
        if self.subject_info_df.empty:
            return None

        # Try different possible column names
        id_col = None
        glyc_col = None

        for col in self.subject_info_df.columns:
            if col.lower() in ['id', 'record_id', 'signal_id', 'signalid']:
                id_col = col
            if 'glyc' in col.lower() or ('glucose' in col.lower() or 'sugar' in col.lower()):
                glyc_col = col

        if id_col is None or glyc_col is None:
            return None

        # Match record
        row = self.subject_info_df[self.subject_info_df[id_col].astype(str) == str(record_id)]

        if row.empty:
            return None

        glyc_value = row[glyc_col].iloc[0]

        # Handle NaN or empty values
        if pd.isna(glyc_value) or str(glyc_value).strip() == '':
            return None

        try:
            glyc = float(glyc_value)
            # Sanity check: typical range 3-30 mmol/l
            if 0 <= glyc <= 50:
                return glyc
        except ValueError:
            pass

        return None

    def create_task_dataset(
        self,
        task: str,
        split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Create dataset for a specific task

        Args:
            task: 'quality', 'heart_rate', or 'motion'
            split_ratio: (train, val, test) ratios

        Returns:
            Dict with 'train', 'val', 'test' splits
            Each split: (signals [N, 5, 1024], labels [N])
        """
        print(f"\n{'='*80}")
        print(f"📊 Creating dataset for task: {task.upper()}")
        print(f"{'='*80}")

        # Collect all record IDs
        ppg_files = sorted(self.raw_dir.glob("*_PPG.hea"))
        record_ids = [f.stem.replace('_PPG', '') for f in ppg_files]

        print(f"Total recordings found: {len(record_ids)}")

        # Collect signals and labels
        all_signals = []
        all_labels = []
        all_subjects = []  # First 3 digits of record ID

        skipped = 0

        for record_id in tqdm(record_ids, desc=f"Processing {task}"):
            # Load signal
            signals, metadata = self.load_recording(record_id)

            if signals is None:
                skipped += 1
                continue

            # Get label based on task
            if task == 'quality':
                label = self.get_quality_label(record_id)
            elif task == 'heart_rate':
                label = self.get_hr_label(record_id)
            elif task == 'motion':
                label = self.get_motion_label(record_id)
            else:
                raise ValueError(f"Unknown task: {task}")

            if label is None:
                skipped += 1
                continue

            # Subject ID (first 3 digits)
            subject_id = int(record_id[:3])

            all_signals.append(signals)
            all_labels.append(label)
            all_subjects.append(subject_id)

        if len(all_signals) == 0:
            print(f"  ❌ No valid data collected for {task}")
            print(f"     Skipped {skipped} recordings")
            return {}

        # Convert to numpy arrays
        signals_array = np.array(all_signals)  # [N, 5, 1024]
        labels_array = np.array(all_labels)    # [N]
        subjects_array = np.array(all_subjects)  # [N]

        print(f"\n  ✓ Collected {len(signals_array)} samples")
        print(f"    Skipped: {skipped} recordings")
        print(f"    Signals shape: {signals_array.shape}")
        print(f"    Labels shape: {labels_array.shape}")

        if task == 'quality':
            unique, counts = np.unique(labels_array, return_counts=True)
            print(f"    Quality distribution: {dict(zip(unique, counts))}")
        elif task == 'motion':
            unique, counts = np.unique(labels_array, return_counts=True)
            print(f"    Motion classes: {dict(zip(unique, counts))}")

        # Subject-level split (CRITICAL for medical ML)
        unique_subjects = np.unique(subjects_array)
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(unique_subjects)

        n_train = int(len(unique_subjects) * split_ratio[0])
        n_val = int(len(unique_subjects) * split_ratio[1])

        train_subjects = unique_subjects[:n_train]
        val_subjects = unique_subjects[n_train:n_train+n_val]
        test_subjects = unique_subjects[n_train+n_val:]

        print(f"\n  Subject-level split:")
        print(f"    Train subjects: {len(train_subjects)}")
        print(f"    Val subjects: {len(val_subjects)}")
        print(f"    Test subjects: {len(test_subjects)}")

        # Create splits
        train_mask = np.isin(subjects_array, train_subjects)
        val_mask = np.isin(subjects_array, val_subjects)
        test_mask = np.isin(subjects_array, test_subjects)

        splits = {
            'train': (signals_array[train_mask], labels_array[train_mask]),
            'val': (signals_array[val_mask], labels_array[val_mask]),
            'test': (signals_array[test_mask], labels_array[test_mask])
        }

        print(f"\n  ✓ Split: Train={train_mask.sum()}, Val={val_mask.sum()}, Test={test_mask.sum()}")

        return splits

    def save_task_dataset(
        self,
        splits: Dict,
        output_dir: Path,
        task: str
    ):
        """Save task dataset to disk"""
        task_dir = Path(output_dir) / task
        task_dir.mkdir(parents=True, exist_ok=True)

        for split_name, (signals, labels) in splits.items():
            output_file = task_dir / f'{split_name}.npz'

            np.savez_compressed(
                output_file,
                signals=signals,
                labels=labels
            )

            print(f"  ✓ Saved {split_name}: {output_file}")
            print(f"    Shape: {signals.shape}, Labels: {labels.shape}")

    def create_multitask_dataset(
        self,
        split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    ) -> Dict[str, Dict]:
        """
        Create unified multi-task dataset with all available labels.

        Args:
            split_ratio: (train, val, test) ratios

        Returns:
            Dict with 'train', 'val', 'test' splits
            Each split contains:
            - signals: [N, 5, 1024] arrays
            - labels: Dict with all available labels for each sample
        """
        print(f"\n{'='*80}")
        print(f"📊 Creating MULTI-TASK dataset with all labels")
        print(f"{'='*80}")

        # Collect all record IDs (need to look in subdirectories)
        ppg_files = sorted(self.raw_dir.glob("*/*_PPG.hea"))
        record_ids = [f.parent.name for f in ppg_files]

        print(f"Total recordings found: {len(record_ids)}")

        # Collect signals and all labels
        all_signals = []
        all_labels = {
            'quality': [],
            'hr': [],
            'motion': [],
            'bp_systolic': [],
            'bp_diastolic': [],
            'spo2': [],
            'glycaemia': []
        }
        all_subjects = []

        skipped = 0
        label_availability = {key: 0 for key in all_labels.keys()}

        for record_id in tqdm(record_ids, desc="Processing multi-task"):
            # Load signal
            signals, metadata = self.load_recording(record_id)

            if signals is None:
                skipped += 1
                continue

            # Get all labels
            quality = self.get_quality_label(record_id)
            hr = self.get_hr_label(record_id)
            motion = self.get_motion_label(record_id)
            bp = self.get_bp_label(record_id)
            spo2 = self.get_spo2_label(record_id)
            glycaemia = self.get_glycaemia_label(record_id)

            # Subject ID (first 3 digits)
            subject_id = int(record_id[:3])

            # Store signals
            all_signals.append(signals)
            all_subjects.append(subject_id)

            # Store labels (use -1 for missing)
            all_labels['quality'].append(quality if quality is not None else -1)
            all_labels['hr'].append(hr if hr is not None else -1)
            all_labels['motion'].append(motion if motion is not None else -1)

            # Blood pressure
            if bp is not None:
                all_labels['bp_systolic'].append(bp[0])
                all_labels['bp_diastolic'].append(bp[1] if bp[1] is not None else -1)
                label_availability['bp_systolic'] += 1
                if bp[1] is not None:
                    label_availability['bp_diastolic'] += 1
            else:
                all_labels['bp_systolic'].append(-1)
                all_labels['bp_diastolic'].append(-1)

            all_labels['spo2'].append(spo2 if spo2 is not None else -1)
            all_labels['glycaemia'].append(glycaemia if glycaemia is not None else -1)

            # Track label availability
            if quality is not None:
                label_availability['quality'] += 1
            if hr is not None:
                label_availability['hr'] += 1
            if motion is not None:
                label_availability['motion'] += 1
            if spo2 is not None:
                label_availability['spo2'] += 1
            if glycaemia is not None:
                label_availability['glycaemia'] += 1

        if len(all_signals) == 0:
            print(f"  ❌ No valid data collected")
            print(f"     Skipped {skipped} recordings")
            return {}

        # Convert to numpy arrays
        signals_array = np.array(all_signals)  # [N, 5, 1024]
        subjects_array = np.array(all_subjects)  # [N]

        # Convert label lists to arrays
        labels_dict = {
            key: np.array(vals) for key, vals in all_labels.items()
        }

        print(f"\n  ✓ Collected {len(signals_array)} samples")
        print(f"    Skipped: {skipped} recordings")
        print(f"    Signals shape: {signals_array.shape}")

        print(f"\n  📋 Label availability:")
        total_samples = len(signals_array)
        for label_name, count in label_availability.items():
            pct = (count / total_samples) * 100
            print(f"    {label_name:15s}: {count:4d}/{total_samples} ({pct:5.1f}%)")

        # Subject-level split (CRITICAL for medical ML)
        unique_subjects = np.unique(subjects_array)
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(unique_subjects)

        n_train = int(len(unique_subjects) * split_ratio[0])
        n_val = int(len(unique_subjects) * split_ratio[1])

        train_subjects = unique_subjects[:n_train]
        val_subjects = unique_subjects[n_train:n_train+n_val]
        test_subjects = unique_subjects[n_train+n_val:]

        print(f"\n  Subject-level split:")
        print(f"    Train subjects: {len(train_subjects)}")
        print(f"    Val subjects: {len(val_subjects)}")
        print(f"    Test subjects: {len(test_subjects)}")

        # Create splits
        train_mask = np.isin(subjects_array, train_subjects)
        val_mask = np.isin(subjects_array, val_subjects)
        test_mask = np.isin(subjects_array, test_subjects)

        splits = {
            'train': {
                'signals': signals_array[train_mask],
                'labels': {key: vals[train_mask] for key, vals in labels_dict.items()}
            },
            'val': {
                'signals': signals_array[val_mask],
                'labels': {key: vals[val_mask] for key, vals in labels_dict.items()}
            },
            'test': {
                'signals': signals_array[test_mask],
                'labels': {key: vals[test_mask] for key, vals in labels_dict.items()}
            }
        }

        print(f"\n  ✓ Split: Train={train_mask.sum()}, Val={val_mask.sum()}, Test={test_mask.sum()}")

        return splits

    def save_multitask_dataset(
        self,
        splits: Dict,
        output_dir: Path
    ):
        """Save multi-task dataset to disk"""
        multitask_dir = Path(output_dir) / 'multitask'
        multitask_dir.mkdir(parents=True, exist_ok=True)

        for split_name, split_data in splits.items():
            output_file = multitask_dir / f'{split_name}.npz'

            signals = split_data['signals']
            labels = split_data['labels']

            # Save with all labels
            np.savez_compressed(
                output_file,
                signals=signals,
                **labels  # Unpack all label arrays
            )

            print(f"  ✓ Saved {split_name}: {output_file}")
            print(f"    Signals shape: {signals.shape}")
            print(f"    Labels: {list(labels.keys())}")
            print(f"    Size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--raw-dir',
        type=str,
        required=True,
        help='Directory with raw BUT-PPG recordings (contains *_PPG.dat files)'
    )

    parser.add_argument(
        '--annotations-dir',
        type=str,
        required=True,
        help='Directory with annotation CSV files'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed/butppg',
        help='Output directory for processed data'
    )

    parser.add_argument(
        '--target-fs',
        type=int,
        default=125,
        help='Target sampling frequency (Hz)'
    )

    parser.add_argument(
        '--window-size',
        type=int,
        default=1024,
        help='Window size in samples'
    )

    parser.add_argument(
        '--tasks',
        type=str,
        default='quality,heart_rate,motion',
        help='Tasks to process (comma-separated, or "multitask" for all labels)'
    )

    parser.add_argument(
        '--multitask',
        action='store_true',
        help='Create unified multi-task dataset with all labels'
    )

    args = parser.parse_args()

    print("="*80)
    print("BUT-PPG CLINICAL DATA PROCESSOR")
    print("="*80)
    print(f"Raw data directory: {args.raw_dir}")
    print(f"Annotations directory: {args.annotations_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target sampling rate: {args.target_fs} Hz")
    print(f"Window size: {args.window_size} samples ({args.window_size / args.target_fs:.2f} seconds)")

    # Initialize processor
    processor = BUTPPGProcessor(
        raw_dir=Path(args.raw_dir),
        annotations_dir=Path(args.annotations_dir),
        target_fs=args.target_fs,
        window_size=args.window_size
    )

    # Create multi-task dataset if requested
    if args.multitask or 'multitask' in args.tasks.lower():
        print(f"\n{'='*80}")
        print(f"CREATING MULTI-TASK DATASET")
        print(f"{'='*80}")

        splits = processor.create_multitask_dataset()

        if splits:
            processor.save_multitask_dataset(
                splits,
                Path(args.output_dir)
            )
    else:
        # Process each task separately
        tasks = [t.strip() for t in args.tasks.split(',')]

        for task in tasks:
            print(f"\n{'='*80}")
            print(f"PROCESSING TASK: {task.upper()}")
            print(f"{'='*80}")

            splits = processor.create_task_dataset(task)

            if splits:
                processor.save_task_dataset(
                    splits,
                    Path(args.output_dir),
                    task
                )

    print("\n" + "="*80)
    print("✅ PROCESSING COMPLETE!")
    print("="*80)
    print(f"✓ Processed data saved to: {args.output_dir}")
    print("\n📖 Next step:")
    print("  Run downstream evaluation: python scripts/run_downstream_evaluation.py")


if __name__ == '__main__':
    main()
