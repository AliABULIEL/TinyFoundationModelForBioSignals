#!/usr/bin/env python3
"""
BUT-PPG Clinical Data Processor

Processes raw BUT-PPG data and extracts clinical labels for:
1. Signal Quality (binary classification)
2. Heart Rate (regression)
3. Motion Type (8-class classification)

Usage:
    python scripts/process_butppg_clinical.py \
        --raw-dir data/but_ppg/raw \
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
from typing import Dict, Tuple, List


class BUTPPGProcessor:
    """Process BUT-PPG raw data into task-specific datasets"""

    # Channel order for 5-channel input
    CHANNEL_ORDER = ['ACC_X', 'ACC_Y', 'ACC_Z', 'PPG', 'ECG']

    # Motion classes (from BUT-PPG documentation)
    MOTION_CLASSES = {
        0: 'sitting',
        1: 'standing',
        2: 'walking_slow',
        3: 'walking_normal',
        4: 'walking_fast',
        5: 'running',
        6: 'cycling',
        7: 'hand_movement'
    }

    def __init__(
        self,
        raw_dir: Path,
        target_fs: int = 125,
        window_size: int = 1024
    ):
        self.raw_dir = Path(raw_dir)
        self.target_fs = target_fs
        self.window_size = window_size

        # Load annotations
        print("\n" + "="*80)
        print("LOADING ANNOTATIONS")
        print("="*80)
        self.quality_labels = self._load_quality_labels()
        self.hr_labels = self._load_hr_labels()
        self.motion_labels = self._load_motion_labels()

    def _load_quality_labels(self) -> pd.DataFrame:
        """Load signal quality labels"""
        ann_file = self.raw_dir / 'annotations' / 'PPGQualityLabels.csv'

        if not ann_file.exists():
            print(f"⚠️  Quality labels not found: {ann_file}")
            return pd.DataFrame()

        df = pd.read_csv(ann_file)
        print(f"✓ Loaded quality labels: {len(df)} recordings")
        return df

    def _load_hr_labels(self) -> pd.DataFrame:
        """Load heart rate reference labels"""
        ann_file = self.raw_dir / 'annotations' / 'HRReference.csv'

        if not ann_file.exists():
            print(f"⚠️  HR labels not found: {ann_file}")
            return pd.DataFrame()

        df = pd.read_csv(ann_file)
        print(f"✓ Loaded HR labels: {len(df)} recordings")
        return df

    def _load_motion_labels(self) -> pd.DataFrame:
        """Load motion type labels"""
        ann_file = self.raw_dir / 'annotations' / 'MotionLabels.csv'

        if not ann_file.exists():
            print(f"⚠️  Motion labels not found: {ann_file}")
            # Try alternative filename
            ann_file = self.raw_dir / 'annotations' / 'MotionTypes.csv'
            if not ann_file.exists():
                return pd.DataFrame()

        df = pd.read_csv(ann_file)
        print(f"✓ Loaded motion labels: {len(df)} recordings")
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
            Preprocessed signal at target_fs
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

    def load_recording(self, record_path: Path) -> Tuple[np.ndarray, Dict]:
        """
        Load and preprocess a single BUT-PPG recording

        Returns:
            signals: [5, 1024] array (ACC_X, ACC_Y, ACC_Z, PPG, ECG)
            metadata: Dict with original sampling rates and signal names
        """
        try:
            record = wfdb.rdrecord(str(record_path.with_suffix('')))

            # BUT-PPG signals: typically ACC (3-axis), PPG, ECG
            sig_names = [name.upper() for name in record.sig_name]

            # Initialize 5-channel array
            signals_5ch = np.zeros((5, self.window_size))

            # Map signals to correct channels
            for i, sig_name in enumerate(sig_names):
                signal = record.p_signal[:, i]
                original_fs = record.fs

                # Determine signal type and channel index
                if 'ACC' in sig_name or 'ACCELEROMETER' in sig_name:
                    if 'X' in sig_name:
                        ch_idx = 0  # ACC_X
                        sig_type = 'ACC'
                    elif 'Y' in sig_name:
                        ch_idx = 1  # ACC_Y
                        sig_type = 'ACC'
                    elif 'Z' in sig_name:
                        ch_idx = 2  # ACC_Z
                        sig_type = 'ACC'
                    else:
                        continue
                elif 'PPG' in sig_name or 'PLETH' in sig_name:
                    ch_idx = 3  # PPG
                    sig_type = 'PPG'
                elif 'ECG' in sig_name:
                    ch_idx = 4  # ECG
                    sig_type = 'ECG'
                else:
                    continue

                # Preprocess signal
                processed = self.preprocess_signal(signal, original_fs, sig_type)
                signals_5ch[ch_idx] = processed

            metadata = {
                'sig_names': sig_names,
                'original_fs': record.fs,
                'record_id': record_path.stem
            }

            return signals_5ch, metadata

        except Exception as e:
            print(f"  ⚠️  Error loading {record_path}: {e}")
            return None, None

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
        print(f"\n📊 Creating dataset for task: {task}")

        # Get labels for this task
        if task == 'quality':
            labels_df = self.quality_labels
            label_column = 'Quality'  # Adjust based on actual CSV column name
        elif task == 'heart_rate':
            labels_df = self.hr_labels
            label_column = 'HR'
        elif task == 'motion':
            labels_df = self.motion_labels
            label_column = 'MotionType'
        else:
            raise ValueError(f"Unknown task: {task}")

        if labels_df.empty:
            print(f"  ❌ No labels available for {task}")
            return {}

        # Collect all recordings with labels
        all_signals = []
        all_labels = []
        all_subjects = []

        for subject_dir in tqdm(sorted(self.raw_dir.glob("subject_*")), desc="Processing subjects"):
            subject_id = int(subject_dir.name.replace("subject_", ""))

            for record_file in subject_dir.glob("*.hea"):
                record_id = record_file.stem

                # Check if this recording has a label
                if 'RecordID' in labels_df.columns:
                    label_row = labels_df[labels_df['RecordID'] == record_id]
                else:
                    # Try matching by subject and recording number
                    label_row = labels_df[labels_df['Subject'] == subject_id]

                if label_row.empty:
                    continue

                # Load signal
                signals, metadata = self.load_recording(record_file)

                if signals is None:
                    continue

                # Get label
                try:
                    label = label_row[label_column].iloc[0]

                    # Convert quality label (if string)
                    if task == 'quality':
                        if isinstance(label, str):
                            label = 1 if label.lower() in ['good', 'high', '1'] else 0

                    all_signals.append(signals)
                    all_labels.append(label)
                    all_subjects.append(subject_id)

                except Exception as e:
                    print(f"  ⚠️  Error processing {record_id}: {e}")
                    continue

        if len(all_signals) == 0:
            print(f"  ❌ No valid data collected for {task}")
            return {}

        # Convert to numpy arrays
        signals_array = np.array(all_signals)  # [N, 5, 1024]
        labels_array = np.array(all_labels)    # [N]
        subjects_array = np.array(all_subjects)  # [N]

        print(f"  ✓ Collected {len(signals_array)} samples")
        print(f"    Signals shape: {signals_array.shape}")
        print(f"    Labels shape: {labels_array.shape}")

        # Subject-level split (CRITICAL for medical ML)
        unique_subjects = np.unique(subjects_array)
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(unique_subjects)

        n_train = int(len(unique_subjects) * split_ratio[0])
        n_val = int(len(unique_subjects) * split_ratio[1])

        train_subjects = unique_subjects[:n_train]
        val_subjects = unique_subjects[n_train:n_train+n_val]
        test_subjects = unique_subjects[n_train+n_val:]

        # Create splits
        train_mask = np.isin(subjects_array, train_subjects)
        val_mask = np.isin(subjects_array, val_subjects)
        test_mask = np.isin(subjects_array, test_subjects)

        splits = {
            'train': (signals_array[train_mask], labels_array[train_mask]),
            'val': (signals_array[val_mask], labels_array[val_mask]),
            'test': (signals_array[test_mask], labels_array[test_mask])
        }

        print(f"  ✓ Split: Train={train_mask.sum()}, Val={val_mask.sum()}, Test={test_mask.sum()}")

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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--raw-dir',
        type=str,
        default='data/but_ppg/raw',
        help='Directory with raw BUT-PPG data'
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
        help='Tasks to process (comma-separated)'
    )

    args = parser.parse_args()

    print("="*80)
    print("BUT-PPG CLINICAL DATA PROCESSOR")
    print("="*80)

    # Initialize processor
    processor = BUTPPGProcessor(
        raw_dir=Path(args.raw_dir),
        target_fs=args.target_fs,
        window_size=args.window_size
    )

    # Process each task
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
    print("PROCESSING COMPLETE!")
    print("="*80)
    print(f"✓ Processed data saved to: {args.output_dir}")
    print("\nNext step:")
    print("  Run downstream evaluation: python scripts/run_downstream_evaluation.py")


if __name__ == '__main__':
    main()
