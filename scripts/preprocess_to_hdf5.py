"""
CAPTURE-24 HDF5 Preprocessor

This script converts the raw CSV.gz files to a single HDF5 file for fast loading.
Run this ONCE, then all future data loading will be instant.

Usage:
    python preprocess_to_hdf5.py --data_path /path/to/capture24 --output capture24.h5

The output HDF5 file structure:
    /metadata
        - num_participants
        - original_sampling_rate
        - target_sampling_rate
        - num_classes
    /P001
        /signal  (N, 3) float32 - resampled accelerometry
        /labels  (N,) int64 - activity labels
    /P002
        ...
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from math import gcd
from typing import Tuple, List

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Walmsley 5-class label mapping
LABEL_MAP = {
    'sleep': 0,
    'sedentary': 1,
    'light': 2,
    'moderate-vigorous': 3,
    'bicycling': 4,
}


def find_participants(data_path: Path) -> List[str]:
    """Find all participant CSV.gz files."""
    csv_files = sorted(data_path.glob("P*.csv.gz"))
    participants = []
    for f in csv_files:
        match = re.match(r'(P\d{3})\.csv\.gz', f.name)
        if match:
            participants.append(match.group(1))
    return participants


def load_and_resample_participant(
    csv_path: Path,
    original_rate: int = 100,
    target_rate: int = 30,
    num_classes: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a participant's CSV.gz file and resample to target rate.
    
    Returns:
        signal: (N, 3) float32 array
        labels: (N,) int64 array
    """
    from scipy.signal import resample_poly
    
    # Load CSV
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Find accelerometer columns
    accel_cols = None
    for cols in [['x', 'y', 'z'], ['X', 'Y', 'Z']]:
        if all(c in df.columns for c in cols):
            accel_cols = cols
            break
    
    if accel_cols is None:
        raise ValueError(f"No x,y,z columns found in {csv_path}")
    
    # Extract signal
    signal = df[accel_cols].values.astype(np.float32)
    signal = np.nan_to_num(signal, nan=0.0)
    
    # Extract labels (Walmsley2020 preferred)
    label_col = None
    for col in ['Walmsley2020', 'annotation', 'label']:
        if col in df.columns:
            label_col = col
            break
    
    if label_col is None:
        labels = np.zeros(len(df), dtype=np.int64)
    else:
        raw = df[label_col].values
        if raw.dtype == object:
            # Convert string labels
            labels = np.array([
                LABEL_MAP.get(str(l).lower().strip(), 2) if pd.notna(l) else 0 
                for l in raw
            ], dtype=np.int64)
        else:
            labels = np.clip(raw.astype(np.int64), 0, num_classes - 1)
    
    # Resample if needed
    if target_rate != original_rate:
        g = gcd(original_rate, target_rate)
        up, down = target_rate // g, original_rate // g
        
        # Resample signal
        ch0 = resample_poly(signal[:, 0], up, down)
        n_new = len(ch0)
        
        resampled = np.zeros((n_new, 3), dtype=np.float32)
        resampled[:, 0] = ch0
        for i in range(1, 3):
            ch = resample_poly(signal[:, i], up, down)
            resampled[:, i] = ch[:n_new]
        
        # Resample labels
        idx = np.linspace(0, len(labels) - 1, n_new)
        labels = labels[np.round(idx).astype(int)]
        signal = resampled
    
    return signal, labels


def preprocess_to_hdf5(
    data_path: Path,
    output_path: Path,
    original_rate: int = 100,
    target_rate: int = 30,
    num_classes: int = 5,
    chunk_size: int = 512,  # Window size for optimal chunking
    compression: str = 'gzip',
    compression_opts: int = 4,
):
    """
    Convert all CAPTURE-24 CSV.gz files to a single HDF5 file.
    
    Args:
        data_path: Directory containing P*.csv.gz files
        output_path: Output HDF5 file path
        original_rate: Original sampling rate (100 Hz for CAPTURE-24)
        target_rate: Target sampling rate (30 Hz for TTM)
        num_classes: Number of activity classes
        chunk_size: Chunk size for HDF5 (matches window size for optimal access)
        compression: HDF5 compression type
        compression_opts: Compression level (1-9)
    """
    data_path = Path(data_path)
    output_path = Path(output_path)
    
    # Find participants
    participants = find_participants(data_path)
    if not participants:
        raise FileNotFoundError(f"No P*.csv.gz files found in {data_path}")
    
    logger.info(f"Found {len(participants)} participants")
    logger.info(f"Output: {output_path}")
    logger.info(f"Resampling: {original_rate}Hz → {target_rate}Hz")
    logger.info(f"Compression: {compression} (level {compression_opts})")
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create HDF5 file
    with h5py.File(output_path, 'w') as h5f:
        # Store metadata
        meta = h5f.create_group('metadata')
        meta.attrs['num_participants'] = len(participants)
        meta.attrs['original_sampling_rate'] = original_rate
        meta.attrs['target_sampling_rate'] = target_rate
        meta.attrs['num_classes'] = num_classes
        meta.attrs['num_channels'] = 3
        meta.attrs['label_scheme'] = 'walmsley'
        meta.attrs['chunk_size'] = chunk_size
        
        # Store label names
        label_names = ['sleep', 'sedentary', 'light', 'moderate-vigorous', 'bicycling']
        meta.create_dataset('label_names', data=[n.encode() for n in label_names])
        
        # Process each participant
        total_samples = 0
        failed = []
        
        for pid in tqdm(participants, desc="Processing participants"):
            csv_path = data_path / f"{pid}.csv.gz"
            
            try:
                signal, labels = load_and_resample_participant(
                    csv_path,
                    original_rate=original_rate,
                    target_rate=target_rate,
                    num_classes=num_classes
                )
                
                # Create participant group
                grp = h5f.create_group(pid)
                
                # Store signal with chunking (optimized for window-based access)
                grp.create_dataset(
                    'signal',
                    data=signal,
                    dtype='float32',
                    chunks=(min(chunk_size, len(signal)), 3),
                    compression=compression,
                    compression_opts=compression_opts,
                )
                
                # Store labels
                grp.create_dataset(
                    'labels',
                    data=labels,
                    dtype='int64',
                    chunks=(min(chunk_size, len(labels)),),
                    compression=compression,
                    compression_opts=compression_opts,
                )
                
                # Store participant metadata
                grp.attrs['num_samples'] = len(signal)
                grp.attrs['duration_seconds'] = len(signal) / target_rate
                grp.attrs['duration_hours'] = len(signal) / target_rate / 3600
                
                total_samples += len(signal)
                
            except Exception as e:
                logger.error(f"Failed to process {pid}: {e}")
                failed.append(pid)
                continue
        
        # Update metadata with totals
        meta.attrs['total_samples'] = total_samples
        meta.attrs['total_hours'] = total_samples / target_rate / 3600
        meta.attrs['num_processed'] = len(participants) - len(failed)
        
        # Store participant list
        processed = [p for p in participants if p not in failed]
        meta.create_dataset('participant_ids', data=[p.encode() for p in processed])
    
    # Summary
    file_size_gb = output_path.stat().st_size / 1e9
    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Processed: {len(participants) - len(failed)}/{len(participants)} participants")
    logger.info(f"  Total samples: {total_samples:,}")
    logger.info(f"  Total hours: {total_samples / target_rate / 3600:.1f}")
    logger.info(f"  Output file: {output_path}")
    logger.info(f"  File size: {file_size_gb:.2f} GB")
    
    if failed:
        logger.warning(f"  Failed participants: {failed}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert CAPTURE-24 CSV.gz files to HDF5 format"
    )
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to directory containing P*.csv.gz files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output HDF5 file path (default: {data_path}/capture24.h5)'
    )
    parser.add_argument(
        '--target_rate',
        type=int,
        default=30,
        help='Target sampling rate in Hz (default: 30)'
    )
    parser.add_argument(
        '--compression',
        type=str,
        default='gzip',
        choices=['gzip', 'lzf', None],
        help='Compression type (default: gzip)'
    )
    parser.add_argument(
        '--compression_level',
        type=int,
        default=4,
        help='Compression level 1-9 (default: 4)'
    )
    
    args = parser.parse_args()
    
    data_path = Path(args.data_path)
    output_path = Path(args.output) if args.output else data_path / 'capture24.h5'
    
    preprocess_to_hdf5(
        data_path=data_path,
        output_path=output_path,
        target_rate=args.target_rate,
        compression=args.compression,
        compression_opts=args.compression_level,
    )


if __name__ == '__main__':
    main()
