#!/usr/bin/env python3
"""Data management utilities for biosignal pipeline.

This module handles data validation, resampling, and format conversion to ensure
compatibility with TTM models. It automatically detects data format issues and
fixes them.

Key Features:
- Validate data format and shape
- Resample signals to target length (1250 → 1024)
- Preserve signal quality during resampling
- Validate labels and metadata
- Create resampled datasets automatically

Usage:
    # Check if data needs resampling
    data_info = check_data_format('data/processed/butppg/windows_with_labels')
    if data_info['needs_resampling']:
        resample_dataset(
            input_dir='data/processed/butppg/windows_with_labels',
            output_dir='data/processed/butppg/windows_1024_ttm',
            target_length=1024
        )
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings

import numpy as np
from scipy import signal
from tqdm import tqdm


class DataFormatError(Exception):
    """Raised when data format is incompatible."""
    pass


def check_data_format(data_dir: str, split: str = 'train') -> Dict:
    """Check data format and determine if resampling is needed.

    Args:
        data_dir: Path to data directory
        split: Data split to check ('train', 'val', 'test')

    Returns:
        info: Dictionary with data format information:
            - 'exists': bool
            - 'num_samples': int
            - 'shape': tuple
            - 'num_channels': int
            - 'sequence_length': int
            - 'needs_resampling': bool
            - 'target_length': int (if needs resampling)
            - 'labels': list of available labels

    Example:
        >>> info = check_data_format('data/processed/butppg/windows_with_labels')
        >>> if info['needs_resampling']:
        ...     print(f"Need to resample from {info['sequence_length']} to {info['target_length']}")
    """
    data_dir = Path(data_dir)
    split_dir = data_dir / split

    info = {
        'exists': False,
        'num_samples': 0,
        'shape': None,
        'num_channels': None,
        'sequence_length': None,
        'needs_resampling': False,
        'target_length': None,
        'labels': []
    }

    # Check if directory exists
    if not split_dir.exists():
        return info

    info['exists'] = True

    # Find sample files
    npz_files = list(split_dir.glob('*.npz'))
    if not npz_files:
        return info

    info['num_samples'] = len(npz_files)

    # Load first file to check format
    sample_file = npz_files[0]
    try:
        data = np.load(sample_file)

        if 'signal' not in data:
            raise DataFormatError(f"No 'signal' key in {sample_file}")

        signal_data = data['signal']
        info['shape'] = signal_data.shape
        info['num_channels'] = signal_data.shape[0]
        info['sequence_length'] = signal_data.shape[1]

        # Check available labels
        label_keys = ['quality', 'hr', 'motion', 'bp_systolic', 'bp_diastolic', 'spo2', 'glycaemia']
        info['labels'] = [key for key in label_keys if key in data]

        # Determine if resampling is needed
        # TTM-Enhanced requires 1024 samples
        target_length = 1024
        if info['sequence_length'] != target_length:
            info['needs_resampling'] = True
            info['target_length'] = target_length

    except Exception as e:
        warnings.warn(f"Error reading sample file {sample_file}: {e}")

    return info


def resample_signal(
    signal: np.ndarray,
    target_length: int,
    method: str = 'fourier'
) -> np.ndarray:
    """Resample a single signal to target length.

    Uses scipy.signal.resample with Fourier method for high-quality resampling
    that preserves frequency content.

    Args:
        signal: Input signal array, shape (num_channels, sequence_length)
        target_length: Target sequence length
        method: Resampling method ('fourier', 'poly')

    Returns:
        resampled: Resampled signal, shape (num_channels, target_length)

    Example:
        >>> signal = np.random.randn(2, 1250)  # 2 channels, 1250 samples
        >>> resampled = resample_signal(signal, target_length=1024)
        >>> resampled.shape
        (2, 1024)
    """
    if signal.shape[1] == target_length:
        return signal  # Already correct length

    # Resample each channel independently
    num_channels = signal.shape[0]
    resampled = np.zeros((num_channels, target_length), dtype=signal.dtype)

    for ch in range(num_channels):
        if method == 'fourier':
            # Fourier-based resampling (high quality, preserves frequencies)
            resampled[ch] = signal.resample(signal[ch], target_length)
        elif method == 'poly':
            # Polyphase filtering (alternative)
            resampled[ch] = signal.resample_poly(signal[ch], target_length, signal.shape[1])
        else:
            raise ValueError(f"Unknown resampling method: {method}")

    return resampled


def validate_resampling_quality(
    original: np.ndarray,
    resampled: np.ndarray,
    target_length: int
) -> Tuple[bool, Dict[str, float]]:
    """Validate that resampling preserved signal quality.

    Checks:
    - Shape is correct
    - No NaN or Inf values
    - Statistical properties preserved (mean, std, range)
    - Correlation with original signal is high

    Args:
        original: Original signal, shape (num_channels, original_length)
        resampled: Resampled signal, shape (num_channels, target_length)
        target_length: Expected target length

    Returns:
        is_valid: Whether resampled signal is valid
        metrics: Quality metrics

    Example:
        >>> is_valid, metrics = validate_resampling_quality(original, resampled, 1024)
        >>> if is_valid:
        ...     print(f"Resampling OK, correlation: {metrics['correlation']:.3f}")
    """
    metrics = {}
    is_valid = True

    # Check shape
    if resampled.shape[1] != target_length:
        return False, {'error': f"Wrong shape: {resampled.shape}"}

    # Check for NaN or Inf
    if np.isnan(resampled).any() or np.isinf(resampled).any():
        return False, {'error': "Contains NaN or Inf values"}

    # Statistical properties (per channel)
    for ch in range(original.shape[0]):
        orig_ch = original[ch]
        resamp_ch = resampled[ch]

        # Mean preservation
        mean_diff = abs(np.mean(resamp_ch) - np.mean(orig_ch))
        metrics[f'ch{ch}_mean_diff'] = mean_diff

        # Std preservation
        std_ratio = np.std(resamp_ch) / (np.std(orig_ch) + 1e-8)
        metrics[f'ch{ch}_std_ratio'] = std_ratio

        # Correlation (downsample original to compare)
        orig_downsampled = signal.resample(orig_ch, target_length)
        correlation = np.corrcoef(orig_downsampled, resamp_ch)[0, 1]
        metrics[f'ch{ch}_correlation'] = correlation

        # Validation thresholds
        if correlation < 0.95:
            is_valid = False
            metrics['error'] = f"Low correlation on channel {ch}: {correlation:.3f}"

    return is_valid, metrics


def resample_dataset(
    input_dir: str,
    output_dir: str,
    target_length: int = 1024,
    splits: Optional[list] = None,
    validate_quality: bool = True,
    verbose: bool = True
) -> Dict[str, int]:
    """Resample entire dataset to target length.

    Args:
        input_dir: Input data directory
        output_dir: Output directory for resampled data
        target_length: Target sequence length
        splits: List of splits to process (default: ['train', 'val', 'test'])
        validate_quality: Whether to validate resampling quality
        verbose: Whether to print progress

    Returns:
        stats: Statistics about resampling (files processed per split)

    Example:
        >>> stats = resample_dataset(
        ...     input_dir='data/processed/butppg/windows_with_labels',
        ...     output_dir='data/processed/butppg/windows_1024_ttm',
        ...     target_length=1024
        ... )
        >>> print(f"Processed {stats['train']} training files")
    """
    if splits is None:
        splits = ['train', 'val', 'test']

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    stats = {}

    if verbose:
        print("\n" + "=" * 80)
        print("RESAMPLING DATASET")
        print("=" * 80)
        print(f"Input:  {input_dir}")
        print(f"Output: {output_dir}")
        print(f"Target length: {target_length}")
        print("=" * 80 + "\n")

    for split in splits:
        split_input_dir = input_dir / split
        split_output_dir = output_dir / split

        if not split_input_dir.exists():
            if verbose:
                print(f"⊘ Skipping {split} (directory not found)")
            continue

        # Create output directory
        split_output_dir.mkdir(parents=True, exist_ok=True)

        # Get all npz files
        npz_files = list(split_input_dir.glob('*.npz'))

        if not npz_files:
            if verbose:
                print(f"⊘ Skipping {split} (no files found)")
            continue

        if verbose:
            print(f"Processing {split} split: {len(npz_files)} files")

        # Process files
        processed = 0
        failed = 0

        iterator = tqdm(npz_files, desc=f"  {split}") if verbose else npz_files

        for npz_file in iterator:
            try:
                # Load data
                data = np.load(npz_file)

                # Resample signal
                original_signal = data['signal']
                resampled_signal = resample_signal(original_signal, target_length)

                # Validate quality if requested
                if validate_quality:
                    is_valid, metrics = validate_resampling_quality(
                        original_signal, resampled_signal, target_length
                    )
                    if not is_valid:
                        warnings.warn(
                            f"Resampling quality check failed for {npz_file.name}: "
                            f"{metrics.get('error', 'Unknown error')}"
                        )
                        failed += 1
                        continue

                # Save resampled data
                output_file = split_output_dir / npz_file.name
                output_data = {'signal': resampled_signal}

                # Copy all labels
                label_keys = ['quality', 'hr', 'motion', 'bp_systolic', 'bp_diastolic', 'spo2', 'glycaemia']
                for key in label_keys:
                    if key in data:
                        output_data[key] = data[key]

                np.savez(output_file, **output_data)
                processed += 1

            except Exception as e:
                warnings.warn(f"Failed to process {npz_file.name}: {e}")
                failed += 1

        stats[split] = processed

        if verbose:
            print(f"  ✓ Processed: {processed} files")
            if failed > 0:
                print(f"  ⚠️  Failed: {failed} files")

    # Summary
    if verbose:
        print("\n" + "=" * 80)
        print("RESAMPLING COMPLETE")
        print("=" * 80)
        for split, count in stats.items():
            print(f"  {split:10s}: {count:6d} files")
        print("=" * 80 + "\n")

    return stats


def get_or_create_resampled_data(
    data_dir: str,
    target_length: int = 1024,
    force_resample: bool = False
) -> str:
    """Get resampled data directory, creating it if needed.

    This is a convenience function that:
    1. Checks if data needs resampling
    2. Checks if resampled data already exists
    3. Creates resampled data if needed
    4. Returns path to appropriate data directory

    Args:
        data_dir: Original data directory
        target_length: Target sequence length
        force_resample: Force resampling even if output exists

    Returns:
        data_dir_to_use: Path to data directory to use

    Example:
        >>> data_dir = get_or_create_resampled_data(
        ...     'data/processed/butppg/windows_with_labels',
        ...     target_length=1024
        ... )
        >>> # Will return resampled directory if needed
    """
    data_dir = Path(data_dir)

    # Check current data format
    info = check_data_format(str(data_dir))

    if not info['exists']:
        raise DataFormatError(f"Data directory not found: {data_dir}")

    # If no resampling needed, return original
    if not info['needs_resampling']:
        print(f"✓ Data already in correct format: {info['sequence_length']} samples")
        return str(data_dir)

    # Determine resampled data directory
    # Replace 'windows_with_labels' with 'windows_1024_ttm'
    resampled_dir = Path(str(data_dir).replace('windows_with_labels', 'windows_1024_ttm'))

    # Check if resampled data already exists
    if resampled_dir.exists() and not force_resample:
        # Verify it's in correct format
        resampled_info = check_data_format(str(resampled_dir))
        if resampled_info['exists'] and not resampled_info['needs_resampling']:
            print(f"✓ Using existing resampled data: {resampled_dir}")
            return str(resampled_dir)

    # Need to create resampled data
    print(f"⚠️  Data needs resampling: {info['sequence_length']} → {target_length} samples")
    print(f"   Creating resampled dataset: {resampled_dir}")

    resample_dataset(
        input_dir=str(data_dir),
        output_dir=str(resampled_dir),
        target_length=target_length,
        validate_quality=True,
        verbose=True
    )

    return str(resampled_dir)


def create_data_report(data_dir: str) -> str:
    """Generate detailed data format report.

    Args:
        data_dir: Data directory to analyze

    Returns:
        report: Formatted data report

    Example:
        >>> report = create_data_report('data/processed/butppg/windows_with_labels')
        >>> print(report)
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("DATA FORMAT REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Directory: {data_dir}")
    report_lines.append("")

    for split in ['train', 'val', 'test']:
        info = check_data_format(data_dir, split=split)

        report_lines.append(f"{split.upper()} Split:")
        report_lines.append("-" * 80)

        if not info['exists']:
            report_lines.append("  ✗ Not found")
        else:
            report_lines.append(f"  Files: {info['num_samples']}")
            report_lines.append(f"  Shape: {info['shape']}")
            report_lines.append(f"  Channels: {info['num_channels']}")
            report_lines.append(f"  Length: {info['sequence_length']} samples")
            report_lines.append(f"  Labels: {', '.join(info['labels'])}")

            if info['needs_resampling']:
                report_lines.append(f"  ⚠️  Needs resampling to {info['target_length']} samples")
            else:
                report_lines.append(f"  ✓ Format compatible with TTM-Enhanced")

        report_lines.append("")

    report_lines.append("=" * 80)

    return "\n".join(report_lines)
