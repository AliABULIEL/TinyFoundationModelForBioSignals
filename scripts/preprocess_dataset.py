#!/usr/bin/env python3
"""Dataset preprocessing script for TTM-HAR.

This script preprocesses raw accelerometry data for HAR:
- Resampling to target sampling rate
- Gravity removal (optional)
- Windowing into fixed-length segments
- Normalization
- Data quality checks
- Export preprocessed data

Usage:
    # Preprocess CAPTURE-24 dataset
    python scripts/preprocess_dataset.py \
        --input_dir data/raw/capture24 \
        --output_dir data/processed/capture24 \
        --config configs/default.yaml

    # Preview preprocessing without saving
    python scripts/preprocess_dataset.py \
        --input_dir data/raw \
        --preview_only \
        --num_preview_samples 5

    # Process specific participants
    python scripts/preprocess_dataset.py \
        --input_dir data/raw \
        --output_dir data/processed \
        --participant_ids P001 P002 P003
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.data.capture24_adapter import CAPTURE24Dataset
from src.preprocessing.pipeline import PreprocessingPipeline

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess accelerometry data for HAR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess entire dataset
  python scripts/preprocess_dataset.py \\
      --input_dir data/raw/capture24 \\
      --output_dir data/processed/capture24

  # Preprocess with custom config
  python scripts/preprocess_dataset.py \\
      --input_dir data/raw \\
      --output_dir data/processed \\
      --config configs/my_preprocessing.yaml

  # Preview preprocessing for 5 participants
  python scripts/preprocess_dataset.py \\
      --input_dir data/raw \\
      --preview_only \\
      --num_preview_samples 5

  # Process specific participants
  python scripts/preprocess_dataset.py \\
      --input_dir data/raw \\
      --output_dir data/processed \\
      --participant_ids P001 P002 P003

  # Skip gravity removal
  python scripts/preprocess_dataset.py \\
      --input_dir data/raw \\
      --output_dir data/processed \\
      --no_gravity_removal
        """
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing raw data",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Output directory for preprocessed data (default: data/processed)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file (default: configs/default.yaml)",
    )

    parser.add_argument(
        "--participant_ids",
        nargs="+",
        default=None,
        help="List of participant IDs to process (default: all)",
    )

    parser.add_argument(
        "--preview_only",
        action="store_true",
        help="Preview preprocessing without saving",
    )

    parser.add_argument(
        "--num_preview_samples",
        type=int,
        default=5,
        help="Number of samples to preview (default: 5)",
    )

    parser.add_argument(
        "--no_gravity_removal",
        action="store_true",
        help="Skip gravity removal step",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, set to -1 for all CPUs)",
    )

    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


def print_preprocessing_summary(config: dict, args: argparse.Namespace) -> None:
    """
    Print preprocessing configuration summary.

    Args:
        config: Configuration dictionary
        args: Command-line arguments
    """
    print("\n" + "=" * 80)
    print("PREPROCESSING CONFIGURATION")
    print("=" * 80)

    # Data config
    data_config = config.get("dataset", {})
    print("\nData Configuration:")
    print(f"  Input directory:     {args.input_dir}")
    print(f"  Output directory:    {args.output_dir}")
    print(f"  Target sample rate:  {data_config.get('target_sample_rate', 30)} Hz")

    # Preprocessing config
    preproc_config = config.get("preprocessing", {})
    print("\nPreprocessing Steps:")
    print(f"  Resampling:          Enabled")
    print(f"  Gravity removal:     {'Disabled' if args.no_gravity_removal else 'Enabled'}")
    print(f"  Windowing:           {preproc_config.get('window_size_sec', 10)}s windows, "
          f"{preproc_config.get('stride_sec', 5)}s stride")
    print(f"  Normalization:       {preproc_config.get('normalization', 'zscore')}")

    # Processing config
    print("\nProcessing Configuration:")
    if args.participant_ids:
        print(f"  Participants:        {len(args.participant_ids)} specified")
    else:
        print(f"  Participants:        All available")
    print(f"  Preview mode:        {'Yes' if args.preview_only else 'No'}")
    print(f"  Workers:             {args.num_workers if args.num_workers > 0 else 'All CPUs'}")

    print("=" * 80 + "\n")


def preview_preprocessing(
    dataset: CAPTURE24Dataset,
    pipeline: PreprocessingPipeline,
    participant_ids: List[str],
    num_samples: int,
) -> None:
    """
    Preview preprocessing results without saving.

    Args:
        dataset: Dataset instance
        pipeline: Preprocessing pipeline
        participant_ids: List of participant IDs
        num_samples: Number of samples to preview
    """
    print("\n" + "=" * 80)
    print("PREPROCESSING PREVIEW")
    print("=" * 80)

    num_preview = min(num_samples, len(participant_ids))

    for i, participant_id in enumerate(participant_ids[:num_preview]):
        print(f"\nParticipant {i + 1}/{num_preview}: {participant_id}")
        print("-" * 40)

        try:
            # Load raw data
            signal, labels = dataset.load_participant(participant_id)

            print(f"  Raw data:")
            print(f"    Signal shape:  {signal.shape}")
            print(f"    Labels shape:  {labels.shape}")
            print(f"    Duration:      {len(signal) / dataset.original_sample_rate:.1f}s")
            print(f"    Sample rate:   {dataset.original_sample_rate} Hz")

            # Apply preprocessing
            windows, window_labels = pipeline.process_participant(
                signal, labels, dataset.original_sample_rate
            )

            print(f"  Preprocessed data:")
            print(f"    Windows shape: {windows.shape}")
            print(f"    Labels shape:  {window_labels.shape}")
            print(f"    Num windows:   {len(windows)}")

            # Show label distribution
            unique, counts = np.unique(window_labels, return_counts=True)
            print(f"  Label distribution:")
            for label, count in zip(unique, counts):
                label_name = dataset.label_map.get(int(label), f"Unknown ({label})")
                print(f"    {label_name}: {count} ({count / len(window_labels) * 100:.1f}%)")

        except Exception as e:
            print(f"  ERROR: {e}")
            logger.error(f"Failed to preview {participant_id}: {e}", exc_info=True)

    print("\n" + "=" * 80)


def preprocess_dataset(
    dataset: CAPTURE24Dataset,
    pipeline: PreprocessingPipeline,
    participant_ids: List[str],
    output_dir: Path,
) -> dict:
    """
    Preprocess entire dataset and save results.

    Args:
        dataset: Dataset instance
        pipeline: Preprocessing pipeline
        participant_ids: List of participant IDs to process
        output_dir: Output directory

    Returns:
        Dictionary with preprocessing statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create participant directory
    participant_dir = output_dir / "participants"
    participant_dir.mkdir(exist_ok=True)

    # Statistics
    stats = {
        "total_participants": len(participant_ids),
        "successful": 0,
        "failed": 0,
        "total_windows": 0,
        "failed_participants": [],
    }

    # Process each participant
    logger.info(f"Processing {len(participant_ids)} participants...")

    for participant_id in tqdm(participant_ids, desc="Preprocessing"):
        try:
            # Load raw data
            signal, labels = dataset.load_participant(participant_id)

            # Apply preprocessing
            windows, window_labels = pipeline.process_participant(
                signal, labels, dataset.original_sample_rate
            )

            # Save preprocessed data
            output_file = participant_dir / f"{participant_id}.npz"
            np.savez_compressed(
                output_file,
                windows=windows,
                labels=window_labels,
                participant_id=participant_id,
            )

            # Update statistics
            stats["successful"] += 1
            stats["total_windows"] += len(windows)

            logger.debug(
                f"Processed {participant_id}: {len(windows)} windows saved to {output_file}"
            )

        except Exception as e:
            stats["failed"] += 1
            stats["failed_participants"].append(participant_id)
            logger.error(f"Failed to process {participant_id}: {e}", exc_info=True)

    # Save metadata
    metadata = {
        "num_participants": stats["successful"],
        "total_windows": stats["total_windows"],
        "participant_ids": [p for p in participant_ids if p not in stats["failed_participants"]],
        "label_map": dataset.label_map,
        "preprocessing_config": pipeline.get_config(),
    }

    metadata_file = output_dir / "metadata.npz"
    np.savez(metadata_file, **metadata)

    logger.info(f"Saved metadata to {metadata_file}")

    return stats


def print_preprocessing_results(stats: dict) -> None:
    """
    Print preprocessing results summary.

    Args:
        stats: Statistics dictionary
    """
    print("\n" + "=" * 80)
    print("PREPROCESSING RESULTS")
    print("=" * 80)

    print(f"\nTotal participants:     {stats['total_participants']}")
    print(f"Successfully processed: {stats['successful']}")
    print(f"Failed:                 {stats['failed']}")
    print(f"Total windows created:  {stats['total_windows']}")

    if stats['failed_participants']:
        print(f"\nFailed participants:")
        for participant_id in stats['failed_participants']:
            print(f"  - {participant_id}")

    print("=" * 80 + "\n")


def main():
    """Main preprocessing function."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    log_file = Path(args.output_dir) / "preprocess.log" if not args.preview_only else None
    setup_logging(
        log_level=args.log_level,
        log_file=str(log_file) if log_file else None,
    )

    logger.info("=" * 80)
    logger.info("TTM-HAR Dataset Preprocessing Script")
    logger.info("=" * 80)

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Print preprocessing summary
    print_preprocessing_summary(config, args)

    # Create dataset
    logger.info(f"Loading dataset from {args.input_dir}")
    dataset = CAPTURE24Dataset(data_dir=args.input_dir)

    # Get participant IDs
    if args.participant_ids:
        participant_ids = args.participant_ids
        logger.info(f"Processing {len(participant_ids)} specified participants")
    else:
        participant_ids = dataset.get_participant_ids()
        logger.info(f"Found {len(participant_ids)} participants in dataset")

    # Create preprocessing pipeline
    logger.info("Creating preprocessing pipeline...")

    preproc_config = config.get("preprocessing", {})
    target_sample_rate = config.get("dataset", {}).get("target_sample_rate", 30)

    pipeline = PreprocessingPipeline(
        target_sample_rate=target_sample_rate,
        window_size_sec=preproc_config.get("window_size_sec", 10),
        stride_sec=preproc_config.get("stride_sec", 5),
        remove_gravity=not args.no_gravity_removal,
        normalization=preproc_config.get("normalization", "zscore"),
    )

    # Preview or process
    if args.preview_only:
        logger.info("Running in preview mode (no data will be saved)")
        preview_preprocessing(
            dataset, pipeline, participant_ids, args.num_preview_samples
        )
    else:
        logger.info("Starting preprocessing...")
        stats = preprocess_dataset(
            dataset, pipeline, participant_ids, Path(args.output_dir)
        )

        # Print results
        print_preprocessing_results(stats)

        logger.info(f"Preprocessing complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
