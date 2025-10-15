#!/usr/bin/env python3
"""
BUT-PPG Window Builder Script
Uses existing BUTPPGDataset and BUTPPGLoader to prepare windows

This script integrates with the master pipeline to build BUT-PPG windows
using the existing robust implementation in src/data/butppg_dataset.py
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.butppg_dataset import BUTPPGDataset, create_butppg_dataloaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_butppg_windows(
    data_dir: str,
    splits_file: str,
    output_dir: str,
    modality: str = 'all',  # 'all' = PPG + ECG + ACC (5 channels)
    window_sec: float = 10.0,
    fs: int = 125,
    quality_filter: bool = True,
    batch_size: int = 32
):
    """
    Build preprocessed windows for BUT-PPG using existing BUTPPGDataset.
    
    Args:
        data_dir: BUT-PPG data directory
        splits_file: JSON file with splits
        output_dir: Output directory for windows
        modality: 'all' (PPG+ECG+ACC), 'ppg', 'ecg', or list
        window_sec: Window size in seconds
        fs: Target sampling rate
        quality_filter: Apply quality filtering
        batch_size: Batch size for processing
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load splits
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    
    logger.info(f"Building BUT-PPG windows...")
    logger.info(f"  Data dir: {data_dir}")
    logger.info(f"  Modality: {modality}")
    logger.info(f"  Window: {window_sec}s @ {fs}Hz")
    logger.info(f"  Quality filter: {quality_filter}")
    
    # Process each split
    for split_name in ['train', 'val', 'test']:
        if split_name not in splits:
            logger.info(f"Skipping {split_name} (not in splits)")
            continue
        
        logger.info(f"\nProcessing {split_name} split...")
        
        try:
            # Create dataset using existing implementation
            dataset = BUTPPGDataset(
                data_dir=data_dir,
                modality=modality,
                split=split_name,
                window_sec=window_sec,
                fs=fs,
                quality_filter=quality_filter,
                return_participant_id=False,
                return_labels=False
            )
            
            logger.info(f"  Dataset created: {len(dataset)} samples")
            
            # Create dataloader
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,  # Single process for reliability
                drop_last=False
            )
            
            # Collect all windows
            all_windows = []
            all_labels = []
            
            for batch_idx, batch in enumerate(dataloader):
                # Handle different return formats
                if len(batch) == 2:
                    seg1, seg2 = batch
                elif len(batch) >= 3:
                    seg1, seg2 = batch[0], batch[1]
                else:
                    continue
                
                # Use seg1 (first segment from each pair)
                # Shape: [B, C, T] where C=channels, T=time
                windows = seg1.numpy()
                
                # Transpose to [B, T, C] for consistency with VitalDB
                windows = np.transpose(windows, (0, 2, 1))
                
                all_windows.append(windows)
                # Placeholder labels
                all_labels.extend([0] * len(windows))
                
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"  Processed {(batch_idx + 1) * batch_size} samples...")
            
            if not all_windows:
                logger.warning(f"  No valid windows for {split_name}")
                continue
            
            # Stack all windows
            windows_array = np.vstack(all_windows)
            labels_array = np.array(all_labels)
            
            logger.info(f"  Total windows: {len(windows_array)}")
            logger.info(f"  Shape: {windows_array.shape}")
            
            # Save
            output_file = output_dir / f'{split_name}_windows.npz'
            np.savez_compressed(
                output_file,
                data=windows_array,
                labels=labels_array
            )
            
            logger.info(f"  ✓ Saved to {output_file}")
            logger.info(f"    Size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
            
        except Exception as e:
            logger.error(f"  ✗ Error processing {split_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    logger.info("\n✓ BUT-PPG window building complete!")


def main():
    parser = argparse.ArgumentParser(description="Build BUT-PPG preprocessed windows")
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/but_ppg/dataset',
        help='BUT-PPG data directory'
    )
    
    parser.add_argument(
        '--splits-file',
        type=str,
        required=True,
        help='JSON file with splits'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for windows'
    )
    
    parser.add_argument(
        '--modality',
        type=str,
        default='all',
        help='Modality: all (PPG+ECG+ACC), ppg, ecg, acc'
    )
    
    parser.add_argument(
        '--window-sec',
        type=float,
        default=10.0,
        help='Window size in seconds'
    )
    
    parser.add_argument(
        '--fs',
        type=int,
        default=125,
        help='Target sampling rate'
    )
    
    parser.add_argument(
        '--no-quality-filter',
        action='store_true',
        help='Disable quality filtering'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for processing'
    )
    
    args = parser.parse_args()
    
    build_butppg_windows(
        data_dir=args.data_dir,
        splits_file=args.splits_file,
        output_dir=args.output_dir,
        modality=args.modality,
        window_sec=args.window_sec,
        fs=args.fs,
        quality_filter=not args.no_quality_filter,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
