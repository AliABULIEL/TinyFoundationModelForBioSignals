#!/usr/bin/env python3
"""
Quick script to resize existing windows from 1000 to 512 samples
without rebuilding from scratch.
"""

import numpy as np
from pathlib import Path
import argparse

def resize_windows(input_path, output_path=None, target_length=512, method='truncate'):
    """
    Resize windows to target length.
    
    Args:
        input_path: Path to input .npz file
        output_path: Path to save resized data (if None, overwrites input)
        target_length: Target window length (default 512 for TTM)
        method: 'truncate' (use first 512) or 'downsample' (skip samples)
    """
    
    # Load data
    print(f"Loading {input_path}...")
    data = np.load(input_path)
    
    original_shape = data['data'].shape
    print(f"Original shape: {original_shape}")
    
    if original_shape[1] == target_length:
        print(f"Already correct size ({target_length}), skipping...")
        return original_shape
    
    # Resize based on method
    if method == 'truncate':
        # Use first 512 samples
        resized_data = data['data'][:, :target_length, :]
        print(f"Truncated to first {target_length} samples")
        
    elif method == 'downsample':
        # Downsample by skipping samples
        ratio = original_shape[1] / target_length
        indices = np.linspace(0, original_shape[1]-1, target_length, dtype=int)
        resized_data = data['data'][:, indices, :]
        print(f"Downsampled from {original_shape[1]} to {target_length} samples")
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Handle labels
    if 'labels' in data:
        labels = data['labels']
    else:
        labels = np.zeros(len(resized_data), dtype=np.int64)
        print("No labels found, creating zeros")
    
    # Save resized
    if output_path is None:
        # Backup original
        backup_path = Path(input_path).with_suffix('.npz.backup')
        print(f"Backing up original to {backup_path}")
        Path(input_path).rename(backup_path)
        output_path = input_path
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_path,
        data=resized_data,
        labels=labels
    )
    
    print(f"✓ Saved resized data: {resized_data.shape} to {output_path}")
    return resized_data.shape

def main():
    parser = argparse.ArgumentParser(description="Resize windows to 512 samples for TTM")
    parser.add_argument('--input-dir', type=str, default='artifacts/raw_windows',
                       help='Directory containing window files')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (if None, overwrites input)')
    parser.add_argument('--target-length', type=int, default=512,
                       help='Target window length (default 512)')
    parser.add_argument('--method', type=str, default='truncate',
                       choices=['truncate', 'downsample'],
                       help='Resizing method')
    parser.add_argument('--splits', type=str, nargs='+', 
                       default=['train', 'val', 'test'],
                       help='Splits to process')
    
    args = parser.parse_args()
    
    print("="*60)
    print(f"Resizing Windows to {args.target_length} Samples")
    print("="*60)
    
    for split in args.splits:
        print(f"\nProcessing {split} split...")
        print("-"*40)
        
        input_path = Path(args.input_dir) / split / f'{split}_windows.npz'
        
        if not input_path.exists():
            print(f"⚠️  File not found: {input_path}")
            continue
        
        if args.output_dir:
            output_path = Path(args.output_dir) / split / f'{split}_windows.npz'
        else:
            output_path = None
        
        try:
            shape = resize_windows(
                input_path=input_path,
                output_path=output_path,
                target_length=args.target_length,
                method=args.method
            )
        except Exception as e:
            print(f"❌ Error processing {split}: {e}")
    
    print("\n" + "="*60)
    print("✓ Resizing complete!")
    print("="*60)

if __name__ == "__main__":
    main()
