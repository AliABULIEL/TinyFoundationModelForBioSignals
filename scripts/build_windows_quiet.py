#!/usr/bin/env python3
"""
Build VitalDB windows with warnings suppressed.
This is a wrapper script that disables all warnings before running.

Usage:
    python3 scripts/build_windows_quiet.py train        # Build train windows
    python3 scripts/build_windows_quiet.py val          # Build val windows
    python3 scripts/build_windows_quiet.py test         # Build test windows
    python3 scripts/build_windows_quiet.py train 8      # Use 8 workers
"""

import os
import sys
import warnings
from pathlib import Path

# CRITICAL: Suppress ALL warnings before any imports
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Now import everything
import subprocess

def main():
    """Run build-windows with all required arguments."""
    
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/build_windows_quiet.py [train|val|test] [num_workers]")
        print("\nExamples:")
        print("  python3 scripts/build_windows_quiet.py train      # 4 workers (default)")
        print("  python3 scripts/build_windows_quiet.py val 8      # 8 workers")
        sys.exit(1)
    
    split = sys.argv[1]
    num_workers = sys.argv[2] if len(sys.argv) > 2 else '4'
    
    # Validate split
    if split not in ['train', 'val', 'test']:
        print(f"Error: Invalid split '{split}'. Must be 'train', 'val', or 'test'")
        sys.exit(1)
    
    # Build command
    cmd = [
        'python3',
        'scripts/ttm_vitaldb.py',
        'build-windows',
        '--channels-yaml', 'configs/channels.yaml',
        '--windows-yaml', 'configs/windows.yaml',
        '--split-file', 'configs/splits/splits_fallback.json',
        '--split', split,
        '--outdir', 'data/vitaldb_windows',
        '--multiprocess',
        '--num-workers', num_workers
    ]
    
    print("=" * 70)
    print(f"Building {split.upper()} windows (workers={num_workers})")
    print("=" * 70)
    print()
    
    # Run with suppressed warnings
    env = os.environ.copy()
    env['PYTHONWARNINGS'] = 'ignore'
    env['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    result = subprocess.run(cmd, env=env)
    
    print()
    print("=" * 70)
    if result.returncode == 0:
        print(f"✓ {split.upper()} windows built successfully!")
        print(f"Output: data/vitaldb_windows/{split}_windows.npz")
        print("=" * 70)
        print()
        
        # Next steps
        if split == 'train':
            print("Next: Build validation windows")
            print(f"  python3 scripts/build_windows_quiet.py val")
        elif split == 'val':
            print("Next: Run smoke test or start pretraining")
            print("  python3 scripts/smoke_realdata_5min.py --data-dir data/vitaldb_windows")
        
    else:
        print(f"✗ Failed to build {split} windows (exit code: {result.returncode})")
        print("=" * 70)
        sys.exit(result.returncode)

if __name__ == '__main__':
    main()
