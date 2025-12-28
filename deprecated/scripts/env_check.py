#!/usr/bin/env python3
"""Environment check and system diagnostics.

Verifies:
- PyTorch and torchaudio versions
- CUDA availability and version
- TTM model accessibility
- Real data loading (if --data-dir provided)
- Single forward pass on real data

Usage:
    # Basic environment check
    python scripts/env_check.py

    # Check with real data
    python scripts/env_check.py --data-dir data/vitaldb_windows
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import platform
import warnings

import torch
import numpy as np


def check_pytorch():
    """Check PyTorch installation."""
    print("\n" + "="*70)
    print("PYTORCH ENVIRONMENT")
    print("="*70)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    
    # CUDA
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"    Compute capability: {props.major}.{props.minor}")
    else:
        print("  Running on CPU")
    
    # MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f"\nMPS (Apple Silicon) available: True")
    
    print("="*70)


def check_torchaudio():
    """Check torchaudio installation."""
    print("\n" + "="*70)
    print("TORCHAUDIO")
    print("="*70)
    
    try:
        import torchaudio
        print(f"torchaudio version: {torchaudio.__version__}")
        print("✓ torchaudio installed")
    except ImportError:
        print("✗ torchaudio NOT installed")
        print("  Install: pip install torchaudio")
    
    print("="*70)


def check_ttm():
    """Check TTM model availability."""
    print("\n" + "="*70)
    print("TTM MODEL (IBM GRANITE)")
    print("="*70)
    
    try:
        from tsfm_public import get_model
        print("✓ tsfm_public installed")
        
        # Try to load model (will download if not cached)
        print("\nAttempting to load TTM model...")
        print("Model: ibm-granite/granite-timeseries-ttm-r1")
        print("(This will download ~50MB on first use)")
        
        try:
            model = get_model(
                "ibm-granite/granite-timeseries-ttm-r1",
                context_length=1250,
                prediction_length=96
            )
            print("✓ TTM model loaded successfully")
            
            # Count parameters
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {n_params:,} (~{n_params/1e6:.1f}M)")
            
        except Exception as e:
            print(f"✗ TTM model load failed: {e}")
            print("  Check internet connection")
            print("  Verify HuggingFace access")
    
    except ImportError as e:
        print(f"✗ tsfm_public NOT installed: {e}")
        print("  Install: pip install tsfm_public")
    
    print("="*70)


def check_dependencies():
    """Check other dependencies."""
    print("\n" + "="*70)
    print("OTHER DEPENDENCIES")
    print("="*70)
    
    deps = {
        'numpy': 'numpy',
        'scipy': 'scipy',
        'yaml': 'pyyaml',
        'tqdm': 'tqdm',
    }
    
    for import_name, package_name in deps.items():
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {package_name}: {version}")
        except ImportError:
            print(f"✗ {package_name}: NOT installed")
    
    print("="*70)


def test_real_data(data_dir: Path):
    """Test loading and forward pass with real data."""
    print("\n" + "="*70)
    print("REAL DATA TEST")
    print("="*70)
    print(f"Data directory: {data_dir}")
    
    # Check if data exists
    train_file = data_dir / 'train_windows.npz'
    
    if not train_file.exists():
        print(f"\n✗ Training data not found: {train_file}")
        print("\nTo preprocess VitalDB data:")
        print("  python scripts/ttm_vitaldb.py prepare-splits --output data")
        print("  python scripts/ttm_vitaldb.py build-windows \\")
        print("    --split train --outdir data/vitaldb_windows")
        print("\n⚠ Skipping real data test (no synthetic fallback provided)")
        print("="*70)
        return
    
    # Load data
    print(f"\nLoading: {train_file}")
    data = np.load(train_file)
    
    if 'signals' not in data:
        print(f"✗ Invalid data format")
        print(f"  Expected 'signals' key, found: {list(data.keys())}")
        print("="*70)
        return
    
    signals = torch.from_numpy(data['signals']).float()
    N, C, T = signals.shape
    
    print(f"✓ Data loaded successfully")
    print(f"  Shape: {signals.shape} = [N={N}, C={C}, T={T}]")
    print(f"  Expected: C=2 (PPG+ECG), T=1250 (10s @ 125Hz)")
    
    # Validate shape
    if C != 2:
        print(f"  ⚠ Warning: Expected 2 channels, got {C}")
    if T != 1250:
        print(f"  ⚠ Warning: Expected 1250 timesteps, got {T}")
    
    # Test forward pass
    print("\nTesting forward pass with real data...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Take one batch
    batch_size = min(4, N)
    batch = signals[:batch_size].to(device)
    
    print(f"Batch shape: {batch.shape}")
    
    # Check for NaNs
    if torch.isnan(batch).any():
        print("✗ NaNs detected in input data!")
        print("="*70)
        return
    
    # Build model
    print("\nBuilding TTM encoder...")
    try:
        from src.models.ttm_adapter import create_ttm_model
        
        encoder = create_ttm_model(
            variant='ibm-granite/granite-timeseries-ttm-r1',
            task='ssl',
            input_channels=C,
            context_length=T,
            patch_size=125,
            use_real_ttm=True
        )
        encoder = encoder.to(device)
        encoder.eval()
        
        print("✓ Model built successfully")
        
        # Forward pass
        print("\nRunning forward pass...")
        with torch.no_grad():
            output = encoder(batch)
            
            if isinstance(output, tuple):
                output = output[0]
            
            print(f"Output shape: {output.shape}")
            
            # Check for NaNs in output
            if torch.isnan(output).any():
                print("✗ NaNs detected in model output!")
            else:
                print("✓ Forward pass successful (no NaNs)")
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Environment check and diagnostics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Optional: path to real data for testing (no synthetic fallback)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("ENVIRONMENT CHECK")
    print("="*70)
    print("\nVerifying installation and system configuration...")
    
    # Run checks
    check_pytorch()
    check_torchaudio()
    check_dependencies()
    check_ttm()
    
    # Test real data if provided
    if args.data_dir:
        test_real_data(Path(args.data_dir))
    else:
        print("\n" + "="*70)
        print("REAL DATA TEST")
        print("="*70)
        print("⚠ No --data-dir provided, skipping data test")
        print("\nTo test with real data:")
        print("  python scripts/env_check.py --data-dir data/vitaldb_windows")
        print("\n(No synthetic data fallback is provided)")
        print("="*70)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    cuda_available = torch.cuda.is_available()
    device = "GPU" if cuda_available else "CPU"
    
    print(f"✓ PyTorch {torch.__version__} installed")
    print(f"✓ Device: {device}")
    
    try:
        import tsfm_public
        print("✓ TTM (tsfm_public) available")
    except ImportError:
        print("✗ TTM NOT available (install: pip install tsfm_public)")
    
    if args.data_dir and (Path(args.data_dir) / 'train_windows.npz').exists():
        print("✓ Real data accessible")
    elif args.data_dir:
        print("✗ Real data not found at specified path")
    else:
        print("⚠ Real data not tested (no --data-dir provided)")
    
    print("\n✓ Environment check complete")
    print("="*70)


if __name__ == '__main__':
    main()
