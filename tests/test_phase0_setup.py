#!/usr/bin/env python3
"""
Phase 0: Environment Setup Test

Tests that all dependencies and paths are ready for the research pipeline.

Run:
    python tests/test_phase0_setup.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np


def test_pytorch_installation():
    """Test PyTorch is installed and working."""
    print("\n[1/5] Testing PyTorch...")
    try:
        x = torch.randn(2, 3)
        y = torch.randn(3, 2)
        z = torch.mm(x, y)
        assert z.shape == (2, 2)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  ✓ PyTorch working (device: {device})")
        return True
    except Exception as e:
        print(f"  ✗ PyTorch failed: {e}")
        return False


def test_ibm_ttm_access():
    """Test IBM TTM model can be loaded."""
    print("\n[2/5] Testing IBM TTM access...")
    try:
        from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
        print("  ✓ tsfm_public installed")
        
        # Try to create a small model config
        from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
        print("  ✓ TTM toolkit available")
        
        return True
        
    except ImportError as e:
        print(f"  ✗ Cannot import IBM TTM: {e}")
        print("  → Fix: pip install git+https://github.com/ibm-granite/granite-tsfm.git")
        return False
    except Exception as e:
        print(f"  ⚠ TTM import issue: {e}")
        return False


def test_biosignal_libraries():
    """Test biosignal processing libraries."""
    print("\n[3/5] Testing biosignal libraries...")
    
    all_ok = True
    
    try:
        import neurokit2 as nk
        print("  ✓ NeuroKit2 installed")
    except:
        print("  ✗ NeuroKit2 missing")
        print("    → Fix: pip install neurokit2")
        all_ok = False
    
    try:
        import scipy.signal
        print("  ✓ SciPy installed")
    except:
        print("  ✗ SciPy missing")
        print("    → Fix: pip install scipy")
        all_ok = False
    
    try:
        import wfdb
        print("  ✓ WFDB installed")
    except:
        print("  ✗ WFDB missing")
        print("    → Fix: pip install wfdb")
        all_ok = False
    
    try:
        import vitaldb
        print("  ✓ VitalDB installed")
    except:
        print("  ✗ VitalDB missing")
        print("    → Fix: pip install vitaldb")
        all_ok = False
    
    return all_ok


def test_project_structure():
    """Test project directories exist."""
    print("\n[4/5] Checking project structure...")
    
    base_dir = Path(__file__).parent.parent
    
    required_dirs = {
        'src': base_dir / 'src',
        'src/ssl': base_dir / 'src' / 'ssl',
        'src/models': base_dir / 'src' / 'models',
        'src/data': base_dir / 'src' / 'data',
        'configs': base_dir / 'configs',
        'scripts': base_dir / 'scripts',
        'tests': base_dir / 'tests'
    }
    
    all_ok = True
    for name, path in required_dirs.items():
        if path.exists():
            print(f"  ✓ {name}: {path.relative_to(base_dir)}")
        else:
            print(f"  ✗ {name}: {path.relative_to(base_dir)} MISSING")
            all_ok = False
    
    return all_ok


def test_critical_files():
    """Test critical code files exist."""
    print("\n[5/5] Checking critical files...")
    
    base_dir = Path(__file__).parent.parent
    
    critical_files = {
        'SSL masking': base_dir / 'src' / 'ssl' / 'masking.py',
        'SSL objectives': base_dir / 'src' / 'ssl' / 'objectives.py',
        'SSL pretrainer': base_dir / 'src' / 'ssl' / 'pretrainer.py',
        'TTM adapter': base_dir / 'src' / 'models' / 'ttm_adapter.py',
        'Decoders': base_dir / 'src' / 'models' / 'decoders.py',
        'Channel utils': base_dir / 'src' / 'models' / 'channel_utils.py',
        'VitalDB dataset': base_dir / 'src' / 'data' / 'vitaldb_dataset.py',
        'SSL config': base_dir / 'configs' / 'ssl_pretrain.yaml',
        'Pretrain script': base_dir / 'scripts' / 'pretrain_vitaldb_ssl.py',
    }
    
    all_ok = True
    for name, path in critical_files.items():
        if path.exists():
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} MISSING: {path.relative_to(base_dir)}")
            all_ok = False
    
    return all_ok


def main():
    print("="*70)
    print("PHASE 0: ENVIRONMENT SETUP TEST")
    print("="*70)
    print("\nThis test verifies your environment is ready for the research pipeline.")
    
    results = []
    
    results.append(("PyTorch", test_pytorch_installation()))
    results.append(("IBM TTM", test_ibm_ttm_access()))
    results.append(("Biosignal libs", test_biosignal_libraries()))
    results.append(("Project structure", test_project_structure()))
    results.append(("Critical files", test_critical_files()))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    
    for name, ok in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"{status:8} | {name}")
    
    print(f"\nResult: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 Environment is ready!")
        print("\nNext step:")
        print("  → Run Phase 1 data preparation test")
        print("  → python tests/test_phase1_data_prep.py")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        return 1


if __name__ == '__main__':
    exit(main())
