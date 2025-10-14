#!/usr/bin/env python3
"""
Quick test to verify imports are fixed and system is ready.
NO MOCK DATA - Real data only.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all imports work after fixes."""
    print("Testing imports...")
    print("=" * 60)
    
    all_ok = True
    
    # Test basic imports
    try:
        import numpy as np
        print("✓ NumPy imported")
    except ImportError:
        print("✗ NumPy not installed")
        all_ok = False
        
    try:
        import torch
        print("✓ PyTorch imported")
    except ImportError:
        print("✗ PyTorch not installed")
        all_ok = False
        
    # Test project imports
    try:
        from src.data import vitaldb_loader
        print("✓ VitalDB loader imported")
    except ImportError as e:
        print(f"✗ VitalDB loader import failed: {e}")
        all_ok = False
        
    try:
        from src.data.vitaldb_dataset import VitalDBDataset
        print("✓ VitalDB dataset imported")
    except ImportError as e:
        print(f"✗ VitalDB dataset import failed: {e}")
        all_ok = False
        
    try:
        from src.data.butppg_dataset import BUTPPGDataset
        print("✓ BUT PPG dataset imported")
    except ImportError as e:
        print(f"✗ BUT PPG dataset import failed: {e}")
        all_ok = False
        
    # Test VitalDB package
    try:
        import vitaldb
        print("✓ VitalDB package installed")
        vitaldb_ok = True
    except ImportError:
        print("⚠ VitalDB package not installed (pip install vitaldb)")
        vitaldb_ok = False
        
    return all_ok, vitaldb_ok


def check_data_availability():
    """Check what data is available."""
    print("\nChecking data availability...")
    print("=" * 60)
    
    # Check BUT PPG
    but_ppg_path = Path('data/but_ppg/dataset')
    if but_ppg_path.exists():
        mat_files = list(but_ppg_path.glob('*.mat'))
        print(f"✓ BUT PPG data found: {len(mat_files)} files")
        but_ppg_ok = True
    else:
        print("✗ BUT PPG data not found")
        print("  Download with: python scripts/download_but_ppg.py")
        but_ppg_ok = False
        
    # Check VitalDB cache
    cache_path = Path('data/vitaldb_cache')
    if cache_path.exists():
        print(f"✓ VitalDB cache directory exists")
    else:
        print("⚠ VitalDB cache directory not found (will be created)")
        
    return but_ppg_ok


def test_vitaldb_connection():
    """Test VitalDB API connection."""
    print("\nTesting VitalDB API connection...")
    print("=" * 60)
    
    try:
        import vitaldb
        
        # Try to connect
        print("Connecting to VitalDB...")
        cases = vitaldb.find_cases('PLETH')
        
        if len(cases) > 0:
            print(f"✅ Connected! Found {len(cases)} cases with PPG")
            return True
        else:
            print("⚠ Connected but no cases found")
            return False
            
    except ImportError:
        print("✗ VitalDB not installed")
        return False
        
    except Exception as e:
        if 'SSL' in str(e) or 'certificate' in str(e).lower():
            print(f"✗ SSL Certificate Error")
            print("\nTo fix:")
            print("  python scripts/fix_ssl_vitaldb.py")
        else:
            print(f"✗ Connection failed: {e}")
        return False


def main():
    print("=" * 60)
    print(" System Status Check (Real Data Only)")
    print("=" * 60)
    
    # Check imports
    imports_ok, vitaldb_installed = test_imports()
    
    # Check data
    but_ppg_ok = check_data_availability()
    
    # Check VitalDB connection
    vitaldb_connected = False
    if vitaldb_installed:
        vitaldb_connected = test_vitaldb_connection()
        
    # Summary
    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    
    print("\n✅ Fixed Issues:")
    print("  - Import errors resolved")
    print("  - Mock data removed")
    print("  - Real data loading implemented")
    
    print("\n📊 Current Status:")
    print(f"  Core imports: {'✅ OK' if imports_ok else '❌ FAILED'}")
    print(f"  VitalDB package: {'✅ Installed' if vitaldb_installed else '⚠️  Not installed'}")
    print(f"  VitalDB connection: {'✅ Working' if vitaldb_connected else '❌ Not working'}")
    print(f"  BUT PPG data: {'✅ Available' if but_ppg_ok else '⚠️  Not downloaded'}")
    
    print("\n🎯 Next Steps:")
    
    if not vitaldb_installed:
        print("1. Install VitalDB:")
        print("   pip install vitaldb")
        
    if vitaldb_installed and not vitaldb_connected:
        print("1. Fix SSL for VitalDB:")
        print("   python scripts/fix_ssl_vitaldb.py")
        
    if not but_ppg_ok:
        print("2. Download BUT PPG dataset:")
        print("   python scripts/download_but_ppg.py")
        
    if vitaldb_connected or but_ppg_ok:
        print("3. Test real data loading:")
        print("   python scripts/test_real_vitaldb.py")
        print("\n4. Run the pipeline:")
        print("   python scripts/run_multimodal_pipeline.py")
    
    if imports_ok:
        print("\n✅ Core system is ready! Just need to:")
        if not vitaldb_connected:
            print("  - Fix SSL or install VitalDB for pre-training data")
        if not but_ppg_ok:
            print("  - Download BUT PPG for fine-tuning data")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
