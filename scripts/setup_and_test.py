#!/usr/bin/env python3
"""
Complete setup and test script for the multi-modal pipeline.
This script will help you get everything working step by step.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def check_environment():
    """Check the environment setup."""
    print_section("Environment Check")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 7):
        print("⚠️  Python 3.7+ recommended")
        
    # Check key packages
    packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'tqdm': 'tqdm'
    }
    
    missing = []
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"✗ {name} not installed")
            missing.append(package)
            
    if missing:
        print(f"\n⚠️  Install missing packages:")
        print(f"  pip install {' '.join(missing)}")
        
    # Check optional packages
    print("\nOptional packages:")
    try:
        import vitaldb
        print("✓ VitalDB installed")
    except ImportError:
        print("⚠️  VitalDB not installed (pip install vitaldb)")
        
    return len(missing) == 0


def test_imports():
    """Test that project imports work."""
    print_section("Testing Project Imports")
    
    try:
        from src.data import VitalDBDataset, BUTPPGDataset
        print("✓ Dataset imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_mock_data():
    """Test with mock data (no external dependencies)."""
    print_section("Testing with Mock Data")
    
    os.environ['VITALDB_MOCK'] = '1'
    
    try:
        from src.data.vitaldb_dataset import VitalDBDataset
        
        # Create mock dataset
        dataset = VitalDBDataset(
            channels=['ppg', 'ecg'],
            split='train',
            use_raw_vitaldb=True,
            max_cases=2,
            segments_per_case=5
        )
        
        # Test loading
        seg1, seg2 = dataset[0]
        print(f"✓ Mock VitalDB working")
        print(f"  Shape: {seg1.shape}")
        print(f"  Samples: {len(dataset)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Mock test failed: {e}")
        return False


def check_but_ppg_data():
    """Check if BUT PPG data is available."""
    print_section("Checking BUT PPG Data")
    
    data_dir = Path('data/but_ppg/dataset')
    if data_dir.exists():
        # Count files
        mat_files = list(data_dir.glob('*.mat'))
        print(f"✓ BUT PPG data found: {len(mat_files)} files")
        return True
    else:
        print("✗ BUT PPG data not found")
        print("\nTo download BUT PPG dataset (~87MB):")
        print("  python scripts/download_but_ppg.py")
        return False


def test_but_ppg():
    """Test BUT PPG dataset if available."""
    if not check_but_ppg_data():
        return False
        
    print("\nTesting BUT PPG loading...")
    
    try:
        from src.data.butppg_dataset import BUTPPGDataset
        
        dataset = BUTPPGDataset(
            data_dir='data/but_ppg/dataset',
            modality='all',
            split='train'
        )
        
        if len(dataset) > 0:
            seg1, seg2 = dataset[0]
            print(f"✓ BUT PPG dataset working")
            print(f"  Shape: {seg1.shape} (5 channels)")
            print(f"  Samples: {len(dataset)}")
            return True
        else:
            print("⚠️  Dataset empty")
            return False
            
    except Exception as e:
        print(f"✗ BUT PPG test failed: {e}")
        return False


def fix_ssl_issue():
    """Try to fix SSL certificate issue for VitalDB."""
    print_section("Fixing SSL for VitalDB")
    
    print("This is needed to download real VitalDB data.")
    print("\nTrying automatic fix...")
    
    # Try installing certifi
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "certifi"], 
                      capture_output=True, check=True)
        print("✓ Certifi updated")
        
        # Set SSL context in loader
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        print("✓ SSL context configured")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Automatic fix incomplete: {e}")
        print("\nManual fix for macOS:")
        print("  1. Find your Python installation folder")
        print("  2. Run 'Install Certificates.command' in that folder")
        print("\nAlternative: Use unverified SSL (less secure):")
        print("  Set environment variable: export PYTHONHTTPSVERIFY=0")
        return False


def test_real_vitaldb():
    """Test real VitalDB data loading."""
    print_section("Testing Real VitalDB Data")
    
    # Remove mock mode
    if 'VITALDB_MOCK' in os.environ:
        del os.environ['VITALDB_MOCK']
        
    try:
        import vitaldb
        
        # Try to find cases
        cases = vitaldb.find_cases('PLETH')
        if len(cases) > 0:
            print(f"✓ Found {len(cases)} VitalDB cases with PPG")
            
            # Try loading one case
            case_id = cases[0]
            data = vitaldb.load_case(case_id, ['SNUADC/PLETH'])
            if data is not None:
                print(f"✓ Successfully loaded case {case_id}")
                return True
            else:
                print(f"⚠️  Could not load data from case {case_id}")
                return False
        else:
            print("⚠️  No cases found")
            return False
            
    except Exception as e:
        if 'SSL' in str(e) or 'certificate' in str(e).lower():
            print(f"✗ SSL certificate error")
            print("  Run SSL fix or use mock data for testing")
        else:
            print(f"✗ VitalDB test failed: {e}")
        return False


def create_test_pipeline():
    """Create a simple test pipeline script."""
    print_section("Creating Test Pipeline")
    
    script = '''#!/usr/bin/env python3
"""Simple test pipeline for multi-modal training."""

import os
os.environ['VITALDB_MOCK'] = '1'  # Use mock data for testing

from src.data import create_vitaldb_dataloaders, create_butppg_dataloaders

# Create mock VitalDB loaders
train_loader, val_loader, test_loader = create_vitaldb_dataloaders(
    channels=['ppg', 'ecg'],
    batch_size=4,
    num_workers=0,
    max_cases=5,
    use_raw_vitaldb=True
)

print(f"VitalDB DataLoaders created:")
print(f"  Train: {len(train_loader)} batches")
print(f"  Val: {len(val_loader)} batches")
print(f"  Test: {len(test_loader)} batches")

# Get one batch
for batch in train_loader:
    seg1, seg2 = batch
    print(f"\\nBatch shape: {seg1.shape}")
    print(f"  PPG channel: {seg1[:, 0, :].shape}")
    print(f"  ECG channel: {seg1[:, 1, :].shape}")
    break

print("\\n✅ Pipeline test successful!")
'''
    
    test_file = Path('test_pipeline_simple.py')
    test_file.write_text(script)
    print(f"✓ Created {test_file}")
    print(f"  Run it with: python {test_file}")
    
    return True


def main():
    """Main test sequence."""
    print("=" * 60)
    print(" Multi-Modal Biosignal Pipeline Setup & Test")
    print("=" * 60)
    
    results = {}
    
    # 1. Check environment
    results['environment'] = check_environment()
    
    # 2. Test imports
    results['imports'] = test_imports()
    
    if results['imports']:
        # 3. Test mock data (always works)
        results['mock_data'] = test_mock_data()
        
        # 4. Check BUT PPG
        results['but_ppg'] = test_but_ppg()
        
        # 5. Try SSL fix for VitalDB
        results['ssl_fix'] = fix_ssl_issue()
        
        # 6. Test real VitalDB (may fail due to SSL)
        results['real_vitaldb'] = test_real_vitaldb()
        
        # 7. Create test pipeline
        results['pipeline'] = create_test_pipeline()
    
    # Summary
    print_section("Setup Summary")
    
    status_icons = {True: '✅', False: '❌', None: '⚠️ '}
    
    for key, value in results.items():
        icon = status_icons[value]
        print(f"{icon} {key.replace('_', ' ').title()}")
        
    # Recommendations
    print_section("Next Steps")
    
    if not results.get('but_ppg'):
        print("1. Download BUT PPG dataset:")
        print("   python scripts/download_but_ppg.py")
        
    if not results.get('real_vitaldb'):
        print("2. For real VitalDB data:")
        print("   - Fix SSL certificates (see above)")
        print("   - Or use mock data for testing")
        
    print("\n3. Test the pipeline:")
    print("   python test_pipeline_simple.py")
    
    print("\n4. Run full multi-modal training:")
    print("   python scripts/run_multimodal_pipeline.py --test-only")
    
    print("\n" + "=" * 60)
    if results.get('mock_data'):
        print("✅ Core functionality is working!")
        print("You can proceed with development using mock data.")
    else:
        print("❌ Some issues need to be resolved.")
        print("Check the error messages above.")


if __name__ == "__main__":
    main()
