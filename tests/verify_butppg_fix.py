#!/usr/bin/env python3
"""Quick fix verification for BUTPPG loader JSON file handling."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tempfile
import shutil
import numpy as np
import json

print("Testing BUTPPG Loader JSON file handling...")

test_dir = tempfile.mkdtemp()
try:
    from src.data.butppg_loader import BUTPPGLoader
    
    data_dir = Path(test_dir) / "but_ppg"
    data_dir.mkdir()
    
    # Create test signal
    fs = 125.0
    duration = 60.0
    n_samples = int(fs * duration)
    signal = np.sin(2 * np.pi * 1.0 * np.linspace(0, duration, n_samples))
    
    # Save NPY signal
    np.save(data_dir / "subject_windowing.npy", signal)
    
    # Save companion JSON metadata
    with open(data_dir / "subject_windowing.json", 'w') as f:
        json.dump({'fs': fs}, f)
    
    print(f"Created test files in: {data_dir}")
    print(f"  - subject_windowing.npy")
    print(f"  - subject_windowing.json")
    
    # Initialize loader
    loader = BUTPPGLoader(
        data_dir=data_dir,
        fs=125.0,
        window_duration=10.0,
        apply_windowing=True
    )
    
    print(f"\n✓ Loader initialized")
    print(f"  Subjects discovered: {loader.subjects}")
    
    # Verify only one subject (not counting JSON)
    assert len(loader.subjects) == 1, f"Expected 1 subject, got {len(loader.subjects)}"
    assert loader.subjects[0] == "subject_windowing", f"Expected 'subject_windowing', got '{loader.subjects[0]}'"
    print(f"  ✓ JSON files correctly excluded from subject discovery")
    
    # Try to load with windowing
    result = loader.load_subject("subject_windowing", return_windows=True)
    
    assert result is not None, "Failed to load subject"
    windowed_signal, metadata, indices = result
    
    print(f"\n✓ Subject loaded successfully")
    print(f"  Shape: {windowed_signal.shape}")
    print(f"  Expected windows: ~{duration / 10}")
    print(f"  Actual windows: {windowed_signal.shape[0]}")
    print(f"  Metadata fs: {metadata.get('fs')}")
    
    # Verify windowing worked
    assert windowed_signal.ndim == 3, f"Expected 3D array, got {windowed_signal.ndim}D"
    assert windowed_signal.shape[1] == 1250, f"Expected 1250 samples per window, got {windowed_signal.shape[1]}"
    assert windowed_signal.shape[0] > 0, "Expected at least one window"
    
    print(f"\n✅ All checks passed!")
    print(f"  - JSON files excluded from discovery")
    print(f"  - Subject loaded correctly") 
    print(f"  - Windowing working properly")
    
except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    shutil.rmtree(test_dir)

print("\n" + "="*60)
print("✅ BUTPPG Loader JSON handling verified!")
print("="*60)
