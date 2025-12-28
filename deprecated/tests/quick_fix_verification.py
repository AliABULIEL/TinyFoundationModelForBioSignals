#!/usr/bin/env python3
"""Quick test to verify the fixes."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tempfile
import shutil
import numpy as np
import pandas as pd
from scipy.io import savemat

# Test 1: Label Alignment JSON serialization
print("Test 1: Label Alignment JSON/NPZ fixes...")
from src.data.clinical_labels import ClinicalLabelExtractor
from src.data.label_alignment import LabelWindowAligner

test_dir = tempfile.mkdtemp()
try:
    metadata_path = Path(test_dir) / "test.csv"
    pd.DataFrame({
        'caseid': ['1'],
        'death_inhosp': [0],
        'age': [65]
    }).to_csv(metadata_path, index=False)
    
    extractor = ClinicalLabelExtractor(metadata_path=metadata_path)
    aligner = LabelWindowAligner(extractor, window_duration=10.0)
    
    window_labels = aligner.align_case_windows(case_id="1", n_windows=3)
    
    # Test JSON save
    json_path = Path(test_dir) / "test.json"
    aligner.save_aligned_labels(window_labels, json_path, format="json")
    loaded = aligner.load_aligned_labels(json_path, format="json")
    assert len(loaded) == 3, "JSON save/load failed"
    print("  ✓ JSON serialization fixed")
    
    # Test NPZ save
    npz_path = Path(test_dir) / "test.npz"
    aligner.save_aligned_labels(window_labels, npz_path, format="npz")
    loaded = aligner.load_aligned_labels(npz_path, format="npz")
    assert len(loaded) == 3, "NPZ save/load failed"
    print("  ✓ NPZ loading fixed")
    
finally:
    shutil.rmtree(test_dir)

# Test 2: BUTPPG Loader subject discovery
print("\nTest 2: BUTPPG Loader fixes...")
from src.data.butppg_loader import BUTPPGLoader

test_dir = tempfile.mkdtemp()
try:
    data_dir = Path(test_dir) / "but_ppg"
    data_dir.mkdir()
    
    # Create test signal
    fs = 125.0
    duration = 60.0
    n_samples = int(fs * duration)
    signal = np.sin(2 * np.pi * 1.0 * np.linspace(0, duration, n_samples))
    
    # Save as MAT with metadata
    savemat(str(data_dir / "subject_001.mat"), {'ppg': signal, 'fs': fs})
    
    # Initialize loader
    loader = BUTPPGLoader(data_dir=data_dir, fs=125.0, apply_windowing=False)
    
    # Check subject discovery
    assert len(loader.subjects) > 0, "No subjects discovered"
    print(f"  ✓ Subject discovery fixed ({len(loader.subjects)} subjects found)")
    
    # Load signal
    result = loader.load_subject("subject_001", return_windows=False)
    assert result is not None, "Failed to load subject"
    signal_loaded, metadata = result
    assert signal_loaded.shape[0] == n_samples, f"Signal length mismatch: {signal_loaded.shape[0]} vs {n_samples}"
    print(f"  ✓ Signal loading fixed (no unwanted resampling)")
    
finally:
    shutil.rmtree(test_dir)

print("\n" + "="*60)
print("✅ All quick tests passed! Ready to run full test suite.")
print("="*60)
