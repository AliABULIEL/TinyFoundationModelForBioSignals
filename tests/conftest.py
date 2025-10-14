"""Pytest configuration and fixtures for test suite."""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import json


@pytest.fixture(scope="session")
def data_dir():
    """Create a temporary data directory for testing."""
    temp_dir = tempfile.mkdtemp()
    data_path = Path(temp_dir) / "test_data"
    data_path.mkdir()
    
    # Create sample BUT PPG data for testing
    butppg_dir = data_path / "but_ppg"
    butppg_dir.mkdir()
    
    # Create 3 test subjects with PPG data
    fs = 125.0
    duration = 60.0
    n_samples = int(fs * duration)
    
    for subject_id in range(1, 4):
        # Generate synthetic PPG signal
        t = np.linspace(0, duration, n_samples)
        ppg_signal = np.sin(2 * np.pi * 1.0 * t) + 0.1 * np.random.randn(n_samples)
        
        # Save as NPY
        np.save(butppg_dir / f"subject_{subject_id:03d}.npy", ppg_signal)
        
        # Save companion metadata
        with open(butppg_dir / f"subject_{subject_id:03d}.json", 'w') as f:
            json.dump({'fs': fs, 'subject_id': f"subject_{subject_id:03d}"}, f)
    
    yield str(butppg_dir)
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def butppg_dir(data_dir):
    """Alias for data_dir for compatibility."""
    return data_dir


@pytest.fixture(scope="session")
def vitaldb_dataset():
    """Placeholder for VitalDB dataset (returns None for now)."""
    # This would load VitalDB dataset if available
    # For now, return None to skip VitalDB-specific tests
    return None
