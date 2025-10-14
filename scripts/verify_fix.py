#!/usr/bin/env python3
"""Verify that all imports work after fix."""

import sys
from pathlib import Path

project_root = Path(r"/Users/aliab/Desktop/TinyFoundationModelForBioSignals")
sys.path.insert(0, str(project_root))

print("Testing imports after fix...")
print("-" * 80)

try:
    from src.data.vitaldb_dataset import VitalDBDataset
    print("✓ VitalDBDataset")
except Exception as e:
    print(f"✗ VitalDBDataset: {e}")

try:
    from src.data.butppg_dataset import BUTPPGDataset
    print("✓ BUTPPGDataset")
except Exception as e:
    print(f"✗ BUTPPGDataset: {e}")

try:
    from src.data.butppg_loader import BUTPPGLoader
    print("✓ BUTPPGLoader")
except Exception as e:
    print(f"✗ BUTPPGLoader: {e}")

print("-" * 80)
print("Import test complete!")
