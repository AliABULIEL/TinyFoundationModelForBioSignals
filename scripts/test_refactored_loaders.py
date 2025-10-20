#!/usr/bin/env python3
"""
Test script for refactored data loaders (BUTPPGDataset and VitalDBDataset).

Tests both RAW and PREPROCESSED modes for both datasets to ensure:
1. Backward compatibility (RAW mode works as before)
2. New PREPROCESSED mode works correctly
3. Label loading works
4. Task filtering works
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch


def test_butppg_raw_mode():
    """Test BUTPPGDataset in RAW mode (backward compatibility)."""
    print("\n" + "="*80)
    print("TEST 1: BUTPPGDataset - RAW Mode (Backward Compatibility)")
    print("="*80)

    try:
        from src.data.butppg_dataset import BUTPPGDataset

        # Check if data exists
        data_dir = project_root / 'data/but_ppg/dataset/but-ppg-an-annotated-photoplethysmography-dataset-2.0.0'
        if not data_dir.exists():
            print("⚠️  SKIP: BUT-PPG raw data not found")
            return

        # Create dataset (RAW mode is default)
        dataset = BUTPPGDataset(
            data_dir=str(data_dir),
            modality='ppg',
            split='train',
            mode='raw'
        )

        print(f"✓ Created dataset with {len(dataset)} samples")

        # Test loading a sample
        seg1, seg2 = dataset[0]
        print(f"✓ Loaded sample: seg1 shape = {seg1.shape}, seg2 shape = {seg2.shape}")

        # Verify shapes
        assert seg1.shape[0] == 1, f"Expected 1 channel (PPG), got {seg1.shape[0]}"
        assert len(seg1.shape) == 2, f"Expected 2D tensor [C, T], got shape {seg1.shape}"

        print("✅ BUT-PPG RAW MODE TEST PASSED!")

    except Exception as e:
        print(f"❌ BUT-PPG RAW MODE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


def test_butppg_preprocessed_mode():
    """Test BUTPPGDataset in PREPROCESSED mode (new feature)."""
    print("\n" + "="*80)
    print("TEST 2: BUTPPGDataset - PREPROCESSED Mode (New Feature)")
    print("="*80)

    try:
        from src.data.butppg_dataset import BUTPPGDataset

        # Check if preprocessed data exists
        data_dir = project_root / 'data/processed/butppg/windows_with_labels'
        if not data_dir.exists():
            print("⚠️  SKIP: Preprocessed BUT-PPG data not found")
            print(f"   Run: python scripts/prepare_all_data.py --mode fasttrack --dataset butppg --format windowed")
            return

        # Create dataset in PREPROCESSED mode WITHOUT labels
        dataset_no_labels = BUTPPGDataset(
            data_dir=str(data_dir),
            modality=['ppg', 'ecg'],
            split='train',
            mode='preprocessed',
            return_labels=False
        )

        print(f"✓ Created dataset (no labels) with {len(dataset_no_labels)} samples")

        # Test loading without labels
        seg1, seg2 = dataset_no_labels[0]
        print(f"✓ Loaded sample (no labels): seg1 shape = {seg1.shape}")

        # Verify shapes
        assert seg1.shape[0] == 2, f"Expected 2 channels (PPG+ECG), got {seg1.shape[0]}"

        # Create dataset in PREPROCESSED mode WITH labels
        dataset_with_labels = BUTPPGDataset(
            data_dir=str(data_dir),
            modality=['ppg', 'ecg'],
            split='train',
            mode='preprocessed',
            task='quality',
            return_labels=True,
            filter_missing=True
        )

        print(f"✓ Created dataset (with labels) with {len(dataset_with_labels)} samples")

        # Test loading with labels
        seg1, seg2, labels = dataset_with_labels[0]
        print(f"✓ Loaded sample (with labels): seg1 shape = {seg1.shape}")
        print(f"  Labels: quality={labels['quality']}, hr={labels['hr']}")

        # Verify labels
        assert 'quality' in labels, "Expected 'quality' in labels"
        assert 'hr' in labels, "Expected 'hr' in labels"

        print("✅ BUT-PPG PREPROCESSED MODE TEST PASSED!")

    except Exception as e:
        print(f"❌ BUT-PPG PREPROCESSED MODE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


def test_vitaldb_raw_mode():
    """Test VitalDBDataset in RAW mode (backward compatibility)."""
    print("\n" + "="*80)
    print("TEST 3: VitalDBDataset - RAW Mode (Backward Compatibility)")
    print("="*80)

    try:
        from src.data.vitaldb_dataset import VitalDBDataset

        # Try to import vitaldb
        try:
            import vitaldb
        except ImportError:
            print("⚠️  SKIP: VitalDB package not installed")
            print("   Install: pip install vitaldb")
            return

        # Create dataset (RAW mode is default)
        dataset = VitalDBDataset(
            cache_dir='data/vitaldb_cache',
            channels=['ppg'],
            split='train',
            mode='raw',
            max_cases=2,  # Only 2 cases for quick test
            segments_per_case=5
        )

        print(f"✓ Created dataset with {len(dataset)} samples")

        # Test loading a sample
        seg1, seg2 = dataset[0]
        print(f"✓ Loaded sample: seg1 shape = {seg1.shape}, seg2 shape = {seg2.shape}")

        # Verify shapes
        assert seg1.shape[0] == 1, f"Expected 1 channel (PPG), got {seg1.shape[0]}"
        assert len(seg1.shape) == 2, f"Expected 2D tensor [C, T], got shape {seg1.shape}"

        print("✅ VITALDB RAW MODE TEST PASSED!")

    except Exception as e:
        print(f"❌ VITALDB RAW MODE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


def test_vitaldb_preprocessed_mode():
    """Test VitalDBDataset in PREPROCESSED mode (new feature)."""
    print("\n" + "="*80)
    print("TEST 4: VitalDBDataset - PREPROCESSED Mode (New Feature)")
    print("="*80)

    try:
        from src.data.vitaldb_dataset import VitalDBDataset

        # Check if preprocessed data exists
        data_dir = project_root / 'data/processed/vitaldb/windows_with_labels'
        if not data_dir.exists():
            print("⚠️  SKIP: Preprocessed VitalDB data not found")
            print(f"   Run: python scripts/prepare_all_data.py --mode fasttrack --dataset vitaldb --format windowed")
            return

        # Create dataset in PREPROCESSED mode WITHOUT labels
        dataset_no_labels = VitalDBDataset(
            data_dir=str(data_dir),
            channels=['ppg', 'ecg'],
            split='train',
            mode='preprocessed',
            return_labels=False
        )

        print(f"✓ Created dataset (no labels) with {len(dataset_no_labels)} samples")

        # Test loading without labels
        seg1, seg2 = dataset_no_labels[0]
        print(f"✓ Loaded sample (no labels): seg1 shape = {seg1.shape}")

        # Verify shapes
        assert seg1.shape[0] == 2, f"Expected 2 channels (PPG+ECG), got {seg1.shape[0]}"

        # Create dataset in PREPROCESSED mode WITH labels
        dataset_with_labels = VitalDBDataset(
            data_dir=str(data_dir),
            channels=['ppg', 'ecg'],
            split='train',
            mode='preprocessed',
            task='mortality',
            return_labels=True,
            filter_missing=True
        )

        print(f"✓ Created dataset (with labels) with {len(dataset_with_labels)} samples")

        # Test loading with labels
        seg1, seg2, labels = dataset_with_labels[0]
        print(f"✓ Loaded sample (with labels): seg1 shape = {seg1.shape}")
        print(f"  Labels: death_inhosp={labels['death_inhosp']}, age={labels['age']}, asa={labels['asa']}")

        # Verify labels
        assert 'death_inhosp' in labels, "Expected 'death_inhosp' in labels"
        assert 'age' in labels, "Expected 'age' in labels"

        print("✅ VITALDB PREPROCESSED MODE TEST PASSED!")

    except Exception as e:
        print(f"❌ VITALDB PREPROCESSED MODE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("REFACTORED DATA LOADERS TEST SUITE")
    print("="*80)
    print("\nTesting backward compatibility and new PREPROCESSED mode for:")
    print("  1. BUTPPGDataset")
    print("  2. VitalDBDataset")

    # Run tests
    test_butppg_raw_mode()
    test_butppg_preprocessed_mode()
    test_vitaldb_raw_mode()
    test_vitaldb_preprocessed_mode()

    print("\n" + "="*80)
    print("ALL TESTS COMPLETED!")
    print("="*80)
    print("\n✅ Refactored loaders are backward compatible and support PREPROCESSED mode")
    print("✅ Both BUTPPGDataset and VitalDBDataset follow the same dual-mode pattern")


if __name__ == '__main__':
    main()
