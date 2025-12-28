#!/usr/bin/env python3
"""
Quick Test: Generate ONE Sample from Each Dataset

Validates the window generation pipeline by creating a single window from:
1. BUT-PPG (one recording)
2. VitalDB (one case)

Checks:
- Signal synchronization (all channels same length)
- Label embedding
- No NaN/Inf values
- Correct shape and format
- Unified loader compatibility

Usage:
    python scripts/test_one_sample_generation.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import tempfile
import shutil
from typing import Optional

# Import the window processors
import subprocess


def test_butppg_one_sample() -> bool:
    """Generate and validate ONE BUT-PPG window."""
    print("\n" + "="*80)
    print("TEST 1: BUT-PPG Single Sample Generation")
    print("="*80)

    # Create temporary output directory
    temp_dir = Path(tempfile.mkdtemp())
    output_dir = temp_dir / 'butppg_test'

    try:
        # Find BUT-PPG data directory
        butppg_dir = Path('data/but_ppg/dataset/')

        if not butppg_dir.exists():
            print(f"❌ BUT-PPG data not found at: {butppg_dir}")
            print("   Run: python scripts/download_butppg_dataset.py --output-dir data/but_ppg/raw --method zip")
            return False

        print(f"✓ Found BUT-PPG data at: {butppg_dir}")

        # Find first recording
        # BUT-PPG structure: subdirectories with 6-digit IDs
        recording_dirs = sorted(butppg_dir.glob('*'))
        recording_dirs = [d for d in recording_dirs if d.is_dir() and d.name.isdigit()]

        if len(recording_dirs) == 0:
            print(f"❌ No recording directories found")
            return False

        first_recording = recording_dirs[0].name
        print(f"✓ Using recording: {first_recording}")

        # Create minimal splits file with just one recording
        splits_file = temp_dir / 'splits.json'
        import json
        with open(splits_file, 'w') as f:
            json.dump({
                'train': [first_recording],
                'val': [],
                'test': []
            }, f)

        print(f"✓ Created test splits file")

        # Run window generation
        print("\nGenerating window...")
        cmd = [
            sys.executable,
            'scripts/create_butppg_windows_with_labels.py',
            '--data-dir', str(butppg_dir),
            '--output-dir', str(output_dir),
            '--splits-file', str(splits_file),
            '--window-sec', '8.192',
            '--fs', '125',
            '--no-quality-filter'  # Don't filter for testing
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            print(f"❌ Generation failed:")
            print(result.stderr)
            return False

        print(result.stdout)

        # Check output
        train_dir = output_dir / 'train'
        if not train_dir.exists():
            print(f"❌ Output directory not created: {train_dir}")
            return False

        window_files = list(train_dir.glob('window_*.npz'))
        if len(window_files) == 0:
            print(f"❌ No window files created")
            return False

        print(f"✓ Created {len(window_files)} window(s)")

        # Load and validate first window
        print("\nValidating window...")
        window_file = window_files[0]
        data = np.load(window_file)

        # Check required fields
        required_fields = {
            'metadata': ['signal', 'record_id', 'window_idx', 'fs'],
            'clinical_labels': ['quality', 'hr', 'motion', 'bp_systolic', 'bp_diastolic', 'spo2', 'glycaemia'],
            'demographics': ['age', 'sex', 'bmi', 'height', 'weight'],
            'quality_metrics': ['ppg_quality', 'ecg_quality'],
            'normalization': ['ppg_mean', 'ppg_std', 'ecg_mean', 'ecg_std', 'acc_mean', 'acc_std']
        }

        print(f"\nChecking ALL expected fields...")
        all_fields_present = True
        for category, fields in required_fields.items():
            missing = [f for f in fields if f not in data]
            if missing:
                print(f"  ❌ Missing {category}: {missing}")
                all_fields_present = False
            else:
                print(f"  ✓ All {category} present ({len(fields)} fields)")

        if not all_fields_present:
            return False

        print(f"\n✓ ALL {sum(len(fields) for fields in required_fields.values())} FIELDS PRESENT!")

        # Validate signal
        signal = data['signal']
        print(f"\nSignal validation:")
        print(f"  Shape: {signal.shape}")

        if signal.shape != (5, 1024):
            print(f"  ❌ Wrong shape (expected [5, 1024])")
            return False
        print(f"  ✓ Correct shape [5, 1024]")

        if np.any(np.isnan(signal)):
            print(f"  ❌ Contains NaN values")
            return False
        print(f"  ✓ No NaN values")

        if np.any(np.isinf(signal)):
            print(f"  ❌ Contains Inf values")
            return False
        print(f"  ✓ No Inf values")

        # Check channel synchronization (all same length)
        channel_lengths = [signal[i, :].shape[0] for i in range(5)]
        if len(set(channel_lengths)) != 1:
            print(f"  ❌ Channels not synchronized: {channel_lengths}")
            return False
        print(f"  ✓ All channels synchronized (length={channel_lengths[0]})")

        # Validate ALL labels
        print(f"\nLabel validation:")
        print(f"  Clinical labels (7 types):")
        print(f"    quality: {data['quality']}")
        print(f"    hr: {data['hr']:.1f} BPM" if not np.isnan(data['hr']) else f"    hr: {data['hr']}")
        print(f"    motion: {data['motion']}")
        print(f"    bp_systolic: {data['bp_systolic']:.1f} mmHg" if not np.isnan(data['bp_systolic']) else f"    bp_systolic: {data['bp_systolic']}")
        print(f"    bp_diastolic: {data['bp_diastolic']:.1f} mmHg" if not np.isnan(data['bp_diastolic']) else f"    bp_diastolic: {data['bp_diastolic']}")
        print(f"    spo2: {data['spo2']:.1f}%" if not np.isnan(data['spo2']) else f"    spo2: {data['spo2']}")
        print(f"    glycaemia: {data['glycaemia']:.1f} mmol/l" if not np.isnan(data['glycaemia']) else f"    glycaemia: {data['glycaemia']}")

        print(f"  Demographics (5 fields):")
        print(f"    age: {data['age']:.1f}" if not np.isnan(data['age']) else f"    age: {data['age']}")
        print(f"    sex: {data['sex']}")
        print(f"    bmi: {data['bmi']:.1f}" if not np.isnan(data['bmi']) else f"    bmi: {data['bmi']}")
        print(f"    height: {data['height']:.1f} cm" if not np.isnan(data['height']) else f"    height: {data['height']}")
        print(f"    weight: {data['weight']:.1f} kg" if not np.isnan(data['weight']) else f"    weight: {data['weight']}")

        print(f"  Quality metrics:")
        print(f"    ppg_quality: {data['ppg_quality']:.3f}")
        print(f"    ecg_quality: {data['ecg_quality']:.3f}")

        # Count valid (non-missing) labels
        clinical_labels = ['quality', 'hr', 'motion', 'bp_systolic', 'bp_diastolic', 'spo2', 'glycaemia']
        valid_count = sum(1 for label in clinical_labels if not np.isnan(data[label]))
        print(f"\n  ✓ {valid_count}/7 clinical labels have valid values")

        # Test unified loader
        print(f"\nTesting unified loader...")
        from src.data.unified_window_loader import UnifiedWindowDataset

        try:
            dataset = UnifiedWindowDataset(
                data_dir=train_dir,
                task='quality',
                channels=['PPG', 'ECG'],
                filter_missing=False
            )

            if len(dataset) == 0:
                print(f"  ❌ Loader found 0 samples")
                return False

            signal_loaded, label_loaded = dataset[0]
            print(f"  ✓ Loaded sample via UnifiedWindowDataset")
            print(f"    Signal shape: {signal_loaded.shape}")
            print(f"    Label: {label_loaded}")

        except Exception as e:
            print(f"  ❌ Loader failed: {e}")
            return False

        print("\n" + "✅ BUT-PPG TEST PASSED!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"✓ Cleaned up temp directory")


def test_vitaldb_one_sample() -> bool:
    """Generate and validate ONE VitalDB window."""
    print("\n" + "="*80)
    print("TEST 2: VitalDB Single Sample Generation")
    print("="*80)

    # Create temporary output directory
    temp_dir = Path(tempfile.mkdtemp())
    output_dir = temp_dir / 'vitaldb_test'

    try:
        # Check if vitaldb package is available
        try:
            import vitaldb
            print("✓ VitalDB package available")
        except ImportError:
            print("❌ VitalDB package not installed")
            print("   Run: pip install vitaldb")
            return False

        # Create minimal splits file with first case
        print("\nFinding VitalDB case...")

        # Try case 1 (commonly available)
        test_case_id = 1

        splits_file = temp_dir / 'splits.json'
        import json
        with open(splits_file, 'w') as f:
            json.dump({
                'train': [str(test_case_id)],
                'val': [],
                'test': []
            }, f)

        print(f"✓ Using case: {test_case_id}")

        # Run window generation (limit to 1 case)
        print("\nGenerating window...")
        cmd = [
            sys.executable,
            'scripts/create_vitaldb_windows_with_labels.py',
            '--output-dir', str(output_dir),
            '--splits-file', str(splits_file),
            '--window-sec', '8.192',
            '--fs', '125',
            '--max-cases', '1'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            print(f"❌ Generation failed:")
            print(result.stderr)
            return False

        print(result.stdout)

        # Check output
        train_dir = output_dir / 'train'
        if not train_dir.exists():
            print(f"❌ Output directory not created: {train_dir}")
            return False

        window_files = list(train_dir.glob('window_*.npz'))
        if len(window_files) == 0:
            print(f"❌ No window files created (case may not have PPG/ECG data)")
            return False

        print(f"✓ Created {len(window_files)} window(s)")

        # Load and validate first window
        print("\nValidating window...")
        window_file = window_files[0]
        data = np.load(window_file)

        # Check required fields
        required_fields = {
            'metadata': ['signal', 'case_id', 'window_idx', 'fs'],
            'case_labels': ['age', 'sex', 'bmi', 'asa', 'emergency', 'death_inhosp', 'icu_days'],
            'quality_metrics': ['ppg_quality', 'ecg_quality'],
            'normalization': ['ppg_mean', 'ppg_std', 'ecg_mean', 'ecg_std']
        }

        print(f"\nChecking ALL expected fields...")
        all_fields_present = True
        for category, fields in required_fields.items():
            missing = [f for f in fields if f not in data]
            if missing:
                print(f"  ❌ Missing {category}: {missing}")
                all_fields_present = False
            else:
                print(f"  ✓ All {category} present ({len(fields)} fields)")

        if not all_fields_present:
            return False

        print(f"\n✓ ALL {sum(len(fields) for fields in required_fields.values())} FIELDS PRESENT!")

        # Validate signal
        signal = data['signal']
        print(f"\nSignal validation:")
        print(f"  Shape: {signal.shape}")

        if signal.shape != (2, 1024):
            print(f"  ❌ Wrong shape (expected [2, 1024])")
            return False
        print(f"  ✓ Correct shape [2, 1024]")

        if np.any(np.isnan(signal)):
            print(f"  ❌ Contains NaN values")
            return False
        print(f"  ✓ No NaN values")

        if np.any(np.isinf(signal)):
            print(f"  ❌ Contains Inf values")
            return False
        print(f"  ✓ No Inf values")

        # Check channel synchronization
        ppg_len = signal[0, :].shape[0]
        ecg_len = signal[1, :].shape[0]
        if ppg_len != ecg_len:
            print(f"  ❌ PPG and ECG not synchronized: PPG={ppg_len}, ECG={ecg_len}")
            return False
        print(f"  ✓ PPG and ECG synchronized (length={ppg_len})")

        # Validate ALL labels
        print(f"\nLabel validation:")
        print(f"  Case-level labels (7 types):")
        print(f"    case_id: {data['case_id']}")
        print(f"    age: {data['age']:.1f}" if not np.isnan(data['age']) else f"    age: {data['age']}")
        print(f"    sex: {data['sex']}")
        print(f"    bmi: {data['bmi']:.1f}" if not np.isnan(data['bmi']) else f"    bmi: {data['bmi']}")
        print(f"    asa: {int(data['asa'])}" if not np.isnan(data['asa']) else f"    asa: {data['asa']}")
        print(f"    emergency: {bool(data['emergency'])}")
        print(f"    death_inhosp: {bool(data['death_inhosp'])}")
        print(f"    icu_days: {data['icu_days']:.1f}" if not np.isnan(data['icu_days']) else f"    icu_days: {data['icu_days']}")

        print(f"  Quality metrics:")
        print(f"    ppg_quality: {data['ppg_quality']:.3f}")
        print(f"    ecg_quality: {data['ecg_quality']:.3f}")

        # Count valid (non-missing) labels
        case_labels = ['age', 'sex', 'bmi', 'asa', 'emergency', 'death_inhosp', 'icu_days']
        valid_count = sum(1 for label in case_labels if label in data and not (isinstance(data[label], float) and np.isnan(data[label])))
        print(f"\n  ✓ {valid_count}/7 case-level labels have valid values")

        # Test unified loader
        print(f"\nTesting unified loader...")
        from src.data.unified_window_loader import UnifiedWindowDataset

        try:
            dataset = UnifiedWindowDataset(
                data_dir=train_dir,
                task='mortality',
                channels=['PPG', 'ECG'],
                filter_missing=False
            )

            if len(dataset) == 0:
                print(f"  ❌ Loader found 0 samples")
                return False

            signal_loaded, label_loaded = dataset[0]
            print(f"  ✓ Loaded sample via UnifiedWindowDataset")
            print(f"    Signal shape: {signal_loaded.shape}")
            print(f"    Label: {label_loaded}")

        except Exception as e:
            print(f"  ❌ Loader failed: {e}")
            return False

        print("\n" + "✅ VITALDB TEST PASSED!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"✓ Cleaned up temp directory")


def main():
    """Run all tests."""
    print("="*80)
    print("ONE SAMPLE GENERATION TEST SUITE")
    print("="*80)
    print("\nThis test validates the window generation pipeline by creating")
    print("a single window from each dataset and checking correctness.")
    print()

    results = {}

    # Test BUT-PPG
    results['butppg'] = test_butppg_one_sample()

    # Test VitalDB
    results['vitaldb'] = test_vitaldb_one_sample()

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for dataset, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{dataset.upper()}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
