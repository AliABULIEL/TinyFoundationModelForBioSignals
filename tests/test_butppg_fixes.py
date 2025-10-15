#!/usr/bin/env python3
"""Test script to verify BUT-PPG loader fixes."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.butppg_loader import BUTPPGLoader

def test_butppg_loader():
    """Test BUT-PPG loader can find subjects."""
    print("=" * 70)
    print("Testing BUT-PPG Loader")
    print("=" * 70)

    data_dir = Path("data/but_ppg/dataset")

    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("   Run: python scripts/download_but_ppg.py")
        return False

    print(f"\n✓ Data directory found: {data_dir}")

    # Test loader initialization
    try:
        loader = BUTPPGLoader(
            data_dir=data_dir,
            fs=125.0,
            window_duration=8.192,  # Match VitalDB config
            window_stride=8.192,
            apply_windowing=False  # Don't apply windowing for discovery test
        )

        print(f"\n✓ Loader initialized successfully")
        print(f"  Subjects found: {len(loader.subjects)}")

        if len(loader.subjects) == 0:
            print("\n❌ No subjects found!")
            return False

        if len(loader.subjects) < 100:
            print(f"\n⚠️  Only {len(loader.subjects)} subjects found (expected ~3,888)")
            print(f"  First 5: {loader.subjects[:5]}")
        else:
            print(f"\n✓ Found {len(loader.subjects)} subjects")
            print(f"  First 10: {loader.subjects[:10]}")
            print(f"  Last 10: {loader.subjects[-10:]}")

        # Test loading a sample subject
        if loader.subjects:
            test_subject = loader.subjects[0]
            print(f"\n✓ Testing load for subject: {test_subject}")

            try:
                result = loader.load_subject(test_subject, signal_type='ppg', return_windows=False)

                if result is None:
                    print(f"  ⚠️  Could not load subject {test_subject}")
                else:
                    signal, metadata = result
                    print(f"  ✓ Loaded successfully")
                    print(f"    Signal shape: {signal.shape}")
                    print(f"    Sampling rate: {metadata.get('fs')} Hz")
                    print(f"    Duration: {len(signal) / metadata.get('fs', 64):.1f} seconds")

            except Exception as e:
                print(f"  ❌ Error loading subject: {e}")
                import traceback
                traceback.print_exc()

        print("\n" + "=" * 70)
        print("✅ BUT-PPG Loader Test Complete")
        print("=" * 70)

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_butppg_loader()
    sys.exit(0 if success else 1)
