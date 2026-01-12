"""
Synthetic CAPTURE-24 dataset generator for smoke testing.

Generates tiny synthetic accelerometry datasets that match the exact schema
expected by CAPTURE24Dataset for full end-to-end smoke testing.
"""

import numpy as np
from pathlib import Path
from typing import List, Optional
import json


def generate_synthetic_capture24(
    root_dir: Path,
    participants: List[str],
    duration_sec: int,
    fs: int = 100,
    seed: int = 123,
    num_classes: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Generate a synthetic CAPTURE-24 dataset on disk.

    Creates participant folders with accelerometry.npy and labels.npy files
    matching the exact schema expected by CAPTURE24Dataset.

    Args:
        root_dir: Root directory to create dataset in
        participants: List of participant IDs (e.g., ["P001", "P002"])
        duration_sec: Duration of data per participant in seconds
        fs: Sampling frequency in Hz (default: 100, original CAPTURE-24 rate)
        seed: Random seed for reproducibility
        num_classes: Number of activity classes (default: 5)
        verbose: Print progress messages

    Returns:
        Dictionary with dataset metadata

    Generated Structure:
        root_dir/
        ├── P001/
        │   ├── accelerometry.npy    # (N, 3) float32
        │   └── labels.npy            # (N,) int64
        ├── P002/
        │   ├── accelerometry.npy
        │   └── labels.npy
        └── ...

    File Formats:
        - accelerometry.npy: shape (num_samples, 3), dtype=float32
          3 channels: X, Y, Z acceleration in g (±8g range)
        - labels.npy: shape (num_samples,), dtype=int64
          Integer labels in range [0, num_classes-1]
    """
    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    # Set seed for reproducibility
    rng = np.random.RandomState(seed)

    num_samples = duration_sec * fs

    if verbose:
        print(f"Generating synthetic CAPTURE-24 dataset:")
        print(f"  Root: {root_dir}")
        print(f"  Participants: {len(participants)}")
        print(f"  Duration: {duration_sec}s @ {fs}Hz")
        print(f"  Samples/participant: {num_samples}")
        print(f"  Classes: {num_classes}")

    metadata = {
        "dataset_name": "SYNTHETIC-CAPTURE24",
        "sampling_rate": fs,
        "num_channels": 3,
        "num_participants": len(participants),
        "num_classes": num_classes,
        "duration_per_participant_sec": duration_sec,
        "num_samples_per_participant": num_samples,
        "seed": seed,
        "participants": participants,
    }

    for idx, participant_id in enumerate(participants):
        participant_dir = root_dir / participant_id
        participant_dir.mkdir(exist_ok=True)

        # Generate realistic accelerometry signal with multiple components
        accelerometry = _generate_realistic_accelerometry(
            num_samples=num_samples,
            fs=fs,
            rng=rng,
            participant_idx=idx,
        )

        # Generate labels ensuring all classes appear across dataset
        labels = _generate_realistic_labels(
            num_samples=num_samples,
            num_classes=num_classes,
            rng=rng,
            participant_idx=idx,
        )

        # Save as numpy arrays (preferred format)
        accel_path = participant_dir / "accelerometry.npy"
        labels_path = participant_dir / "labels.npy"

        np.save(accel_path, accelerometry.astype(np.float32))
        np.save(labels_path, labels.astype(np.int64))

        if verbose:
            print(f"  ✓ {participant_id}: "
                  f"accel={accelerometry.shape} {accelerometry.dtype}, "
                  f"labels={labels.shape} {labels.dtype}, "
                  f"classes={sorted(np.unique(labels))}")

    # Save metadata
    metadata_path = root_dir / "synthetic_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"\n✓ Dataset generated at: {root_dir}")
        print(f"  Metadata: {metadata_path}")

    return metadata


def _generate_realistic_accelerometry(
    num_samples: int,
    fs: int,
    rng: np.random.RandomState,
    participant_idx: int,
) -> np.ndarray:
    """
    Generate realistic 3-channel accelerometry data.

    Simulates wrist-worn accelerometer with:
    - Gravity component (dominant on Z-axis when stationary)
    - Movement oscillations (walking ~1-2 Hz, arm swing)
    - Higher frequency components (impacts, transitions)
    - Gaussian noise

    Args:
        num_samples: Number of time samples
        fs: Sampling frequency
        rng: Random number generator
        participant_idx: Participant index for variation

    Returns:
        Array of shape (num_samples, 3) with X, Y, Z acceleration in g
    """
    t = np.arange(num_samples) / fs

    # Initialize with gravity + small random offset
    # Typical wrist orientation: Z ~ 1g (vertical), X/Y ~ 0
    gravity_offset = rng.uniform(-0.2, 0.2, size=3)
    gravity = np.array([gravity_offset[0], gravity_offset[1], 1.0 + gravity_offset[2]])

    signal = np.tile(gravity, (num_samples, 1))

    # Add movement components (vary by participant for diversity)
    phase_offset = participant_idx * 0.5

    # 1. Walking cadence ~1.5 Hz (typical step frequency)
    walk_freq = 1.5 + rng.uniform(-0.3, 0.3)
    walk_amplitude = 0.5 + rng.uniform(0, 0.3)
    signal[:, 0] += walk_amplitude * np.sin(2 * np.pi * walk_freq * t + phase_offset)
    signal[:, 1] += walk_amplitude * 0.7 * np.cos(2 * np.pi * walk_freq * t + phase_offset)
    signal[:, 2] += walk_amplitude * 0.5 * np.sin(2 * np.pi * walk_freq * t + phase_offset + np.pi/4)

    # 2. Arm swing harmonic ~3 Hz
    swing_freq = 3.0 + rng.uniform(-0.5, 0.5)
    swing_amplitude = 0.3
    signal[:, 0] += swing_amplitude * np.sin(2 * np.pi * swing_freq * t + phase_offset * 2)
    signal[:, 1] += swing_amplitude * 0.5 * np.cos(2 * np.pi * swing_freq * t + phase_offset * 2)

    # 3. Higher frequency components (sudden movements) ~8-12 Hz
    high_freq = 10.0 + rng.uniform(-2, 2)
    high_amplitude = 0.15
    signal[:, 0] += high_amplitude * np.sin(2 * np.pi * high_freq * t + phase_offset * 3)
    signal[:, 2] += high_amplitude * 0.5 * np.cos(2 * np.pi * high_freq * t + phase_offset * 3)

    # 4. Add realistic Gaussian noise
    noise_std = 0.05  # Typical sensor noise
    noise = rng.normal(0, noise_std, size=(num_samples, 3))
    signal += noise

    # 5. Add occasional "events" (high-intensity bursts)
    num_events = rng.randint(5, 15)
    for _ in range(num_events):
        event_start = rng.randint(0, num_samples - fs)
        event_duration = rng.randint(int(0.5 * fs), int(2 * fs))
        event_end = min(event_start + event_duration, num_samples)
        event_amplitude = rng.uniform(1.0, 3.0)
        event_signal = event_amplitude * rng.randn(event_end - event_start, 3)
        signal[event_start:event_end] += event_signal

    # Clip to realistic ±8g range (typical accelerometer range)
    signal = np.clip(signal, -8.0, 8.0)

    return signal


def _generate_realistic_labels(
    num_samples: int,
    num_classes: int,
    rng: np.random.RandomState,
    participant_idx: int,
) -> np.ndarray:
    """
    Generate realistic activity labels with temporal coherence.

    Activities don't change every sample - they persist for "bouts"
    (e.g., walking for 30 seconds, then sedentary for 2 minutes).

    CAPTURE-24 5-class taxonomy:
        0: Sleep
        1: Sedentary (sitting, standing still)
        2: Light activity (slow walking, household tasks)
        3: Moderate activity (brisk walking, climbing stairs)
        4: Vigorous activity (running, sports)

    Args:
        num_samples: Number of time samples
        num_classes: Number of activity classes (typically 5)
        rng: Random number generator
        participant_idx: Participant index for variation

    Returns:
        Array of shape (num_samples,) with integer labels in [0, num_classes-1]
    """
    labels = np.zeros(num_samples, dtype=np.int64)

    # Generate activity bouts with realistic durations
    current_pos = 0

    # Ensure all participants have diverse activities
    # Participant 0 might favor sleep/sedentary, participant 1 more active, etc.
    class_bias = participant_idx % num_classes

    while current_pos < num_samples:
        # Sample an activity class with bias toward certain classes
        if rng.rand() < 0.3:
            # Biased toward participant-specific class
            activity_class = class_bias
        else:
            # Random class (but ensure variety)
            activity_class = rng.randint(0, num_classes)

        # Activity bout duration (in samples)
        # Sleep: 30-120 sec, Sedentary: 30-90 sec, Active: 10-60 sec
        if activity_class == 0:  # Sleep
            bout_duration = rng.randint(1000, 3000)  # 10-30 sec (reduced for smoke test)
        elif activity_class == 1:  # Sedentary
            bout_duration = rng.randint(800, 2000)   # 8-20 sec
        else:  # Active classes
            bout_duration = rng.randint(500, 1500)   # 5-15 sec

        bout_end = min(current_pos + bout_duration, num_samples)
        labels[current_pos:bout_end] = activity_class
        current_pos = bout_end

    return labels


def _verify_dataset_integrity(root_dir: Path, verbose: bool = True) -> bool:
    """
    Verify generated dataset matches CAPTURE24Dataset requirements.

    Args:
        root_dir: Root directory of generated dataset
        verbose: Print verification messages

    Returns:
        True if dataset is valid, False otherwise
    """
    root_dir = Path(root_dir)

    if not root_dir.exists():
        if verbose:
            print(f"✗ Dataset directory does not exist: {root_dir}")
        return False

    # Find all participant folders
    participant_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir()])

    if len(participant_dirs) == 0:
        if verbose:
            print(f"✗ No participant directories found in {root_dir}")
        return False

    if verbose:
        print(f"\nVerifying dataset integrity:")
        print(f"  Found {len(participant_dirs)} participants")

    all_classes = set()

    for p_dir in participant_dirs:
        # Check required files
        accel_path = p_dir / "accelerometry.npy"
        labels_path = p_dir / "labels.npy"

        if not accel_path.exists():
            if verbose:
                print(f"  ✗ {p_dir.name}: missing accelerometry.npy")
            return False

        if not labels_path.exists():
            if verbose:
                print(f"  ✗ {p_dir.name}: missing labels.npy")
            return False

        # Load and verify shapes
        accel = np.load(accel_path)
        labels = np.load(labels_path)

        if accel.ndim != 2 or accel.shape[1] != 3:
            if verbose:
                print(f"  ✗ {p_dir.name}: accelerometry shape {accel.shape}, expected (N, 3)")
            return False

        if labels.ndim != 1:
            if verbose:
                print(f"  ✗ {p_dir.name}: labels shape {labels.shape}, expected (N,)")
            return False

        if accel.shape[0] != labels.shape[0]:
            if verbose:
                print(f"  ✗ {p_dir.name}: length mismatch - "
                      f"accel={accel.shape[0]}, labels={labels.shape[0]}")
            return False

        # Check label range
        unique_labels = np.unique(labels)
        all_classes.update(unique_labels.tolist())

        if np.any(labels < 0):
            if verbose:
                print(f"  ✗ {p_dir.name}: negative labels found")
            return False

    if verbose:
        print(f"  ✓ All files present and valid")
        print(f"  ✓ Classes represented: {sorted(all_classes)}")
        print(f"  ✓ Dataset integrity verified")

    return True


if __name__ == "__main__":
    # Quick test
    import tempfile
    import shutil

    with tempfile.TemporaryDirectory() as tmpdir:
        test_root = Path(tmpdir) / "test_dataset"

        metadata = generate_synthetic_capture24(
            root_dir=test_root,
            participants=["P001", "P002", "P003"],
            duration_sec=60,
            fs=100,
            seed=42,
            verbose=True,
        )

        print("\nGenerated metadata:")
        print(json.dumps(metadata, indent=2))

        print("\nVerifying integrity:")
        is_valid = _verify_dataset_integrity(test_root, verbose=True)

        if is_valid:
            print("\n✓ Synthetic dataset generator working correctly!")
        else:
            print("\n✗ Dataset verification failed")
            exit(1)
