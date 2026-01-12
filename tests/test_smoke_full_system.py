"""
Pytest integration for full-system smoke test.

This test runs the complete smoke test via subprocess and validates
that it completes successfully.

Usage:
    # Run smoke test
    pytest tests/test_smoke_full_system.py -v

    # Run with verbose output
    pytest tests/test_smoke_full_system.py -v -s

    # Run with custom markers
    pytest -m smoke tests/test_smoke_full_system.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent


@pytest.mark.slow
@pytest.mark.integration
def test_smoke_full_system_fast():
    """
    Test full system smoke test in fast mode.

    This is a comprehensive integration test that:
    1. Generates synthetic dataset
    2. Runs full training pipeline
    3. Runs full evaluation pipeline
    4. Validates all outputs

    Marked as slow because it runs the entire pipeline (typically < 90s on CPU).
    """
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)

        # Build smoke test command
        smoke_script = PROJECT_ROOT / "scripts" / "smoke_test_full_system.py"

        cmd = [
            sys.executable,
            str(smoke_script),
            "--workdir", str(workdir),
            "--fast",
            # Don't cleanup so we can inspect on failure
        ]

        # Run smoke test
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout for safety
        )

        # Print output for debugging
        print("\n" + "=" * 80)
        print("SMOKE TEST OUTPUT")
        print("=" * 80)
        print(result.stdout)
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        print("=" * 80)

        # Check exit code
        assert result.returncode == 0, (
            f"Smoke test failed with exit code {result.returncode}\n"
            f"See output above for details"
        )

        # Verify results file exists
        results_file = workdir / "smoke_test_results.json"
        assert results_file.exists(), f"Results file not found: {results_file}"

        # Load and validate results
        with open(results_file, "r") as f:
            results = json.load(f)

        # Check status
        assert results["status"] == "passed", (
            f"Smoke test status is '{results['status']}', expected 'passed'\n"
            f"Failures: {results.get('failures', [])}"
        )

        # Check all steps completed
        expected_steps = [
            "generate_dataset",
            "run_training",
            "validate_training_artifacts",
            "run_evaluation",
            "validate_evaluation_outputs",
            "validate_data_integrity",
            "validate_determinism",
        ]
        assert results["steps_completed"] == expected_steps, (
            f"Not all steps completed. "
            f"Expected: {expected_steps}, "
            f"Got: {results['steps_completed']}"
        )

        # Validate key checks
        checks = results["validation_checks"]
        assert checks.get("dataset_created") is True
        assert checks.get("checkpoint_exists") is True
        assert checks.get("metrics_json_exists") is True
        assert checks.get("no_leakage") is True
        assert checks.get("shapes_valid") is True
        assert checks.get("determinism") is True

        # Validate metrics are reasonable (not NaN, in valid range)
        if "metric_accuracy" in checks:
            accuracy = checks["metric_accuracy"]
            assert 0.0 <= accuracy <= 1.0, f"Invalid accuracy: {accuracy}"

        # Check timing is reasonable (< 10 minutes total)
        total_time = results["timing"]["total"]
        assert total_time < 600, f"Smoke test took too long: {total_time:.2f}s"

        print(f"\n✓ All smoke test validations passed in {total_time:.2f}s")


@pytest.mark.slow
@pytest.mark.integration
def test_smoke_synthetic_dataset_generator():
    """
    Test just the synthetic dataset generator in isolation.

    This is a faster sanity check that the generator works correctly.
    """
    from src.testing.synthetic_capture24 import (
        generate_synthetic_capture24,
        _verify_dataset_integrity,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        data_root = Path(tmpdir) / "test_data"

        # Generate small dataset
        participants = ["P001", "P002", "P003"]
        metadata = generate_synthetic_capture24(
            root_dir=data_root,
            participants=participants,
            duration_sec=60,  # 1 minute
            fs=100,
            seed=42,
            num_classes=5,
            verbose=False,
        )

        # Verify metadata
        assert metadata["num_participants"] == 3
        assert metadata["sampling_rate"] == 100
        assert metadata["num_classes"] == 5

        # Verify dataset integrity
        is_valid = _verify_dataset_integrity(data_root, verbose=False)
        assert is_valid, "Dataset integrity check failed"

        # Check files exist
        for p_id in participants:
            p_dir = data_root / p_id
            assert p_dir.exists(), f"Participant directory not found: {p_dir}"

            accel_file = p_dir / "accelerometry.npy"
            labels_file = p_dir / "labels.npy"

            assert accel_file.exists(), f"Missing: {accel_file}"
            assert labels_file.exists(), f"Missing: {labels_file}"

        print(f"✓ Synthetic dataset generator test passed")


@pytest.mark.unit
def test_smoke_config_exists():
    """
    Test that smoke test config file exists and is valid YAML.

    This is a fast sanity check.
    """
    config_path = PROJECT_ROOT / "configs" / "smoke_synthetic.yaml"
    assert config_path.exists(), f"Smoke config not found: {config_path}"

    # Try to load config
    from src.utils.config import load_config

    config = load_config(str(config_path))

    # Verify key sections exist
    assert "experiment" in config
    assert "dataset" in config
    assert "model" in config
    assert "training" in config
    assert "hardware" in config

    # Verify smoke test specific settings
    assert config["training"]["epochs"] <= 2, "Smoke config should have minimal epochs"
    assert config["hardware"]["device"] == "cpu", "Smoke config should use CPU"
    assert config["hardware"]["num_workers"] == 0, "Smoke config should use 0 workers"

    print("✓ Smoke config validation passed")


if __name__ == "__main__":
    # Allow running directly
    pytest.main([__file__, "-v", "-s"])
