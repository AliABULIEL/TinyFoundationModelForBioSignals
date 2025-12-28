"""
Verification module - compares results against IBM baselines.
FAILS LOUDLY if results diverge beyond acceptable thresholds.
"""
from typing import Dict
from .config import CONFIG

class VerificationError(Exception):
    """Raised when verification fails."""
    pass


def verify_zero_shot_results(results: Dict[str, float]) -> None:
    """
    Verify zero-shot results against expected baselines.

    IBM reports approximately 0.39 MSE for TTM 512-192 zero-shot on ETTh1.

    Args:
        results: Dictionary containing 'eval_loss' (MSE)

    Raises:
        VerificationError: If results outside acceptable range
    """
    mse = results.get("eval_loss")

    if mse is None:
        raise VerificationError("eval_loss (MSE) not found in results")

    print("\n" + "="*60)
    print("VERIFICATION REPORT")
    print("="*60)
    print(f"Measured MSE: {mse:.6f}")
    print(f"Expected range: [{CONFIG.EXPECTED_MSE_LOWER}, {CONFIG.EXPECTED_MSE_UPPER}]")

    # Check if within expected range
    if CONFIG.EXPECTED_MSE_LOWER <= mse <= CONFIG.EXPECTED_MSE_UPPER:
        status = "PASS"
        print(f"Status: ✅ {status}")
        print("Result is within expected range for TTM zero-shot on ETTh1")
    else:
        status = "FAIL"
        print(f"Status: ❌ {status}")

        if mse < CONFIG.EXPECTED_MSE_LOWER:
            deviation = ((CONFIG.EXPECTED_MSE_LOWER - mse) / CONFIG.EXPECTED_MSE_LOWER) * 100
            msg = f"MSE {mse:.6f} is {deviation:.1f}% BELOW expected lower bound {CONFIG.EXPECTED_MSE_LOWER}"
        else:
            deviation = ((mse - CONFIG.EXPECTED_MSE_UPPER) / CONFIG.EXPECTED_MSE_UPPER) * 100
            msg = f"MSE {mse:.6f} is {deviation:.1f}% ABOVE expected upper bound {CONFIG.EXPECTED_MSE_UPPER}"

        print(f"Deviation: {msg}")
        print("="*60)
        raise VerificationError(msg)

    print("="*60 + "\n")


def generate_verification_summary(results: Dict[str, float]) -> str:
    """
    Generate a human-readable verification summary.
    """
    summary = []
    summary.append("\n" + "#"*60)
    summary.append("# TTM REPRODUCTION VERIFICATION SUMMARY")
    summary.append("#"*60)
    summary.append(f"# Model: {CONFIG.MODEL_PATH} ({CONFIG.MODEL_REVISION})")
    summary.append(f"# Dataset: {CONFIG.TARGET_DATASET}")
    summary.append(f"# Context Length: {CONFIG.CONTEXT_LENGTH}")
    summary.append(f"# Prediction Length: {CONFIG.ROLLING_PREDICTION_LENGTH}")
    summary.append(f"# Batch Size: {CONFIG.BATCH_SIZE}")
    summary.append(f"# Seed: {CONFIG.SEED}")
    summary.append("#"*60)
    summary.append("# RESULTS:")

    for key, value in results.items():
        if isinstance(value, float):
            summary.append(f"#   {key}: {value:.6f}")
        else:
            summary.append(f"#   {key}: {value}")

    summary.append("#"*60)

    return "\n".join(summary)
