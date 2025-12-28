"""
Main entry point for TTM reproduction.
Run this script to verify IBM's TTM results can be reproduced.
"""
import sys
import torch
import os
from transformers import set_seed

# Add parent directory to path for imports
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, repo_root)

from src.ttm_reproduction.config import CONFIG
from src.ttm_reproduction.data_loader import load_etth1_dataset, verify_data_statistics
from src.ttm_reproduction.model_wrapper import load_ttm_model, load_rolling_predictor
from src.ttm_reproduction.evaluation import run_zero_shot_evaluation
from src.ttm_reproduction.verification import verify_zero_shot_results, generate_verification_summary, VerificationError


def main():
    """
    Main reproduction pipeline.

    Exit codes:
        0: Verification passed
        1: Verification failed
        2: Runtime error
    """
    print("\n" + "="*60)
    print("TTM REPRODUCTION PIPELINE")
    print("="*60)
    print(f"Device: {CONFIG.DEVICE}")
    print(f"Seed: {CONFIG.SEED}")
    print("="*60 + "\n")

    try:
        # Step 1: Set seed for reproducibility
        print("[STEP 1/5] Setting random seed...")
        set_seed(CONFIG.SEED)
        torch.manual_seed(CONFIG.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(CONFIG.SEED)

        # Step 2: Load data
        print("\n[STEP 2/5] Loading ETTh1 dataset...")
        _, _, test_dataset = load_etth1_dataset()
        verify_data_statistics(test_dataset)

        # Step 3: Load model
        print("\n[STEP 3/5] Loading TTM model...")
        base_model = load_ttm_model()

        # Step 4: Create rolling predictor
        print("\n[STEP 4/5] Creating rolling predictor...")
        rolling_model = load_rolling_predictor(base_model)

        # Step 5: Run evaluation
        print("\n[STEP 5/5] Running zero-shot evaluation...")
        results = run_zero_shot_evaluation(rolling_model, test_dataset)

        # Print summary
        summary = generate_verification_summary(results)
        print(summary)

        # Verify results
        verify_zero_shot_results(results)

        print("\n✅ REPRODUCTION SUCCESSFUL")
        print("TTM results have been verified against IBM baselines.")
        return 0

    except VerificationError as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        print("Results diverge from expected IBM baselines.")
        return 1

    except Exception as e:
        print(f"\n💥 RUNTIME ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
