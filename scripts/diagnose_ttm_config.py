#!/usr/bin/env python3
"""Diagnostic script to check IBM TTM actual configuration.

This script loads IBM TTM directly and prints its actual runtime configuration,
helping diagnose architecture mismatches.

Usage:
    python scripts/diagnose_ttm_config.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings('ignore')

def diagnose_ttm():
    """Load and inspect IBM TTM configuration."""

    print("="*80)
    print("IBM TTM CONFIGURATION DIAGNOSTIC")
    print("="*80)

    # Check if tsfm_public is available
    try:
        from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
        print("✓ tsfm_public library is available")
    except ImportError as e:
        print(f"✗ tsfm_public not available: {e}")
        print("\nInstall with: pip install tsfm[notebooks]")
        return False

    print("\nLoading IBM TTM variant: ibm-granite/granite-timeseries-ttm-r1")
    print("Context length: 1024")
    print("-" * 80)

    try:
        # Load TTM with context_length=1024
        # Use from_pretrained directly (correct API)
        ttm_model = TinyTimeMixerForPrediction.from_pretrained(
            'ibm-granite/granite-timeseries-ttm-r1',
            context_length=1024,
            prediction_length=96  # Standard
        )

        print("✓ TTM loaded successfully!")

        # Get config
        config = ttm_model.backbone.config

        print("\n" + "="*80)
        print("TTM ACTUAL CONFIGURATION")
        print("="*80)
        print(f"  context_length:        {config.context_length}")
        print(f"  prediction_length:     {config.prediction_length}")
        # Try different attribute names (patch_len vs patch_length)
        patch_size = getattr(config, 'patch_len', getattr(config, 'patch_length', None))
        print(f"  patch_length:          {patch_size}")
        print(f"  num_patches:           {config.num_patches}")
        print(f"  d_model:               {config.d_model}")
        print(f"  num_input_channels:    {config.num_input_channels}")
        print(f"  num_layers:            {config.num_layers}")
        print(f"  expansion_factor:      {config.expansion_factor}")
        print(f"  dropout:               {config.dropout}")
        print(f"  mode:                  {config.mode}")
        print(f"  scaling:               {config.scaling}")
        print("="*80)

        # Validate expected values for TTM-Enhanced
        print("\nVALIDATION:")

        expected_patch_len = 64
        expected_num_patches = 8  # IBM TTM creates 8 patches (with striding/downsampling)
        expected_d_model = 192

        patch_ok = patch_size == expected_patch_len
        patches_ok = config.num_patches == expected_num_patches
        d_model_ok = config.d_model == expected_d_model

        print(f"  patch_length = {patch_size} (expected {expected_patch_len}): {'✓' if patch_ok else '✗'}")
        print(f"  num_patches = {config.num_patches} (expected {expected_num_patches}): {'✓' if patches_ok else '✗'}")
        print(f"  d_model = {config.d_model} (expected {expected_d_model}): {'✓' if d_model_ok else '✗'}")

        if patch_ok and patches_ok and d_model_ok:
            print("\n" + "="*80)
            print("✅ TTM-Enhanced configuration is CORRECT!")
            print("="*80)
            print("\nYou should use these values in your training scripts:")
            print(f"  --ibm-context-length 1024")
            print(f"  --ibm-patch-size {patch_size}")
            print(f"  Expected d_model: {config.d_model}")
            print(f"  Expected num_patches: {config.num_patches}")
            return True
        else:
            print("\n" + "="*80)
            print("⚠️  TTM configuration differs from expected TTM-Enhanced values!")
            print("="*80)

            if not patch_ok:
                print(f"\n  Patch size mismatch:")
                print(f"    Expected: {expected_patch_len}")
                print(f"    Actual: {patch_size}")
                print(f"    Impact: Will create {1024 // patch_size if patch_size else 'unknown'} patches instead of {expected_num_patches}")

            if not patches_ok:
                print(f"\n  Patch count mismatch:")
                print(f"    Expected: {expected_num_patches}")
                print(f"    Actual: {config.num_patches}")
                print(f"    Impact: Encoder/decoder dimension mismatch!")

            if not d_model_ok:
                print(f"\n  d_model mismatch:")
                print(f"    Expected: {expected_d_model}")
                print(f"    Actual: {config.d_model}")
                print(f"    Impact: Feature dimension will differ!")

            print("\nRECOMMENDATION:")
            print("  Use the ACTUAL values from TTM in your configuration:")
            print(f"    patch_size = {patch_size}")
            print(f"    d_model = {config.d_model}")
            print(f"    num_patches = {config.num_patches}")

            return False

    except Exception as e:
        print(f"\n✗ Failed to load TTM: {e}")
        print("\nPossible causes:")
        print("  1. HuggingFace Hub connectivity issue")
        print("  2. Model not cached locally")
        print("  3. Insufficient memory")
        print("\nTry:")
        print("  huggingface-cli login")
        print("  Or set TRANSFORMERS_OFFLINE=1 if model is cached")
        return False


if __name__ == '__main__':
    print("\nThis script checks what configuration IBM TTM actually uses.")
    print("Run this BEFORE training to verify your configuration is correct.\n")

    success = diagnose_ttm()

    if success:
        print("\n" + "="*80)
        print("DIAGNOSIS COMPLETE: TTM configuration is correct! ✓")
        print("="*80)
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("DIAGNOSIS FAILED: Configuration issues detected")
        print("="*80)
        sys.exit(1)
