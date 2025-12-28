#!/usr/bin/env python3
"""
Phase 1.1b: Deep Weight Verification
=====================================

The initial diagnostic showed:
- ⚠️  First layer weights look random (mean~0, std~0.06)
- ⚠️  Patch dimension mismatch (8 expected, 16 actual)

This script performs deeper verification:
1. Check multiple layers (not just first layer)
2. Compare pretrained vs random initialization
3. Identify if specific layers are pretrained

Author: Claude Code Foundation Model Audit
Date: October 2025
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

# Add project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models.ttm_adapter import TTMAdapter

print("=" * 80)
print("DEEP WEIGHT VERIFICATION: Are TTM weights actually pretrained?")
print("=" * 80)

device = 'cpu'  # Use CPU for consistency

# Test 1: Load model that SHOULD have pretrained weights
print("\n[TEST 1] Loading TTM with context=1024, patch=128 (TTM-Enhanced)")
print("-" * 80)

model_pretrained = TTMAdapter(
    variant='ibm-granite/granite-timeseries-ttm-r1',
    task='ssl',
    input_channels=2,
    context_length=1024,
    patch_size=128,
    freeze_encoder=False,
    use_real_ttm=True
).to(device)

print("\n[TEST 2] Loading SAME config again (should have identical pretrained weights)")
print("-" * 80)

# Load the same config again - if pretrained, weights should be IDENTICAL
model_pretrained2 = TTMAdapter(
    variant='ibm-granite/granite-timeseries-ttm-r1',
    task='ssl',
    input_channels=2,
    context_length=1024,  # Same as TEST 1
    patch_size=128,  # Same as TEST 1
    freeze_encoder=False,
    use_real_ttm=True
).to(device)

print("\n" + "=" * 80)
print("WEIGHT COMPARISON: Two instances of pretrained model")
print("=" * 80)
print("\nBoth models use SAME config (context=1024, patch=128)")
print("If weights are truly pretrained from HuggingFace:")
print("  → They should be IDENTICAL (loaded from same checkpoint)")
print("If weights are random:")
print("  → They would be DIFFERENT (different random seeds)")

# Get all parameters
pretrained_params = list(model_pretrained.backbone.parameters())
pretrained2_params = list(model_pretrained2.backbone.parameters())

print(f"\nNumber of parameters: {len(pretrained_params)}")

# Analyze first 10 layers
print("\n" + "-" * 80)
print("Layer Statistics (first 10 layers)")
print("-" * 80)
print(f"{'Layer':<6} {'Shape':<20} {'Model 1 Mean':<16} {'Model 2 Mean':<16} {'Model 1 Std':<16} {'Model 2 Std':<16}")
print("-" * 80)

for i in range(min(10, len(pretrained_params))):
    p1_param = pretrained_params[i]
    p2_param = pretrained2_params[i]

    p1_mean = p1_param.data.mean().item()
    p1_std = p1_param.data.std().item()
    p2_mean = p2_param.data.mean().item()
    p2_std = p2_param.data.std().item()

    shape_str = str(list(p1_param.shape))

    print(f"{i:<6} {shape_str:<20} {p1_mean:<16.6f} {p2_mean:<16.6f} {p1_std:<16.6f} {p2_std:<16.6f}")

print("\n" + "=" * 80)
print("WEIGHT SIMILARITY TEST")
print("=" * 80)

# Compare first 10 layers - if pretrained, they should be IDENTICAL
# If both random, they would be DIFFERENT
print("\nIf TTM loads pretrained weights correctly:")
print("  - Both models should have IDENTICAL encoder weights")
print("  - Cosine similarity should be ~1.0")
print("  - L2 distance should be ~0.0")
print("\nIf TTM is randomly initialized:")
print("  - Weights would be DIFFERENT between models")
print("  - Cosine similarity would be ~0.0")
print("  - L2 distance would be large")

similarities = []
l2_distances = []

for i in range(min(10, len(pretrained_params))):
    p1_param = pretrained_params[i].data.flatten()
    p2_param = pretrained2_params[i].data.flatten()

    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(p1_param.unsqueeze(0), p2_param.unsqueeze(0)).item()
    similarities.append(cos_sim)

    # L2 distance
    l2_dist = torch.norm(p1_param - p2_param).item()
    l2_distances.append(l2_dist)

print(f"\n{'Layer':<6} {'Cosine Similarity':<20} {'L2 Distance':<20}")
print("-" * 80)
for i in range(len(similarities)):
    print(f"{i:<6} {similarities[i]:<20.6f} {l2_distances[i]:<20.6f}")

avg_similarity = np.mean(similarities)
avg_l2 = np.mean(l2_distances)

print(f"\nAverage cosine similarity: {avg_similarity:.6f}")
print(f"Average L2 distance: {avg_l2:.6f}")

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

# The KEY question: Are the weights identical or different?
# For pretrained weights from HuggingFace, they should be IDENTICAL
# because both models load from the same checkpoint

if avg_similarity > 0.99:
    print("\n✅ PRETRAINED WEIGHTS CONFIRMED!")
    print("   Both models have IDENTICAL weights (cosine similarity > 0.99)")
    print("   This proves weights are loaded from HuggingFace checkpoint")
    print("   The 'random-looking' statistics (mean~0) are just a property of")
    print("   how TTM was trained - NOT an indication of random initialization!")
else:
    print("\n❌ CRITICAL: WEIGHTS ARE RANDOM!")
    print(f"   Cosine similarity: {avg_similarity:.6f} (expected > 0.99)")
    print("   This proves weights are NOT loaded from pretrained checkpoint")
    print("   Each model initialization creates different random weights")

print("\n" + "=" * 80)
print("PATCH SIZE INVESTIGATION")
print("=" * 80)

print("\nConfig says patch_size=128, but encoder outputs 16 patches (not 8)")
print("This suggests TTM internally uses patch_size=64")
print("\nRunning test to verify...")

with torch.no_grad():
    dummy_input = torch.randn(1, 2, 1024).to(device)
    encoder_output = model_pretrained.get_encoder_output(dummy_input)

    print(f"\nInput shape: {list(dummy_input.shape)} [B, C, T]")
    print(f"Output shape: {list(encoder_output.shape)} [B, P, D]")
    print(f"  - Output patches (P): {encoder_output.size(1)}")
    print(f"  - Expected patches: {1024 // 128} = 8")
    print(f"  - Actual patches: {encoder_output.size(1)}")
    print(f"  - Implied patch_size: {1024 // encoder_output.size(1)} = {1024 // encoder_output.size(1)}")

# Check TTM config
if hasattr(model_pretrained, 'backbone') and hasattr(model_pretrained.backbone, 'config'):
    ttm_config = model_pretrained.backbone.config
    print(f"\nTTM backbone config:")
    print(f"  - patch_length: {ttm_config.patch_length}")
    print(f"  - num_patches: {ttm_config.num_patches}")
    print(f"  - context_length: {ttm_config.context_length}")
    print(f"  - d_model: {ttm_config.d_model}")

    if ttm_config.patch_length != 128:
        print(f"\n⚠️  TTM config patch_length ({ttm_config.patch_length}) != our patch_size (128)")
        print("   This is why we get different number of patches!")
        print("   TTM-Enhanced pretrained uses adaptive_patching_levels which changes patch_size")

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

if avg_similarity > 0.99:
    print("\n✅ Pretrained weights are loading correctly!")
    print("   The encoder IS using IBM's pretrained TTM-Enhanced weights")
    print("\n⚠️  However, patch_size mismatch needs attention:")
    print("   - Config specifies patch_size=128")
    print("   - TTM actually uses patch_size=64 (adaptive patching)")
    print("   - This affects SSL masking and decoder reconstruction")
    print("\n📋 Action items:")
    print("   1. ✓ Pretrained weights: WORKING")
    print("   2. ⚠️  Patch size sync: Use encoder's actual patch_size (64, not 128)")
    print("   3. ⚠️  SSL decoder: Must match actual patch_size=64")
else:
    print("\n❌ CRITICAL FIX NEEDED:")
    print("   1. TTM is NOT loading pretrained weights")
    print("   2. Check ttm_adapter.py:_init_real_ttm()")
    print("   3. Verify HuggingFace model ID is correct")
    print("   4. Check if tsfm_public.get_model() is working")

print("=" * 80)
