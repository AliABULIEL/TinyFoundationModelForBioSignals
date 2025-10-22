#!/usr/bin/env python3
"""
Phase 1.1: IBM TTM Baseline Architecture Diagnostic
====================================================

This script verifies:
1. IBM TTM loads pretrained weights (not random initialization)
2. Encoder output shape is correct [B, P, D]
3. Dimensions match TTM-Enhanced configuration
4. Weights are non-random (pretrained)

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
print("PHASE 1.1: IBM TTM BASELINE ARCHITECTURE DIAGNOSTIC")
print("=" * 80)

# Test configuration - matching SSL pretraining
CONFIG = {
    'variant': 'ibm-granite/granite-timeseries-ttm-r1',
    'task': 'ssl',
    'input_channels': 2,  # PPG + ECG
    'context_length': 1024,  # Should match TTM-Enhanced
    'patch_size': 128,  # Should match TTM-Enhanced
    'freeze_encoder': False,
    'use_real_ttm': True
}

EXPECTED_CONFIG = {
    'context_length': 1024,
    'patch_size': 128,
    'n_patches': 8,  # 1024 / 128
    'hidden_dim': 192,  # TTM-Enhanced d_model
    'input_channels': 2  # PPG + ECG
}

print("\n1. Creating TTM Encoder with SSL task...")
print(f"   Config: {CONFIG}")
print()

# Create encoder
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"   Using device: {device}")
encoder = TTMAdapter(**CONFIG).to(device)

print("\n" + "=" * 80)
print("TEST 1: Verify Pretrained Weight Loading")
print("=" * 80)

# Check if using real TTM
using_real_ttm = encoder.is_using_real_ttm()
print(f"\n✓ Using real TTM: {using_real_ttm}")

if not using_real_ttm:
    print("❌ CRITICAL: Not using real TTM! Using fallback encoder instead!")
    print("   This means pretrained weights are NOT loaded!")
    sys.exit(1)
else:
    print("✓ Real TTM loaded successfully")

# Check if weights are pretrained (not random)
print("\n2. Checking if weights are pretrained (not random)...")

# Get first layer weights
try:
    if hasattr(encoder, 'backbone'):
        # Get some weights from the backbone
        all_params = list(encoder.backbone.parameters())
        if len(all_params) > 0:
            first_param = all_params[0]

            # Calculate statistics
            mean = first_param.data.mean().item()
            std = first_param.data.std().item()
            min_val = first_param.data.min().item()
            max_val = first_param.data.max().item()

            print(f"   First layer statistics:")
            print(f"     Mean: {mean:.6f}")
            print(f"     Std:  {std:.6f}")
            print(f"     Min:  {min_val:.6f}")
            print(f"     Max:  {max_val:.6f}")

            # Check if weights look pretrained
            # Random init would have mean ~0, std ~0.01-0.1
            # Pretrained weights have more varied statistics
            if abs(mean) < 0.001 and std < 0.1:
                print("   ⚠️  WARNING: Weights look randomly initialized!")
                print("      Mean ~0 and small std suggests no pretrained weights")
            else:
                print("   ✓ Weights appear pretrained (non-uniform distribution)")
        else:
            print("   ⚠️  No parameters found in backbone")
except Exception as e:
    print(f"   ⚠️  Could not check weights: {e}")

# Count parameters
total_params = sum(p.numel() for p in encoder.parameters())
trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
print(f"\n3. Parameter counts:")
print(f"   Total parameters: {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")
print(f"   Frozen parameters: {total_params - trainable_params:,}")

print("\n" + "=" * 80)
print("TEST 2: Verify Encoder Architecture & Output Shape")
print("=" * 80)

# Check configuration
print(f"\n1. Verifying configuration...")
actual_config = {
    'context_length': encoder.context_length,
    'patch_size': encoder.patch_size,
    'n_patches': encoder.num_patches,
    'hidden_dim': encoder.encoder_dim,
    'input_channels': encoder.input_channels
}

print(f"\n   Expected config: {EXPECTED_CONFIG}")
print(f"   Actual config:   {actual_config}")

# Check each dimension
mismatches = []
for key in EXPECTED_CONFIG:
    if actual_config[key] != EXPECTED_CONFIG[key]:
        mismatches.append(f"{key}: expected {EXPECTED_CONFIG[key]}, got {actual_config[key]}")

if mismatches:
    print(f"\n   ⚠️  Configuration mismatches:")
    for mismatch in mismatches:
        print(f"      - {mismatch}")
else:
    print(f"\n   ✓ All configuration parameters match TTM-Enhanced!")

print(f"\n2. Testing encoder output shape...")

# Create dummy input [B, C, T]
batch_size = 4
dummy_input = torch.randn(batch_size, CONFIG['input_channels'], CONFIG['context_length']).to(device)
print(f"   Input shape: {list(dummy_input.shape)} [B, C, T]")

# Get encoder output
with torch.no_grad():
    try:
        encoder_output = encoder.get_encoder_output(dummy_input)
        print(f"   Encoder output shape: {list(encoder_output.shape)}")

        # Expected shape: [B, P, D]
        expected_shape = [batch_size, EXPECTED_CONFIG['n_patches'], EXPECTED_CONFIG['hidden_dim']]
        print(f"   Expected shape: {expected_shape} [B, P, D]")

        # Verify dimensions
        if encoder_output.dim() != 3:
            print(f"   ❌ CRITICAL: Output is {encoder_output.dim()}D, expected 3D!")
        else:
            print(f"   ✓ Output is 3D as expected")

        if encoder_output.size(0) != batch_size:
            print(f"   ❌ Batch size mismatch: {encoder_output.size(0)} != {batch_size}")
        else:
            print(f"   ✓ Batch dimension correct")

        # Check patch dimension
        actual_patches = encoder_output.size(1)
        expected_patches = EXPECTED_CONFIG['n_patches']

        if actual_patches != expected_patches:
            print(f"   ⚠️  Patch dimension: {actual_patches} (expected {expected_patches})")
            print(f"      TTM may be using different patch_size internally!")
            print(f"      Calculated actual patch_size: {CONFIG['context_length'] // actual_patches}")
        else:
            print(f"   ✓ Patch dimension correct: {actual_patches}")

        # Check hidden dimension
        actual_hidden = encoder_output.size(2)
        expected_hidden = EXPECTED_CONFIG['hidden_dim']

        if actual_hidden != expected_hidden:
            print(f"   ❌ Hidden dimension mismatch: {actual_hidden} != {expected_hidden}")
        else:
            print(f"   ✓ Hidden dimension correct: {actual_hidden}")

        # Overall verdict
        shape_correct = (
            encoder_output.dim() == 3 and
            encoder_output.size(0) == batch_size and
            encoder_output.size(2) == expected_hidden
        )

        if shape_correct:
            print(f"\n   ✅ ENCODER OUTPUT SHAPE CORRECT!")
        else:
            print(f"\n   ❌ ENCODER OUTPUT SHAPE INCORRECT!")

    except Exception as e:
        print(f"   ❌ CRITICAL: Encoder forward pass failed!")
        print(f"      Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("TEST 3: Verify Encoder Does NOT Pool Patches (for SSL)")
print("=" * 80)

print("\nFor SSL masking to work, encoder must preserve patch dimension!")
print("Expected: [B, P, D] where P = number of patches")
print("NOT: [B, D] (pooled)")

with torch.no_grad():
    encoder_output = encoder.get_encoder_output(dummy_input)

    if encoder_output.dim() == 2:
        print("\n❌ CRITICAL: Encoder pooled patches too early!")
        print(f"   Output shape: {list(encoder_output.shape)} [B, D]")
        print("   This will break SSL masking!")
        print("   FIX: Ensure get_encoder_output() returns [B, P, D] not [B, D]")
    elif encoder_output.dim() == 3:
        print("\n✅ Encoder preserves patch dimension!")
        print(f"   Output shape: {list(encoder_output.shape)} [B, P, D]")
        print("   SSL masking will work correctly")
    else:
        print(f"\n⚠️  Unexpected output dimension: {encoder_output.dim()}D")

print("\n" + "=" * 80)
print("TEST 4: Verify Forward Pass (for classification/regression)")
print("=" * 80)

print("\nTesting regular forward pass (with pooling for task heads)...")

with torch.no_grad():
    try:
        # Regular forward should pool to [B, D]
        forward_output = encoder(dummy_input, return_features=False)
        print(f"   Forward output shape: {list(forward_output.shape)}")

        # For SSL task with no head, should return pooled features [B, D]
        # OR encoder features [B, P, D]
        if forward_output.dim() == 2:
            print(f"   ✓ Output is 2D [B, D] - pooled for task head")
            if forward_output.size(1) == EXPECTED_CONFIG['hidden_dim']:
                print(f"   ✓ Hidden dimension correct: {forward_output.size(1)}")
            else:
                print(f"   ⚠️  Hidden dimension: {forward_output.size(1)} (expected {EXPECTED_CONFIG['hidden_dim']})")
        elif forward_output.dim() == 3:
            print(f"   ✓ Output is 3D [B, P, D] - encoder features")
        else:
            print(f"   ⚠️  Unexpected output dimension: {forward_output.dim()}D")

    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

# Collect all findings
findings = []
issues = []

if using_real_ttm:
    findings.append("✓ Using real TTM (not fallback)")
else:
    issues.append("❌ Using fallback encoder (pretrained weights NOT loaded)")

if actual_config == EXPECTED_CONFIG:
    findings.append("✓ Configuration matches TTM-Enhanced")
else:
    issues.append(f"⚠️  Configuration mismatch: {mismatches}")

try:
    with torch.no_grad():
        encoder_output = encoder.get_encoder_output(dummy_input)
        if encoder_output.dim() == 3:
            findings.append(f"✓ Encoder preserves patch dimension [B, P, D]")
        else:
            issues.append(f"❌ Encoder output is {encoder_output.dim()}D (expected 3D)")
except:
    issues.append("❌ Encoder forward pass failed")

print("\nFindings:")
for finding in findings:
    print(f"  {finding}")

if issues:
    print("\nIssues:")
    for issue in issues:
        print(f"  {issue}")
else:
    print("\n✅ ALL CHECKS PASSED!")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)

if issues:
    print("\n⚠️  Issues found! Recommendations:")
    print("  1. Check TTM loading in ttm_adapter.py:_init_real_ttm()")
    print("  2. Verify HuggingFace model ID is correct")
    print("  3. Ensure context_length=1024, patch_size=128 for TTM-Enhanced")
    print("  4. Check encoder output shape in get_encoder_output()")
else:
    print("\n✅ IBM TTM baseline looks correct!")
    print("   Moving to Phase 1.2: SSL Architecture Audit...")

print("=" * 80)
