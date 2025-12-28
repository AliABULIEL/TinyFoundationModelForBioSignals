#!/usr/bin/env python3
"""
Debug checkpoint key structure to fix fine-tuning loading.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

checkpoint_path = "artifacts/foundation_model/best_model.pt"
print("=" * 80)
print("DEBUGGING CHECKPOINT KEY STRUCTURE")
print("=" * 80)
print(f"Checkpoint: {checkpoint_path}\n")

checkpoint = torch.load(checkpoint_path, map_location='cpu')
state_dict = checkpoint['encoder_state_dict']

print(f"Total keys: {len(state_dict)}\n")

# Analyze key prefixes
print("First 20 keys:")
for i, key in enumerate(list(state_dict.keys())[:20]):
    shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
    print(f"{i+1:2d}. {key}")
    print(f"    Shape: {shape}\n")

print("=" * 80)
print("KEY PREFIX ANALYSIS")
print("=" * 80)

# Count by prefix
prefixes = {}
for key in state_dict.keys():
    # Get first two parts of key
    parts = key.split('.')
    if len(parts) >= 2:
        prefix = f"{parts[0]}.{parts[1]}"
    else:
        prefix = parts[0]

    if prefix not in prefixes:
        prefixes[prefix] = 0
    prefixes[prefix] += 1

print("\nKey prefix counts:")
for prefix, count in sorted(prefixes.items()):
    print(f"  {prefix}: {count} keys")

print("\n" + "=" * 80)
