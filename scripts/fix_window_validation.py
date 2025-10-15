#!/usr/bin/env python3
"""
Quick fix: Patch ttm_vitaldb.py to remove strict shape validation

The current code filters out ALL windows due to strict shape matching.
This script fixes it by allowing ±5% tolerance in window length.
"""

import sys
from pathlib import Path

script_path = Path(__file__).parent / "ttm_vitaldb.py"

print(f"Patching {script_path}...")

# Read the file
with open(script_path, 'r') as f:
    content = f.read()

# Pattern 1: Remove strict shape validation in first batch
old_pattern1 = """                for w in first_windows:
                    # Ensure window is 2D [T, C]
                    if w.ndim == 1:
                        w = w.reshape(-1, 1)
                    elif w.ndim == 3:
                        w = w.squeeze(0)
                    
                    # Validate shape matches expected dimensions
                    expected_samples = int(window_s * 125)  # Assuming 125Hz
                    if w.shape[0] == expected_samples:
                        first_windows.append(w)
                    else:
                        logger.debug(f"Skipping window with shape {w.shape}, expected {expected_samples} samples")"""

new_pattern1 = """                for w in first_windows:
                    # Ensure window is 2D [T, C]
                    if w.ndim == 1:
                        w = w.reshape(-1, 1)
                    elif w.ndim == 3:
                        w = w.squeeze(0)
                    
                    # Allow ±5% tolerance in window length
                    expected_samples = int(window_s * 125)  # Assuming 125Hz
                    tolerance = int(expected_samples * 0.05)  # 5% tolerance
                    if abs(w.shape[0] - expected_samples) <= tolerance:
                        # Pad or trim to exact size if needed
                        if w.shape[0] < expected_samples:
                            pad_width = ((0, expected_samples - w.shape[0]), (0, 0))
                            w = np.pad(w, pad_width, mode='edge')
                        elif w.shape[0] > expected_samples:
                            w = w[:expected_samples, :]
                        first_windows.append(w)
                    else:
                        logger.debug(f"Skipping window with shape {w.shape}, expected {expected_samples}±{tolerance} samples")"""

# Pattern 2: Remove strict validation in main loop
old_pattern2 = """            if windows is not None and len(windows) > 0:
                # Validate window shapes before adding
                for w in windows:
                    # Ensure window is 2D [T, C]
                    if w.ndim == 1:
                        w = w.reshape(-1, 1)
                    elif w.ndim == 3:
                        w = w.squeeze(0)
                    
                    # Only add if shape is valid
                    expected_samples = int(window_s * 125)  # Assuming 125Hz
                    if w.shape[0] == expected_samples:
                        all_windows.append(w)
                        all_labels.append(0)  # Placeholder label
                successful_cases += 1"""

new_pattern2 = """            if windows is not None and len(windows) > 0:
                # Validate window shapes before adding (with tolerance)
                for w in windows:
                    # Ensure window is 2D [T, C]
                    if w.ndim == 1:
                        w = w.reshape(-1, 1)
                    elif w.ndim == 3:
                        w = w.squeeze(0)
                    
                    # Allow ±5% tolerance and fix shape
                    expected_samples = int(window_s * 125)  # Assuming 125Hz
                    tolerance = int(expected_samples * 0.05)
                    if abs(w.shape[0] - expected_samples) <= tolerance:
                        # Pad or trim to exact size
                        if w.shape[0] < expected_samples:
                            pad_width = ((0, expected_samples - w.shape[0]), (0, 0))
                            w = np.pad(w, pad_width, mode='edge')
                        elif w.shape[0] > expected_samples:
                            w = w[:expected_samples, :]
                        all_windows.append(w)
                        all_labels.append(0)  # Placeholder label
                successful_cases += 1"""

# Apply patches
if old_pattern1 in content:
    content = content.replace(old_pattern1, new_pattern1)
    print("✓ Fixed first batch validation")
else:
    print("⚠ First batch validation pattern not found")

if old_pattern2 in content:
    content = content.replace(old_pattern2, new_pattern2)
    print("✓ Fixed main loop validation")
else:
    print("⚠ Main loop validation pattern not found")

# Write back
with open(script_path, 'w') as f:
    f.write(content)

print(f"✓ Patched {script_path}")
print("\nNow run:")
print("python scripts/prepare_all_data.py --mode fasttrack --dataset vitaldb --num-workers 8")
