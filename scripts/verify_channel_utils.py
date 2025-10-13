#!/usr/bin/env python3
"""Quick verification script for channel inflation utilities."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

print("=" * 70)
print("CHANNEL INFLATION UTILITIES - QUICK VERIFICATION")
print("=" * 70)

# Test 1: Import check
print("\n1. Testing imports...")
try:
    from src.models.channel_utils import (
        load_pretrained_with_channel_inflate,
        unfreeze_last_n_blocks,
        verify_channel_inflation,
        get_channel_inflation_report
    )
    print("   ✓ All imports successful")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Module structure check
print("\n2. Checking module structure...")
try:
    from src.models import (
        load_pretrained_with_channel_inflate,
        unfreeze_last_n_blocks,
        verify_channel_inflation,
        get_channel_inflation_report
    )
    print("   ✓ Module exports work correctly")
except ImportError as e:
    print(f"   ✗ Module export failed: {e}")
    sys.exit(1)

# Test 3: Documentation check
print("\n3. Checking function documentation...")
functions = [
    load_pretrained_with_channel_inflate,
    unfreeze_last_n_blocks,
    verify_channel_inflation,
    get_channel_inflation_report
]

for func in functions:
    if func.__doc__:
        print(f"   ✓ {func.__name__}: documented")
    else:
        print(f"   ✗ {func.__name__}: missing docstring")

# Test 4: Quick functionality test
print("\n4. Testing basic functionality...")

# Create a simple test model
class SimpleModel(nn.Module):
    def __init__(self, channels=2):
        super().__init__()
        self.backbone = nn.ModuleList([
            nn.Linear(channels, 64),
            nn.Linear(64, 64)
        ])
        self.head = nn.Linear(64, 2)
    
    def forward(self, x):
        for layer in self.backbone:
            x = layer(x)
        return self.head(x)

try:
    model = SimpleModel(channels=2)
    
    # Freeze all
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze with n=1
    unfreeze_last_n_blocks(model, n=1, verbose=False)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable > 0:
        print("   ✓ unfreeze_last_n_blocks works")
    else:
        print("   ✗ No parameters unfrozen")
        
except Exception as e:
    print(f"   ✗ Functionality test failed: {e}")

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
print("\nNext steps:")
print("  1. Run full test suite: pytest tests/test_channel_utils.py")
print("  2. See docs/channel_inflation_guide.md for usage examples")
print("  3. Integrate with SSL pretraining and fine-tuning pipelines")
print("=" * 70)
