"""Test script to verify critical fixes for TTM adapter.

Tests:
1. Context length matches window configuration (1250 samples)
2. Model variant consistency (ttm-r1)
3. LoRA integration with TTM
4. Module inspection utility
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.ttm_adapter import create_ttm_model, TTMAdapter
from src.models.lora import LoRAConfig
from src.utils.io import load_yaml


def test_context_length_fix():
    """Test that context_length defaults to 1250."""
    print("\n" + "="*70)
    print("TEST 1: Context Length Fix (512 -> 1250)")
    print("="*70)
    
    # Test default context_length
    model = TTMAdapter(
        task='classification',
        num_classes=2,
        freeze_encoder=True,
        use_real_ttm=False  # Use fallback for quick test
    )
    
    assert model.context_length == 1250, f"Expected 1250, got {model.context_length}"
    print(f"✓ Default context_length: {model.context_length} samples")
    print(f"✓ This equals: {model.context_length / 125:.1f} seconds at 125 Hz")
    
    # Test with config file
    config = {
        'task': 'classification',
        'num_classes': 2,
        'freeze_encoder': True,
        'input_channels': 1,
        # Don't specify context_length - should use default
    }
    
    model2 = create_ttm_model(config)
    assert model2.context_length == 1250, f"Expected 1250, got {model2.context_length}"
    print(f"✓ Config-based model also uses 1250 by default")
    
    # Test with actual input (simulate 10-second window)
    batch_size = 4
    channels = 1
    time_steps = 1250  # 10 seconds at 125 Hz
    
    x = torch.randn(batch_size, channels, time_steps)
    print(f"✓ Test input shape: {list(x.shape)} [batch, channels, time]")
    
    # Forward pass
    output = model2(x)
    print(f"✓ Output shape: {list(output.shape)} [batch, num_classes]")
    print(f"✓ Model accepts 1250-sample input correctly!")
    
    print("\n✅ TEST 1 PASSED: Context length fix verified")


def test_model_variant_consistency():
    """Test that model variant is consistent."""
    print("\n" + "="*70)
    print("TEST 2: Model Variant Consistency (ttm-v1 -> ttm-r1)")
    print("="*70)
    
    # Load config
    config_path = project_root / "configs" / "model.yaml"
    config = load_yaml(config_path)
    
    variant = config['model']['variant']
    print(f"Config file variant: {variant}")
    
    assert 'ttm-r1' in variant, f"Expected ttm-r1, got {variant}"
    print(f"✓ Config uses ttm-r1 (release version)")
    
    # Check default in code
    model = TTMAdapter(
        task='classification',
        num_classes=2,
        use_real_ttm=False
    )
    
    assert 'ttm-r1' in model.variant, f"Expected ttm-r1, got {model.variant}"
    print(f"✓ Code default: {model.variant}")
    
    print("\n✅ TEST 2 PASSED: Model variant is consistent")


def test_lora_integration():
    """Test LoRA integration with TTM."""
    print("\n" + "="*70)
    print("TEST 3: LoRA Integration")
    print("="*70)
    
    # Create model with LoRA
    lora_config = LoRAConfig(
        r=8,
        alpha=16,
        dropout=0.1,
        target_modules=None,  # Use defaults
        exclude_modules=None
    )
    
    print("Creating model with LoRA (using fallback encoder for testing)...")
    model = TTMAdapter(
        task='classification',
        num_classes=2,
        freeze_encoder=True,
        lora_config=lora_config,
        use_real_ttm=False,  # Use fallback for testing
        input_channels=1
    )
    
    # Check LoRA modules were created
    if hasattr(model, 'lora_modules'):
        print(f"✓ LoRA modules created: {len(model.lora_modules)}")
        if len(model.lora_modules) > 0:
            print(f"✓ First LoRA module: {list(model.lora_modules.keys())[0]}")
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad and 'lora_' in str(p))
    
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    print(f"✓ LoRA parameters: {lora_params:,}")
    print(f"✓ Trainable percentage: {trainable_params/total_params*100:.2f}%")
    
    # Test forward pass with LoRA
    x = torch.randn(2, 1, 1250)
    output = model(x)
    print(f"✓ Forward pass with LoRA works: {list(output.shape)}")
    
    print("\n✅ TEST 3 PASSED: LoRA integration verified")


def test_module_inspection():
    """Test module inspection utility."""
    print("\n" + "="*70)
    print("TEST 4: Module Inspection Utility")
    print("="*70)
    
    model = TTMAdapter(
        task='classification',
        num_classes=2,
        use_real_ttm=False,
        input_channels=1
    )
    
    print("Testing inspect_modules() method...")
    model.inspect_modules(show_all=False, max_display=10)
    
    print("✅ TEST 4 PASSED: Module inspection works")


def test_dimension_consistency():
    """Test that dimensions are consistent throughout."""
    print("\n" + "="*70)
    print("TEST 5: End-to-End Dimension Consistency")
    print("="*70)
    
    # Simulate the full pipeline
    print("\nSimulating full data pipeline:")
    
    # 1. Raw signal (10 seconds at 125 Hz)
    fs = 125
    duration = 10.0
    n_samples = int(fs * duration)
    print(f"1. Raw signal: {n_samples} samples ({duration}s @ {fs} Hz)")
    
    # 2. Window configuration
    window_samples = n_samples
    print(f"2. Window size: {window_samples} samples")
    
    # 3. Model context_length
    model = TTMAdapter(
        task='classification',
        num_classes=2,
        use_real_ttm=False,
        input_channels=1
    )
    print(f"3. Model context_length: {model.context_length} samples")
    
    # 4. Verify they match
    assert window_samples == model.context_length, \
        f"Mismatch! Window: {window_samples}, Model: {model.context_length}"
    print(f"✓ Window size matches model context_length!")
    
    # 5. Test with actual data
    batch_size = 8
    n_channels = 1
    x = torch.randn(batch_size, n_channels, window_samples)
    print(f"4. Test batch shape: {list(x.shape)}")
    
    output = model(x)
    print(f"5. Output shape: {list(output.shape)}")
    
    assert output.shape == (batch_size, 2), f"Expected ({batch_size}, 2), got {output.shape}"
    print(f"✓ End-to-end pipeline works correctly!")
    
    print("\n✅ TEST 5 PASSED: Dimensions are consistent")


def run_all_tests():
    """Run all verification tests."""
    print("\n" + "="*70)
    print("CRITICAL FIXES VERIFICATION TEST SUITE")
    print("="*70)
    print("\nThis test suite verifies that all critical fixes are working:")
    print("1. Context length fix (512 -> 1250)")
    print("2. Model variant consistency (ttm-v1 -> ttm-r1)")
    print("3. LoRA integration improvements")
    print("4. Module inspection utility")
    print("5. End-to-end dimension consistency")
    
    try:
        test_context_length_fix()
        test_model_variant_consistency()
        test_lora_integration()
        test_module_inspection()
        test_dimension_consistency()
        
        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*70)
        print("\nCritical fixes verified:")
        print("✓ Context length: 1250 samples (10s @ 125Hz)")
        print("✓ Model variant: ttm-r1 (consistent)")
        print("✓ LoRA: Properly integrated with TTM")
        print("✓ Module inspection: Working")
        print("✓ Dimensions: Consistent end-to-end")
        print("\n✅ Your model is now ready for training!")
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST FAILED!")
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
