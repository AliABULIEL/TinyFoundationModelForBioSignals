"""Quick test to verify TTM configuration is correct"""

import torch
import sys
sys.path.insert(0, '/Users/aliab/Desktop/TinyFoundationModelForBioSignals/src')

from models.ttm_adapter import TTMAdapter

def test_ttm_configuration():
    """Test that TTM can be initialized and forward pass works"""
    
    print("=" * 70)
    print("TTM CONFIGURATION VERIFICATION TEST")
    print("=" * 70)
    
    # Configuration for VitalDB SSL pretraining
    config = {
        'task': 'ssl',
        'input_channels': 2,        # PPG + ECG
        'context_length': 1250,     # 10 seconds @ 125Hz
        'patch_size': 125,          # 1 second patches
        'freeze_encoder': False,    # Training from scratch
        'use_real_ttm': True,
        'decoder_mode': 'mix_channel'
    }
    
    print("\n1. Creating TTM model with configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    try:
        model = TTMAdapter(**config)
        print("\n✅ Model created successfully!")
        
        # Print model summary
        model.print_parameter_summary()
        
        # Test forward pass
        print("\n2. Testing forward pass...")
        batch_size = 4
        dummy_input = torch.randn(batch_size, 2, 1250)  # [B, C, T]
        print(f"   Input shape: {dummy_input.shape}")
        
        with torch.no_grad():
            output = model.get_encoder_output(dummy_input)
        
        print(f"   Output shape: {output.shape}")
        expected_shape = (batch_size, 10, 384)  # [B, patches, d_model]
        
        if output.shape == expected_shape:
            print(f"✅ Output shape correct! Expected {expected_shape}, got {output.shape}")
        else:
            print(f"❌ Output shape mismatch! Expected {expected_shape}, got {output.shape}")
            return False
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("\nYou can now run SSL pretraining:")
        print("  python scripts/pretrain_vitaldb_ssl.py")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 70)
        print("❌ TEST FAILED - Check the error above")
        print("=" * 70)
        return False

if __name__ == "__main__":
    success = test_ttm_configuration()
    sys.exit(0 if success else 1)
