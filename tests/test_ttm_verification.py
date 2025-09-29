"""Test to verify TTM model is actually loaded from IBM's tsfm library."""

import warnings
import torch
import pytest


def test_ttm_actual_loading():
    """Test that we're actually loading TTM, not the fallback."""
    
    # Try importing tsfm
    try:
        from tsfm import TinyTimeMixerForPrediction
        tsfm_available = True
    except ImportError:
        tsfm_available = False
        warnings.warn("tsfm not installed - TTM tests will use fallback")
    
    # Import our adapter
    from src.models.ttm_adapter import create_ttm_model, TTMAdapter
    
    # Create model with IBM TTM variant
    config = {
        'variant': 'ibm-granite/granite-timeseries-ttm-v1',
        'task': 'classification',
        'num_classes': 2,
        'input_channels': 3,
        'context_length': 512,
        'freeze_encoder': True,
        'use_tsfm': True  # Explicitly request tsfm
    }
    
    model = create_ttm_model(config)
    
    # Check if we're using real TTM
    print("\n" + "=" * 60)
    print("TTM MODEL VERIFICATION TEST")
    print("=" * 60)
    
    if tsfm_available:
        print("✓ tsfm library is installed")
        assert model.is_using_real_ttm(), "Model should be using real TTM when tsfm is available"
        print("✓ Model is using REAL IBM TTM (not fallback)")
        
        # Check model has expected attributes
        assert hasattr(model.encoder, 'config'), "TTM should have config attribute"
        print(f"✓ Model config found: {type(model.encoder.config)}")
        
        # Print model info
        model.print_parameter_summary()
        
    else:
        print("⚠️ tsfm library NOT installed")
        assert not model.is_using_real_ttm(), "Model should use fallback when tsfm is not available"
        print("⚠️ Model is using FALLBACK encoder (not real TTM)")
        print("\nTo use real TTM, install tsfm:")
        print("  pip install tsfm[notebooks]")
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, config['input_channels'], config['context_length'])
    
    with torch.no_grad():
        output = model(x)
    
    assert output is not None, "Model should produce output"
    assert output.shape[0] == batch_size, "Batch size should be preserved"
    
    if config['task'] == 'classification':
        assert output.shape[-1] == config['num_classes'], "Output should match num_classes"
    
    print(f"✓ Forward pass successful: {x.shape} -> {output.shape}")
    
    print("=" * 60)
    
    return model.is_using_real_ttm()


def test_ttm_vs_fallback_comparison():
    """Compare TTM model vs fallback to ensure different behavior."""
    
    from src.models.ttm_adapter import TTMAdapter
    
    # Create TTM model (will use real or fallback based on availability)
    ttm_model = TTMAdapter(
        variant='ibm-granite/granite-timeseries-ttm-v1',
        task='classification',
        num_classes=2,
        input_channels=3,
        context_length=512,
        use_tsfm=True
    )
    
    # Explicitly create fallback model
    fallback_model = TTMAdapter(
        variant='fallback',
        task='classification', 
        num_classes=2,
        input_channels=3,
        context_length=512,
        use_tsfm=False  # Force fallback
    )
    
    # Test data
    x = torch.randn(2, 3, 512)
    
    with torch.no_grad():
        ttm_output = ttm_model(x)
        fallback_output = fallback_model(x)
    
    print("\n" + "=" * 60)
    print("TTM vs FALLBACK COMPARISON")
    print("=" * 60)
    print(f"TTM using real model: {ttm_model.is_using_real_ttm()}")
    print(f"Fallback using real model: {fallback_model.is_using_real_ttm()}")
    
    # Count parameters
    ttm_params = sum(p.numel() for p in ttm_model.parameters())
    fallback_params = sum(p.numel() for p in fallback_model.parameters())
    
    print(f"TTM parameters: {ttm_params:,}")
    print(f"Fallback parameters: {fallback_params:,}")
    
    if ttm_model.is_using_real_ttm():
        # If using real TTM, should have different architecture
        assert ttm_params != fallback_params, "Real TTM should have different param count than fallback"
        print("✓ Parameter counts differ (as expected)")
    else:
        print("⚠️ Both models using fallback (install tsfm for real TTM)")
    
    print("=" * 60)


def test_installation_instructions():
    """Provide clear instructions for installing TTM."""
    
    print("\n" + "=" * 60)
    print("TTM INSTALLATION INSTRUCTIONS")
    print("=" * 60)
    
    try:
        import tsfm
        print("✅ tsfm is installed!")
        print(f"   Version: {tsfm.__version__ if hasattr(tsfm, '__version__') else 'unknown'}")
    except ImportError:
        print("❌ tsfm is NOT installed")
        print("\nTo install IBM's TTM models:")
        print("\n1. Install tsfm library:")
        print("   pip install tsfm[notebooks]")
        print("\n2. Or from source:")
        print("   git clone https://github.com/IBM/tsfm")
        print("   cd tsfm")
        print("   pip install -e .")
        print("\n3. Verify installation:")
        print("   python3 -c 'from tsfm import TinyTimeMixerForPrediction; print(\"Success!\")'")
    
    print("\nAvailable TTM models from IBM:")
    print("  - ibm-granite/granite-timeseries-ttm-v1")
    print("  - Context lengths: 512, 1024, 1536")
    print("  - Pre-trained on diverse time series data")
    
    print("=" * 60)


if __name__ == "__main__":
    # Run all tests
    print("Running TTM verification tests...\n")
    
    # Test 1: Installation check
    test_installation_instructions()
    
    # Test 2: Model loading verification
    is_using_real_ttm = test_ttm_actual_loading()
    
    # Test 3: Comparison test
    test_ttm_vs_fallback_comparison()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if is_using_real_ttm:
        print("✅ SUCCESS: Using real IBM TTM model")
        print("   You can expect high accuracy with pre-trained weights")
    else:
        print("⚠️ WARNING: Using fallback model (not pre-trained)")
        print("   Install tsfm for real TTM: pip install tsfm[notebooks]")
        print("   Without real TTM, accuracy will be significantly lower")
    
    print("=" * 60)
