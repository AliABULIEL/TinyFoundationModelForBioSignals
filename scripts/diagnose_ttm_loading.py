#!/usr/bin/env python3
"""Diagnostic script to troubleshoot TTM loading issues."""

import sys
import torch

print("="*80)
print("TTM LOADING DIAGNOSTICS")
print("="*80)

# Check Python version
print(f"\n[1/8] Python version: {sys.version}")

# Check PyTorch
print(f"\n[2/8] PyTorch:")
print(f"  Version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU device: {torch.cuda.get_device_name(0)}")
    print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Check tsfm_public
print(f"\n[3/8] tsfm_public:")
try:
    import tsfm_public
    print(f"  ✓ Installed: {tsfm_public.__version__ if hasattr(tsfm_public, '__version__') else 'version unknown'}")

    from tsfm_public import get_model, count_parameters
    print(f"  ✓ get_model importable")
    print(f"  ✓ count_parameters importable")

    from tsfm_public.models.tinytimemixer.modeling_tinytimemixer import TinyTimeMixerForPrediction
    print(f"  ✓ TinyTimeMixerForPrediction importable")

except ImportError as e:
    print(f"  ❌ Import error: {e}")
    print(f"\n  To install: pip install tsfm[notebooks]")
    sys.exit(1)

# Check HuggingFace Hub access
print(f"\n[4/8] HuggingFace Hub:")
try:
    from huggingface_hub import hf_hub_download, cached_assets_path
    print(f"  ✓ huggingface_hub installed")

    # Try to download model info
    try:
        from huggingface_hub import model_info
        info = model_info("ibm-granite/granite-timeseries-ttm-r1")
        print(f"  ✓ Model accessible: ibm-granite/granite-timeseries-ttm-r1")
        print(f"  Model size: {info.safetensors.total / 1e6:.2f} MB")
    except Exception as e:
        print(f"  ⚠️  Model access check failed: {e}")

except ImportError as e:
    print(f"  ❌ huggingface_hub not installed: {e}")

# Check transformers
print(f"\n[5/8] transformers:")
try:
    import transformers
    print(f"  ✓ Installed: {transformers.__version__}")
except ImportError as e:
    print(f"  ❌ Not installed: {e}")

# Try loading TTM model
print(f"\n[6/8] Attempting to load TTM model...")
try:
    from tsfm_public import get_model

    print(f"  Loading ibm-granite/granite-timeseries-ttm-r1...")
    print(f"  Config: context=1024, prediction=96, channels=2")

    model = get_model(
        "ibm-granite/granite-timeseries-ttm-r1",
        context_length=1024,
        prediction_length=96,
        num_input_channels=2,
        decoder_mode="mix_channel"
    )

    print(f"  ✓ Model loaded successfully!")
    print(f"  Type: {type(model)}")

    # Test forward pass
    print(f"\n  Testing forward pass...")
    dummy_input = torch.randn(2, 1024, 2)  # [batch, time, channels]

    with torch.no_grad():
        output = model(dummy_input)

    print(f"  ✓ Forward pass successful!")
    print(f"  Output type: {type(output)}")
    if hasattr(output, 'last_hidden_state'):
        print(f"  last_hidden_state shape: {output.last_hidden_state.shape}")
    if hasattr(output, 'prediction_outputs'):
        print(f"  prediction_outputs shape: {output.prediction_outputs.shape}")

    # Test encoder output extraction
    print(f"\n  Testing encoder output extraction...")
    backbone = model.backbone if hasattr(model, 'backbone') else model
    backbone_output = backbone(dummy_input)

    if hasattr(backbone_output, 'last_hidden_state'):
        features = backbone_output.last_hidden_state
        print(f"  ✓ Encoder features shape: {features.shape}")
        print(f"  Expected: [batch=2, patches=16, d_model=192]")
    else:
        print(f"  ⚠️  backbone_output type: {type(backbone_output)}")

except Exception as e:
    print(f"  ❌ Loading failed!")
    print(f"\nException type: {type(e).__name__}")
    print(f"Exception message: {e}")
    print(f"\nFull traceback:")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check disk space
print(f"\n[7/8] Disk space:")
try:
    import shutil
    total, used, free = shutil.disk_usage("/")
    print(f"  Total: {total / 1e9:.2f} GB")
    print(f"  Used: {used / 1e9:.2f} GB")
    print(f"  Free: {free / 1e9:.2f} GB")

    if free < 1e9:  # Less than 1 GB free
        print(f"  ⚠️  WARNING: Low disk space may cause model download to fail")
except:
    print(f"  (Unable to check disk space)")

# Check memory
print(f"\n[8/8] System memory:")
try:
    import psutil
    mem = psutil.virtual_memory()
    print(f"  Total: {mem.total / 1e9:.2f} GB")
    print(f"  Available: {mem.available / 1e9:.2f} GB")
    print(f"  Used: {mem.used / 1e9:.2f} GB ({mem.percent}%)")

    if mem.available < 2e9:  # Less than 2 GB available
        print(f"  ⚠️  WARNING: Low memory may cause model loading to fail")
except:
    print(f"  (psutil not installed - cannot check memory)")

print("\n" + "="*80)
print("DIAGNOSTICS COMPLETE")
print("="*80)
print("\nIf TTM loaded successfully above, the issue may be:")
print("  1. Environment difference between this script and SSL training")
print("  2. CUDA/GPU memory issues during training")
print("  3. Checkpoint loading issues")
print("\nIf TTM failed to load, check:")
print("  1. tsfm[notebooks] installation: pip install tsfm[notebooks]")
print("  2. Internet connection for HuggingFace Hub")
print("  3. Disk space and memory availability")
print("="*80)
