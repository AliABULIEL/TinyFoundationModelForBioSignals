#!/usr/bin/env python3
"""
Test script to verify the actual TTM model loading from tsfm_public
Based on the IBM notebook
"""

import sys
import torch

print("=" * 70)
print("Testing REAL TTM Model Loading")
print("=" * 70)

# Try the correct imports from the notebook
try:
    from tsfm_public import (
        TimeSeriesPreprocessor,
        get_model,
        count_parameters
    )
    print("✅ tsfm_public module imported successfully!")
    
    # Try loading the actual model
    TTM_MODEL_PATH = "ibm-granite/granite-timeseries-ttm-r1"
    
    print(f"\nAttempting to load: {TTM_MODEL_PATH}")
    
    model = get_model(
        TTM_MODEL_PATH,
        context_length=512,
        prediction_length=96,
        num_input_channels=3,  # ECG, PPG, ABP
    )
    
    print("✅ TTM Model loaded successfully!")
    print(f"Model type: {type(model)}")
    print(f"Number of parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 512, 3)  # [batch, time, channels]
    
    with torch.no_grad():
        outputs = model(x)
    
    print(f"\nForward pass successful!")
    print(f"Input shape: {x.shape}")
    print(f"Output type: {type(outputs)}")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nThe tsfm_public module is not accessible.")
    print("This suggests the pip package doesn't include the actual model code.")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")

print("\n" + "=" * 70)
print("DIAGNOSIS:")
print("=" * 70)

print("""
The notebook shows TTM is real and works, but the pip package 
seems incomplete. You need to either:

1. Clone and install from source with the tsfm_public module:
   git clone https://github.com/IBM/tsfm.git
   cd tsfm
   pip install -e .

2. Or use the model directly from HuggingFace (if available):
   from transformers import AutoModel
   model = AutoModel.from_pretrained("ibm-granite/granite-timeseries-ttm-r1")
""")
