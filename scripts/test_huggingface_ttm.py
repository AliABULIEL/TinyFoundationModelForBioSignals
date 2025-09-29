#!/usr/bin/env python3
"""
Alternative approach: Load TTM directly from HuggingFace
"""

import torch
from transformers import AutoModel, AutoConfig

print("=" * 70)
print("Attempting to load TTM directly from HuggingFace")
print("=" * 70)

try:
    model_id = "ibm-granite/granite-timeseries-ttm-r1"
    
    print(f"Loading: {model_id}")
    
    # Try loading config first
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    print(f"✅ Config loaded: {config.model_type}")
    
    # Load model
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,  # Required for custom models
    )
    
    print("✅ Model loaded successfully!")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
except Exception as e:
    print(f"❌ Failed: {e}")
    print("\nThis model might require the tsfm_public library to be installed")

print("=" * 70)
