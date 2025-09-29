# How to Use TinyTimeMixers from Hugging Face

## Installation

First, install the transformers library:
```bash
pip install transformers torch
```

## Available Models

IBM has released several TTM models on Hugging Face:
- `ibm/TTM` - The base TinyTimeMixer model
- Check https://huggingface.co/ibm for the latest models

## Usage Example

```python
from src.models import create_ttm_model

# Configure the model to use HuggingFace
config = {
    'variant': 'ibm/TTM',  # HuggingFace model ID
    'task': 'classification',
    'num_classes': 2,
    'freeze_encoder': True,
    'input_channels': 1,
    'context_length': 96
}

model = create_ttm_model(config)
```

## Alternative Time Series Models

If TTM isn't available, consider these alternatives from HuggingFace:
- PatchTST: `amazon/chronos-t5-mini`
- TimesFM: Google's foundation model
- Lag-Llama: Time series forecasting model

## Fallback Behavior

If the specified model can't be loaded from HuggingFace, the adapter will:
1. Warn about the loading failure
2. Create a simple transformer encoder as fallback
3. Continue working with reduced performance

This ensures your code can run even during development or testing.
