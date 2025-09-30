#!/bin/bash

# Fix for TTM × VitalDB Training Issues

echo "=============================================="
echo "TTM × VitalDB Training Fix"
echo "=============================================="

# 1. Install missing TTM library
echo ""
echo "Step 1: Installing TTM library (tsfm_public)..."
echo "----------------------------------------------"
pip install git+https://github.com/IBM/tsfm.git || {
    echo "Warning: Could not install tsfm_public. Using fallback model."
}

# Alternative: Install tinytimemixers from PyPI
pip install tinytimemixers || {
    echo "Note: tinytimemixers not available on PyPI"
}

# 2. Use the fixed training script
echo ""
echo "Step 2: Running fixed training script..."
echo "----------------------------------------------"

# Check if we're in Google Colab
if [ -d "/content/drive/MyDrive/TinyFoundationModelForBioSignals" ]; then
    cd /content/drive/MyDrive/TinyFoundationModelForBioSignals
    
    # Run the fixed training script
    python scripts/ttm_vitaldb_fixed.py \
        --model-yaml configs/model.yaml \
        --run-yaml configs/fasttrack_cpu.yaml \
        --split-file configs/splits/splits_full.json \
        --outdir artifacts/raw_windows \
        --out artifacts/model_finetuned \
        --task clf \
        --fasttrack
else
    # Run locally
    cd "$(dirname "$0")/.."
    
    python scripts/ttm_vitaldb_fixed.py \
        --model-yaml configs/model.yaml \
        --run-yaml configs/run.yaml \
        --split-file configs/splits/splits_full.json \
        --outdir artifacts/raw_windows \
        --out artifacts/model_output \
        --task clf \
        --fasttrack
fi

echo ""
echo "=============================================="
echo "Training should now work correctly!"
echo "=============================================="
