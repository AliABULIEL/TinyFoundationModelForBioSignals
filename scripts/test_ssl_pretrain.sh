#!/bin/bash
# Quick test of SSL pretraining script

set -e

echo "========================================================================"
echo "SSL PRETRAINING - QUICK TEST"
echo "========================================================================"

# Check if data directory exists
if [ ! -d "data/vitaldb_windows" ]; then
    echo "Error: data/vitaldb_windows not found"
    echo "Please prepare VitalDB windows first using:"
    echo "  python scripts/ttm_vitaldb.py build-windows ..."
    exit 1
fi

# Check if train_windows.npz exists
if [ ! -f "data/vitaldb_windows/train_windows.npz" ]; then
    echo "Error: data/vitaldb_windows/train_windows.npz not found"
    echo "Please build VitalDB windows first"
    exit 1
fi

echo ""
echo "Running SSL pretraining test (5 epochs, fast mode)..."
echo ""

python scripts/pretrain_vitaldb_ssl.py \
    --config configs/ssl_pretrain.yaml \
    --data-dir data/vitaldb_windows \
    --output-dir artifacts/ssl_test \
    --epochs 5 \
    --batch-size 32 \
    --mask-ratio 0.4 \
    --fast \
    --device cpu

echo ""
echo "========================================================================"
echo "Test complete! Check artifacts/ssl_test/ for outputs"
echo "========================================================================"
