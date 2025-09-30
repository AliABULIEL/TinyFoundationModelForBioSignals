#!/bin/bash
# Rebuild windows with 512 samples for TTM compatibility

echo "=================================================="
echo "Rebuilding Windows with 512 Samples (4.096s)"
echo "=================================================="

# Configuration has been updated to use 512 samples (4.096 seconds at 125 Hz)
# This matches TTM's expected context_length

echo ""
echo "Step 1: Clean old windows..."
echo "--------------------------------------------------"
# Optional: backup old windows
if [ -d "artifacts/raw_windows" ]; then
    echo "Backing up old windows to artifacts/raw_windows_1000..."
    mv artifacts/raw_windows artifacts/raw_windows_1000
fi

echo ""
echo "Step 2: Rebuild training windows (512 samples)..."
echo "--------------------------------------------------"
python scripts/ttm_vitaldb.py build-windows \
    --channels-yaml configs/channels.yaml \
    --windows-yaml configs/windows.yaml \
    --split-file configs/splits/splits_full.json \
    --split train \
    --channel PPG \
    --outdir artifacts/raw_windows/train \
    --multiprocess \
    --num-workers 8 \
    --min-sqi 0.5

echo ""
echo "Step 3: Rebuild validation windows (512 samples)..."
echo "--------------------------------------------------"
python scripts/ttm_vitaldb.py build-windows \
    --channels-yaml configs/channels.yaml \
    --windows-yaml configs/windows.yaml \
    --split-file configs/splits/splits_full.json \
    --split val \
    --channel PPG \
    --outdir artifacts/raw_windows/val \
    --multiprocess \
    --num-workers 8 \
    --min-sqi 0.5

echo ""
echo "Step 4: Rebuild test windows (512 samples)..."
echo "--------------------------------------------------"
python scripts/ttm_vitaldb.py build-windows \
    --channels-yaml configs/channels.yaml \
    --windows-yaml configs/windows.yaml \
    --split-file configs/splits/splits_full.json \
    --split test \
    --channel PPG \
    --outdir artifacts/raw_windows/test \
    --multiprocess \
    --num-workers 8 \
    --min-sqi 0.5

echo ""
echo "Step 5: Verify window sizes..."
echo "--------------------------------------------------"
python -c "
import numpy as np
import os

for split in ['train', 'val', 'test']:
    path = f'artifacts/raw_windows/{split}/{split}_windows.npz'
    if os.path.exists(path):
        data = np.load(path)
        print(f'{split}: {data[\"data\"].shape} - Expected: (N, 512, 1)')
    else:
        print(f'{split}: File not found')
"

echo ""
echo "=================================================="
echo "✓ Window rebuilding complete!"
echo "  Windows are now 512 samples (4.096s at 125Hz)"
echo "  Compatible with TTM model expectations"
echo "=================================================="
