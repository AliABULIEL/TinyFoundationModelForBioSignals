#!/bin/bash
# FastTrack Pipeline - Complete training in ~3 hours

# Exit on error
set -e

echo "=========================================="
echo "TTM × VitalDB FastTrack Pipeline"
echo "Expected runtime: ~3 hours on single GPU"
echo "=========================================="

# Set paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="${PYTHON:-python3}"

echo "Running in FastTrack mode (50 train, 20 test cases)"

# Create output directories
mkdir -p artifacts/raw_windows
mkdir -p artifacts/checkpoints
mkdir -p data

echo ""
echo "Step 1/4: Preparing train/test splits..."
echo "------------------------------------------"
$PYTHON scripts/ttm_vitaldb.py prepare-splits \
    --mode fasttrack \
    --case-set bis \
    --output data \
    --seed 42

echo ""
echo "Step 2/4: Building preprocessed windows..."
echo "------------------------------------------"
# Build training windows
$PYTHON scripts/ttm_vitaldb.py build-windows \
    --channels-yaml configs/channels.yaml \
    --windows-yaml configs/windows.yaml \
    --split-file data/splits_fasttrack.json \
    --split train \
    --outdir artifacts/raw_windows/train \
    --duration-sec 60 \
    --min-sqi 0.8

# Build test windows
$PYTHON scripts/ttm_vitaldb.py build-windows \
    --channels-yaml configs/channels.yaml \
    --windows-yaml configs/windows.yaml \
    --split-file data/splits_fasttrack.json \
    --split test \
    --outdir artifacts/raw_windows/test \
    --duration-sec 60 \
    --min-sqi 0.8

echo ""
echo "Step 3/4: Training TTM model..."
echo "------------------------------------------"
$PYTHON scripts/ttm_vitaldb.py train \
    --model-yaml configs/model.yaml \
    --run-yaml configs/run.yaml \
    --split-file data/splits_fasttrack.json \
    --outdir artifacts/raw_windows \
    --out artifacts/run_ft_fast \
    --fasttrack

echo ""
echo "Step 4/4: Testing and evaluation..."
echo "------------------------------------------"
$PYTHON scripts/ttm_vitaldb.py test \
    --ckpt artifacts/run_ft_fast/best_model.pt \
    --model-yaml configs/model.yaml \
    --run-yaml configs/run.yaml \
    --split-file data/splits_fasttrack.json \
    --outdir artifacts/raw_windows \
    --out artifacts/run_ft_fast

echo ""
echo "=========================================="
echo "FastTrack Pipeline Complete!"
echo "=========================================="
echo ""
echo "Results saved to: artifacts/run_ft_fast/"
echo "  - best_model.pt: Trained model checkpoint"
echo "  - test_results.json: Test set evaluation"
echo ""
echo "To run with higher accuracy (full fine-tuning):"
echo "  bash scripts/run_high_accuracy.sh"
echo ""
echo "View training logs (if available):"
echo "  tail -f artifacts/run_ft_fast/train.log"
