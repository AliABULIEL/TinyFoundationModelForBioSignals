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

# Check if running in FastTrack mode (default)
FASTTRACK="${FASTTRACK:-true}"
MODE_FLAG=""
if [ "$FASTTRACK" = true ]; then
    MODE_FLAG="--fasttrack"
    echo "Running in FastTrack mode (50 train, 20 test cases)"
else
    echo "Running in full mode (all VitalDB cases)"
fi

# Create output directories
mkdir -p artifacts/raw_windows
mkdir -p artifacts/checkpoints
mkdir -p configs/splits

echo ""
echo "Step 1/4: Preparing train/test splits..."
echo "------------------------------------------"
$PYTHON scripts/ttm_vitaldb.py prepare-splits \
    --train-ratio 0.71 \
    --val-ratio 0.0 \
    --test-ratio 0.29 \
    --seed 42 \
    --out configs/splits/train_test.json \
    $MODE_FLAG

echo ""
echo "Step 2/4: Building preprocessed windows..."
echo "------------------------------------------"
# Build training windows
$PYTHON scripts/ttm_vitaldb.py build-windows \
    --channels-yaml configs/channels.yaml \
    --windows-yaml configs/windows.yaml \
    --split-file configs/splits/train_test.json \
    --split train \
    --outdir artifacts/raw_windows/train \
    --ecg-mode analysis \
    $MODE_FLAG

# Build test windows
$PYTHON scripts/ttm_vitaldb.py build-windows \
    --channels-yaml configs/channels.yaml \
    --windows-yaml configs/windows.yaml \
    --split-file configs/splits/train_test.json \
    --split test \
    --outdir artifacts/raw_windows/test \
    --ecg-mode analysis \
    $MODE_FLAG

echo ""
echo "Step 3/4: Training TTM model..."
echo "------------------------------------------"
$PYTHON scripts/ttm_vitaldb.py train \
    --model-yaml configs/model.yaml \
    --run-yaml configs/run.yaml \
    --split-file configs/splits/train_test.json \
    --split train \
    --task clf \
    --out artifacts/run_ft_fast \
    $MODE_FLAG

echo ""
echo "Step 4/4: Testing and evaluation..."
echo "------------------------------------------"
$PYTHON scripts/ttm_vitaldb.py test \
    --model-yaml configs/model.yaml \
    --run-yaml configs/run.yaml \
    --split-file configs/splits/train_test.json \
    --split test \
    --task clf \
    --ckpt artifacts/run_ft_fast/model.pt \
    --out artifacts/run_ft_fast \
    --calibration temperature

echo ""
echo "=========================================="
echo "FastTrack Pipeline Complete!"
echo "=========================================="
echo ""
echo "Results saved to: artifacts/run_ft_fast/"
echo "  - model.pt: Trained model checkpoint"
echo "  - metrics.json: Performance metrics"
echo "  - test_results.json: Test set evaluation"
echo ""
echo "To run with higher accuracy (full fine-tuning):"
echo "  1. Set FASTTRACK=false"
echo "  2. Edit configs/model.yaml:"
echo "     - freeze_encoder: false"
echo "     - unfreeze_last_n_blocks: 2"
echo "     - lora.enabled: true"
echo "  3. Re-run this script"
echo ""
echo "View training logs:"
echo "  tail -f artifacts/run_ft_fast/train.log"
echo ""
echo "Launch TensorBoard:"
echo "  tensorboard --logdir artifacts/run_ft_fast/tensorboard"
