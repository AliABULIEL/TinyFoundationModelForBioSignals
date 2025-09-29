#!/bin/bash
# Complete End-to-End FastTrack Pipeline
# Train → Evaluate → Downstream Tasks (using BEST MODEL)
# Expected runtime: ~3-4 hours on single GPU

set -e

echo "======================================================================"
echo "TTM × VitalDB - COMPLETE FASTTRACK PIPELINE"
echo "======================================================================"
echo "Flow: Data Prep → Training → Evaluation → Downstream Tasks"
echo "Model: Foundation Model (Frozen Encoder + Linear Head)"
echo "Expected runtime: ~3-4 hours on single GPU"
echo "======================================================================"
echo ""

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="${PYTHON:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/fasttrack_complete}"
FASTTRACK="--fasttrack"

# Print configuration
echo "Configuration:"
echo "  Output directory: $OUTPUT_DIR"
echo "  FastTrack mode: Enabled (50 train cases, 20 test cases)"
echo "  Device: $(python3 -c 'import torch; print("GPU" if torch.cuda.is_available() else "CPU")')"
echo ""

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/raw_windows"
mkdir -p "$OUTPUT_DIR/checkpoints"
mkdir -p "$OUTPUT_DIR/evaluation"
mkdir -p "$OUTPUT_DIR/downstream_tasks"
mkdir -p configs/splits

# Record start time
START_TIME=$(date +%s)

echo ""
echo "========================================================================"
echo "STEP 1/6: Prepare Train/Val/Test Splits"
echo "========================================================================"
$PYTHON scripts/ttm_vitaldb.py prepare-splits \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --seed 42 \
    --out "$OUTPUT_DIR/splits.json" \
    $FASTTRACK

echo ""
echo "========================================================================"
echo "STEP 2/6: Build Preprocessed Windows (Train)"
echo "========================================================================"
$PYTHON scripts/ttm_vitaldb.py build-windows \
    --channels-yaml configs/channels.yaml \
    --windows-yaml configs/windows.yaml \
    --split-file "$OUTPUT_DIR/splits.json" \
    --split train \
    --outdir "$OUTPUT_DIR/raw_windows/train" \
    --ecg-mode analysis \
    $FASTTRACK

echo ""
echo "========================================================================"
echo "STEP 3/6: Build Preprocessed Windows (Val + Test)"
echo "========================================================================"
# Validation windows
$PYTHON scripts/ttm_vitaldb.py build-windows \
    --channels-yaml configs/channels.yaml \
    --windows-yaml configs/windows.yaml \
    --split-file "$OUTPUT_DIR/splits.json" \
    --split val \
    --outdir "$OUTPUT_DIR/raw_windows/val" \
    --ecg-mode analysis \
    $FASTTRACK

# Test windows
$PYTHON scripts/ttm_vitaldb.py build-windows \
    --channels-yaml configs/channels.yaml \
    --windows-yaml configs/windows.yaml \
    --split-file "$OUTPUT_DIR/splits.json" \
    --split test \
    --outdir "$OUTPUT_DIR/raw_windows/test" \
    --ecg-mode analysis \
    $FASTTRACK

echo ""
echo "========================================================================"
echo "STEP 4/6: Training TTM Model (Foundation Model Mode)"
echo "========================================================================"
echo "Configuration:"
echo "  - Frozen encoder (805K params frozen)"
echo "  - Linear head only (~290K trainable params)"
echo "  - 10 epochs with early stopping"
echo "  - Batch size: 128"
echo ""

$PYTHON scripts/ttm_vitaldb.py train \
    --model-yaml configs/model.yaml \
    --run-yaml configs/run.yaml \
    --split-file "$OUTPUT_DIR/splits.json" \
    --split train \
    --task clf \
    --out "$OUTPUT_DIR/checkpoints" \
    $FASTTRACK

echo ""
echo "✓ Training complete!"
echo "  Best model: $OUTPUT_DIR/checkpoints/best_model.pt"
echo "  Last checkpoint: $OUTPUT_DIR/checkpoints/last_checkpoint.pt"
echo "  Metrics: $OUTPUT_DIR/checkpoints/metrics.json"
echo ""

# Find the best model
BEST_MODEL="$OUTPUT_DIR/checkpoints/best_model.pt"
if [ ! -f "$BEST_MODEL" ]; then
    echo "ERROR: Best model not found at $BEST_MODEL"
    echo "Checking for alternatives..."
    if [ -f "$OUTPUT_DIR/checkpoints/model.pt" ]; then
        BEST_MODEL="$OUTPUT_DIR/checkpoints/model.pt"
        echo "Using $BEST_MODEL instead"
    else
        echo "No model checkpoint found. Exiting."
        exit 1
    fi
fi

echo "========================================================================"
echo "STEP 5/6: Evaluation on Test Set (using BEST MODEL)"
echo "========================================================================"
echo "Using model: $BEST_MODEL"
echo ""

$PYTHON scripts/ttm_vitaldb.py test \
    --model-yaml configs/model.yaml \
    --run-yaml configs/run.yaml \
    --split-file "$OUTPUT_DIR/splits.json" \
    --split test \
    --task clf \
    --ckpt "$BEST_MODEL" \
    --out "$OUTPUT_DIR/evaluation" \
    --calibration temperature

echo ""
echo "✓ Evaluation complete!"
echo "  Test results: $OUTPUT_DIR/evaluation/test_results.json"
echo ""

# Display test results
echo "Test Results Summary:"
if [ -f "$OUTPUT_DIR/evaluation/test_results.json" ]; then
    python3 -c "
import json
import sys
try:
    with open('$OUTPUT_DIR/evaluation/test_results.json') as f:
        results = json.load(f)
    print('  Accuracy: {:.4f}'.format(results.get('accuracy', 0)))
    print('  Loss: {:.4f}'.format(results.get('loss', 0)))
    if 'auroc' in results:
        print('  AUROC: {:.4f}'.format(results['auroc']))
except Exception as e:
    print('  Could not parse results:', e)
"
fi

echo ""
echo "========================================================================"
echo "STEP 6/6: Downstream Tasks Evaluation (using BEST MODEL)"
echo "========================================================================"
echo "Evaluating on 8 downstream tasks..."
echo ""

# Run downstream tasks evaluation
bash scripts/evaluate_all_tasks.sh \
    "$BEST_MODEL" \
    "$OUTPUT_DIR/downstream_tasks"

echo ""
echo "✓ Downstream tasks complete!"
echo "  Results: $OUTPUT_DIR/downstream_tasks/"
echo ""

# Calculate total runtime
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))

echo ""
echo "======================================================================"
echo "COMPLETE FASTTRACK PIPELINE FINISHED!"
echo "======================================================================"
echo ""
echo "Total Runtime: ${HOURS}h ${MINUTES}m"
echo ""
echo "Results Summary:"
echo "  Output directory: $OUTPUT_DIR"
echo "  Best model: $BEST_MODEL"
echo "  Test results: $OUTPUT_DIR/evaluation/test_results.json"
echo "  Downstream tasks: $OUTPUT_DIR/downstream_tasks/"
echo ""
echo "Generated Files:"
ls -lh "$OUTPUT_DIR/checkpoints/"*.pt 2>/dev/null | awk '{print "  "$9" ("$5")"}'
echo ""
echo "View Results:"
echo "  Training metrics: cat $OUTPUT_DIR/checkpoints/metrics.json"
echo "  Test results: cat $OUTPUT_DIR/evaluation/test_results.json"
echo "  Downstream comparison: cat $OUTPUT_DIR/downstream_tasks/aggregate_comparison.html"
echo ""
echo "Next Steps:"
echo "  1. View training curves: tensorboard --logdir $OUTPUT_DIR/checkpoints/tensorboard"
echo "  2. Compare to benchmarks: python scripts/benchmark_comparison.py --results-dir $OUTPUT_DIR/downstream_tasks"
echo "  3. Try high-accuracy mode: bash scripts/run_high_accuracy_complete.sh"
echo ""
echo "======================================================================"
