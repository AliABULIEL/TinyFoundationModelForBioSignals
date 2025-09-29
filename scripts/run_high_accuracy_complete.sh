#!/bin/bash
# Complete End-to-End High-Accuracy Pipeline
# Train with LoRA + Partial Unfreezing → Evaluate → Downstream Tasks
# Expected runtime: ~12-24 hours on GPU

set -e

echo "======================================================================"
echo "TTM × VitalDB - COMPLETE HIGH-ACCURACY PIPELINE"
echo "======================================================================"
echo "Flow: Data Prep → Fine-tuning → Evaluation → Downstream Tasks"
echo "Model: Fine-tuned (LoRA + Partial Unfreezing)"
echo "Expected runtime: 12-24 hours on GPU"
echo "======================================================================"
echo ""

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="${PYTHON:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/high_accuracy_complete}"

# Print configuration
echo "Configuration:"
echo "  Output directory: $OUTPUT_DIR"
echo "  Mode: Full dataset (all VitalDB cases)"
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
echo "STEP 1/7: Prepare Train/Val/Test Splits (Full Dataset)"
echo "========================================================================"
$PYTHON scripts/ttm_vitaldb.py prepare-splits \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --seed 42 \
    --out "$OUTPUT_DIR/splits.json"

echo ""
echo "========================================================================"
echo "STEP 2/7: Build Preprocessed Windows (Train)"
echo "========================================================================"
echo "Processing full training set (this may take a while)..."
echo ""
$PYTHON scripts/ttm_vitaldb.py build-windows \
    --channels-yaml configs/channels.yaml \
    --windows-yaml configs/windows.yaml \
    --split-file "$OUTPUT_DIR/splits.json" \
    --split train \
    --outdir "$OUTPUT_DIR/raw_windows/train" \
    --ecg-mode analysis

echo ""
echo "========================================================================"
echo "STEP 3/7: Build Preprocessed Windows (Val + Test)"
echo "========================================================================"
# Validation windows
echo "Processing validation set..."
$PYTHON scripts/ttm_vitaldb.py build-windows \
    --channels-yaml configs/channels.yaml \
    --windows-yaml configs/windows.yaml \
    --split-file "$OUTPUT_DIR/splits.json" \
    --split val \
    --outdir "$OUTPUT_DIR/raw_windows/val" \
    --ecg-mode analysis

# Test windows
echo "Processing test set..."
$PYTHON scripts/ttm_vitaldb.py build-windows \
    --channels-yaml configs/channels.yaml \
    --windows-yaml configs/windows.yaml \
    --split-file "$OUTPUT_DIR/splits.json" \
    --split test \
    --outdir "$OUTPUT_DIR/raw_windows/test" \
    --ecg-mode analysis

echo ""
echo "========================================================================"
echo "STEP 4/7: Create High-Accuracy Model Configuration"
echo "========================================================================"

# Create high-accuracy config
cat > "$OUTPUT_DIR/model_high_accuracy.yaml" << 'EOF'
# High-Accuracy Model Configuration

model:
  variant: "ibm-granite/granite-timeseries-ttm-r1"
  input_channels: 3
  context_length: 1250
  patch_size: 16
  stride: 8

task:
  type: "classification"
  num_classes: 2

# Fine-tuning mode (NOT frozen)
training_mode:
  freeze_encoder: false
  unfreeze_last_n_blocks: 2
  
  gradual_unfreeze:
    enabled: true
    schedule: [0, 0, 2, 4]

# LoRA configuration
lora:
  enabled: true
  r: 16
  alpha: 32
  dropout: 0.1
  target_modules:
    - "attention"
    - "mixer"
    - "dense"

# Enhanced MLP head
head:
  type: "mlp"
  mlp:
    hidden_dims: [512, 256]
    activation: "gelu"
    dropout: 0.2
    batch_norm: false

# Focal loss for imbalanced data
loss:
  classification:
    type: "focal"
    focal:
      alpha: 0.25
      gamma: 2.0
    label_smoothing: 0.1

regularization:
  weight_decay: 0.01
  gradient_clip: 1.0
  early_stopping:
    patience: 10
    monitor: "val_loss"
    mode: "min"
EOF

echo "✓ High-accuracy configuration created"
echo ""

echo "========================================================================"
echo "STEP 5/7: Fine-tuning TTM Model"
echo "========================================================================"
echo "Configuration:"
echo "  - Partial unfreezing (last 2 blocks)"
echo "  - LoRA adapters (r=16, alpha=32)"
echo "  - MLP head with 2 hidden layers"
echo "  - Focal loss + label smoothing"
echo "  - Up to 50 epochs with early stopping (patience=10)"
echo ""

$PYTHON scripts/ttm_vitaldb.py train \
    --model-yaml "$OUTPUT_DIR/model_high_accuracy.yaml" \
    --run-yaml configs/run.yaml \
    --split-file "$OUTPUT_DIR/splits.json" \
    --split train \
    --task clf \
    --out "$OUTPUT_DIR/checkpoints" \
    --epochs 50 \
    --early-stopping-patience 10

echo ""
echo "✓ Fine-tuning complete!"
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
echo "STEP 6/7: Evaluation on Test Set (using BEST MODEL)"
echo "========================================================================"
echo "Using model: $BEST_MODEL"
echo "With isotonic calibration and overlapping windows"
echo ""

$PYTHON scripts/ttm_vitaldb.py test \
    --model-yaml "$OUTPUT_DIR/model_high_accuracy.yaml" \
    --run-yaml configs/run.yaml \
    --split-file "$OUTPUT_DIR/splits.json" \
    --split test \
    --task clf \
    --ckpt "$BEST_MODEL" \
    --out "$OUTPUT_DIR/evaluation" \
    --calibration isotonic

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
    if 'f1' in results:
        print('  F1-Score: {:.4f}'.format(results['f1']))
except Exception as e:
    print('  Could not parse results:', e)
"
fi

echo ""
echo "========================================================================"
echo "STEP 7/7: Downstream Tasks Evaluation (using BEST MODEL)"
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

# Generate benchmark comparison
echo "Generating benchmark comparison report..."
$PYTHON scripts/benchmark_comparison.py \
    --results-dir "$OUTPUT_DIR/downstream_tasks" \
    --format html \
    --plot

# Calculate total runtime
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))

echo ""
echo "======================================================================"
echo "COMPLETE HIGH-ACCURACY PIPELINE FINISHED!"
echo "======================================================================"
echo ""
echo "Total Runtime: ${HOURS}h ${MINUTES}m"
echo ""
echo "Results Summary:"
echo "  Output directory: $OUTPUT_DIR"
echo "  Best model: $BEST_MODEL"
echo "  Test results: $OUTPUT_DIR/evaluation/test_results.json"
echo "  Downstream tasks: $OUTPUT_DIR/downstream_tasks/"
echo "  Benchmark comparison: $OUTPUT_DIR/downstream_tasks/aggregate_comparison.html"
echo ""
echo "Performance Improvements (vs FastTrack):"
echo "  ✓ Fine-tuned encoder (last 2 blocks unfrozen)"
echo "  ✓ LoRA adapters for efficient parameter updates"
echo "  ✓ Enhanced MLP head for better capacity"
echo "  ✓ Focal loss for imbalanced data"
echo "  ✓ Isotonic calibration for better uncertainty"
echo ""
echo "Generated Files:"
ls -lh "$OUTPUT_DIR/checkpoints/"*.pt 2>/dev/null | awk '{print "  "$9" ("$5")"}'
echo ""
echo "View Results:"
echo "  Training metrics: cat $OUTPUT_DIR/checkpoints/metrics.json"
echo "  Test results: cat $OUTPUT_DIR/evaluation/test_results.json"
echo "  Benchmark report: open $OUTPUT_DIR/downstream_tasks/aggregate_comparison.html"
echo ""
echo "Compare with FastTrack:"
echo "  diff artifacts/fasttrack_complete/evaluation/test_results.json \\"
echo "       artifacts/high_accuracy_complete/evaluation/test_results.json"
echo ""
echo "======================================================================"
