#!/bin/bash
# Complete End-to-End Full Fine-tuning Pipeline
# Deep fine-tuning with extensive unfreezing
# Expected runtime: ~24-48 hours on GPU

set -e

echo "======================================================================"
echo "TTM × VitalDB - FULL FINE-TUNING PIPELINE"
echo "======================================================================"
echo "Flow: Data Prep → Deep Fine-tuning → Evaluation → Downstream Tasks"
echo "Model: Deep Fine-tuning (Unfreeze ALL or most blocks)"
echo "Expected runtime: 24-48 hours on GPU"
echo "======================================================================"
echo ""

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="${PYTHON:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/full_finetune_complete}"

echo "Configuration:"
echo "  Output directory: $OUTPUT_DIR"
echo "  Mode: Full dataset with deep fine-tuning"
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
echo "STEP 1/7: Prepare Train/Val/Test Splits"
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
$PYTHON scripts/ttm_vitaldb.py build-windows \
    --channels-yaml configs/channels.yaml \
    --windows-yaml configs/windows.yaml \
    --split-file "$OUTPUT_DIR/splits.json" \
    --split val \
    --outdir "$OUTPUT_DIR/raw_windows/val" \
    --ecg-mode analysis

# Test windows
$PYTHON scripts/ttm_vitaldb.py build-windows \
    --channels-yaml configs/channels.yaml \
    --windows-yaml configs/windows.yaml \
    --split-file "$OUTPUT_DIR/splits.json" \
    --split test \
    --outdir "$OUTPUT_DIR/raw_windows/test" \
    --ecg-mode analysis

echo ""
echo "========================================================================"
echo "STEP 4/7: Create Full Fine-tuning Configuration"
echo "========================================================================"

# Create full fine-tuning config
cat > "$OUTPUT_DIR/model_full_finetune.yaml" << 'EOF'
# Full Fine-tuning Configuration

model:
  variant: "ibm-granite/granite-timeseries-ttm-r1"
  input_channels: 3
  context_length: 1250
  patch_size: 16
  stride: 8

task:
  type: "classification"
  num_classes: 2

# Full fine-tuning mode
training_mode:
  freeze_encoder: false
  unfreeze_last_n_blocks: 6  # Unfreeze MOST blocks (out of 8 total)
  
  gradual_unfreeze:
    enabled: true
    schedule: [0, 2, 4, 6, 6, 6]  # Progressive unfreezing

# LoRA for efficient adaptation on top of unfreezing
lora:
  enabled: true
  r: 32  # Higher rank for more capacity
  alpha: 64
  dropout: 0.05
  target_modules:
    - "attention"
    - "mixer"
    - "dense"
    - "query"
    - "key"
    - "value"

# Deep MLP head
head:
  type: "mlp"
  mlp:
    hidden_dims: [768, 512, 256]
    activation: "gelu"
    dropout: 0.15
    batch_norm: false

# Advanced loss
loss:
  classification:
    type: "focal"
    focal:
      alpha: 0.25
      gamma: 2.5
    label_smoothing: 0.15

regularization:
  weight_decay: 0.005  # Lower for fine-tuning
  gradient_clip: 0.5
  early_stopping:
    patience: 15
    monitor: "val_loss"
    mode: "min"

mixed_precision:
  enabled: true
EOF

echo "✓ Full fine-tuning configuration created"
echo "  - Unfreezing 6/8 transformer blocks"
echo "  - LoRA rank 32 (high capacity)"
echo "  - 3-layer MLP head"
echo ""

echo "========================================================================"
echo "STEP 5/7: Deep Fine-tuning"
echo "========================================================================"
echo "Configuration:"
echo "  - Unfreeze 6 out of 8 blocks (75% of encoder)"
echo "  - LoRA adapters (r=32, alpha=64) on top"
echo "  - Deep MLP head (3 hidden layers)"
echo "  - Lower weight decay for fine-tuning"
echo "  - Up to 100 epochs with early stopping (patience=15)"
echo "  - Mixed precision training"
echo ""
echo "⚠️  This will take a LONG time (24-48 hours)"
echo ""

$PYTHON scripts/ttm_vitaldb.py train \
    --model-yaml "$OUTPUT_DIR/model_full_finetune.yaml" \
    --run-yaml configs/run.yaml \
    --split-file "$OUTPUT_DIR/splits.json" \
    --split train \
    --task clf \
    --out "$OUTPUT_DIR/checkpoints" \
    --epochs 100 \
    --early-stopping-patience 15

echo ""
echo "✓ Deep fine-tuning complete!"
echo "  Best model: $OUTPUT_DIR/checkpoints/best_model.pt"
echo "  Last checkpoint: $OUTPUT_DIR/checkpoints/last_checkpoint.pt"
echo ""

# Find the best model
BEST_MODEL="$OUTPUT_DIR/checkpoints/best_model.pt"
if [ ! -f "$BEST_MODEL" ]; then
    if [ -f "$OUTPUT_DIR/checkpoints/model.pt" ]; then
        BEST_MODEL="$OUTPUT_DIR/checkpoints/model.pt"
    else
        echo "ERROR: No model checkpoint found. Exiting."
        exit 1
    fi
fi

echo "========================================================================"
echo "STEP 6/7: Evaluation on Test Set (using BEST MODEL)"
echo "========================================================================"
echo "Using model: $BEST_MODEL"
echo ""

$PYTHON scripts/ttm_vitaldb.py test \
    --model-yaml "$OUTPUT_DIR/model_full_finetune.yaml" \
    --run-yaml configs/run.yaml \
    --split-file "$OUTPUT_DIR/splits.json" \
    --split test \
    --task clf \
    --ckpt "$BEST_MODEL" \
    --out "$OUTPUT_DIR/evaluation" \
    --calibration isotonic

echo ""
echo "✓ Evaluation complete!"
echo ""

# Display results
if [ -f "$OUTPUT_DIR/evaluation/test_results.json" ]; then
    echo "Test Results Summary:"
    python3 -c "
import json
try:
    with open('$OUTPUT_DIR/evaluation/test_results.json') as f:
        results = json.load(f)
    print('  Accuracy: {:.4f}'.format(results.get('accuracy', 0)))
    print('  Loss: {:.4f}'.format(results.get('loss', 0)))
    if 'auroc' in results:
        print('  AUROC: {:.4f}'.format(results['auroc']))
    if 'f1' in results:
        print('  F1-Score: {:.4f}'.format(results['f1']))
except:
    pass
"
fi

echo ""
echo "========================================================================"
echo "STEP 7/7: Downstream Tasks Evaluation (using BEST MODEL)"
echo "========================================================================"

bash scripts/evaluate_all_tasks.sh \
    "$BEST_MODEL" \
    "$OUTPUT_DIR/downstream_tasks"

echo ""
echo "✓ Downstream tasks complete!"
echo ""

# Generate benchmark comparison
if [ -f "scripts/benchmark_comparison.py" ]; then
    $PYTHON scripts/benchmark_comparison.py \
        --results-dir "$OUTPUT_DIR/downstream_tasks" \
        --format html \
        --plot || true
fi

# Calculate total runtime
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))

echo ""
echo "======================================================================"
echo "FULL FINE-TUNING PIPELINE COMPLETE!"
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
echo "Fine-tuning Strategy:"
echo "  ✓ 6/8 transformer blocks unfrozen (75%)"
echo "  ✓ LoRA adapters (rank 32) on top"
echo "  ✓ Deep MLP head (3 layers)"
echo "  ✓ ~60% of encoder parameters trained"
echo "  ✓ Optimal for maximum performance"
echo ""
echo "Compare Performance:"
echo "  FastTrack:     artifacts/fasttrack_complete/evaluation/test_results.json"
echo "  High-Accuracy: artifacts/high_accuracy_complete/evaluation/test_results.json"
echo "  Full Finetune: $OUTPUT_DIR/evaluation/test_results.json"
echo ""
echo "======================================================================"
