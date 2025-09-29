#!/bin/bash
# Evaluation-Only Pipeline (using existing BEST MODEL)
# Evaluate → Downstream Tasks → Benchmark Comparison
# Expected runtime: ~30-60 minutes

set -e

echo "======================================================================"
echo "TTM × VitalDB - EVALUATION ONLY PIPELINE"
echo "======================================================================"
echo "Flow: Evaluate Test Set → Downstream Tasks → Benchmark Comparison"
echo "Using existing trained model"
echo "Expected runtime: 30-60 minutes"
echo "======================================================================"
echo ""

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="${PYTHON:-python3}"

# Parse arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_checkpoint> [output_dir]"
    echo ""
    echo "Example:"
    echo "  $0 artifacts/fasttrack_complete/checkpoints/best_model.pt"
    echo "  $0 artifacts/high_accuracy_complete/checkpoints/best_model.pt artifacts/eval_results"
    echo ""
    echo "Available models:"
    find artifacts -name "best_model.pt" -o -name "model.pt" | head -5
    exit 1
fi

MODEL_CHECKPOINT="$1"
OUTPUT_DIR="${2:-artifacts/evaluation_$(date +%Y%m%d_%H%M%S)}"

# Validate model exists
if [ ! -f "$MODEL_CHECKPOINT" ]; then
    echo "ERROR: Model checkpoint not found: $MODEL_CHECKPOINT"
    echo ""
    echo "Available checkpoints:"
    find artifacts -name "*.pt" | head -10
    exit 1
fi

echo "Configuration:"
echo "  Model: $MODEL_CHECKPOINT"
echo "  Output directory: $OUTPUT_DIR"
echo "  Device: $(python3 -c 'import torch; print("GPU" if torch.cuda.is_available() else "CPU")')"
echo ""

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/test_evaluation"
mkdir -p "$OUTPUT_DIR/downstream_tasks"

# Get model directory to find associated configs
MODEL_DIR=$(dirname "$MODEL_CHECKPOINT")
MODEL_PARENT=$(dirname "$MODEL_DIR")

# Try to find associated config files
MODEL_YAML="configs/model.yaml"
RUN_YAML="configs/run.yaml"
SPLITS_JSON="$MODEL_PARENT/splits.json"

if [ -f "$MODEL_DIR/model_high_accuracy.yaml" ]; then
    MODEL_YAML="$MODEL_DIR/model_high_accuracy.yaml"
    echo "  Using high-accuracy config: $MODEL_YAML"
elif [ -f "$MODEL_PARENT/model_high_accuracy.yaml" ]; then
    MODEL_YAML="$MODEL_PARENT/model_high_accuracy.yaml"
    echo "  Using high-accuracy config: $MODEL_YAML"
fi

if [ ! -f "$SPLITS_JSON" ]; then
    SPLITS_JSON="configs/splits/train_test.json"
    if [ ! -f "$SPLITS_JSON" ]; then
        SPLITS_JSON="configs/splits/train_val_test.json"
    fi
fi

echo "  Model config: $MODEL_YAML"
echo "  Run config: $RUN_YAML"
echo "  Splits file: $SPLITS_JSON"
echo ""

# Record start time
START_TIME=$(date +%s)

echo "========================================================================"
echo "STEP 1/3: Test Set Evaluation"
echo "========================================================================"
echo "Using model: $MODEL_CHECKPOINT"
echo ""

$PYTHON scripts/ttm_vitaldb.py test \
    --model-yaml "$MODEL_YAML" \
    --run-yaml "$RUN_YAML" \
    --split-file "$SPLITS_JSON" \
    --split test \
    --task clf \
    --ckpt "$MODEL_CHECKPOINT" \
    --out "$OUTPUT_DIR/test_evaluation" \
    --calibration temperature

echo ""
echo "✓ Test evaluation complete!"
echo "  Results: $OUTPUT_DIR/test_evaluation/test_results.json"
echo ""

# Display results
if [ -f "$OUTPUT_DIR/test_evaluation/test_results.json" ]; then
    echo "Test Results Summary:"
    python3 -c "
import json
try:
    with open('$OUTPUT_DIR/test_evaluation/test_results.json') as f:
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
echo "STEP 2/3: Downstream Tasks Evaluation"
echo "========================================================================"
echo "Evaluating on 8 downstream tasks..."
echo ""

# Run downstream tasks
bash scripts/evaluate_all_tasks.sh \
    "$MODEL_CHECKPOINT" \
    "$OUTPUT_DIR/downstream_tasks"

echo ""
echo "✓ Downstream tasks complete!"
echo ""

echo "========================================================================"
echo "STEP 3/3: Benchmark Comparison"
echo "========================================================================"

if [ -f "scripts/benchmark_comparison.py" ]; then
    echo "Generating benchmark comparison report..."
    $PYTHON scripts/benchmark_comparison.py \
        --results-dir "$OUTPUT_DIR/downstream_tasks" \
        --format html \
        --plot || echo "Warning: Benchmark comparison failed (non-critical)"
    echo ""
fi

# Calculate total runtime
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
MINUTES=$((TOTAL_TIME / 60))
SECONDS=$((TOTAL_TIME % 60))

echo ""
echo "======================================================================"
echo "EVALUATION PIPELINE COMPLETE!"
echo "======================================================================"
echo ""
echo "Total Runtime: ${MINUTES}m ${SECONDS}s"
echo ""
echo "Results Summary:"
echo "  Model evaluated: $MODEL_CHECKPOINT"
echo "  Output directory: $OUTPUT_DIR"
echo "  Test results: $OUTPUT_DIR/test_evaluation/test_results.json"
echo "  Downstream tasks: $OUTPUT_DIR/downstream_tasks/"
echo ""
echo "View Results:"
echo "  Test results: cat $OUTPUT_DIR/test_evaluation/test_results.json"
if [ -f "$OUTPUT_DIR/downstream_tasks/aggregate_comparison.html" ]; then
    echo "  Benchmark report: open $OUTPUT_DIR/downstream_tasks/aggregate_comparison.html"
fi
echo ""
echo "Compare Multiple Models:"
echo "  1. Evaluate each model:"
echo "     bash scripts/run_evaluation_only.sh model1.pt output1/"
echo "     bash scripts/run_evaluation_only.sh model2.pt output2/"
echo "  2. Compare results:"
echo "     diff output1/test_evaluation/test_results.json \\"
echo "          output2/test_evaluation/test_results.json"
echo ""
echo "======================================================================"
