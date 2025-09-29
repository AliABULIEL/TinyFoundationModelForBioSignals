#!/bin/bash
# Evaluate TTM model on all VitalDB downstream tasks
#
# Usage:
#   bash scripts/evaluate_all_tasks.sh <checkpoint_path> <output_dir> [max_cases]
#
# Example:
#   bash scripts/evaluate_all_tasks.sh artifacts/model.pt results/evaluation 100

set -e  # Exit on error

# Parse arguments
CHECKPOINT=$1
OUTPUT_DIR=$2
MAX_CASES=${3:-""}

# Validate inputs
if [ -z "$CHECKPOINT" ]; then
    echo "Error: Checkpoint path required"
    echo "Usage: bash scripts/evaluate_all_tasks.sh <checkpoint> <output_dir> [max_cases]"
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Output directory required"
    echo "Usage: bash scripts/evaluate_all_tasks.sh <checkpoint> <output_dir> [max_cases]"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "============================================================"
echo "VitalDB Downstream Tasks - Batch Evaluation"
echo "============================================================"
echo "Checkpoint: $CHECKPOINT"
echo "Output dir: $OUTPUT_DIR"
if [ -n "$MAX_CASES" ]; then
    echo "Max cases: $MAX_CASES"
fi
echo "============================================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Define tasks to evaluate
TASKS=(
    "hypotension_5min"
    "hypotension_10min"
    "blood_pressure_both"
    "cardiac_output"
    "mortality_30day"
    "icu_admission"
    "signal_quality"
)

# Track success/failure
SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_TASKS=()

# Evaluate each task
for TASK in "${TASKS[@]}"
do
    echo ""
    echo "============================================================"
    echo "Evaluating: $TASK"
    echo "============================================================"
    
    # Build command
    CMD="python scripts/evaluate_task.py \
        --task $TASK \
        --checkpoint $CHECKPOINT \
        --output-dir $OUTPUT_DIR/$TASK \
        --compare-benchmarks \
        --generate-report"
    
    # Add max cases if specified
    if [ -n "$MAX_CASES" ]; then
        CMD="$CMD --max-cases $MAX_CASES"
    fi
    
    # Run evaluation
    if eval $CMD; then
        echo "✓ SUCCESS: $TASK"
        ((SUCCESS_COUNT++))
    else
        echo "✗ FAILED: $TASK"
        ((FAIL_COUNT++))
        FAILED_TASKS+=("$TASK")
    fi
done

echo ""
echo "============================================================"
echo "BATCH EVALUATION SUMMARY"
echo "============================================================"
echo "Total tasks: ${#TASKS[@]}"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT"

if [ $FAIL_COUNT -gt 0 ]; then
    echo ""
    echo "Failed tasks:"
    for TASK in "${FAILED_TASKS[@]}"; do
        echo "  - $TASK"
    done
fi

echo "============================================================"

# Generate aggregate comparison
if [ $SUCCESS_COUNT -gt 0 ]; then
    echo ""
    echo "Generating aggregate comparison report..."
    python scripts/benchmark_comparison.py \
        --results-dir "$OUTPUT_DIR" \
        --format all \
        --output "$OUTPUT_DIR/aggregate_comparison" \
        --plot
    
    echo "✓ Aggregate report saved to: $OUTPUT_DIR/aggregate_comparison.html"
fi

echo ""
echo "============================================================"
echo "EVALUATION COMPLETE"
echo "============================================================"
echo "Results directory: $OUTPUT_DIR"
echo ""
echo "View results:"
echo "  - Individual tasks: $OUTPUT_DIR/<task_name>/"
echo "  - Aggregate report: $OUTPUT_DIR/aggregate_comparison.html"
echo "  - Summary CSV: $OUTPUT_DIR/aggregate_comparison.csv"
echo "  - Plots: $OUTPUT_DIR/plots/"
echo "============================================================"

# Exit with appropriate code
if [ $FAIL_COUNT -gt 0 ]; then
    exit 1
else
    exit 0
fi
