#!/bin/bash
# Compare Results from Multiple Training Modes
# Generates side-by-side comparison of different models

set -e

echo "======================================================================"
echo "TTM × VitalDB - TRAINING MODES COMPARISON"
echo "======================================================================"
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Find all test_results.json files
echo "Searching for completed training runs..."
echo ""

declare -a RESULTS_FILES
declare -a RESULTS_NAMES

# Check common output directories
SEARCH_DIRS=(
    "artifacts/fasttrack_complete/evaluation"
    "artifacts/high_accuracy_complete/evaluation"
    "artifacts/full_finetune_complete/evaluation"
)

for dir in "${SEARCH_DIRS[@]}"; do
    if [ -f "$PROJECT_ROOT/$dir/test_results.json" ]; then
        RESULTS_FILES+=("$PROJECT_ROOT/$dir/test_results.json")
        # Extract name from path
        NAME=$(echo "$dir" | sed 's/artifacts\///' | sed 's/\/evaluation//')
        RESULTS_NAMES+=("$NAME")
    fi
done

# Also find any other results
while IFS= read -r -d '' file; do
    # Skip if already in list
    if [[ ! " ${RESULTS_FILES[@]} " =~ " ${file} " ]]; then
        RESULTS_FILES+=("$file")
        # Extract relative path
        rel_path="${file#$PROJECT_ROOT/}"
        NAME=$(echo "$rel_path" | sed 's/\/test_results.json//')
        RESULTS_NAMES+=("$NAME")
    fi
done < <(find "$PROJECT_ROOT/artifacts" -name "test_results.json" -print0 2>/dev/null)

if [ ${#RESULTS_FILES[@]} -eq 0 ]; then
    echo "No test results found!"
    echo ""
    echo "Run one of these pipelines first:"
    echo "  bash scripts/run_fasttrack_complete.sh"
    echo "  bash scripts/run_high_accuracy_complete.sh"
    echo "  bash scripts/run_full_finetune_complete.sh"
    exit 1
fi

echo "Found ${#RESULTS_FILES[@]} completed runs:"
echo ""

# Print comparison table
echo "======================================================================"
echo "RESULTS COMPARISON"
echo "======================================================================"
printf "%-30s | %8s | %8s | %8s | %8s\n" "Training Mode" "Accuracy" "Loss" "AUROC" "F1"
echo "----------------------------------------------------------------------"

for i in "${!RESULTS_FILES[@]}"; do
    file="${RESULTS_FILES[$i]}"
    name="${RESULTS_NAMES[$i]}"
    
    # Extract metrics
    python3 << EOF
import json
import sys

try:
    with open("$file") as f:
        results = json.load(f)
    
    accuracy = results.get('accuracy', 0)
    loss = results.get('loss', 0)
    auroc = results.get('auroc', results.get('roc_auc', 0))
    f1 = results.get('f1', results.get('f1_score', 0))
    
    # Truncate name if too long
    name = "$name"
    if len(name) > 30:
        name = name[:27] + "..."
    
    print(f"{name:30s} | {accuracy:8.4f} | {loss:8.4f} | {auroc:8.4f} | {f1:8.4f}")
except Exception as e:
    print(f"$name: Error reading file")
EOF

done

echo "======================================================================"
echo ""

# Find best model
echo "Identifying best model..."
echo ""

BEST_ACC=0
BEST_ACC_NAME=""
BEST_ACC_FILE=""

for i in "${!RESULTS_FILES[@]}"; do
    file="${RESULTS_FILES[$i]}"
    name="${RESULTS_NAMES[$i]}"
    
    ACC=$(python3 -c "import json; print(json.load(open('$file')).get('accuracy', 0))")
    
    if (( $(echo "$ACC > $BEST_ACC" | bc -l) )); then
        BEST_ACC=$ACC
        BEST_ACC_NAME=$name
        BEST_ACC_FILE=$file
    fi
done

echo "🏆 Best Model (by accuracy):"
echo "  Name: $BEST_ACC_NAME"
echo "  Accuracy: $BEST_ACC"
echo "  Results: $BEST_ACC_FILE"
echo ""

# Generate performance comparison chart
echo "Generating visual comparison..."

python3 << 'PYTHON_SCRIPT'
import json
import sys
from pathlib import Path

# Read all results
results_data = []
files = [
PYTHON_SCRIPT

# Add file paths to Python script
for file in "${RESULTS_FILES[@]}"; do
    echo "    Path('$file')," >> /tmp/compare_script.py
done

cat >> /tmp/compare_script.py << 'PYTHON_SCRIPT'
]

names = [
PYTHON_SCRIPT

# Add names
for name in "${RESULTS_NAMES[@]}"; do
    echo "    '$name'," >> /tmp/compare_script.py
done

cat >> /tmp/compare_script.py << 'PYTHON_SCRIPT'
]

for i, file_path in enumerate(files):
    try:
        with open(file_path) as f:
            data = json.load(f)
        results_data.append({
            'name': names[i],
            'accuracy': data.get('accuracy', 0),
            'loss': data.get('loss', 0),
            'auroc': data.get('auroc', data.get('roc_auc', 0)),
            'f1': data.get('f1', data.get('f1_score', 0))
        })
    except:
        pass

# Sort by accuracy
results_data.sort(key=lambda x: x['accuracy'], reverse=True)

print("\n" + "="*70)
print("RANKING BY ACCURACY")
print("="*70)
for i, result in enumerate(results_data, 1):
    print(f"{i}. {result['name']}")
    print(f"   Accuracy: {result['accuracy']:.4f}")
    print(f"   AUROC: {result['auroc']:.4f}")
    print(f"   F1: {result['f1']:.4f}")
    print()

# Performance improvements
if len(results_data) >= 2:
    baseline = results_data[-1]  # Worst performing
    best = results_data[0]  # Best performing
    
    acc_improvement = (best['accuracy'] - baseline['accuracy']) / baseline['accuracy'] * 100
    auroc_improvement = (best['auroc'] - baseline['auroc']) / baseline['auroc'] * 100 if baseline['auroc'] > 0 else 0
    
    print("="*70)
    print("PERFORMANCE IMPROVEMENT")
    print("="*70)
    print(f"Baseline: {baseline['name']}")
    print(f"Best: {best['name']}")
    print(f"Accuracy improvement: +{acc_improvement:.2f}%")
    if auroc_improvement > 0:
        print(f"AUROC improvement: +{auroc_improvement:.2f}%")
    print("="*70)
PYTHON_SCRIPT

python3 /tmp/compare_script.py
rm /tmp/compare_script.py

echo ""
echo "======================================================================"
echo "RECOMMENDATIONS"
echo "======================================================================"
echo ""

if [ ${#RESULTS_FILES[@]} -eq 1 ]; then
    echo "Only one training mode completed."
    echo ""
    echo "Try other modes for comparison:"
    echo "  - FastTrack (~3 hours): bash scripts/run_fasttrack_complete.sh"
    echo "  - High-Accuracy (~12-24 hours): bash scripts/run_high_accuracy_complete.sh"
    echo "  - Full Fine-tune (~24-48 hours): bash scripts/run_full_finetune_complete.sh"
elif (( $(echo "$BEST_ACC < 0.85" | bc -l) )); then
    echo "⚠️  All models have accuracy < 85%"
    echo ""
    echo "Suggestions to improve:"
    echo "  1. Check data quality and preprocessing"
    echo "  2. Try full fine-tuning: bash scripts/run_full_finetune_complete.sh"
    echo "  3. Increase training epochs"
    echo "  4. Adjust learning rate or batch size"
    echo "  5. Enable data augmentation in configs/channels.yaml"
else
    echo "✓ Models performing well (accuracy ≥ 85%)"
    echo ""
    echo "Next steps:"
    echo "  1. Use best model for downstream tasks"
    echo "  2. Compare with published benchmarks"
    echo "  3. Deploy for production use"
fi

echo ""
echo "======================================================================"
echo ""
echo "View detailed results:"
for i in "${!RESULTS_FILES[@]}"; do
    echo "  ${RESULTS_NAMES[$i]}:"
    echo "    cat ${RESULTS_FILES[$i]}"
done

echo ""
echo "Run downstream tasks on best model:"
MODEL_DIR=$(dirname "$BEST_ACC_FILE")
MODEL_DIR=$(dirname "$MODEL_DIR")
BEST_MODEL="$MODEL_DIR/checkpoints/best_model.pt"
if [ -f "$BEST_MODEL" ]; then
    echo "  bash scripts/run_evaluation_only.sh $BEST_MODEL"
fi

echo ""
echo "======================================================================"
