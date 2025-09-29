#!/bin/bash
"""
Quick end-to-end test of TTM VitalDB pipeline
Runs in ~2-3 minutes with minimal data
Tests: data loading → window creation → training → inference
"""

set -e  # Exit on error

echo "============================================"
echo "TTM × VitalDB Pipeline Quick Test (~2-3 min)"
echo "============================================"

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
TEST_DIR="$PROJECT_ROOT/test_run_$(date +%Y%m%d_%H%M%S)"

# Time limit (kill after 5 minutes)
TIMEOUT_SECONDS=300

echo ""
echo "Project root: $PROJECT_ROOT"
echo "Test directory: $TEST_DIR"
echo "Time limit: ${TIMEOUT_SECONDS}s"
echo ""

# Create test directory
mkdir -p "$TEST_DIR"

# Function to run command with timeout
run_with_timeout() {
    local cmd="$1"
    local desc="$2"
    
    echo ""
    echo "[$desc]"
    echo "Command: $cmd"
    echo "---"
    
    # Run with timeout
    timeout ${TIMEOUT_SECONDS} bash -c "$cmd" 2>&1 | tee "$TEST_DIR/${desc// /_}.log"
    
    if [ ${PIPESTATUS[0]} -eq 124 ]; then
        echo "✗ Timeout after ${TIMEOUT_SECONDS}s"
        exit 1
    elif [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "✗ Command failed"
        exit 1
    else
        echo "✓ Success"
    fi
}

# Start timer
START_TIME=$(date +%s)

echo "============================================"
echo "STEP 1: Test VitalDB Data Loading"
echo "============================================"

# Quick test of VitalDB loader
run_with_timeout "cd $PROJECT_ROOT && python3 -c '
from src.data.vitaldb_loader import load_channel, list_cases, get_available_case_sets
import numpy as np

# Test case sets
sets = get_available_case_sets()
print(f\"Available sets: {list(sets.keys())}\")

# Test loading 1 case
cases = list_cases(case_set=\"bis\", max_cases=1)
if cases:
    case_id = cases[0][\"case_id\"]
    print(f\"Testing case {case_id}...\")
    signal, fs = load_channel(case_id, \"PLETH\", duration_sec=10, auto_fix_alternating=True)
    print(f\"Loaded {len(signal)} samples at {fs} Hz\")
    print(f\"Signal OK: mean={np.mean(signal):.2f}, std={np.std(signal):.2f}\")
else:
    print(\"No cases found\")
'" "Test VitalDB Loader"

echo ""
echo "============================================"
echo "STEP 2: Create Mini Splits (3 train, 2 test)"
echo "============================================"

# Create minimal splits for testing
run_with_timeout "cd $PROJECT_ROOT && python3 scripts/ttm_vitaldb_fixed.py prepare-splits \
    --train-ratio 0.6 \
    --val-ratio 0.0 \
    --test-ratio 0.4 \
    --mode fasttrack \
    --out $TEST_DIR/splits.json" "Create Splits"

# Check splits
echo ""
echo "Splits created:"
python3 -c "
import json
with open('$TEST_DIR/splits.json') as f:
    splits = json.load(f)
    for k, v in splits.items():
        print(f'  {k}: {len(v)} cases')
"

echo ""
echo "============================================"
echo "STEP 3: Build Windows (10s windows)"
echo "============================================"

# Create minimal configs
cat > "$TEST_DIR/channels.yaml" << EOF
ppg:
  sampling_rate: 50
  filter:
    low_freq: 0.5
    high_freq: 10
    type: butterworth
    order: 4
EOF

cat > "$TEST_DIR/windows.yaml" << EOF
window_size_sec: 10
hop_size_sec: 10  # No overlap for speed
min_quality: 0.5  # Lenient for testing
EOF

# Build windows for train (limit to 3 cases)
run_with_timeout "cd $PROJECT_ROOT && python3 scripts/ttm_vitaldb_fixed.py build-windows \
    --channels-yaml $TEST_DIR/channels.yaml \
    --windows-yaml $TEST_DIR/windows.yaml \
    --split-file $TEST_DIR/splits.json \
    --split train \
    --outdir $TEST_DIR" "Build Train Windows"

# Build windows for test (limit to 2 cases)
run_with_timeout "cd $PROJECT_ROOT && python3 scripts/ttm_vitaldb_fixed.py build-windows \
    --channels-yaml $TEST_DIR/channels.yaml \
    --windows-yaml $TEST_DIR/windows.yaml \
    --split-file $TEST_DIR/splits.json \
    --split test \
    --outdir $TEST_DIR" "Build Test Windows"

# Check windows
echo ""
echo "Windows created:"
for file in $TEST_DIR/*_windows.npz; do
    if [ -f "$file" ]; then
        python3 -c "
import numpy as np
data = np.load('$file')
print(f'  $(basename $file): {data[\"windows\"].shape}')
"
    fi
done

echo ""
echo "============================================"
echo "STEP 4: Quick Training (2 epochs)"
echo "============================================"

# Create minimal model config
cat > "$TEST_DIR/model.yaml" << EOF
model_source: huggingface
freeze_encoder: true
head_type: linear
num_classes: 2
task: classification
EOF

cat > "$TEST_DIR/run.yaml" << EOF
batch_size: 8
num_epochs: 2
learning_rate: 0.001
num_workers: 0
seed: 42
EOF

# Train model (just 2 epochs for testing)
run_with_timeout "cd $PROJECT_ROOT && python3 scripts/ttm_vitaldb_fixed.py train \
    --model-yaml $TEST_DIR/model.yaml \
    --run-yaml $TEST_DIR/run.yaml \
    --split-file $TEST_DIR/splits.json \
    --out $TEST_DIR \
    --fasttrack" "Train Model (2 epochs)"

echo ""
echo "============================================"
echo "STEP 5: Quick Inference Test"
echo "============================================"

# Test inference
run_with_timeout "cd $PROJECT_ROOT && python3 scripts/ttm_vitaldb_fixed.py test \
    --model-yaml $TEST_DIR/model.yaml \
    --run-yaml $TEST_DIR/run.yaml \
    --split-file $TEST_DIR/splits.json \
    --ckpt $TEST_DIR/model.pt \
    --out $TEST_DIR" "Test Inference"

# Calculate total time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo ""
echo "============================================"
echo "PIPELINE TEST COMPLETED"
echo "============================================"
echo "Total time: ${TOTAL_TIME}s"
echo "Test directory: $TEST_DIR"
echo ""

# Check for output files
echo "Output files created:"
ls -lh "$TEST_DIR" | grep -E "\.(json|npz|pt|log)$" | awk '{print "  - "$9" ("$5")"}'

echo ""
if [ $TOTAL_TIME -lt 300 ]; then
    echo "✅ SUCCESS: Pipeline completed in under 5 minutes!"
else
    echo "⚠️ WARNING: Pipeline took longer than expected"
fi

echo ""
echo "To inspect results:"
echo "  cat $TEST_DIR/*.log"
echo "  python3 -c \"import json; print(json.load(open('$TEST_DIR/test_results.json')))\""
echo ""
echo "To clean up:"
echo "  rm -rf $TEST_DIR"
