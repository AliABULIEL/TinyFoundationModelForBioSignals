#!/bin/bash
# One-command quick start for VitalDB Downstream Tasks
#
# This script runs a complete demo:
# 1. Lists available tasks
# 2. Runs quick start example
# 3. Shows sample evaluation
#
# Usage:
#   bash scripts/quickstart_demo.sh

set -e

echo "============================================================"
echo "🏥 VitalDB Downstream Tasks - Quick Start Demo"
echo "============================================================"
echo ""

# Check Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Error: Python not found"
    exit 1
fi

echo "✓ Python found: $(python --version)"
echo ""

# Step 1: List available tasks
echo "============================================================"
echo "📋 Step 1: Available Tasks"
echo "============================================================"
echo ""

python scripts/evaluate_task.py --list-tasks

echo ""
read -p "Press Enter to continue..."
echo ""

# Step 2: Show task information
echo "============================================================"
echo "📖 Step 2: Task Information (Hypotension Prediction)"
echo "============================================================"
echo ""

python scripts/evaluate_task.py --task-info hypotension_5min

echo ""
read -p "Press Enter to continue..."
echo ""

# Step 3: Run quick start example
echo "============================================================"
echo "🚀 Step 3: Running Quick Start Example"
echo "============================================================"
echo ""
echo "This will demonstrate:"
echo "  - Task initialization"
echo "  - Synthetic data generation"
echo "  - Evaluation metrics"
echo "  - Benchmark comparison"
echo ""

python examples/quick_start_tasks.py

echo ""
echo "============================================================"
echo "✅ Demo Complete!"
echo "============================================================"
echo ""
echo "📁 Results saved to: results/quick_start/"
echo ""
echo "Next steps:"
echo "  1. Review the HTML report:"
echo "     open results/quick_start/benchmark_report.html"
echo ""
echo "  2. Train your model:"
echo "     python scripts/ttm_vitaldb.py train --fasttrack"
echo ""
echo "  3. Evaluate on real data:"
echo "     python scripts/evaluate_task.py \\"
echo "       --task hypotension_5min \\"
echo "       --checkpoint artifacts/model.pt \\"
echo "       --compare-benchmarks"
echo ""
echo "  4. Evaluate all tasks:"
echo "     bash scripts/evaluate_all_tasks.sh \\"
echo "       artifacts/model.pt \\"
echo "       results/evaluation"
echo ""
echo "📚 Read the full guide: DOWNSTREAM_TASKS.md"
echo "============================================================"
