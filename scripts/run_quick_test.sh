#!/bin/bash
# Quick End-to-End Pipeline Test Runner
# Runs in 5-10 minutes with 2 VitalDB cases on CPU

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=================================="
echo "Quick E2E Pipeline Test"
echo "=================================="
echo "Project: $PROJECT_ROOT"
echo "Python: $(which python3)"
echo ""

# Check Python version
python3 -c "import sys; assert sys.version_info >= (3, 8), 'Python 3.8+ required'" || {
    echo "❌ Error: Python 3.8 or higher required"
    exit 1
}

# Check dependencies
echo "Checking dependencies..."
python3 -c "
import sys
missing = []
try:
    import torch
except ImportError:
    missing.append('torch')
try:
    import numpy
except ImportError:
    missing.append('numpy')
try:
    import scipy
except ImportError:
    missing.append('scipy')
try:
    import vitaldb
except ImportError:
    missing.append('vitaldb')
try:
    import neurokit2
except ImportError:
    missing.append('neurokit2')

if missing:
    print(f'❌ Missing packages: {missing}')
    print('Install with: pip install -r requirements.txt')
    sys.exit(1)
else:
    print('✓ All dependencies installed')
" || exit 1

echo ""
echo "Starting quick E2E test..."
echo "This will:"
echo "  1. Load 2 VitalDB cases"
echo "  2. Build preprocessed windows"
echo "  3. Train for 2 epochs (CPU)"
echo "  4. Evaluate on test case"
echo ""
echo "Expected runtime: 5-10 minutes"
echo ""

# Run the test
cd "$PROJECT_ROOT"
python3 scripts/quick_e2e_test.py

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "✅ TEST PASSED!"
    echo "=================================="
else
    echo ""
    echo "=================================="
    echo "❌ TEST FAILED!"
    echo "=================================="
    exit $exit_code
fi
