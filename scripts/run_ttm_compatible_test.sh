#!/bin/bash
# Run TTM-Compatible E2E Test
# This version ensures data matches TTM's architecture requirements

set -e

PROJECT_ROOT="/Users/aliab/Desktop/TinyFoundationModelForBioSignals"
cd "$PROJECT_ROOT"

echo "======================================"
echo "TTM-Compatible E2E Pipeline Test"
echo "======================================"
echo "Project: $PROJECT_ROOT"
echo "Python: $(which python3)"
echo ""
echo "This test ensures data matches TTM requirements:"
echo "  - Context length: 512 samples"
echo "  - Input channels: 3 (multi-channel)"
echo "  - Sampling rate adjusted accordingly"
echo ""
echo "Expected runtime: 5-10 minutes"
echo ""

# Check dependencies
echo "Checking dependencies..."
python3 -c "import torch, numpy, tqdm, yaml" 2>/dev/null && echo "✓ Core dependencies installed" || {
    echo "❌ Missing dependencies!"
    exit 1
}

# Check TTM availability
echo "Checking TTM availability..."
python3 -c "from tsfm_public import get_model; print('✓ TTM (tsfm_public) available')" 2>/dev/null || {
    echo "⚠️ TTM not available - will use fallback"
    echo "To install: pip install git+https://github.com/ibm-granite/granite-tsfm.git"
    echo ""
}

echo "Starting TTM-compatible test..."
echo ""

# Run the test
python3 scripts/quick_e2e_test_ttm_compatible.py

echo ""
echo "======================================"
echo "Test completed!"
echo "======================================"
