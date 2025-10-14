#!/bin/bash
# Build VitalDB windows with all warnings suppressed
# Usage: bash scripts/run_build_windows.sh [train|val|test]

# Get split from argument, default to 'train'
SPLIT=${1:-train}

# Disable Python warnings
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL=3

echo "========================================================================"
echo "Building VitalDB Windows - Split: $SPLIT"
echo "========================================================================"
echo ""

# Run the build-windows command with all required arguments
python3 scripts/ttm_vitaldb.py build-windows \
    --channels-yaml configs/channels.yaml \
    --windows-yaml configs/windows.yaml \
    --split-file configs/splits/splits_fallback.json \
    --split "$SPLIT" \
    --outdir data/vitaldb_windows \
    --multiprocess \
    --num-workers 4

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✓ $SPLIT windows built successfully!"
    echo "Output: data/vitaldb_windows/${SPLIT}_windows.npz"
    echo "========================================================================"
    echo ""
    echo "Next steps:"
    if [ "$SPLIT" = "train" ]; then
        echo "  1. Build validation windows:"
        echo "     bash scripts/run_build_windows.sh val"
    else
        echo "  1. Run smoke test:"
        echo "     python3 scripts/smoke_realdata_5min.py --data-dir data/vitaldb_windows"
    fi
    echo "  2. Start SSL pretraining:"
    echo "     python3 scripts/pretrain_vitaldb_ssl.py --data-dir data/vitaldb_windows"
else
    echo ""
    echo "✗ Failed to build $SPLIT windows (exit code: $EXIT_CODE)"
    exit $EXIT_CODE
fi
