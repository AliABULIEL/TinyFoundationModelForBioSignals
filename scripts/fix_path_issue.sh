#!/bin/bash
# Quick fix for the path issue

echo "=========================================="
echo "Fixing Split File Path Issue"
echo "=========================================="

# Remove the incorrectly created directory
if [ -d "configs/splits/splits_full.json" ]; then
    echo "Removing incorrect directory: configs/splits/splits_full.json"
    rm -rf configs/splits/splits_full.json
fi

# Clean up any partial data
echo "Cleaning up partial artifacts..."
rm -rf artifacts/raw_windows/train
rm -rf artifacts/raw_windows/val  
rm -rf artifacts/raw_windows/test

# Create fresh directories
echo "Creating fresh directories..."
mkdir -p configs/splits
mkdir -p artifacts/raw_windows
mkdir -p artifacts/checkpoints

echo ""
echo "✓ Cleanup complete!"
echo ""
echo "Now you can run:"
echo "  bash scripts/run_high_accuracy.sh"
echo ""
echo "The script has been fixed to use:"
echo "  --output configs/splits"
echo "which will create: configs/splits/splits_full.json"
echo ""
