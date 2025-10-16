#!/bin/bash

# Complete BUT-PPG setup pipeline
# This script downloads, processes, and prepares BUT-PPG data for evaluation

set -e  # Exit on error

echo "========================================================================"
echo "BUT-PPG COMPLETE SETUP PIPELINE"
echo "========================================================================"

# Step 1: Install dependencies
echo ""
echo "Step 1: Installing dependencies..."
pip install wfdb pandas scipy numpy tqdm

# Step 2: Download raw data
echo ""
echo "Step 2: Downloading BUT-PPG dataset from PhysioNet..."
python scripts/download_butppg_dataset.py \
    --output-dir data/but_ppg/raw \
    --method zip \
    --skip-if-exists

# Step 3: Process clinical data
echo ""
echo "Step 3: Processing clinical data for all tasks..."
# After download, dataset is extracted to:
# data/but_ppg/raw/but-ppg-an-annotated-photoplethysmography-dataset-2.0.0/
DATASET_DIR="data/but_ppg/raw/but-ppg-an-annotated-photoplethysmography-dataset-2.0.0"
ANNOTATIONS_DIR="data/but_ppg/raw/annotations"

python scripts/process_butppg_clinical.py \
    --raw-dir "$DATASET_DIR" \
    --annotations-dir "$ANNOTATIONS_DIR" \
    --output-dir data/processed/butppg \
    --target-fs 125 \
    --window-size 1024 \
    --tasks quality,heart_rate,motion

# Step 4: Verify data structure
echo ""
echo "Step 4: Verifying processed data..."
if [ -f "scripts/check_butppg_data.py" ]; then
    python scripts/check_butppg_data.py \
        --data-dir data/processed/butppg
fi

echo ""
echo "========================================================================"
echo "✓ BUT-PPG SETUP COMPLETE!"
echo "========================================================================"
echo ""
echo "Data structure:"
echo "  data/processed/butppg/"
echo "  ├── quality/"
echo "  │   ├── train.npz"
echo "  │   ├── val.npz"
echo "  │   └── test.npz"
echo "  ├── heart_rate/"
echo "  │   ├── train.npz"
echo "  │   ├── val.npz"
echo "  │   └── test.npz"
echo "  └── motion/"
echo "      ├── train.npz"
echo "      ├── val.npz"
echo "      └── test.npz"
echo ""
echo "Next steps:"
echo "  1. Run downstream evaluation:"
echo "     python scripts/run_downstream_evaluation.py \\"
echo "       --butppg-checkpoint artifacts/but_ppg_finetuned/best_model.pt \\"
echo "       --butppg-data data/processed/butppg \\"
echo "       --output-dir artifacts/butppg_evaluation"
echo ""
