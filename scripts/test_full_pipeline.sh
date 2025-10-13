#!/bin/bash
# Complete Pipeline Test: SSL Pretraining → Fine-tuning
# This script tests the entire pipeline from SSL pretraining to BUT-PPG fine-tuning

set -e  # Exit on error

echo "=============================================================================="
echo "COMPLETE PIPELINE TEST: SSL PRETRAINING → BUT-PPG FINE-TUNING"
echo "=============================================================================="
echo ""

# Configuration
EPOCHS_SSL=1
EPOCHS_FINETUNE=1
BATCH_SIZE=8
DATA_DIR_VITALDB="data/vitaldb_windows"
DATA_DIR_BUTPPG="data/but_ppg"
OUTPUT_SSL="artifacts/foundation_model"
OUTPUT_FINETUNE="artifacts/but_ppg_finetuned"

echo "Configuration:"
echo "  SSL epochs: $EPOCHS_SSL"
echo "  Fine-tune epochs: $EPOCHS_FINETUNE"
echo "  Batch size: $BATCH_SIZE"
echo ""

# Step 0: Generate synthetic BUT-PPG data if not exists
echo "=============================================================================="
echo "STEP 0: PREPARE BUT-PPG TEST DATA"
echo "=============================================================================="
if [ ! -d "$DATA_DIR_BUTPPG" ] || [ ! -f "$DATA_DIR_BUTPPG/train.npz" ]; then
    echo "Generating synthetic BUT-PPG test data..."
    python scripts/generate_butppg_test_data.py \
        --output-dir "$DATA_DIR_BUTPPG" \
        --train-samples 50 \
        --val-samples 20 \
        --test-samples 30 \
        --quality-ratio 0.6 \
        --seed 42
    echo "✓ BUT-PPG data generated"
else
    echo "✓ BUT-PPG data already exists"
fi
echo ""

# Step 1: SSL Pretraining (if checkpoint doesn't exist)
echo "=============================================================================="
echo "STEP 1: SSL PRETRAINING ON VITALDB (2 channels: PPG + ECG)"
echo "=============================================================================="
if [ ! -f "$OUTPUT_SSL/best_model.pt" ]; then
    echo "Running SSL pretraining..."
    python scripts/pretrain_vitaldb_ssl.py \
        --config configs/ssl_pretrain.yaml \
        --data-dir "$DATA_DIR_VITALDB" \
        --channels PPG ECG \
        --output-dir "$OUTPUT_SSL" \
        --mask-ratio 0.4 \
        --epochs $EPOCHS_SSL \
        --batch-size $BATCH_SIZE \
        --device cpu \
        --fast
    
    echo ""
    echo "✓ SSL pretraining complete"
else
    echo "✓ SSL checkpoint already exists: $OUTPUT_SSL/best_model.pt"
    echo "  Skipping pretraining (use --force to retrain)"
fi
echo ""

# Step 2: Fine-tuning on BUT-PPG
echo "=============================================================================="
echo "STEP 2: FINE-TUNING ON BUT-PPG (5 channels: ACC_X,Y,Z + PPG + ECG)"
echo "=============================================================================="
echo "Inflating channels: 2 → 5"
echo "Strategy: Staged unfreezing"
echo "  - Stage 1: Head-only training"
echo "  - Stage 2: Unfreeze last 2 blocks"
echo ""

python scripts/finetune_butppg.py \
    --pretrained "$OUTPUT_SSL/best_model.pt" \
    --data-dir "$DATA_DIR_BUTPPG" \
    --pretrain-channels 2 \
    --finetune-channels 5 \
    --unfreeze-last-n 2 \
    --epochs $EPOCHS_FINETUNE \
    --head-only-epochs 1 \
    --lr 2e-5 \
    --batch-size $BATCH_SIZE \
    --device cpu \
    --output-dir "$OUTPUT_FINETUNE"

echo ""
echo "✓ Fine-tuning complete"
echo ""

# Step 3: Display Results
echo "=============================================================================="
echo "RESULTS SUMMARY"
echo "=============================================================================="
echo ""

echo "SSL Pretraining:"
if [ -f "$OUTPUT_SSL/training_history.json" ]; then
    echo "  History: $OUTPUT_SSL/training_history.json"
    echo "  Best model: $OUTPUT_SSL/best_model.pt"
else
    echo "  ⚠ Training history not found"
fi
echo ""

echo "BUT-PPG Fine-tuning:"
if [ -f "$OUTPUT_FINETUNE/training_history.json" ]; then
    echo "  History: $OUTPUT_FINETUNE/training_history.json"
    echo "  Best model: $OUTPUT_FINETUNE/best_model.pt"
    
    if [ -f "$OUTPUT_FINETUNE/test_metrics.json" ]; then
        echo ""
        echo "  Test Metrics:"
        cat "$OUTPUT_FINETUNE/test_metrics.json"
    fi
else
    echo "  ⚠ Training history not found"
fi
echo ""

echo "=============================================================================="
echo "PIPELINE TEST COMPLETE ✓"
echo "=============================================================================="
echo ""
echo "Next steps:"
echo "  1. Inspect training curves: $OUTPUT_FINETUNE/training_history.json"
echo "  2. Load best model: $OUTPUT_FINETUNE/best_model.pt"
echo "  3. Run full training with more epochs for better performance"
echo ""
echo "For full training:"
echo "  ./scripts/run_full_pipeline.sh"
echo ""
