#!/bin/bash
# High-Accuracy Pipeline - Full fine-tuning with LoRA

# Exit on error
set -e

echo "=========================================="
echo "TTM × VitalDB High-Accuracy Pipeline"
echo "Fine-tuning with partial unfreezing + LoRA"
echo "Expected runtime: 12-24 hours on GPU"
echo "=========================================="

# Set paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="${PYTHON:-python3}"

# Create output directories
mkdir -p artifacts/raw_windows
mkdir -p artifacts/checkpoints
mkdir -p configs/splits

echo ""
echo "Step 1/5: Preparing train/val/test splits..."
echo "----------------------------------------------"
$PYTHON scripts/ttm_vitaldb.py prepare-splits \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --seed 42 \
    --out configs/splits/train_val_test.json

echo ""
echo "Step 2/5: Building preprocessed windows..."
echo "--------------------------------------------"
# Build training windows
echo "Processing training set..."
$PYTHON scripts/ttm_vitaldb.py build-windows \
    --channels-yaml configs/channels.yaml \
    --windows-yaml configs/windows.yaml \
    --split-file configs/splits/train_val_test.json \
    --split train \
    --outdir artifacts/raw_windows/train \
    --ecg-mode analysis

# Build validation windows
echo "Processing validation set..."
$PYTHON scripts/ttm_vitaldb.py build-windows \
    --channels-yaml configs/channels.yaml \
    --windows-yaml configs/windows.yaml \
    --split-file configs/splits/train_val_test.json \
    --split val \
    --outdir artifacts/raw_windows/val \
    --ecg-mode analysis

# Build test windows
echo "Processing test set..."
$PYTHON scripts/ttm_vitaldb.py build-windows \
    --channels-yaml configs/channels.yaml \
    --windows-yaml configs/windows.yaml \
    --split-file configs/splits/train_val_test.json \
    --split test \
    --outdir artifacts/raw_windows/test \
    --ecg-mode analysis

echo ""
echo "Step 3/5: Creating high-accuracy model config..."
echo "-------------------------------------------------"
cat > configs/model_high_accuracy.yaml << 'EOF'
# High-Accuracy Model Configuration

model:
  variant: "ibm/TTM"
  input_channels: 3
  context_length: 1250
  d_model: 512

task:
  type: "classification"
  num_classes: 2

# Fine-tuning with partial unfreezing
training_mode:
  freeze_encoder: false
  unfreeze_last_n_blocks: 2  # Unfreeze last 2 transformer blocks
  
  # Gradual unfreezing
  gradual_unfreeze:
    enabled: true
    schedule: [0, 0, 2, 4, 6]  # Progressive unfreezing

# LoRA for efficient adaptation
lora:
  enabled: true
  r: 16  # Higher rank for more capacity
  alpha: 32
  dropout: 0.1
  target_modules:
    - "q_proj"
    - "v_proj"
    - "k_proj"

# Enhanced MLP head
head:
  type: "mlp"
  mlp:
    hidden_dims: [512, 256, 128]
    activation: "gelu"
    dropout: 0.2
    batch_norm: true

# Advanced loss function
loss:
  classification:
    type: "focal"  # Better for imbalanced data
    focal:
      alpha: 0.25
      gamma: 2.0
    label_smoothing: 0.1

regularization:
  weight_decay: 0.01
  gradient_clip: 1.0
  early_stopping:
    patience: 10
    monitor: "val_f1"
    mode: "max"
EOF

echo ""
echo "Step 4/5: Training with fine-tuning..."
echo "----------------------------------------"
$PYTHON scripts/ttm_vitaldb.py train \
    --model-yaml configs/model_high_accuracy.yaml \
    --run-yaml configs/run.yaml \
    --split-file configs/splits/train_val_test.json \
    --split train \
    --task clf \
    --out artifacts/run_ft_full \
    --epochs 50 \
    --early-stopping-patience 10

echo ""
echo "Step 5/5: Advanced testing with calibration..."
echo "-----------------------------------------------"
# Test with overlapping windows and calibration
$PYTHON scripts/ttm_vitaldb.py test \
    --model-yaml configs/model_high_accuracy.yaml \
    --run-yaml configs/run.yaml \
    --split-file configs/splits/train_val_test.json \
    --split test \
    --task clf \
    --ckpt artifacts/run_ft_full/model.pt \
    --out artifacts/run_ft_full \
    --calibration isotonic \
    --overlap 0.5 \
    --context-length 1500

echo ""
echo "=========================================="
echo "High-Accuracy Pipeline Complete!"
echo "=========================================="
echo ""
echo "Results saved to: artifacts/run_ft_full/"
echo "  - model.pt: Fine-tuned model with LoRA"
echo "  - metrics.json: Full performance metrics"
echo "  - test_results.json: Detailed test evaluation"
echo ""
echo "Performance improvements over FastTrack:"
echo "  - Higher accuracy from fine-tuning"
echo "  - Better calibration with isotonic regression"
echo "  - Improved temporal resolution with overlapping windows"
echo "  - LoRA adapters for efficient parameter updates"
echo ""
echo "Next steps:"
echo "  1. Compare results: diff artifacts/run_ft_fast/metrics.json artifacts/run_ft_full/metrics.json"
echo "  2. Deploy model: python scripts/export_model.py --ckpt artifacts/run_ft_full/model.pt"
echo "  3. Run inference: python scripts/inference.py --model artifacts/run_ft_full/model.pt --data new_data.npz"
