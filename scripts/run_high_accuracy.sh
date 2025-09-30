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
mkdir -p data

echo ""
echo "Step 1/5: Preparing train/val/test splits..."
echo "----------------------------------------------"
$PYTHON scripts/ttm_vitaldb.py prepare-splits \
    --mode full \
    --case-set bis \
    --output configs/splits \
    --seed 42

echo ""
echo "Step 2/5: Building preprocessed windows..."
echo "--------------------------------------------"
# Build training windows
echo "Processing training set..."
$PYTHON scripts/ttm_vitaldb_multiprocess.py build-windows \
    --channels-yaml configs/channels.yaml \
    --windows-yaml configs/windows.yaml \
    --split-file configs/splits/splits_full.json \
    --split train \
    --outdir artifacts/raw_windows/train \
    --duration-sec 60 \
    --min-sqi 0.8

# Build validation windows
echo "Processing validation set..."
$PYTHON scripts/ttm_vitaldb_multiprocess.py build-windows \
    --channels-yaml configs/channels.yaml \
    --windows-yaml configs/windows.yaml \
    --split-file configs/splits/splits_full.json \
    --split val \
    --outdir artifacts/raw_windows/val \
    --duration-sec 60 \
    --min-sqi 0.8

# Build test windows
echo "Processing test set..."
$PYTHON scripts/ttm_vitaldb_multiprocess.py build-windows \
    --channels-yaml configs/channels.yaml \
    --windows-yaml configs/windows.yaml \
    --split-file configs/splits/splits_full.json \
    --split test \
    --outdir artifacts/raw_windows/test \
    --duration-sec 60 \
    --min-sqi 0.8

echo ""
echo "Step 3/5: Creating high-accuracy model config..."
echo "-------------------------------------------------"
cat > configs/model_high_accuracy.yaml << 'EOF'
# High-Accuracy Model Configuration

model:
  variant: "ibm-granite/granite-timeseries-ttm-r1"
  input_channels: 3
  context_length: 1250
  d_model: 512
  patch_size: 16
  stride: 8

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
    - "o_proj"

# Enhanced MLP head
head:
  type: "mlp"
  mlp:
    hidden_dims: [512, 256, 128]
    activation: "gelu"
    dropout: 0.2
    use_batch_norm: false  # Avoid dimension issues

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

# Mixed precision
mixed_precision:
  enabled: true
  opt_level: "O1"
EOF

echo ""
echo "Step 4/5: Training with fine-tuning..."
echo "----------------------------------------"
$PYTHON scripts/ttm_vitaldb.py train \
    --model-yaml configs/model_high_accuracy.yaml \
    --run-yaml configs/run.yaml \
    --split-file configs/splits/splits_full.json \
    --outdir artifacts/raw_windows \
    --out artifacts/run_ft_full

echo ""
echo "Step 5/5: Testing..."
echo "-----------------------------------------------"
$PYTHON scripts/ttm_vitaldb.py test \
    --ckpt artifacts/run_ft_full/best_model.pt \
    --model-yaml configs/model_high_accuracy.yaml \
    --run-yaml configs/run.yaml \
    --split-file configs/splits/splits_full.json \
    --outdir artifacts/raw_windows \
    --out artifacts/run_ft_full

echo ""
echo "=========================================="
echo "High-Accuracy Pipeline Complete!"
echo "=========================================="
echo ""
echo "Results saved to: artifacts/run_ft_full/"
echo "  - best_model.pt: Fine-tuned model with LoRA"
echo "  - test_results.json: Detailed test evaluation"
echo ""
echo "Performance improvements over FastTrack:"
echo "  - Higher accuracy from fine-tuning"
echo "  - LoRA adapters for efficient parameter updates"
echo "  - Full dataset (70% train, 15% val, 15% test)"
echo ""
echo "Next steps:"
echo "  1. Evaluate on tasks: python scripts/evaluate_task.py --task hypotension_5min --checkpoint artifacts/run_ft_full/best_model.pt"
echo "  2. Compare to benchmarks: python scripts/benchmark_comparison.py --results-dir artifacts/"
echo "  3. Generate report: Look at test_results.json for detailed metrics"
