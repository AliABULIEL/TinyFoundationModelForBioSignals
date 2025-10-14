#!/bin/bash
# ============================================================================
# End-to-End VAE Pipeline for VitalDB Biosignals
# ============================================================================
# This script runs the complete VAE training pipeline:
# 1. Data preparation (windows from VitalDB)
# 2. VAE unsupervised pretraining
# 3. VAE supervised fine-tuning
# 4. Model evaluation
# 5. Results visualization
# ============================================================================

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Paths
PROJECT_DIR="/Users/aliab/Desktop/TinyFoundationModelForBioSignals"
SCRIPTS_DIR="$PROJECT_DIR/scripts"
CONFIG_DIR="$PROJECT_DIR/configs"
DATA_DIR="$PROJECT_DIR/artifacts"
RESULTS_DIR="$PROJECT_DIR/results_vae"

# Python environment
PYTHON="python3"

# Mode selection
MODE="${1:-full}"  # 'fasttrack' or 'full'
echo "Running in $MODE mode"

# Create necessary directories
mkdir -p "$DATA_DIR"
mkdir -p "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR/checkpoints"
mkdir -p "$RESULTS_DIR/logs"
mkdir -p "$RESULTS_DIR/plots"

# Logging
LOG_FILE="$RESULTS_DIR/logs/vae_pipeline_$(date +%Y%m%d_%H%M%S).log"
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1

echo "============================================================================"
echo "VAE Pipeline for Biosignals - $(date)"
echo "============================================================================"
echo "Mode: $MODE"
echo "Project directory: $PROJECT_DIR"
echo "Results directory: $RESULTS_DIR"
echo "Log file: $LOG_FILE"
echo ""

# ============================================================================
# Step 0: Run Tests
# ============================================================================
echo "Step 0: Running VAE tests..."
echo "----------------------------------------"

cd "$PROJECT_DIR"
if $PYTHON tests/test_vae.py; then
    echo "✓ All VAE tests passed"
else
    echo "✗ VAE tests failed. Please fix issues before continuing."
    exit 1
fi
echo ""

# ============================================================================
# Step 1: Prepare Data Splits
# ============================================================================
echo "Step 1: Preparing data splits..."
echo "----------------------------------------"

if [ ! -f "$CONFIG_DIR/splits/splits_${MODE}.json" ]; then
    $PYTHON "$SCRIPTS_DIR/ttm_vitaldb_fixed.py" prepare-splits \
        --mode "$MODE" \
        --output "$CONFIG_DIR/splits" \
        --seed 42
    echo "✓ Splits created: $CONFIG_DIR/splits/splits_${MODE}.json"
else
    echo "✓ Splits already exist: $CONFIG_DIR/splits/splits_${MODE}.json"
fi
echo ""

# ============================================================================
# Step 2: Build Preprocessed Windows
# ============================================================================
echo "Step 2: Building preprocessed windows..."
echo "----------------------------------------"

# Function to build windows for a split
build_windows() {
    local split=$1
    local output_file="$DATA_DIR/raw_windows/$split/${split}_windows.npz"
    
    if [ ! -f "$output_file" ]; then
        echo "Building $split windows..."
        
        # Use multiprocessing version if available
        if [ -f "$SCRIPTS_DIR/ttm_vitaldb_multiprocess.py" ]; then
            echo "Using multiprocessing (8 workers)..."
            $PYTHON "$SCRIPTS_DIR/ttm_vitaldb_multiprocess.py" build-windows \
                --channels-yaml "$CONFIG_DIR/channels.yaml" \
                --windows-yaml "$CONFIG_DIR/windows.yaml" \
                --split-file "$CONFIG_DIR/splits/splits_${MODE}.json" \
                --split "$split" \
                --outdir "$DATA_DIR/raw_windows/$split" \
                --channel PPG \
                --duration-sec 60 \
                --min-sqi 0.5 \
                --num-workers 8
        else
            $PYTHON "$SCRIPTS_DIR/ttm_vitaldb_fixed.py" build-windows \
                --channels-yaml "$CONFIG_DIR/channels.yaml" \
                --windows-yaml "$CONFIG_DIR/windows.yaml" \
                --split-file "$CONFIG_DIR/splits/splits_${MODE}.json" \
                --split "$split" \
                --outdir "$DATA_DIR/raw_windows/$split" \
                --channel PPG \
                --duration-sec 60 \
                --min-sqi 0.5
        fi
        echo "✓ $split windows created: $output_file"
    else
        echo "✓ $split windows already exist: $output_file"
    fi
}

# Build windows for each split
build_windows "train"

if [ "$MODE" == "full" ]; then
    build_windows "val"
fi

build_windows "test"
echo ""

# ============================================================================
# Step 3: VAE Unsupervised Pretraining
# ============================================================================
echo "Step 3: VAE unsupervised pretraining..."
echo "----------------------------------------"

PRETRAINED_MODEL="$RESULTS_DIR/checkpoints/vae_pretrained.pt"

if [ ! -f "$PRETRAINED_MODEL" ]; then
    echo "Starting VAE pretraining (reconstruction only)..."
    
    # Create temporary config for pretraining
    cat > "$CONFIG_DIR/model_vae_pretrain.yaml" <<EOF
model_type: vae
task: reconstruction  # No supervised task for pretraining
latent_dim: 64
beta: 1.0
freeze_encoder: false
head_type: none
kl_annealing: true
EOF
    
    $PYTHON "$SCRIPTS_DIR/train_vae.py" \
        --model-yaml "$CONFIG_DIR/model_vae_pretrain.yaml" \
        --run-yaml "$CONFIG_DIR/run_vae.yaml" \
        --outdir "$DATA_DIR/raw_windows" \
        --out "$RESULTS_DIR/checkpoints"
    
    echo "✓ VAE pretrained model saved: $PRETRAINED_MODEL"
else
    echo "✓ Pretrained VAE already exists: $PRETRAINED_MODEL"
fi
echo ""

# ============================================================================
# Step 4: VAE Supervised Fine-tuning
# ============================================================================
echo "Step 4: VAE supervised fine-tuning..."
echo "----------------------------------------"

FINETUNED_MODEL="$RESULTS_DIR/checkpoints/vae_finetuned.pt"

if [ ! -f "$FINETUNED_MODEL" ]; then
    echo "Starting VAE fine-tuning (with classification head)..."
    
    $PYTHON "$SCRIPTS_DIR/train_vae.py" \
        --model-yaml "$CONFIG_DIR/model_vae.yaml" \
        --run-yaml "$CONFIG_DIR/run_vae.yaml" \
        --outdir "$DATA_DIR/raw_windows" \
        --out "$RESULTS_DIR/checkpoints" \
        --load-pretrained "$PRETRAINED_MODEL"
    
    echo "✓ Fine-tuned VAE saved: $FINETUNED_MODEL"
else
    echo "✓ Fine-tuned VAE already exists: $FINETUNED_MODEL"
fi
echo ""

# ============================================================================
# Step 5: Evaluate VAE Model
# ============================================================================
echo "Step 5: Evaluating VAE model..."
echo "----------------------------------------"

# Test with fixed script (using VAE model)
$PYTHON "$SCRIPTS_DIR/ttm_vitaldb_fixed.py" test \
    --ckpt "$FINETUNED_MODEL" \
    --model-yaml "$CONFIG_DIR/model_vae.yaml" \
    --run-yaml "$CONFIG_DIR/run_vae.yaml" \
    --split-file "$CONFIG_DIR/splits/splits_${MODE}.json" \
    --outdir "$DATA_DIR/raw_windows" \
    --out "$RESULTS_DIR/evaluation"

echo "✓ Evaluation results saved to: $RESULTS_DIR/evaluation"
echo ""

# ============================================================================
# Step 6: Generate VAE Analysis and Visualizations
# ============================================================================
echo "Step 6: Generating VAE analysis..."
echo "----------------------------------------"

# Create analysis script
cat > "$RESULTS_DIR/analyze_vae.py" <<'ANALYSIS_SCRIPT'
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.vae_adapter import VAEAdapter
from src.models.ttm_adapter import create_ttm_model

def analyze_vae_results(results_dir):
    """Analyze VAE training and evaluation results."""
    
    # Load test results
    test_results_file = Path(results_dir) / "evaluation" / "test_results.json"
    if test_results_file.exists():
        with open(test_results_file, 'r') as f:
            test_results = json.load(f)
        
        print("Test Results:")
        print("-" * 40)
        for metric, value in test_results['metrics'].items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
    
    # Load model for latent space analysis
    model_config = {
        'model_type': 'vae',
        'task': 'classification',
        'num_classes': 2,
        'input_channels': 1,
        'context_length': 1250,
        'latent_dim': 64
    }
    
    model = create_ttm_model(model_config)
    checkpoint_path = Path(results_dir) / "checkpoints" / "vae_finetuned.pt"
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Generate samples from prior
        print("\nGenerating samples from VAE...")
        device = torch.device('cpu')
        samples = model.generate_samples(num_samples=10, device=device)
        
        # Plot samples
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for i, ax in enumerate(axes.flat):
            ax.plot(samples[i, 0, :].detach().numpy())
            ax.set_title(f'Sample {i+1}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Amplitude')
        plt.tight_layout()
        plt.savefig(Path(results_dir) / "plots" / "vae_generated_samples.png")
        print(f"✓ Generated samples plot saved")
        
        # Load training history
        if 'train_losses' in checkpoint and 'val_losses' in checkpoint:
            # Plot loss curves
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Placeholder for actual epoch tracking
            ax1.set_title('Reconstruction Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            
            ax2.set_title('KL Divergence')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('KL Loss')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(Path(results_dir) / "plots" / "vae_training_curves.png")
            print(f"✓ Training curves saved")
    
    print("\n✓ VAE analysis complete!")

if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results_vae"
    analyze_vae_results(results_dir)
ANALYSIS_SCRIPT

$PYTHON "$RESULTS_DIR/analyze_vae.py" "$RESULTS_DIR"
echo ""

# ============================================================================
# Step 7: Compare VAE with TTM (Optional)
# ============================================================================
echo "Step 7: Comparing VAE with TTM (optional)..."
echo "----------------------------------------"

if [ "$2" == "--compare" ]; then
    echo "Training TTM for comparison..."
    
    # Train TTM model
    $PYTHON "$SCRIPTS_DIR/ttm_vitaldb_fixed.py" train \
        --model-yaml "$CONFIG_DIR/model.yaml" \
        --run-yaml "$CONFIG_DIR/run.yaml" \
        --split-file "$CONFIG_DIR/splits/splits_${MODE}.json" \
        --outdir "$DATA_DIR/raw_windows" \
        --out "$RESULTS_DIR/ttm_comparison" \
        --fasttrack
    
    # Test TTM model
    $PYTHON "$SCRIPTS_DIR/ttm_vitaldb_fixed.py" test \
        --ckpt "$RESULTS_DIR/ttm_comparison/best_model.pt" \
        --model-yaml "$CONFIG_DIR/model.yaml" \
        --run-yaml "$CONFIG_DIR/run.yaml" \
        --split-file "$CONFIG_DIR/splits/splits_${MODE}.json" \
        --outdir "$DATA_DIR/raw_windows" \
        --out "$RESULTS_DIR/ttm_evaluation"
    
    echo "✓ TTM comparison results saved"
else
    echo "Skipping TTM comparison (add --compare flag to enable)"
fi
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "============================================================================"
echo "VAE Pipeline Complete!"
echo "============================================================================"
echo ""
echo "Results summary:"
echo "  - Pretrained VAE: $PRETRAINED_MODEL"
echo "  - Fine-tuned VAE: $FINETUNED_MODEL"
echo "  - Test results: $RESULTS_DIR/evaluation/test_results.json"
echo "  - Plots: $RESULTS_DIR/plots/"
echo "  - Log file: $LOG_FILE"
echo ""

# Display key metrics
if [ -f "$RESULTS_DIR/evaluation/test_results.json" ]; then
    echo "Key metrics:"
    $PYTHON -c "
import json
with open('$RESULTS_DIR/evaluation/test_results.json', 'r') as f:
    results = json.load(f)
    metrics = results.get('metrics', {})
    for key in ['accuracy', 'f1', 'auc', 'loss']:
        if key in metrics:
            print(f'  {key}: {metrics[key]:.4f}')
"
fi

echo ""
echo "✓ All steps completed successfully!"
echo ""
echo "Next steps:"
echo "  1. Review results in: $RESULTS_DIR"
echo "  2. Analyze generated samples and latent space"
echo "  3. Compare with TTM baseline: ./run_vae_pipeline.sh $MODE --compare"
echo "  4. Fine-tune hyperparameters in configs/model_vae.yaml"
echo ""
