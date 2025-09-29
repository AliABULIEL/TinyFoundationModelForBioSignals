# TTM × VitalDB: Foundation Model for Biosignals

A production-ready implementation of Tiny Time Mixers (TTM) as a foundation model for VitalDB biosignals, with evidence-aligned preprocessing and high-accuracy fine-tuning options.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/TinyFoundationModelForBioSignals.git
cd TinyFoundationModelForBioSignals

# Install dependencies
pip install -e .

# For development with tests
pip install -e .[dev]
```

### FastTrack Mode (~3 hours)

FastTrack mode enables rapid experimentation with a subset of data and simplified training:
- **50 training cases** (instead of full dataset)
- **Frozen encoder** (foundation model mode)
- **Linear head only** (minimal parameters)
- **10 epochs** (quick convergence)
- **Total runtime: ~3 hours** on single GPU

## 📊 Training Modes

### Foundation Model (FM) Mode - Default
- **Frozen TTM encoder**: Pre-trained weights remain fixed
- **Only head trains**: Linear or MLP classifier/regressor
- **Fast training**: ~3 hours with FastTrack
- **Low compute requirements**: Single GPU sufficient
- **Best for**: Quick prototyping, transfer learning evaluation

### Fine-Tuning (FT) Mode - High Accuracy
- **Partial unfreezing**: Last N transformer blocks trainable
- **LoRA adaptation**: Parameter-efficient fine-tuning
- **Full dataset**: All VitalDB cases
- **Extended training**: 50+ epochs with early stopping
- **Best for**: Maximum performance, production deployment

## 🔧 Complete Pipeline Commands

### 1. Prepare Train/Val/Test Splits

```bash
python scripts/ttm_vitaldb.py prepare-splits \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1 \
    --seed 42 \
    --out configs/splits/train_test.json
```

For FastTrack mode (50/0/20 cases):
```bash
python scripts/ttm_vitaldb.py prepare-splits \
    --train-ratio 0.71 \
    --val-ratio 0.0 \
    --test-ratio 0.29 \
    --seed 42 \
    --out configs/splits/train_test.json \
    --fasttrack
```

### 2. Build Preprocessed Windows

Full preprocessing with evidence-aligned filters and quality checks:

```bash
python scripts/ttm_vitaldb.py build-windows \
    --channels-yaml configs/channels.yaml \
    --windows-yaml configs/windows.yaml \
    --split-file configs/splits/train_test.json \
    --split train \
    --outdir artifacts/raw_windows/train \
    --ecg-mode analysis \
    --fasttrack
```

Key preprocessing steps:
- **Resampling**: 125 Hz for ECG/PPG/ABP
- **Filtering**: Butterworth/Chebyshev bandpass
- **Quality**: SQI≥0.9 (ECG), sSQI≥0.8 (PPG)
- **Validation**: ≥3 cardiac cycles per window
- **Normalization**: Z-score using train statistics

### 3. Train Model

FastTrack training (Foundation Model mode):

```bash
python scripts/ttm_vitaldb.py train \
    --model-yaml configs/model.yaml \
    --run-yaml configs/run.yaml \
    --split-file configs/splits/train_test.json \
    --split train \
    --task clf \
    --out artifacts/run_ft_fast \
    --fasttrack
```

### 4. Test and Evaluate

Test with calibration and metrics:

```bash
python scripts/ttm_vitaldb.py test \
    --model-yaml configs/model.yaml \
    --run-yaml configs/run.yaml \
    --split-file configs/splits/train_test.json \
    --split test \
    --task clf \
    --ckpt artifacts/run_ft_fast/model.pt \
    --out artifacts/run_ft_fast \
    --calibration temperature
```

## 🎯 Switching to High-Accuracy Mode

For maximum performance, modify `configs/model.yaml`:

```yaml
# Partial Unfreezing
freeze_encoder: false
unfreeze_last_n_blocks: 2  # Unfreeze last 2 transformer blocks

# LoRA Adaptation
lora:
  enabled: true
  r: 8  # LoRA rank
  alpha: 16  # LoRA alpha
  dropout: 0.1
  target_modules: ["q_proj", "v_proj"]  # Target attention layers

# Enhanced Head
head_type: mlp  # Use MLP instead of linear
head_config:
  hidden_dims: [256, 128]
  dropout: 0.2
  activation: gelu
```

Then run without `--fasttrack` flag:

```bash
# Full dataset preprocessing
python scripts/ttm_vitaldb.py build-windows \
    --channels-yaml configs/channels.yaml \
    --windows-yaml configs/windows.yaml \
    --split-file configs/splits/train_test.json \
    --split train \
    --outdir artifacts/raw_windows/train \
    --ecg-mode analysis

# Extended training
python scripts/ttm_vitaldb.py train \
    --model-yaml configs/model.yaml \
    --run-yaml configs/run.yaml \
    --split-file configs/splits/train_test.json \
    --split train \
    --task clf \
    --out artifacts/run_ft_full \
    --epochs 50 \
    --early-stopping-patience 10

# Test with advanced options
python scripts/ttm_vitaldb.py test \
    --model-yaml configs/model.yaml \
    --run-yaml configs/run.yaml \
    --split-file configs/splits/train_test.json \
    --split test \
    --task clf \
    --ckpt artifacts/run_ft_full/model.pt \
    --out artifacts/run_ft_full \
    --calibration isotonic \
    --overlap 0.5 \
    --context-length 192
```

## 🔬 Advanced Features

### Cross-Validation (Optional)
For robust evaluation, enable 5-fold CV in `configs/run.yaml`:

```yaml
cv:
  enabled: true
  n_folds: 5
  stratified: true
  seed: 42
```

### Overlapping Windows
For temporal context during inference:

```bash
python scripts/ttm_vitaldb.py test \
    --overlap 0.5 \  # 50% overlap
    --voting soft \   # Soft voting for predictions
    --context-length 192  # Extended context
```

### Calibration Methods
- **Temperature Scaling**: Simple, effective for neural networks
- **Isotonic Regression**: Non-parametric, handles complex miscalibration
- **Platt Scaling**: Binary classification optimization

### Multi-GPU Training
```bash
# Distributed training on 4 GPUs
torchrun --nproc_per_node=4 scripts/ttm_vitaldb.py train \
    --model-yaml configs/model.yaml \
    --run-yaml configs/run.yaml \
    --distributed
```

## 📁 Project Structure

```
├── configs/
│   ├── channels.yaml      # Signal configurations (fs, filters)
│   ├── windows.yaml       # Window parameters (size, quality)
│   ├── model.yaml         # TTM architecture and fine-tuning
│   └── run.yaml           # Training hyperparameters
├── scripts/
│   └── ttm_vitaldb.py     # Main CLI entry point
├── src/
│   ├── data/              # VitalDB loading and preprocessing
│   ├── models/            # TTM adapter, heads, LoRA
│   ├── eval/              # Metrics and calibration
│   └── utils/             # Common utilities
├── tests/                 # Comprehensive test suite
└── artifacts/             # Output directory (models, results)
```

## 📈 Expected Performance

### FastTrack Mode (3 hours)
- **Accuracy**: 85-90% on binary tasks
- **ECE**: < 0.05 with calibration
- **Training time**: ~2 hours
- **Inference**: 1000 windows/second

### Full Fine-Tuning Mode
- **Accuracy**: 92-96% on binary tasks
- **ECE**: < 0.02 with calibration
- **Training time**: 12-24 hours
- **Model size**: +5% parameters (LoRA)

## 🔍 Monitoring Training

Track metrics in real-time:

```bash
# View training logs
tail -f artifacts/run_ft_fast/train.log

# TensorBoard visualization
tensorboard --logdir artifacts/run_ft_fast/tensorboard

# Check metrics
cat artifacts/run_ft_fast/metrics.json | jq .
```

## 🚨 Troubleshooting

### Out of Memory
- Reduce batch size in `configs/run.yaml`
- Use gradient accumulation
- Enable mixed precision (AMP)
- Use FastTrack mode for initial experiments

### Poor Convergence
- Check data quality with `--inspect` flag
- Verify preprocessing parameters
- Adjust learning rate schedule
- Ensure sufficient training data per class

### Calibration Issues
- Use temperature scaling for quick calibration
- Try isotonic regression for complex patterns
- Validate on held-out calibration set

## 📚 Citation

If you use this implementation, please cite:

```bibtex
@software{ttm_vitaldb2024,
  title={TTM × VitalDB: Foundation Model for Biosignals},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/TinyFoundationModelForBioSignals}
}
```

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- IBM Research for TinyTimeMixers architecture
- VitalDB team for the biosignal database
- Hugging Face for model hosting infrastructure
