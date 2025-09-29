# TTM × VitalDB: Foundation Model for Biosignals

A production-ready implementation of Tiny Time Mixers (TTM) as a foundation model for VitalDB biosignals, with evidence-aligned preprocessing and high-accuracy fine-tuning options.

## 🚀 Quick Start

### Installation
```bash
pip install -e .
# For development tools:
pip install -e .[dev]
```

### Modes

#### FastTrack Mode (3 hours)
- **Purpose**: Quick validation and prototyping
- **Data**: 50 training / 20 test cases
- **Model**: Frozen TTM encoder + linear head
- **Time**: ~3 hours on single GPU

#### Full Mode
- **Purpose**: Production-grade training
- **Data**: Full VitalDB dataset
- **Model**: TTM with optional unfreezing/LoRA
- **Time**: 12-24 hours on single GPU

### Core CLI Commands

```bash
# 1. Download and preprocess VitalDB data
python scripts/ttm_vitaldb.py preprocess \
    --config configs/run.yaml \
    --mode fasttrack  # or 'full'

# 2. Pre-train TTM on VitalDB
python scripts/ttm_vitaldb.py pretrain \
    --config configs/run.yaml \
    --mode fasttrack

# 3. Fine-tune for specific task (e.g., PPG quality)
python scripts/ttm_vitaldb.py finetune \
    --config configs/run.yaml \
    --task ppg_quality \
    --unfreeze-last-n 2 \
    --lora-rank 8

# 4. Evaluate on test set
python scripts/ttm_vitaldb.py evaluate \
    --checkpoint artifacts/best_model.pt \
    --config configs/run.yaml
```

## 📊 Evidence-Aligned Processing

This implementation strictly follows evidence-based preprocessing:

- **Sampling**: 125 Hz (ECG/PPG/ABP)
- **Windows**: 10-second non-overlapping segments
- **Filters**: 
  - PPG: Chebyshev-II(4) 0.4–7 Hz
  - ECG: Butterworth(4) 0.5–40 Hz
  - ABP: Butterworth(4) 0.5–20 Hz
- **Quality Gates**: SQI≥0.9 (ECG), sSQI≥0.8 (PPG), ≥3 cardiac cycles
- **Normalization**: Population z-score using train stats only
- **Splits**: Patient-level (no subject leakage)

## 🏗️ Architecture

```
TTM Foundation Model (Frozen/Trainable)
    ↓
Optional LoRA Adapters
    ↓
Task-Specific Head (MLP/Linear)
    ↓
Output (Classification/Regression)
```

## 📁 Project Structure

```
├── configs/           # YAML configuration files
├── scripts/          # CLI entry points
├── src/
│   ├── data/        # VitalDB loading and preprocessing
│   ├── models/      # TTM adapters, heads, LoRA
│   ├── eval/        # Metrics and calibration
│   └── utils/       # Common utilities
└── tests/           # Unit tests
```

## 🔧 Configuration

All hyperparameters are controlled via YAML configs:
- `channels.yaml`: Signal selection and sampling rates
- `windows.yaml`: Segmentation and quality thresholds
- `model.yaml`: TTM architecture and fine-tuning
- `run.yaml`: Training hyperparameters

## 📝 Notes

- TTM package import: Uses official `tinytimemixers` package
- GPU recommended for full training
- FastTrack mode ideal for experimentation
- Patient-level splits ensure no data leakage

## 📄 License

MIT
