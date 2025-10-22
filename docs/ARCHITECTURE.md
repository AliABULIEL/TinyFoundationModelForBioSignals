# Foundation Model Architecture

**Tiny Foundation Model for Biosignals** - Complete architecture documentation for the 3-stage hybrid SSL pipeline.

---

## Table of Contents

1. [Overview](#overview)
2. [3-Stage Pipeline](#3-stage-pipeline)
3. [Detailed Component Architecture](#detailed-component-architecture)
4. [Data Flow & Transformations](#data-flow--transformations)
5. [Key Design Principles](#key-design-principles)
6. [Architecture Parameters](#architecture-parameters)

---

## Overview

This project implements a **foundation model** for biosignal analysis using a 3-stage training pipeline:

1. **Stage 1**: Self-supervised pre-training on VitalDB (general biosignal patterns)
2. **Stage 2**: Quality-aware SSL on BUT-PPG (domain adaptation with contrastive learning)
3. **Stage 3**: Supervised fine-tuning on BUT-PPG (task-specific classification)

**Core Innovation**: Combining IBM's Tiny Time Mixers (TTM) transformer architecture with quality-aware contrastive learning to bridge the domain gap between datasets.

---

## 3-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 1: FOUNDATION MODEL                     │
│                    (VitalDB SSL Pre-training)                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
        artifacts/foundation_model/best_model.pt
        - Encoder: IBM TTM (192-dim, 8 patches)
        - Trained on: VitalDB (PPG + ECG, 2 channels)
        - Task: Masked Signal Modeling (MSM)
        - Knowledge: General biosignal patterns
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 2: DOMAIN ADAPTATION (Quality SSL)            │
│                    (BUT-PPG Contrastive Learning)                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
        artifacts/hybrid_vitaldb_fixed/stage2_butppg_quality_ssl/
        - Same encoder (fine-tuned)
        - New dataset: BUT-PPG (PPG + ECG + ACC, 5→2 channels)
        - Task: Quality-aware contrastive learning
        - Knowledge: Quality-relevant features
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 3: SUPERVISED FINE-TUNING                     │
│                    (BUT-PPG Quality Classification)              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
        artifacts/hybrid_vitaldb_fixed/stage3_supervised_finetune/
        - Same encoder + Classification head
        - Task: Binary quality classification (Good/Bad)
        - Knowledge: Task-specific decision boundaries
```

---

## Detailed Component Architecture

### 1. Foundation Model (Stage 1) - IBM TTM Backbone

```
INPUT: [Batch, 2 channels (PPG+ECG), 1024 timesteps]
  ↓
┌─────────────────────────────────────────────────────────────────┐
│                       PATCHIFICATION                             │
│  Splits signal into patches: 1024 samples → 8 patches × 128     │
│  Output: [B, 8 patches, 2 channels, 128 samples/patch]          │
└─────────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────────┐
│                    IBM TTM ENCODER (Frozen)                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Linear Projection: [8, 2, 128] → [8, 192]              │    │
│  │    (Embeds each patch to 192-dim latent space)          │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ↓                                                               │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  TTM Transformer Blocks (Mix Time + Mix Channels)       │    │
│  │  - Block 1: Self-attention across patches               │    │
│  │  - Block 2: Cross-channel mixing                        │    │
│  │  - ...repeated...                                       │    │
│  │  - Block N: Final representation                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ↓                                                               │
│  OUTPUT: [B, 8 patches, 192-dim features]                       │
│          ↓                             ↓                         │
│     Patch-level [B,P,D]         Pooled [B,D]                    │
│     (for SSL tasks)             (for classification)            │
└─────────────────────────────────────────────────────────────────┘
```

**Key Parameters:**
- `d_model = 192`: Latent dimension (fixed by IBM TTM)
- `patch_size = 128`: Adaptive (depends on context_length)
- `num_patches = 8`: context_length / patch_size = 1024 / 128
- `trainable_params = 946,904`: Total encoder parameters
- `frozen_params = 556,352`: After freezing backbone

**IBM TTM Details:**
- **Model**: `ibm-granite/granite-timeseries-ttm-r1`
- **Architecture**: Tiny Time Mixer (Transformer-based)
- **Mixing**: Alternates between time-mixing and channel-mixing blocks
- **Pretrained**: Yes (on time-series forecasting tasks)
- **Adaptation**: We use the encoder part for representation learning

---

### 2. SSL Reconstruction Head (Stage 1 & 2)

```
INPUT: Patch-level features [B, 8 patches, 192-dim]
  ↓
┌─────────────────────────────────────────────────────────────────┐
│              RECONSTRUCTION HEAD (Lightweight)                   │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  Linear Projection: 192 → (2 channels × 128 samples)   │     │
│  │                     192 → 256                          │     │
│  └────────────────────────────────────────────────────────┘     │
│  ↓                                                               │
│  Reshape: [B, 8, 256] → [B, 8, 2 channels, 128 samples]         │
│  ↓                                                               │
│  Permute & Fold: [B, 8, 2, 128] → [B, 2, 1024]                  │
│  (Concatenates patches along time dimension)                    │
└─────────────────────────────────────────────────────────────────┘
  ↓
OUTPUT: Reconstructed signal [B, 2 channels, 1024 timesteps]
```

**Design Philosophy:**
Following **MAE (Masked Autoencoder)** design principle:
- **Asymmetric encoder-decoder**: Heavy encoder, lightweight decoder
- **Why?** The encoder does the heavy lifting of learning representations
- **Decoder role**: Simple projection back to signal space for reconstruction
- **Parameters**: ~49k (only 5% of encoder size)

**Implementation:**
- File: `src/models/decoders.py:ReconstructionHead1D`
- Single linear layer + reshape + permute
- No convolutions or complex upsampling
- Allows dynamic patch_size updates via `update_patch_size()`

---

### 3. Stage 2: Quality-Aware SSL Architecture

```
INPUT: BUT-PPG signal [B, 5 channels, 1250 samples]
  ↓
┌─────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING                                 │
│  - Extract first 2 channels: [B, 5, 1250] → [B, 2, 1250]       │
│  - Crop to 1024 samples: [B, 2, 1250] → [B, 2, 1024]           │
└─────────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────────┐
│                      MASKING (40% random)                        │
│  Masks 40% of patches for reconstruction                        │
│  Output: masked_signal [B, 2, 1024], mask [B, 8 patches]        │
└─────────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────────┐
│                      TTM ENCODER                                 │
│  (Pretrained from Stage 1, partially frozen)                    │
│  Output: features [B, 8, 192]                                   │
└─────────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────────┐
│                 DUAL-OBJECTIVE TRAINING                          │
│  ┌──────────────────────┐  ┌────────────────────────────────┐   │
│  │ Reconstruction Head  │  │ Quality Contrastive Learning   │   │
│  │ (MSM Loss)          │  │ (InfoNCE Loss)                 │   │
│  │                     │  │                                │   │
│  │ Reconstructed       │  │ Positive pairs: Similar quality│   │
│  │ signal [B,2,1024]   │  │ Negative pairs: Different Q    │   │
│  │                     │  │                                │   │
│  │ Loss_recon × 0.3    │  │ Loss_contrast × 1.0            │   │
│  └──────────────────────┘  └────────────────────────────────┘   │
│                     ↓              ↓                             │
│              Combined Loss = Loss_contrast + 0.3 × Loss_recon    │
└─────────────────────────────────────────────────────────────────┘
```

**Quality Contrastive Learning (InfoNCE):**

```python
# Positive pairs: Signals with similar quality scores
positive_mask = |quality_i - quality_j| < threshold

# Negative pairs: Signals with different quality scores
negative_mask = |quality_i - quality_j| >= threshold

# InfoNCE Loss
similarity = exp(cosine_similarity(features_i, features_j) / temperature)
loss = -log(sum(positive_similarities) / sum(all_similarities))
```

**Hyperparameters:**
- **Temperature (τ)**: 0.07 (controls softness of similarity)
- **Contrastive weight**: 1.0
- **Reconstruction weight**: 0.3
- **Balanced sampling**: Equal samples per quality bin (Low/Medium/High)

**Implementation:**
- File: `scripts/continue_ssl_butppg_quality.py`
- Loss: `src/ssl/quality_contrastive.py:QualityContrastiveLoss`

---

### 4. Stage 3: Supervised Classification Architecture

```
INPUT: BUT-PPG signal [B, 5 channels, 1250 samples]
  ↓
(Same preprocessing as Stage 2)
  ↓
┌─────────────────────────────────────────────────────────────────┐
│                      TTM ENCODER                                 │
│  (Fine-tuned from Stage 2)                                      │
│  Output: features [B, 192] (pooled across patches)              │
└─────────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────────┐
│              CLASSIFICATION HEAD (Linear)                        │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  Linear: 192 → 2 classes (Good/Bad quality)            │     │
│  │  Softmax: Convert logits to probabilities              │     │
│  └────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
  ↓
OUTPUT: Class probabilities [B, 2]
  ↓
LOSS: Cross-Entropy Loss
```

**3-Stage Fine-Tuning Strategy:**

| Stage | Epochs | Frozen Components | Trainable Components | Learning Rate |
|-------|--------|-------------------|---------------------|---------------|
| **1. Head-only** | 3 | Entire encoder | Classification head only | 1e-4 |
| **2. Progressive** | 5-7 | First N-2 blocks | Last 2 blocks + head | 1e-4 |
| **3. Full FT** | 5 | None | All weights | 1e-5 (very low) |

**Why this strategy?**
- **Prevents catastrophic forgetting**: Gradually unfreezing preserves pretrained knowledge
- **Head warmup**: First trains the random head to avoid corrupting encoder
- **Top-down adaptation**: Fine-tunes task-specific features first (top layers)
- **Conservative full FT**: Very low LR ensures stable convergence

**Implementation:**
- File: `scripts/finetune_butppg.py`
- Trainer: `src/models/trainers.py:TrainerClf`

---

## Data Flow & Transformations

### VitalDB Dataset (Stage 1)

```
┌─────────────────────────────────────────────────────────────────┐
│                        VITALDB (Stage 1)                         │
├─────────────────────────────────────────────────────────────────┤
│ Dataset:        3,667 ICU patients (subject-level splits)       │
│ Modalities:     PPG (PLETH) + ECG (ECG_II)                      │
│ Sampling rate:  125 Hz                                          │
│ Window size:    1024 samples @ 125Hz = 8.192 seconds            │
│ Channel order:  [0: PPG, 1: ECG]                                │
│                                                                  │
│ Preprocessing:                                                   │
│  1. Band-pass filtering: 0.5-8 Hz (PPG), 0.5-40 Hz (ECG)       │
│  2. Z-score normalization (per-window)                          │
│  3. Quality filtering: Remove low-quality segments              │
│                                                                  │
│ SSL Task:       Masked Signal Modeling (MSM)                    │
│  - Mask ratio: 40% of patches                                   │
│  - Mask type:  Random patch-level masking                       │
│  - Objective:  Reconstruct masked patches from context          │
│                                                                  │
│ Output:         Foundation model checkpoint                      │
│                 artifacts/foundation_model/best_model.pt         │
└─────────────────────────────────────────────────────────────────┘
```

**Subject-Level Splits:**
- **Train**: 2,933 cases (80%)
- **Val**: 367 cases (10%)
- **Test**: 367 cases (10%)
- **Split file**: `configs/splits/splits_full.json`
- **Critical**: Always use subject-level splits to prevent data leakage!

---

### BUT-PPG Dataset (Stage 2 & 3)

```
┌─────────────────────────────────────────────────────────────────┐
│                      BUT-PPG (Stage 2 & 3)                       │
├─────────────────────────────────────────────────────────────────┤
│ Dataset:        3,888 smartphone PPG recordings                 │
│ Modalities:     PPG + ECG + ACC_X + ACC_Y + ACC_Z (5 channels)  │
│ Sampling rate:  125 Hz                                          │
│ Window size:    1250 samples @ 125Hz = 10 seconds               │
│ Channel order:  [0: PPG, 1: ECG, 2: ACC_X, 3: ACC_Y, 4: ACC_Z]  │
│                                                                  │
│ Splits:                                                          │
│  - Train: 3,110 samples (80%)                                   │
│  - Val:   388 samples (10%)                                     │
│  - Test:  390 samples (10%)                                     │
│                                                                  │
│ Quality Stratification (Train):                                 │
│  - Low quality:    32 samples (1.0%)                            │
│  - Medium quality: 3,074 samples (98.8%)                        │
│  - High quality:   4 samples (0.1%)                             │
│                                                                  │
│ Preprocessing for TTM:                                          │
│  1. Extract channels: [5, 1250] → [2, 1250] (PPG + ECG only)   │
│  2. Crop length: [2, 1250] → [2, 1024] (match VitalDB)         │
│  3. Z-score normalization                                       │
│                                                                  │
│ Stage 2 Task:   Quality-aware SSL (contrastive + reconstruction)│
│ Stage 3 Task:   Binary classification (Good vs Bad quality)     │
│                                                                  │
│ Labels:         quality-hr-ann.csv (quality scores 0-1)         │
│                 Binary: score ≥ 0.5 = Good, < 0.5 = Bad         │
└─────────────────────────────────────────────────────────────────┘
```

**Quality Score Distribution:**
- Quality scores range from 0.0 (worst) to 1.0 (best)
- Most samples are medium quality (~0.6-0.65)
- Very few high-quality samples (>0.8)
- Highly imbalanced → Use balanced sampling in Stage 2

---

### Cross-Dataset Adaptation

**Challenge**: VitalDB and BUT-PPG have different:
- Number of channels (2 vs 5)
- Window lengths (1024 vs 1250 samples)
- Recording conditions (ICU vs smartphone)
- Quality distributions

**Solution**: Preprocessing layer handles mismatches:

```python
# 1. Extract first 2 channels (PPG + ECG)
signals = signals[:, :2, :]  # [B, 5, 1250] → [B, 2, 1250]

# 2. Crop/pad to encoder's context_length
expected_length = encoder.context_length  # 1024
current_length = signals.shape[2]  # 1250

if current_length > expected_length:
    signals = signals[:, :, :expected_length]  # Crop
elif current_length < expected_length:
    pad_length = expected_length - current_length
    signals = F.pad(signals, (0, pad_length))  # Pad

# 3. Z-score normalization (per-channel, per-sample)
signals = (signals - signals.mean(dim=2, keepdim=True)) / \
          (signals.std(dim=2, keepdim=True) + 1e-8)
```

---

## Key Design Principles

### 1. Transfer Learning via Self-Supervised Pretraining

**Motivation**: Labeled biosignal data is expensive and scarce.

**Approach**:
1. **Stage 1 (VitalDB)**: Learn general biosignal patterns from large unlabeled dataset
   - Heart rate patterns, respiratory variations, noise characteristics
   - No labels needed → Scalable to millions of samples
2. **Stage 2 (BUT-PPG)**: Adapt to new domain and task (quality assessment)
   - Bridge domain gap between ICU and smartphone recordings
   - Learn quality-relevant features via contrastive learning
3. **Stage 3 (BUT-PPG)**: Fine-tune for specific classification task
   - Task-specific decision boundaries
   - Minimal labeled data needed (leverages Stage 1 & 2 knowledge)

---

### 2. Patch-Based Signal Processing

**Inspiration**: Vision Transformer (ViT) treats images as sequences of patches.

**Adaptation for 1D signals**:
- **Patchification**: Split 1024-sample signal into 8 patches × 128 samples
- **Benefits**:
  - **Local patterns**: Each patch captures ~1 second of physiological activity
  - **Global structure**: Transformer learns relationships across patches
  - **Efficiency**: Fewer tokens (8 patches) vs 1024 individual timesteps
  - **Flexibility**: Can adapt patch size based on context length

**Example**: For heart rate ~60 bpm @ 125Hz:
- 1 cardiac cycle ≈ 125 samples
- 1 patch (128 samples) ≈ 1 heartbeat
- Transformer learns inter-beat patterns across 8 beats

---

### 3. Dual-Objective SSL (Stage 2)

**Why combine reconstruction + contrastive learning?**

| Objective | What it learns | Limitation |
|-----------|----------------|------------|
| **Reconstruction** | Signal fidelity, temporal structure | Doesn't explicitly learn quality features |
| **Contrastive** | Quality-discriminative features | May ignore signal details |
| **Combined** | Both signal fidelity AND quality semantics | Best of both worlds! |

**Mathematical formulation**:
```
L_total = λ_contrast × L_InfoNCE + λ_recon × L_MSE

where:
- L_InfoNCE: Quality-aware contrastive loss
- L_MSE: Reconstruction loss (MSE between original & reconstructed)
- λ_contrast = 1.0 (emphasize quality learning)
- λ_recon = 0.3 (maintain signal structure)
```

---

### 4. Progressive Unfreezing Strategy

**Problem**: Naively fine-tuning all weights can cause **catastrophic forgetting**.

**Solution**: 3-stage progressive unfreezing:

```
Stage 1 (Head-only):
┌────────────────────┐
│ Classification Head│ ← Trainable (random init)
├────────────────────┤
│ TTM Block N        │ ← Frozen
│ ...                │ ← Frozen
│ TTM Block 1        │ ← Frozen
└────────────────────┘

Stage 2 (Progressive):
┌────────────────────┐
│ Classification Head│ ← Trainable
├────────────────────┤
│ TTM Block N        │ ← Trainable (task-specific features)
│ TTM Block N-1      │ ← Trainable
├────────────────────┤
│ ...                │ ← Frozen (preserve pretrained features)
│ TTM Block 1        │ ← Frozen
└────────────────────┘

Stage 3 (Full FT):
┌────────────────────┐
│ Classification Head│ ← Trainable (LR = 1e-5)
├────────────────────┤
│ TTM Block N        │ ← Trainable (LR = 1e-5)
│ ...                │ ← Trainable (LR = 1e-5)
│ TTM Block 1        │ ← Trainable (LR = 1e-5)
└────────────────────┘
```

**Why this works**:
- **Bottom layers**: Learn low-level features (universal across tasks) → Keep frozen
- **Top layers**: Learn task-specific features → Fine-tune first
- **Very low LR**: Final full fine-tuning with 10× lower LR → Stable convergence

---

### 5. Runtime Configuration Synchronization

**Challenge**: TTM auto-adapts patch_size and d_model during first forward pass.

**Problem**: Hard-coded configs become stale after IBM weights are loaded.

**Solution**: Always query runtime values from loaded model:

```python
# ✓ CORRECT: Query runtime values
if hasattr(encoder, 'patch_size'):
    patch_size = encoder.patch_size  # Runtime value
elif hasattr(encoder, 'backbone') and hasattr(encoder.backbone, 'config'):
    patch_size = encoder.backbone.config.patch_length  # TTM config

# Query d_model from actual loaded weights (not initialization value)
if hasattr(encoder, 'backbone') and hasattr(encoder.backbone, 'config'):
    d_model = encoder.backbone.config.d_model  # TTM actual d_model (192)
elif hasattr(encoder, 'encoder_dim'):
    d_model = encoder.encoder_dim  # May be stale if set before loading

# ✗ WRONG: Use checkpoint config or hard-coded values
patch_size = 125  # May not match encoder's actual patch_size!
d_model = 64      # May not match TTM's actual d_model (192)!
```

**Why this matters**:
- TTM adapts `patch_size = context_length / num_patches`
- IBM pretrained weights have fixed `d_model = 192`
- Using mismatched values causes dimension errors during forward pass

---

## Architecture Parameters

### Model Configuration

| Component | Parameter | Value | Notes |
|-----------|-----------|-------|-------|
| **Encoder** | variant | `ibm-granite/granite-timeseries-ttm-r1` | IBM TTM pretrained |
| | d_model | 192 | Fixed by IBM TTM |
| | patch_size | 128 | Adaptive (1024 / 8) |
| | num_patches | 8 | context_length / patch_size |
| | context_length | 1024 | 8.192s @ 125Hz |
| | input_channels | 2 | PPG + ECG |
| | total_params | 946,904 | Full model |
| | frozen_params | 556,352 | After freezing backbone |
| **Decoder** | d_model | 192 | Match encoder |
| | patch_size | 128 | Match encoder |
| | n_channels | 2 | PPG + ECG |
| | output_dim | 256 | 2 × 128 |
| | parameters | ~49k | 5% of encoder |
| **Classification Head** | input_dim | 192 | From encoder pooling |
| | output_dim | 2 | Binary (Good/Bad) |
| | parameters | ~390 | Single linear layer |

---

### Training Hyperparameters

#### Stage 1: VitalDB SSL Pre-training

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Data** | | |
| Train samples | ~14,000 windows | 2,933 patients |
| Val samples | ~1,800 windows | 367 patients |
| Batch size | 128 | Fits in 16GB GPU |
| **SSL** | | |
| Mask ratio | 0.4 | 40% patches masked |
| Mask type | Random | Patch-level |
| Loss | MSE + STFT | Reconstruction loss |
| **Optimization** | | |
| Epochs | 100 | Full training |
| Learning rate | 1e-4 | Base LR |
| Optimizer | AdamW | Weight decay 0.01 |
| Scheduler | CosineAnnealing | T_max = epochs |
| Gradient clip | 1.0 | Prevent explosion |
| AMP | Enabled | Mixed precision |

#### Stage 2: BUT-PPG Quality SSL

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Data** | | |
| Train samples | 3,110 | Balanced sampling |
| Val samples | 388 | |
| Batch size | 128 | |
| **SSL** | | |
| Contrastive weight | 1.0 | InfoNCE loss |
| Reconstruction weight | 0.3 | MSE loss |
| Temperature (τ) | 0.07 | Contrastive softness |
| Balanced sampling | Enabled | Equal per quality bin |
| **Optimization** | | |
| Epochs | 5 | Quick adaptation |
| Learning rate | 5e-5 | Lower than Stage 1 |
| Optimizer | AdamW | Weight decay 0.01 |

#### Stage 3: BUT-PPG Supervised Fine-tuning

**Stage 3.1: Head-only (3 epochs)**

| Parameter | Value |
|-----------|-------|
| Frozen | Entire encoder |
| Trainable | Classification head only |
| Learning rate | 1e-4 |
| Batch size | 64 |

**Stage 3.2: Progressive unfreezing (5-7 epochs)**

| Parameter | Value |
|-----------|-------|
| Frozen | First N-2 blocks |
| Trainable | Last 2 blocks + head |
| Learning rate | 1e-4 |
| Batch size | 64 |

**Stage 3.3: Full fine-tuning (5 epochs)**

| Parameter | Value |
|-----------|-------|
| Frozen | None |
| Trainable | All weights |
| Learning rate | 1e-5 (very low) |
| Batch size | 64 |

---

## File Reference

### Core Architecture Files

| File | Purpose |
|------|---------|
| `src/models/ttm_adapter.py` | TTMAdapter wrapper around IBM TTM |
| `src/models/decoders.py` | ReconstructionHead1D for SSL |
| `src/models/heads.py` | Task-specific classification/regression heads |
| `src/models/trainers.py` | TrainerClf, TrainerReg with progressive unfreezing |

### SSL Components

| File | Purpose |
|------|---------|
| `src/ssl/masking.py` | Random/block patch masking |
| `src/ssl/objectives.py` | MSM + STFT loss |
| `src/ssl/quality_contrastive.py` | Quality-aware InfoNCE loss |
| `src/ssl/pretrainer.py` | SSL training loop (Stage 1) |

### Training Scripts

| File | Purpose |
|------|---------|
| `scripts/pretrain_vitaldb_ssl.py` | Stage 1: VitalDB SSL pre-training |
| `scripts/continue_ssl_butppg_quality.py` | Stage 2: BUT-PPG quality SSL |
| `scripts/finetune_butppg.py` | Stage 3: Supervised fine-tuning |
| `scripts/train_hybrid_ssl_pipeline.py` | End-to-end 3-stage pipeline |

### Data Loaders

| File | Purpose |
|------|---------|
| `src/data/vitaldb_dataset.py` | VitalDB dataset (PPG + ECG) |
| `src/data/butppg_dataset.py` | BUT-PPG dataset (5 channels) |
| `src/data/butppg_quality_dataset.py` | Quality-stratified wrapper |

---

## Why This Architecture Works

### 1. Foundation Model Benefits

The **foundation model** (Stage 1) provides:
- **General biosignal understanding**: Heart rate patterns, noise, artifacts
- **Transfer learning**: Pretrained on 3,667 ICU patients
- **Task-agnostic**: Can fine-tune for multiple downstream tasks
- **Data efficiency**: Reduces labeled data requirements for downstream tasks

### 2. Quality-Aware SSL (Stage 2)

**Problem**: Direct transfer from VitalDB (ICU) to BUT-PPG (smartphone) has domain gap.

**Solution**: Intermediate SSL stage bridges the gap:
- **Contrastive learning**: Learns quality-discriminative features
- **Reconstruction**: Maintains signal fidelity
- **Balanced sampling**: Handles class imbalance

**Result**: Better representations for quality classification than direct fine-tuning.

### 3. Progressive Unfreezing (Stage 3)

**Problem**: Full fine-tuning can cause catastrophic forgetting.

**Solution**: Gradual adaptation preserves pretrained knowledge:
- **Head warmup**: Trains random head first (avoid corrupting encoder)
- **Top-down**: Fine-tunes task-specific features (top layers) first
- **Conservative full FT**: Very low LR for stable convergence

**Result**: Best of both worlds - task adaptation + preserved pretrained knowledge.

---

## Expected Performance

### Stage 2 Results (Quality SSL)

Based on training logs:
- **Validation loss**: 0.8774 → 0.8641 (improving)
- **Contrastive loss**: 0.5671 (stable, quality embeddings learned)
- **Reconstruction loss**: 1.0344 → 0.9900 (signal fidelity improving)

### Stage 3 Targets (Supervised Fine-tuning)

- **Target AUROC**: ≥0.80 (pipeline threshold)
- **Expected**: 0.85-0.90 (with hybrid SSL approach)
- **Baseline** (direct fine-tuning): ~0.75-0.80
- **Improvement**: ~5-10% AUROC gain from quality SSL stage

---

## Troubleshooting Common Issues

### 1. Dimension Mismatch Errors

**Error**: `RuntimeError: size mismatch (2048 vs 1024)`

**Cause**: TTM's adaptive patching outputs more patches than expected.

**Fix**: Crop decoder output to match target length:
```python
target_length = signals.shape[2]
if reconstructed.shape[2] > target_length:
    reconstructed = reconstructed[:, :, :target_length]
```

### 2. Patch Size Errors

**Error**: `AssertionError: T=1024 must be divisible by patch_size=125`

**Cause**: Using checkpoint config patch_size instead of runtime value.

**Fix**: Query encoder's actual patch_size:
```python
patch_size = encoder.patch_size  # Not from config!
```

### 3. d_model Mismatch

**Error**: `ValueError: Input latent dimension 192 doesn't match expected d_model=64`

**Cause**: Using stale `encoder_dim` set before IBM weights loaded.

**Fix**: Query from backbone config:
```python
d_model = encoder.backbone.config.d_model  # Actual TTM d_model
```

### 4. Channel/Length Mismatch

**Error**: Signal shape errors when loading BUT-PPG into VitalDB encoder.

**Cause**: BUT-PPG has 5 channels × 1250 samples, VitalDB expects 2 × 1024.

**Fix**: Preprocessing layer:
```python
signals = signals[:, :2, :1024]  # Extract PPG+ECG, crop to 1024
```

---

## References

### Papers

1. **Masked Autoencoders (MAE)**: He et al. (2022) "Masked Autoencoders Are Scalable Vision Learners"
2. **Vision Transformer (ViT)**: Dosovitskiy et al. (2021) "An Image is Worth 16x16 Words"
3. **InfoNCE Loss**: Oord et al. (2018) "Representation Learning with Contrastive Predictive Coding"
4. **IBM TTM**: Das et al. (2024) "Tiny Time Mixers: Fast Pretraining for 512-Length Context"

### Datasets

1. **VitalDB**: Lee et al. (2018) "VitalDB: Open-source vital sign database"
   - https://vitaldb.net
   - 3,667 ICU patients, multi-modal biosignals
2. **BUT-PPG**: Nemcova et al. (2020) "Brno University of Technology Smartphone PPG Database"
   - https://physionet.org/content/but-ppg/2.0.0/
   - 3,888 smartphone PPG recordings with quality labels

### Code

- **IBM TTM**: https://github.com/IBM/tsfm
- **This repository**: https://github.com/AliABULIEL/TinyFoundationModelForBioSignals

---

## Changelog

- **2025-10-22**: Added comprehensive architecture documentation
- **2025-10-22**: Fixed Stage 3 argument mapping in hybrid pipeline
- **2025-10-22**: Fixed all dimension mismatch errors (patch_size, d_model, signal length)
- **2025-10-22**: Successfully completed Stage 2 quality SSL training

---

**Last Updated**: 2025-10-22
**Status**: Architecture stable, Stage 2 complete, Stage 3 ready to run
