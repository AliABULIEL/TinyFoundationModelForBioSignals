# Complete Training and Evaluation Pipelines

This directory contains end-to-end bash scripts for training, evaluating, and benchmarking TTM models on VitalDB data.

## 📋 Overview

All scripts now **automatically use the BEST MODEL** (saved as `best_model.pt`) for evaluation and downstream tasks. The best model is saved during training based on validation performance.

## 🚀 Available Pipelines

### 1. **FastTrack Complete** (`run_fasttrack_complete.sh`)
**Foundation Model Mode - Quick Experiment**

```bash
bash scripts/run_fasttrack_complete.sh
```

**What it does:**
- ✅ Data preparation (50 train cases, 20 test cases)
- ✅ Training with frozen encoder (~3 hours)
- ✅ Evaluation on test set using **best_model.pt**
- ✅ Downstream tasks evaluation
- ✅ Benchmark comparison

**Configuration:**
- Frozen TTM encoder (805K params)
- Linear head only (~290K trainable)
- 10 epochs with early stopping
- Output: `artifacts/fasttrack_complete/`

**Runtime:** ~3-4 hours on single GPU

---

### 2. **High-Accuracy Complete** (`run_high_accuracy_complete.sh`)
**Fine-tuning with LoRA - Best Balance**

```bash
bash scripts/run_high_accuracy_complete.sh
```

**What it does:**
- ✅ Full VitalDB dataset processing
- ✅ LoRA + partial unfreezing (last 2 blocks)
- ✅ Extended training (up to 50 epochs)
- ✅ Evaluation using **best_model.pt**
- ✅ Downstream tasks + benchmarking
- ✅ HTML report generation

**Configuration:**
- Unfreeze last 2 transformer blocks
- LoRA adapters (r=16, alpha=32)
- MLP head (512→256 dims)
- Focal loss + label smoothing
- Output: `artifacts/high_accuracy_complete/`

**Runtime:** ~12-24 hours on GPU

---

### 3. **Full Fine-tuning Complete** (`run_full_finetune_complete.sh`)
**Maximum Performance - Deep Fine-tuning**

```bash
bash scripts/run_full_finetune_complete.sh
```

**What it does:**
- ✅ Full VitalDB dataset
- ✅ Deep fine-tuning (6/8 blocks unfrozen)
- ✅ High-capacity LoRA (r=32)
- ✅ 100 epochs with patience=15
- ✅ Evaluation using **best_model.pt**
- ✅ Comprehensive benchmarking

**Configuration:**
- Unfreeze 6/8 transformer blocks (75%)
- LoRA r=32, alpha=64
- Deep MLP head (768→512→256)
- Lower weight decay (0.005)
- Output: `artifacts/full_finetune_complete/`

**Runtime:** ~24-48 hours on GPU

---

### 4. **Evaluation Only** (`run_evaluation_only.sh`)
**Evaluate Existing Model**

```bash
bash scripts/run_evaluation_only.sh <model_checkpoint> [output_dir]
```

**Examples:**
```bash
# Evaluate best model from FastTrack
bash scripts/run_evaluation_only.sh artifacts/fasttrack_complete/checkpoints/best_model.pt

# Evaluate with custom output directory
bash scripts/run_evaluation_only.sh artifacts/high_accuracy_complete/checkpoints/best_model.pt my_results/
```

**What it does:**
- ✅ Test set evaluation
- ✅ Downstream tasks (8 tasks)
- ✅ Benchmark comparison
- ❌ No training

**Runtime:** ~30-60 minutes

---

### 5. **Compare Training Modes** (`compare_training_modes.sh`)
**Side-by-Side Comparison**

```bash
bash scripts/compare_training_modes.sh
```

**What it does:**
- Finds all completed training runs
- Generates comparison table
- Ranks models by accuracy
- Shows performance improvements
- Provides recommendations

**Output:**
```
====================================================================
RESULTS COMPARISON
====================================================================
Training Mode                  | Accuracy |     Loss |    AUROC |       F1
----------------------------------------------------------------------
full_finetune_complete         |   0.9234 |   0.2145 |   0.9456 |   0.9123
high_accuracy_complete         |   0.9012 |   0.2456 |   0.9234 |   0.8912
fasttrack_complete             |   0.8734 |   0.3012 |   0.8967 |   0.8645
====================================================================
```

---

## 📊 Training Modes Comparison

| Mode | Trainable Params | Training Time | Expected Accuracy | Best For |
|------|-----------------|---------------|-------------------|----------|
| **FastTrack** | ~290K (26%) | 3-4 hours | 85-89% | Quick validation, prototyping |
| **High-Accuracy** | ~500K (45%) | 12-24 hours | 89-92% | Production deployment |
| **Full Fine-tune** | ~700K (64%) | 24-48 hours | 92-94% | Maximum performance |

---

## 🎯 Recommended Workflow

### For Research/Development:

```bash
# 1. Start with FastTrack (quick validation)
bash scripts/run_fasttrack_complete.sh

# 2. Review results
cat artifacts/fasttrack_complete/evaluation/test_results.json

# 3. If promising, try High-Accuracy
bash scripts/run_high_accuracy_complete.sh

# 4. Compare performance
bash scripts/compare_training_modes.sh

# 5. If needed, go all-in with Full Fine-tuning
bash scripts/run_full_finetune_complete.sh
```

### For Production Deployment:

```bash
# Train with High-Accuracy mode (best balance)
bash scripts/run_high_accuracy_complete.sh

# Evaluate thoroughly
bash scripts/run_evaluation_only.sh \
    artifacts/high_accuracy_complete/checkpoints/best_model.pt \
    production_eval/

# Use the best model for deployment
cp artifacts/high_accuracy_complete/checkpoints/best_model.pt production/model.pt
```

### For Research Paper:

```bash
# Train all modes for comparison
bash scripts/run_fasttrack_complete.sh
bash scripts/run_high_accuracy_complete.sh
bash scripts/run_full_finetune_complete.sh

# Generate comprehensive comparison
bash scripts/compare_training_modes.sh

# Analyze downstream tasks
python scripts/benchmark_comparison.py \
    --results-dir artifacts/ \
    --format html \
    --plot
```

---

## 📁 Output Structure

Each pipeline creates the following structure:

```
artifacts/<mode>_complete/
├── splits.json                          # Train/val/test splits
├── raw_windows/
│   ├── train/                           # Preprocessed training windows
│   ├── val/                             # Validation windows
│   └── test/                            # Test windows
├── checkpoints/
│   ├── best_model.pt                    # ⭐ BEST MODEL (use this!)
│   ├── last_checkpoint.pt               # Latest checkpoint
│   ├── model.pt                         # Final model (backward compat)
│   └── metrics.json                     # Training history
├── evaluation/
│   └── test_results.json                # Test set evaluation
└── downstream_tasks/
    ├── hypotension_5min.json            # Task-specific results
    ├── blood_pressure_both.json
    ├── cardiac_output.json
    ├── ...
    └── aggregate_comparison.html        # Benchmark report
```

---

## 🔍 Key Files

### **best_model.pt** ⭐
- **What:** The model checkpoint with best validation performance
- **When saved:** Automatically during training when validation metric improves
- **Use for:** All evaluation, downstream tasks, production deployment

### **last_checkpoint.pt**
- **What:** The most recent training checkpoint
- **When saved:** After every epoch
- **Use for:** Resuming training, debugging

### **model.pt**
- **What:** Backward compatibility alias (same as best_model.pt)
- **Use for:** Legacy scripts

---

## ⚙️ Environment Variables

Customize pipeline behavior:

```bash
# Use CPU instead of GPU
CUDA_VISIBLE_DEVICES="" bash scripts/run_fasttrack_complete.sh

# Custom output directory
OUTPUT_DIR="my_experiment/" bash scripts/run_high_accuracy_complete.sh

# Use different Python interpreter
PYTHON=python3.9 bash scripts/run_full_finetune_complete.sh

# Disable FastTrack mode (use full dataset)
FASTTRACK="" bash scripts/run_fasttrack_complete.sh
```

---

## 🐛 Troubleshooting

### "Best model not found"
**Problem:** Script can't find `best_model.pt`
**Solution:** Check training completed successfully. Look for error messages in training output.

### "Out of memory"
**Problem:** GPU runs out of memory during training
**Solution:** 
1. Reduce batch size in `configs/run.yaml`
2. Use gradient accumulation
3. Try FastTrack mode first

### "Training too slow"
**Problem:** Training takes too long
**Solution:**
1. Start with FastTrack mode
2. Use smaller dataset (keep `--fasttrack` flag)
3. Reduce number of epochs
4. Check GPU utilization

### "Poor performance"
**Problem:** Model accuracy < 80%
**Solution:**
1. Check data quality and preprocessing
2. Try different training modes
3. Adjust hyperparameters
4. Enable data augmentation
5. Run comparison script to identify issues

---

## 📚 Additional Resources

- **Model Configuration:** `configs/model.yaml`
- **Training Parameters:** `configs/run.yaml`
- **Channel Settings:** `configs/channels.yaml`
- **Window Configuration:** `configs/windows.yaml`

- **Main Documentation:** `../README.md`
- **Implementation Details:** `../IMPLEMENTATION_SUMMARY.md`
- **Downstream Tasks Guide:** `../DOWNSTREAM_TASKS.md`
- **Critical Fixes:** `../CRITICAL_FIXES.md`

---

## 🎓 Examples

### Quick Test (30 minutes):
```bash
# Use minimal data
FASTTRACK="--fasttrack" OUTPUT_DIR="test_run/" bash scripts/run_fasttrack_complete.sh
```

### Production Training:
```bash
# Full pipeline with High-Accuracy
bash scripts/run_high_accuracy_complete.sh

# Deploy best model
cp artifacts/high_accuracy_complete/checkpoints/best_model.pt production/
```

### Research Experiment:
```bash
# Train all modes
for mode in fasttrack high_accuracy full_finetune; do
    bash scripts/run_${mode}_complete.sh
done

# Generate comparison
bash scripts/compare_training_modes.sh > results_comparison.txt
```

### Evaluate External Model:
```bash
# If you have a model from elsewhere
bash scripts/run_evaluation_only.sh \
    path/to/external_model.pt \
    external_model_evaluation/
```

---

## ✅ Verification

After running a pipeline, verify:

1. **Training completed:**
   ```bash
   ls artifacts/<mode>_complete/checkpoints/best_model.pt
   ```

2. **Evaluation ran:**
   ```bash
   cat artifacts/<mode>_complete/evaluation/test_results.json
   ```

3. **Downstream tasks completed:**
   ```bash
   ls artifacts/<mode>_complete/downstream_tasks/*.json
   ```

4. **Results look reasonable:**
   ```bash
   python3 -c "import json; print(json.load(open('artifacts/<mode>_complete/evaluation/test_results.json')))"
   ```

---

## 🏆 Success Criteria

Your pipeline is successful if:

- ✅ `best_model.pt` exists and is > 50MB
- ✅ Test accuracy > 0.80 (80%)
- ✅ Training completed without errors
- ✅ All downstream tasks have results
- ✅ Benchmark comparison generated

---

## 🚨 Important Notes

1. **Always use best_model.pt** for evaluation and deployment
2. **FastTrack is for validation only** - not production
3. **High-Accuracy mode** is recommended for most use cases
4. **Full Fine-tuning** only if you need maximum performance
5. **Monitor GPU memory** during training
6. **Save your configs** if you modify hyperparameters

---

## 📞 Need Help?

If you encounter issues:

1. Check the error messages carefully
2. Review the training logs in `artifacts/<mode>_complete/checkpoints/`
3. Compare with successful runs using `compare_training_modes.sh`
4. Verify your environment setup
5. Check available GPU memory and disk space

---

**Happy Training! 🎉**

All scripts automatically save and use the best model based on validation performance. No manual intervention needed!
