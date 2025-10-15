# Downstream Task Evaluation Guide

**Status:** ✅ COMPLETE & READY TO USE
**Date:** October 15, 2025

This guide explains how to use the comprehensive downstream task evaluation system for benchmarking your foundation model against article targets.

---

## Overview

The evaluation system implements **6 downstream tasks** across 2 datasets:

### VitalDB Tasks (3 tasks)
1. **Hypotension Prediction (10-min ahead)** - Binary classification, Target: AUROC ≥0.91
2. **Blood Pressure (MAP) Estimation** - Regression, Target: MAE ≤5.0 mmHg
3. **AAMI Compliance** - BP estimation meeting AAMI standard (ME≤5, SDE≤8)

### BUT-PPG Tasks (3 tasks)
4. **Signal Quality Classification** - Binary classification, Target: AUROC ≥0.88
5. **Heart Rate Estimation** - Regression, Target: MAE ≤2.0 bpm
6. **Motion Classification** - 8-class classification, Target: Accuracy ≥0.85

---

## System Architecture

```
src/eval/
├── tasks/
│   ├── __init__.py
│   ├── vitaldb_tasks.py      # VitalDB task evaluators
│   └── butppg_tasks.py       # BUT-PPG task evaluators
└── reports/
    ├── __init__.py
    └── benchmark_comparison.py  # Report generator

scripts/
└── run_downstream_evaluation.py  # Master script
```

### Key Features

✅ **Subject-level evaluation** - Prevents data leakage
✅ **Bootstrap confidence intervals** - 95% CIs for all metrics (1000 iterations)
✅ **Benchmark comparison** - Automatic comparison to article targets and SOTA
✅ **Comprehensive reporting** - HTML reports, markdown tables, visualization plots
✅ **AAMI compliance** - Proper per-subject aggregation for BP estimation
✅ **Flexible** - Evaluate VitalDB only, BUT-PPG only, or both

---

## Usage

### Basic Usage

```bash
# Evaluate both VitalDB and BUT-PPG
python3 scripts/run_downstream_evaluation.py \
  --vitaldb-checkpoint artifacts/vitaldb_finetuned/best_model.pt \
  --butppg-checkpoint artifacts/butppg_finetuned/best_model.pt \
  --vitaldb-data data/processed/vitaldb \
  --butppg-data data/processed/butppg \
  --output-dir artifacts/downstream_evaluation
```

### Evaluate VitalDB Only

```bash
python3 scripts/run_downstream_evaluation.py \
  --vitaldb-checkpoint artifacts/vitaldb_finetuned/best_model.pt \
  --vitaldb-data data/processed/vitaldb \
  --output-dir artifacts/vitaldb_evaluation
```

### Evaluate BUT-PPG Only

```bash
python3 scripts/run_downstream_evaluation.py \
  --butppg-checkpoint artifacts/butppg_finetuned/best_model.pt \
  --butppg-data data/processed/butppg \
  --output-dir artifacts/butppg_evaluation
```

### Options

```bash
--batch-size 64             # Batch size for evaluation (default: 64)
--no-ci                     # Skip confidence interval computation (faster)
--device cuda               # Device to run on (default: auto-detect)
--model-name "My Model"     # Model name for report
--model-params "1.2M"       # Parameter count for report
```

---

## Expected Data Format

### VitalDB Data Structure

```
data/processed/vitaldb/
├── hypotension/
│   └── test.npz          # Keys: 'signals', 'labels', 'subject_ids' (optional)
└── blood_pressure/
    └── test.npz          # Keys: 'signals', 'labels', 'subject_ids' (optional)
```

**Format:**
- `signals`: `[N, C, T]` - N samples, C channels, T timesteps (e.g., [1000, 2, 1024])
- `labels`: `[N]` - Binary labels (hypotension) or float values (BP)
- `subject_ids`: `[N]` - Subject IDs for subject-level evaluation (optional but recommended)

### BUT-PPG Data Structure

```
data/processed/butppg/
├── quality/
│   └── test.npz          # Keys: 'signals', 'labels', 'subject_ids'
├── heart_rate/
│   └── test.npz          # Optional
└── motion/
    └── test.npz          # Optional
```

**Format:**
- `signals`: `[N, C, T]` - N samples, C channels (5 for BUT-PPG: ACC_X/Y/Z + PPG + ECG)
- `labels`: `[N]` - Binary (quality), float (HR), or class labels (motion)
- `subject_ids`: `[N]` - Subject IDs (recommended)

---

## Output

### Generated Files

```
artifacts/downstream_evaluation/
├── benchmark_report.html          # Main HTML report (open in browser)
├── benchmark_comparison.png       # Comparison plots
├── vitaldb_comparison.md          # VitalDB results table
├── butppg_comparison.md           # BUT-PPG results table
└── all_results.json              # Complete results JSON
```

### Report Contents

**HTML Report includes:**
- Model information and timestamp
- VitalDB results table with all metrics and benchmarks
- BUT-PPG results table with all metrics and benchmarks
- Visual comparison plots (bar charts)
- Pass/Fail status for each task
- Confidence intervals (95% CIs)
- Gap to SOTA analysis

**Markdown Tables:**
- Formatted tables for easy copying to papers/docs
- All metrics, targets, baselines, and SOTA values
- Clear pass/fail indicators

**Visualization Plots:**
- 4-panel comparison showing your model vs targets vs SOTA/baselines
- Error bars showing 95% confidence intervals
- Color-coded for easy interpretation

---

## Example Output

### Console Output

```
================================================================================
DOWNSTREAM TASK EVALUATION & BENCHMARK COMPARISON
================================================================================
Output directory: artifacts/downstream_evaluation
Device: cuda
Batch size: 64
Compute CI: True

Loading checkpoint: artifacts/vitaldb_finetuned/best_model.pt
  ✓ Model loaded: TTMForClassification
  ✓ Parameters: 1,234,567

Loading VitalDB test data...
  ✓ Hypotension: 5000 samples
  ✓ Blood Pressure: 5000 samples

================================================================================
TASK 1: Hypotension Prediction (10-min ahead)
================================================================================
AUROC: 0.925 (Target: 0.910)
  95% CI: [0.918, 0.932]
AUPRC: 0.889
Sensitivity: 0.876
Specificity: 0.845
Samples: 5000 (Positive: 1200, Negative: 3800)
Target Met: ✅ YES
Gap to SOTA: -0.009

================================================================================
TASK 2 & 3: Blood Pressure Estimation (MAE + AAMI)
================================================================================
MAE: 4.82 mmHg (Target: 5.00)
  95% CI: [4.65, 4.98]
RMSE: 6.23 mmHg
R²: 0.845

AAMI Compliance:
  ME: 0.45 mmHg (Limit: 5.0) ✅
  SDE: 6.89 mmHg (Limit: 8.0) ✅
  AAMI Compliant: ✅ YES
Samples: 5000
MAE Target Met: ✅ YES
Gap to SOTA: +1.02 mmHg

...

================================================================================
GENERATING BENCHMARK COMPARISON REPORT
================================================================================
✓ Saved: artifacts/downstream_evaluation/benchmark_comparison.png
✓ Saved: artifacts/downstream_evaluation/benchmark_report.html
✓ Saved: artifacts/downstream_evaluation/vitaldb_comparison.md
✓ Saved: artifacts/downstream_evaluation/butppg_comparison.md
✓ Saved: artifacts/downstream_evaluation/all_results.json

================================================================================
EVALUATION COMPLETE!
================================================================================

Results saved to: artifacts/downstream_evaluation
  📊 HTML Report: artifacts/downstream_evaluation/benchmark_report.html
  📈 Plots: artifacts/downstream_evaluation/benchmark_comparison.png
  📄 JSON: artifacts/downstream_evaluation/all_results.json

================================================================================
SUMMARY
================================================================================

VitalDB Tasks:
  Hypotension: AUROC=0.925 ✅ PASS
  BP Estimation: MAE=4.82 mmHg ✅ PASS
  AAMI Compliance: ✅ PASS

BUT-PPG Tasks:
  Quality: AUROC=0.892 ✅ PASS
  Heart Rate: MAE=1.87 bpm ✅ PASS
  Motion: Accuracy=0.863 ✅ PASS

================================================================================
```

### Markdown Table Example

```markdown
# VitalDB Benchmark Comparison

| Task | Metric | Your Model | 95% CI | Target | SOTA | Status | Gap to SOTA |
|------|--------|------------|--------|--------|------|--------|-------------|
| Hypotension (10-min) | AUROC | 0.925 | [0.918, 0.932] | 0.910 | 0.934 | ✅ PASS | -0.009 |
| | AUPRC | 0.889 | - | - | - | - | - |
| | Sensitivity | 0.876 | - | - | - | - | - |
| | Specificity | 0.845 | - | - | - | - | - |
| BP Estimation (MAP) | MAE (mmHg) | 4.82 | [4.65, 4.98] | 5.00 | 3.80 | ✅ PASS | +1.02 |
| | RMSE (mmHg) | 6.23 | - | - | - | - | - |
| BP AAMI Compliance | ME (mmHg) | 0.45 | - | 5.0 | - | ✅ | - |
| | SDE (mmHg) | 6.89 | - | 8.0 | - | ✅ | - |
```

---

## Programmatic Usage

You can also use the evaluators programmatically in your own scripts:

```python
from src.eval.tasks.vitaldb_tasks import run_all_vitaldb_tasks
from src.eval.tasks.butppg_tasks import run_all_butppg_tasks
from src.eval.reports.benchmark_comparison import BenchmarkComparator

# Load your model and data loaders
model = ...
hypotension_loader = ...
bp_loader = ...

# Run VitalDB tasks
vitaldb_results = run_all_vitaldb_tasks(
    model=model,
    hypotension_loader=hypotension_loader,
    bp_loader=bp_loader,
    device='cuda',
    compute_ci=True
)

# Generate report
comparator = BenchmarkComparator(output_dir='results')
comparator.generate_comparison_plots(vitaldb_results, butppg_results)
comparator.generate_full_report(vitaldb_results, butppg_results, model_info)
```

---

## Metrics Explained

### Classification Metrics

- **AUROC** (Area Under ROC Curve): Primary metric for binary classification. Range [0, 1], higher is better.
- **AUPRC** (Area Under Precision-Recall Curve): Useful for imbalanced datasets.
- **Sensitivity** (Recall): True positive rate, TP/(TP+FN)
- **Specificity**: True negative rate, TN/(TN+FP)
- **F1 Score**: Harmonic mean of precision and recall

### Regression Metrics

- **MAE** (Mean Absolute Error): Average absolute difference. Primary metric for BP and HR.
- **RMSE** (Root Mean Squared Error): More sensitive to outliers than MAE.
- **MAPE** (Mean Absolute Percentage Error): Percentage-based error metric.
- **R²**: Coefficient of determination. Warning: Can be misleading, use with caution.

### AAMI Compliance (Blood Pressure)

Per ANSI/AAMI SP10 standard:
- **ME** (Mean Error): Average error across subjects, must be ≤5 mmHg
- **SDE** (Standard Deviation of Error): Std dev of per-subject errors, must be ≤8 mmHg
- **Compliant**: Both ME and SDE conditions met

**Important:** Article requires per-subject aggregation first, then cross-subject statistics.

---

## Confidence Intervals

All metrics include 95% bootstrap confidence intervals (1000 bootstrap samples):
- Shows uncertainty in metric estimates
- Useful for comparing models statistically
- Reported as [CI_lower, CI_upper]

**Example:**
```
AUROC: 0.925
  95% CI: [0.918, 0.932]
```
This means we are 95% confident the true AUROC is between 0.918 and 0.932.

---

## Benchmark Targets

### VitalDB

| Task | Metric | Target | SOTA | SOTA Paper |
|------|--------|--------|------|------------|
| Hypotension (10-min) | AUROC | 0.91 | 0.934 | SAFDNet (2024) |
| BP Estimation | MAE | 5.0 mmHg | 3.8 mmHg | AnesthNet (2025) |
| AAMI Compliance | ME/SDE | 5.0/8.0 | - | AAMI Standard |

### BUT-PPG

| Task | Metric | Target | Baseline |
|------|--------|--------|----------|
| Quality | AUROC | 0.88 | 0.758 (traditional), 0.85 (DL) |
| Heart Rate | MAE | 2.0 bpm | 1.5 bpm (human expert) |
| Motion (8-class) | Accuracy | 0.85 | 0.70 (traditional) |

---

## Troubleshooting

### Issue: "Checkpoint missing 'config' key"

**Solution:** Ensure your checkpoint includes model config:
```python
torch.save({
    'model_state_dict': model.state_dict(),
    'config': model_config,
    'epoch': epoch,
}, 'checkpoint.pt')
```

### Issue: "Data not found"

**Solution:** Check data directory structure matches expected format (see "Expected Data Format" above).

### Issue: "Subject IDs not found"

**Solution:** Add subject IDs to your .npz files:
```python
np.savez('test.npz',
         signals=signals,
         labels=labels,
         subject_ids=subject_ids)
```

Subject IDs are optional but highly recommended for proper subject-level evaluation.

### Issue: Confidence intervals take too long

**Solution:** Use `--no-ci` flag to skip CI computation:
```bash
python3 scripts/run_downstream_evaluation.py ... --no-ci
```

---

## Integration with Paper/Reports

### For Papers

1. Use the generated markdown tables directly in your paper
2. Include the visualization plots as figures
3. Report confidence intervals for statistical rigor
4. Compare against SOTA and highlight improvements

### For Documentation

1. Copy the HTML report to your documentation site
2. Link to the report from your README
3. Show the comparison plots in your project overview

### For Presentations

1. Extract key metrics from the summary tables
2. Use the bar charts for visual comparison
3. Highlight pass/fail status for each task

---

## Next Steps

1. **Run evaluation on your fine-tuned models**
   ```bash
   python3 scripts/run_downstream_evaluation.py \
     --vitaldb-checkpoint your_checkpoint.pt \
     --vitaldb-data data/processed/vitaldb \
     --output-dir results
   ```

2. **Review the HTML report** to see detailed results

3. **Analyze gaps to targets** - Focus on tasks that didn't meet targets

4. **Iterate on fine-tuning** - Use insights to improve model

5. **Generate final report** for publication/documentation

---

## Citation

If you use this evaluation system, please cite:

```bibtex
@article{tinyfoundationmodel2025,
  title={Tiny Foundation Model for Biosignals},
  author={...},
  year={2025}
}
```

---

## Support

For issues or questions:
- Check `FINETUNE_AUDIT_REPORT.md` for fine-tuning guidance
- Review `CHANNEL_INFLATION_FIX.md` for channel mapping details
- See `BUTPPG_FIX_SUMMARY.md` for BUT-PPG data preparation

---

**Status:** ✅ System complete and ready for use!

Run your first evaluation:
```bash
python3 scripts/run_downstream_evaluation.py --help
```
