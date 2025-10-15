# Downstream Evaluation - Quick Reference

## 🚀 Quick Start

```bash
# Complete evaluation (VitalDB + BUT-PPG)
python3 scripts/run_downstream_evaluation.py \
  --vitaldb-checkpoint artifacts/vitaldb_finetuned/best_model.pt \
  --butppg-checkpoint artifacts/butppg_finetuned/best_model.pt \
  --vitaldb-data data/processed/vitaldb \
  --butppg-data data/processed/butppg \
  --output-dir artifacts/evaluation
```

## 📊 6 Downstream Tasks

### VitalDB (3 tasks)
| Task | Type | Target Metric | Target Value |
|------|------|---------------|--------------|
| Hypotension Prediction | Classification | AUROC | ≥0.91 |
| BP Estimation | Regression | MAE | ≤5.0 mmHg |
| AAMI Compliance | Regression | ME/SDE | ≤5.0/8.0 |

### BUT-PPG (3 tasks)
| Task | Type | Target Metric | Target Value |
|------|------|---------------|--------------|
| Signal Quality | Classification | AUROC | ≥0.88 |
| Heart Rate | Regression | MAE | ≤2.0 bpm |
| Motion (8-class) | Classification | Accuracy | ≥0.85 |

## 📁 Output Files

```
artifacts/evaluation/
├── benchmark_report.html          # 📊 Open this in browser
├── benchmark_comparison.png       # 📈 Plots
├── vitaldb_comparison.md          # 📝 VitalDB table
├── butppg_comparison.md           # 📝 BUT-PPG table
└── all_results.json              # 💾 Complete results
```

## 🎯 Key Features

✅ Subject-level evaluation (prevents data leakage)
✅ 95% bootstrap confidence intervals
✅ Automatic benchmark comparison
✅ HTML + markdown reports
✅ Visualization plots

## ⚙️ Common Options

```bash
--batch-size 64              # Evaluation batch size
--no-ci                      # Skip CIs (faster)
--device cuda                # Force device
--model-name "My Model"      # Report name
```

## 📖 Full Documentation

See `DOWNSTREAM_EVALUATION_GUIDE.md` for complete guide.

## 🔗 Related Files

- `src/eval/tasks/vitaldb_tasks.py` - VitalDB evaluators
- `src/eval/tasks/butppg_tasks.py` - BUT-PPG evaluators
- `src/eval/reports/benchmark_comparison.py` - Report generator
