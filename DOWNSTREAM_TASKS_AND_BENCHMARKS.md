# Downstream Tasks and Benchmarks for Biosignal Foundation Models
## Based on VitalDB and MIMIC-III Research Literature (2020-2025)

**Document Version:** 1.0  
**Last Updated:** October 11, 2025  
**Sources:** 20+ papers synthesized from the VitalDB Foundation Models Technical Implementation Guide

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Classification Tasks](#classification-tasks)
3. [Regression Tasks](#regression-tasks)
4. [Forecasting Tasks](#forecasting-tasks)
5. [Segmentation Tasks](#segmentation-tasks)
6. [Clinical Standards](#clinical-standards)
7. [Evaluation Protocols](#evaluation-protocols)
8. [Quick Reference Table](#quick-reference-table)

---

## Overview

Foundation models for biosignals should be evaluated on tasks that:
- Have **clinical relevance** (impact patient care)
- Have **established benchmarks** (published SOTA results)
- Have **standardized metrics** (comparable across studies)
- Cover **diverse signal types** (ECG, PPG, ABP, EEG)
- Test **transfer learning** ability (zero-shot to few-shot)

**Key Principle from Article:**
> "Foundation models pre-trained on clean clinical VitalDB data require domain-specific adaptation when deployed to noisy real-world settings—the key challenge for the next generation of biosignal AI systems."

---

## Classification Tasks

### 1. Arrhythmia Detection

**Clinical Definition:** Detect abnormal heart rhythms including AFib, PVC, VT, and normal sinus rhythm.

**Input Signals:**
- ECG (primary, 500 Hz)
- PPG (secondary, 50-125 Hz)

**Benchmarks:**

| Paper | Year | Dataset | N | Metric | Value | Method |
|-------|------|---------|---|--------|-------|--------|
| Hannun et al. | 2019 | Stanford | 91,232 | F1-score | 0.95 | 34-layer CNN |
| Ribeiro et al. | 2020 | CODE-15 | 827,337 | AUROC | 0.83 (AFib) | ResNet |
| MIT-BIH Baseline | 1980 | MIT-BIH | 48 | Accuracy | >95% | Various |

**Target Performance:**
- AUROC ≥ 0.90 for binary classification
- F1-score ≥ 0.85 for multi-class
- Per-class sensitivity ≥ 80%

**VitalDB/MIMIC-III Availability:** ✅ Extensive ECG data available

**Label Generation:**
- Automated: R-R interval analysis, QRS morphology
- Expert: Cardiologist annotations (gold standard)
- ICD codes: Diagnostic codes as weak labels

---

### 2. Hypotension Prediction

**Clinical Definition:** Predict MAP < 65 mmHg sustained for ≥60 seconds, 5-15 minutes before onset.

**Input Signals:**
- ABP waveform (primary, 100-500 Hz)
- ECG (secondary)
- PPG (tertiary)

**Benchmarks:**

| Paper | Year | Dataset | N_Patients | Prediction Window | AUROC | AUPRC | Method |
|-------|------|---------|------------|-------------------|-------|-------|--------|
| **Jo et al.** | 2022 | VitalDB | 5,230 | 5 min | 0.935 | 0.882 | ABP+EEG ResNet |
| **STEP-OP (Choe)** | 2021 | VitalDB | 18,813 | 5 min | 0.900 | 0.716 | CNN-RNN ensemble |
| **Jeong et al.** | 2024 | VitalDB | 3,200 | 10 min | 0.917 | 0.85 | Non-invasive (ECG+PPG+Cap+BIS) |
| Lee et al. | 2020 | SNUH | 1,545 | 10 min | 0.920 | - | DeepVP |
| Hatib et al. | 2018 | Single center | 1,334 | 15 min | 0.880 | - | HPI algorithm |

**Target Performance:**
- AUROC ≥ 0.90
- AUPRC ≥ 0.70 (accounts for class imbalance)
- Sensitivity ≥ 80% at 95% specificity
- Lead time: 5-15 minutes before onset

**VitalDB/MIMIC-III Availability:** ✅ Extensive ABP and hemodynamic data

**Label Generation (Consensus):**
1. Extract MAP values (2-second intervals from Solar8000/ART_MBP)
2. Identify 1-minute intervals where MAP < 65 mmHg
3. Group consecutive hypotensive minutes into episodes
4. Create prediction windows 5-15 min before each episode
5. Apply quality filtering (jSQI > 0.8)

**Critical Implementation Notes:**
- Use ≥20-minute intervals between samples to prevent temporal leakage
- Exclude periods already in hypotension
- Validate with clinical outcomes

---

### 3. Mortality Prediction (ICU/In-Hospital)

**Clinical Definition:** Predict patient mortality within 24h, 48h, 7-day, or in-hospital.

**Input Signals:**
- Multi-modal: ECG, PPG, ABP, Respiration
- Clinical: Age, SAPS II, SOFA scores
- Labs: Lactate, creatinine, WBC

**Benchmarks:**

| Paper | Year | Dataset | N | Timeframe | AUROC | Method |
|-------|------|---------|---|-----------|-------|--------|
| Purushotham et al. | 2018 | MIMIC-III | 17,903 | 48h | 0.851 | Multimodal LSTM |
| Tang et al. | 2020 | MIMIC-III | 21,139 | In-hospital | 0.871 | Time-aware LSTM |
| Harutyunyan et al. | 2019 | MIMIC-III | 17,903 | In-hospital | 0.857 | Channel-wise LSTM |
| **Target** | - | - | - | ICU | **≥0.90** | Foundation model |

**Target Performance:**
- AUROC ≥ 0.88 (current SOTA)
- AUPRC ≥ 0.45 (highly imbalanced)
- Early prediction (within 24h of admission)

**VitalDB/MIMIC-III Availability:** ✅✅ Extensive outcome data

**Label Generation:**
- Ground truth: Hospital discharge status
- Time-to-event analysis
- Censored data handling

---

### 4. Acute Kidney Injury (AKI) Prediction

**Clinical Definition:** Predict AKI stages (KDIGO criteria) within 48 hours.

**KDIGO Criteria:**
- Stage 1: SCr increase ≥0.3 mg/dL within 48h OR ≥1.5× baseline within 7 days
- Stage 2: SCr ≥2× baseline
- Stage 3: SCr ≥3× baseline OR initiation of RRT

**Benchmarks:**

| Paper | Year | Dataset | N | Prediction Window | AUROC | Method |
|-------|------|---------|---|-------------------|-------|--------|
| Rank et al. | 2020 | Mayo Clinic | 71,098 | 24-48h | 0.78-0.80 | XGBoost |
| Tomašev et al. | 2019 | VA Healthcare | 703,782 | 48h | 0.925 | Deep learning |
| **Target** | - | VitalDB/MIMIC | - | 48h | **≥0.85** | Foundation model |

**VitalDB/MIMIC-III Availability:** ✅ Lab values (creatinine) available

---

### 5. ICU Admission Prediction

**Clinical Definition:** Predict whether patient will require ICU admission from ED or ward.

**Benchmarks:**

| Paper | Year | Dataset | N | AUROC | Sensitivity@90% Spec | Method |
|-------|------|---------|---|-------|---------------------|--------|
| Shashikumar et al. | 2021 | Emory | 21,644 | 0.82 | 53% | Deep learning |
| **Target** | - | MIMIC-III | - | **≥0.85** | **≥60%** | Foundation model |

**VitalDB/MIMIC-III Availability:** ✅ MIMIC-III has ED and ward data

---

### 6. Signal Quality Assessment

**Clinical Definition:** Classify biosignal quality as acceptable (SQI ≥ 0.8) or poor (SQI < 0.8).

**Input Signals:**
- ECG: Template correlation, baseline wander, QRS detection rate
- PPG: Skewness, perfusion, pulse detection
- ABP: Physiological range, pulse pressure

**Benchmarks:**

| Paper | Year | Dataset | Signal | Accuracy | F1-Score | Method |
|-------|------|---------|--------|----------|----------|--------|
| Clifford et al. | 2012 | CinC Challenge | ECG | 91% | - | SQI algorithms |
| Elgendi et al. | 2016 | MIMIC-II | PPG | 95.8% | - | Quality indices |
| **Target** | - | VitalDB | Multi | **≥90%** | **≥0.88** | Foundation model |

**VitalDB/MIMIC-III Availability:** ✅ Real-world noisy signals preserved

**Label Generation:**
- Automated: Calculate SQI, sSQI, jSQI
- Rules: Flatline detection, saturation, physiological bounds
- Expert: Manual quality annotation (expensive)

**Target: 72% suitable signals after filtering**

---

## Regression Tasks

### 7. Blood Pressure Estimation (Cuffless)

**Clinical Definition:** Estimate SBP and DBP from PPG/ECG without cuff measurements.

**Input Signals:**
- PPG (primary, fingertip or wrist)
- ECG (for pulse arrival time)

**Clinical Standards:**

**AAMI (Association for Advancement of Medical Instrumentation):**
- Mean Error (ME) ≤ 5 mmHg
- Standard Deviation (SD) ≤ 8 mmHg
- Both criteria must be met

**BHS (British Hypertension Society) Grades:**
| Grade | Cumulative % within Error Bounds |
|-------|----------------------------------|
| A | 60% ≤ 5 mmHg, 85% ≤ 10 mmHg, 95% ≤ 15 mmHg |
| B | 50% ≤ 5 mmHg, 75% ≤ 10 mmHg, 90% ≤ 15 mmHg |
| C | 40% ≤ 5 mmHg, 65% ≤ 10 mmHg, 85% ≤ 15 mmHg |

**Benchmarks:**

| Paper | Year | Dataset | N | SBP MAE | DBP MAE | AAMI | BHS | Method |
|-------|------|---------|---|---------|---------|------|-----|--------|
| **Tóth et al.** | 2025 | MIMIC-III | - | 2.72 | 1.57 | ✅ | A | EEG→ECG/PPG transfer |
| **Tóth et al.** | 2025 | VitalDB | - | 3.14 | 1.92 | ✅ | A | CEReBrO fine-tuned |
| Hsu et al. | 2020 | MIMIC-III | 510 | 6.0±7.5 | 3.7±5.6 | ❌ | B | PPG morphology |
| Schlesinger et al. | 2020 | MIMIC-II | 1,593 | 7.8±9.2 | 4.5±6.1 | ❌ | C | ResNet |
| Paviglianiti et al. | 2022 | MIMIC-III | 2,651 | 4.12 | 2.23 | ✅ | A | ResNet-LSTM hybrid |
| **Target (Clinical)** | - | - | - | **≤5.0** | **≤5.0** | **✅** | **A** | Foundation model |

**VitalDB/MIMIC-III Availability:** ✅✅ Extensive synchronized PPG and ABP

**Evaluation Protocol:**
1. Calibration-based setting (patient-specific calibration allowed)
2. Report ME, SD, MAE, RMSE
3. Check AAMI compliance (ME ≤ 5, SD ≤ 8)
4. Calculate BHS grade
5. Bland-Altman plots
6. Per-demographic subgroup analysis

---

### 8. Heart Rate Estimation

**Clinical Definition:** Estimate instantaneous heart rate from PPG or ECG.

**Input Signals:**
- PPG (wearable, 50-125 Hz)
- ECG (clinical, 125-500 Hz)

**Benchmarks:**

| Paper | Year | Dataset | Signal | N | MAE (BPM) | RMSE | Correlation | Method |
|-------|------|---------|--------|---|-----------|------|-------------|--------|
| Biswas et al. | 2019 | TROIKA | PPG | 22 | 3.8 | 5.9 | 0.96 | JOSS + SpaMa |
| Reiss et al. | 2019 | IEEE SPC | PPG | 12 | 4.2 | - | - | Spectral analysis |
| **Baseline (autocorr)** | - | - | PPG | - | 8-10 | 12-15 | 0.85 | Traditional DSP |
| **Target** | - | VitalDB | PPG | - | **≤5.0** | **≤8.0** | **≥0.95** | Foundation model |

**Clinical Target:** ±5 BPM MAE (acceptable for consumer wearables)

**VitalDB/MIMIC-III Availability:** ✅ Both have continuous HR monitoring

**Label Generation:**
- Ground truth: R-R intervals from ECG (Pan-Tompkins)
- Validation: Compare PPG-derived HR vs ECG-derived HR
- Quality filtering: Exclude segments with poor SQI

---

### 9. Cardiac Output Estimation

**Clinical Definition:** Estimate cardiac output (CO) and stroke volume (SV) from PPG morphology.

**Input Signals:**
- PPG waveform features (pulse contour)
- ABP (if available, for calibration)
- Demographics (height, weight, age)

**Benchmarks:**

| Paper | Year | Dataset | N | Correlation (r) | PE% | Bias | Method |
|-------|------|---------|---|-----------------|-----|------|--------|
| Solà et al. | 2011 | Clinical | 21 | 0.87 | 29% | - | Pulse contour |
| Huynh et al. | 2019 | MIMIC-III | 120 | 0.85 | 32% | - | Deep learning |
| **Target** | - | VitalDB | - | **≥0.90** | **<30%** | **<20%** | Foundation model |

**Clinical Criteria:**
- Correlation ≥ 0.85
- Percentage error < 30% (Critchley-Critchley criteria)
- Bias < 20% (mean difference from reference)

**VitalDB/MIMIC-III Availability:** ⚠️ Limited gold-standard CO measurements

---

### 10. Heart Rate Variability (HRV) Metrics

**Clinical Definition:** Estimate time-domain and frequency-domain HRV metrics.

**Time-Domain Metrics:**
- SDNN: Standard deviation of NN intervals
- RMSSD: Root mean square of successive differences
- pNN50: Percentage of successive NN intervals > 50ms

**Frequency-Domain Metrics:**
- LF: Low frequency power (0.04-0.15 Hz)
- HF: High frequency power (0.15-0.4 Hz)
- LF/HF ratio: Sympathovagal balance

**Benchmarks:**

| Paper | Year | Dataset | N | SDNN MAE | RMSSD MAE | Correlation | Method |
|-------|------|---------|---|----------|-----------|-------------|--------|
| Task Force | 1996 | Standard | - | - | - | - | Guidelines |
| **Target** | - | VitalDB | - | **≤5 ms** | **≤8 ms** | **≥0.90** | Foundation model |

**VitalDB/MIMIC-III Availability:** ✅ Can derive from ECG R-R intervals

---

### 11. Anesthesia Depth (BIS Prediction)

**Clinical Definition:** Predict Bispectral Index (BIS) score from EEG (0-100, 40-60 = adequate anesthesia).

**Input Signals:**
- EEG (frontal, 128-256 Hz)
- Clinical: Propofol/Sevoflurane concentration

**Benchmarks:**

| Paper | Year | Dataset | N | MAE (BIS units) | RMSE | Method |
|-------|------|---------|---|-----------------|------|--------|
| Lee et al. | 2018 | VitalDB | 231 | 6.5 | 9.2 | CNN-LSTM |
| Yoon et al. | 2019 | SNUH | 500 | 5.8 | 8.5 | Deep learning |
| **Target** | - | VitalDB | - | **4-6** | **6-9** | Foundation model |

**Clinical Target:** MAE < 5 BIS units (adequate for monitoring)

**VitalDB Availability:** ✅✅ BIS specifically collected in VitalDB

---

## Forecasting Tasks

### 12. Vital Sign Forecasting

**Clinical Definition:** Predict future vital sign trajectories 5-15 minutes ahead.

**Forecast Targets:**
- Heart rate (next 5-15 min)
- Blood pressure (MAP, SBP, DBP)
- SpO2 trends
- Respiratory rate

**Benchmarks:**

| Paper | Year | Dataset | Signal | Horizon | MAE | Method |
|-------|------|---------|--------|---------|-----|--------|
| Temporal Fusion Transformer | 2024 | VitalDB | MAP | 5-10 min | 7 mmHg | Attention-based |
| **Target** | - | VitalDB | Multi | 5-15 min | **<10%** | Foundation model |

**VitalDB/MIMIC-III Availability:** ✅ Continuous monitoring data

**This Task Aligns with TTM's Pre-training!**

---

### 13. Signal Imputation

**Clinical Definition:** Fill missing signal segments with physiologically plausible predictions.

**Use Cases:**
- Motion artifact removal
- Sensor disconnection recovery
- Data compression

**Benchmarks:**

| Paper | Year | Dataset | Signal | MSE | SSIM | Method |
|-------|------|---------|--------|-----|------|--------|
| BRITS | 2018 | PhysioNet | Multi | - | - | Bi-directional RNN |
| **Target** | - | VitalDB | PPG/ECG | **Low** | **>0.85** | Foundation model |

**VitalDB/MIMIC-III Availability:** ✅ Has natural missing segments

---

## Segmentation Tasks

### 14. Hemodynamic Event Detection

**Clinical Definition:** Detect and segment temporal boundaries of clinical events.

**Event Types:**
- Hypotensive episodes (start, duration, end)
- Arrhythmia bursts
- Vasopressor infusion periods
- Fluid resuscitation

**Benchmarks:**

| Paper | Year | Dataset | Event | Precision | Recall | F1 | Method |
|-------|------|---------|-------|-----------|--------|----|-|
| Jo et al. | 2022 | VitalDB | Hypotension | 0.83 | 0.81 | 0.82 | Segmentation |
| **Target** | - | VitalDB | Multi | **≥0.85** | **≥0.85** | **≥0.85** | Foundation model |

**VitalDB/MIMIC-III Availability:** ✅ Can derive from clinical notes and signals

---

## Clinical Standards

### AAMI Blood Pressure Standard (ISO 81060-2:2018)

**Requirements for Cuffless BP Devices:**
- Mean Error (ME) ≤ 5 mmHg
- Standard Deviation (SD) ≤ 8 mmHg
- Validation on ≥85 subjects
- Include diverse demographics (age, BMI, BP range)
- Report per-subgroup performance

**Reference:** ANSI/AAMI/ISO 81060-2:2018

### BHS Blood Pressure Grades

| Grade | Performance Description | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg |
|-------|-------------------------|---------|----------|----------|
| A | Excellent | ≥60% | ≥85% | ≥95% |
| B | Good | ≥50% | ≥75% | ≥90% |
| C | Acceptable | ≥40% | ≥65% | ≥85% |
| D | Poor | <40% | <65% | <85% |

**Reference:** O'Brien et al., British Hypertension Society Protocol

### KDIGO AKI Criteria

**Stage 1:**
- SCr increase ≥0.3 mg/dL within 48h, OR
- SCr ≥1.5-1.9× baseline within 7 days, OR
- Urine output <0.5 mL/kg/h for 6-12h

**Stage 2:**
- SCr 2.0-2.9× baseline, OR
- Urine output <0.5 mL/kg/h for ≥12h

**Stage 3:**
- SCr ≥3× baseline, OR
- SCr increase to ≥4.0 mg/dL, OR
- Initiation of RRT, OR
- Urine output <0.3 mL/kg/h for ≥24h, OR
- Anuria for ≥12h

**Reference:** KDIGO Clinical Practice Guideline 2012

---

## Evaluation Protocols

### 1. Zero-Shot Evaluation

**Protocol:**
```python
# Freeze entire encoder
model.freeze_encoder()

# Add linear head only
head = LinearClassifier(hidden_dim, num_classes)

# Evaluate on downstream task
metrics = evaluate(model, test_data, task)
```

**Expected Performance:**
- 60-75% of full fine-tuning performance
- Validates quality of pre-trained representations

### 2. Few-Shot Evaluation

**Protocol:**
- N-way K-shot: N classes, K examples per class
- K ∈ {1, 5, 10, 50}
- Repeat 5-10 times with different support sets
- Report mean ± std

**Expected Performance:**
- K=10: 70-85% of full performance
- K=50: 85-95% of full performance

### 3. Full Fine-Tuning

**Protocol:**
- Unfreeze encoder (full or partial)
- Train on full downstream dataset
- Early stopping on validation set
- Report test performance

**Expected Performance:**
- Match or exceed SOTA benchmarks
- 5-15% improvement over training from scratch

### 4. Artifact Tolerance Curves

**Novel Evaluation Method (from SiamQuality):**

Map performance across signal quality levels from 0 (perfect) to 1.0 (all samples):

```python
for quality_threshold in [0.0, 0.1, 0.2, ..., 1.0]:
    filtered_data = data[data['sqi'] >= quality_threshold]
    metrics = evaluate(model, filtered_data)
    plot(quality_threshold, metrics['auroc'])
```

**Expected Performance:**
- Quality-aware models: 85-90% performance at quality=0.7
- Baseline models: 60-70% performance at quality=0.7

---

## Quick Reference Table

### All Tasks at a Glance

| Task | Type | Primary Signal | Target Metric | Target Value | Clinical Standard | VitalDB | MIMIC-III |
|------|------|----------------|---------------|--------------|-------------------|---------|-----------|
| Arrhythmia | Classification | ECG | AUROC | ≥0.90 | None | ✅ | ✅✅ |
| Hypotension 5min | Classification | ABP | AUROC | ≥0.90 | None | ✅✅ | ✅ |
| Hypotension 5min | Classification | ABP | AUPRC | ≥0.70 | None | ✅✅ | ✅ |
| Mortality 48h | Classification | Multi | AUROC | ≥0.88 | None | ✅ | ✅✅ |
| AKI | Classification | Clinical | AUROC | ≥0.85 | KDIGO | ⚠️ | ✅ |
| ICU Admission | Classification | Multi | AUROC | ≥0.85 | None | ❌ | ✅ |
| Signal Quality | Classification | Any | Accuracy | ≥90% | None | ✅ | ✅ |
| BP (SBP) | Regression | PPG+ECG | MAE | ≤5 mmHg | AAMI/BHS A | ✅ | ✅✅ |
| BP (DBP) | Regression | PPG+ECG | MAE | ≤5 mmHg | AAMI/BHS A | ✅ | ✅✅ |
| Heart Rate | Regression | PPG/ECG | MAE | ≤5 BPM | None | ✅ | ✅ |
| Cardiac Output | Regression | PPG+ABP | Correlation | ≥0.90 | PE<30% | ⚠️ | ⚠️ |
| HRV (SDNN) | Regression | ECG | MAE | ≤5 ms | None | ✅ | ✅ |
| Anesthesia (BIS) | Regression | EEG | MAE | 4-6 units | None | ✅✅ | ❌ |
| Vital Forecasting | Forecasting | Multi | MAE | <10% | None | ✅ | ✅ |
| Signal Imputation | Forecasting | Any | MSE | Low | None | ✅ | ✅ |
| Event Detection | Segmentation | ABP | F1 | ≥0.85 | None | ✅ | ✅ |

**Legend:**
- ✅✅ : Excellent availability, many papers
- ✅ : Good availability
- ⚠️ : Limited availability or quality
- ❌ : Not available or very limited

---

## Implementation Priority

### Phase 1: Essential Tasks (Weeks 1-4)

**Must implement for paper:**
1. ✅ **Blood Pressure Estimation** - Most standardized (AAMI/BHS)
2. ✅ **Hypotension Prediction** - Best VitalDB benchmarks
3. ✅ **Arrhythmia Detection** - Standard task for ECG
4. ✅ **Heart Rate Estimation** - Simple baseline

**Rationale:** Direct comparison with published benchmarks, clinical standards available

### Phase 2: High-Impact Tasks (Weeks 5-8)

5. ✅ **Mortality Prediction** - High clinical impact
6. ✅ **Signal Quality** - Novel contribution (AT curves)
7. ✅ **HRV Analysis** - Autonomic function assessment
8. ✅ **Vital Sign Forecasting** - Aligns with TTM pre-training!

### Phase 3: Advanced Tasks (Months 3-6)

9. ⚠️ **Cardiac Output** - Challenging, limited gold standard
10. ⚠️ **AKI Prediction** - Requires lab integration
11. ⚠️ **Anesthesia Depth** - VitalDB-specific
12. ⚠️ **Event Segmentation** - Complex temporal modeling

---

## Benchmark Comparison Strategy

### For Each Task, Report:

1. **Zero-Shot Performance**
   - Frozen encoder + linear head
   - Compare to random baseline

2. **Few-Shot Performance**
   - K={1, 5, 10, 50} examples per class
   - Show sample efficiency

3. **Full Fine-Tuning**
   - Compare to SOTA benchmarks
   - Report improvement over scratch

4. **Ablation Studies**
   - With/without pre-training
   - Different pre-training datasets (VitalDB vs MIMIC-III)
   - Encoder freezing strategies

5. **Clinical Validation**
   - Check against standards (AAMI, BHS, KDIGO)
   - Subgroup analysis (age, sex, BMI)
   - Artifact tolerance curves

---

## Expected Results Summary

### Conservative Estimates (Foundation Model)

| Task Category | Zero-Shot | Few-Shot (K=10) | Full Fine-Tune | SOTA Comparison |
|---------------|-----------|-----------------|----------------|-----------------|
| Classification | 0.70-0.75 | 0.80-0.85 | 0.88-0.92 | -2% to +3% |
| Regression | 60-70% | 80-90% | 95-100% | -5% to +10% |
| Forecasting | - | 70-80% | 90-95% | +5% to +15% |

### Optimistic Estimates (with proper pre-training)

| Task Category | Zero-Shot | Few-Shot (K=10) | Full Fine-Tune | SOTA Comparison |
|---------------|-----------|-----------------|----------------|-----------------|
| Classification | 0.75-0.80 | 0.85-0.90 | 0.90-0.94 | Match to +5% |
| Regression | 70-80% | 85-95% | 100-105% | Match to +15% |
| Forecasting | - | 80-90% | 95-100% | +10% to +20% |

---

## Data Availability Summary

### VitalDB Strengths
- ✅✅ Surgical patient data (unique)
- ✅✅ BIS/Anesthesia depth
- ✅✅ High-quality ABP (500 Hz)
- ✅✅ Synchronized multi-modal (PPG+ECG+ABP+EEG)

### VitalDB Limitations
- ❌ Single-center (Seoul National University Hospital)
- ❌ Asian population only
- ❌ Surgical setting (may not generalize to ICU)
- ⚠️ Limited outcome data (mostly intraoperative)

### MIMIC-III Strengths
- ✅✅ Multi-condition ICU patients
- ✅✅ Rich clinical outcomes (mortality, AKI, etc.)
- ✅✅ Lab values, medications, notes
- ✅✅ 6× more patients than VitalDB

### MIMIC-III Limitations
- ⚠️ Lower sampling rate (125 Hz vs 500 Hz)
- ⚠️ Single-center (Beth Israel Deaconess)
- ⚠️ Older data (2001-2012)
- ❌ No BIS/anesthesia data

### Recommendation: Use Both!
- **Pre-train on MIMIC-III** (scale, diversity)
- **Fine-tune on VitalDB** (surgery-specific, higher quality)
- **Validate on both** (generalization)

---

## Citation Format

When reporting results, use this format:

```markdown
### Blood Pressure Estimation

**Our Method:**
- SBP MAE: 4.2 ± 6.8 mmHg (BHS Grade A)
- DBP MAE: 2.8 ± 5.1 mmHg (BHS Grade A)
- AAMI Compliant: ✅ (ME < 5, SD < 8)

**Comparison to SOTA:**
| Method | Year | SBP MAE | DBP MAE | AAMI |
|--------|------|---------|---------|------|
| Ours (TTM) | 2025 | 4.2 | 2.8 | ✅ |
| Tóth et al. | 2025 | 2.72 | 1.57 | ✅ |
| Paviglianiti | 2022 | 4.12 | 2.23 | ✅ |
| Hsu et al. | 2020 | 6.0 | 3.7 | ❌ |

**Interpretation:** Competitive with current SOTA, meets clinical standards.
```

---

## References

### Key Papers for Each Task

**Hypotension:**
- Jo et al. (2022): "Prediction of Intraoperative Hypotension Using Deep Learning", PLOS ONE
- Choe et al. (2021): "STEP-OP Weighted CNN-RNN", VitalDB

**Blood Pressure:**
- Tóth et al. (2025): "EEG-based Foundation Models for ECG/PPG BP Estimation"
- Paviglianiti et al. (2022): "ResNet-LSTM Hybrid for BP Estimation"

**Arrhythmia:**
- Hannun et al. (2019): "Cardiologist-level arrhythmia detection"
- Ribeiro et al. (2020): "Automatic diagnosis of 12-lead ECGs"

**Foundation Models:**
- Apple (2024): "Large-scale Training of Foundation Models for Wearable Biosignals", ICLR
- PaPaGei (2024): "Morphology-aware Contrastive Learning for PPG", ICLR Best Paper
- SiamQuality (2024): "Quality-aware Pre-training for PPG", PMC

---

## Next Steps

1. ✅ **Review this document** and prioritize tasks
2. ✅ **Implement Phase 1 tasks** (BP, Hypotension, Arrhythmia, HR)
3. ✅ **Set up evaluation framework** (zero-shot, few-shot, full)
4. ✅ **Collect SOTA baselines** from literature
5. ✅ **Run experiments** and compare to benchmarks
6. ✅ **Write paper** with comprehensive comparisons

---

**Document Maintained By:** Foundation Model Research Team  
**Last Updated:** October 11, 2025  
**Version:** 1.0  

**For questions or updates, see:** `docs/` directory or raise an issue on GitHub.
