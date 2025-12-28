#!/usr/bin/env python3
"""
Demonstration of Complete Pipeline Flow

Shows how:
1. Windows are created from recordings
2. Labels are connected to windows (same participant = same labels for all windows)
3. Pre-trained foundation model is loaded
4. Fine-tuning adds task-specific head
5. Evaluation aggregates windows by participant

Usage:
    python scripts/demo_pipeline_flow.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch


def demonstrate_window_label_connection():
    """Show how windows from same participant share labels"""

    print("\n" + "="*80)
    print("PART 1: WINDOW-TO-LABEL CONNECTION")
    print("="*80)

    # Load multi-task dataset
    data = np.load('data/test_multitask/multitask/train.npz')

    signals = data['signals']     # [N, 5, 1024]
    quality = data['quality']     # [N]
    hr = data['hr']               # [N]
    bp_sys = data['bp_systolic']  # [N]

    print(f"\nDataset: {signals.shape[0]} windows")
    print(f"  Signals: {signals.shape}")
    print(f"  Labels: quality, hr, bp_systolic, bp_diastolic, spo2, glycaemia")

    print("\n" + "-"*80)
    print("KEY INSIGHT: Each window = 10 seconds from a recording")
    print("-"*80)

    print("\nBUT-PPG Structure:")
    print("  Recording 100001 → 10 second signal → 1 window")
    print("  ├── Window shape: [5 channels, 1024 samples]")
    print("  ├── Quality: 1 (good)")
    print("  ├── HR: 83 BPM")
    print("  └── BP: 110/75 mmHg")

    print("\nVitalDB Structure (different!):")
    print("  Case 1 → 2 hour recording → many 10s windows")
    print("  ├── Window 1: [2 channels, 1250 samples]  → Labels: MAP at t=0")
    print("  ├── Window 2: [2 channels, 1250 samples]  → Labels: MAP at t=10s")
    print("  └── Window 3: [2 channels, 1250 samples]  → Labels: MAP at t=20s")

    print("\n" + "-"*80)
    print("Label Connection Strategies:")
    print("-"*80)

    print("\n1. BUT-PPG (Recording-level labels):")
    print("   - Each 10s recording has ONE set of labels")
    print("   - Labels are constant for entire recording")
    print("   - No sliding windows needed")

    print("\n2. VitalDB (Time-varying labels):")
    print("   - Labels change over time (e.g., blood pressure varies)")
    print("   - Each window has unique label from its timepoint")
    print("   - Sliding windows extract temporal patterns")

    # Show sample windows with labels
    print("\n" + "-"*80)
    print("Sample Windows with Labels (BUT-PPG):")
    print("-"*80)
    for i in range(min(5, len(signals))):
        signal_range = f"[{signals[i].min():.1f}, {signals[i].max():.1f}]"
        bp = f"{bp_sys[i]:.0f}/?" if bp_sys[i] > 0 else "N/A"
        print(f"  Window {i}: quality={quality[i]}, hr={hr[i]:.0f}, bp={bp}, signal_range={signal_range}")


def demonstrate_pretrain_to_finetune():
    """Show how pre-trained model is loaded for fine-tuning"""

    print("\n\n" + "="*80)
    print("PART 2: PRE-TRAINING → FINE-TUNING PIPELINE")
    print("="*80)

    print("\nPhase 1: Self-Supervised Pre-training (VitalDB)")
    print("-"*80)
    print("Input:  VitalDB PPG + ECG (2 channels, no labels)")
    print("Task:   Masked Signal Modeling (MSM)")
    print("Output: Foundation model checkpoint")
    print("        ├── Encoder: Learned representations")
    print("        └── Decoder: Reconstruction head (discarded)")

    print("\nWhat's saved in checkpoint:")
    print("  - encoder.state_dict() → TTM backbone weights")
    print("  - Input channels: 2 (PPG + ECG)")
    print("  - Context length: 1250 samples (10s @ 125Hz)")
    print("  - Patch size: ~125 (adapted by TTM)")

    print("\n" + "-"*80)
    print("Phase 2: Fine-tuning (BUT-PPG)")
    print("-"*80)
    print("Input:  BUT-PPG PPG + ECG + ACC (5 channels, WITH labels)")
    print("Task:   Quality classification / HR regression / BP regression")
    print("Output: Task-specific model")

    print("\nChannel Inflation Process:")
    print("  1. Load pretrained encoder (2 channels)")
    print("  2. Inflate to 5 channels:")
    print("     - Original 2 channels: Copy pretrained weights")
    print("     - New 3 channels (ACC): Initialize randomly")
    print("  3. Add task-specific head:")
    print("     - Quality: 2-class classifier")
    print("     - HR: Regression head (1 output)")
    print("     - BP: Regression head (2 outputs: systolic/diastolic)")

    print("\nStaged Unfreezing Strategy:")
    print("  Stage 1 (5 epochs):  Train head only, encoder frozen")
    print("  Stage 2 (25 epochs): Unfreeze last 2 encoder blocks")
    print("  Stage 3 (optional):  Full fine-tuning at low LR")

    # Check if pretrained model exists
    pretrained_path = Path('artifacts/foundation_model/best_model.pt')
    if pretrained_path.exists():
        print(f"\n✓ Found pretrained model: {pretrained_path}")
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            print(f"  Keys in checkpoint: {list(checkpoint.keys())}")
            if 'model_state_dict' in checkpoint:
                state = checkpoint['model_state_dict']
                print(f"  Model parameters: {len(state)} tensors")
                # Show first few keys
                for i, key in enumerate(list(state.keys())[:5]):
                    print(f"    {i+1}. {key}: {state[key].shape}")
        except Exception as e:
            print(f"  (Could not load: {e})")
    else:
        print(f"\n⚠️  Pretrained model not found: {pretrained_path}")
        print("     Run: python scripts/pretrain_vitaldb_ssl.py")


def demonstrate_evaluation():
    """Show how evaluation works with subject-level aggregation"""

    print("\n\n" + "="*80)
    print("PART 3: EVALUATION WITH SUBJECT-LEVEL AGGREGATION")
    print("="*80)

    print("\nWhy Subject-Level Evaluation?")
    print("-"*80)
    print("❌ WRONG: Evaluate at window level")
    print("   Problem: Windows from same participant leak into train/test")
    print("   Example:")
    print("     - Participant 001: Windows 1-10 → 8 in train, 2 in test")
    print("     - Model memorizes participant patterns")
    print("     - Inflated test accuracy (data leakage!)")

    print("\n✅ CORRECT: Evaluate at participant level")
    print("   Solution: Split by participant, not by window")
    print("   Example:")
    print("     - Train participants: 001-035")
    print("     - Val participants: 036-042")
    print("     - Test participants: 043-050")
    print("     - ALL windows from participant stay in same split")

    print("\n" + "-"*80)
    print("Evaluation Process:")
    print("-"*80)

    print("\nStep 1: Load test data (subject-level split)")
    test_data = np.load('data/test_multitask/multitask/test.npz')
    print(f"  Test set: {test_data['signals'].shape[0]} windows")
    print(f"  From: 8 participants (completely unseen during training)")

    print("\nStep 2: Run inference on all windows")
    print("  - Each window gets prediction")
    print("  - Quality: [0.2, 0.8] → class 1 (good)")
    print("  - HR: 82.5 BPM")
    print("  - BP: [118.3, 76.8]")

    print("\nStep 3: Aggregate by participant (optional)")
    print("  For BUT-PPG: Not needed (1 window = 1 participant)")
    print("  For VitalDB: Average predictions per participant")
    print("    Example: Participant 043")
    print("      - Window 1: HR=80.2")
    print("      - Window 2: HR=82.1")
    print("      - Window 3: HR=81.5")
    print("      → Participant-level prediction: HR=81.3 (mean)")

    print("\nStep 4: Compute metrics")
    print("  - Classification: AUROC, AUPRC, Accuracy")
    print("  - Regression: MAE, RMSE, Pearson correlation")
    print("  - Report participant-level performance")


def demonstrate_complete_workflow():
    """Show complete end-to-end workflow"""

    print("\n\n" + "="*80)
    print("COMPLETE WORKFLOW SUMMARY")
    print("="*80)

    workflow = """
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 1: SELF-SUPERVISED PRE-TRAINING (No labels needed)                │
├─────────────────────────────────────────────────────────────────────────┤
│ Data:   VitalDB (PPG + ECG, 2 channels)                                │
│ Task:   Masked Signal Modeling (MSM)                                   │
│ Output: Foundation model checkpoint                                    │
│                                                                         │
│ Command:                                                                │
│   python scripts/pretrain_vitaldb_ssl.py \\                           │
│     --config configs/ssl_pretrain.yaml \\                             │
│     --epochs 50                                                         │
│                                                                         │
│ Creates: artifacts/foundation_model/best_model.pt                      │
└─────────────────────────────────────────────────────────────────────────┘

                                    ↓

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 2: DATA PREPARATION (Connect windows to labels)                   │
├─────────────────────────────────────────────────────────────────────────┤
│ Process BUT-PPG with all clinical labels:                              │
│                                                                         │
│ Command:                                                                │
│   python scripts/process_butppg_clinical.py \\                        │
│     --raw-dir data/but_ppg/dataset \\                                 │
│     --annotations-dir data/but_ppg/dataset \\                         │
│     --output-dir data/processed/butppg \\                             │
│     --multitask                                                         │
│                                                                         │
│ Creates:                                                                │
│   data/processed/butppg/multitask/train.npz                            │
│   ├── signals: [2742, 5, 1024]                                         │
│   ├── quality: [2742]  (binary)                                        │
│   ├── hr: [2742]       (regression)                                    │
│   ├── bp_systolic: [2742]                                              │
│   └── ... (7 total label types)                                        │
└─────────────────────────────────────────────────────────────────────────┘

                                    ↓

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 3: FINE-TUNING (Transfer learning with labels)                    │
├─────────────────────────────────────────────────────────────────────────┤
│ Load pretrained model + inflate channels + add task head:              │
│                                                                         │
│ Command:                                                                │
│   python scripts/finetune_butppg.py \\                                │
│     --pretrained artifacts/foundation_model/best_model.pt \\          │
│     --data-dir data/processed/butppg/multitask \\                     │
│     --pretrain-channels 2 \\                                          │
│     --finetune-channels 5 \\                                          │
│     --epochs 30                                                         │
│                                                                         │
│ Process:                                                                │
│   1. Load encoder (2ch) from pretrained checkpoint                     │
│   2. Inflate to 5ch (copy weights + init new channels)                 │
│   3. Add classification head (2 classes for quality)                   │
│   4. Train with staged unfreezing                                      │
│                                                                         │
│ Creates: artifacts/but_ppg_finetuned/best_model.pt                    │
└─────────────────────────────────────────────────────────────────────────┘

                                    ↓

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 4: EVALUATION (Subject-level metrics)                             │
├─────────────────────────────────────────────────────────────────────────┤
│ Evaluate on held-out test participants:                                │
│                                                                         │
│ - Load fine-tuned model                                                 │
│ - Load test data (8 unseen participants)                                │
│ - Run inference on all windows                                          │
│ - Compute participant-level metrics                                     │
│ - Report AUROC, Accuracy, MAE, etc.                                     │
└─────────────────────────────────────────────────────────────────────────┘
"""

    print(workflow)

    print("\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)

    print("""
1. Windows → Labels Connection:
   - BUT-PPG: 1 recording = 1 window = 1 set of labels
   - VitalDB: 1 case = many windows = time-varying labels

2. Pre-training → Fine-tuning:
   - Pre-training: Learn representations WITHOUT labels
   - Fine-tuning: Transfer + adapt WITH labels
   - Channel inflation: Reuse pretrained weights, add new channels

3. Subject-Level Evaluation:
   - CRITICAL: Split by participant, not by window
   - Prevents data leakage
   - Realistic performance estimate

4. Label Availability:
   - Quality: 100% (all recordings)
   - HR: 100% (all recordings)
   - BP/SpO2/Glycaemia: 98.8% (most recordings)
   - Motion: 0% (data issue - all coded as 0)
""")


if __name__ == '__main__':
    demonstrate_window_label_connection()
    demonstrate_pretrain_to_finetune()
    demonstrate_evaluation()
    demonstrate_complete_workflow()

    print("\n" + "="*80)
    print("✅ PIPELINE DEMONSTRATION COMPLETE")
    print("="*80)
