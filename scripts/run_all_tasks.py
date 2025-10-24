#!/usr/bin/env python3
"""Run all 7 BUT-PPG tasks using the enhanced fine-tuner.

This script automatically trains models for all clinical tasks:
1. quality (classification)
2. hr_estimation (regression)
3. motion (classification)
4. bp_systolic (regression)
5. bp_diastolic (regression)
6. spo2 (regression)
7. glycaemia (regression)

Usage:
    python scripts/run_all_tasks.py \
        --pretrained artifacts/hybrid_full_corrected/stage2_butppg_quality_ssl/best_model.pt \
        --data-dir data/processed/butppg/windows_with_labels \
        --output-dir artifacts/all_tasks_results \
        --epochs 30 \
        --device cuda
"""

import subprocess
import argparse
import torch
import os
from pathlib import Path
from datetime import datetime

# All 7 BUT-PPG tasks
TASKS = [
    'quality',        # PPG quality classification (2 classes)
    'hr_estimation',  # Heart rate estimation
    'motion',         # Motion classification (8 classes)
    'bp_systolic',    # Systolic BP estimation
    'bp_diastolic',   # Diastolic BP estimation
    'spo2',           # SpO2 estimation
    'glycaemia',      # Blood glucose estimation
]


def run_task(task: str, args: argparse.Namespace) -> bool:
    """Run fine-tuning for a single task.

    Args:
        task: Task name
        args: Command-line arguments

    Returns:
        True if successful, False if failed
    """
    print("\n" + "=" * 80)
    print(f"TASK {TASKS.index(task)+1}/7: {task.upper()}")
    print("=" * 80)

    # Create task-specific output directory
    task_output_dir = Path(args.output_dir) / f'task_{task}'

    # Build command
    cmd = [
        'python3', 'scripts/finetune_enhanced.py',
        '--pretrained', args.pretrained,
        '--data-dir', args.data_dir,
        '--task', task,
        '--output-dir', str(task_output_dir),
        '--epochs', str(args.epochs),
        '--batch-size', str(args.batch_size),
        '--lr', str(args.lr),
        '--device', args.device,
    ]

    # Add optional flags
    if args.no_adaptation:
        cmd.append('--no-adaptation')
    elif args.adaptation:
        cmd.extend(['--adaptation', args.adaptation])

    print(f"\nRunning: {' '.join(cmd)}\n")

    try:
        # Run fine-tuning
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)  # Print stdout for visibility
        print(f"\n✓ Task '{task}' completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Task '{task}' failed with error: {e}")

        # Print error details
        if e.stderr:
            print(f"\nError output:\n{e.stderr}")

        # Check for architecture mismatch errors
        error_text = str(e.stderr) if e.stderr else ""
        if 'Expected size' in error_text or 'dimension' in error_text.lower():
            print("\n" + "="*80)
            print("⚠️  ARCHITECTURE MISMATCH DETECTED")
            print("="*80)
            print("This usually means the checkpoint format is incompatible.")
            print("\nPossible solutions:")
            print("  1. The checkpoint should have been auto-converted above")
            print("  2. Check if converted_encoder.pt was created successfully")
            print("  3. Try using --skip-ssl flag with finetune_butppg.py for simpler architecture")
            print("  4. Train quality task with finetune_enhanced.py directly:")
            print("     python3 scripts/finetune_enhanced.py \\")
            print("         --pretrained <SSL_checkpoint> \\")
            print("         --task quality \\")
            print("         --data-dir <data_dir> \\")
            print("         --output-dir <output_dir>")
            print("="*80 + "\n")

        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run all 7 BUT-PPG tasks with enhanced fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--pretrained', type=str, required=True,
                       help='Path to SSL pretrained checkpoint')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing BUT-PPG windowed data')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for all task results')

    # Training configuration
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs per task')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')

    # Domain adaptation
    parser.add_argument('--adaptation', type=str,
                       choices=['projection', 'adversarial', 'none'],
                       help='Domain adaptation method')
    parser.add_argument('--no-adaptation', action='store_true',
                       help='Disable domain adaptation (baseline)')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')

    # Task selection
    parser.add_argument('--tasks', type=str, nargs='+',
                       choices=TASKS,
                       help='Specific tasks to run (default: all 7 tasks)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select tasks to run
    tasks_to_run = args.tasks if args.tasks else TASKS

    # Print configuration
    print("\n" + "=" * 80)
    print("MULTI-TASK FINE-TUNING")
    print("=" * 80)
    print(f"Pretrained checkpoint: {args.pretrained}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Tasks ({len(tasks_to_run)}): {', '.join(tasks_to_run)}")
    print(f"Epochs per task: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Domain adaptation: {args.adaptation if not args.no_adaptation else 'None'}")
    print(f"Device: {args.device}")
    print("=" * 80)

    # ======================================================================
    # CHECKPOINT CONVERSION FOR MULTI-TASK COMPATIBILITY
    # ======================================================================
    print("\n" + "=" * 80)
    print("CHECKPOINT CONVERSION")
    print("=" * 80)

    # Check if checkpoint needs conversion
    original_checkpoint = args.pretrained
    converted_checkpoint = None

    # Check if this is a multi-scale checkpoint (from finetune_butppg.py)
    try:
        print(f"Checking checkpoint format: {original_checkpoint}")
        checkpoint = torch.load(original_checkpoint, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))

        # Check for multi-scale components
        has_multi_scale = any('pooling' in key or 'fusion' in key for key in state_dict.keys())

        if has_multi_scale:
            print("✓ Detected multi-scale checkpoint (from finetune_butppg.py)")
            print("  Converting to simple encoder format for multi-task evaluation...")

            # Convert checkpoint
            converted_path = os.path.join(args.output_dir, 'converted_encoder.pt')

            conversion_cmd = [
                'python3', 'scripts/convert_checkpoint_for_multitask.py',
                '--input', original_checkpoint,
                '--output', converted_path,
                '--quiet'
            ]

            result = subprocess.run(conversion_cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"✓ Converted checkpoint saved to: {converted_path}")
                converted_checkpoint = converted_path
                args.pretrained = converted_checkpoint
                print(f"✓ Using converted checkpoint for all tasks")
            else:
                print(f"⚠️  Conversion failed: {result.stderr}")
                print("  Attempting to use original checkpoint...")
        else:
            print("✓ Checkpoint is already in simple encoder format")

    except Exception as e:
        print(f"⚠️  Could not check checkpoint format: {e}")
        print("  Attempting to use original checkpoint...")

    print("=" * 80)

    # ======================================================================
    # MULTI-TASK FINE-TUNING
    # ======================================================================

    # Run all tasks
    start_time = datetime.now()
    results = {}

    for task in tasks_to_run:
        success = run_task(task, args)
        results[task] = 'SUCCESS' if success else 'FAILED'

    end_time = datetime.now()
    duration = end_time - start_time

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total time: {duration}")
    print(f"\nResults:")
    for task, status in results.items():
        symbol = "✓" if status == "SUCCESS" else "✗"
        print(f"  {symbol} {task}: {status}")

    # Count successes
    successful = sum(1 for status in results.values() if status == "SUCCESS")
    total = len(results)

    print(f"\nCompleted: {successful}/{total} tasks")
    print("=" * 80)

    # Exit with error if any task failed
    if successful < total:
        exit(1)


if __name__ == "__main__":
    main()
