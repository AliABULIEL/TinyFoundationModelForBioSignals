#!/usr/bin/env python3
"""
SSL Pre-training Runner for VitalDB
====================================

Ensures proper 3-stage foundation model pipeline:
  Stage 1: IBM TTM pretrained (baseline) ✓
  Stage 2: SSL on VitalDB (THIS SCRIPT) → Foundation Model
  Stage 3: Fine-tune on BUT-PPG → Task-specific Model

This script ENFORCES:
1. Uses VitalDB data (NOT BUT-PPG)
2. Validates setup before training
3. Saves proper checkpoints for fine-tuning
4. Monitors training progress

Usage:
    # Quick test (5 epochs, small dataset)
    python scripts/run_ssl_pretraining_vitaldb.py --mode fasttrack --epochs 5

    # Full training (100 epochs, full VitalDB)
    python scripts/run_ssl_pretraining_vitaldb.py --mode full --epochs 100

Author: Claude Code Foundation Model Audit
Date: October 2025
"""

import argparse
import sys
from pathlib import Path
import subprocess

# Add project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("SSL PRE-TRAINING ON VITALDB - Foundation Model Creation")
print("=" * 80)


def check_vitaldb_data():
    """Check if VitalDB data is available."""
    print("\n[Step 1/4] Checking VitalDB data availability...")

    possible_paths = [
        Path('data/processed/vitaldb/windows'),
        Path('artifacts/raw_windows'),
        Path('data/vitaldb_windows')
    ]

    vitaldb_found = False
    vitaldb_path = None

    for path in possible_paths:
        train_dir = path / 'train'
        if train_dir.exists():
            # Check for data files
            has_data = (
                list(train_dir.glob('*.npz')) or
                list(train_dir.glob('case_*.npz')) or
                (train_dir / 'ppg').exists()
            )
            if has_data:
                vitaldb_found = True
                vitaldb_path = path
                print(f"  ✓ Found VitalDB data at: {path}")
                break

    if not vitaldb_found:
        print("\n  ❌ VitalDB data NOT found!")
        print("\n  Please prepare VitalDB data first:")
        print("    python scripts/prepare_all_data.py --dataset vitaldb")
        print("\n  Or download preprocessed VitalDB windows")
        sys.exit(1)

    return vitaldb_path


def run_validation():
    """Run validation before training."""
    print("\n[Step 2/4] Validating SSL setup...")

    validation_script = project_root / 'scripts' / 'validate_ssl_setup.py'
    config_file = project_root / 'configs' / 'ssl_pretrain.yaml'

    if not validation_script.exists():
        print(f"  ⚠️  Validation script not found: {validation_script}")
        print(f"     Skipping validation...")
        return True

    result = subprocess.run(
        ['python3', str(validation_script), '--config', str(config_file)],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("  ✓ Validation PASSED - safe to proceed!")
        return True
    else:
        print("  ❌ Validation FAILED!")
        print("\nValidation output:")
        print(result.stdout)
        print(result.stderr)
        print("\n  Fix validation errors before training!")
        sys.exit(1)


def confirm_training_params(args):
    """Confirm training parameters with user."""
    print("\n[Step 3/4] Training Configuration")
    print("-" * 80)

    print(f"  Dataset: VitalDB (hospital ICU biosignals)")
    print(f"  Mode: {args.mode}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {args.device}")
    print(f"  Output: {args.output_dir}")
    print(f"\n  Config: {args.config}")

    if args.mode == 'fasttrack':
        print(f"\n  ⚡ FastTrack mode: Uses subset of data for quick testing")
        print(f"     Expected time: ~30-60 minutes")
    else:
        print(f"\n  🔬 Full mode: Uses complete VitalDB dataset")
        print(f"     Expected time: ~4-8 hours for {args.epochs} epochs")

    print("-" * 80)

    if not args.yes:
        response = input("\nProceed with SSL pre-training? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted by user")
            sys.exit(0)


def run_ssl_pretraining(args, vitaldb_path):
    """Run SSL pre-training."""
    print("\n[Step 4/4] Running SSL Pre-training...")
    print("=" * 80)

    # Build command
    ssl_script = project_root / 'scripts' / 'pretrain_vitaldb_ssl.py'

    if not ssl_script.exists():
        print(f"  ❌ SSL training script not found: {ssl_script}")
        sys.exit(1)

    cmd = [
        'python3', str(ssl_script),
        '--config', str(args.config),
        '--mode', args.mode,
        '--epochs', str(args.epochs),
        '--batch-size', str(args.batch_size),
        '--lr', str(args.lr),
        '--device', args.device,
        '--output-dir', args.output_dir,
        '--seed', str(args.seed)
    ]

    if args.num_workers:
        cmd.extend(['--num-workers', str(args.num_workers)])

    if args.resume:
        cmd.extend(['--resume', args.resume])

    print(f"\nCommand:")
    print(f"  {' '.join(cmd)}")
    print("\n" + "=" * 80)

    # Run training
    try:
        result = subprocess.run(cmd, check=True)

        print("\n" + "=" * 80)
        print("SSL PRE-TRAINING COMPLETE!")
        print("=" * 80)

        output_path = Path(args.output_dir)
        best_model = output_path / 'best_model.pt'

        if best_model.exists():
            print(f"\n✓ Foundation model saved: {best_model}")
            print(f"\nNext steps:")
            print(f"  1. Evaluate foundation model:")
            print(f"       python scripts/evaluate_ssl_model.py \\")
            print(f"         --checkpoint {best_model}")
            print(f"\n  2. Fine-tune on BUT-PPG:")
            print(f"       python scripts/finetune_butppg.py \\")
            print(f"         --pretrained {best_model} \\")
            print(f"         --output-dir artifacts/butppg_finetuned")
        else:
            print(f"\n⚠️  Best model not found at: {best_model}")
            print(f"   Check training logs for errors")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        print(f"   Check logs above for errors")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Run SSL pre-training on VitalDB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (5 epochs, subset)
  python scripts/run_ssl_pretraining_vitaldb.py --mode fasttrack --epochs 5

  # Full training (100 epochs)
  python scripts/run_ssl_pretraining_vitaldb.py --mode full --epochs 100

  # Resume from checkpoint
  python scripts/run_ssl_pretraining_vitaldb.py --resume artifacts/ssl_vitaldb/last_model.pt
        """
    )

    parser.add_argument('--config', type=str,
                       default='configs/ssl_pretrain.yaml',
                       help='Path to SSL config file')
    parser.add_argument('--mode', type=str, choices=['fasttrack', 'full'],
                       default='fasttrack',
                       help='Training mode (fasttrack=quick test, full=complete)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--output-dir', type=str,
                       default='artifacts/ssl_vitaldb',
                       help='Output directory for checkpoints')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--resume', type=str,
                       help='Resume from checkpoint')
    parser.add_argument('--yes', '-y', action='store_true',
                       help='Skip confirmation prompt')

    args = parser.parse_args()

    try:
        # Step 1: Check VitalDB data
        vitaldb_path = check_vitaldb_data()

        # Step 2: Validate setup
        run_validation()

        # Step 3: Confirm parameters
        confirm_training_params(args)

        # Step 4: Run training
        run_ssl_pretraining(args, vitaldb_path)

    except KeyboardInterrupt:
        print("\n\nAborted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
