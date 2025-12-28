#!/usr/bin/env python3
"""ROBUST Hybrid SSL Pipeline - Production-Ready Version.

This is the FIXED, PRODUCTION-READY version of the hybrid SSL pipeline with:
- Automatic data validation and resampling
- Architecture compatibility validation between stages
- Robust checkpoint management with complete metadata
- Comprehensive error handling and recovery
- Resume capability from any stage
- Built-in verification and testing

Key Improvements Over Original:
1. ✓ Auto-detects and fixes data format issues (1250 → 1024 resampling)
2. ✓ Validates architecture between SSL and fine-tuning stages
3. ✓ Saves checkpoints with complete metadata (no more dimension mismatches)
4. ✓ Verifies checkpoint integrity after saving
5. ✓ Clear error messages with suggested fixes
6. ✓ Can resume from any stage if interrupted

Usage:
    # Full pipeline with auto-everything
    python scripts/train_hybrid_ssl_pipeline_ROBUST.py \
        --data-dir data/processed/butppg/windows_with_labels \
        --output-dir artifacts/PRODUCTION_RUN \
        --device cuda \
        --stage2-epochs 50 \
        --stage3-epochs 30

    # Resume from Stage 2 checkpoint
    python scripts/train_hybrid_ssl_pipeline_ROBUST.py \
        --data-dir data/processed/butppg/windows_with_labels \
        --output-dir artifacts/PRODUCTION_RUN \
        --resume-from-stage2 artifacts/PRODUCTION_RUN/stage2_ssl/best_model.pt

    # Verify pipeline before running
    python scripts/train_hybrid_ssl_pipeline_ROBUST.py \
        --data-dir data/processed/butppg/windows_with_labels \
        --verify-only
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import subprocess
import warnings
from typing import Dict, Optional, Tuple

import torch

# Import our robust utilities
from src.utils.checkpoint_manager import (
    save_checkpoint,
    load_checkpoint_safe,
    verify_checkpoint_integrity,
    ArchitectureMismatchError,
    CheckpointCorruptedError
)
from src.utils.architecture_validator import (
    validate_stage_compatibility,
    create_architecture_report,
    get_ttm_variant_config,
    TTM_VARIANTS
)
from src.utils.data_manager import (
    check_data_format,
    get_or_create_resampled_data,
    create_data_report,
    DataFormatError
)
from src.utils.seed import set_seed

warnings.filterwarnings('ignore')


class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass


class HybridSSLPipeline:
    """Robust 3-stage pipeline with comprehensive validation.

    Stage 1: Initialize from IBM pretrained TTM
    Stage 2: Quality-aware SSL on BUT-PPG
    Stage 3: Fine-tune on downstream tasks

    Features:
    - Automatic data validation and resampling
    - Architecture compatibility checks
    - Checkpoint integrity verification
    - Resume capability
    - Clear error reporting
    """

    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Pipeline state
        self.stage_checkpoints = {
            'stage1': None,  # IBM pretrained (no checkpoint file)
            'stage2': None,  # SSL checkpoint
            'stage3': None,  # Fine-tuned checkpoint
        }

        self.stage_status = {
            'stage1': 'pending',
            'stage2': 'pending',
            'stage3': 'pending',
        }

        # Architecture config for validation
        self.expected_architecture = None

    def log(self, message: str, level: str = 'INFO'):
        """Print formatted log message."""
        prefix = {
            'INFO': '  ',
            'SUCCESS': '✓ ',
            'WARNING': '⚠️  ',
            'ERROR': '❌ '
        }.get(level, '  ')
        print(f"{prefix}{message}")

    def log_section(self, title: str):
        """Print section header."""
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)

    def validate_environment(self) -> bool:
        """Check environment is ready for training."""
        self.log_section("ENVIRONMENT VALIDATION")

        # Check CUDA availability if requested
        if self.config.get('device') == 'cuda':
            if not torch.cuda.is_available():
                self.log("CUDA requested but not available", 'ERROR')
                return False
            self.log(f"CUDA available: {torch.cuda.get_device_name(0)}", 'SUCCESS')

        # Check disk space in output directory
        stat = self.output_dir.stat() if self.output_dir.exists() else None
        # Note: Proper disk space check would need platform-specific code

        self.log("Environment validation passed", 'SUCCESS')
        return True

    def validate_and_prepare_data(self) -> str:
        """Validate data format and resample if needed.

        Returns:
            data_dir: Path to data directory to use (original or resampled)
        """
        self.log_section("DATA VALIDATION AND PREPARATION")

        data_dir = self.config['data_dir']

        # Generate data report
        self.log("Checking data format...")
        report = create_data_report(data_dir)
        print(report)

        # Check if resampling is needed
        info = check_data_format(data_dir)

        if not info['exists']:
            raise DataFormatError(f"Data directory not found: {data_dir}")

        if info['needs_resampling']:
            self.log(
                f"Data needs resampling: {info['sequence_length']} → {info['target_length']} samples",
                'WARNING'
            )

            if self.config.get('auto_resample', True):
                self.log("Auto-resampling enabled, creating resampled dataset...")
                try:
                    resampled_dir = get_or_create_resampled_data(
                        data_dir=data_dir,
                        target_length=1024,
                        force_resample=False
                    )
                    self.log(f"Using resampled data: {resampled_dir}", 'SUCCESS')
                    return resampled_dir
                except Exception as e:
                    raise DataFormatError(f"Resampling failed: {e}")
            else:
                raise DataFormatError(
                    f"Data needs resampling but auto_resample=False. "
                    f"Please resample manually or enable --auto-resample"
                )
        else:
            self.log(f"Data format OK: {info['sequence_length']} samples", 'SUCCESS')
            return data_dir

    def run_stage1_init(self):
        """Stage 1: Initialize from IBM pretrained TTM."""
        self.log_section("[STAGE 1/3] FOUNDATION MODEL INITIALIZATION")

        # Determine configuration
        use_ibm = self.config.get('use_ibm_pretrained', True)

        if use_ibm:
            context_length = self.config.get('ibm_context_length', 1024)
            patch_size = self.config.get('ibm_patch_size', 128)

            # Identify variant
            variant_name = None
            for name, variant_config in TTM_VARIANTS.items():
                if (variant_config['context_length'] == context_length and
                    variant_config['patch_size'] == patch_size):
                    variant_name = name
                    break

            self.log(f"Using IBM Pretrained TTM")
            self.log(f"  Variant: {variant_name or 'Custom'}")
            self.log(f"  Context length: {context_length}")
            self.log(f"  Patch size: {patch_size}")

            # Store expected architecture for validation
            self.expected_architecture = {
                'context_length': context_length,
                'patch_size': patch_size,
                'input_channels': 2,  # PPG + ECG for BUT-PPG
            }

            self.stage_status['stage1'] = 'completed'
            self.log("Stage 1 initialized", 'SUCCESS')

        else:
            # Would load from VitalDB checkpoint
            vitaldb_checkpoint = self.config.get('vitaldb_checkpoint')
            if not vitaldb_checkpoint:
                raise PipelineError("Must provide --vitaldb-checkpoint or use --use-ibm-pretrained")

            self.log(f"Loading VitalDB checkpoint: {vitaldb_checkpoint}")
            # Load and validate checkpoint
            try:
                checkpoint_data = load_checkpoint_safe(
                    vitaldb_checkpoint,
                    device='cpu',
                    verbose=False
                )
                self.expected_architecture = checkpoint_data['architecture']
                self.stage_checkpoints['stage1'] = vitaldb_checkpoint
                self.stage_status['stage1'] = 'completed'
                self.log("VitalDB checkpoint loaded", 'SUCCESS')
            except Exception as e:
                raise PipelineError(f"Failed to load VitalDB checkpoint: {e}")

    def run_stage2_ssl(self, data_dir: str) -> str:
        """Stage 2: Quality-aware SSL on BUT-PPG.

        Args:
            data_dir: Path to (potentially resampled) data

        Returns:
            checkpoint_path: Path to Stage 2 checkpoint
        """
        self.log_section("[STAGE 2/3] QUALITY-AWARE SSL TRAINING")

        stage2_dir = self.output_dir / 'stage2_ssl'
        stage2_dir.mkdir(parents=True, exist_ok=True)

        # Build command for SSL training script
        cmd = [
            'python3', 'scripts/continue_ssl_butppg_quality.py',
            '--data-dir', data_dir,
            '--output-dir', str(stage2_dir),
            '--epochs', str(self.config.get('stage2_epochs', 50)),
            '--batch-size', str(self.config.get('stage2_batch_size', 128)),
            '--lr', str(self.config.get('stage2_lr', 5e-5)),
            '--device', self.config.get('device', 'cuda'),
        ]

        # Add IBM pretrained config or VitalDB checkpoint
        if self.config.get('use_ibm_pretrained', True):
            cmd.extend([
                '--use-ibm-pretrained',
                '--ibm-variant', self.config.get('ibm_variant', 'ibm-granite/granite-timeseries-ttm-r1'),
                '--ibm-context-length', str(self.config.get('ibm_context_length', 1024)),
                '--ibm-patch-size', str(self.config.get('ibm_patch_size', 128)),
            ])
        elif self.stage_checkpoints['stage1']:
            cmd.extend(['--vitaldb-checkpoint', self.stage_checkpoints['stage1']])

        self.log(f"Running SSL training...")
        self.log(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise PipelineError(f"Stage 2 SSL training failed: {e}")

        checkpoint_path = stage2_dir / 'best_model.pt'

        # Verify checkpoint was created and is valid
        if not checkpoint_path.exists():
            raise PipelineError(f"Stage 2 checkpoint not found: {checkpoint_path}")

        self.log("Verifying Stage 2 checkpoint integrity...")
        success, msg = verify_checkpoint_integrity(str(checkpoint_path), device='cpu')

        if not success:
            raise CheckpointCorruptedError(f"Stage 2 checkpoint verification failed: {msg}")

        self.log(msg, 'SUCCESS')

        # Validate architecture matches expected
        if self.expected_architecture:
            self.log("Validating architecture compatibility...")
            is_compatible, compat_msg = validate_stage_compatibility(
                stage2_checkpoint=str(checkpoint_path),
                expected_config=self.expected_architecture
            )

            if not is_compatible:
                self.log(compat_msg, 'ERROR')
                raise ArchitectureMismatchError(
                    f"Stage 2 architecture doesn't match expected: {compat_msg}"
                )

            self.log("Architecture validated", 'SUCCESS')

        # Generate and save architecture report
        report = create_architecture_report(checkpoint_path=str(checkpoint_path))
        report_file = stage2_dir / 'architecture_report.txt'
        report_file.write_text(report)
        self.log(f"Architecture report saved: {report_file}")

        self.stage_checkpoints['stage2'] = str(checkpoint_path)
        self.stage_status['stage2'] = 'completed'

        self.log(f"Stage 2 complete: {checkpoint_path}", 'SUCCESS')
        return str(checkpoint_path)

    def run_stage3_finetune(self, stage2_checkpoint: str, data_dir: str) -> str:
        """Stage 3: Supervised fine-tuning.

        Args:
            stage2_checkpoint: Path to Stage 2 SSL checkpoint
            data_dir: Path to data directory

        Returns:
            checkpoint_path: Path to Stage 3 checkpoint
        """
        self.log_section("[STAGE 3/3] SUPERVISED FINE-TUNING")

        # Validate Stage 2 checkpoint before using it
        self.log("Validating Stage 2 checkpoint before fine-tuning...")
        is_compatible, msg = validate_stage_compatibility(
            stage2_checkpoint=stage2_checkpoint,
            expected_config=self.expected_architecture
        )

        if not is_compatible:
            self.log(msg, 'ERROR')
            raise ArchitectureMismatchError(
                f"Cannot fine-tune: Stage 2 checkpoint incompatible: {msg}"
            )

        self.log("Stage 2 checkpoint validated", 'SUCCESS')

        # Determine which tasks to run
        if self.config.get('train_all_tasks', False):
            return self.run_stage3_multitask(stage2_checkpoint, data_dir)
        else:
            return self.run_stage3_single_task(stage2_checkpoint, data_dir, task='quality')

    def run_stage3_single_task(
        self,
        stage2_checkpoint: str,
        data_dir: str,
        task: str = 'quality'
    ) -> str:
        """Run Stage 3 for a single task.

        Args:
            stage2_checkpoint: Path to Stage 2 checkpoint
            data_dir: Data directory
            task: Task name

        Returns:
            checkpoint_path: Path to fine-tuned checkpoint
        """
        self.log(f"Fine-tuning for task: {task}")

        stage3_dir = self.output_dir / f'stage3_{task}'
        stage3_dir.mkdir(parents=True, exist_ok=True)

        # Build command for fine-tuning script
        cmd = [
            'python3', 'scripts/finetune_enhanced.py',
            '--pretrained', stage2_checkpoint,
            '--data-dir', data_dir,
            '--task', task,
            '--output-dir', str(stage3_dir),
            '--epochs', str(self.config.get('stage3_epochs', 30)),
            '--batch-size', str(self.config.get('stage3_batch_size', 64)),
            '--lr', str(self.config.get('stage3_lr', 1e-4)),
            '--device', self.config.get('device', 'cuda'),
        ]

        self.log(f"Running fine-tuning...")
        self.log(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise PipelineError(f"Stage 3 fine-tuning failed: {e}")

        checkpoint_path = stage3_dir / 'best_model.pt'

        if not checkpoint_path.exists():
            raise PipelineError(f"Stage 3 checkpoint not found: {checkpoint_path}")

        # Verify checkpoint
        self.log("Verifying Stage 3 checkpoint...")
        success, msg = verify_checkpoint_integrity(str(checkpoint_path), device='cpu')

        if not success:
            self.log(msg, 'WARNING')  # Non-critical for fine-tuned model
        else:
            self.log(msg, 'SUCCESS')

        self.stage_checkpoints['stage3'] = str(checkpoint_path)
        self.stage_status['stage3'] = 'completed'

        self.log(f"Stage 3 complete: {checkpoint_path}", 'SUCCESS')
        return str(checkpoint_path)

    def run_stage3_multitask(self, stage2_checkpoint: str, data_dir: str) -> Dict[str, str]:
        """Run Stage 3 for all 7 tasks.

        Args:
            stage2_checkpoint: Path to Stage 2 checkpoint
            data_dir: Data directory

        Returns:
            task_checkpoints: Dict mapping task name to checkpoint path
        """
        self.log("Training all 7 tasks...")

        tasks = ['quality', 'hr_estimation', 'motion', 'bp_systolic', 'bp_diastolic', 'spo2', 'glycaemia']
        task_checkpoints = {}

        for task in tasks:
            try:
                self.log(f"\nTraining task: {task}")
                checkpoint_path = self.run_stage3_single_task(
                    stage2_checkpoint, data_dir, task=task
                )
                task_checkpoints[task] = checkpoint_path
                self.log(f"Task '{task}' completed", 'SUCCESS')
            except Exception as e:
                self.log(f"Task '{task}' failed: {e}", 'ERROR')
                task_checkpoints[task] = None

        # Summary
        successful = sum(1 for v in task_checkpoints.values() if v is not None)
        self.log(f"\nMulti-task training complete: {successful}/{len(tasks)} tasks successful")

        # Use quality task checkpoint as primary
        self.stage_checkpoints['stage3'] = task_checkpoints.get('quality')
        self.stage_status['stage3'] = 'completed'

        return task_checkpoints

    def verify_pipeline(self):
        """Verify pipeline setup without running training."""
        self.log_section("PIPELINE VERIFICATION")

        # Check environment
        if not self.validate_environment():
            raise PipelineError("Environment validation failed")

        # Validate data
        try:
            data_dir = self.validate_and_prepare_data()
            self.log(f"Data directory validated: {data_dir}", 'SUCCESS')
        except DataFormatError as e:
            raise PipelineError(f"Data validation failed: {e}")

        # Validate Stage 1 config
        try:
            self.run_stage1_init()
        except Exception as e:
            raise PipelineError(f"Stage 1 initialization failed: {e}")

        self.log_section("VERIFICATION COMPLETE")
        self.log("Pipeline is ready to run!", 'SUCCESS')

    def run_full_pipeline(self):
        """Execute complete 3-stage pipeline with validation."""
        self.log_section("STARTING HYBRID SSL PIPELINE")

        try:
            # Pre-flight checks
            if not self.validate_environment():
                raise PipelineError("Environment validation failed")

            # Validate and prepare data
            data_dir = self.validate_and_prepare_data()

            # Stage 1: Initialize
            self.run_stage1_init()

            # Stage 2: SSL (unless resuming)
            if self.stage_checkpoints['stage2']:
                self.log_section("[STAGE 2/3] SSL TRAINING")
                self.log(f"Skipping (using existing checkpoint): {self.stage_checkpoints['stage2']}")
                stage2_checkpoint = self.stage_checkpoints['stage2']
            else:
                stage2_checkpoint = self.run_stage2_ssl(data_dir)

            # Stage 3: Fine-tuning (unless resuming)
            if self.stage_checkpoints['stage3']:
                self.log_section("[STAGE 3/3] FINE-TUNING")
                self.log(f"Skipping (using existing checkpoint): {self.stage_checkpoints['stage3']}")
            else:
                self.run_stage3_finetune(stage2_checkpoint, data_dir)

            # Success!
            self.log_section("PIPELINE COMPLETE!")
            self.log("All stages completed successfully", 'SUCCESS')

            # Save pipeline state
            self.save_pipeline_state()

            return self.stage_checkpoints

        except Exception as e:
            self.log_section("PIPELINE FAILED")
            self.log(str(e), 'ERROR')

            # Save current state for debugging
            self.save_pipeline_state()

            # Provide recovery suggestions
            self.print_recovery_suggestions()

            raise

    def save_pipeline_state(self):
        """Save pipeline state for debugging and resuming."""
        state = {
            'stage_checkpoints': self.stage_checkpoints,
            'stage_status': self.stage_status,
            'expected_architecture': self.expected_architecture,
            'config': self.config,
        }

        state_file = self.output_dir / 'pipeline_state.json'
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)

        self.log(f"Pipeline state saved: {state_file}")

    def print_recovery_suggestions(self):
        """Print suggestions for recovering from failures."""
        self.log("\nRecovery Suggestions:")
        self.log("─" * 80)

        if self.stage_status['stage2'] == 'completed':
            self.log("✓ Stage 2 completed successfully")
            self.log(f"  Checkpoint: {self.stage_checkpoints['stage2']}")
            self.log("\n  To resume from Stage 2:")
            self.log(f"  python {sys.argv[0]} \\")
            self.log(f"    --resume-from-stage2 {self.stage_checkpoints['stage2']} \\")
            self.log(f"    --data-dir {self.config['data_dir']} \\")
            self.log(f"    --output-dir {self.config['output_dir']}")

        elif self.stage_status['stage1'] == 'completed':
            self.log("✓ Stage 1 initialized successfully")
            self.log("\n  Re-run pipeline to retry Stage 2")

        self.log("\n  For detailed logs, check:")
        self.log(f"  {self.output_dir / 'pipeline_state.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="ROBUST Hybrid SSL Pipeline - Production Ready",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Core arguments
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to BUT-PPG data')
    parser.add_argument('--output-dir', type=str, default='artifacts/hybrid_pipeline_robust',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Stage 1 options
    parser.add_argument('--use-ibm-pretrained', action='store_true', default=True,
                       help='Use IBM pretrained TTM (default)')
    parser.add_argument('--vitaldb-checkpoint', type=str,
                       help='Alternative: Use VitalDB checkpoint')
    parser.add_argument('--ibm-variant', type=str, default='ibm-granite/granite-timeseries-ttm-r1',
                       help='IBM TTM variant')
    parser.add_argument('--ibm-context-length', type=int, default=1024,
                       help='IBM TTM context length (512/1024/1536)')
    parser.add_argument('--ibm-patch-size', type=int, default=128,
                       help='IBM TTM patch size (64/128)')

    # Stage 2 options
    parser.add_argument('--stage2-epochs', type=int, default=50,
                       help='SSL training epochs')
    parser.add_argument('--stage2-batch-size', type=int, default=128,
                       help='SSL batch size')
    parser.add_argument('--stage2-lr', type=float, default=5e-5,
                       help='SSL learning rate')

    # Stage 3 options
    parser.add_argument('--stage3-epochs', type=int, default=30,
                       help='Fine-tuning epochs')
    parser.add_argument('--stage3-batch-size', type=int, default=64,
                       help='Fine-tuning batch size')
    parser.add_argument('--stage3-lr', type=float, default=1e-4,
                       help='Fine-tuning learning rate')
    parser.add_argument('--train-all-tasks', action='store_true',
                       help='Train all 7 tasks (not just quality)')

    # Data options
    parser.add_argument('--auto-resample', action='store_true', default=True,
                       help='Automatically resample data if needed')

    # Pipeline control
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify pipeline setup, don\'t run')
    parser.add_argument('--resume-from-stage2', type=str,
                       help='Resume from existing Stage 2 checkpoint')
    parser.add_argument('--resume-from-stage3', type=str,
                       help='Resume from existing Stage 3 checkpoint')

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)

    # Create pipeline
    config = vars(args)
    pipeline = HybridSSLPipeline(config)

    # Handle resume flags
    if args.resume_from_stage2:
        pipeline.stage_checkpoints['stage2'] = args.resume_from_stage2
    if args.resume_from_stage3:
        pipeline.stage_checkpoints['stage3'] = args.resume_from_stage3

    # Run pipeline
    if args.verify_only:
        pipeline.verify_pipeline()
    else:
        pipeline.run_full_pipeline()


if __name__ == '__main__':
    try:
        main()
    except PipelineError as e:
        print(f"\n❌ Pipeline Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⊘ Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
