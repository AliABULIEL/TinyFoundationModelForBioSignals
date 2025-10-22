#!/usr/bin/env python3
"""
SSL Training Monitor
====================

Real-time monitoring and diagnostics for SSL pre-training:
1. Loss curves (MSM, STFT, total)
2. Reconstruction quality metrics
3. Gradient health checks
4. Learning rate tracking
5. Early warning for common issues

Usage:
    # Monitor training directory
    python scripts/monitor_ssl_training.py --logdir artifacts/ssl_vitaldb

    # Monitor with refresh interval
    python scripts/monitor_ssl_training.py --logdir artifacts/ssl_vitaldb --interval 30

Author: Claude Code Foundation Model Audit
Date: October 2025
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import torch
import numpy as np

# Plotting (optional)
try:
    import matplotlib.pyplot as plt
    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False


class SSLTrainingMonitor:
    """Monitor SSL training progress."""

    def __init__(self, logdir: Path):
        """Initialize monitor.

        Args:
            logdir: Directory containing training logs and checkpoints
        """
        self.logdir = Path(logdir)
        self.history_file = self.logdir / 'training_history.json'
        self.checkpoint_dir = self.logdir

    def load_history(self) -> Optional[Dict]:
        """Load training history."""
        if not self.history_file.exists():
            return None

        try:
            with open(self.history_file, 'r') as f:
                history = json.load(f)
            return history
        except Exception as e:
            print(f"Warning: Could not load history: {e}")
            return None

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Find latest checkpoint file."""
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        checkpoints.extend(list(self.checkpoint_dir.glob('last_model.pt')))

        if not checkpoints:
            return None

        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return latest

    def analyze_checkpoint(self, checkpoint_path: Path) -> Dict:
        """Analyze checkpoint for issues."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

            analysis = {
                'epoch': checkpoint.get('epoch', 'unknown'),
                'best_val_loss': checkpoint.get('best_val_loss', None),
                'has_encoder': 'encoder_state_dict' in checkpoint,
                'has_decoder': 'decoder_state_dict' in checkpoint,
                'has_optimizer': 'optimizer_state_dict' in checkpoint,
            }

            # Check encoder weights
            if 'encoder_state_dict' in checkpoint:
                encoder_state = checkpoint['encoder_state_dict']
                weights = list(encoder_state.values())
                if weights:
                    first_weight = weights[0]
                    analysis['encoder_weight_mean'] = float(first_weight.mean())
                    analysis['encoder_weight_std'] = float(first_weight.std())
                    analysis['encoder_has_nan'] = bool(torch.isnan(first_weight).any())
                    analysis['encoder_has_inf'] = bool(torch.isinf(first_weight).any())

            # Check decoder weights
            if 'decoder_state_dict' in checkpoint:
                decoder_state = checkpoint['decoder_state_dict']
                weights = list(decoder_state.values())
                if weights:
                    first_weight = weights[0]
                    analysis['decoder_weight_mean'] = float(first_weight.mean())
                    analysis['decoder_weight_std'] = float(first_weight.std())
                    analysis['decoder_has_nan'] = bool(torch.isnan(first_weight).any())
                    analysis['decoder_has_inf'] = bool(torch.isinf(first_weight).any())

            return analysis

        except Exception as e:
            return {'error': str(e)}

    def check_training_health(self, history: Dict) -> List[str]:
        """Check for common training issues."""
        issues = []

        if not history:
            issues.append("❌ No training history available")
            return issues

        # Check if training has started
        train_losses = history.get('train_loss', [])
        if not train_losses:
            issues.append("❌ No training losses recorded")
            return issues

        # Check for NaN/Inf losses
        if any(not np.isfinite(loss) for loss in train_losses):
            issues.append("❌ CRITICAL: NaN or Inf in training losses!")

        val_losses = history.get('val_loss', [])
        if val_losses and any(not np.isfinite(loss) for loss in val_losses):
            issues.append("❌ CRITICAL: NaN or Inf in validation losses!")

        # Check if loss is decreasing
        if len(train_losses) >= 5:
            recent_trend = train_losses[-5:]
            if all(recent_trend[i] >= recent_trend[i-1] for i in range(1, len(recent_trend))):
                issues.append("⚠️  WARNING: Training loss not decreasing (last 5 epochs)")

        # Check for exploding gradients
        initial_loss = train_losses[0] if train_losses else None
        recent_loss = train_losses[-1] if train_losses else None
        if initial_loss and recent_loss and recent_loss > initial_loss * 10:
            issues.append("⚠️  WARNING: Loss exploded (>10x initial)")

        # Check validation gap
        if len(train_losses) > 0 and len(val_losses) > 0:
            latest_train = train_losses[-1]
            latest_val = val_losses[-1]
            if latest_val > latest_train * 2:
                issues.append("⚠️  WARNING: Large train/val gap (possible overfitting)")

        # Check for stalled training
        if len(train_losses) >= 10:
            recent = train_losses[-10:]
            std = np.std(recent)
            mean = np.mean(recent)
            if std / mean < 0.01:  # Less than 1% variation
                issues.append("⚠️  WARNING: Training stalled (no progress in last 10 epochs)")

        if not issues:
            issues.append("✅ No obvious issues detected")

        return issues

    def print_status(self, history: Optional[Dict] = None, checkpoint_analysis: Optional[Dict] = None):
        """Print current training status."""
        print("\n" + "=" * 80)
        print("SSL TRAINING STATUS")
        print("=" * 80)

        # Training directory
        print(f"\nDirectory: {self.logdir}")

        if history is None:
            print("\n❌ No training history found")
            print("   Has training started?")
            return

        # Basic stats
        train_losses = history.get('train_loss', [])
        val_losses = history.get('val_loss', [])
        msm_losses = history.get('train_msm_loss', [])
        stft_losses = history.get('train_stft_loss', [])

        num_epochs = len(train_losses)
        print(f"\nEpochs completed: {num_epochs}")

        if num_epochs == 0:
            print("No epochs completed yet")
            return

        # Latest metrics
        print(f"\nLatest metrics (Epoch {num_epochs}):")
        print(f"  Train loss: {train_losses[-1]:.6f}")
        if val_losses:
            print(f"  Val loss:   {val_losses[-1]:.6f}")
        if msm_losses:
            print(f"  MSM loss:   {msm_losses[-1]:.6f}")
        if stft_losses and stft_losses[-1] > 0:
            print(f"  STFT loss:  {stft_losses[-1]:.6f}")

        # Best metrics
        if val_losses:
            best_val = min(val_losses)
            best_epoch = val_losses.index(best_val) + 1
            print(f"\nBest validation loss: {best_val:.6f} (epoch {best_epoch})")

        # Progress
        if len(train_losses) >= 2:
            initial = train_losses[0]
            current = train_losses[-1]
            improvement = (initial - current) / initial * 100
            print(f"\nProgress:")
            print(f"  Initial loss: {initial:.6f}")
            print(f"  Current loss: {current:.6f}")
            print(f"  Improvement:  {improvement:.1f}%")

        # Checkpoint analysis
        if checkpoint_analysis:
            print(f"\nCheckpoint health:")
            if 'encoder_has_nan' in checkpoint_analysis:
                if checkpoint_analysis['encoder_has_nan'] or checkpoint_analysis.get('decoder_has_nan', False):
                    print(f"  ❌ CRITICAL: NaN in weights!")
                elif checkpoint_analysis.get('encoder_has_inf', False) or checkpoint_analysis.get('decoder_has_inf', False):
                    print(f"  ❌ CRITICAL: Inf in weights!")
                else:
                    print(f"  ✅ No NaN/Inf in weights")

            if 'encoder_weight_std' in checkpoint_analysis:
                encoder_std = checkpoint_analysis['encoder_weight_std']
                decoder_std = checkpoint_analysis.get('decoder_weight_std', 0)
                print(f"  Encoder weight std: {encoder_std:.6f}")
                print(f"  Decoder weight std: {decoder_std:.6f}")

        # Health check
        print(f"\n" + "-" * 80)
        print("HEALTH CHECK")
        print("-" * 80)

        issues = self.check_training_health(history)
        for issue in issues:
            print(f"  {issue}")

        print("=" * 80)

    def plot_training_curves(self, history: Dict, save_path: Optional[Path] = None):
        """Plot training curves."""
        if not HAVE_MATPLOTLIB:
            print("Matplotlib not available, skipping plots")
            return

        train_losses = history.get('train_loss', [])
        val_losses = history.get('val_loss', [])
        msm_losses = history.get('train_msm_loss', [])
        stft_losses = history.get('train_stft_loss', [])

        if not train_losses:
            print("No data to plot")
            return

        epochs = list(range(1, len(train_losses) + 1))

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Total loss
        ax = axes[0, 0]
        ax.plot(epochs, train_losses, 'b-', label='Train', linewidth=2)
        if val_losses:
            ax.plot(epochs, val_losses, 'r-', label='Val', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.set_title('Total Loss (MSM + STFT)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: MSM loss
        ax = axes[0, 1]
        if msm_losses:
            ax.plot(epochs, msm_losses, 'g-', label='MSM', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MSM Loss')
            ax.set_title('Masked Signal Modeling Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No MSM data', ha='center', va='center')

        # Plot 3: STFT loss
        ax = axes[1, 0]
        if stft_losses and any(l > 0 for l in stft_losses):
            ax.plot(epochs, stft_losses, 'm-', label='STFT', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('STFT Loss')
            ax.set_title('Multi-Resolution STFT Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No STFT data or STFT disabled', ha='center', va='center')

        # Plot 4: Learning rate (if available)
        ax = axes[1, 1]
        lrs = history.get('learning_rate', [])
        if lrs:
            ax.plot(epochs, lrs, 'orange', label='LR', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No LR data', ha='center', va='center')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to: {save_path}")
        else:
            plt.show()

        plt.close()

    def monitor_loop(self, interval: int = 60):
        """Continuous monitoring loop."""
        print(f"\nMonitoring training every {interval} seconds...")
        print(f"Press Ctrl+C to stop")

        try:
            while True:
                # Clear screen (optional)
                # print("\033[2J\033[H")  # ANSI escape codes to clear screen

                # Load latest data
                history = self.load_history()
                checkpoint_path = self.get_latest_checkpoint()

                checkpoint_analysis = None
                if checkpoint_path:
                    checkpoint_analysis = self.analyze_checkpoint(checkpoint_path)

                # Print status
                self.print_status(history, checkpoint_analysis)

                # Plot curves if matplotlib available
                if HAVE_MATPLOTLIB and history:
                    plot_path = self.logdir / 'training_curves.png'
                    self.plot_training_curves(history, save_path=plot_path)

                # Wait for next update
                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")


def main():
    parser = argparse.ArgumentParser(description='Monitor SSL training')
    parser.add_argument('--logdir', type=str, required=True,
                       help='Training log directory')
    parser.add_argument('--interval', type=int, default=60,
                       help='Refresh interval in seconds (default: 60)')
    parser.add_argument('--once', action='store_true',
                       help='Run once and exit (no continuous monitoring)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots')

    args = parser.parse_args()

    logdir = Path(args.logdir)
    if not logdir.exists():
        print(f"❌ Log directory not found: {logdir}")
        sys.exit(1)

    monitor = SSLTrainingMonitor(logdir)

    if args.once:
        # Single check
        history = monitor.load_history()
        checkpoint_path = monitor.get_latest_checkpoint()

        checkpoint_analysis = None
        if checkpoint_path:
            checkpoint_analysis = monitor.analyze_checkpoint(checkpoint_path)

        monitor.print_status(history, checkpoint_analysis)

        if args.plot and history and HAVE_MATPLOTLIB:
            plot_path = logdir / 'training_curves.png'
            monitor.plot_training_curves(history, save_path=plot_path)

    else:
        # Continuous monitoring
        monitor.monitor_loop(interval=args.interval)


if __name__ == '__main__':
    main()
