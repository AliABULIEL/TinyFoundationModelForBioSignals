"""Training profiler for performance analysis."""

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class TrainingProfiler:
    """
    Profiler for tracking training performance metrics.

    Tracks:
    - Time per epoch, batch, forward pass, backward pass
    - GPU memory usage (if available)
    - Throughput (samples/sec)
    - CPU and RAM usage

    Args:
        output_dir: Directory to save profiling results
        enabled: Whether profiling is enabled

    Example:
        >>> profiler = TrainingProfiler(output_dir="outputs/profiling")
        >>> profiler.start_epoch()
        >>> with profiler.profile_step("forward"):
        >>>     outputs = model(inputs)
        >>> profiler.end_epoch()
        >>> profiler.save_results()
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        """Initialize profiler."""
        self.output_dir = Path(output_dir) if output_dir else None
        self.enabled = enabled

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Timing data
        self.epoch_times: List[float] = []
        self.batch_times: List[float] = []
        self.step_times: Dict[str, List[float]] = {}

        # Memory data
        self.gpu_memory_allocated: List[float] = []
        self.gpu_memory_reserved: List[float] = []
        self.gpu_memory_peak: List[float] = []

        # Throughput data
        self.samples_per_second: List[float] = []
        self.total_samples_processed: int = 0

        # Current timing contexts
        self._epoch_start: Optional[float] = None
        self._batch_start: Optional[float] = None
        self._step_starts: Dict[str, float] = {}

        # System monitoring (optional)
        self.cpu_percent: List[float] = []
        self.ram_percent: List[float] = []

        logger.info(f"Initialized TrainingProfiler (enabled={enabled})")

    @contextmanager
    def profile_step(self, step_name: str):
        """
        Context manager for profiling a specific step.

        Args:
            step_name: Name of the step (e.g., "forward", "backward", "optimizer")

        Example:
            >>> with profiler.profile_step("forward"):
            >>>     outputs = model(inputs)
        """
        if not self.enabled:
            yield
            return

        start_time = time.perf_counter()

        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time

            if step_name not in self.step_times:
                self.step_times[step_name] = []

            self.step_times[step_name].append(elapsed)

    def start_epoch(self) -> None:
        """Mark the start of an epoch."""
        if not self.enabled:
            return

        self._epoch_start = time.perf_counter()

        # Record memory at epoch start
        if torch.cuda.is_available():
            self.gpu_memory_allocated.append(
                torch.cuda.memory_allocated() / 1024**3  # GB
            )
            self.gpu_memory_reserved.append(
                torch.cuda.memory_reserved() / 1024**3  # GB
            )

        # Record system stats
        try:
            import psutil
            self.cpu_percent.append(psutil.cpu_percent())
            self.ram_percent.append(psutil.virtual_memory().percent)
        except ImportError:
            pass

    def end_epoch(self, num_samples: Optional[int] = None) -> Dict[str, float]:
        """
        Mark the end of an epoch and compute statistics.

        Args:
            num_samples: Number of samples processed in this epoch

        Returns:
            Dictionary of epoch statistics
        """
        if not self.enabled or self._epoch_start is None:
            return {}

        epoch_time = time.perf_counter() - self._epoch_start
        self.epoch_times.append(epoch_time)

        # Compute throughput
        if num_samples:
            throughput = num_samples / epoch_time
            self.samples_per_second.append(throughput)
            self.total_samples_processed += num_samples
        else:
            throughput = 0.0

        # Record peak memory
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            self.gpu_memory_peak.append(peak_memory)
            torch.cuda.reset_peak_memory_stats()

        stats = {
            "epoch_time": epoch_time,
            "throughput": throughput,
        }

        if torch.cuda.is_available():
            stats["gpu_memory_gb"] = torch.cuda.memory_allocated() / 1024**3

        logger.debug(
            f"Epoch profiling: {epoch_time:.2f}s, "
            f"{throughput:.0f} samples/s"
        )

        self._epoch_start = None
        return stats

    def start_batch(self) -> None:
        """Mark the start of a batch."""
        if not self.enabled:
            return

        self._batch_start = time.perf_counter()

    def end_batch(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        """
        Mark the end of a batch.

        Args:
            batch_size: Number of samples in this batch

        Returns:
            Dictionary of batch statistics
        """
        if not self.enabled or self._batch_start is None:
            return {}

        batch_time = time.perf_counter() - self._batch_start
        self.batch_times.append(batch_time)

        stats = {"batch_time": batch_time}

        if batch_size:
            stats["batch_throughput"] = batch_size / batch_time

        self._batch_start = None
        return stats

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of all profiling data.

        Returns:
            Dictionary containing summary statistics
        """
        import numpy as np

        summary = {}

        # Epoch statistics
        if self.epoch_times:
            summary["epoch_time"] = {
                "mean": np.mean(self.epoch_times),
                "std": np.std(self.epoch_times),
                "min": np.min(self.epoch_times),
                "max": np.max(self.epoch_times),
                "total": np.sum(self.epoch_times),
            }

        # Batch statistics
        if self.batch_times:
            summary["batch_time"] = {
                "mean": np.mean(self.batch_times),
                "std": np.std(self.batch_times),
                "min": np.min(self.batch_times),
                "max": np.max(self.batch_times),
            }

        # Step-wise statistics
        if self.step_times:
            summary["steps"] = {}
            for step_name, times in self.step_times.items():
                if times:
                    summary["steps"][step_name] = {
                        "mean": np.mean(times),
                        "std": np.std(times),
                        "total": np.sum(times),
                        "percentage": np.sum(times) / np.sum(self.batch_times) * 100
                        if self.batch_times
                        else 0,
                    }

        # GPU memory statistics
        if torch.cuda.is_available() and self.gpu_memory_peak:
            summary["gpu_memory_gb"] = {
                "peak": np.max(self.gpu_memory_peak),
                "mean_allocated": np.mean(self.gpu_memory_allocated),
                "mean_reserved": np.mean(self.gpu_memory_reserved),
            }

        # Throughput statistics
        if self.samples_per_second:
            summary["throughput"] = {
                "mean_samples_per_sec": np.mean(self.samples_per_second),
                "total_samples": self.total_samples_processed,
            }

        # System statistics
        if self.cpu_percent:
            summary["system"] = {
                "mean_cpu_percent": np.mean(self.cpu_percent),
                "mean_ram_percent": np.mean(self.ram_percent),
            }

        return summary

    def print_summary(self) -> None:
        """Print profiling summary to console."""
        if not self.enabled:
            logger.info("Profiling is disabled")
            return

        summary = self.get_summary()

        print("\n" + "=" * 80)
        print("TRAINING PROFILING SUMMARY")
        print("=" * 80)

        # Epoch timing
        if "epoch_time" in summary:
            epoch_stats = summary["epoch_time"]
            print(f"\nEpoch Timing:")
            print(f"  Mean:  {epoch_stats['mean']:.2f}s ± {epoch_stats['std']:.2f}s")
            print(f"  Range: {epoch_stats['min']:.2f}s - {epoch_stats['max']:.2f}s")
            print(f"  Total: {epoch_stats['total'] / 60:.1f} minutes")

        # Batch timing
        if "batch_time" in summary:
            batch_stats = summary["batch_time"]
            print(f"\nBatch Timing:")
            print(f"  Mean: {batch_stats['mean']*1000:.1f}ms ± {batch_stats['std']*1000:.1f}ms")
            print(f"  Range: {batch_stats['min']*1000:.1f}ms - {batch_stats['max']*1000:.1f}ms")

        # Step-wise timing
        if "steps" in summary:
            print(f"\nStep-wise Timing:")
            for step_name, stats in summary["steps"].items():
                print(
                    f"  {step_name:12s}: {stats['mean']*1000:6.1f}ms "
                    f"({stats['percentage']:5.1f}% of batch)"
                )

        # GPU memory
        if "gpu_memory_gb" in summary:
            mem_stats = summary["gpu_memory_gb"]
            print(f"\nGPU Memory:")
            print(f"  Peak allocated: {mem_stats['peak']:.2f} GB")
            print(f"  Mean allocated: {mem_stats['mean_allocated']:.2f} GB")
            print(f"  Mean reserved:  {mem_stats['mean_reserved']:.2f} GB")

        # Throughput
        if "throughput" in summary:
            throughput_stats = summary["throughput"]
            print(f"\nThroughput:")
            print(f"  Mean: {throughput_stats['mean_samples_per_sec']:.0f} samples/sec")
            print(f"  Total samples: {throughput_stats['total_samples']:,}")

        # System resources
        if "system" in summary:
            sys_stats = summary["system"]
            print(f"\nSystem Resources:")
            print(f"  Mean CPU: {sys_stats['mean_cpu_percent']:.1f}%")
            print(f"  Mean RAM: {sys_stats['mean_ram_percent']:.1f}%")

        print("=" * 80 + "\n")

    def save_results(self, filename: str = "profiling_results.json") -> None:
        """
        Save profiling results to file.

        Args:
            filename: Output filename
        """
        if not self.enabled or not self.output_dir:
            logger.warning("Profiling disabled or no output directory specified")
            return

        import json

        summary = self.get_summary()

        # Convert numpy types to Python types for JSON serialization
        def convert(obj):
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            return obj

        summary = convert(summary)

        output_path = self.output_dir / filename

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved profiling results to {output_path}")

    def reset(self) -> None:
        """Reset all profiling data."""
        self.epoch_times.clear()
        self.batch_times.clear()
        self.step_times.clear()
        self.gpu_memory_allocated.clear()
        self.gpu_memory_reserved.clear()
        self.gpu_memory_peak.clear()
        self.samples_per_second.clear()
        self.total_samples_processed = 0
        self.cpu_percent.clear()
        self.ram_percent.clear()

        logger.info("Reset profiler")
