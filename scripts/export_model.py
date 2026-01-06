#!/usr/bin/env python3
"""Model export script for TTM-HAR.

Export trained models to ONNX or TorchScript format for deployment.

Usage:
    # Export to ONNX
    python scripts/export_model.py \
        --checkpoint outputs/checkpoints/best_model.pt \
        --format onnx \
        --output model.onnx

    # Export to TorchScript
    python scripts/export_model.py \
        --checkpoint outputs/checkpoints/best_model.pt \
        --format torchscript \
        --output model.pt

    # Export with quantization
    python scripts/export_model.py \
        --checkpoint outputs/checkpoints/best_model.pt \
        --format onnx \
        --output model.onnx \
        --quantize

    # Benchmark exported model
    python scripts/export_model.py \
        --checkpoint outputs/checkpoints/best_model.pt \
        --format onnx \
        --output model.onnx \
        --benchmark
"""

import argparse
import logging
from pathlib import Path

from src.models.model_factory import create_model
from src.utils.checkpointing import load_checkpoint
from src.utils.device import get_device
from src.utils.export import (
    export_to_onnx,
    export_to_torchscript,
    quantize_onnx_model,
    benchmark_onnx_model,
)
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export TTM-HAR model to ONNX or TorchScript",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export to ONNX
  python scripts/export_model.py \\
      --checkpoint outputs/checkpoints/best_model.pt \\
      --format onnx \\
      --output exports/model.onnx

  # Export to TorchScript (traced)
  python scripts/export_model.py \\
      --checkpoint outputs/checkpoints/best_model.pt \\
      --format torchscript \\
      --output exports/model.pt \\
      --torchscript_method trace

  # Export with quantization
  python scripts/export_model.py \\
      --checkpoint outputs/checkpoints/best_model.pt \\
      --format onnx \\
      --output exports/model.onnx \\
      --quantize

  # Benchmark exported model
  python scripts/export_model.py \\
      --checkpoint outputs/checkpoints/best_model.pt \\
      --format onnx \\
      --output exports/model.onnx \\
      --benchmark \\
      --num_benchmark_runs 100
        """
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["onnx", "torchscript"],
        default="onnx",
        help="Export format (default: onnx)",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for exported model",
    )

    # ONNX-specific options
    parser.add_argument(
        "--opset_version",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)",
    )

    parser.add_argument(
        "--dynamic_batch",
        action="store_true",
        help="Enable dynamic batch size for ONNX",
    )

    parser.add_argument(
        "--dynamic_sequence",
        action="store_true",
        help="Enable dynamic sequence length for ONNX",
    )

    # TorchScript-specific options
    parser.add_argument(
        "--torchscript_method",
        type=str,
        choices=["trace", "script"],
        default="trace",
        help="TorchScript export method (default: trace)",
    )

    # Input shape
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for export (default: 1)",
    )

    parser.add_argument(
        "--sequence_length",
        type=int,
        default=512,
        help="Sequence length for export (default: 512)",
    )

    parser.add_argument(
        "--num_channels",
        type=int,
        default=3,
        help="Number of input channels (default: 3)",
    )

    # Post-export options
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize exported ONNX model",
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark exported model",
    )

    parser.add_argument(
        "--num_benchmark_runs",
        type=int,
        default=100,
        help="Number of benchmark runs (default: 100)",
    )

    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


def main():
    """Main export function."""
    args = parse_args()

    # Setup logging
    setup_logging(log_level=args.log_level)

    logger.info("=" * 80)
    logger.info("TTM-HAR Model Export")
    logger.info("=" * 80)

    # Load checkpoint
    logger.info(f"Loading checkpoint from: {args.checkpoint}")

    device = get_device()
    checkpoint = load_checkpoint(args.checkpoint, device=device)

    config = checkpoint["config"]
    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")

    # Create model
    logger.info("Creating model...")
    model = create_model(config)

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded: {model.__class__.__name__}")

    # Input shape
    input_shape = (args.batch_size, args.sequence_length, args.num_channels)
    logger.info(f"Export input shape: {input_shape}")

    # Export model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "onnx":
        logger.info(f"\nExporting to ONNX format...")

        # Configure dynamic axes
        dynamic_axes = None
        if args.dynamic_batch or args.dynamic_sequence:
            dynamic_axes = {"input": {}, "output": {}}

            if args.dynamic_batch:
                dynamic_axes["input"][0] = "batch_size"
                dynamic_axes["output"][0] = "batch_size"

            if args.dynamic_sequence:
                dynamic_axes["input"][1] = "sequence_length"

        # Export to ONNX
        export_to_onnx(
            model=model,
            output_path=str(output_path),
            input_shape=input_shape,
            opset_version=args.opset_version,
            dynamic_axes=dynamic_axes,
            verify=True,
        )

        # Quantize if requested
        if args.quantize:
            logger.info(f"\nQuantizing ONNX model...")
            quantized_path = output_path.parent / f"{output_path.stem}_quantized.onnx"

            quantize_onnx_model(
                onnx_path=str(output_path),
                output_path=str(quantized_path),
                quantization_mode="dynamic",
            )

            logger.info(f"Quantized model saved to: {quantized_path}")

        # Benchmark if requested
        if args.benchmark:
            logger.info(f"\nBenchmarking ONNX model...")
            benchmark_results = benchmark_onnx_model(
                onnx_path=str(output_path),
                input_shape=input_shape,
                num_runs=args.num_benchmark_runs,
            )

            # Save benchmark results
            import json
            benchmark_path = output_path.parent / f"{output_path.stem}_benchmark.json"

            with open(benchmark_path, "w") as f:
                json.dump(benchmark_results, f, indent=2)

            logger.info(f"Benchmark results saved to: {benchmark_path}")

    elif args.format == "torchscript":
        logger.info(f"\nExporting to TorchScript format...")

        export_to_torchscript(
            model=model,
            output_path=str(output_path),
            input_shape=input_shape,
            method=args.torchscript_method,
        )

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("EXPORT SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Format:      {args.format.upper()}")
    logger.info(f"Output:      {output_path}")
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"File size:   {output_path.stat().st_size / (1024 * 1024):.2f} MB")

    if args.format == "onnx":
        logger.info(f"Opset:       {args.opset_version}")

        if args.dynamic_batch or args.dynamic_sequence:
            logger.info(f"Dynamic axes: batch={args.dynamic_batch}, sequence={args.dynamic_sequence}")

    if args.format == "torchscript":
        logger.info(f"Method:      {args.torchscript_method}")

    logger.info("=" * 80)
    logger.info("✓ Export completed successfully!")


if __name__ == "__main__":
    main()
