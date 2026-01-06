"""Model export utilities for ONNX and TorchScript."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, ...] = (1, 512, 3),
    opset_version: int = 14,
    dynamic_axes: Optional[Dict] = None,
    verify: bool = True,
) -> None:
    """
    Export PyTorch model to ONNX format.

    Args:
        model: PyTorch model to export
        output_path: Path to save ONNX model
        input_shape: Input tensor shape (batch, sequence_length, channels)
        opset_version: ONNX opset version
        dynamic_axes: Dynamic axes specification for variable input sizes
        verify: Whether to verify the exported model

    Example:
        >>> model = create_model(config)
        >>> export_to_onnx(
        >>>     model,
        >>>     "model.onnx",
        >>>     input_shape=(1, 512, 3),
        >>>     dynamic_axes={"input": {0: "batch", 1: "sequence"}}
        >>> )
    """
    try:
        import onnx
    except ImportError:
        raise ImportError(
            "ONNX export requires 'onnx' package.\n"
            "  Install with: pip install onnx onnxruntime"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Set model to eval mode
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(*input_shape)

    # Default dynamic axes if not specified
    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size"},
        }

    logger.info(f"Exporting model to ONNX format...")
    logger.info(f"  Input shape: {input_shape}")
    logger.info(f"  Opset version: {opset_version}")
    logger.info(f"  Dynamic axes: {dynamic_axes}")

    try:
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            verbose=False,
        )

        logger.info(f"✓ Successfully exported model to {output_path}")

        # Verify the model
        if verify:
            _verify_onnx_model(output_path, dummy_input, model)

        # Print model info
        onnx_model = onnx.load(str(output_path))
        model_size_mb = output_path.stat().st_size / (1024 * 1024)

        logger.info(
            f"ONNX Model Info:\n"
            f"  IR version: {onnx_model.ir_version}\n"
            f"  Producer: {onnx_model.producer_name}\n"
            f"  File size: {model_size_mb:.2f} MB"
        )

    except Exception as e:
        logger.error(f"Failed to export model to ONNX: {e}")
        raise


def _verify_onnx_model(
    onnx_path: Path,
    dummy_input: torch.Tensor,
    pytorch_model: nn.Module,
) -> None:
    """
    Verify ONNX model produces same output as PyTorch model.

    Args:
        onnx_path: Path to ONNX model
        dummy_input: Test input tensor
        pytorch_model: Original PyTorch model
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        logger.warning(
            "Skipping ONNX verification: onnxruntime not installed.\n"
            "  Install with: pip install onnxruntime"
        )
        return

    logger.info("Verifying ONNX model...")

    # Check model validity
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    logger.info("  ✓ ONNX model is valid")

    # Run inference with both models
    pytorch_model.eval()

    with torch.no_grad():
        pytorch_output = pytorch_model(dummy_input).cpu().numpy()

    # Run ONNX inference
    ort_session = ort.InferenceSession(str(onnx_path))
    onnx_output = ort_session.run(
        None,
        {"input": dummy_input.cpu().numpy()}
    )[0]

    # Compare outputs
    max_diff = abs(pytorch_output - onnx_output).max()
    mean_diff = abs(pytorch_output - onnx_output).mean()

    logger.info(f"  ✓ Output comparison:")
    logger.info(f"    Max difference:  {max_diff:.2e}")
    logger.info(f"    Mean difference: {mean_diff:.2e}")

    if max_diff > 1e-3:
        logger.warning(
            f"Large difference between PyTorch and ONNX outputs: {max_diff:.2e}\n"
            f"  This may indicate numerical instability or incompatible operators."
        )
    else:
        logger.info("  ✓ ONNX model verified successfully")


def export_to_torchscript(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, ...] = (1, 512, 3),
    method: str = "trace",
) -> None:
    """
    Export PyTorch model to TorchScript format.

    Args:
        model: PyTorch model to export
        output_path: Path to save TorchScript model
        input_shape: Input tensor shape for tracing
        method: Export method ("trace" or "script")

    Example:
        >>> model = create_model(config)
        >>> export_to_torchscript(model, "model.pt", method="trace")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Set model to eval mode
    model.eval()

    logger.info(f"Exporting model to TorchScript format (method={method})...")

    try:
        if method == "trace":
            # Trace the model
            dummy_input = torch.randn(*input_shape)

            with torch.no_grad():
                traced_model = torch.jit.trace(model, dummy_input)

            # Save traced model
            torch.jit.save(traced_model, str(output_path))

        elif method == "script":
            # Script the model
            scripted_model = torch.jit.script(model)

            # Save scripted model
            torch.jit.save(scripted_model, str(output_path))

        else:
            raise ValueError(
                f"Unknown TorchScript export method: {method}\n"
                f"  Supported: ['trace', 'script']"
            )

        logger.info(f"✓ Successfully exported model to {output_path}")

        # Print model info
        model_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"TorchScript Model Info:\n" f"  File size: {model_size_mb:.2f} MB")

    except Exception as e:
        logger.error(f"Failed to export model to TorchScript: {e}")
        raise


def quantize_onnx_model(
    onnx_path: str,
    output_path: str,
    quantization_mode: str = "dynamic",
) -> None:
    """
    Quantize ONNX model for faster inference.

    Args:
        onnx_path: Path to ONNX model
        output_path: Path to save quantized model
        quantization_mode: Quantization mode ("dynamic" or "static")

    Example:
        >>> quantize_onnx_model("model.onnx", "model_quantized.onnx")
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        raise ImportError(
            "ONNX quantization requires 'onnxruntime' package.\n"
            "  Install with: pip install onnxruntime"
        )

    logger.info(f"Quantizing ONNX model ({quantization_mode} quantization)...")

    try:
        if quantization_mode == "dynamic":
            quantize_dynamic(
                model_input=onnx_path,
                model_output=output_path,
                weight_type=QuantType.QUInt8,
            )

            # Compare file sizes
            original_size_mb = Path(onnx_path).stat().st_size / (1024 * 1024)
            quantized_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
            reduction_pct = (1 - quantized_size_mb / original_size_mb) * 100

            logger.info(
                f"✓ Successfully quantized model:\n"
                f"  Original size:  {original_size_mb:.2f} MB\n"
                f"  Quantized size: {quantized_size_mb:.2f} MB\n"
                f"  Reduction:      {reduction_pct:.1f}%"
            )

        else:
            raise ValueError(
                f"Quantization mode '{quantization_mode}' not yet implemented.\n"
                f"  Currently supported: ['dynamic']"
            )

    except Exception as e:
        logger.error(f"Failed to quantize ONNX model: {e}")
        raise


def benchmark_onnx_model(
    onnx_path: str,
    input_shape: Tuple[int, ...] = (1, 512, 3),
    num_runs: int = 100,
) -> Dict[str, float]:
    """
    Benchmark ONNX model inference speed.

    Args:
        onnx_path: Path to ONNX model
        input_shape: Input tensor shape
        num_runs: Number of inference runs

    Returns:
        Dictionary with benchmark results

    Example:
        >>> results = benchmark_onnx_model("model.onnx", num_runs=100)
        >>> print(f"Average latency: {results['latency_ms']:.2f}ms")
    """
    try:
        import onnxruntime as ort
        import numpy as np
        import time
    except ImportError:
        raise ImportError(
            "ONNX benchmarking requires 'onnxruntime' package.\n"
            "  Install with: pip install onnxruntime"
        )

    logger.info(f"Benchmarking ONNX model ({num_runs} runs)...")

    # Create session
    ort_session = ort.InferenceSession(onnx_path)

    # Create dummy input
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(10):
        ort_session.run(None, {"input": dummy_input})

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        ort_session.run(None, {"input": dummy_input})
        latencies.append(time.perf_counter() - start_time)

    # Compute statistics
    latencies = np.array(latencies) * 1000  # Convert to ms

    results = {
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_std_ms": float(np.std(latencies)),
        "latency_min_ms": float(np.min(latencies)),
        "latency_max_ms": float(np.max(latencies)),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "throughput_samples_per_sec": 1000 / float(np.mean(latencies)),
    }

    logger.info(
        f"Benchmark Results:\n"
        f"  Mean latency:   {results['latency_mean_ms']:.2f} ± {results['latency_std_ms']:.2f} ms\n"
        f"  P50 latency:    {results['latency_p50_ms']:.2f} ms\n"
        f"  P95 latency:    {results['latency_p95_ms']:.2f} ms\n"
        f"  P99 latency:    {results['latency_p99_ms']:.2f} ms\n"
        f"  Throughput:     {results['throughput_samples_per_sec']:.1f} samples/sec"
    )

    return results
