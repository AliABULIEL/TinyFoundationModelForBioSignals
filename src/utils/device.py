"""Device management utilities for TTM-HAR."""

import logging
from typing import Optional, Union

import torch

logger = logging.getLogger(__name__)


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Get PyTorch device for computation.

    Args:
        device: Desired device ("cpu", "cuda", "cuda:0", etc.).
                If None, automatically selects CUDA if available, else CPU.

    Returns:
        torch.device object

    Example:
        >>> device = get_device()  # Auto-select
        >>> device = get_device("cuda:0")  # Specific GPU
        >>> device = get_device("cpu")  # Force CPU
    """
    if device is not None:
        if isinstance(device, str):
            device = torch.device(device)
        logger.info(f"Using specified device: {device}")
        return device

    # Auto-select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(
            f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)} "
            f"(Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB)"
        )
    else:
        device = torch.device("cpu")
        logger.warning(
            "CUDA not available. Using CPU. Training will be significantly slower."
        )

    return device


def move_to_device(
    obj: Union[torch.Tensor, torch.nn.Module, dict, list, tuple],
    device: torch.device,
) -> Union[torch.Tensor, torch.nn.Module, dict, list, tuple]:
    """
    Move tensor, model, or nested structure to specified device.

    Handles:
    - Single tensors
    - PyTorch modules
    - Dictionaries of tensors
    - Lists/tuples of tensors
    - Nested combinations

    Args:
        obj: Object to move to device
        device: Target device

    Returns:
        Object moved to device

    Example:
        >>> device = get_device()
        >>> model = move_to_device(model, device)
        >>> batch = {"inputs": x, "labels": y}
        >>> batch = move_to_device(batch, device)
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)

    elif isinstance(obj, torch.nn.Module):
        return obj.to(device)

    elif isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}

    elif isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]

    elif isinstance(obj, tuple):
        return tuple(move_to_device(item, device) for item in obj)

    else:
        # Return as-is for non-tensor types (int, str, etc.)
        return obj


def get_device_memory_info(device: Optional[torch.device] = None) -> dict:
    """
    Get memory usage information for the specified device.

    Args:
        device: Device to query. If None, uses current CUDA device.

    Returns:
        Dictionary with memory information (in bytes)

    Example:
        >>> info = get_device_memory_info()
        >>> print(f"Allocated: {info['allocated'] / 1e9:.2f} GB")
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cpu":
        return {
            "allocated": 0,
            "reserved": 0,
            "free": 0,
            "total": 0,
        }

    device_idx = device.index if device.index is not None else 0

    return {
        "allocated": torch.cuda.memory_allocated(device_idx),
        "reserved": torch.cuda.memory_reserved(device_idx),
        "free": torch.cuda.get_device_properties(device_idx).total_memory
        - torch.cuda.memory_allocated(device_idx),
        "total": torch.cuda.get_device_properties(device_idx).total_memory,
    }


def clear_device_cache(device: Optional[torch.device] = None) -> None:
    """
    Clear device cache to free up memory.

    Args:
        device: Device to clear cache for. If None, clears all CUDA devices.

    Example:
        >>> clear_device_cache()  # Free up unused memory
    """
    if device is None or device.type == "cuda":
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache")
    else:
        logger.info("No cache to clear for CPU device")


def log_device_info(device: torch.device) -> None:
    """
    Log detailed device information.

    Args:
        device: Device to log information about

    Example:
        >>> device = get_device()
        >>> log_device_info(device)
    """
    logger.info(f"Device: {device}")

    if device.type == "cuda":
        device_idx = device.index if device.index is not None else 0
        props = torch.cuda.get_device_properties(device_idx)

        logger.info(f"  Name: {props.name}")
        logger.info(f"  Compute Capability: {props.major}.{props.minor}")
        logger.info(f"  Total Memory: {props.total_memory / 1e9:.2f} GB")
        logger.info(f"  Multi-Processors: {props.multi_processor_count}")

        mem_info = get_device_memory_info(device)
        logger.info(f"  Allocated Memory: {mem_info['allocated'] / 1e9:.2f} GB")
        logger.info(f"  Reserved Memory: {mem_info['reserved'] / 1e9:.2f} GB")
        logger.info(f"  Free Memory: {mem_info['free'] / 1e9:.2f} GB")
    else:
        logger.info("  Type: CPU")
