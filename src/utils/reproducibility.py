"""Reproducibility utilities for TTM-HAR."""

import logging
import random
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - PyTorch CUDA deterministic mode

    Args:
        seed: Random seed value

    Example:
        >>> set_seed(42)
        >>> # All random operations will now be deterministic
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Set deterministic mode for CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Set random seed to {seed} for reproducibility")


def get_random_state() -> dict:
    """
    Get current random state from all random number generators.

    Returns:
        Dictionary containing random states for Python, NumPy, and PyTorch

    Example:
        >>> state = get_random_state()
        >>> # ... some random operations ...
        >>> restore_random_state(state)  # Restore to previous state
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def restore_random_state(state: dict) -> None:
    """
    Restore random state for all random number generators.

    Args:
        state: Dictionary containing random states (from get_random_state())

    Example:
        >>> state = get_random_state()
        >>> # ... some random operations ...
        >>> restore_random_state(state)
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])

    if state["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def verify_determinism(func: callable, seed: int, num_runs: int = 2) -> bool:
    """
    Verify that a function produces deterministic results.

    Runs the function multiple times with the same seed and checks if results match.

    Args:
        func: Function to test (should return a comparable value)
        seed: Seed to use for reproducibility
        num_runs: Number of times to run the function

    Returns:
        True if all runs produce identical results, False otherwise

    Example:
        >>> def random_operation():
        >>>     return torch.rand(10)
        >>> verify_determinism(random_operation, seed=42)
        True
    """
    results = []

    for _ in range(num_runs):
        set_seed(seed)
        result = func()
        results.append(result)

    # Compare all results
    first_result = results[0]
    for result in results[1:]:
        if isinstance(first_result, torch.Tensor):
            if not torch.equal(first_result, result):
                logger.warning("Determinism verification failed: tensors not equal")
                return False
        elif isinstance(first_result, np.ndarray):
            if not np.array_equal(first_result, result):
                logger.warning("Determinism verification failed: arrays not equal")
                return False
        else:
            if first_result != result:
                logger.warning("Determinism verification failed: values not equal")
                return False

    logger.info(f"Determinism verified: {num_runs} runs produced identical results")
    return True
