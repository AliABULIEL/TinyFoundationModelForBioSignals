"""Seed utilities for reproducibility."""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set all random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
        deterministic: Whether to enable deterministic CUDNN operations.
                      May impact performance.
    
    Note:
        This sets seeds for:
        - python3 random
        - NumPy
        - PyTorch (CPU and CUDA)
        - CUDNN deterministic mode
        - Environment variables
    """
    # python3 random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU
    
    # Environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if deterministic:
        # CUDNN deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # PyTorch deterministic algorithms
        torch.use_deterministic_algorithms(True, warn_only=True)
        
        # Additional environment variable for CUBLAS
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def get_random_state() -> dict:
    """Get current random state from all generators.
    
    Returns:
        Dictionary containing random states from all sources.
    """
    state = {
        'python3': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()
    
    return state


def set_random_state(state: dict) -> None:
    """Restore random state for all generators.
    
    Args:
        state: Dictionary containing random states from get_random_state().
    """
    random.setstate(state['python3'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    
    if 'torch_cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def worker_init_fn(worker_id: int, base_seed: Optional[int] = None) -> None:
    """Initialize random seeds for DataLoader workers.
    
    Args:
        worker_id: Worker ID from DataLoader.
        base_seed: Base seed (if None, uses current torch seed).
        
    Note:
        Use this with torch.utils.data.DataLoader:
        DataLoader(..., worker_init_fn=worker_init_fn)
    """
    if base_seed is None:
        base_seed = torch.initial_seed() % (2**32)
    
    worker_seed = base_seed + worker_id
    
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class SeedManager:
    """Context manager for temporary seed setting."""
    
    def __init__(self, seed: int):
        """Initialize seed manager.
        
        Args:
            seed: Seed to use within context.
        """
        self.seed = seed
        self.state = None
    
    def __enter__(self):
        """Enter context and save current state."""
        self.state = get_random_state()
        set_seed(self.seed)
        return self
    
    def __exit__(self, *args):
        """Exit context and restore previous state."""
        if self.state is not None:
            set_random_state(self.state)


def generate_seeds(base_seed: int, n: int) -> list:
    """Generate list of seeds for ensemble training.
    
    Args:
        base_seed: Base seed to start from.
        n: Number of seeds to generate.
        
    Returns:
        List of seeds for ensemble members.
        
    Example:
        >>> seeds = generate_seeds(42, n=5)
        >>> for seed in seeds:
        ...     set_seed(seed)
        ...     train_model()
    """
    # Use a temporary random generator to avoid affecting global state
    rng = np.random.RandomState(base_seed)
    return [int(rng.randint(0, 2**31)) for _ in range(n)]


def check_reproducibility(func, *args, seed: int = 42, **kwargs) -> bool:
    """Check if a function produces reproducible results.
    
    Args:
        func: Function to test.
        *args: Positional arguments for func.
        seed: Seed to use for testing.
        **kwargs: Keyword arguments for func.
        
    Returns:
        True if results are identical, False otherwise.
    """
    # First run
    set_seed(seed)
    result1 = func(*args, **kwargs)
    
    # Second run with same seed
    set_seed(seed)
    result2 = func(*args, **kwargs)
    
    # Compare results
    if isinstance(result1, torch.Tensor):
        return torch.equal(result1, result2)
    elif isinstance(result1, np.ndarray):
        return np.array_equal(result1, result2)
    else:
        return result1 == result2
