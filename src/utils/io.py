"""I/O utilities for YAML and NPZ file operations."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import yaml


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML configuration file.
    
    Args:
        path: Path to YAML file.
        
    Returns:
        Dictionary containing YAML contents.
        
    Raises:
        FileNotFoundError: If file doesn't exist.
        yaml.YAMLError: If file is malformed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    
    with open(path, 'r') as f:
        try:
            return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {path}: {e}")


def save_yaml(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save dictionary to YAML file.
    
    Args:
        data: Dictionary to save.
        path: Path to save YAML file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_npz(path: Union[str, Path], allow_pickle: bool = False) -> Dict[str, np.ndarray]:
    """Load NPZ archive.
    
    Args:
        path: Path to NPZ file.
        allow_pickle: Whether to allow pickle objects.
        
    Returns:
        Dictionary mapping keys to arrays.
        
    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NPZ file not found: {path}")
    
    with np.load(path, allow_pickle=allow_pickle) as data:
        return {key: data[key] for key in data.files}


def save_npz(
    path: Union[str, Path],
    data: Optional[Dict[str, np.ndarray]] = None,
    compress: bool = True,
    **arrays: np.ndarray
) -> None:
    """Save arrays to NPZ file.
    
    Args:
        path: Path to save NPZ file.
        data: Dictionary of arrays to save (alternative to **arrays).
        compress: Whether to compress the archive.
        **arrays: Named arrays to save.
        
    Example:
        >>> save_npz('data.npz', signals=signals_array, labels=labels_array)
        >>> # or
        >>> save_npz('data.npz', {'signals': signals_array, 'labels': labels_array})
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Combine data dict and kwargs
    if data is not None:
        save_dict = data
    else:
        save_dict = arrays
    
    if compress:
        np.savez_compressed(path, **save_dict)
    else:
        np.savez(path, **save_dict)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file.
    
    Args:
        path: Path to JSON file.
        
    Returns:
        Dictionary containing JSON contents.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    with open(path, 'r') as f:
        return json.load(f)


def save_json(
    data: Any,
    path: Union[str, Path],
    indent: Optional[int] = 2
) -> None:
    """Save data to JSON file.
    
    Args:
        data: Data to save (must be JSON serializable).
        path: Path to save JSON file.
        indent: Indentation level for pretty printing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)
