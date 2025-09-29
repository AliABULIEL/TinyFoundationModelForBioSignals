"""Path utilities for safe file operations."""

import os
from pathlib import Path
from typing import List, Optional, Union


def safe_join(*parts: Union[str, Path]) -> Path:
    """Safely join path components.
    
    Args:
        *parts: Path components to join.
        
    Returns:
        Joined path as Path object.
        
    Example:
        >>> safe_join('data', 'vitaldb', 'processed')
        PosixPath('data/vitaldb/processed')
    """
    # Filter out None and empty strings
    valid_parts = [str(p) for p in parts if p]
    if not valid_parts:
        return Path('.')
    
    return Path(*valid_parts).resolve()


def make_dirs(path: Union[str, Path], exist_ok: bool = True) -> Path:
    """Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create.
        exist_ok: If True, don't raise error if directory exists.
        
    Returns:
        Path object of created directory.
        
    Raises:
        FileExistsError: If exist_ok=False and directory exists.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=exist_ok)
    return path


def ensure_parent_dir(filepath: Union[str, Path]) -> Path:
    """Ensure parent directory of a file exists.
    
    Args:
        filepath: Path to file.
        
    Returns:
        Path object of the file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    return filepath


def get_project_root() -> Path:
    """Get project root directory.
    
    Looks for pyproject.toml or .git directory to identify root.
    
    Returns:
        Path to project root.
        
    Raises:
        RuntimeError: If project root cannot be found.
    """
    current = Path.cwd()
    
    for parent in [current, *current.parents]:
        if (parent / 'pyproject.toml').exists() or (parent / '.git').exists():
            return parent
    
    raise RuntimeError("Cannot find project root (no pyproject.toml or .git found)")


def list_files(
    directory: Union[str, Path],
    pattern: str = "*",
    recursive: bool = False
) -> List[Path]:
    """List files matching pattern in directory.
    
    Args:
        directory: Directory to search.
        pattern: Glob pattern for matching files.
        recursive: Whether to search recursively.
        
    Returns:
        List of matching file paths.
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    
    if recursive:
        return sorted([p for p in directory.rglob(pattern) if p.is_file()])
    else:
        return sorted([p for p in directory.glob(pattern) if p.is_file()])


def get_relative_path(
    path: Union[str, Path],
    base: Optional[Union[str, Path]] = None
) -> Path:
    """Get relative path from base directory.
    
    Args:
        path: Path to make relative.
        base: Base directory (defaults to project root).
        
    Returns:
        Relative path from base.
    """
    path = Path(path).resolve()
    
    if base is None:
        base = get_project_root()
    else:
        base = Path(base).resolve()
    
    try:
        return path.relative_to(base)
    except ValueError:
        # Path is not relative to base, return absolute
        return path


def clean_filename(name: str, replacement: str = "_") -> str:
    """Clean filename by replacing invalid characters.
    
    Args:
        name: Original filename.
        replacement: String to replace invalid characters with.
        
    Returns:
        Cleaned filename safe for filesystem.
    """
    # Characters invalid in most filesystems
    invalid_chars = '<>:"|?*/\\'
    
    cleaned = name
    for char in invalid_chars:
        cleaned = cleaned.replace(char, replacement)
    
    # Remove leading/trailing dots and spaces
    cleaned = cleaned.strip('. ')
    
    # Ensure non-empty
    if not cleaned:
        cleaned = "unnamed"
    
    return cleaned
