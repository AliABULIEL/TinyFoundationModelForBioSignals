"""Logging utilities for TTM-HAR."""

import logging
import sys
from pathlib import Path
from typing import Optional, Union


def setup_logging(
    level: Union[int, str] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None,
) -> None:
    """
    Set up logging configuration for the entire application.

    Args:
        level: Logging level (e.g., logging.INFO, "INFO", "DEBUG")
        log_file: Optional path to log file. If None, logs only to console
        format_string: Custom format string. If None, uses default format

    Example:
        >>> setup_logging(level="INFO", log_file="training.log")
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    if format_string is None:
        format_string = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(format_string))
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(format_string))
        root_logger.addHandler(file_handler)

        root_logger.info(f"Logging to file: {log_file}")


def get_logger(name: str, level: Optional[Union[int, str]] = None) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name (typically __name__ of the module)
        level: Optional logging level to set for this logger

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting training")
    """
    logger = logging.getLogger(name)

    if level is not None:
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(level)

    return logger


class LoggerContext:
    """
    Context manager for temporarily changing logger level.

    Example:
        >>> logger = get_logger(__name__)
        >>> with LoggerContext(logger, logging.DEBUG):
        >>>     logger.debug("This will be logged")
        >>> logger.debug("This won't be logged if original level was INFO")
    """

    def __init__(self, logger: logging.Logger, level: Union[int, str]) -> None:
        """
        Initialize logger context.

        Args:
            logger: Logger instance
            level: Temporary logging level
        """
        self.logger = logger
        self.original_level = logger.level

        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        self.temp_level = level

    def __enter__(self) -> logging.Logger:
        """Enter context and set temporary level."""
        self.logger.setLevel(self.temp_level)
        return self.logger

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: object) -> None:
        """Exit context and restore original level."""
        self.logger.setLevel(self.original_level)
