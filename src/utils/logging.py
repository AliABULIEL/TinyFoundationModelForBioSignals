"""Logging utilities for consistent output formatting."""

import logging
import sys
from pathlib import Path
from typing import Optional, Union


# Color codes for terminal output
COLORS = {
    'DEBUG': '\033[36m',    # Cyan
    'INFO': '\033[32m',     # Green
    'WARNING': '\033[33m',  # Yellow
    'ERROR': '\033[31m',    # Red
    'CRITICAL': '\033[35m', # Magenta
    'RESET': '\033[0m'      # Reset
}


class ColoredFormatter(logging.Formatter):
    """Colored formatter for terminal output."""
    
    def __init__(self, use_color: bool = True):
        """Initialize colored formatter.
        
        Args:
            use_color: Whether to use colors in output.
        """
        super().__init__()
        self.use_color = use_color and sys.stderr.isatty()
        
        # Define format strings
        if self.use_color:
            self.formatters = {
                level: logging.Formatter(
                    f"{COLORS[level]}%(asctime)s - %(name)s - %(levelname)s{COLORS['RESET']} - %(message)s",
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                for level in COLORS if level != 'RESET'
            }
        else:
            base_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            self.formatters = {level: base_formatter for level in COLORS if level != 'RESET'}
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors.
        
        Args:
            record: Log record to format.
            
        Returns:
            Formatted log string.
        """
        formatter = self.formatters.get(record.levelname, self.formatters['INFO'])
        return formatter.format(record)


def get_logger(
    name: str,
    level: Union[str, int] = 'INFO',
    log_file: Optional[Union[str, Path]] = None,
    use_color: bool = True
) -> logging.Logger:
    """Create or get a logger with consistent formatting.
    
    Args:
        name: Logger name (usually __name__).
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional file to write logs to.
        use_color: Whether to use colored output for console.
        
    Returns:
        Configured logger instance.
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting training")
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(getattr(logging, level) if isinstance(level, str) else level)
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(ColoredFormatter(use_color=use_color))
        logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
    
    return logger


def set_global_logging_level(level: Union[str, int]) -> None:
    """Set logging level for all loggers.
    
    Args:
        level: Logging level to set globally.
    """
    level_value = getattr(logging, level) if isinstance(level, str) else level
    logging.getLogger().setLevel(level_value)
    
    # Also update existing loggers
    for logger_name in logging.Logger.manager.loggerDict:
        logging.getLogger(logger_name).setLevel(level_value)


def log_dict(logger: logging.Logger, data: dict, level: str = 'INFO') -> None:
    """Log dictionary contents in readable format.
    
    Args:
        logger: Logger instance to use.
        data: Dictionary to log.
        level: Logging level to use.
    """
    log_fn = getattr(logger, level.lower())
    
    # Format dictionary for logging
    max_key_len = max(len(str(k)) for k in data.keys()) if data else 0
    for key, value in data.items():
        log_fn(f"  {str(key).ljust(max_key_len)} : {value}")


def create_experiment_logger(
    experiment_name: str,
    output_dir: Union[str, Path] = "artifacts/logs"
) -> logging.Logger:
    """Create logger for an experiment with file output.
    
    Args:
        experiment_name: Name of the experiment.
        output_dir: Directory for log files.
        
    Returns:
        Configured logger for experiment.
    """
    output_dir = Path(output_dir)
    log_file = output_dir / f"{experiment_name}.log"
    
    return get_logger(
        name=f"exp.{experiment_name}",
        level='DEBUG',
        log_file=log_file,
        use_color=True
    )


def suppress_library_logs(libraries: list = None) -> None:
    """Suppress verbose logging from libraries.
    
    Args:
        libraries: List of library names to suppress.
                  Defaults to common verbose libraries.
    """
    if libraries is None:
        libraries = [
            'matplotlib',
            'PIL',
            'urllib3',
            'requests',
            'transformers',
            'torch.nn.parallel.distributed'
        ]
    
    for lib in libraries:
        logging.getLogger(lib).setLevel(logging.WARNING)
