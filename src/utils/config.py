"""Configuration management for TTM-HAR."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary containing configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"  Hint: Check that the path is correct and the file exists."
        )

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(
            f"Failed to parse YAML configuration file: {config_path}\n" f"  Error: {e}"
        ) from e

    logger.info(f"Loaded configuration from {config_path}")
    return config if config is not None else {}


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.

    Later configs override earlier ones. Nested dictionaries are merged recursively.

    Args:
        *configs: Variable number of configuration dictionaries

    Returns:
        Merged configuration dictionary

    Example:
        >>> base = {"model": {"hidden_dim": 128}, "train": {"lr": 0.001}}
        >>> override = {"train": {"lr": 0.0001, "epochs": 10}}
        >>> merged = merge_configs(base, override)
        >>> merged["train"]["lr"]
        0.0001
    """
    if not configs:
        return {}

    result = {}

    for config in configs:
        if not isinstance(config, dict):
            logger.warning(f"Skipping non-dict config: {type(config)}")
            continue

        for key, value in config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                result[key] = merge_configs(result[key], value)
            else:
                # Override value
                result[key] = value

    return result


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration dictionary.

    Checks that critical parameters are present and valid.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If configuration is invalid
    """
    # Validate preprocessing config
    if "preprocessing" in config:
        preproc = config["preprocessing"]

        if "context_length" in preproc and "patch_length" in preproc:
            context_length = preproc["context_length"]
            patch_length = preproc["patch_length"]

            if context_length % patch_length != 0:
                raise ValueError(
                    f"context_length must be divisible by patch_length.\n"
                    f"  Received: context_length={context_length}, patch_length={patch_length}\n"
                    f"  Hint: Common valid combinations: (512, 16), (512, 32), (256, 16)"
                )

        if "sampling_rate_target" in preproc:
            sampling_rate = preproc["sampling_rate_target"]
            if sampling_rate <= 0:
                raise ValueError(
                    f"sampling_rate_target must be positive.\n"
                    f"  Received: {sampling_rate}\n"
                    f"  Hint: Typical values are 30 or 100 Hz"
                )

    # Validate model config
    if "model" in config:
        model_cfg = config["model"]

        if "num_classes" in model_cfg:
            num_classes = model_cfg["num_classes"]
            if num_classes <= 0:
                raise ValueError(
                    f"num_classes must be positive.\n" f"  Received: {num_classes}"
                )

    # Validate training config
    if "training" in config:
        train_cfg = config["training"]

        if "lr_head" in train_cfg and "lr_backbone" in train_cfg:
            lr_head = train_cfg["lr_head"]
            lr_backbone = train_cfg["lr_backbone"]

            # Head LR should typically be higher when backbone is trainable
            if "freeze_backbone" in train_cfg and not train_cfg["freeze_backbone"]:
                if lr_head <= lr_backbone:
                    logger.warning(
                        f"lr_head ({lr_head}) <= lr_backbone ({lr_backbone}). "
                        f"Typically lr_head should be higher when fine-tuning."
                    )

    # Validate dataset config
    if "dataset" in config:
        dataset_cfg = config["dataset"]

        if "label_map" in dataset_cfg and "num_classes" in config.get("model", {}):
            label_map = dataset_cfg["label_map"]
            num_classes = config["model"]["num_classes"]

            if len(label_map) != num_classes:
                raise ValueError(
                    f"Number of labels in label_map ({len(label_map)}) "
                    f"doesn't match num_classes ({num_classes}).\n"
                    f"  Hint: Ensure label_map and num_classes are consistent"
                )

    logger.info("Configuration validation passed")


def save_config(config: Dict[str, Any], save_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        save_path: Path where to save the configuration

    Raises:
        IOError: If file cannot be written
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(save_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
    except IOError as e:
        raise IOError(
            f"Failed to save configuration to {save_path}\n" f"  Error: {e}"
        ) from e

    logger.info(f"Saved configuration to {save_path}")


def get_nested_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get value from nested dictionary using dot-separated key path.

    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to value (e.g., "model.num_classes")
        default: Default value if key not found

    Returns:
        Value at key_path or default if not found

    Example:
        >>> config = {"model": {"num_classes": 5}}
        >>> get_nested_value(config, "model.num_classes")
        5
        >>> get_nested_value(config, "model.missing", default=10)
        10
    """
    keys = key_path.split(".")
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value


def set_nested_value(config: Dict[str, Any], key_path: str, value: Any) -> None:
    """
    Set value in nested dictionary using dot-separated key path.

    Creates intermediate dictionaries as needed.

    Args:
        config: Configuration dictionary to modify
        key_path: Dot-separated path to value (e.g., "model.num_classes")
        value: Value to set

    Example:
        >>> config = {}
        >>> set_nested_value(config, "model.num_classes", 5)
        >>> config
        {'model': {'num_classes': 5}}
    """
    keys = key_path.split(".")

    # Navigate to parent dictionary
    current = config
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        elif not isinstance(current[key], dict):
            # Overwrite non-dict value with dict
            current[key] = {}
        current = current[key]

    # Set value
    current[keys[-1]] = value


def parse_override_value(value_str: str) -> Any:
    """
    Parse value string from command-line override.

    Attempts to parse as:
    1. Boolean (true/false, True/False)
    2. Integer
    3. Float
    4. List (comma-separated)
    5. String (fallback)

    Args:
        value_str: String value to parse

    Returns:
        Parsed value

    Example:
        >>> parse_override_value("true")
        True
        >>> parse_override_value("42")
        42
        >>> parse_override_value("3.14")
        3.14
        >>> parse_override_value("1,2,3")
        [1, 2, 3]
    """
    value_str = value_str.strip()

    # Boolean
    if value_str.lower() in ("true", "false"):
        return value_str.lower() == "true"

    # Try integer
    try:
        return int(value_str)
    except ValueError:
        pass

    # Try float
    try:
        return float(value_str)
    except ValueError:
        pass

    # Try list (comma-separated)
    if "," in value_str:
        items = [parse_override_value(item.strip()) for item in value_str.split(",")]
        return items

    # String
    return value_str


def merge_config_overrides(config: Dict[str, Any], overrides: list) -> Dict[str, Any]:
    """
    Merge command-line overrides into configuration.

    Overrides should be in format "key.path=value".

    Args:
        config: Base configuration dictionary
        overrides: List of override strings

    Returns:
        Configuration with overrides applied

    Example:
        >>> config = {"training": {"epochs": 10, "lr": 0.001}}
        >>> overrides = ["training.epochs=100", "training.lr=0.0001"]
        >>> merged = merge_config_overrides(config, overrides)
        >>> merged["training"]["epochs"]
        100
    """
    # Create copy to avoid modifying original
    result = dict(config)

    for override in overrides:
        if "=" not in override:
            logger.warning(
                f"Invalid override format '{override}' (expected key=value), skipping"
            )
            continue

        key_path, value_str = override.split("=", 1)
        key_path = key_path.strip()
        value = parse_override_value(value_str)

        set_nested_value(result, key_path, value)
        logger.info(f"Override: {key_path} = {value}")

    return result
