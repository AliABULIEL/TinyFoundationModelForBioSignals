"""Label mapping definitions for various HAR datasets."""

import logging
from typing import Dict

logger = logging.getLogger(__name__)

# CAPTURE-24 5-class taxonomy (MET-based)
CAPTURE24_5CLASS: Dict[int, str] = {
    0: "Sleep",  # MET < 1.0
    1: "Sedentary",  # MET 1.0 - 1.5
    2: "Light",  # MET 1.5 - 3.0
    3: "Moderate",  # MET 3.0 - 6.0
    4: "Vigorous",  # MET >= 6.0
}

# CAPTURE-24 8-class taxonomy (more granular)
CAPTURE24_8CLASS: Dict[int, str] = {
    0: "Sleep",
    1: "Sedentary",
    2: "Light intensity",
    3: "Moderate intensity",
    4: "Vigorous intensity",
    5: "Bicycling",
    6: "Walking",
    7: "Mixed activity",
}

# Alternative datasets (for future extension)
WISDM_6CLASS: Dict[int, str] = {
    0: "Walking",
    1: "Jogging",
    2: "Stairs",
    3: "Sitting",
    4: "Standing",
    5: "LyingDown",
}

PAMAP2_12CLASS: Dict[int, str] = {
    0: "Lying",
    1: "Sitting",
    2: "Standing",
    3: "Walking",
    4: "Running",
    5: "Cycling",
    6: "Nordic walking",
    7: "Ascending stairs",
    8: "Descending stairs",
    9: "Vacuum cleaning",
    10: "Ironing",
    11: "Rope jumping",
}


def get_label_mapping(dataset_name: str, num_classes: int = 5) -> Dict[int, str]:
    """
    Get label mapping for specified dataset.

    Args:
        dataset_name: Name of dataset ("capture24", "wisdm", "pamap2")
        num_classes: Number of classes (for datasets with multiple taxonomies)

    Returns:
        Dictionary mapping class IDs to class names

    Raises:
        ValueError: If dataset or num_classes is unsupported

    Example:
        >>> mapping = get_label_mapping("capture24", num_classes=5)
        >>> mapping[0]
        'Sleep'
    """
    dataset_name = dataset_name.lower()

    if dataset_name == "capture24":
        if num_classes == 5:
            return CAPTURE24_5CLASS
        elif num_classes == 8:
            return CAPTURE24_8CLASS
        else:
            raise ValueError(
                f"CAPTURE-24 doesn't support {num_classes} classes.\n"
                f"  Supported: [5, 8]\n"
                f"  Hint: Use num_classes=5 for MET-based taxonomy"
            )

    elif dataset_name == "wisdm":
        if num_classes == 6:
            return WISDM_6CLASS
        else:
            raise ValueError(f"WISDM only supports 6 classes, got {num_classes}")

    elif dataset_name == "pamap2":
        if num_classes == 12:
            return PAMAP2_12CLASS
        else:
            raise ValueError(f"PAMAP2 only supports 12 classes, got {num_classes}")

    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}\n"
            f"  Supported datasets: ['capture24', 'wisdm', 'pamap2']\n"
            f"  Hint: Implement new dataset by adding adapter and label mapping"
        )


def validate_label_mapping(
    label_map: Dict[int, str], expected_num_classes: int
) -> None:
    """
    Validate label mapping dictionary.

    Args:
        label_map: Label mapping to validate
        expected_num_classes: Expected number of classes

    Raises:
        ValueError: If label mapping is invalid
    """
    if len(label_map) != expected_num_classes:
        raise ValueError(
            f"Label mapping has {len(label_map)} classes, "
            f"expected {expected_num_classes}"
        )

    # Check that keys are sequential starting from 0
    expected_keys = set(range(expected_num_classes))
    actual_keys = set(label_map.keys())

    if actual_keys != expected_keys:
        raise ValueError(
            f"Label mapping keys must be [0, 1, ..., {expected_num_classes-1}].\n"
            f"  Expected: {sorted(expected_keys)}\n"
            f"  Received: {sorted(actual_keys)}\n"
            f"  Hint: Remap labels to start from 0 with no gaps"
        )

    # Check for duplicate names
    names = list(label_map.values())
    if len(names) != len(set(names)):
        duplicates = {name for name in names if names.count(name) > 1}
        raise ValueError(
            f"Duplicate class names found: {duplicates}\n"
            f"  Hint: Each class must have a unique name"
        )

    logger.debug(f"Label mapping validation passed: {expected_num_classes} classes")
