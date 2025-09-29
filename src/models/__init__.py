"""Model components for TTM × VitalDB."""

from .datasets import (
    BalancedBatchSampler,
    StreamingVitalDBDataset,
    VitalDBDataset,
    collate_fn,
    create_data_loaders,
)
from .heads import (
    AttentionHead,
    LinearHead,
    MLPHead,
    PoolingHead,
    create_head,
)
from .lora import (
    LoRALayer,
    LoRALinear,
    add_lora_to_model,
    count_lora_parameters,
    get_lora_parameters,
    merge_lora_weights,
)
from .trainers import (
    DistributedTrainer,
    EarlyStopping,
    TTMTrainer,
    create_criterion,
    create_optimizer,
    create_scheduler,
)
from .ttm_adapter import (
    TTMAdapter,
    TTMForClassification,
    TTMForRegression,
)

__all__ = [
    # Datasets
    "VitalDBDataset",
    "StreamingVitalDBDataset",
    "BalancedBatchSampler",
    "create_data_loaders",
    "collate_fn",
    # Heads
    "MLPHead",
    "LinearHead",
    "AttentionHead",
    "PoolingHead",
    "create_head",
    # TTM
    "TTMAdapter",
    "TTMForClassification",
    "TTMForRegression",
    # LoRA
    "LoRALayer",
    "LoRALinear",
    "add_lora_to_model",
    "get_lora_parameters",
    "count_lora_parameters",
    "merge_lora_weights",
    # Training
    "TTMTrainer",
    "DistributedTrainer",
    "create_optimizer",
    "create_scheduler",
    "create_criterion",
    "EarlyStopping",
]
