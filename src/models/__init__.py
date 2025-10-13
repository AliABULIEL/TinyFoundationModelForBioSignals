"""Models module for TTM integration."""

from .datasets import (
    RawWindowDataset,
    StreamingWindowDataset,
    MultiModalDataset,
    create_dataloader,
    custom_collate_fn
)

from .heads import (
    LinearClassifier,
    LinearRegressor,
    MLPClassifier,
    MLPRegressor,
    AttentionPooling,
    SequenceClassifier
)

from .decoders import (
    ReconstructionHead1D,
    create_reconstruction_head
)

from .lora import (
    LoRALinear,
    LoRAConfig,
    apply_lora,
    get_lora_parameters,
    freeze_non_lora_parameters,
    mark_lora_parameters,
    print_lora_summary
)

# Conditional import for TTM
try:
    from .ttm_adapter import (
        TTMAdapter,
        create_ttm_model
    )
    TTM_AVAILABLE = True
except ImportError:
    TTM_AVAILABLE = False
    TTMAdapter = None
    create_ttm_model = None

__all__ = [
    # Datasets
    'RawWindowDataset',
    'StreamingWindowDataset',
    'MultiModalDataset',
    'create_dataloader',
    'custom_collate_fn',
    # Heads
    'LinearClassifier',
    'LinearRegressor',
    'MLPClassifier',
    'MLPRegressor',
    'AttentionPooling',
    'SequenceClassifier',
    # Decoders
    'ReconstructionHead1D',
    'create_reconstruction_head',
    # LoRA
    'LoRALinear',
    'LoRAConfig',
    'apply_lora',
    'get_lora_parameters',
    'freeze_non_lora_parameters',
    'mark_lora_parameters',
    'print_lora_summary',
    # TTM
    'TTMAdapter',
    'create_ttm_model',
    'TTM_AVAILABLE'
]
