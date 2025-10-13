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

from .channel_utils import (
    load_pretrained_with_channel_inflate,
    unfreeze_last_n_blocks,
    verify_channel_inflation,
    get_channel_inflation_report
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
    # Channel utilities
    'load_pretrained_with_channel_inflate',
    'unfreeze_last_n_blocks',
    'verify_channel_inflation',
    'get_channel_inflation_report',
    # TTM
    'TTMAdapter',
    'create_ttm_model',
    'TTM_AVAILABLE'
]
