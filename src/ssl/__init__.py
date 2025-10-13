"""Self-supervised learning components for biosignal foundation models.

This module implements SSL via masked signal modeling + multi-resolution STFT,
following MAE (He et al., 2022), bioFAME (ICLR'24), and HeartBEiT (Nature'23).

Key components:
- Masking strategies: random and block/contiguous masking
- Loss objectives: MSE on masked patches + multi-resolution spectral loss
"""

from .masking import random_masking, block_masking
from .objectives import MaskedSignalModeling, MultiResolutionSTFT

__all__ = [
    'random_masking',
    'block_masking',
    'MaskedSignalModeling',
    'MultiResolutionSTFT',
]
