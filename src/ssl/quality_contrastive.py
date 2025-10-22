"""Quality-Aware Contrastive Learning for Biosignal Foundation Models.

This module implements contrastive learning that leverages signal quality as
supervision signal, helping the model learn quality-relevant features without
requiring manual labels during pre-training.

Key idea: Pull together samples with similar quality, push apart samples with
different quality. This bridges the domain gap between VitalDB (hospital ICU)
and BUT-PPG (smartphone) datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class QualityContrastiveLoss(nn.Module):
    """Contrastive loss that uses signal quality as supervision.

    Creates positive pairs from samples with similar quality scores and
    negative pairs from samples with different quality. Uses InfoNCE-style
    contrastive learning to learn quality-relevant features.

    This helps bridge domain gaps (e.g., VitalDB → BUT-PPG) by learning
    features that are directly relevant to signal quality assessment.

    Args:
        temperature: Softmax temperature for scaling similarities (default: 0.07)
        quality_threshold: Maximum SQI difference for positive pairs (default: 0.1)
        negative_threshold: Minimum SQI difference for negative pairs (default: 0.3)
        use_hard_negatives: If True, focus on hardest negatives (default: True)
        similarity_metric: 'cosine' or 'l2' (default: 'cosine')

    Example:
        >>> loss_fn = QualityContrastiveLoss(temperature=0.07)
        >>> features = torch.randn(32, 256)  # [B, D]
        >>> quality_scores = torch.rand(32)  # [B] in [0, 1]
        >>> loss = loss_fn(features, quality_scores)
        >>> loss.backward()
    """

    def __init__(
        self,
        temperature: float = 0.07,
        quality_threshold: float = 0.1,
        negative_threshold: float = 0.3,
        use_hard_negatives: bool = True,
        similarity_metric: str = 'cosine'
    ):
        super().__init__()
        self.temperature = temperature
        self.quality_threshold = quality_threshold
        self.negative_threshold = negative_threshold
        self.use_hard_negatives = use_hard_negatives
        self.similarity_metric = similarity_metric

        # For logging
        self.last_num_positives = 0
        self.last_num_negatives = 0

    def forward(
        self,
        features: torch.Tensor,
        quality_scores: torch.Tensor,
        return_metrics: bool = False
    ) -> torch.Tensor:
        """Compute quality-aware contrastive loss.

        Args:
            features: Feature vectors [B, D]
            quality_scores: Quality scores [B] in [0, 1]
            return_metrics: If True, return (loss, metrics_dict)

        Returns:
            loss: Scalar contrastive loss
            metrics: Dict with diagnostic metrics (if return_metrics=True)
        """
        batch_size = features.shape[0]
        device = features.device

        # Normalize features for cosine similarity
        if self.similarity_metric == 'cosine':
            features = F.normalize(features, dim=1, p=2)

        # Compute pairwise similarity matrix
        if self.similarity_metric == 'cosine':
            # Cosine similarity: features @ features.T
            sim_matrix = torch.matmul(features, features.T) / self.temperature
        else:
            # L2 distance: -||f_i - f_j||^2
            diff = features.unsqueeze(1) - features.unsqueeze(0)  # [B, B, D]
            sim_matrix = -torch.sum(diff ** 2, dim=-1) / self.temperature

        # Compute pairwise quality differences
        quality_diff = torch.abs(
            quality_scores.unsqueeze(1) - quality_scores.unsqueeze(0)
        )  # [B, B]

        # Create masks for positive and negative pairs
        # Positive: similar quality (within threshold)
        positive_mask = (quality_diff < self.quality_threshold).float()

        # Negative: different quality (beyond threshold)
        negative_mask = (quality_diff > self.negative_threshold).float()

        # Remove diagonal (self-similarity)
        eye_mask = torch.eye(batch_size, device=device)
        positive_mask = positive_mask * (1 - eye_mask)
        negative_mask = negative_mask * (1 - eye_mask)

        # Count pairs for diagnostics
        num_positives = positive_mask.sum(dim=1)  # [B]
        num_negatives = negative_mask.sum(dim=1)  # [B]

        # Store for logging
        self.last_num_positives = num_positives.mean().item()
        self.last_num_negatives = num_negatives.mean().item()

        # Filter out samples with no positive or negative pairs
        valid_samples = (num_positives > 0) & (num_negatives > 0)

        if valid_samples.sum() == 0:
            # No valid samples, return zero loss
            if return_metrics:
                return torch.tensor(0.0, device=device), {
                    'num_positives': 0.0,
                    'num_negatives': 0.0,
                    'valid_samples': 0,
                    'quality_range': 0.0
                }
            return torch.tensor(0.0, device=device)

        # Compute InfoNCE loss for valid samples
        losses = []

        for i in range(batch_size):
            if not valid_samples[i]:
                continue

            # Get positive and negative similarities for sample i
            pos_sim = sim_matrix[i] * positive_mask[i]  # [B]
            neg_sim = sim_matrix[i] * negative_mask[i]  # [B]

            # Extract non-zero similarities
            pos_sim = pos_sim[positive_mask[i] > 0]  # [num_pos]
            neg_sim = neg_sim[negative_mask[i] > 0]  # [num_neg]

            if len(pos_sim) == 0 or len(neg_sim) == 0:
                continue

            # InfoNCE loss: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
            if self.use_hard_negatives:
                # Focus on hardest negatives (highest similarity to anchor)
                top_k = min(len(neg_sim), 10)
                neg_sim, _ = torch.topk(neg_sim, k=top_k)

            # Compute for each positive
            for pos in pos_sim:
                # Numerator: exp(positive similarity)
                numerator = torch.exp(pos)

                # Denominator: exp(pos) + sum(exp(negatives))
                denominator = numerator + torch.sum(torch.exp(neg_sim))

                # InfoNCE: -log(numerator / denominator)
                loss_i = -torch.log(numerator / (denominator + 1e-8))
                losses.append(loss_i)

        if len(losses) == 0:
            # No valid losses computed
            if return_metrics:
                return torch.tensor(0.0, device=device), {
                    'num_positives': self.last_num_positives,
                    'num_negatives': self.last_num_negatives,
                    'valid_samples': 0,
                    'quality_range': quality_scores.max().item() - quality_scores.min().item()
                }
            return torch.tensor(0.0, device=device)

        # Average over all positive pairs
        loss = torch.mean(torch.stack(losses))

        if return_metrics:
            metrics = {
                'num_positives': self.last_num_positives,
                'num_negatives': self.last_num_negatives,
                'valid_samples': valid_samples.sum().item(),
                'quality_range': quality_scores.max().item() - quality_scores.min().item(),
                'avg_pos_sim': sim_matrix[positive_mask > 0].mean().item() if (positive_mask > 0).any() else 0.0,
                'avg_neg_sim': sim_matrix[negative_mask > 0].mean().item() if (negative_mask > 0).any() else 0.0
            }
            return loss, metrics

        return loss


class HybridSSLLoss(nn.Module):
    """Hybrid loss combining quality-aware contrastive and reconstruction.

    Combines:
    1. Quality-aware contrastive loss (primary): Learn quality-relevant features
    2. Masked signal modeling (secondary): Maintain general reconstruction ability

    Args:
        contrastive_weight: Weight for contrastive loss (default: 1.0)
        reconstruction_weight: Weight for reconstruction loss (default: 0.3)
        **contrastive_kwargs: Arguments for QualityContrastiveLoss

    Example:
        >>> loss_fn = HybridSSLLoss(
        ...     contrastive_weight=1.0,
        ...     reconstruction_weight=0.3,
        ...     temperature=0.07
        ... )
        >>> features = torch.randn(32, 256)
        >>> quality_scores = torch.rand(32)
        >>> pred_signals = torch.randn(32, 2, 1024)
        >>> target_signals = torch.randn(32, 2, 1024)
        >>> mask = torch.rand(32, 8) > 0.6
        >>> loss = loss_fn(features, quality_scores, pred_signals, target_signals, mask)
    """

    def __init__(
        self,
        contrastive_weight: float = 1.0,
        reconstruction_weight: float = 0.3,
        **contrastive_kwargs
    ):
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.reconstruction_weight = reconstruction_weight

        self.contrastive_loss = QualityContrastiveLoss(**contrastive_kwargs)

    def forward(
        self,
        features: torch.Tensor,
        quality_scores: torch.Tensor,
        pred_signals: Optional[torch.Tensor] = None,
        target_signals: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> torch.Tensor:
        """Compute hybrid loss.

        Args:
            features: Feature vectors [B, D]
            quality_scores: Quality scores [B]
            pred_signals: Predicted signals [B, C, T] (optional)
            target_signals: Target signals [B, C, T] (optional)
            mask: Mask for reconstruction [B, P] (optional)
            return_components: If True, return (total_loss, components_dict)

        Returns:
            loss: Combined loss
            components: Dict with individual losses (if return_components=True)
        """
        # Contrastive loss
        contrastive, contrastive_metrics = self.contrastive_loss(
            features, quality_scores, return_metrics=True
        )

        total_loss = self.contrastive_weight * contrastive

        components = {
            'contrastive': contrastive.item(),
            'contrastive_metrics': contrastive_metrics
        }

        # Optional reconstruction loss
        if pred_signals is not None and target_signals is not None and mask is not None:
            # Simple MSE on masked regions
            # Expand mask to signal shape
            B, C, T = pred_signals.shape
            P = mask.shape[1]
            patch_size = T // P

            # Create patch-wise mask [B, P] -> [B, C, T]
            mask_expanded = mask.unsqueeze(1).repeat(1, C, 1)  # [B, C, P]
            mask_expanded = mask_expanded.unsqueeze(-1).repeat(1, 1, 1, patch_size)  # [B, C, P, patch_size]
            mask_expanded = mask_expanded.reshape(B, C, P * patch_size)  # [B, C, T]

            # Truncate if needed
            if mask_expanded.shape[-1] > T:
                mask_expanded = mask_expanded[..., :T]
            elif mask_expanded.shape[-1] < T:
                # Pad
                padding = T - mask_expanded.shape[-1]
                mask_expanded = F.pad(mask_expanded, (0, padding), value=0)

            # MSE only on masked regions
            diff = (pred_signals - target_signals) ** 2
            masked_diff = diff * mask_expanded

            num_masked = mask_expanded.sum()
            if num_masked > 0:
                reconstruction = masked_diff.sum() / num_masked
            else:
                reconstruction = torch.tensor(0.0, device=pred_signals.device)

            total_loss = total_loss + self.reconstruction_weight * reconstruction
            components['reconstruction'] = reconstruction.item()

        if return_components:
            components['total'] = total_loss.item()
            return total_loss, components

        return total_loss
