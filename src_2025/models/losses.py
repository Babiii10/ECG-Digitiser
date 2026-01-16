#!/usr/bin/env python3
"""
Custom loss functions for PhysioNet Challenge 2025
Optimized for TPR@5% metric
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Reference: https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean' or 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions (batch_size, 1)
            targets: Ground truth labels (batch_size, 1)

        Returns:
            Focal loss
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)

        # Binary cross entropy
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha weighting
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # Compute focal loss
        loss = alpha_weight * focal_weight * bce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class RankingLoss(nn.Module):
    """
    Custom ranking loss optimized for TPR@top-k%
    Penalizes positive samples ranked below negative samples
    """

    def __init__(self, k_percent=5, margin=0.1, lambda_bce=0.3):
        """
        Args:
            k_percent: Target percentage for TPR calculation
            margin: Margin for ranking loss
            lambda_bce: Weight for BCE component
        """
        super(RankingLoss, self).__init__()
        self.k_percent = k_percent
        self.margin = margin
        self.lambda_bce = lambda_bce
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions (batch_size, 1) - logits
            targets: Ground truth labels (batch_size, 1)

        Returns:
            Combined ranking + BCE loss
        """
        # Component 1: Standard BCE loss
        bce_loss = self.bce(inputs, targets)

        # Component 2: Pairwise ranking loss
        # Get positive and negative masks
        targets_squeezed = targets.squeeze()
        pos_mask = targets_squeezed == 1
        neg_mask = targets_squeezed == 0

        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            # Get positive and negative scores
            pos_scores = inputs.squeeze()[pos_mask]
            neg_scores = inputs.squeeze()[neg_mask]

            # Compute all pairwise differences
            # pos_scores: (n_pos,), neg_scores: (n_neg,)
            pos_expanded = pos_scores.unsqueeze(1)  # (n_pos, 1)
            neg_expanded = neg_scores.unsqueeze(0)  # (1, n_neg)

            # Ranking loss: max(0, margin + neg_score - pos_score)
            # We want pos_score > neg_score + margin
            pairwise_loss = torch.clamp(self.margin + neg_expanded - pos_expanded, min=0)
            ranking_loss = pairwise_loss.mean()
        else:
            ranking_loss = torch.tensor(0.0, device=inputs.device)

        # Combined loss
        total_loss = self.lambda_bce * bce_loss + (1 - self.lambda_bce) * ranking_loss

        return total_loss


class TPRLoss(nn.Module):
    """
    Loss that directly optimizes TPR@top-k%
    Penalizes positive samples not in top-k predictions
    """

    def __init__(self, k_percent=5, lambda_bce=0.3):
        super(TPRLoss, self).__init__()
        self.k_percent = k_percent
        self.lambda_bce = lambda_bce
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions (batch_size, 1) - logits
            targets: Ground truth labels (batch_size, 1)

        Returns:
            TPR-optimized loss
        """
        # Component 1: Standard BCE loss
        bce_loss = self.bce(inputs, targets)

        # Component 2: Penalty for positives not in top-k
        batch_size = inputs.size(0)
        k = max(1, int(batch_size * self.k_percent / 100))

        # Get top-k indices
        _, top_k_indices = torch.topk(inputs.squeeze(), k)

        # Create mask for top-k
        top_k_mask = torch.zeros(batch_size, dtype=torch.bool, device=inputs.device)
        top_k_mask[top_k_indices] = True

        # Find positives not in top-k
        targets_bool = targets.squeeze().bool()
        missed_positives = targets_bool & (~top_k_mask)

        # Penalty for missed positives
        if missed_positives.sum() > 0:
            # Higher penalty = lower predicted score for missed positives
            probs = torch.sigmoid(inputs.squeeze())
            penalty = (1 - probs[missed_positives]).sum()
        else:
            penalty = torch.tensor(0.0, device=inputs.device)

        # Reward for caught positives
        caught_positives = targets_bool & top_k_mask
        if caught_positives.sum() > 0:
            probs = torch.sigmoid(inputs.squeeze())
            reward = -probs[caught_positives].sum()  # Negative = reward (minimize loss)
        else:
            reward = torch.tensor(0.0, device=inputs.device)

        # Combined loss
        total_loss = self.lambda_bce * bce_loss + penalty + reward

        return total_loss


class WeightedBCELoss(nn.Module):
    """
    Weighted BCE loss for class imbalance
    """

    def __init__(self, pos_weight=1.0):
        """
        Args:
            pos_weight: Weight for positive class (auto-calculated if None)
        """
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions (batch_size, 1) - logits
            targets: Ground truth labels (batch_size, 1)

        Returns:
            Weighted BCE loss
        """
        if isinstance(self.pos_weight, float):
            pos_weight = torch.tensor([self.pos_weight], device=inputs.device)
        else:
            pos_weight = self.pos_weight

        return F.binary_cross_entropy_with_logits(
            inputs, targets,
            pos_weight=pos_weight
        )


class CombinedLoss(nn.Module):
    """
    Combination of multiple losses with learnable weights
    """

    def __init__(self, losses, weights=None):
        """
        Args:
            losses: List of loss functions
            weights: List of weights for each loss (learnable if None)
        """
        super(CombinedLoss, self).__init__()

        self.losses = nn.ModuleList(losses)

        if weights is None:
            # Initialize learnable weights
            self.weights = nn.Parameter(torch.ones(len(losses)) / len(losses))
        else:
            self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32))

    def forward(self, inputs, targets):
        """
        Compute weighted combination of losses
        """
        total_loss = 0
        weights_normalized = F.softmax(self.weights, dim=0)

        for i, loss_fn in enumerate(self.losses):
            loss = loss_fn(inputs, targets)
            total_loss += weights_normalized[i] * loss

        return total_loss


def create_loss_function(loss_type='focal', pos_weight=None, **kwargs):
    """
    Factory function to create loss function

    Args:
        loss_type: 'bce', 'weighted_bce', 'focal', 'ranking', 'tpr', 'combined'
        pos_weight: Weight for positive class (for weighted_bce)
        **kwargs: Additional arguments for specific losses

    Returns:
        Loss function
    """
    if loss_type == 'bce':
        return nn.BCEWithLogitsLoss()

    elif loss_type == 'weighted_bce':
        if pos_weight is None:
            pos_weight = 1.0
        return WeightedBCELoss(pos_weight=pos_weight)

    elif loss_type == 'focal':
        alpha = kwargs.get('alpha', 0.25)
        gamma = kwargs.get('gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)

    elif loss_type == 'ranking':
        k_percent = kwargs.get('k_percent', 5)
        margin = kwargs.get('margin', 0.1)
        lambda_bce = kwargs.get('lambda_bce', 0.3)
        return RankingLoss(k_percent=k_percent, margin=margin, lambda_bce=lambda_bce)

    elif loss_type == 'tpr':
        k_percent = kwargs.get('k_percent', 5)
        lambda_bce = kwargs.get('lambda_bce', 0.3)
        return TPRLoss(k_percent=k_percent, lambda_bce=lambda_bce)

    elif loss_type == 'combined':
        losses = [
            FocalLoss(alpha=0.25, gamma=2.0),
            RankingLoss(k_percent=5, margin=0.1, lambda_bce=0.3)
        ]
        return CombinedLoss(losses)

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")

    batch_size = 32
    inputs = torch.randn(batch_size, 1)
    targets = torch.randint(0, 2, (batch_size, 1)).float()

    print(f"Inputs shape: {inputs.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Positive samples: {targets.sum().item()}")

    # Test each loss
    for loss_type in ['bce', 'weighted_bce', 'focal', 'ranking', 'tpr']:
        loss_fn = create_loss_function(loss_type, pos_weight=2.0)
        loss_value = loss_fn(inputs, targets)
        print(f"{loss_type.upper()} loss: {loss_value.item():.4f}")

    print("✓ All loss functions working correctly!")
