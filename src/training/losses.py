"""Custom loss functions for regression with directional awareness."""

import torch
import torch.nn as nn


class DirectionalMSELoss(nn.Module):
    """MSE Loss with directional penalty for wrong predictions.
    
    Combines MSE with a penalty for predicting the wrong direction,
    which is critical for trading applications.
    """

    def __init__(self, direction_weight: float = 1.0, epsilon: float = 1e-8) -> None:
        """Initialize DirectionalMSELoss.

        Args:
            direction_weight: Weight for directional penalty (higher = more emphasis on direction).
            epsilon: Small value to avoid division by zero.
        """
        super().__init__()
        self.direction_weight = direction_weight
        self.epsilon = epsilon

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute directional MSE loss.

        Args:
            predictions: Model predictions (batch_size, 1).
            targets: True targets (batch_size, 1).

        Returns:
            Combined loss value.
        """
        # Standard MSE loss
        mse_loss = torch.mean((predictions - targets) ** 2)

        # Directional penalty: penalize when signs don't match
        pred_signs = torch.sign(predictions)
        target_signs = torch.sign(targets)
        
        # Create mask where signs differ (wrong direction)
        wrong_direction = (pred_signs != target_signs).float()
        
        # Penalty is proportional to how wrong we are when direction is wrong
        directional_penalty = wrong_direction * torch.abs(predictions - targets)
        directional_loss = torch.mean(directional_penalty)

        # Combined loss
        total_loss = mse_loss + self.direction_weight * directional_loss

        return total_loss


class AsymmetricMSELoss(nn.Module):
    """MSE Loss with asymmetric penalties.
    
    Penalizes predictions that underestimate movements more heavily,
    as missing opportunities is worse than overestimating in trading.
    """

    def __init__(self, underestimate_penalty: float = 1.5) -> None:
        """Initialize AsymmetricMSELoss.

        Args:
            underestimate_penalty: Multiplier for underestimation errors.
        """
        super().__init__()
        self.underestimate_penalty = underestimate_penalty

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute asymmetric MSE loss.

        Args:
            predictions: Model predictions (batch_size, 1).
            targets: True targets (batch_size, 1).

        Returns:
            Asymmetric loss value.
        """
        errors = predictions - targets
        squared_errors = errors ** 2

        # For positive targets, penalize under-prediction more
        mask_positive = (targets > 0).float()
        underestimate_mask = ((predictions < targets) & (targets > 0)).float()
        
        # For negative targets, penalize over-prediction (less negative) more
        mask_negative = (targets < 0).float()
        overestimate_mask = ((predictions > targets) & (targets < 0)).float()

        # Apply asymmetric penalties
        penalties = torch.ones_like(squared_errors)
        penalties = penalties + (underestimate_mask + overestimate_mask) * (self.underestimate_penalty - 1.0)

        weighted_errors = squared_errors * penalties
        loss = torch.mean(weighted_errors)

        return loss


class QuantileLoss(nn.Module):
    """Quantile loss for robust regression.
    
    Less sensitive to outliers than MSE, but still differentiable.
    Can be tuned to prefer over/under prediction via quantile parameter.
    """

    def __init__(self, quantile: float = 0.5) -> None:
        """Initialize QuantileLoss.

        Args:
            quantile: Quantile to optimize (0.5 = median, <0.5 = under-predict, >0.5 = over-predict).
        """
        super().__init__()
        self.quantile = quantile

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute quantile loss.

        Args:
            predictions: Model predictions (batch_size, 1).
            targets: True targets (batch_size, 1).

        Returns:
            Quantile loss value.
        """
        errors = targets - predictions
        loss = torch.mean(
            torch.max(self.quantile * errors, (self.quantile - 1) * errors)
        )
        return loss


class CombinedDirectionalLoss(nn.Module):
    """Combines multiple loss components for optimal trading performance.
    
    - MSE for magnitude accuracy
    - Directional penalty for sign accuracy
    - Asymmetric penalty to avoid underestimating movements
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        direction_weight: float = 2.0,
        asymmetric_weight: float = 0.5,
        underestimate_penalty: float = 1.3,
    ) -> None:
        """Initialize CombinedDirectionalLoss.

        Args:
            mse_weight: Weight for MSE component.
            direction_weight: Weight for directional accuracy.
            asymmetric_weight: Weight for asymmetric penalty.
            underestimate_penalty: Penalty multiplier for underestimation.
        """
        super().__init__()
        self.mse_weight = mse_weight
        self.direction_weight = direction_weight
        self.asymmetric_weight = asymmetric_weight
        self.underestimate_penalty = underestimate_penalty

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined loss.

        Args:
            predictions: Model predictions (batch_size, 1).
            targets: True targets (batch_size, 1).

        Returns:
            Combined loss value.
        """
        # 1. MSE component
        mse_loss = torch.mean((predictions - targets) ** 2)

        # 2. Directional component
        pred_signs = torch.sign(predictions)
        target_signs = torch.sign(targets)
        wrong_direction = (pred_signs != target_signs).float()
        directional_penalty = wrong_direction * torch.abs(predictions - targets)
        directional_loss = torch.mean(directional_penalty)

        # 3. Asymmetric component (penalize underestimation)
        errors = predictions - targets
        squared_errors = errors ** 2
        
        underestimate_mask = ((predictions < targets) & (targets > 0)).float()
        overestimate_mask = ((predictions > targets) & (targets < 0)).float()
        
        penalties = torch.ones_like(squared_errors)
        penalties = penalties + (underestimate_mask + overestimate_mask) * (self.underestimate_penalty - 1.0)
        
        asymmetric_loss = torch.mean(squared_errors * penalties)

        # Combine all components
        total_loss = (
            self.mse_weight * mse_loss +
            self.direction_weight * directional_loss +
            self.asymmetric_weight * asymmetric_loss
        )

        return total_loss
