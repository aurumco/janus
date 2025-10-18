"""Custom loss functions for regression with directional awareness."""

import torch
import torch.nn as nn


class DirectionalMSELoss(nn.Module):
    """MSE Loss with directional penalty and variance regularization.
    
    Prevents mode collapse by penalizing constant predictions.
    """

    def __init__(
        self, 
        direction_weight: float = 2.0,
        variance_weight: float = 0.5,
        epsilon: float = 1e-8
    ) -> None:
        """Initialize DirectionalMSELoss.

        Args:
            direction_weight: Weight for directional penalty.
            variance_weight: Weight for variance regularization (prevents flat predictions).
            epsilon: Small value to avoid division by zero.
        """
        super().__init__()
        self.direction_weight = direction_weight
        self.variance_weight = variance_weight
        self.epsilon = epsilon

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute directional MSE loss with variance regularization.

        Args:
            predictions: Model predictions (batch_size, 1).
            targets: True targets (batch_size, 1).

        Returns:
            Combined loss value.
        """
        # Standard MSE loss
        mse_loss = torch.mean((predictions - targets) ** 2)

        # Directional component: Only penalize WRONG direction
        pred_signs = torch.sign(predictions + self.epsilon)
        target_signs = torch.sign(targets + self.epsilon)
        
        # Mask for wrong direction
        wrong_direction = (pred_signs != target_signs).float()
        
        # Extra penalty for wrong direction, scaled by magnitude of target
        directional_penalty = wrong_direction * (torch.abs(targets) ** 2)
        directional_loss = torch.mean(directional_penalty)

        # Variance regularization: Penalize if predictions have too low variance
        # This prevents mode collapse to constant predictions
        pred_var = torch.var(predictions)
        target_var = torch.var(targets)
        # Penalize when pred_var is much smaller than target_var
        variance_penalty = torch.clamp(target_var - pred_var, min=0.0)
        
        # Combined loss
        total_loss = (
            mse_loss + 
            self.direction_weight * directional_loss +
            self.variance_weight * variance_penalty
        )

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


class SignWeightedMSELoss(nn.Module):
    """MSE Loss that heavily weights sign accuracy.
    
    Uses multiplicative penalty for wrong sign predictions,
    preventing the model from predicting near-zero values.
    """

    def __init__(
        self,
        sign_penalty_multiplier: float = 5.0,
        epsilon: float = 1e-6,
    ) -> None:
        """Initialize SignWeightedMSELoss.

        Args:
            sign_penalty_multiplier: How much worse wrong-sign predictions are.
            epsilon: Small value for numerical stability.
        """
        super().__init__()
        self.sign_penalty_multiplier = sign_penalty_multiplier
        self.epsilon = epsilon

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute sign-weighted MSE loss.

        Args:
            predictions: Model predictions (batch_size, 1).
            targets: True targets (batch_size, 1).

        Returns:
            Weighted loss value.
        """
        # Base squared errors
        squared_errors = (predictions - targets) ** 2
        
        # Check sign agreement
        pred_signs = torch.sign(predictions)
        target_signs = torch.sign(targets)
        correct_sign = (pred_signs == target_signs).float()
        
        # Weight errors based on sign correctness
        # Correct sign: weight = 1.0
        # Wrong sign: weight = sign_penalty_multiplier
        weights = torch.where(
            correct_sign > 0.5,
            torch.ones_like(squared_errors),
            torch.full_like(squared_errors, self.sign_penalty_multiplier)
        )
        
        # Also penalize predictions that are too timid
        # If |target| > threshold but |pred| is small, add penalty
        target_magnitude = torch.abs(targets)
        pred_magnitude = torch.abs(predictions)
        timid_mask = (target_magnitude > 0.005) & (pred_magnitude < 0.002)
        weights = torch.where(
            timid_mask,
            weights * 2.0,  # Double penalty for being too conservative
            weights
        )
        
        # Weighted MSE
        weighted_loss = torch.mean(weights * squared_errors)
        
        return weighted_loss
