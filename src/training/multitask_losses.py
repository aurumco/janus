"""Multi-task loss functions for simultaneous direction and magnitude prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class MultiTaskRegressionLoss(nn.Module):
    """Combined loss for multi-task regression.
    
    Optimizes three objectives simultaneously:
    1. Direction classification (sign correctness)
    2. Magnitude regression (absolute value accuracy)
    3. Confidence calibration (uncertainty estimation)
    """

    def __init__(
        self,
        direction_weight: float = 2.0,
        magnitude_weight: float = 1.0,
        confidence_weight: float = 0.5,
        regression_weight: float = 1.5,
        focal_gamma: float = 2.0,
    ) -> None:
        """Initialize multi-task loss.

        Args:
            direction_weight: Weight for direction classification loss.
            magnitude_weight: Weight for magnitude regression loss.
            confidence_weight: Weight for confidence calibration loss.
            regression_weight: Weight for final regression loss.
            focal_gamma: Gamma parameter for focal loss (reduces easy examples).
        """
        super().__init__()
        self.direction_weight = direction_weight
        self.magnitude_weight = magnitude_weight
        self.confidence_weight = confidence_weight
        self.regression_weight = regression_weight
        self.focal_gamma = focal_gamma

    def forward(
        self, 
        outputs: Dict[str, torch.Tensor], 
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss.

        Args:
            outputs: Dictionary with keys:
                - 'regression': Final prediction
                - 'direction_logits': Direction classification logits (batch, 3)
                - 'magnitude': Magnitude prediction (batch, 1)
                - 'confidence': Confidence scores (batch, 1)
            targets: True target values (batch, 1).

        Returns:
            Dictionary with total loss and component losses.
        """
        regression_pred = outputs['regression']
        direction_logits = outputs['direction_logits']
        magnitude_pred = outputs['magnitude']
        confidence = outputs['confidence']

        # 1. Direction Classification Loss (Focal Cross-Entropy)
        # Convert continuous targets to direction classes: 0=negative, 1=neutral, 2=positive
        direction_targets = self._targets_to_direction_classes(targets)
        direction_loss = self._focal_cross_entropy(
            direction_logits, 
            direction_targets,
            gamma=self.focal_gamma
        )

        # 2. Magnitude Regression Loss (Huber on absolute values)
        target_magnitude = torch.abs(targets)
        magnitude_loss = F.huber_loss(
            magnitude_pred, 
            target_magnitude,
            delta=0.01  # Small delta for sensitivity to small changes
        )

        # 3. Confidence Calibration Loss
        # Confidence should correlate with actual prediction quality
        prediction_error = torch.abs(regression_pred - targets)
        # Low error -> high confidence, high error -> low confidence
        ideal_confidence = torch.exp(-prediction_error / 0.01)  # Exponential decay
        confidence_loss = F.mse_loss(confidence, ideal_confidence)

        # 4. Final Regression Loss (Main task)
        # Use confidence-weighted MSE
        weighted_errors = confidence * (regression_pred - targets) ** 2
        regression_loss = torch.mean(weighted_errors)
        
        # Also add standard MSE to ensure we don't just reduce confidence
        regression_loss = regression_loss + 0.5 * F.mse_loss(regression_pred, targets)

        # 5. Direction Consistency Penalty
        # Penalize if predicted sign doesn't match direction classification
        pred_sign = torch.sign(regression_pred)
        direction_probs = F.softmax(direction_logits, dim=-1)
        expected_sign = direction_probs[:, 2:3] - direction_probs[:, 0:1]  # P(pos) - P(neg)
        expected_sign_discrete = torch.sign(expected_sign)
        
        sign_mismatch = (pred_sign != expected_sign_discrete).float()
        consistency_loss = torch.mean(sign_mismatch * torch.abs(regression_pred))

        # Total loss
        total_loss = (
            self.direction_weight * direction_loss +
            self.magnitude_weight * magnitude_loss +
            self.confidence_weight * confidence_loss +
            self.regression_weight * regression_loss +
            0.5 * consistency_loss
        )

        return {
            'loss': total_loss,
            'direction_loss': direction_loss.detach(),
            'magnitude_loss': magnitude_loss.detach(),
            'confidence_loss': confidence_loss.detach(),
            'regression_loss': regression_loss.detach(),
            'consistency_loss': consistency_loss.detach(),
        }

    def _targets_to_direction_classes(self, targets: torch.Tensor) -> torch.Tensor:
        """Convert continuous targets to direction classes.
        
        Args:
            targets: Continuous target values (batch, 1).
            
        Returns:
            Class indices (batch,): 0=negative, 1=neutral, 2=positive.
        """
        # Use thresholds to define neutral zone
        neutral_threshold = 0.001  # ±0.1%
        
        classes = torch.zeros(targets.size(0), dtype=torch.long, device=targets.device)
        classes[targets.squeeze() < -neutral_threshold] = 0  # Negative
        classes[targets.squeeze() > neutral_threshold] = 2   # Positive
        classes[(targets.squeeze() >= -neutral_threshold) & 
                (targets.squeeze() <= neutral_threshold)] = 1  # Neutral
        
        return classes

    def _focal_cross_entropy(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor,
        gamma: float = 2.0
    ) -> torch.Tensor:
        """Focal loss for handling class imbalance.
        
        Reduces loss contribution from easy examples, focusing on hard ones.
        
        Args:
            logits: Prediction logits (batch, num_classes).
            targets: Target class indices (batch,).
            gamma: Focusing parameter (higher = more focus on hard examples).
            
        Returns:
            Scalar loss value.
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = ((1 - p_t) ** gamma) * ce_loss
        return torch.mean(focal_loss)


class AdaptiveMultiTaskLoss(nn.Module):
    """Multi-task loss with learned task weights.
    
    Automatically balances task weights during training based on
    relative task difficulties (homoscedastic uncertainty).
    
    Reference: "Multi-Task Learning Using Uncertainty to Weigh Losses"
    (Kendall et al., CVPR 2018)
    """

    def __init__(
        self,
        base_loss: MultiTaskRegressionLoss,
        init_log_vars: Dict[str, float] = None,
    ) -> None:
        """Initialize adaptive multi-task loss.

        Args:
            base_loss: Base multi-task loss function.
            init_log_vars: Initial log variances for each task.
        """
        super().__init__()
        self.base_loss = base_loss
        
        # Learnable log variances (one per task)
        if init_log_vars is None:
            init_log_vars = {
                'direction': 0.0,
                'magnitude': 0.0,
                'confidence': 0.0,
                'regression': 0.0,
            }
        
        self.log_vars = nn.ParameterDict({
            task: nn.Parameter(torch.tensor(log_var))
            for task, log_var in init_log_vars.items()
        })

    def forward(
        self, 
        outputs: Dict[str, torch.Tensor], 
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute adaptive multi-task loss.

        Args:
            outputs: Model outputs dictionary.
            targets: True target values.

        Returns:
            Dictionary with total loss and components.
        """
        # Get base losses
        loss_dict = self.base_loss(outputs, targets)
        
        # Apply learned weights (inverse of variance)
        precision_direction = torch.exp(-self.log_vars['direction'])
        precision_magnitude = torch.exp(-self.log_vars['magnitude'])
        precision_confidence = torch.exp(-self.log_vars['confidence'])
        precision_regression = torch.exp(-self.log_vars['regression'])
        
        # Weighted loss with regularization on log variances
        total_loss = (
            precision_direction * loss_dict['direction_loss'] + self.log_vars['direction'] +
            precision_magnitude * loss_dict['magnitude_loss'] + self.log_vars['magnitude'] +
            precision_confidence * loss_dict['confidence_loss'] + self.log_vars['confidence'] +
            precision_regression * loss_dict['regression_loss'] + self.log_vars['regression']
        )
        
        loss_dict['loss'] = total_loss
        loss_dict['precision_direction'] = precision_direction.detach()
        loss_dict['precision_magnitude'] = precision_magnitude.detach()
        loss_dict['precision_confidence'] = precision_confidence.detach()
        loss_dict['precision_regression'] = precision_regression.detach()
        
        return loss_dict
