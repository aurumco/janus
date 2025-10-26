"""Loss functions for self-supervised pre-training."""

from typing import Dict

import torch
import torch.nn as nn


class PretrainLoss(nn.Module):
    """Combined loss for self-supervised pre-training tasks."""

    def __init__(
        self,
        masked_price_weight: float = 1.0,
        volatility_weight: float = 0.5,
        use_huber: bool = True,
        huber_delta: float = 1.0,
    ) -> None:
        """Initialize pre-training loss.

        Args:
            masked_price_weight: Weight for masked price reconstruction loss.
            volatility_weight: Weight for volatility prediction loss.
            use_huber: Whether to use Huber loss instead of MSE.
            huber_delta: Delta parameter for Huber loss.
        """
        super().__init__()
        self.masked_price_weight = masked_price_weight
        self.volatility_weight = volatility_weight

        if use_huber:
            self.reconstruction_loss_fn = nn.HuberLoss(delta=huber_delta)
            self.volatility_loss_fn = nn.HuberLoss(delta=huber_delta)
        else:
            self.reconstruction_loss_fn = nn.MSELoss()
            self.volatility_loss_fn = nn.MSELoss()

    def forward(
        self, model_outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute combined pre-training loss.

        Args:
            model_outputs: Dictionary containing model predictions.
            batch: Dictionary containing batch data.

        Returns:
            Dictionary with total loss and individual loss components.
        """
        reconstructed_sequence = model_outputs["reconstructed_sequence"]
        predicted_volatility = model_outputs["predicted_volatility"]

        mask_indices = batch["mask_indices"]
        original_masked_values = batch["original_masked_values"]
        volatility_target = batch["volatility_target"]

        batch_size = reconstructed_sequence.size(0)

        masked_price_loss = 0.0
        for i in range(batch_size):
            indices = mask_indices[i]
            if len(indices) > 0:
                pred_masked = reconstructed_sequence[i, indices]
                true_masked = original_masked_values[i]
                masked_price_loss += self.reconstruction_loss_fn(
                    pred_masked, true_masked
                )

        masked_price_loss = masked_price_loss / batch_size

        volatility_loss = self.volatility_loss_fn(
            predicted_volatility, volatility_target
        )

        total_loss = (
            self.masked_price_weight * masked_price_loss
            + self.volatility_weight * volatility_loss
        )

        return {
            "total_loss": total_loss,
            "masked_price_loss": masked_price_loss,
            "volatility_loss": volatility_loss,
        }
