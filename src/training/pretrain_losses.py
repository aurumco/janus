"""Loss functions for self-supervised pre-training."""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class PretrainLoss(nn.Module):
    """Combined loss for self-supervised pre-training tasks."""

    def __init__(
        self,
        masked_price_weight: float = 1.0,
        volatility_weight: float = 0.5,
        contrastive_weight: float = 0.2,
        temporal_consistency_weight: float = 0.1,
        temperature: float = 0.07,
        use_huber: bool = True,
        huber_delta: float = 1.0,
    ) -> None:
        """Initialize pre-training loss.

        Args:
            masked_price_weight: Weight for masked price reconstruction loss.
            volatility_weight: Weight for volatility prediction loss.
            contrastive_weight: Weight for contrastive learning across assets.
            temporal_consistency_weight: Weight for temporal smoothness.
            temperature: Temperature parameter for contrastive loss.
            use_huber: Whether to use Huber loss instead of MSE.
            huber_delta: Delta parameter for Huber loss.
        """
        super().__init__()

        self.masked_price_weight = masked_price_weight
        self.volatility_weight = volatility_weight
        self.contrastive_weight = contrastive_weight
        self.temporal_consistency_weight = temporal_consistency_weight
        self.temperature = temperature

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

        mask_binary = batch["mask_binary"]
        original_sequence = batch["original_sequence"]
        volatility_target = batch["volatility_target"]

        batch_size = reconstructed_sequence.size(0)

        masked_price_loss = 0.0
        total_masked = 0
        
        for i in range(batch_size):
            mask_i = mask_binary[i]
            if mask_i.sum() > 0:
                pred_masked = reconstructed_sequence[i, mask_i]
                true_masked = original_sequence[i, mask_i]
                masked_price_loss += self.reconstruction_loss_fn(
                    pred_masked, true_masked
                )
                total_masked += 1

        masked_price_loss = masked_price_loss / max(total_masked, 1)

        volatility_loss = self.volatility_loss_fn(
            predicted_volatility, volatility_target
        )

        total_loss = (
            self.masked_price_weight * masked_price_loss
            + self.volatility_weight * volatility_loss
        )
        
        loss_dict = {
            "masked_price_loss": masked_price_loss,
            "volatility_loss": volatility_loss,
        }

        if self.contrastive_weight > 0 and "asset_id" in batch:
            contrastive_loss = self._contrastive_loss(
                reconstructed_sequence, batch["asset_id"]
            )
            total_loss = total_loss + self.contrastive_weight * contrastive_loss
            loss_dict["contrastive_loss"] = contrastive_loss

        if self.temporal_consistency_weight > 0:
            temporal_loss = self._temporal_consistency_loss(reconstructed_sequence)
            total_loss = total_loss + self.temporal_consistency_weight * temporal_loss
            loss_dict["temporal_consistency_loss"] = temporal_loss

        loss_dict["total_loss"] = total_loss
        return loss_dict

    def _contrastive_loss(
        self, embeddings: torch.Tensor, asset_ids: torch.Tensor
    ) -> torch.Tensor:
        """Contrastive loss to separate different assets, group similar ones.

        Args:
            embeddings: Sequence representations (batch, seq_len, features).
            asset_ids: Asset identifiers (batch,).

        Returns:
            Contrastive loss value.
        """
        pooled = embeddings.mean(dim=1)
        
        pooled_normalized = F.normalize(pooled, p=2, dim=1)
        
        sim_matrix = torch.matmul(pooled_normalized, pooled_normalized.T) / self.temperature
        
        same_asset_mask = (asset_ids.unsqueeze(1) == asset_ids.unsqueeze(0)).float()
        
        same_asset_mask = same_asset_mask - torch.eye(
            same_asset_mask.size(0), device=same_asset_mask.device
        )
        
        exp_sim = torch.exp(sim_matrix)
        
        positive_pairs = (exp_sim * same_asset_mask).sum(dim=1)
        all_pairs = exp_sim.sum(dim=1)
        
        loss = -torch.log((positive_pairs + 1e-8) / (all_pairs + 1e-8))
        
        return loss.mean()

    def _temporal_consistency_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Penalize abrupt changes in temporal embeddings.

        Args:
            embeddings: Sequence representations (batch, seq_len, features).

        Returns:
            Temporal consistency loss.
        """
        diff = embeddings[:, 1:, :] - embeddings[:, :-1, :]
        
        temporal_variance = torch.norm(diff, p=2, dim=-1)
        
        return temporal_variance.mean()
