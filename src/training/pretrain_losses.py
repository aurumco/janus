"""Custom loss functions for SSL pre-training with multi-asset awareness."""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class PretrainLoss(nn.Module):
    """Combined loss for self-supervised pre-training tasks."""

    def __init__(
        self,
        masked_price_weight: float = 1.0,
        volatility_weight: float = 1.5,
        direction_weight: float = 0.5,
        contrastive_weight: float = 0.3,
        temporal_consistency_weight: float = 0.2,
        temperature: float = 0.07,
        use_huber: bool = True,
        huber_delta: float = 1.0,
    ) -> None:
        """Initialize pre-training loss.

        Args:
            masked_price_weight: Weight for masked price reconstruction loss.
            volatility_weight: Weight for volatility prediction loss.
            direction_weight: Weight for direction prediction loss.
            contrastive_weight: Weight for contrastive learning across assets.
            temporal_consistency_weight: Weight for temporal smoothness.
            temperature: Temperature parameter for contrastive loss.
            use_huber: Whether to use Huber loss instead of MSE.
            huber_delta: Delta parameter for Huber loss.
        """
        super().__init__()

        self.masked_price_weight = masked_price_weight
        self.volatility_weight = volatility_weight
        self.direction_weight = direction_weight
        self.contrastive_weight = contrastive_weight
        self.temporal_consistency_weight = temporal_consistency_weight
        self.temperature = temperature

        if use_huber:
            self.reconstruction_loss_fn = nn.HuberLoss(delta=huber_delta)
            self.volatility_loss_fn = nn.HuberLoss(delta=huber_delta)
        else:
            self.reconstruction_loss_fn = nn.MSELoss()
            self.volatility_loss_fn = nn.MSELoss()
        
        self.direction_loss_fn = nn.CrossEntropyLoss()

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
        reconstructed_sequence = model_outputs.get("reconstructed_sequence", None)
        predicted_volatility = model_outputs.get("predicted_volatility", None)
        predicted_direction = model_outputs.get("predicted_direction", None)

        mask_binary = batch.get("mask_binary", None)
        original_sequence = batch.get("original_sequence", None)
        volatility_target = batch.get("volatility_target", None)
        direction_target = batch.get("direction_target", None)

        batch_size = reconstructed_sequence.size(0)

        if reconstructed_sequence is not None and original_sequence is not None and mask_binary is not None:
            if isinstance(self.reconstruction_loss_fn, nn.HuberLoss):
                per_elem = torch.nn.functional.smooth_l1_loss(
                    reconstructed_sequence, original_sequence, reduction="none", beta=self.reconstruction_loss_fn.delta
                )
            else:
                per_elem = (reconstructed_sequence - original_sequence) ** 2

            mask_exp = mask_binary.unsqueeze(-1).expand_as(per_elem)
            masked_losses = per_elem[mask_exp]
            if masked_losses.numel() == 0:
                masked_price_loss = torch.tensor(0.0, device=per_elem.device)
            else:
                masked_price_loss = masked_losses.mean()
                if torch.isnan(masked_price_loss) or torch.isinf(masked_price_loss):
                    masked_price_loss = torch.tensor(0.0, device=per_elem.device)
        else:
            masked_price_loss = torch.tensor(0.0, device=next(self.parameters()).device)

        if predicted_volatility is not None and volatility_target is not None:
            vol_valid = batch.get("volatility_valid", None)
            if vol_valid is not None:
                vol_valid = vol_valid.to(predicted_volatility.device).view(-1)
                pred_vol = predicted_volatility.view(-1)
                true_vol = volatility_target.view(-1)
                if vol_valid.any():
                    vloss_all = self.volatility_loss_fn(pred_vol[vol_valid], true_vol[vol_valid])
                    volatility_loss = vloss_all
                else:
                    volatility_loss = torch.tensor(0.0, device=predicted_volatility.device)
            else:
                volatility_loss = self.volatility_loss_fn(
                    predicted_volatility, volatility_target
                )
            if torch.isnan(volatility_loss) or torch.isinf(volatility_loss):
                volatility_loss = torch.tensor(0.0, device=predicted_volatility.device)
        else:
            volatility_loss = torch.tensor(0.0, device=next(self.parameters()).device)

        if predicted_direction is not None and direction_target is not None:
            direction_loss = self.direction_loss_fn(predicted_direction, direction_target)
            if torch.isnan(direction_loss) or torch.isinf(direction_loss):
                direction_loss = torch.tensor(0.0, device=predicted_direction.device)
        else:
            direction_loss = torch.tensor(0.0, device=reconstructed_sequence.device)

        total_loss = (
            self.masked_price_weight * masked_price_loss
            + self.volatility_weight * volatility_loss
            + self.direction_weight * direction_loss
        )
        
        loss_dict = {
            "masked_price_loss": masked_price_loss,
            "volatility_loss": volatility_loss,
            "direction_loss": direction_loss,
        }

        if self.contrastive_weight > 0 and "asset_id" in batch:
            contrastive_loss = self._contrastive_loss(
                reconstructed_sequence, batch["asset_id"]
            )
            if torch.isnan(contrastive_loss) or torch.isinf(contrastive_loss):
                contrastive_loss = torch.tensor(0.0, device=reconstructed_sequence.device)
            total_loss = total_loss + self.contrastive_weight * contrastive_loss
            loss_dict["contrastive_loss"] = contrastive_loss

        if self.temporal_consistency_weight > 0:
            temporal_loss = self._temporal_consistency_loss(reconstructed_sequence)
            if torch.isnan(temporal_loss) or torch.isinf(temporal_loss):
                temporal_loss = torch.tensor(0.0, device=reconstructed_sequence.device)
            total_loss = total_loss + self.temporal_consistency_weight * temporal_loss
            loss_dict["temporal_consistency_loss"] = temporal_loss

        loss_dict["total_loss"] = total_loss
        return loss_dict

    def _contrastive_loss(
        self, embeddings: torch.Tensor, asset_ids: torch.Tensor
    ) -> torch.Tensor:
        """Fully vectorized contrastive loss using InfoNCE with improved pooling.

        Args:
            embeddings: Sequence representations (batch, seq_len, features).
            asset_ids: Asset identifiers (batch,).

        Returns:
            Contrastive loss value.
        """
        # Improved pooling: combine mean and max for richer representation
        mean_pool = embeddings.mean(dim=1)
        max_pool, _ = embeddings.max(dim=1)
        pooled = (mean_pool + max_pool) / 2.0
        pooled_normalized = F.normalize(pooled, p=2, dim=1)
        
        batch_size = pooled.size(0)
        
        # Compute similarity matrix: (batch, batch)
        sim_matrix = (pooled_normalized @ pooled_normalized.T) / self.temperature
        
        # Check for NaN in similarity matrix
        if torch.isnan(sim_matrix).any():
            return torch.tensor(0.0, device=embeddings.device)
        
        # Create masks for same asset (positives)
        asset_eq = (asset_ids.unsqueeze(0) == asset_ids.unsqueeze(1))  # (batch, batch)
        
        # Remove self-similarity
        mask_self = torch.eye(batch_size, device=embeddings.device, dtype=torch.bool)
        asset_eq = asset_eq & ~mask_self
        
        # Check which samples have at least one positive
        has_positive = asset_eq.any(dim=1)
        if not has_positive.any():
            return torch.tensor(0.0, device=embeddings.device)
        
        # InfoNCE loss: -log( sum(exp(pos_sim)) / sum(exp(all_sim)) )
        # Use temperature-aware masking to avoid overflow
        # With temperature=0.07, max safe value in float16 is ~4500
        mask_value = min(-10.0 / self.temperature, -1000.0)  # Adaptive to temperature
        
        # Numerator: sum of similarities with positives
        pos_sim = sim_matrix.masked_fill(~asset_eq, mask_value)
        pos_exp_sum = torch.exp(torch.clamp(pos_sim, min=-50, max=50)).sum(dim=1)
        
        # Denominator: sum of all similarities except self
        all_sim = sim_matrix.masked_fill(mask_self, mask_value)
        all_exp_sum = torch.exp(torch.clamp(all_sim, min=-50, max=50)).sum(dim=1)
        
        # Compute loss for samples with positives (add eps to prevent log(0))
        loss = -torch.log((pos_exp_sum + 1e-7) / (all_exp_sum + 1e-7))
        loss = loss[has_positive]
        
        # Final NaN check
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            return torch.tensor(0.0, device=embeddings.device)
        
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
