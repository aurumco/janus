"""Evaluation module for SSL pre-training metrics."""

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

try:
    from src.utils.logger import logger
except:
    from ..utils.logger import logger


class EnhancedPretrainEvaluator:
    """Enhanced evaluator for self-supervised pre-training tasks."""

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        """Initialize pretrain evaluator.

        Args:
            model: Pre-training model to evaluate.
            device: Device to run evaluation on.
        """
        self.model = model
        self.device = device

    def evaluate(self, dataloader: DataLoader, max_batches: int = 100) -> Dict[str, float]:
        """Evaluate SSL pre-training performance with advanced metrics.

        Args:
            dataloader: DataLoader with PretrainDataset.
            max_batches: Maximum number of batches to evaluate (for speed).

        Returns:
            Dictionary with comprehensive SSL evaluation metrics.
        """
        self.model.eval()

        total_recon_loss = 0.0
        total_vol_loss = 0.0
        total_samples = 0

        all_embeddings: List[torch.Tensor] = []
        all_asset_ids: List[torch.Tensor] = []
        all_pred_vols: List[torch.Tensor] = []
        all_true_vols: List[torch.Tensor] = []

        logger.info(f"Evaluating on {min(len(dataloader), max_batches)} batches", indent=1)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", total=min(len(dataloader), max_batches))):
                if batch_idx >= max_batches:
                    break
                input_seq = batch["input_sequence"].to(self.device)
                mask_binary = batch["mask_binary"].to(self.device)
                original_seq = batch["original_sequence"].to(self.device)
                vol_target = batch["volatility_target"].to(self.device)
                asset_id = batch["asset_id"].to(self.device)

                outputs = self.model(input_seq, asset_id)
                recon_seq = outputs["reconstructed_sequence"]
                pred_vol = outputs["predicted_volatility"]

                batch_size = input_seq.size(0)
                
                mask_exp = mask_binary.unsqueeze(-1).expand_as(recon_seq)
                masked_recon = recon_seq[mask_exp]
                masked_orig = original_seq[mask_exp]
                
                if masked_recon.numel() > 0:
                    recon_loss_batch = torch.mean((masked_recon - masked_orig) ** 2).item()
                else:
                    recon_loss_batch = 0.0

                vol_diff = (pred_vol - vol_target) ** 2
                if torch.isnan(vol_diff).any() or torch.isinf(vol_diff).any():
                    vol_loss_batch = 0.0
                else:
                    vol_loss_batch = torch.mean(vol_diff).item()

                total_recon_loss += recon_loss_batch * batch_size
                total_vol_loss += vol_loss_batch * batch_size
                total_samples += batch_size

                pooled_embedding = recon_seq.mean(dim=1)
                all_embeddings.append(pooled_embedding.cpu())
                all_asset_ids.append(asset_id.cpu())
                all_pred_vols.append(pred_vol.cpu())
                all_true_vols.append(vol_target.cpu())

        if total_samples == 0 or len(all_embeddings) == 0:
            return {
                "masked_reconstruction_mse": 0.0,
                "volatility_mse": 0.0,
                "embedding_silhouette_score": 0.0,
                "volatility_correlation": 0.0,
                "temporal_consistency": 0.0,
            }

        all_embeddings_cat = torch.cat(all_embeddings, dim=0)
        all_asset_ids_cat = torch.cat(all_asset_ids, dim=0)
        all_pred_vols_cat = torch.cat(all_pred_vols, dim=0)
        all_true_vols_cat = torch.cat(all_true_vols, dim=0)

        metrics = {
            "masked_reconstruction_mse": total_recon_loss / total_samples,
            "volatility_mse": total_vol_loss / total_samples,
        }

        embedding_quality = self._evaluate_embedding_quality(
            all_embeddings_cat, all_asset_ids_cat
        )
        metrics["embedding_silhouette_score"] = embedding_quality

        vol_correlation = self._evaluate_volatility_correlation(
            all_pred_vols_cat, all_true_vols_cat
        )
        metrics["volatility_correlation"] = vol_correlation

        temporal_consistency = self._evaluate_temporal_consistency(all_embeddings)
        metrics["temporal_consistency"] = temporal_consistency

        return metrics

    def _evaluate_embedding_quality(
        self, embeddings: torch.Tensor, asset_ids: torch.Tensor
    ) -> float:
        """Evaluate how well embeddings separate different assets.

        Args:
            embeddings: All embeddings (n_samples, embedding_dim).
            asset_ids: Asset identifiers (n_samples,).

        Returns:
            Silhouette score (higher is better).
        """
        try:
            from sklearn.metrics import silhouette_score

            embeddings_np = embeddings.numpy()
            asset_ids_np = asset_ids.numpy()

            if len(np.unique(asset_ids_np)) > 1:
                score = silhouette_score(embeddings_np, asset_ids_np)
                return float(score)
            else:
                return 0.0
        except Exception:
            return 0.0

    def _evaluate_volatility_correlation(
        self, pred_vols: torch.Tensor, true_vols: torch.Tensor
    ) -> float:
        """Evaluate correlation between predicted and true volatility.

        Args:
            pred_vols: Predicted volatilities.
            true_vols: True volatilities.

        Returns:
            Pearson correlation coefficient.
        """
        pred_np = pred_vols.squeeze().numpy()
        true_np = true_vols.squeeze().numpy()

        if len(pred_np) > 1:
            corr = np.corrcoef(pred_np, true_np)[0, 1]
            return float(corr) if not np.isnan(corr) else 0.0
        return 0.0

    def _evaluate_temporal_consistency(
        self, embeddings_list: List[torch.Tensor]
    ) -> float:
        """Evaluate temporal smoothness of embeddings.

        Args:
            embeddings_list: List of embedding batches.

        Returns:
            Average temporal variance (lower is better, more consistent).
        """
        if len(embeddings_list) < 2:
            return 0.0

        total_variance = 0.0
        count = 0

        for i in range(len(embeddings_list) - 1):
            diff = embeddings_list[i + 1] - embeddings_list[i]
            variance = torch.norm(diff, p=2, dim=-1).mean().item()
            total_variance += variance
            count += 1

        return total_variance / max(count, 1)

    def print_metrics(self, metrics: Dict[str, float]) -> None:
        logger.blank_line()
        logger.info("Reconstruction Performance:", indent=1)
        logger.metric("Masked Reconstruction MSE", f"{metrics.get('masked_reconstruction_mse', 0):.6f}", indent=2)

        logger.blank_line()
        logger.info("Volatility Prediction:", indent=1)
        vol_mse = metrics.get('volatility_mse', 0)
        vol_mse_str = "nan" if (vol_mse != vol_mse or vol_mse == float('inf')) else f"{vol_mse:.6f}"
        logger.metric("Volatility MSE", vol_mse_str, indent=2)
        logger.metric("Volatility Correlation", f"{metrics.get('volatility_correlation', 0):.4f}", indent=2)

        logger.blank_line()
        logger.info("Embedding Quality:", indent=1)
        logger.metric("Silhouette Score", f"{metrics.get('embedding_silhouette_score', 0):.4f}", indent=2)
        logger.metric("Temporal Consistency", f"{metrics.get('temporal_consistency', 0):.6f}", indent=2)
        logger.blank_line()

PretrainEvaluator = EnhancedPretrainEvaluator
