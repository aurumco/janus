"""Evaluation module for SSL pre-training metrics."""

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class PretrainEvaluator:
    """Evaluator for self-supervised pre-training tasks."""

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        """Initialize pretrain evaluator.

        Args:
            model: Pre-training model to evaluate.
            device: Device to run evaluation on.
        """
        self.model = model
        self.device = device

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate SSL pre-training performance.

        Args:
            dataloader: DataLoader with PretrainDataset.

        Returns:
            Dictionary with SSL evaluation metrics.
        """
        self.model.eval()

        total_recon_loss = 0.0
        total_vol_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                input_seq = batch["input_sequence"].to(self.device)
                mask_binary = batch["mask_binary"].to(self.device)
                original_seq = batch["original_sequence"].to(self.device)
                vol_target = batch["volatility_target"].to(self.device)
                asset_id = batch["asset_id"].to(self.device)

                outputs = self.model(input_seq, asset_id)
                recon_seq = outputs["reconstructed_sequence"]
                pred_vol = outputs["predicted_volatility"]

                batch_size = input_seq.size(0)
                recon_loss_batch = 0.0

                for i in range(batch_size):
                    mask_i = mask_binary[i]
                    if mask_i.sum() > 0:
                        recon_masked = recon_seq[i, mask_i]
                        orig_masked = original_seq[i, mask_i]
                        recon_loss_batch += torch.mean(
                            (recon_masked - orig_masked) ** 2
                        ).item()

                recon_loss_batch /= batch_size

                vol_loss_batch = torch.mean((pred_vol - vol_target) ** 2).item()

                total_recon_loss += recon_loss_batch * batch_size
                total_vol_loss += vol_loss_batch * batch_size
                total_samples += batch_size

        metrics = {
            "masked_reconstruction_mse": total_recon_loss / total_samples,
            "volatility_mse": total_vol_loss / total_samples,
        }

        return metrics

    def print_metrics(self, metrics: Dict[str, float]) -> None:
        """Print evaluation metrics.

        Args:
            metrics: Dictionary of metric names and values.
        """
        print("\n" + "=" * 60)
        print("SSL Pre-training Evaluation Metrics")
        print("=" * 60)
        for name, value in metrics.items():
            print(f"{name:30s}: {value:.6f}")
        print("=" * 60 + "\n")
