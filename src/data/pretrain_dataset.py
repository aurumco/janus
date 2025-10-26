"""PyTorch dataset for SSL pre-training with masking."""

from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    """Dataset for self-supervised pre-training with masked reconstruction."""

    def __init__(
        self,
        X: np.ndarray,
        asset_ids: np.ndarray,
        sequence_length: int,
        masking_ratio: float = 0.15,
        volatility_lookahead: int = 60,
        price_column_idx: int = 0,
    ) -> None:
        """Initialize pre-training dataset.

        Args:
            X: Feature sequences of shape (n_samples, seq_len, n_features).
            asset_ids: Asset identifiers of shape (n_samples,).
            sequence_length: Length of input sequences.
            masking_ratio: Ratio of timesteps to mask.
            volatility_lookahead: Steps ahead for volatility prediction.
            price_column_idx: Index of price column for volatility calc.
        """
        self.X = torch.FloatTensor(X)
        self.asset_ids = torch.LongTensor(asset_ids)
        self.sequence_length = sequence_length
        self.masking_ratio = masking_ratio
        self.volatility_lookahead = volatility_lookahead
        self.price_column_idx = price_column_idx
        self.n_samples, self.seq_len, self.n_features = self.X.shape

    def __len__(self) -> int:
        """Get dataset length.

        Returns:
            Number of valid samples.
        """
        return max(0, self.n_samples - self.volatility_lookahead)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single pre-training sample with masking.

        Args:
            idx: Sample index.

        Returns:
            Dictionary containing masked input, targets, and metadata.
        """
        original_sequence = self.X[idx].clone()
        asset_id = self.asset_ids[idx]
        
        num_to_mask = max(1, int(self.seq_len * self.masking_ratio))
        mask_indices = torch.randperm(self.seq_len)[:num_to_mask]
        
        original_masked_values = original_sequence[mask_indices].clone()
        
        masked_sequence = original_sequence.clone()
        masked_sequence[mask_indices] = 0.0
        
        mask_binary = torch.zeros(self.seq_len, dtype=torch.bool)
        mask_binary[mask_indices] = True
        
        future_end_idx = min(idx + 1 + self.volatility_lookahead, self.n_samples)
        if future_end_idx > idx + 1:
            future_prices = self.X[
                idx + 1 : future_end_idx,
                :,
                self.price_column_idx,
            ]
            
            if len(future_prices) > 1:
                log_returns = torch.log(
                    (future_prices[1:] + 1e-8) / (future_prices[:-1] + 1e-8)
                )
                volatility = torch.std(log_returns)
            else:
                volatility = torch.tensor(0.0)
        else:
            volatility = torch.tensor(0.0)

        return {
            "input_sequence": masked_sequence,
            "mask_binary": mask_binary,
            "original_masked_values": original_masked_values,
            "original_sequence": original_sequence,
            "volatility_target": volatility.unsqueeze(0),
            "asset_id": asset_id,
        }
