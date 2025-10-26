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
        sequence_length: int,
        masking_ratio: float = 0.15,
        volatility_lookahead: int = 60,
    ) -> None:
        """Initialize pre-training dataset.

        Args:
            X: Feature sequences of shape (n_samples, seq_len, n_features).
            sequence_length: Length of input sequences.
            masking_ratio: Ratio of timesteps to mask.
            volatility_lookahead: Steps ahead for volatility prediction.
        """
        self.X = torch.FloatTensor(X)
        self.sequence_length = sequence_length
        self.masking_ratio = masking_ratio
        self.volatility_lookahead = volatility_lookahead
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
            Dictionary containing masked input and targets.
        """
        sequence = self.X[idx].clone()
        
        num_to_mask = max(1, int(self.seq_len * self.masking_ratio))
        mask_indices = torch.randperm(self.seq_len)[:num_to_mask]
        
        original_values = sequence[mask_indices].clone()
        
        masked_sequence = sequence.clone()
        masked_sequence[mask_indices] = 0.0
        
        future_prices = self.X[
            idx + 1 : min(idx + 1 + self.volatility_lookahead, self.n_samples),
            :,
            0,
        ]
        
        if len(future_prices) > 1:
            log_returns = torch.log(future_prices[1:] / (future_prices[:-1] + 1e-8))
            volatility = torch.std(log_returns)
        else:
            volatility = torch.tensor(0.0)

        return {
            "input_sequence": masked_sequence,
            "mask_indices": mask_indices,
            "original_masked_values": original_values,
            "volatility_target": volatility.unsqueeze(0),
        }
