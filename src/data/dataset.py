"""PyTorch dataset implementation for Bitcoin trend classification."""

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class BitcoinTrendDataset(Dataset):
    """PyTorch dataset for Bitcoin trend sequences."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """Initialize dataset.

        Args:
            X: Feature sequences of shape (n_samples, seq_len, n_features).
            y: Target labels of shape (n_samples,).
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self) -> int:
        """Get dataset length.

        Returns:
            Number of samples in the dataset.
        """
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (features, label).
        """
        return self.X[idx], self.y[idx]
