"""PyTorch dataset for fine-tuning regression task."""

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class FineTuneDataset(Dataset):
    """PyTorch dataset for cryptocurrency regression fine-tuning."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """Initialize fine-tuning dataset.

        Args:
            X: Feature sequences of shape (n_samples, seq_len, n_features).
            y: Target values of shape (n_samples,) for regression.
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

        if self.y.ndim == 1:
            self.y = self.y.unsqueeze(1)

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
