"""PyTorch dataset for fine-tuning regression task."""

from typing import Tuple, Dict, Union

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
        # Assume asset_id is the last column of features if present
        # We'll split it during getitem
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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary containing input_sequence, asset_id, and targets.
        """
        x_full = self.X[idx]

        # Separate features and asset_id
        # x_core: (seq_len, n_features-1)
        # asset_id: scalar (taken from first timestep, assumed constant)
        x_core = x_full[:, :-1]
        asset_id = x_full[0, -1].long()

        return {
            "input_sequence": x_core,
            "asset_id": asset_id,
            "targets": self.y[idx]
        }


class LazyFineTuneDataset(Dataset):
    """Memory-efficient dataset that slices windows on-the-fly.

    This avoids materializing the full (N, L, F) array, reducing memory usage
    by a factor of L (sequence length).
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        sequence_length: int,
    ) -> None:
        """Initialize lazy fine-tuning dataset.

        Args:
            features: Continuous feature array of shape (n_timesteps, n_features).
            targets: Continuous target array of shape (n_timesteps,).
            sequence_length: Length of each sequence window.
        """
        # Keep data as FloatTensor for fast slicing
        # If passed as tensor, use as is; if numpy, convert
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)

        self.sequence_length = sequence_length

        # Calculate number of valid sequences
        # We need indices i such that i + sequence_length <= n_timesteps
        self.n_samples = max(0, len(self.features) - sequence_length + 1)

        if self.targets.ndim == 1:
            self.targets = self.targets.unsqueeze(1)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample using on-the-fly slicing.

        Args:
            idx: Sample index.

        Returns:
            Dictionary containing input_sequence, asset_id, and targets.
        """
        if idx >= self.n_samples:
             raise IndexError(f"Index {idx} out of bounds for size {self.n_samples}")

        # Slice features: (L, F)
        # Slicing a tensor returns a view, but creating the batch later copies it.
        # This is O(1) here.
        x_full = self.features[idx : idx + self.sequence_length]

        # Separate features and asset_id
        # x_core: (L, F-1)
        x_core = x_full[:, :-1]

        # asset_id: scalar (taken from first timestep of window)
        asset_id = x_full[0, -1].long()

        # Get target corresponding to the end of the window
        # For window [t, t+L], target is at t+L-1
        y = self.targets[idx + self.sequence_length - 1]

        return {
            "input_sequence": x_core,
            "asset_id": asset_id,
            "targets": y
        }
