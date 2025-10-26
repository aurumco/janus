"""Sequence-based data processing strategy."""

from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from .base_strategy import DataProcessingStrategy


class SequenceProcessingStrategy(DataProcessingStrategy):
    """Strategy for converting tabular data into sequences."""

    def __init__(
        self,
        feature_columns: Optional[List[str]],
        target_column: Optional[str],
        sequence_length: int,
    ) -> None:
        """Initialize sequence processing strategy.

        Args:
            feature_columns: List of feature column names.
            target_column: Name of the target column.
            sequence_length: Length of each sequence window.
        """
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.sequence_length = sequence_length

    def validate(self, data: pd.DataFrame) -> bool:
        """Validate that data contains required columns.

        Args:
            data: Input DataFrame to validate.

        Returns:
            True if all required columns are present.
        """
        if self.feature_columns is None and self.target_column is None:
            return True
        required: List[str] = []
        if self.feature_columns is not None:
            required.extend(self.feature_columns)
        if self.target_column is not None:
            required.append(self.target_column)
        return all(col in data.columns for col in required)

    def process(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Convert DataFrame to sequences.

        Args:
            data: Input DataFrame with features and target.

        Returns:
            Tuple of (X, y) where X has shape (n_samples, seq_len, n_features)
            and y has shape (n_samples,).
        """
        if not self.validate(data):
            missing = set((self.feature_columns or []) + ([self.target_column] if self.target_column else [])) - set(data.columns)
            raise ValueError(f"Missing required columns: {missing}")

        if self.feature_columns is None:
            cols = [c for c in data.columns if c != 'asset_id']
            if self.target_column is not None:
                cols = [c for c in cols if c != self.target_column]
            feature_cols = cols
        else:
            feature_cols = [c for c in self.feature_columns if c != 'asset_id']

        features = data[feature_cols].values
        if self.target_column is None or self.target_column not in data.columns:
            targets = np.zeros(len(data), dtype=np.float32)
        else:
            targets = data[self.target_column].values

        X, y = self._create_sequences(features, targets)

        return X, y

    def _create_sequences(
        self,
        features: np.ndarray,
        targets: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences from features and targets (vectorized)."""
        n_timesteps = features.shape[0]
        if n_timesteps < self.sequence_length:
            raise ValueError(
                f"Not enough data points. Need at least {self.sequence_length}, got {n_timesteps}"
            )

        try:
            from numpy.lib.stride_tricks import sliding_window_view
            X_view = sliding_window_view(features, window_shape=(self.sequence_length, features.shape[1]))
            if X_view.ndim == 4:
                X = X_view[:, 0, :, :]
            else:
                X = sliding_window_view(features, window_shape=self.sequence_length)
                X = X.reshape(-1, self.sequence_length, features.shape[1])
        except Exception:
            n_samples = n_timesteps - self.sequence_length + 1
            X = np.zeros((n_samples, self.sequence_length, features.shape[1]), dtype=np.float32)
            for i in range(n_samples):
                X[i] = features[i:i + self.sequence_length]

        y = targets[self.sequence_length - 1: self.sequence_length - 1 + X.shape[0]]
        if y.ndim != 1:
            y = y.reshape(-1)
        y = y.astype(np.float32, copy=False)
        if X.dtype != np.float32:
            X = X.astype(np.float32, copy=False)
        return X, y
