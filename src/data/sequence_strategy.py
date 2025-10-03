"""Sequence-based data processing strategy."""

from typing import List, Tuple

import numpy as np
import pandas as pd

from .base_strategy import DataProcessingStrategy


class SequenceProcessingStrategy(DataProcessingStrategy):
    """Strategy for converting tabular data into sequences."""

    def __init__(
        self,
        feature_columns: List[str],
        target_column: str,
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
        required_columns = self.feature_columns + [self.target_column]
        return all(col in data.columns for col in required_columns)

    def process(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Convert DataFrame to sequences.

        Args:
            data: Input DataFrame with features and target.

        Returns:
            Tuple of (X, y) where X has shape (n_samples, seq_len, n_features)
            and y has shape (n_samples,).
        """
        if not self.validate(data):
            missing = set(self.feature_columns + [self.target_column]) - set(data.columns)
            raise ValueError(f"Missing required columns: {missing}")

        features = data[self.feature_columns].values
        targets = data[self.target_column].values

        X, y = self._create_sequences(features, targets)

        return X, y

    def _create_sequences(
        self,
        features: np.ndarray,
        targets: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences from features and targets.

        Args:
            features: Feature array of shape (n_timesteps, n_features).
            targets: Target array of shape (n_timesteps,).

        Returns:
            Tuple of (X, y) sequences.
        """
        n_samples = len(features) - self.sequence_length + 1

        if n_samples <= 0:
            raise ValueError(
                f"Not enough data points. Need at least {self.sequence_length}, "
                f"got {len(features)}"
            )

        X = np.zeros((n_samples, self.sequence_length, features.shape[1]))
        y = np.zeros(n_samples, dtype=np.int64)

        for i in range(n_samples):
            X[i] = features[i:i + self.sequence_length]
            y[i] = targets[i + self.sequence_length - 1]

        return X, y
