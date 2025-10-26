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
        # If feature_columns is None, we'll derive features dynamically; skip strict validation.
        if self.feature_columns is None and self.target_column is None:
            return True
        # Build required columns list based on provided inputs
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
            # Compute missing only from provided columns
            missing = set((self.feature_columns or []) + ([self.target_column] if self.target_column else [])) - set(data.columns)
            raise ValueError(f"Missing required columns: {missing}")

        # Determine feature columns dynamically if not provided
        if self.feature_columns is None:
            cols = [c for c in data.columns if c != 'asset_id']
            if self.target_column is not None:
                cols = [c for c in cols if c != self.target_column]
            feature_cols = cols
        else:
            # Always exclude 'asset_id' from features if accidentally included
            feature_cols = [c for c in self.feature_columns if c != 'asset_id']

        features = data[feature_cols].values
        # If no target column provided (pretrain mode), create a dummy target vector
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

        X = np.zeros((n_samples, self.sequence_length, features.shape[1]), dtype=np.float32)
        y = np.zeros(n_samples, dtype=np.float32)

        for i in range(n_samples):
            X[i] = features[i:i + self.sequence_length]
            y[i] = targets[i + self.sequence_length - 1]

        return X, y
