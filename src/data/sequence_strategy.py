"""Sequence-based data processing strategy."""

import os
import tempfile
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .base_strategy import DataProcessingStrategy


class SequenceProcessingStrategy(DataProcessingStrategy):
    """Strategy for converting tabular data into sequences."""

    def __init__(
        self,
        feature_columns: Optional[List[str]],
        target_column: Optional[Union[str, List[str]]],
        sequence_length: int,
    ) -> None:
        """Initialize sequence processing strategy.

        Args:
            feature_columns: List of feature column names.
            target_column: Name of the target column(s).
            sequence_length: Length of each sequence window.
        """
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.sequence_length = sequence_length
        self._asset_ids_out: Optional[np.ndarray] = None
        self._asset_ids_full: Optional[np.ndarray] = None

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
            if isinstance(self.target_column, list):
                required.extend(self.target_column)
            else:
                required.append(self.target_column)
        return all(col in data.columns for col in required)

    def process(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Convert DataFrame to sequences.

        Args:
            data: Input DataFrame with features and target.

        Returns:
            Tuple of (X, y) where X has shape (n_samples, seq_len, n_features)
            and y has shape (n_samples,) or (n_samples, output_dim).
        """
        if not self.validate(data):
            # Helper to safely handle None/List for missing set calculation
            cols = self.feature_columns or []
            if self.target_column:
                cols.extend(
                    self.target_column
                    if isinstance(self.target_column, list)
                    else [self.target_column]
                )
            missing = set(cols) - set(data.columns)
            raise ValueError(f"Missing required columns: {missing}")

        if self.feature_columns is None:
            cols = [c for c in data.columns if c not in ("asset_id", "timestamp")]
            if self.target_column is not None:
                if isinstance(self.target_column, list):
                    cols = [c for c in cols if c not in self.target_column]
                else:
                    cols = [c for c in cols if c != self.target_column]
            feature_cols = cols
        else:
            feature_cols = [
                c
                for c in self.feature_columns
                if c not in ("asset_id", "timestamp")
            ]

        features = data[feature_cols].values

        # APPEND ASSET ID IF AVAILABLE
        # The FineTuneDataset assumes asset_id is the last column of X.
        if "asset_id" in data.columns:
            asset_ids = data["asset_id"].values.reshape(-1, 1)
            features = np.hstack([features, asset_ids])

        if self.target_column is None:
            targets = np.zeros(len(data), dtype=np.float32)
        elif isinstance(self.target_column, list):
            targets = data[self.target_column].values
        elif self.target_column not in data.columns:
            targets = np.zeros(len(data), dtype=np.float32)
        else:
            targets = data[self.target_column].values

        X, y = self._create_sequences(features, targets)

        if "asset_id" in data.columns:
            asset_ids_raw = data["asset_id"].values
            # Extract asset_ids corresponding to the last step of each window
            self._asset_ids_out = asset_ids_raw[
                self.sequence_length - 1 : self.sequence_length - 1 + X.shape[0]
            ]
        else:
            self._asset_ids_out = None

        return X, y

    def process_gpu(self, parquet_path: str) -> Tuple[str, str, int, int]:
        """GPU-accelerated chunked processing writing base features memmap.

        Args:
            parquet_path: Path to parquet file to read in chunks.

        Returns:
            Tuple (features_path, asset_ids_path, n_timesteps, n_features).
        """
        import cudf
        import cupy as cp
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(parquet_path)
        total_rows = parquet_file.metadata.num_rows
        schema_cols = [
            parquet_file.schema[i].name for i in range(len(parquet_file.schema))
        ]

        if self.feature_columns is None:
            cols = [
                c for c in schema_cols if c not in ("asset_id", "timestamp")
            ]
            if self.target_column is not None:
                if isinstance(self.target_column, list):
                    cols = [c for c in cols if c not in self.target_column]
                else:
                    if self.target_column in cols:
                        cols.remove(self.target_column)
            feature_cols = cols
        else:
            feature_cols = [
                c
                for c in self.feature_columns
                if c in schema_cols and c not in ("asset_id", "timestamp")
            ]

        if len(feature_cols) == 0:
            raise ValueError("No valid feature columns available on GPU path")

        n_timesteps = total_rows
        n_features = len(feature_cols)
        if n_timesteps < self.sequence_length:
            raise ValueError(
                f"Not enough data points. Need at least {self.sequence_length}, got {n_timesteps}"
            )

        tmpdir = (
            "/kaggle/working"
            if os.path.exists("/kaggle/working")
            else tempfile.gettempdir()
        )
        features_path = os.path.join(tmpdir, f"features_{os.getpid()}.dat")
        asset_ids_path = os.path.join(tmpdir, f"asset_ids_{os.getpid()}.dat")

        features_mmap = np.memmap(
            features_path,
            dtype=np.float32,
            mode="w+",
            shape=(n_timesteps, n_features),
        )
        asset_ids_mmap = np.memmap(
            asset_ids_path, dtype=np.int64, mode="w+", shape=(n_timesteps,)
        )

        read_chunk_size = 500000
        print(f"  GPU chunked read: {n_timesteps} rows, chunk={read_chunk_size}")

        try:
            from tqdm import tqdm

            pbar = tqdm(total=n_timesteps, desc="  Reading features", unit="row")
        except ImportError:
            pbar = None

        written = 0

        for batch in parquet_file.iter_batches(batch_size=read_chunk_size):
            chunk_pd = batch.to_pandas()
            # Optimization: Use built-in pandas/numpy checks
            numeric_cols = chunk_pd.select_dtypes(include=[np.number]).columns
            present_cols = [c for c in feature_cols if c in numeric_cols]
            missing_cols = [c for c in feature_cols if c not in present_cols]

            if missing_cols:
                chunk_pd[missing_cols] = 0.0

            feat_pd = chunk_pd[feature_cols]
            chunk = cudf.from_pandas(feat_pd)
            gfeat = chunk.to_cupy().astype(cp.float32)
            feat_cpu = cp.asnumpy(gfeat)

            chunk_len = len(feat_cpu)
            end_write = min(written + chunk_len, n_timesteps)
            actual = end_write - written
            features_mmap[written:end_write] = feat_cpu[:actual]

            if "asset_id" in chunk_pd.columns:
                aids = chunk_pd["asset_id"].values.astype(np.int64)
            else:
                aids = np.zeros(chunk_len, dtype=np.int64)
            asset_ids_mmap[written:end_write] = aids[:actual]

            written = end_write
            if pbar:
                pbar.update(actual)

            # Cleanup GPU memory without full GC
            del gfeat, feat_cpu, chunk, chunk_pd, aids
            cp.get_default_memory_pool().free_all_blocks()
            # Removed gc.collect() as per optimization guidelines

        if pbar:
            pbar.close()

        features_mmap.flush()
        asset_ids_mmap.flush()
        del features_mmap, asset_ids_mmap

        return features_path, asset_ids_path, written, n_features

    def _create_sequences(
        self,
        features: np.ndarray,
        targets: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences from features and targets using strided views.

        Args:
            features: Input features array of shape (n_samples, n_features).
            targets: Input targets array of shape (n_samples,) or (n_samples, output_dim).

        Returns:
            Tuple of (X, y) where:
                X: Windowed features (n_windows, seq_len, n_features)
                y: Targets corresponding to the last step of each window.
        """
        n_timesteps = features.shape[0]
        if n_timesteps < self.sequence_length:
            raise ValueError(
                f"Not enough data points. Need at least {self.sequence_length}, got {n_timesteps}"
            )

        # Use efficient sliding_window_view (numpy >= 1.20)
        # For a 2D array (N, F) with window (L, F), returns (N-L+1, 1, L, F)
        # The '1' dim comes because we match the second dimension fully.
        X_view = sliding_window_view(
            features, window_shape=(self.sequence_length, features.shape[1])
        )

        # Squeeze the singleton dimension corresponding to features
        X = X_view.squeeze(axis=1)

        # Extract targets corresponding to the last step of each window
        # For window [t, t+L], the target is at index t+L-1
        # The number of windows is X.shape[0]
        y = targets[
            self.sequence_length - 1 : self.sequence_length - 1 + X.shape[0]
        ]

        # Ensure targets are float32
        y = y.astype(np.float32, copy=False)

        # Flatten target if it's strictly 1D (vector), preserve if multi-dimensional
        if y.ndim == 1:
            y = y.reshape(-1)

        # Ensure features are float32.
        # Note: casting a view might force a copy, which is inevitable if we need float32
        # and input is double.
        if X.dtype != np.float32:
            X = X.astype(np.float32, copy=False)

        return X, y

    def process_lazy(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Convert DataFrame to raw features and targets for lazy windowing.

        Args:
            data: Input DataFrame with features and target.

        Returns:
            Tuple of (features, targets) where:
            - features has shape (n_timesteps, n_features)
            - targets has shape (n_timesteps,) or (n_timesteps, output_dim)
        """
        if not self.validate(data):
            cols = self.feature_columns or []
            if self.target_column:
                cols.extend(
                    self.target_column
                    if isinstance(self.target_column, list)
                    else [self.target_column]
                )
            missing = set(cols) - set(data.columns)
            raise ValueError(f"Missing required columns: {missing}")

        if self.feature_columns is None:
            cols = [c for c in data.columns if c not in ("asset_id", "timestamp")]
            if self.target_column is not None:
                if isinstance(self.target_column, list):
                    cols = [c for c in cols if c not in self.target_column]
                else:
                    if self.target_column in cols:
                        cols = [c for c in cols if c != self.target_column]
            feature_cols = cols
        else:
            feature_cols = [
                c
                for c in self.feature_columns
                if c not in ("asset_id", "timestamp")
            ]

        features = data[feature_cols].values.astype(np.float32)

        # APPEND ASSET ID IF AVAILABLE
        # The LazyFineTuneDataset assumes asset_id is the last column of X.
        if "asset_id" in data.columns:
            asset_ids = (
                data["asset_id"].values.astype(np.float32).reshape(-1, 1)
            )
            features = np.hstack([features, asset_ids])

        if self.target_column is None:
            targets = np.zeros(len(data), dtype=np.float32)
        elif isinstance(self.target_column, list):
            targets = data[self.target_column].values.astype(np.float32)
        elif self.target_column not in data.columns:
            targets = np.zeros(len(data), dtype=np.float32)
        else:
            targets = data[self.target_column].values.astype(np.float32)

        if "asset_id" in data.columns:
            self._asset_ids_full = data["asset_id"].values

            if len(features) >= self.sequence_length:
                start_idx = self.sequence_length - 1
                end_idx = start_idx + (
                    len(features) - self.sequence_length + 1
                )
                self._asset_ids_out = self._asset_ids_full[start_idx:end_idx]
            else:
                self._asset_ids_out = None
        else:
            self._asset_ids_out = None

        return features, targets
