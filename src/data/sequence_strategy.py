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

        # Determine feature columns excluding identifiers/target
        if self.feature_columns is None:
            cols = [c for c in data.columns if c not in ('asset_id', 'timestamp')]
            if self.target_column is not None:
                cols = [c for c in cols if c != self.target_column]
            feature_cols = cols
        else:
            feature_cols = [c for c in self.feature_columns if c not in ('asset_id', 'timestamp')]

        # MEMORY-EFFICIENT: Assume data is already sorted chronologically in parquet
        # DO NOT sort or groupby - these operations copy the entire DataFrame!
        
        # Extract features and asset_ids as numpy arrays directly (no copy)
        features = data[feature_cols].values
        if self.target_column is None or self.target_column not in data.columns:
            targets = np.zeros(len(data), dtype=np.float32)
        else:
            targets = data[self.target_column].values
        
        # Create sequences from the full dataset (chronological order preserved)
        X, y = self._create_sequences(features, targets)
        
        # If asset_id exists, extract it aligned to sequence windows
        if 'asset_id' in data.columns:
            # Get asset_id at the END of each sequence window (last timestep)
            asset_ids_raw = data['asset_id'].values
            # Align to sequence ends: window i ends at position (i + sequence_length - 1)
            self._asset_ids_out = asset_ids_raw[self.sequence_length - 1 : self.sequence_length - 1 + X.shape[0]]
        else:
            self._asset_ids_out = None
        
        return X, y

    def process_gpu(self, parquet_path: str) -> Tuple[str, str, int, int]:
        """GPU-accelerated chunked processing writing base features memmap.

        Args:
            parquet_path: Path to parquet file to read in chunks.

        Returns:
            Tuple (features_path, asset_ids_path, n_timesteps, n_features) for streaming dataset.
        """
        import cupy as cp
        import cudf
        import tempfile
        import os
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(parquet_path)
        total_rows = parquet_file.metadata.num_rows
        schema_cols = [parquet_file.schema[i].name for i in range(len(parquet_file.schema))]
        
        if self.feature_columns is None:
            cols = [c for c in schema_cols if c not in ('asset_id', 'timestamp')]
            if self.target_column is not None and self.target_column in cols:
                cols.remove(self.target_column)
            feature_cols = cols
        else:
            feature_cols = [c for c in self.feature_columns if c in schema_cols and c not in ('asset_id', 'timestamp')]

        if len(feature_cols) == 0:
            raise ValueError("No valid feature columns available on GPU path")

        n_timesteps = total_rows
        n_features = len(feature_cols)
        if n_timesteps < self.sequence_length:
            raise ValueError(
                f"Not enough data points. Need at least {self.sequence_length}, got {n_timesteps}"
            )

        tmpdir = "/kaggle/working" if os.path.exists("/kaggle/working") else tempfile.gettempdir()
        features_path = os.path.join(tmpdir, f"features_{os.getpid()}.dat")
        asset_ids_path = os.path.join(tmpdir, f"asset_ids_{os.getpid()}.dat")
        
        features_mmap = np.memmap(features_path, dtype=np.float32, mode='w+', shape=(n_timesteps, n_features))
        asset_ids_mmap = np.memmap(asset_ids_path, dtype=np.int64, mode='w+', shape=(n_timesteps,))

        read_chunk_size = 500000
        print(f"  GPU chunked read: {n_timesteps} rows, chunk={read_chunk_size}")
        
        try:
            from tqdm import tqdm
            pbar = tqdm(total=n_timesteps, desc="  Reading features", unit="row")
        except Exception:
            pbar = None

        written = 0
        
        for batch in parquet_file.iter_batches(batch_size=read_chunk_size):
            chunk_pd = batch.to_pandas()
            import numpy as _np
            numeric_cols = [c for c in chunk_pd.columns if _np.issubdtype(chunk_pd[c].dtype, _np.number)]
            present_cols = [c for c in feature_cols if c in numeric_cols]
            missing_cols = [c for c in feature_cols if c not in present_cols]
            for mc in missing_cols:
                chunk_pd[mc] = 0.0
            feat_pd = chunk_pd[feature_cols]
            chunk = cudf.from_pandas(feat_pd)
            gfeat = chunk.to_cupy().astype(cp.float32)
            feat_cpu = cp.asnumpy(gfeat)
            
            chunk_len = len(feat_cpu)
            end_write = min(written + chunk_len, n_timesteps)
            actual = end_write - written
            features_mmap[written:end_write] = feat_cpu[:actual]
            
            if 'asset_id' in chunk_pd.columns:
                aids = chunk_pd['asset_id'].values.astype(np.int64)
            else:
                aids = np.zeros(chunk_len, dtype=np.int64)
            asset_ids_mmap[written:end_write] = aids[:actual]
            
            written = end_write
            if pbar:
                pbar.update(actual)
            
            del gfeat, feat_cpu, chunk, chunk_pd, aids
            cp.get_default_memory_pool().free_all_blocks()
            import gc
            gc.collect()
        
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
