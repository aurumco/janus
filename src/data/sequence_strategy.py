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

    def process_gpu(self, parquet_path: str) -> Tuple[np.ndarray, np.ndarray]:  # type: ignore[name-defined]
        """GPU-accelerated chunked processing to avoid OOM.

        Args:
            parquet_path: Path to parquet file to read in chunks.

        Returns:
            Tuple (X, y) as NumPy arrays (backed by memmap for efficiency).
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

        n_samples = n_timesteps - self.sequence_length + 1
        
        tmpdir = "/kaggle/working" if os.path.exists("/kaggle/working") else tempfile.gettempdir()
        X_path = os.path.join(tmpdir, f"X_seq_{os.getpid()}.dat")
        y_path = os.path.join(tmpdir, f"y_seq_{os.getpid()}.dat")
        
        X_mmap = np.memmap(X_path, dtype=np.float32, mode='w+', shape=(n_samples, self.sequence_length, n_features))
        y_mmap = np.memmap(y_path, dtype=np.float32, mode='w+', shape=(n_samples,))

        read_chunk_size = 200000
        print(f"  GPU chunked processing: {n_timesteps} rows, read_chunk={read_chunk_size}")
        
        try:
            from tqdm import tqdm
            pbar = tqdm(total=n_samples, desc="  Building sequences", unit="seq")
        except Exception:
            pbar = None

        written = 0
        row_offset = 0
        
        for batch in parquet_file.iter_batches(batch_size=read_chunk_size):
            chunk_pd = batch.to_pandas()
            import numpy as _np
            numeric_cols = [c for c in chunk_pd.columns if _np.issubdtype(chunk_pd[c].dtype, _np.number)]
            present_feature_cols = [c for c in feature_cols if c in numeric_cols]
            missing_cols = [c for c in feature_cols if c not in present_feature_cols]
            
            for mc in missing_cols:
                chunk_pd[mc] = 0.0
            feat_pd = chunk_pd[feature_cols]

            if self.target_column and self.target_column in chunk_pd.columns and _np.issubdtype(chunk_pd[self.target_column].dtype, _np.number):
                cudf_pd = _np.concatenate
                chunk = cudf.from_pandas(feat_pd.join(chunk_pd[[self.target_column]]))
            else:
                chunk = cudf.from_pandas(feat_pd)
            
            gfeat = chunk[feature_cols].to_cupy().astype(cp.float32)
            
            if self.target_column is None or self.target_column not in chunk.columns:
                gtargets = cp.zeros(len(chunk), dtype=cp.float32)
            else:
                gtargets = chunk[self.target_column].to_cupy().astype(cp.float32)
            
            row_offset += len(chunk)
            
            chunk_len = len(chunk)
            if chunk_len < self.sequence_length:
                continue
                
            try:
                from cupy.lib.stride_tricks import as_strided
                n_win = chunk_len - self.sequence_length + 1
                shape = (n_win, self.sequence_length, n_features)
                strides = (gfeat.strides[0], gfeat.strides[0], gfeat.strides[1])
                X_chunk = as_strided(gfeat, shape=shape, strides=strides)
            except Exception:
                n_win = chunk_len - self.sequence_length + 1
                X_chunk = cp.empty((n_win, self.sequence_length, n_features), dtype=cp.float32)
                for i in range(n_win):
                    X_chunk[i] = gfeat[i:i + self.sequence_length]
            
            y_chunk = gtargets[self.sequence_length - 1: self.sequence_length - 1 + n_win]
            
            X_cpu = cp.asnumpy(X_chunk)
            y_cpu = cp.asnumpy(y_chunk)
            
            end_write = min(written + len(X_cpu), n_samples)
            actual_write = end_write - written
            X_mmap[written:end_write] = X_cpu[:actual_write]
            y_mmap[written:end_write] = y_cpu[:actual_write]
            written = end_write
            
            if pbar:
                pbar.update(actual_write)
            
            del gfeat, gtargets, X_chunk, y_chunk, X_cpu, y_cpu, chunk, chunk_pd
            cp.get_default_memory_pool().free_all_blocks()
            import gc
            gc.collect()
        
        if pbar:
            pbar.close()
        
        X_mmap.flush()
        y_mmap.flush()
        X = np.array(X_mmap[:written], dtype=np.float32)
        y = np.array(y_mmap[:written], dtype=np.float32)
        
        del X_mmap, y_mmap
        try:
            os.unlink(X_path)
            os.unlink(y_path)
        except Exception:
            pass
        
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
