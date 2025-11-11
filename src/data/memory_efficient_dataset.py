"""Memory-efficient datasets that read data on-the-fly from parquet files.

This module provides lazy-loading datasets that avoid loading the entire
dataset into RAM, instead reading only the required windows on demand.
"""

import gc
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


class ParquetStreamingDataset(Dataset):
    """Base class for streaming datasets that read from parquet on-the-fly.

    This dataset reads only the required rows from disk per __getitem__ call,
    dramatically reducing memory usage for large datasets.
    """

    def __init__(
        self,
        parquet_path: str,
        sequence_length: int,
        stride: int = 1,
    ) -> None:
        """Initialize streaming dataset.

        Args:
            parquet_path: Path to parquet file.
            sequence_length: Number of timesteps per sequence.
            stride: Step size between consecutive sequences (1=overlapping).
        """
        self.parquet_path = parquet_path
        self.sequence_length = sequence_length
        self.stride = stride

        pq_file = pq.ParquetFile(parquet_path)
        self.total_rows = pq_file.metadata.num_rows
        self.schema = pq_file.schema
        pq_file = None
        gc.collect()

        self.num_sequences = (self.total_rows - sequence_length) // stride + 1
        print(
            f"  Streaming dataset: {self.num_sequences:,} sequences "
            f"from {self.total_rows:,} rows (stride={stride})"
        )

    def __len__(self) -> int:
        return self.num_sequences

    def _read_window(self, start_row: int, end_row: int) -> pd.DataFrame:
        """Read a window of rows from parquet file efficiently.

        Args:
            start_row: Starting row index.
            end_row: Ending row index (exclusive).

        Returns:
            DataFrame with requested rows.
        """
        try:
            df = pd.read_parquet(
                self.parquet_path,
                engine='pyarrow',
            )
            result = df.iloc[start_row:end_row].copy()
            del df
            gc.collect()
            return result
        except Exception as e:
            print(f"Warning: Error reading rows {start_row}-{end_row}: {e}")
            df = pd.read_parquet(self.parquet_path)
            result = df.iloc[start_row:end_row].copy()
            del df
            gc.collect()
            return result


class MemoryEfficientPretrainDataset(ParquetStreamingDataset):
    """Memory-efficient SSL pre-training dataset with on-the-fly masking."""

    def __init__(
        self,
        parquet_path: str,
        sequence_length: int,
        masking_ratio: float = 0.15,
        volatility_lookahead: int = 60,
        smart_masking_prob: float = 0.4,
        stride: int = 1,
    ) -> None:
        """Initialize memory-efficient pretrain dataset.

        Args:
            parquet_path: Path to parquet file.
            sequence_length: Sequence length for windows.
            masking_ratio: Ratio of tokens to mask.
            volatility_lookahead: Timesteps ahead for volatility target.
            smart_masking_prob: Probability of using smart masking.
            stride: Stride between windows (default 1 for overlap).
        """
        super().__init__(parquet_path, sequence_length, stride)
        self.masking_ratio = masking_ratio
        self.volatility_lookahead = volatility_lookahead
        self.smart_masking_prob = smart_masking_prob

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sequence with on-the-fly masking.

        Args:
            idx: Sequence index.

        Returns:
            Dictionary with masked sequence and targets.
        """
        start_row = idx * self.stride
        end_row = start_row + self.sequence_length

        chunk = self._read_window(start_row, end_row)

        exclude_cols = ["timestamp", "target", "asset_id"]
        feature_cols = [c for c in chunk.columns if c not in exclude_cols]
        features = chunk[feature_cols].values.astype(np.float32)

        if "asset_id" in chunk.columns:
            asset_id = int(chunk["asset_id"].iloc[-1])
        else:
            asset_id = 0

        original_sequence = torch.from_numpy(features)

        mask_binary = self._generate_smart_mask(original_sequence)

        masked_sequence = original_sequence.clone()
        masked_sequence[mask_binary] = 0.0

        future_end = min(end_row + self.volatility_lookahead, self.total_rows)
        if future_end > end_row + 5:
            future_chunk = self._read_window(end_row, future_end)
            close_col = feature_cols[min(3, len(feature_cols) - 1)]
            if len(future_chunk) > 5 and close_col in future_chunk.columns:
                prices = future_chunk[close_col].values.astype(np.float32)
                returns = np.diff(prices)
                volatility = float(np.std(returns) + 1e-6)
            else:
                volatility = 0.0
            future_chunk = None
        else:
            volatility = 0.0
        
        volatility = volatility * 100.0

        chunk = None
        features = None
        gc.collect()

        return {
            "input_sequence": masked_sequence,
            "mask_binary": mask_binary,
            "original_sequence": original_sequence,
            "volatility_target": torch.tensor([volatility], dtype=torch.float32),
            "asset_id": torch.tensor(asset_id, dtype=torch.long),
        }

    def _generate_smart_mask(self, sequence: torch.Tensor) -> torch.Tensor:
        """Generate smart mask for sequence.

        Args:
            sequence: Input sequence tensor (seq_len, n_features).

        Returns:
            Boolean mask tensor.
        """
        seq_len = sequence.size(0)
        mask_binary = torch.zeros(seq_len, dtype=torch.bool)

        use_smart = np.random.random() < self.smart_masking_prob

        if use_smart and sequence.size(1) >= 4:
            price_features = sequence[:, :4]
            price_volatility = torch.std(price_features, dim=1)
            high_vol_threshold = torch.quantile(price_volatility, 0.8)
            high_vol_idx = (price_volatility > high_vol_threshold).nonzero(as_tuple=True)[0]

            if len(high_vol_idx) > 0:
                m = high_vol_idx[np.random.randint(len(high_vol_idx))]
                length = np.random.randint(1, 4)
                end_idx = min(m + length, seq_len)
                mask_binary[m:end_idx] = True
        else:
            num_mask = max(1, int(seq_len * self.masking_ratio))
            mask_idx = np.random.choice(seq_len, size=num_mask, replace=False)
            mask_binary[mask_idx] = True

        return mask_binary


class MemoryEfficientFinetuneDataset(ParquetStreamingDataset):
    """Memory-efficient fine-tuning dataset for regression."""

    def __init__(
        self,
        parquet_path: str,
        feature_columns: List[str],
        target_column: str,
        sequence_length: int,
        stride: int = 1,
    ) -> None:
        """Initialize memory-efficient finetune dataset.

        Args:
            parquet_path: Path to parquet file.
            feature_columns: List of feature column names.
            target_column: Target column name.
            sequence_length: Sequence length.
            stride: Stride between windows.
        """
        super().__init__(parquet_path, sequence_length, stride)
        self.feature_columns = feature_columns
        self.target_column = target_column

    def __getitem__(self, idx: int) -> tuple:
        """Get a single sequence and target.

        Args:
            idx: Sequence index.

        Returns:
            Tuple of (features, target).
        """
        start_row = idx * self.stride
        end_row = start_row + self.sequence_length

        chunk = self._read_window(start_row, end_row)

        features = chunk[self.feature_columns].values.astype(np.float32)
        target = float(chunk[self.target_column].iloc[-1])

        features_tensor = torch.from_numpy(features)
        target_tensor = torch.tensor([target], dtype=torch.float32)

        chunk = None
        features = None
        gc.collect()

        return features_tensor, target_tensor
