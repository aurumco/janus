import os
from typing import Tuple, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset

# Defaults; keep in sync with dataset builder
INPUT_WINDOW_CANDLES = 27
DATA_PATH_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "Dataset", "janus_m15_dataset.parquet")


def _chronological_split(df: pd.DataFrame, cutoff: str = "2025-07-01") -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp']).set_index('timestamp')
        else:
            raise ValueError("Dataset requires a DatetimeIndex or a 'timestamp' column")
    cutoff_ts = pd.to_datetime(cutoff)
    train_val = df[df.index < cutoff_ts]
    test = df[df.index >= cutoff_ts]
    return train_val, test


class TimeseriesDataset(Dataset):
    def __init__(self, parquet_path: str = DATA_PATH_DEFAULT, seq_len: int = INPUT_WINDOW_CANDLES, df: Optional[pd.DataFrame] = None):
        super().__init__()
        self.seq_len = int(seq_len)
        if df is None:
            df = pd.read_parquet(parquet_path)
        # Drop explicit timestamp column if present (index should carry time)
        if 'timestamp' in df.columns:
            df = df.drop(columns=['timestamp'])
        # Ensure target present
        if 'target' not in df.columns:
            raise ValueError("Dataset must contain a 'target' column")
        # Separate features/labels
        self.features = df.drop(columns=['target'])
        self.labels = df['target']
        # Cache numpy for speed
        self.X_np = self.features.values.astype('float32')
        self.y_np = self.labels.values.astype('int64')

    def __len__(self) -> int:
        return max(0, len(self.features) - self.seq_len)

    def __getitem__(self, idx: int):
        s = idx
        e = idx + self.seq_len
        X = self.X_np[s:e]                    # [seq_len, num_features]
        y = self.y_np[e]                      # label at t+seq_len
        return torch.from_numpy(X), torch.tensor(y, dtype=torch.long)


def create_dataloaders(
    parquet_path: str = DATA_PATH_DEFAULT,
    seq_len: int = INPUT_WINDOW_CANDLES,
    batch_size: int = 512,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """Create train/val/test dataloaders with chronological split.
    Train/Val: 2023-01-01 .. 2025-06-30 (90/10 split)
    Test:     >= 2025-07-01
    Returns loaders and inferred num_features
    """
    df = pd.read_parquet(parquet_path)
    # Normalize index/time
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp']).set_index('timestamp')
        else:
            raise ValueError("Dataset requires a DatetimeIndex or a 'timestamp' column")

    # Ensure 'target'
    if 'target' not in df.columns:
        raise ValueError("Dataset must contain a 'target' column")

    # Chronological split (and clamp ranges)
    train_val_df, test_df = _chronological_split(df, cutoff="2025-07-01")
    train_val_df = train_val_df[(train_val_df.index >= pd.to_datetime("2023-01-01")) & (train_val_df.index <= pd.to_datetime("2025-06-30"))]
    test_df = test_df[test_df.index >= pd.to_datetime("2025-07-01")]

    # Build full datasets
    full_train_val = TimeseriesDataset(df=train_val_df, seq_len=seq_len)
    test_dataset = TimeseriesDataset(df=test_df, seq_len=seq_len)

    # 90/10 split on train_val chronologically (no leakage)
    n = len(full_train_val)
    split_idx = int(n * 0.9)
    train_idx = list(range(0, split_idx))
    val_idx = list(range(split_idx, n))

    train_set = Subset(full_train_val, train_idx)
    val_set = Subset(full_train_val, val_idx)

    # Infer num_features from dataset
    num_features = full_train_val.features.shape[1]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader, num_features
