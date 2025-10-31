"""Data loading and preparation utilities."""

from pathlib import Path
from typing import Dict, Optional
import gc
import time

import pandas as pd
from torch.utils.data import DataLoader

try:
    from src.utils.logger import logger
except Exception:
    logger = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):  # type: ignore
        return iterable if iterable is not None else range(0)

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

from .base_strategy import DataProcessingStrategy
from .finetune_dataset import FineTuneDataset
from .pretrain_dataset import PretrainDataset
from .memory_efficient_dataset import (
    MemoryEfficientPretrainDataset,
    MemoryEfficientFinetuneDataset,
)


class DataLoaderFactory:
    """Factory for creating data loaders using the Strategy Pattern."""

    def __init__(
        self,
        data_path: str,
        processing_strategy: DataProcessingStrategy,
        mode: str = "finetune",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle_train: bool = True,
        random_seed: int = 87,
        masking_ratio: float = 0.15,
        volatility_lookahead: int = 60,
        sequence_length: int = 96,
        smart_masking_prob: float = 0.4,
        cross_asset_masking_prob: float = 0.3,
        use_gpu_preprocess: bool = True,
        use_streaming_fallback: bool = False,
        verbose: bool = False,
        stride: int = 4,
    ) -> None:
        """Initialize data loader factory.

        Args:
            data_path: Path to the parquet data file.
            processing_strategy: Strategy for processing data.
            mode: Training mode ('pretrain' or 'finetune').
            train_ratio: Proportion of data for training.
            val_ratio: Proportion of data for validation.
            test_ratio: Proportion of data for testing.
            batch_size: Batch size for data loaders.
            num_workers: Number of worker processes.
            shuffle_train: Whether to shuffle training data.
            random_seed: Random seed for reproducibility.
            masking_ratio: Masking ratio for pre-training.
            volatility_lookahead: Lookahead for volatility prediction.
            sequence_length: Length of input sequences.
        """
        self.data_path = Path(data_path)
        self.processing_strategy = processing_strategy
        self.mode = mode
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train
        self.random_seed = random_seed
        self.masking_ratio = masking_ratio
        self.volatility_lookahead = volatility_lookahead
        self.sequence_length = sequence_length
        self.smart_masking_prob = smart_masking_prob
        self.cross_asset_masking_prob = cross_asset_masking_prob
        self.use_gpu_preprocess = use_gpu_preprocess
        self.use_streaming_fallback = use_streaming_fallback
        self.verbose = verbose
        self.stride = stride

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Train, val, and test ratios must sum to 1.0")

    def create_data_loaders(self) -> Dict[str, DataLoader]:
        """Create train, validation, and test data loaders using time-based split.

        Returns:
            Dictionary with 'train', 'val', and 'test' DataLoaders.
        """
        start_time = time.time()
        def mem_info(prefix: str) -> None:
            if psutil and self.verbose:
                mi = psutil.Process().memory_info()
                if logger:
                    logger.info(f"{prefix}: RSS={mi.rss/1e9:.2f}GB", indent=2)

        try:
            if self.verbose and logger:
                logger.info("Reading parquet file", indent=1)
            data = None
            gdata = None
            if self.use_gpu_preprocess:
                try:
                    import cudf  # type: ignore
                    gdata = str(self.data_path)
                    if self.verbose and logger:
                        logger.info("Using cuDF GPU processing", indent=2)
                except Exception as ge:
                    if self.verbose and logger:
                        logger.warning(f"GPU read failed, using CPU: {type(ge).__name__}", indent=2)
                    gdata = None
            if gdata is None:
                data = pd.read_parquet(self.data_path, engine="pyarrow")
                if self.verbose and logger:
                    logger.info(f"Loaded {data.shape[0]} rows", indent=2)

            if self.verbose and logger:
                logger.info("Processing sequences", indent=1)
            if gdata is not None and self.use_gpu_preprocess:
                features_path, asset_ids_path, n_timesteps, n_features = self.processing_strategy.process_gpu(gdata)
                if self.verbose and logger:
                    logger.info(f"Features: {n_timesteps} × {n_features}", indent=2)
                    mem_info("After GPU process")
                X = None
                y = None
            else:
                X, y = self.processing_strategy.process(data)
                if self.verbose and logger:
                    logger.info(f"Sequences: {X.shape[0]} samples", indent=2)
                features_path = None
                asset_ids_path = None
                n_timesteps = None
                n_features = None

            if self.verbose and logger:
                logger.info("Splitting train/val/test", indent=1)
            if X is not None:
                n_samples = len(X)
            else:
                n_samples = n_timesteps - self.sequence_length + 1
            train_end = int(n_samples * self.train_ratio)
            val_end = int(n_samples * (self.train_ratio + self.val_ratio))

            if X is not None:
                X_train = X[:train_end]
                y_train = y[:train_end]
                X_val = X[train_end:val_end]
                y_val = y[train_end:val_end]
                X_test = X[val_end:]
                y_test = y[val_end:]
            else:
                X_train = X_val = X_test = None
                y_train = y_val = y_test = None

            if X is not None:
                asset_ids = None
                if self.mode == "pretrain" and data is not None and "asset_id" in data.columns:
                    import numpy as np
                    asset_ids = data["asset_id"].values.astype(np.int64)
                    asset_ids = asset_ids[self.sequence_length - 1: self.sequence_length - 1 + n_samples]
                    asset_ids_train = asset_ids[:train_end]
                    asset_ids_val = asset_ids[train_end:val_end]
                    asset_ids_test = asset_ids[val_end:]
                else:
                    import numpy as np
                    asset_ids_train = np.zeros(len(X_train), dtype=np.int64)
                    asset_ids_val = np.zeros(len(X_val), dtype=np.int64)
                    asset_ids_test = np.zeros(len(X_test), dtype=np.int64)
            else:
                asset_ids_train = asset_ids_val = asset_ids_test = None

            if self.verbose and logger:
                logger.info("Building datasets", indent=1)
            if self.mode == "pretrain":
                if self.use_streaming_fallback:
                    if self.verbose and logger:
                        logger.info("Using streaming mode", indent=2)
                    full_dataset = MemoryEfficientPretrainDataset(
                        parquet_path=str(self.data_path),
                        sequence_length=self.sequence_length,
                        masking_ratio=self.masking_ratio,
                        volatility_lookahead=self.volatility_lookahead,
                        stride=1,
                    )
                    n_samples = len(full_dataset)
                    train_end = int(n_samples * self.train_ratio)
                    val_end = int(n_samples * (self.train_ratio + self.val_ratio))
                    
                    from torch.utils.data import Subset
                    train_dataset = Subset(full_dataset, range(0, train_end))
                    val_dataset = Subset(full_dataset, range(train_end, val_end))
                    test_dataset = Subset(full_dataset, range(val_end, n_samples))
                elif features_path is not None:
                    # Local import to avoid environment-specific import errors
                    try:
                        from .pretrain_window_dataset import PretrainWindowDataset  # type: ignore
                    except Exception:
                        try:
                            from src.data.pretrain_window_dataset import PretrainWindowDataset  # type: ignore
                        except Exception as imp_err:
                            raise ImportError(
                                f"Failed to import PretrainWindowDataset: {imp_err}"
                            )
                    train_dataset = PretrainWindowDataset(
                        features_memmap_path=features_path,
                        asset_ids_memmap_path=asset_ids_path,
                        n_timesteps=n_timesteps,
                        n_features=n_features,
                        sequence_length=self.sequence_length,
                        start_index=0,
                        end_index=train_end,
                        masking_ratio=self.masking_ratio,
                        volatility_lookahead=self.volatility_lookahead,
                        smart_masking_prob=self.smart_masking_prob,
                        cross_asset_masking_prob=self.cross_asset_masking_prob,
                        stride=self.stride,
                    )
                    val_dataset = PretrainWindowDataset(
                        features_memmap_path=features_path,
                        asset_ids_memmap_path=asset_ids_path,
                        n_timesteps=n_timesteps,
                        n_features=n_features,
                        sequence_length=self.sequence_length,
                        start_index=train_end,
                        end_index=val_end,
                        masking_ratio=self.masking_ratio,
                        volatility_lookahead=self.volatility_lookahead,
                        smart_masking_prob=self.smart_masking_prob,
                        cross_asset_masking_prob=self.cross_asset_masking_prob,
                        stride=self.stride,
                        deterministic=True,
                    )
                    test_dataset = PretrainWindowDataset(
                        features_memmap_path=features_path,
                        asset_ids_memmap_path=asset_ids_path,
                        n_timesteps=n_timesteps,
                        n_features=n_features,
                        sequence_length=self.sequence_length,
                        start_index=val_end,
                        end_index=n_samples,
                        masking_ratio=self.masking_ratio,
                        volatility_lookahead=self.volatility_lookahead,
                        smart_masking_prob=self.smart_masking_prob,
                        cross_asset_masking_prob=self.cross_asset_masking_prob,
                        stride=self.stride,
                        deterministic=True,
                    )
                else:
                    train_dataset = PretrainDataset(
                        X_train,
                        asset_ids=asset_ids_train,
                        sequence_length=self.sequence_length,
                        masking_ratio=self.masking_ratio,
                        volatility_lookahead=self.volatility_lookahead,
                        smart_masking_prob=self.smart_masking_prob,
                        cross_asset_masking_prob=self.cross_asset_masking_prob,
                    )
                    val_dataset = PretrainDataset(
                        X_val,
                        asset_ids=asset_ids_val,
                        sequence_length=self.sequence_length,
                        masking_ratio=self.masking_ratio,
                        volatility_lookahead=self.volatility_lookahead,
                        smart_masking_prob=self.smart_masking_prob,
                        cross_asset_masking_prob=self.cross_asset_masking_prob,
                    )
                    test_dataset = PretrainDataset(
                        X_test,
                        asset_ids=asset_ids_test,
                        sequence_length=self.sequence_length,
                        masking_ratio=self.masking_ratio,
                        volatility_lookahead=self.volatility_lookahead,
                        smart_masking_prob=self.smart_masking_prob,
                        cross_asset_masking_prob=self.cross_asset_masking_prob,
                    )
            else:
                if self.use_streaming_fallback:
                    print("  Using streaming fallback mode (direct parquet read)")
                    full_dataset = MemoryEfficientFinetuneDataset(
                        parquet_path=str(self.data_path),
                        feature_columns=self.processing_strategy.feature_columns,
                        target_column=self.processing_strategy.target_column,
                        sequence_length=self.sequence_length,
                        stride=1,
                    )
                    n_samples = len(full_dataset)
                    train_end = int(n_samples * self.train_ratio)
                    val_end = int(n_samples * (self.train_ratio + self.val_ratio))
                    
                    from torch.utils.data import Subset
                    train_dataset = Subset(full_dataset, range(0, train_end))
                    val_dataset = Subset(full_dataset, range(train_end, val_end))
                    test_dataset = Subset(full_dataset, range(val_end, n_samples))
                else:
                    train_dataset = FineTuneDataset(X_train, y_train)
                    val_dataset = FineTuneDataset(X_val, y_val)
                    test_dataset = FineTuneDataset(X_test, y_test)

            if self.verbose and logger:
                logger.info("Creating DataLoaders", indent=1)
            use_streaming_memmap = features_path is not None and self.mode == "pretrain"
            streaming_active = self.use_streaming_fallback
            if self.verbose:
                if self.use_streaming_fallback:
                    backend = "Streaming Parquet"
                    dataset_type = "MemoryEfficientPretrainDataset"
                elif use_streaming_memmap:
                    backend = "Memmap Windows (PretrainWindowDataset)"
                    dataset_type = "PretrainWindowDataset"
                else:
                    backend = "In-Memory Dataset"
                    dataset_type = "PretrainDataset"
                
                if logger:
                    logger.info(f"Backend: {backend}", indent=2)

            workers = 0 if streaming_active else self.num_workers
            pin_mem = self.num_workers > 0 and not streaming_active
            do_shuffle = self.shuffle_train
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=do_shuffle,
                num_workers=workers,
                pin_memory=pin_mem,
                persistent_workers=workers > 0,
                prefetch_factor=4 if workers > 0 else None,
                drop_last=False,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=workers,
                pin_memory=pin_mem,
                persistent_workers=False,
                prefetch_factor=3 if workers > 0 else None,
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=workers,
                pin_memory=pin_mem,
                persistent_workers=False,
                prefetch_factor=3 if workers > 0 else None,
            )

            if X is not None:
                del X, y, X_train, y_train, X_val, y_val, X_test, y_test
            if 'data' in locals():
                del data
            if 'gdata' in locals():
                del gdata
            gc.collect()

            return {
                "train": train_loader,
                "val": val_loader,
                "test": test_loader,
            }
        except Exception as e:
            if logger:
                logger.error(f"Data loading failed: {type(e).__name__}: {e}", indent=1)
            gc.collect()
            raise

    def get_dataset_info(self) -> Dict[str, int]:
        """Get information about the dataset.

        Returns:
            Dictionary with dataset statistics.
        """
        data = pd.read_parquet(self.data_path, engine="pyarrow")
        X, y = self.processing_strategy.process(data)

        import numpy as np
        return {
            "total_samples": len(X),
            "sequence_length": X.shape[1],
            "num_features": X.shape[2],
            "target_mean": float(np.mean(y)),
            "target_std": float(np.std(y)),
        }
