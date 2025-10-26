"""Data loading and preparation utilities."""

from pathlib import Path
from typing import Dict, Optional
import gc
import time

import pandas as pd
from torch.utils.data import DataLoader

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
        random_seed: int = 42,
        masking_ratio: float = 0.15,
        volatility_lookahead: int = 60,
        sequence_length: int = 96,
        smart_masking_prob: float = 0.4,
        cross_asset_masking_prob: float = 0.3,
        use_gpu_preprocess: bool = False,
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
            if psutil:
                mi = psutil.Process().memory_info()
                print(f"{prefix} | RSS={mi.rss/1e9:.2f} GB, VMS={mi.vms/1e9:.2f} GB")

        try:
            print("- Reading parquet file...")
            mem_info("  Before read")
            data = None
            gdata = None
            if self.use_gpu_preprocess:
                try:
                    import cudf  # type: ignore
                    gdata = str(self.data_path)
                    print(f"  Using cuDF GPU processing (chunked read)")
                except Exception as ge:
                    print(f"  GPU read failed ({type(ge).__name__}: {ge}), falling back to CPU")
                    gdata = None
            if gdata is None:
                data = pd.read_parquet(self.data_path, engine="pyarrow")
                print(f"  Loaded DataFrame: shape={data.shape}")
                mem_info("  After read")

            print("- Processing sequences (vectorized sliding windows)...")
            if gdata is not None and self.use_gpu_preprocess:
                features_path, asset_ids_path, n_timesteps, n_features = self.processing_strategy.process_gpu(gdata)
                print(f"  Base features: {n_timesteps} timesteps x {n_features} features")
                mem_info("  After GPU process")
                X = None
                y = None
            else:
                X, y = self.processing_strategy.process(data)
                print(f"  Sequences: X={X.shape}, y={y.shape}")
                mem_info("  After process")
                features_path = None
                asset_ids_path = None
                n_timesteps = None
                n_features = None

            print("- Splitting train/val/test...")
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

            print("- Building PyTorch datasets...")
            if self.mode == "pretrain":
                if features_path is not None:
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
                train_dataset = FineTuneDataset(X_train, y_train)
                val_dataset = FineTuneDataset(X_val, y_val)
                test_dataset = FineTuneDataset(X_test, y_test)

            print("- Creating DataLoader objects...")
            use_streaming = features_path is not None and self.mode == "pretrain"
            workers = 0 if use_streaming else self.num_workers
            pin_mem = False if use_streaming else True
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle_train,
                num_workers=workers,
                pin_memory=pin_mem,
                persistent_workers=False,
                prefetch_factor=1 if workers > 0 else None,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=workers,
                pin_memory=pin_mem,
                persistent_workers=False,
                prefetch_factor=1 if workers > 0 else None,
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=workers,
                pin_memory=pin_mem,
                persistent_workers=False,
                prefetch_factor=1 if workers > 0 else None,
            )

            elapsed = time.time() - start_time
            print(f"✓ Data loaders ready in {elapsed:.2f}s")
            mem_info("  Final")

            return {
                "train": train_loader,
                "val": val_loader,
                "test": test_loader,
            }
        except Exception as e:
            print("! Error while creating data loaders")
            print(f"  Exception: {type(e).__name__}: {e}")
            mem_info("  On error")
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
