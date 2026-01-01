"""Data loading and preparation utilities."""

from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import gc
import time
import numpy as np

import pandas as pd
from torch.utils.data import DataLoader, Subset, Dataset

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
from .finetune_dataset import FineTuneDataset, LazyFineTuneDataset
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
        self._log_memory("Start create_data_loaders")

        try:
            # 1. Load Data
            data, gdata = self._load_data()

            # 2. Process Data (GPU or CPU)
            # We split the process to ensure data is deleted before tensor allocation
            processed_data = self._process_data(data, gdata)

            # CPU PATH: We have temporary splits, need to delete data, then process
            if "_temp_splits" in processed_data:
                # 2b. Explicitly delete raw data before heavy processing
                del data
                data = None
                gc.collect()

                # 2c. Process the splits into tensors
                processed_data["data_splits"] = self._process_splits(processed_data.pop("_temp_splits"))
            else:
                # GPU PATH or other path where data is already handled or not used
                if data is not None:
                    del data
                data = None
                gc.collect()

            # 3. Create Datasets
            datasets = self._create_datasets(processed_data, gdata is None)

            # 4. Create Loaders
            loaders = self._create_loaders(datasets, processed_data)

            # Cleanup
            del gdata, processed_data
            gc.collect()

            return loaders

        except Exception as e:
            if logger:
                logger.error(f"Data loading failed: {type(e).__name__}: {e}", indent=1)
            gc.collect()
            raise

    def _log_memory(self, prefix: str) -> None:
        """Log memory usage if verbose and psutil available."""
        if psutil and self.verbose:
            mi = psutil.Process().memory_info()
            if logger:
                logger.info(f"{prefix}: RSS={mi.rss/1e9:.2f}GB", indent=2)

    def _load_data(self) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Load data from parquet file."""
        if self.verbose and logger:
            logger.info("Reading parquet file", indent=1)

        gdata = None
        data = None

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

        return data, gdata

    def _process_data(
        self,
        data: Optional[pd.DataFrame],
        gdata: Optional[str]
    ) -> Dict[str, Any]:
        """Process raw data into features and targets or memory maps."""
        if self.verbose and logger:
            logger.info("Processing sequences", indent=1)

        result = {
            "features_path": None,
            "asset_ids_path": None,
            "targets_path": None,
            "n_timesteps": None,
            "n_features": None,
            "X": None,
            "y": None,
            "data_splits": None
        }

        if gdata is not None and self.use_gpu_preprocess:
            features_path, asset_ids_path, targets_path, n_timesteps, n_features = self.processing_strategy.process_gpu(gdata)
            result.update({
                "features_path": features_path,
                "asset_ids_path": asset_ids_path,
                "targets_path": targets_path,
                "n_timesteps": n_timesteps,
                "n_features": n_features
            })
            if self.verbose and logger:
                logger.info(f"Features: {n_timesteps} × {n_features}", indent=2)
                self._log_memory("After GPU process")

        elif data is not None:
            # CPU processing logic split to ensure memory safety
            splits = self._create_splits(data)

            # IMPORTANT: The caller is responsible for deleting 'data' immediately
            # after we return from _create_splits if we were refactoring further,
            # but here we can't easily signal the caller to delete 'data' halfway.
            # So instead, we just create the splits here and return them.
            # But wait, we need to process the splits.
            # The 'data' variable is a reference.

            # To strictly follow the "Delete Data -> Process Splits" flow:
            # We must process splits AFTER 'data' is deleted.
            # Since we cannot delete the caller's reference to 'data',
            # we rely on the caller to not use 'data' anymore, but the object exists.

            # However, we can modify the flow:
            # We return the splits (DataFrames) in a temporary structure,
            # then the caller deletes 'data', then calls a new method to process splits.
            # BUT, to keep the interface simple, we can do the splitting and processing here
            # IF we accept that 'data' reference is still held by caller.
            # The issue is that `_process_data` takes `data`.

            # Solution: We can't delete caller's reference.
            # The original code had everything in one function so it could `del data`.
            # To replicate this, we need to ensure `data` is not held.
            # Since `data` is passed as argument, we can't clear caller's scope.
            pass

            # Since I am already inside _process_data, I will proceed with the implementation
            # that assumes the caller will handle `del data` if I return early? No.

            # The only way to strictly enforce "Create Splits -> Delete Data -> Process"
            # with this signature is if we don't process here, OR if the caller handles the flow.
            # I will refactor `_process_data` to `_create_splits_and_process`.
            # Actually, I'll stick to the current implementation but verify correct behavior:
            # 1. Create split copies (train_df, etc.)
            # 2. del data (removes local reference)
            # 3. gc.collect()
            # 4. process splits.

            # This works if the caller doesn't keep other references.
            # In `create_data_loaders`, `data` is a local var.
            # If `_process_data` finishes, `data` is still in `create_data_loaders`.
            # So `del data` inside `_process_data` only deletes the local arg.
            # The large object remains in memory until `_process_data` returns and `create_data_loaders` deletes it.
            # BUT `_process_data` creates the tensors! So we have Peak = Data + Splits + Tensors.

            # To fix this, `_process_data` for CPU path must NOT do the tensor processing.
            # It should just return the splits.
            # Then `create_data_loaders` deletes `data`.
            # Then we call a NEW method `_process_splits`.

            splits = self._create_splits(data)
            # We return the splits in a special key to signal the caller
            result["_temp_splits"] = splits

        return result

    def _create_splits(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create train/val/test splits from dataframe."""
        n_rows = len(data)
        train_row_end = int(n_rows * self.train_ratio)
        val_row_end = int(n_rows * (self.train_ratio + self.val_ratio))

        train_df = data.iloc[:train_row_end].copy()
        val_df = data.iloc[train_row_end:val_row_end].copy()
        test_df = data.iloc[val_row_end:].copy()

        return {
            "train": train_df,
            "val": val_df,
            "test": test_df
        }

    def _process_splits(self, splits: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Process splits into tensors."""
        return {
            "train": self._process_split(splits["train"]),
            "val": self._process_split(splits["val"]),
            "test": self._process_split(splits["test"])
        }

    def _process_split(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Process a single data split."""
        if len(df) < self.sequence_length:
            return {"X": None, "y": None, "asset_ids": None}

        # Check if we should use lazy processing for finetune mode
        if self.mode == "finetune" and not self.use_streaming_fallback and hasattr(self.processing_strategy, "process_lazy"):
            X, y = self.processing_strategy.process_lazy(df)
            is_lazy = True
        else:
            X, y = self.processing_strategy.process(df)
            is_lazy = False

        aid_out = getattr(self.processing_strategy, "_asset_ids_out", None)

        return {
            "X": X,
            "y": y,
            "asset_ids": aid_out,
            "is_lazy": is_lazy
        }

    def _create_datasets(
        self,
        processed_data: Dict[str, Any],
        is_streaming: bool
    ) -> Dict[str, Any]:
        """Create PyTorch datasets from processed data."""
        if self.verbose and logger:
            logger.info("Building datasets", indent=1)

        # 1. Determine split indices if using memmap or streaming
        n_samples = 0
        if processed_data["n_timesteps"] is not None:
             n_samples = processed_data["n_timesteps"] - self.sequence_length + 1
        elif is_streaming or self.use_streaming_fallback:
             # Streaming length estimation logic would go here,
             # but we handle it inside dataset creation logic for streaming
             pass

        train_end = int(n_samples * self.train_ratio)
        val_end = int(n_samples * (self.train_ratio + self.val_ratio))

        train_dataset = val_dataset = test_dataset = None

        if self.mode == "pretrain":
            if self.use_streaming_fallback:
                train_dataset, val_dataset, test_dataset = self._create_streaming_pretrain_datasets()
            elif processed_data["features_path"] is not None:
                train_dataset, val_dataset, test_dataset = self._create_memmap_pretrain_datasets(
                    processed_data, train_end, val_end, n_samples
                )
            else:
                train_dataset, val_dataset, test_dataset = self._create_memory_pretrain_datasets(
                    processed_data["data_splits"]
                )
        else:
            if self.use_streaming_fallback:
                train_dataset, val_dataset, test_dataset = self._create_streaming_finetune_datasets()
            elif processed_data["features_path"] is not None:
                train_dataset, val_dataset, test_dataset = self._create_memmap_finetune_datasets(
                    processed_data, train_end, val_end, n_samples
                )
            else:
                train_dataset, val_dataset, test_dataset = self._create_memory_finetune_datasets(
                    processed_data["data_splits"]
                )
                
        return {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset
        }

    def _create_streaming_pretrain_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
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

        return (
            Subset(full_dataset, range(0, train_end)),
            Subset(full_dataset, range(train_end, val_end)),
            Subset(full_dataset, range(val_end, n_samples))
        )

    def _create_memmap_pretrain_datasets(
        self,
        processed_data: Dict[str, Any],
        train_end: int,
        val_end: int,
        n_samples: int
    ) -> Tuple[Dataset, Dataset, Dataset]:

        # Local import to avoid environment-specific import errors
        try:
            from .pretrain_window_dataset import PretrainWindowDataset  # type: ignore
        except Exception:
            try:
                from src.data.pretrain_window_dataset import PretrainWindowDataset  # type: ignore
            except Exception as imp_err:
                raise ImportError(f"Failed to import PretrainWindowDataset: {imp_err}")

        common_args = {
            "features_memmap_path": processed_data["features_path"],
            "asset_ids_memmap_path": processed_data["asset_ids_path"],
            "n_timesteps": processed_data["n_timesteps"],
            "n_features": processed_data["n_features"],
            "sequence_length": self.sequence_length,
            "masking_ratio": self.masking_ratio,
            "volatility_lookahead": self.volatility_lookahead,
            "smart_masking_prob": self.smart_masking_prob,
            "cross_asset_masking_prob": self.cross_asset_masking_prob,
            "stride": self.stride,
        }

        train_ds = PretrainWindowDataset(
            start_index=0,
            end_index=train_end,
            **common_args
        )
        val_ds = PretrainWindowDataset(
            start_index=train_end,
            end_index=val_end,
            deterministic=True,
            **common_args
        )
        test_ds = PretrainWindowDataset(
            start_index=val_end,
            end_index=n_samples,
            deterministic=True,
            **common_args
        )

        return train_ds, val_ds, test_ds

    def _create_memory_pretrain_datasets(self, splits: Dict[str, Any]) -> Tuple[Dataset, Dataset, Dataset]:
        def create_ds(split_data):
            if split_data["X"] is None:
                return None
            return PretrainDataset(
                split_data["X"],
                asset_ids=split_data["asset_ids"],
                sequence_length=self.sequence_length,
                masking_ratio=self.masking_ratio,
                volatility_lookahead=self.volatility_lookahead,
                smart_masking_prob=self.smart_masking_prob,
                cross_asset_masking_prob=self.cross_asset_masking_prob,
            )

        return (
            create_ds(splits["train"]),
            create_ds(splits["val"]),
            create_ds(splits["test"])
        )

    def _create_streaming_finetune_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
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

        return (
            Subset(full_dataset, range(0, train_end)),
            Subset(full_dataset, range(train_end, val_end)),
            Subset(full_dataset, range(val_end, n_samples))
        )

    def _create_memmap_finetune_datasets(
        self,
        processed_data: Dict[str, Any],
        train_end: int,
        val_end: int,
        n_samples: int
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """Create fine-tuning datasets backed by memory maps."""

        # Load memmaps
        n_timesteps = processed_data["n_timesteps"]
        n_features = processed_data["n_features"]

        features_mmap = np.memmap(
            processed_data["features_path"],
            dtype=np.float32,
            mode='r',
            shape=(n_timesteps, n_features)
        )

        targets_path = processed_data.get("targets_path")
        if targets_path:
            # Check if we can infer target dim or if it's 1D
            # LazyFineTuneDataset expects 1D or 2D.
            # process_gpu writes (N,) or (N, D).
            # We need to peek at the file size or assume from strategy?
            # Actually, np.memmap needs shape.
            # Strategy: Try 1D first, if size mismatch, try infer?
            # Better: DataLoaderFactory doesn't know D.
            # But process_gpu calculated n_timesteps.
            # Size of file in floats = n_timesteps * D.
            import os
            size_bytes = os.path.getsize(targets_path)
            size_floats = size_bytes // 4
            target_dim = size_floats // n_timesteps

            t_shape = (n_timesteps,) if target_dim == 1 else (n_timesteps, target_dim)

            targets_mmap = np.memmap(
                targets_path,
                dtype=np.float32,
                mode='r',
                shape=t_shape
            )
        else:
            # Fallback if no targets (should not happen in finetune usually)
            targets_mmap = np.zeros(n_timesteps, dtype=np.float32)

        def create_subset(start, end):
            if start >= end:
                return None

            # Slice the memmaps for the subset
            # IMPORTANT: LazyFineTuneDataset takes the WHOLE array and slices internally by index.
            # But here we want to pass a VIEW of the subset?
            # LazyFineTuneDataset.__init__ takes (features, targets).
            # If we pass features[start:end], it's a slice.
            # Does it copy? np.memmap slice returns a new memmap object (view) usually.
            # Let's trust it works as view.

            feat_subset = features_mmap[start:end]
            tgt_subset = targets_mmap[start:end]

            return LazyFineTuneDataset(
                features=feat_subset,
                targets=tgt_subset,
                sequence_length=self.sequence_length
            )

        return (
            create_subset(0, train_end),
            create_subset(train_end, val_end),
            create_subset(val_end, n_samples)
        )

    def _create_memory_finetune_datasets(self, splits: Dict[str, Any]) -> Tuple[Dataset, Dataset, Dataset]:
        def create_ds(split_data):
            if split_data["X"] is None:
                return None

            if split_data.get("is_lazy", False):
                return LazyFineTuneDataset(
                    features=split_data["X"],
                    targets=split_data["y"],
                    sequence_length=self.sequence_length
                )
            else:
                return FineTuneDataset(split_data["X"], split_data["y"])

        return (
            create_ds(splits["train"]),
            create_ds(splits["val"]),
            create_ds(splits["test"])
        )

    def _create_loaders(
        self,
        datasets: Dict[str, Dataset],
        processed_data: Dict[str, Any]
    ) -> Dict[str, DataLoader]:
        """Create DataLoader instances."""
        if self.verbose and logger:
            logger.info("Creating DataLoaders", indent=1)

        use_streaming_memmap = processed_data.get("features_path") is not None and self.mode == "pretrain"
        streaming_active = self.use_streaming_fallback

        if self.verbose:
            self._log_backend_info(streaming_active, use_streaming_memmap)

        workers = 0 if streaming_active else self.num_workers
        pin_mem = self.num_workers > 0 and not streaming_active
        do_shuffle = self.shuffle_train

        loaders = {}

        # Train loader
        if datasets["train"]:
            loaders["train"] = DataLoader(
                datasets["train"],
                batch_size=self.batch_size,
                shuffle=do_shuffle,
                num_workers=workers,
                pin_memory=pin_mem,
                persistent_workers=workers > 0,
                prefetch_factor=4 if workers > 0 else None,
                drop_last=False,
            )
        else:
            loaders["train"] = None

        # Val loader
        if datasets["val"]:
            loaders["val"] = DataLoader(
                datasets["val"],
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=workers,
                pin_memory=pin_mem,
                persistent_workers=False,
                prefetch_factor=3 if workers > 0 else None,
            )
        else:
            loaders["val"] = None

        # Test loader
        if datasets["test"]:
            loaders["test"] = DataLoader(
                datasets["test"],
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=workers,
                pin_memory=pin_mem,
                persistent_workers=False,
                prefetch_factor=3 if workers > 0 else None,
            )
        else:
            loaders["test"] = None

        return loaders

    def _log_backend_info(self, streaming_active: bool, use_streaming_memmap: bool) -> None:
        if streaming_active:
            backend = "Streaming Parquet"
        elif use_streaming_memmap:
            backend = "Memmap Windows (PretrainWindowDataset)"
        else:
            backend = "In-Memory Dataset"

        if logger:
            logger.info(f"Backend: {backend}", indent=2)

    def get_dataset_info(self) -> Dict[str, int]:
        """Get information about the dataset.

        Returns:
            Dictionary with dataset statistics.
        """
        data = pd.read_parquet(self.data_path, engine="pyarrow")
        X, y = self.processing_strategy.process(data)

        return {
            "total_samples": len(X),
            "sequence_length": X.shape[1],
            "num_features": X.shape[2],
            "target_mean": float(np.mean(y)),
            "target_std": float(np.std(y)),
        }
