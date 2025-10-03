"""Data loading and preparation utilities."""

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .base_strategy import DataProcessingStrategy
from .dataset import BitcoinTrendDataset


class DataLoaderFactory:
    """Factory for creating data loaders using the Strategy Pattern."""

    def __init__(
        self,
        data_path: str,
        processing_strategy: DataProcessingStrategy,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle_train: bool = True,
        random_seed: int = 42,
    ) -> None:
        """Initialize data loader factory.

        Args:
            data_path: Path to the parquet data file.
            processing_strategy: Strategy for processing data.
            train_ratio: Proportion of data for training.
            val_ratio: Proportion of data for validation.
            test_ratio: Proportion of data for testing.
            batch_size: Batch size for data loaders.
            num_workers: Number of worker processes.
            shuffle_train: Whether to shuffle training data.
            random_seed: Random seed for reproducibility.
        """
        self.data_path = Path(data_path)
        self.processing_strategy = processing_strategy
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train
        self.random_seed = random_seed

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Train, val, and test ratios must sum to 1.0")

    def create_data_loaders(self) -> Dict[str, DataLoader]:
        """Create train, validation, and test data loaders.

        Returns:
            Dictionary with 'train', 'val', and 'test' DataLoaders.
        """
        data = pd.read_parquet(self.data_path)

        X, y = self.processing_strategy.process(data)

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.test_ratio,
            random_state=self.random_seed,
            stratify=y,
        )

        val_ratio_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio_adjusted,
            random_state=self.random_seed,
            stratify=y_temp,
        )

        train_dataset = BitcoinTrendDataset(X_train, y_train)
        val_dataset = BitcoinTrendDataset(X_val, y_val)
        test_dataset = BitcoinTrendDataset(X_test, y_test)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        return {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }

    def get_dataset_info(self) -> Dict[str, int]:
        """Get information about the dataset.

        Returns:
            Dictionary with dataset statistics.
        """
        data = pd.read_parquet(self.data_path)
        X, y = self.processing_strategy.process(data)

        return {
            "total_samples": len(X),
            "sequence_length": X.shape[1],
            "num_features": X.shape[2],
            "num_classes": len(set(y)),
        }
