"""Ultra-optimized data loaders for extreme memory constraints.

Implements strategies from DeepSeek, TimeMachine, and other large-scale TS training.
"""

import gc
from typing import Dict

import torch
from torch.utils.data import DataLoader

from .streaming_dataset import StreamingParquetDataset, UltraLightweightPretrainDataset


def create_ultra_optimized_loaders(
    data_path: str,
    mode: str = "pretrain",
    sequence_length: int = 32,
    batch_size: int = 4,
    max_samples: int = 5000,
    feature_columns: list = None,
    target_column: str = None,
    masking_ratio: float = 0.15,
    use_streaming: bool = False,
) -> Dict[str, DataLoader]:
    """Create ultra-optimized data loaders with minimal memory footprint.

    Args:
        data_path: Path to parquet file.
        mode: 'pretrain' or 'finetune'.
        sequence_length: Sequence length (keep short: 32-64).
        batch_size: Batch size (keep small: 4-8).
        max_samples: Max samples for ultra-lightweight mode.
        feature_columns: Feature column names.
        target_column: Target column name (finetune only).
        masking_ratio: Masking ratio for pretrain.
        use_streaming: Use true streaming (IterableDataset) vs sampled.

    Returns:
        Dictionary with 'train' and 'val' loaders.
    """
    print(f"\n{'='*60}")
    print(f"Creating Ultra-Optimized {mode.upper()} Loaders")
    print(f"{'='*60}")
    print(f"  Mode: {mode}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Batch size: {batch_size}")
    print(f"  Streaming: {use_streaming}")
    print(f"{'='*60}\n")

    if mode == "pretrain":
        if use_streaming:
            print("→ Using StreamingParquetDataset (true streaming)")
            train_dataset = StreamingParquetDataset(
                parquet_path=data_path,
                sequence_length=sequence_length,
                feature_columns=feature_columns,
                mode="pretrain",
                chunk_size=500,
                masking_ratio=masking_ratio,
            )

            val_dataset = StreamingParquetDataset(
                parquet_path=data_path,
                sequence_length=sequence_length,
                feature_columns=feature_columns,
                mode="pretrain",
                chunk_size=500,
                masking_ratio=masking_ratio,
            )

            # For IterableDataset, no shuffle in DataLoader
            shuffle_train = False
        else:
            print("→ Using UltraLightweightPretrainDataset (sampled)")
            full_dataset = UltraLightweightPretrainDataset(
                parquet_path=data_path,
                sequence_length=sequence_length,
                max_samples=max_samples,
                masking_ratio=masking_ratio,
            )

            # Split into train/val
            train_size = int(0.85 * len(full_dataset))
            val_size = len(full_dataset) - train_size

            from torch.utils.data import random_split

            train_dataset, val_dataset = random_split(
                full_dataset, [train_size, val_size]
            )

            shuffle_train = True

        # Create loaders with extreme memory optimization
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
            persistent_workers=False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
        )

    else:  # finetune
        print("→ Using StreamingParquetDataset for finetune")
        train_dataset = StreamingParquetDataset(
            parquet_path=data_path,
            sequence_length=sequence_length,
            feature_columns=feature_columns,
            target_column=target_column,
            mode="finetune",
            chunk_size=500,
        )

        val_dataset = StreamingParquetDataset(
            parquet_path=data_path,
            sequence_length=sequence_length,
            feature_columns=feature_columns,
            target_column=target_column,
            mode="finetune",
            chunk_size=500,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )

    # Aggressive cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"✓ Loaders created successfully\n")

    return {"train": train_loader, "val": val_loader, "test": val_loader}
