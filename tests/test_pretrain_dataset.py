"""Unit tests for PretrainDataset."""

import numpy as np
import pytest
import torch

from src.data.pretrain_dataset import PretrainDataset


def test_pretrain_dataset_initialization():
    """Test PretrainDataset initialization."""
    X = np.random.randn(100, 256, 16).astype(np.float32)
    asset_ids = np.random.randint(0, 15, size=100)
    
    dataset = PretrainDataset(
        X=X,
        asset_ids=asset_ids,
        sequence_length=256,
        masking_ratio=0.15,
        volatility_lookahead=60,
    )
    
    assert len(dataset) == 100 - 60
    assert dataset.n_features == 16


def test_pretrain_dataset_getitem():
    """Test PretrainDataset __getitem__."""
    X = np.random.randn(100, 256, 16).astype(np.float32)
    asset_ids = np.random.randint(0, 15, size=100)
    
    dataset = PretrainDataset(
        X=X,
        asset_ids=asset_ids,
        sequence_length=256,
        masking_ratio=0.15,
        volatility_lookahead=60,
    )
    
    sample = dataset[0]
    
    assert "input_sequence" in sample
    assert "mask_binary" in sample
    assert "original_sequence" in sample
    assert "volatility_target" in sample
    assert "asset_id" in sample
    
    assert sample["input_sequence"].shape == (256, 16)
    assert sample["mask_binary"].shape == (256,)
    assert sample["original_sequence"].shape == (256, 16)
    assert sample["volatility_target"].shape == (1,)
    assert sample["asset_id"].ndim == 0


def test_pretrain_dataset_masking():
    """Test that masking is applied correctly."""
    X = np.ones((100, 256, 16), dtype=np.float32)
    asset_ids = np.zeros(100, dtype=np.int64)
    
    dataset = PretrainDataset(
        X=X,
        asset_ids=asset_ids,
        sequence_length=256,
        masking_ratio=0.15,
        smart_masking_prob=0.0,
        cross_asset_masking_prob=0.0,
    )
    
    sample = dataset[0]
    
    mask_binary = sample["mask_binary"]
    masked_seq = sample["input_sequence"]
    
    num_masked = mask_binary.sum().item()
    expected_masked = int(256 * 0.15)
    
    assert abs(num_masked - expected_masked) <= 2
    
    assert torch.all(masked_seq[mask_binary] == 0.0)
    assert torch.all(masked_seq[~mask_binary] == 1.0)
