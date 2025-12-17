
import pytest
import torch
import numpy as np
from src.data.pretrain_dataset import PretrainDataset

class TestPretrainDataset:
    def test_volatility_calculation_vectorized_vs_loop(self):
        """Verify that the vectorized volatility calculation matches the loop-based one."""
        # Setup random data
        n_samples = 100
        seq_len = 50
        n_features = 10
        # Make data predictable to debug if needed, but random is fine for equivalence check
        X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
        asset_ids = np.zeros(n_samples, dtype=np.int64)

        lookahead = 10
        price_col_idx = 3

        ds = PretrainDataset(
            X,
            asset_ids,
            sequence_length=seq_len,
            volatility_lookahead=lookahead,
            price_column_idx=price_col_idx
        )

        # Calculate expected volatility using the loop method (simplified from original code)
        expected_targets = torch.zeros(n_samples, dtype=torch.float32)
        X_torch = torch.from_numpy(X)

        target_price_idx = price_col_idx

        for idx in range(n_samples):
            future_end_idx = min(idx + 1 + lookahead, n_samples)
            if future_end_idx > idx + 5: # Original condition
                future_prices = X_torch[
                    idx + 1 : future_end_idx,
                    :,
                    target_price_idx
                ]

                if len(future_prices) > 5:
                    returns = future_prices[1:] - future_prices[:-1]
                    expected_targets[idx] = torch.std(returns) + 1e-6

        # Check if they match
        assert torch.allclose(ds.volatility_targets, expected_targets, atol=1e-5), \
            f"Max diff: {torch.max(torch.abs(ds.volatility_targets - expected_targets))}"

    def test_price_column_idx_usage(self):
        """Ensure price_column_idx is actually used."""
        n_samples = 50
        seq_len = 20
        n_features = 5
        X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
        # Make column 0 very volatile, column 4 constant
        X[:, :, 0] = np.random.randn(n_samples, seq_len) * 100
        X[:, :, 4] = np.ones((n_samples, seq_len))

        asset_ids = np.zeros(n_samples, dtype=np.int64)

        # Dataset using col 0 (high vol)
        ds_high = PretrainDataset(X, asset_ids, volatility_lookahead=5, price_column_idx=0)

        # Dataset using col 4 (zero vol)
        ds_low = PretrainDataset(X, asset_ids, volatility_lookahead=5, price_column_idx=4)

        assert ds_high.volatility_targets.mean() > ds_low.volatility_targets.mean()
        assert ds_low.volatility_targets.mean() < 1.0 # Should be basically 1e-6

    def test_default_price_column_behavior(self):
        """Ensure legacy behavior (defaulting to col 3) works when price_column_idx is None."""
        n_samples = 50
        seq_len = 20
        n_features = 5 # So col 3 is valid
        X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)

        # Make col 3 distinct
        X[:, :, 3] = np.random.randn(n_samples, seq_len) * 50

        asset_ids = np.zeros(n_samples, dtype=np.int64)

        # Default init (price_column_idx=None)
        ds_default = PretrainDataset(X, asset_ids, volatility_lookahead=5)

        # Explicit init (price_column_idx=3)
        ds_explicit = PretrainDataset(X, asset_ids, volatility_lookahead=5, price_column_idx=3)

        assert torch.allclose(ds_default.volatility_targets, ds_explicit.volatility_targets)
