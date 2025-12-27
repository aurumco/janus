"""PyTorch dataset for SSL pre-training with masking."""

import random
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    """Dataset for self-supervised pre-training with masked reconstruction."""

    def __init__(
        self,
        X: np.ndarray,
        asset_ids: np.ndarray,
        sequence_length: int = 256,
        masking_ratio: float = 0.15,
        volatility_lookahead: int = 60,
        smart_masking_prob: float = 0.4,
        cross_asset_masking_prob: float = 0.3,
        price_column_idx: Optional[int] = None,
        price_feature_indices: Optional[List[int]] = None,
    ) -> None:
        """Initialize pre-training dataset.

        Args:
            X: Feature sequences of shape (n_samples, seq_len, n_features).
            asset_ids: Asset identifiers of shape (n_samples,).
            sequence_length: Length of input sequences.
            masking_ratio: Ratio of timesteps to mask.
            volatility_lookahead: Steps ahead for volatility prediction.
            smart_masking_prob: Probability of using smart masking.
            cross_asset_masking_prob: Probability of using cross-asset masking.
            price_column_idx: Index of price column for volatility calc.
            price_feature_indices: Indices of price-related features for masking.
        """
        self.X = torch.FloatTensor(X)
        self.asset_ids = torch.LongTensor(asset_ids)
        self.sequence_length = sequence_length
        self.masking_ratio = masking_ratio
        self.volatility_lookahead = volatility_lookahead
        self.smart_masking_prob = smart_masking_prob
        self.cross_asset_masking_prob = cross_asset_masking_prob
        self.n_samples, self.seq_len, self.n_features = self.X.shape

        # Restore legacy behavior: default to col 3 (Close) if available
        if price_column_idx is None:
            self.price_column_idx = min(3, self.n_features - 1)
        else:
            self.price_column_idx = price_column_idx

        if price_feature_indices is None:
            # Fallback to first 4 or less if not enough features
            limit = min(4, self.n_features)
            self.price_feature_indices = list(range(limit))
        else:
            self.price_feature_indices = price_feature_indices

        self._precompute_volatility_targets()

    def _precompute_volatility_targets(self) -> None:
        """Precompute volatility targets for all samples to speed up training."""
        self.volatility_targets = torch.zeros(self.n_samples, dtype=torch.float32)

        # Use configurable price column index
        target_idx = self.price_column_idx
        if target_idx >= self.n_features:
            target_idx = 0 # Fallback if invalid

        # Vectorized calculation using unfold
        prices = self.X[:, :, target_idx]

        # We process in chunks to avoid OOM
        chunk_size = 10000
        lookahead = self.volatility_lookahead
        valid_n = max(0, self.n_samples - lookahead - 1)

        if valid_n > 0:
            prices_source = prices[1:] # Shift by 1

            for start_i in range(0, valid_n, chunk_size):
                end_i = min(start_i + chunk_size, valid_n)

                sub_slice = prices_source[start_i : end_i + lookahead - 1]
                if sub_slice.size(0) < lookahead:
                    break

                # unfold(dim, size, step)
                windows = sub_slice.unfold(0, lookahead, 1) # (B, L, lookahead)
                diffs = windows[:, :, 1:] - windows[:, :, :-1] # (B, L, lookahead-1)

                # Flatten last two dims for std calculation
                diffs_flat = diffs.reshape(diffs.size(0), -1)
                stds = torch.std(diffs_flat, dim=1) + 1e-6

                self.volatility_targets[start_i:end_i] = stds

        # Handle the tail
        tail_start = valid_n
        for idx in range(tail_start, self.n_samples):
            future_end_idx = min(idx + 1 + lookahead, self.n_samples)
            if future_end_idx > idx + 5:
                future_prices = prices[idx + 1 : future_end_idx, :]
                if future_prices.size(0) > 5:
                    returns = future_prices[1:] - future_prices[:-1]
                    self.volatility_targets[idx] = torch.std(returns) + 1e-6

    def __len__(self) -> int:
        """Get dataset length.

        Returns:
            Number of valid samples.
        """
        return max(0, self.n_samples - self.volatility_lookahead)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample with optional masking for SSL pre-training.

        Args:
            idx: Sample index.

        Returns:
            Dictionary containing masked sequence, mask, targets, and asset_id.
        """
        # Optimize: Avoid initial clone since we need a clean original for return.
        original_sequence = self.X[idx]
        asset_id = self.asset_ids[idx]
        
        if self.masking_ratio > 0.0:
            mask_binary = self._generate_smart_mask(original_sequence)
            masked_sequence = original_sequence.clone()
            masked_sequence[mask_binary] = 0.0
        else:
            mask_binary = torch.zeros(self.seq_len, dtype=torch.bool)
            masked_sequence = original_sequence
        
        # Use precomputed volatility
        volatility = self.volatility_targets[idx] * 100.0

        return {
            "input_sequence": masked_sequence,
            "mask_binary": mask_binary,
            "original_sequence": original_sequence,
            "volatility_target": volatility.unsqueeze(0),
            "asset_id": asset_id,
        }
    
    def _generate_smart_mask(self, sequence: torch.Tensor) -> torch.Tensor:
        """Generate smart mask using volatility-aware and cross-asset strategies.
        
        Args:
            sequence: Input sequence tensor (seq_len, n_features).
            
        Returns:
            Binary mask tensor (seq_len,).
        """
        mask_binary = torch.zeros(self.sequence_length, dtype=torch.bool)
        
        # Use standard random for scalar probabilities (faster than torch.rand(1).item())
        use_smart_masking = random.random() < self.smart_masking_prob
        use_cross_asset = random.random() < self.cross_asset_masking_prob
        
        if use_smart_masking:
            mask_binary = self._volatility_aware_mask(sequence, mask_binary)
        
        if use_cross_asset:
            mask_binary = self._cross_asset_mask(sequence, mask_binary)
        
        # Ensure we meet the minimum masking ratio
        current_masked_count = mask_binary.sum().item()
        target_masked_count = int(self.sequence_length * self.masking_ratio)

        if current_masked_count < target_masked_count:
            # Add random masking to meet the quota
            needed = target_masked_count - current_masked_count

            # Optimized: Get indices directly on torch tensor
            unmasked_indices = (~mask_binary).nonzero(as_tuple=True)[0]
            n_unmasked = unmasked_indices.size(0)

            if n_unmasked > 0:
                needed = min(needed, n_unmasked)

                # Optimized: Use torch.randperm for sampling without replacement
                # This stays on the same device/memory type as indices
                perm = torch.randperm(n_unmasked)[:needed]
                new_mask_indices = unmasked_indices[perm]
                mask_binary[new_mask_indices] = True
        
        return mask_binary
    
    def _volatility_aware_mask(
        self, sequence: torch.Tensor, mask_binary: torch.Tensor
    ) -> torch.Tensor:
        """Mask high-volatility periods (important market events).
        
        Args:
            sequence: Input sequence (seq_len, n_features).
            mask_binary: Current mask to update.
            
        Returns:
            Updated mask.
        """
        # Use configurable price feature indices instead of hardcoded slice
        price_features = sequence[:, self.price_feature_indices]
        
        # torch.std along dim 1
        price_volatility = torch.std(price_features, dim=1)
        
        # quantile is somewhat expensive, but needed for logic
        high_vol_threshold = torch.quantile(price_volatility, 0.8)
        high_vol_indices = (price_volatility > high_vol_threshold).nonzero(as_tuple=True)[0]
        n_high_vol = high_vol_indices.size(0)
        
        if n_high_vol > 0:
            # Optimized: Use random.randint for scalar indices
            rand_idx = random.randint(0, n_high_vol - 1)
            mask_idx = high_vol_indices[rand_idx].item()
            mask_length = random.randint(1, 3) # randint is inclusive for python random

            end_idx = min(mask_idx + mask_length, self.sequence_length)
            mask_binary[mask_idx:end_idx] = True
        
        return mask_binary
    
    def _cross_asset_mask(
        self, sequence: torch.Tensor, mask_binary: torch.Tensor
    ) -> torch.Tensor:
        """Mask cross-asset correlated features.
        
        Args:
            sequence: Input sequence (seq_len, n_features).
            mask_binary: Current mask to update.
            
        Returns:
            Updated mask.
        """
        # Use configurable price feature indices
        
        for _ in self.price_feature_indices:
            if random.random() < 0.15:
                num_positions = max(1, int(self.sequence_length * self.masking_ratio * 0.5))

                # Optimized: Use torch.randperm
                positions = torch.randperm(self.sequence_length)[:num_positions]

                mask_binary[positions] = True
        
        return mask_binary
