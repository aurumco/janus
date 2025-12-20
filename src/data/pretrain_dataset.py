"""PyTorch dataset for SSL pre-training with masking."""

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
        # We need to compute std(diff(future_prices)) for each idx.
        # future_prices for idx comes from X[idx+1 : idx+1+lookahead]
        # X shape: (N, L, F)
        # We extract the relevant feature: prices = X[:, :, target_idx] -> (N, L)

        prices = self.X[:, :, target_idx]

        # We process in chunks to avoid OOM
        chunk_size = 10000
        lookahead = self.volatility_lookahead

        # We need X[idx+1...idx+lookahead].
        # The last sample is at n_samples - 1.
        # So valid idx goes up to n_samples - lookahead - 1.
        # Actually __len__ is n_samples - lookahead.
        # The loop range was range(self.n_samples).
        # Inside loop: future_end_idx = min(idx + 1 + lookahead, n_samples)
        # If idx is near end, the window is smaller.

        # Vectorized approach is easier if we ignore the tail edge cases or handle them separately.
        # However, to be robust and match the logic:

        # Let's handle the main block where we have full lookahead.
        valid_n = max(0, self.n_samples - lookahead - 1)

        if valid_n > 0:
            # We take prices starting from index 1 (since loop uses idx+1)
            # prices_shifted = prices[1:]

            # We unfold dimension 0 with size=lookahead, step=1
            # prices_unfolded shape: (N_unfolded, L, lookahead)
            # N_unfolded = (N-1) - lookahead + 1 = N - lookahead.
            # This covers idx 0 to N-lookahead-1.

            # Since unfolding creates a view, it's cheap. But operations on it are expensive.
            # We chunk the operations.

            prices_source = prices[1:] # Shift by 1

            for start_i in range(0, valid_n, chunk_size):
                end_i = min(start_i + chunk_size, valid_n)

                # Unfold creates a window view.
                # We need a window of size 'lookahead' starting at each position.
                # slice source: prices_source[start_i : end_i + lookahead - 1]
                # length needed: (end_i - start_i) + lookahead - 1?
                # No. Unfold on T elements gives T - size + 1 windows.
                # We want (end_i - start_i) windows.
                # So we need input of length (end_i - start_i) + lookahead - 1.

                # Correction: to get N windows, we need input size N + size - 1.
                # Here N = (end_i - start_i). size = lookahead.
                # So we need slice length = (end_i - start_i) + lookahead - 1.

                sub_slice = prices_source[start_i : end_i + lookahead - 1]
                if sub_slice.size(0) < lookahead:
                    break

                # unfold(dim, size, step)
                windows = sub_slice.unfold(0, lookahead, 1) # (B, L, lookahead)

                # We want diff along the window dimension (dim 2)
                # windows: (B, L, lookahead)
                # diffs: windows[..., 1:] - windows[..., :-1]
                diffs = windows[:, :, 1:] - windows[:, :, :-1] # (B, L, lookahead-1)

                # Std over last two dims (L * (lookahead-1))
                # torch.std doesn't support multiple dims until recently?
                # Actually it does. But to be safe and efficient:
                # flatten the last two dims
                diffs_flat = diffs.reshape(diffs.size(0), -1)

                # Calculate std
                stds = torch.std(diffs_flat, dim=1) + 1e-6

                self.volatility_targets[start_i:end_i] = stds

        # Handle the tail (where we don't have full lookahead)
        # This matches the "if len(future_prices) > 5" logic in the loop
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
        # Slicing returns a view, which is safe to read from.
        # The DataLoader will copy it when batching anyway.
        original_sequence = self.X[idx]
        asset_id = self.asset_ids[idx]
        
        if self.masking_ratio > 0.0:
            mask_binary = self._generate_smart_mask(original_sequence)
            masked_sequence = original_sequence.clone()
            masked_sequence[mask_binary] = 0.0
        else:
            mask_binary = torch.zeros(self.seq_len, dtype=torch.bool)
            # If no masking, we must clone if we want to ensure the returned tensor
            # is not a view into the large self.X (though usually fine for read-only).
            # But consistent behavior suggests we should likely return a copy or view.
            # Here we return the view as original_sequence, and masked_sequence is same.
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
        
        use_smart_masking = np.random.random() < self.smart_masking_prob
        use_cross_asset = np.random.random() < self.cross_asset_masking_prob
        
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
            # Get indices that are not yet masked
            unmasked_indices = (~mask_binary).nonzero(as_tuple=True)[0].numpy()

            if len(unmasked_indices) > 0:
                # Limit needed to available unmasked spots
                needed = min(needed, len(unmasked_indices))

                new_mask_indices = np.random.choice(
                    unmasked_indices, size=needed, replace=False
                )
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
        
        price_volatility = torch.std(price_features, dim=1)
        
        high_vol_threshold = torch.quantile(price_volatility, 0.8)
        high_vol_indices = (price_volatility > high_vol_threshold).nonzero(as_tuple=True)[0]
        
        if len(high_vol_indices) > 0:
            mask_idx = high_vol_indices[np.random.randint(len(high_vol_indices))]
            mask_length = np.random.randint(1, 4)
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
        
        for feat_idx in self.price_feature_indices:
            if np.random.random() < 0.15:
                num_positions = max(1, int(self.sequence_length * self.masking_ratio * 0.5))
                positions = np.random.choice(
                    self.sequence_length, size=num_positions, replace=False
                )
                # Optimize: Vectorized assignment
                mask_binary[positions] = True
        
        return mask_binary
