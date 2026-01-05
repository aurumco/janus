"""PyTorch dataset for SSL pre-training with masking."""

from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    """Dataset for self-supervised pre-training with masked reconstruction."""

    def __init__(
        self,
        X: torch.Tensor,
        asset_ids: torch.Tensor,
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
        # Ensure input is Tensor
        if not isinstance(X, torch.Tensor):
            self.X = torch.FloatTensor(X)
        else:
            self.X = X.float()

        if not isinstance(asset_ids, torch.Tensor):
            self.asset_ids = torch.LongTensor(asset_ids)
        else:
            self.asset_ids = asset_ids.long()

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

        target_idx = self.price_column_idx
        if target_idx >= self.n_features:
            target_idx = 0

        prices = self.X[:, :, target_idx]

        chunk_size = 10000
        lookahead = self.volatility_lookahead

        valid_n = max(0, self.n_samples - lookahead - 1)

        if valid_n > 0:
            prices_source = prices[1:]

            for start_i in range(0, valid_n, chunk_size):
                end_i = min(start_i + chunk_size, valid_n)

                # Slicing logic for unfold
                sub_slice = prices_source[start_i : end_i + lookahead - 1]
                if sub_slice.size(0) < lookahead:
                    break

                windows = sub_slice.unfold(0, lookahead, 1)
                diffs = windows[:, :, 1:] - windows[:, :, :-1]

                # Flatten the last two dims to compute std over all diffs in the window
                diffs_flat = diffs.reshape(diffs.size(0), -1)

                # Calculate std
                stds = torch.std(diffs_flat, dim=1) + 1e-6

                self.volatility_targets[start_i:end_i] = stds

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
        # View avoids copy
        original_sequence = self.X[idx]
        asset_id = self.asset_ids[idx]
        
        if self.masking_ratio > 0.0:
            mask_binary = self._generate_smart_mask(original_sequence)
            masked_sequence = original_sequence.clone()
            masked_sequence[mask_binary] = 0.0
        else:
            mask_binary = torch.zeros(self.seq_len, dtype=torch.bool)
            masked_sequence = original_sequence
        
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
        
        # Use torch.rand instead of np.random
        use_smart_masking = torch.rand(1).item() < self.smart_masking_prob
        use_cross_asset = torch.rand(1).item() < self.cross_asset_masking_prob
        
        if use_smart_masking:
            mask_binary = self._volatility_aware_mask(sequence, mask_binary)
        
        if use_cross_asset:
            mask_binary = self._cross_asset_mask(sequence, mask_binary)
        
        # Ensure we meet the minimum masking ratio
        current_masked_count = mask_binary.sum().item()
        target_masked_count = int(self.sequence_length * self.masking_ratio)

        if current_masked_count < target_masked_count:
            needed = target_masked_count - current_masked_count
            unmasked_indices = (~mask_binary).nonzero(as_tuple=True)[0]

            if len(unmasked_indices) > 0:
                needed = min(needed, len(unmasked_indices))

                # Torch randperm for sampling
                perm = torch.randperm(len(unmasked_indices))
                new_mask_indices = unmasked_indices[perm[:needed]]
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
        price_features = sequence[:, self.price_feature_indices]
        price_volatility = torch.std(price_features, dim=1)
        
        # Approximate quantile using topk for speed if tensor is large, or sort.
        # For small seq_len (e.g. 72, 96), sort is fast.
        # But 'quantile' function exists in newer torch versions.
        # To match previous behavior safely:
        # quantile = torch.quantile(price_volatility, 0.8) # requires torch 1.7+
        # Let's use sort which is robust.
        # 0.8 quantile means we want top 20%.

        k = max(1, int(len(price_volatility) * 0.2))
        # topk returns (values, indices)
        # We want the threshold value, which is the min of the top k
        top_vals, _ = torch.topk(price_volatility, k)
        high_vol_threshold = top_vals[-1]

        high_vol_indices = (price_volatility > high_vol_threshold).nonzero(as_tuple=True)[0]
        
        if len(high_vol_indices) > 0:
            # Select random index from high vol indices
            idx_in_indices = torch.randint(0, len(high_vol_indices), (1,)).item()
            mask_idx = high_vol_indices[idx_in_indices].item()

            mask_length = torch.randint(1, 4, (1,)).item()
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
        for _ in self.price_feature_indices:
            if torch.rand(1).item() < 0.15:
                num_positions = max(1, int(self.sequence_length * self.masking_ratio * 0.5))

                # Sample random positions
                positions = torch.randperm(self.sequence_length)[:num_positions]
                mask_binary[positions] = True
        
        return mask_binary
