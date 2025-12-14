"""PyTorch dataset for SSL pre-training with masking."""

from typing import Dict

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
        price_column_idx: int = 0,
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
        """
        self.X = torch.FloatTensor(X)
        self.asset_ids = torch.LongTensor(asset_ids)
        self.sequence_length = sequence_length
        self.masking_ratio = masking_ratio
        self.volatility_lookahead = volatility_lookahead
        self.smart_masking_prob = smart_masking_prob
        self.cross_asset_masking_prob = cross_asset_masking_prob
        self.price_column_idx = price_column_idx
        self.n_samples, self.seq_len, self.n_features = self.X.shape

        self._precompute_volatility_targets()

    def _precompute_volatility_targets(self) -> None:
        """Precompute volatility targets for all samples to speed up training."""
        self.volatility_targets = torch.zeros(self.n_samples, dtype=torch.float32)
        close_price_idx = min(3, self.n_features - 1)

        # Vectorized calculation would be complex due to sliding window on already windowed data
        # But we can optimize the loop significantly or use a rolling calculation if X was contiguous.
        # Since X is (N, seq_len, features), and we look at X[idx+1:idx+lookahead],
        # we are effectively looking at next samples.

        # We can accept a small startup cost for faster iteration.
        # Let's do a simple loop for now, but optimize inside.

        for idx in range(self.n_samples):
            future_end_idx = min(idx + 1 + self.volatility_lookahead, self.n_samples)
            if future_end_idx > idx + 5:
                # Direct slicing on the tensor is efficient enough for init time
                future_prices = self.X[
                    idx + 1 : future_end_idx,
                    :,
                    close_price_idx,
                ]

                if len(future_prices) > 5:
                    # Calculate returns along time dimension (dim 0 of the slice)
                    # future_prices shape: (lookahead, seq_len)
                    returns = future_prices[1:] - future_prices[:-1]
                    # std over all elements
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
        original_sequence = self.X[idx].clone()
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
        price_features = sequence[:, :4]
        
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
        price_feature_indices = [0, 1, 2, 3]
        
        for feat_idx in price_feature_indices:
            if np.random.random() < 0.15:
                num_positions = max(1, int(self.sequence_length * self.masking_ratio * 0.5))
                positions = np.random.choice(
                    self.sequence_length, size=num_positions, replace=False
                )
                for pos in positions:
                    mask_binary[pos] = True
        
        return mask_binary
