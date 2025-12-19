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

        # Optimization: Use batched vectorization with torch.unfold
        # We need at least 'lookahead' samples to perform the vectorized sliding window.
        # For indices where idx + 1 + lookahead <= n_samples, we can vectorize.
        cutoff = self.n_samples - self.volatility_lookahead - 1

        if cutoff <= 0:
            # Dataset too small for lookahead, fall back to loop everywhere
            self._compute_volatility_loop(0, self.n_samples, close_price_idx)
            return

        # 1. Vectorized chunk
        # Process in batches to keep memory usage low
        batch_size = 1024

        # Extract the relevant column once to avoid repeated slicing overhead
        # Shape: (n_samples, seq_len)
        all_prices = self.X[:, :, close_price_idx]

        for start_idx in range(0, cutoff, batch_size):
            end_idx = min(start_idx + batch_size, cutoff)

            # We need prices from start_idx + 1 to end_idx + lookahead
            # Slice range: [start_idx + 1, end_idx + lookahead] (exclusive end)
            window_slice = all_prices[start_idx + 1 : end_idx + self.volatility_lookahead]

            # Use unfold to create sliding windows
            # input shape: (batch + lookahead - 1, seq_len)
            # unfold output shape: (batch, seq_len, lookahead)
            windows_unfolded = window_slice.unfold(0, self.volatility_lookahead, 1)

            # Permute to (batch, lookahead, seq_len) so slicing matches logic
            windows = windows_unfolded.permute(0, 2, 1)

            # Calculate returns: diff along the time dimension (dim 1 of windows)
            # shape: (batch, lookahead-1, seq_len)
            returns = windows[:, 1:] - windows[:, :-1]

            # std over (lookahead-1, seq_len) -> dims (1, 2)
            stds = torch.std(returns, dim=(1, 2))

            self.volatility_targets[start_idx:end_idx] = stds + 1e-6

        # 2. Tail chunk (process remaining samples with loop)
        if cutoff < self.n_samples:
            self._compute_volatility_loop(cutoff, self.n_samples, close_price_idx)

    def _compute_volatility_loop(self, start_idx: int, end_idx: int, close_price_idx: int) -> None:
        """Fallback loop for volatility computation."""
        for idx in range(start_idx, end_idx):
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
