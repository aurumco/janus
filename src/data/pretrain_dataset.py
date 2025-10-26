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

    def __len__(self) -> int:
        """Get dataset length.

        Returns:
            Number of valid samples.
        """
        return max(0, self.n_samples - self.volatility_lookahead)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single pre-training sample with masking.

        Args:
            idx: Sample index.

        Returns:
            Dictionary containing masked input, targets, and metadata.
        """
        original_sequence = self.X[idx].clone()
        asset_id = self.asset_ids[idx]
        
        num_to_mask = max(1, int(self.seq_len * self.masking_ratio))
        mask_binary = self._generate_smart_mask(original_sequence)
        
        masked_sequence = original_sequence.clone()
        masked_sequence[mask_binary] = 0.0
        
        future_end_idx = min(idx + 1 + self.volatility_lookahead, self.n_samples)
        if future_end_idx > idx + 1:
            future_prices = self.X[
                idx + 1 : future_end_idx,
                :,
                self.price_column_idx,
            ]
            
            if len(future_prices) > 1:
                log_returns = torch.log(
                    (future_prices[1:] + 1e-8) / (future_prices[:-1] + 1e-8)
                )
                volatility = torch.std(log_returns)
            else:
                volatility = torch.tensor(0.0)
        else:
            volatility = torch.tensor(0.0)

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
        
        if not (use_smart_masking or use_cross_asset):
            num_mask = max(1, int(self.sequence_length * self.masking_ratio))
            mask_positions = np.random.choice(
                self.sequence_length, size=num_mask, replace=False
            )
            mask_binary[mask_positions] = True
        
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
