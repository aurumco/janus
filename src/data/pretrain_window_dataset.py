"""Streaming pre-train dataset that builds windows on-the-fly from base memmaps."""

from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class PretrainWindowDataset(Dataset):
    """Dataset for SSL pre-training using base (timesteps x features) memmaps.

    Windows are created on-the-fly to avoid storing full (n_samples x seq_len x n_feat) arrays.
    """

    def __init__(
        self,
        features_memmap_path: str,
        asset_ids_memmap_path: Optional[str],
        n_timesteps: int,
        n_features: int,
        sequence_length: int,
        start_index: int,
        end_index: int,
        masking_ratio: float = 0.15,
        volatility_lookahead: int = 60,
        smart_masking_prob: float = 0.4,
        cross_asset_masking_prob: float = 0.3,
        price_column_idx: int = 0,
    ) -> None:
        self.features_mm = np.memmap(
            features_memmap_path, dtype=np.float32, mode="r", shape=(n_timesteps, n_features)
        )
        self.asset_ids_mm = (
            np.memmap(asset_ids_memmap_path, dtype=np.int64, mode="r", shape=(n_timesteps,))
            if asset_ids_memmap_path
            else np.zeros(n_timesteps, dtype=np.int64)
        )
        
        try:
            import mmap
            import ctypes
            libc = ctypes.CDLL('libc.so.6')
            advise_flag = getattr(mmap, 'MADV_RANDOM', None)
            if advise_flag is not None:
                libc.madvise(self.features_mm.ctypes.data, self.features_mm.nbytes, advise_flag)
        except Exception:
            pass
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.sequence_length = sequence_length
        self.start_index = start_index
        self.end_index = end_index
        self.n_samples = max(0, min(end_index, n_timesteps - sequence_length + 1) - start_index)

        self.masking_ratio = masking_ratio
        self.volatility_lookahead = volatility_lookahead
        self.smart_masking_prob = smart_masking_prob
        self.cross_asset_masking_prob = cross_asset_masking_prob
        self.price_column_idx = price_column_idx

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        i = self.start_index + idx
        end = i + self.sequence_length

        window_view = self.features_mm[i:end, :]
        if window_view.dtype != np.float32:
            window_view = window_view.astype(np.float32)
        original_sequence = torch.tensor(window_view, dtype=torch.float32)
        try:
            import ctypes, mmap
            libc = ctypes.CDLL('libc.so.6')
            addr = window_view.ctypes.data
            nbytes = int(window_view.nbytes)
            if hasattr(mmap, 'MADV_DONTNEED'):
                libc.madvise(addr, nbytes, mmap.MADV_DONTNEED)
        except Exception:
            pass
        del window_view
        asset_id = torch.tensor(int(self.asset_ids_mm[end - 1]), dtype=torch.long)

        mask_binary = self._generate_smart_mask(original_sequence)
        masked_sequence = original_sequence.clone()
        masked_sequence[mask_binary] = 0.0

        future_end_idx = min(end + self.volatility_lookahead, self.n_timesteps)
        if future_end_idx > end:
            prices_view = self.features_mm[end:future_end_idx, self.price_column_idx]
            if len(prices_view) > 1:
                prices_arr = np.array(prices_view, dtype=np.float32, copy=False)
                log_returns = np.log((prices_arr[1:] + 1e-8) / (prices_arr[:-1] + 1e-8))
                volatility = torch.tensor(float(np.std(log_returns)), dtype=torch.float32)
            else:
                volatility = torch.tensor(0.0, dtype=torch.float32)
            try:
                import ctypes, mmap
                libc = ctypes.CDLL('libc.so.6')
                addr = prices_view.ctypes.data
                nbytes = int(prices_view.nbytes)
                if hasattr(mmap, 'MADV_DONTNEED'):
                    libc.madvise(addr, nbytes, mmap.MADV_DONTNEED)
            except Exception:
                pass
            del prices_view
        else:
            volatility = torch.tensor(0.0, dtype=torch.float32)

        return {
            "input_sequence": masked_sequence,
            "mask_binary": mask_binary,
            "original_sequence": original_sequence,
            "volatility_target": volatility.unsqueeze(0),
            "asset_id": asset_id,
        }

    def _generate_smart_mask(self, sequence: torch.Tensor) -> torch.Tensor:
        mask_binary = torch.zeros(self.sequence_length, dtype=torch.bool)
        use_smart = np.random.random() < self.smart_masking_prob
        use_cross = np.random.random() < self.cross_asset_masking_prob

        if use_smart:
            mask_binary = self._volatility_aware_mask(sequence, mask_binary)
        if use_cross:
            mask_binary = self._cross_asset_mask(sequence, mask_binary)
        if not (use_smart or use_cross):
            num_mask = max(1, int(self.sequence_length * self.masking_ratio))
            pos = np.random.choice(self.sequence_length, size=num_mask, replace=False)
            mask_binary[pos] = True
        return mask_binary

    def _volatility_aware_mask(self, sequence: torch.Tensor, mask_binary: torch.Tensor) -> torch.Tensor:
        price_features = sequence[:, :4]
        price_volatility = torch.std(price_features, dim=1)
        high_vol_threshold = torch.quantile(price_volatility, 0.8)
        idx = (price_volatility > high_vol_threshold).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            m = idx[np.random.randint(len(idx))]
            length = np.random.randint(1, 4)
            end_idx = min(m + length, self.sequence_length)
            mask_binary[m:end_idx] = True
        return mask_binary

    def _cross_asset_mask(self, sequence: torch.Tensor, mask_binary: torch.Tensor) -> torch.Tensor:
        price_indices = [0, 1, 2, 3]
        for k in price_indices:
            if np.random.random() < 0.15:
                num_pos = max(1, int(self.sequence_length * self.masking_ratio * 0.5))
                positions = np.random.choice(self.sequence_length, size=num_pos, replace=False)
                for pos in positions:
                    mask_binary[pos] = True
        return mask_binary
