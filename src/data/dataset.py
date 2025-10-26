"""PyTorch dataset implementation for Bitcoin trend prediction.

This module maintains backward compatibility while delegating to FineTuneDataset.
"""

from .finetune_dataset import FineTuneDataset

# Backward compatibility alias
BitcoinTrendDataset = FineTuneDataset
