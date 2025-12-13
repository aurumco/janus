"""Backtest engine."""

import pandas as pd
import numpy as np
from .config import BacktestConfig

class BacktestEngine:
    def __init__(self, config: BacktestConfig):
        self.config = config

    def run(self, ohlcv_data: pd.DataFrame, signals: np.ndarray) -> dict:
        """Run backtest simulation."""
        # Dummy implementation since we only want to optimize prediction generation
        return {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0
        }
