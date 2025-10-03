"""Labeling strategies for cryptocurrency price movements."""

import numpy as np
import pandas as pd


class PriceLabelingStrategy:
    """Strategy for labeling price movements into classes."""

    def __init__(self, lookahead: int, sl_filter_pct: float) -> None:
        """Initialize labeling strategy.

        Args:
            lookahead: Number of candles to look ahead.
            sl_filter_pct: Stop loss filter percentage.
        """
        self.lookahead = lookahead
        self.sl_filter_pct = sl_filter_pct
        self.sl_neutralized_count = 0

    def label_hybrid_5class(self, ohlcv: pd.DataFrame) -> pd.Series:
        """Create hybrid 5-class labels based on price changes.

        Classes:
            0: Strong Sell (< -2%)
            1: Sell (-2% to -0.5%)
            2: Neutral (-0.5% to 0.5%)
            3: Buy (0.5% to 2%)
            4: Strong Buy (> 2%)

        Args:
            ohlcv: DataFrame with OHLCV data.

        Returns:
            Series with class labels.
        """
        n = len(ohlcv)
        close = ohlcv['close'].values
        high = ohlcv['high'].values
        low = ohlcv['low'].values

        labels = np.full(n, 2, dtype=np.int8)
        self.sl_neutralized_count = 0

        for i in range(n - self.lookahead):
            current_close = close[i]
            future_close = close[i + self.lookahead]

            window_start = i + 1
            window_end = i + self.lookahead

            window_high = high[window_start:window_end + 1]
            window_low = low[window_start:window_end + 1]

            if len(window_high) == 0:
                continue

            price_change_pct = (future_close - current_close) / current_close * 100.0

            upward_excursion = (np.max(window_high) - current_close) / current_close
            downward_excursion = (current_close - np.min(window_low)) / current_close

            if price_change_pct > 0 and downward_excursion > self.sl_filter_pct:
                labels[i] = 2
                self.sl_neutralized_count += 1
                continue

            if price_change_pct < 0 and upward_excursion > self.sl_filter_pct:
                labels[i] = 2
                self.sl_neutralized_count += 1
                continue

            if price_change_pct > 2.0:
                labels[i] = 4
            elif 0.5 < price_change_pct <= 2.0:
                labels[i] = 3
            elif -0.5 <= price_change_pct <= 0.5:
                labels[i] = 2
            elif -2.0 <= price_change_pct < -0.5:
                labels[i] = 1
            elif price_change_pct < -2.0:
                labels[i] = 0

        series = pd.Series(labels, index=ohlcv.index, dtype=np.int8)
        series.attrs['sl_neutralized'] = self.sl_neutralized_count

        return series
