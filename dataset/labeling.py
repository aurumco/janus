"""Labeling strategies for cryptocurrency price movements."""

import numpy as np
import pandas as pd
import pandas_ta as ta


class PriceLabelingStrategy:
    """Strategy for labeling price movements into classes."""

    def __init__(
        self,
        lookahead: int,
        sl_filter_pct: float,
        use_atr_stop: bool = True,
        atr_multiplier: float = 1.4,
        atr_period: int = 14,
    ) -> None:
        """Initialize labeling strategy.

        Args:
            lookahead: Number of candles to look ahead.
            sl_filter_pct: Fixed stop loss filter percentage (fallback).
            use_atr_stop: Whether to use ATR-based dynamic stop loss.
            atr_multiplier: Multiplier for ATR to calculate stop distance.
            atr_period: Period for ATR calculation.
        """
        self.lookahead = lookahead
        self.sl_filter_pct = sl_filter_pct
        self.use_atr_stop = use_atr_stop
        self.atr_multiplier = atr_multiplier
        self.atr_period = atr_period
        self.sl_neutralized_count = 0

    def label_hybrid_5class(self, ohlcv: pd.DataFrame) -> pd.Series:
        """Create hybrid 5-class labels based on price changes with ATR-based stops.

        Classes:
            0: Strong Sell (< -2.0%)
            1: Sell (-2.0% to -0.5%)
            2: Neutral (-0.5% to 0.5%)
            3: Buy (0.5% to 2.0%)
            4: Strong Buy (> 2.0%)

        Args:
            ohlcv: DataFrame with OHLCV data.

        Returns:
            Series with class labels.
        """
        n = len(ohlcv)
        close = ohlcv['close'].values
        high = ohlcv['high'].values
        low = ohlcv['low'].values

        if self.use_atr_stop:
            atr = ta.atr(
                high=ohlcv['high'],
                low=ohlcv['low'],
                close=ohlcv['close'],
                length=self.atr_period
            )
            atr_values = atr.bfill().fillna(self.sl_filter_pct).values
        else:
            atr_values = np.full(n, self.sl_filter_pct)

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

            if self.use_atr_stop:
                stop_distance = (atr_values[i] / current_close) * self.atr_multiplier
            else:
                stop_distance = self.sl_filter_pct

            price_change = (future_close - current_close) / current_close

            upward_excursion = (np.max(window_high) - current_close) / current_close
            downward_excursion = (current_close - np.min(window_low)) / current_close

            if price_change > 0 and downward_excursion > stop_distance:
                labels[i] = 2
                self.sl_neutralized_count += 1
                continue

            if price_change < 0 and upward_excursion > stop_distance:
                labels[i] = 2
                self.sl_neutralized_count += 1
                continue

            if price_change > 0.02:
                labels[i] = 4
            elif 0.005 < price_change <= 0.02:
                labels[i] = 3
            elif -0.005 <= price_change <= 0.005:
                labels[i] = 2
            elif -0.02 <= price_change < -0.005:
                labels[i] = 1
            elif price_change < -0.02:
                labels[i] = 0

        series = pd.Series(labels, index=ohlcv.index, dtype=np.int8)
        series.attrs['sl_neutralized'] = self.sl_neutralized_count

        return series
