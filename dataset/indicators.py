"""Technical indicator calculation utilities."""

from typing import Optional

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except ImportError:
    print("Warning: pandas_ta not installed. Install with: pip install pandas-ta")
    ta = None

try:
    from arch import arch_model
except ImportError:
    print("Warning: arch not installed. Install with: pip install arch")
    arch_model = None


class IndicatorCalculator:
    """Calculates technical indicators for cryptocurrency data."""

    @staticmethod
    def calculate_rsi(data: pd.Series, length: int) -> pd.Series:
        """Calculate Relative Strength Index.

        Args:
            data: Price series.
            length: RSI period.

        Returns:
            RSI values.
        """
        return ta.rsi(data, length=length)

    @staticmethod
    def calculate_atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int
    ) -> pd.Series:
        """Calculate Average True Range.

        Args:
            high: High prices.
            low: Low prices.
            close: Close prices.
            length: ATR period.

        Returns:
            ATR values.
        """
        return ta.atr(high=high, low=low, close=close, length=length)

    @staticmethod
    def calculate_ema(data: pd.Series, length: int) -> pd.Series:
        """Calculate Exponential Moving Average.

        Args:
            data: Price series.
            length: EMA period.

        Returns:
            EMA values.
        """
        return ta.ema(data, length=length)

    @staticmethod
    def calculate_adx(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int
    ) -> pd.Series:
        """Calculate Average Directional Index.

        Args:
            high: High prices.
            low: Low prices.
            close: Close prices.
            length: ADX period.

        Returns:
            ADX values.
        """
        adx_df = ta.adx(high=high, low=low, close=close, length=length)
        return adx_df[f'ADX_{length}']

    @staticmethod
    def calculate_pvo(
        volume: pd.Series,
        fast: int,
        slow: int,
        signal: int
    ) -> pd.Series:
        """Calculate Percentage Volume Oscillator.

        Args:
            volume: Volume series.
            fast: Fast period.
            slow: Slow period.
            signal: Signal period.

        Returns:
            PVO values.
        """
        pvo = ta.pvo(volume=volume, fast=fast, slow=slow, signal=signal)
        return pvo.iloc[:, 0] if pvo is not None else pd.Series(np.nan, index=volume.index)

    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume.

        Args:
            close: Close prices.
            volume: Volume series.

        Returns:
            OBV values.
        """
        return ta.obv(close=close, volume=volume)

    @staticmethod
    def calculate_garch_volatility(returns: pd.Series) -> float:
        """Calculate GARCH(1,1) volatility forecast.

        Args:
            returns: Return series.

        Returns:
            Forecasted volatility or NaN if calculation fails.
        """
        if returns.isnull().any() or len(returns) < 50:
            return np.nan

        try:
            model = arch_model(returns * 100, vol='Garch', p=1, q=1, rescale=False)
            result = model.fit(disp='off', show_warning=False)

            if result.convergence_flag == 0:
                forecast = result.forecast(horizon=1)
                return np.sqrt(forecast.variance.iloc[-1, 0])
            else:
                return np.nan
        except Exception:
            return np.nan

    @staticmethod
    def calculate_cyclical_time_features(
        timestamps: pd.DatetimeIndex,
        period: int = 24
    ) -> tuple[pd.Series, pd.Series]:
        """Calculate cyclical time features using sine and cosine.

        Args:
            timestamps: DateTime index.
            period: Period for cyclical encoding (default 24 for hours).

        Returns:
            Tuple of (sin_values, cos_values).
        """
        hour_of_day = timestamps.hour
        sin_values = np.sin(2 * np.pi * hour_of_day / period)
        cos_values = np.cos(2 * np.pi * hour_of_day / period)

        return pd.Series(sin_values, index=timestamps), pd.Series(cos_values, index=timestamps)
