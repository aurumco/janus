"""Technical indicator calculation utilities."""

import numpy as np
import pandas as pd


class IndicatorCalculator:
    """Calculates technical indicators for cryptocurrency data."""

    @staticmethod
    def calculate_rsi(data: pd.Series, length: int) -> pd.Series:
        delta = data.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
        avg_loss = loss.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(method='bfill')

    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
        return atr

    @staticmethod
    def calculate_ema(data: pd.Series, length: int) -> pd.Series:
        return data.ewm(span=length, adjust=False, min_periods=length).mean()

    @staticmethod
    def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        tr = IndicatorCalculator.calculate_atr(high, low, close, 1)
        atr = tr.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/length, adjust=False, min_periods=length).mean() / atr.replace(0, np.nan))
        minus_di = 100 * (minus_dm.ewm(alpha=1/length, adjust=False, min_periods=length).mean() / atr.replace(0, np.nan))
        dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).abs()).replace([np.inf, -np.inf], np.nan)
        adx = dx.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
        return adx.fillna(method='bfill')

    @staticmethod
    def calculate_pvo(volume: pd.Series, fast: int, slow: int, signal: int) -> pd.Series:
        ema_fast = volume.ewm(span=fast, adjust=False, min_periods=fast).mean()
        ema_slow = volume.ewm(span=slow, adjust=False, min_periods=slow).mean()
        pvo = 100 * (ema_fast - ema_slow) / ema_slow.replace(0, np.nan)
        signal_line = pvo.ewm(span=signal, adjust=False, min_periods=signal).mean()
        return signal_line

    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        direction = np.sign(close.diff().fillna(0))
        obv = (direction * volume).fillna(0).cumsum()
        return obv

    @staticmethod
    def calculate_garch_volatility(returns: pd.Series) -> float:
        if returns.isnull().any() or len(returns) < 50:
            return np.nan
        try:
            from arch import arch_model
        except Exception:
            return np.nan
        try:
            model = arch_model(returns * 100, vol='Garch', p=1, q=1, rescale=False)
            result = model.fit(disp='off', show_warning=False)
            if getattr(result, 'convergence_flag', 1) == 0:
                forecast = result.forecast(horizon=1)
                return float(np.sqrt(forecast.variance.iloc[-1, 0]))
            return np.nan
        except Exception:
            return np.nan

    @staticmethod
    def calculate_cyclical_time_features(timestamps: pd.DatetimeIndex, period: int = 24, time_component: str = 'hour') -> tuple[pd.Series, pd.Series]:
        if time_component == 'dayofweek':
            time_values = timestamps.dayofweek
        else:
            time_values = timestamps.hour
        sin_values = np.sin(2 * np.pi * time_values / period)
        cos_values = np.cos(2 * np.pi * time_values / period)
        return pd.Series(sin_values, index=timestamps), pd.Series(cos_values, index=timestamps)
