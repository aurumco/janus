"""Data processing pipeline for multi-timeframe cryptocurrency data."""

from typing import Dict, List

import pandas as pd

from .config import DatasetConfig
from .indicators import IndicatorCalculator


class MultiTimeframeProcessor:
    """Processes cryptocurrency data across multiple timeframes."""

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize processor.

        Args:
            config: Dataset configuration.
        """
        self.config = config
        self.indicator_calc = IndicatorCalculator()

    def load_and_resample(self, csv_path: str) -> Dict[str, pd.DataFrame]:
        """Load raw data and create multiple timeframe DataFrames.

        Args:
            csv_path: Path to raw CSV file.

        Returns:
            Dictionary with timeframe DataFrames.
        """
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.lower()

        timestamps = pd.to_numeric(df['timestamp'], errors='coerce')
        unit = 'ms' if timestamps.max() > 1e12 else 's'
        df['timestamp'] = pd.to_datetime(timestamps, unit=unit, errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
        df.set_index('timestamp', inplace=True)

        df_filtered = df[
            (df.index >= self.config.train_start) &
            (df.index <= self.config.train_end)
        ]

        if df_filtered.empty:
            raise ValueError(f"No data found between {self.config.train_start} and {self.config.train_end}")

        ohlcv_logic = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }

        timeframes = {
            'm15': df_filtered.resample(self.config.base_timeframe).apply(ohlcv_logic).dropna(),
            'h1': df_filtered.resample('1h').apply(ohlcv_logic).dropna(),
            'h4': df_filtered.resample('4h').apply(ohlcv_logic).dropna(),
            'daily': df_filtered.resample('D').apply(ohlcv_logic).dropna(),
        }

        for tf_name, tf_df in timeframes.items():
            timeframes[tf_name] = tf_df[
                (tf_df.index >= self.config.train_start) &
                (tf_df.index <= self.config.train_end)
            ]

        return timeframes

    def calculate_higher_timeframe_features(
        self,
        df_h1: pd.DataFrame,
        df_h4: pd.DataFrame,
        df_daily: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculate features from higher timeframes.

        Args:
            df_h1: Hourly DataFrame.
            df_h4: 4-hour DataFrame.
            df_daily: Daily DataFrame.

        Returns:
            H1 DataFrame with all higher timeframe features.
        """
        ema21_h4 = self.indicator_calc.calculate_ema(df_h4['close'], self.config.h4_ema_length)
        df_h4['EMA_diff_21_H4pct'] = (df_h4['close'] - ema21_h4) / df_h4['close']
        df_h4['RSI_H4'] = self.indicator_calc.calculate_rsi(df_h4['close'], self.config.h4_rsi_length)
        df_h4['ADX_H4'] = self.indicator_calc.calculate_adx(
            df_h4['high'], df_h4['low'], df_h4['close'], self.config.h4_adx_length
        )

        df_daily['RSI_Daily'] = self.indicator_calc.calculate_rsi(
            df_daily['close'], self.config.daily_rsi_length
        )

        h4_features = df_h4[['EMA_diff_21_H4pct', 'RSI_H4']]
        daily_features = df_daily[['RSI_Daily']]

        df_h1 = df_h1.join(h4_features).join(daily_features)
        df_h1 = df_h1.ffill()

        df_h1['log_return'] = pd.Series(
            df_h1['close'] / df_h1['close'].shift(1)
        ).apply(lambda x: 0 if x <= 0 else pd.np.log(x))

        df_h1['RSI_14_H1'] = self.indicator_calc.calculate_rsi(df_h1['close'], self.config.h1_rsi_length)
        df_h1['ADX_14_H1'] = self.indicator_calc.calculate_adx(
            df_h1['high'], df_h1['low'], df_h1['close'], self.config.h1_adx_length
        )

        if self.config.enable_garch and len(df_h1) >= self.config.garch_roll_window:
            df_h1['garch_volatility'] = df_h1['log_return'].rolling(
                window=self.config.garch_roll_window
            ).apply(self.indicator_calc.calculate_garch_volatility, raw=False)
        else:
            df_h1['garch_volatility'] = pd.Series(dtype=float)

        essential_cols = ['log_return', 'RSI_14_H1', 'ADX_14_H1']
        df_h1[essential_cols] = df_h1[essential_cols].ffill().bfill()

        if self.config.enable_garch:
            df_h1['garch_volatility'] = df_h1['garch_volatility'].ffill().bfill()

        return df_h1

    def calculate_m15_features(self, df_m15: pd.DataFrame) -> pd.DataFrame:
        """Calculate M15 timeframe features.

        Args:
            df_m15: M15 DataFrame.

        Returns:
            M15 DataFrame with calculated features.
        """
        df_m15['RSI_14_M15'] = self.indicator_calc.calculate_rsi(
            df_m15['close'], self.config.m15_rsi_length
        )

        atr = self.indicator_calc.calculate_atr(
            df_m15['high'], df_m15['low'], df_m15['close'], self.config.m15_atr_length
        )
        df_m15['ATR_5_pct_M15'] = atr / df_m15['close']

        ema10 = self.indicator_calc.calculate_ema(df_m15['close'], self.config.m15_ema_length)
        df_m15['dist_from_ema_10_M15'] = (df_m15['close'] - ema10) / df_m15['close']
        df_m15['ema10_slope_M15'] = ema10.diff(self.config.ema_slope_lag)

        df_m15['volume_oscillator_M15'] = self.indicator_calc.calculate_pvo(
            df_m15['volume'], self.config.pvo_fast, self.config.pvo_slow, self.config.pvo_signal
        )

        df_m15['obv_M15'] = self.indicator_calc.calculate_obv(df_m15['close'], df_m15['volume'])

        hour_sin, hour_cos = self.indicator_calc.calculate_cyclical_time_features(
            df_m15.index, self.config.hour_period
        )
        df_m15['hour_of_day'] = df_m15.index.hour
        df_m15['hour_sin'] = hour_sin
        df_m15['hour_cos'] = hour_cos

        return df_m15

    def merge_timeframes(
        self,
        df_m15: pd.DataFrame,
        df_h1: pd.DataFrame,
        df_h4: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge features from all timeframes into M15 base.

        Args:
            df_m15: M15 DataFrame with features.
            df_h1: H1 DataFrame with features.
            df_h4: H4 DataFrame with features.

        Returns:
            Merged M15 DataFrame.
        """
        h1_cols = ['RSI_14_H1', 'ADX_14_H1', 'garch_volatility']
        h4_cols = ['EMA_diff_21_H4pct']

        h1_aligned = df_h1[h1_cols].reindex(df_m15.index).ffill().bfill()
        h4_aligned = df_h4[h4_cols].reindex(df_m15.index).ffill().bfill()

        df_m15 = df_m15.join(h1_aligned).join(h4_aligned)

        ny_hours_mask = df_m15['hour_of_day'].between(
            self.config.ny_hours_start, self.config.ny_hours_end
        ).astype(int)
        df_m15['RSI_15_x_NYHours'] = df_m15['RSI_14_M15'] * ny_hours_mask

        return df_m15

    def get_feature_columns(self) -> List[str]:
        """Get list of feature column names.

        Returns:
            List of feature column names.
        """
        features = [
            'RSI_14_M15', 'ATR_5_pct_M15', 'dist_from_ema_10_M15', 'ema10_slope_M15',
            'volume_oscillator_M15', 'obv_M15',
            'hour_sin', 'hour_cos',
            'RSI_14_H1', 'ADX_14_H1', 'EMA_diff_21_H4pct',
            'RSI_15_x_NYHours',
        ]

        if self.config.enable_garch:
            features.append('garch_volatility')

        return features
