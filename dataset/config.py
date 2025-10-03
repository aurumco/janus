"""Configuration parameters for dataset creation."""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class DatasetConfig:
    """Configuration for Janus dataset creation."""

    asset: str = "BTC/USDT"
    base_timeframe: str = "15min"
    train_start: str = "2023-01-01"
    train_end: str = "2025-06-30"
    test_start: str = "2025-07-01"

    input_window_candles: int = 64
    target_window_candles: int = 20
    stop_loss_pct: float = 0.004
    use_atr_stop_loss: bool = True
    atr_multiplier: float = 1.4
    atr_period: int = 14
    profit_thresholds: List[float] = None
    risk_reward_ratios: List[float] = None

    scaler_feature_range: Tuple[float, float] = (-1.0, 1.0)
    round_decimals: int = 5

    m15_rsi_length: int = 14
    m15_atr_length: int = 5
    m15_ema_length: int = 10
    h1_rsi_length: int = 14
    h1_adx_length: int = 14
    h4_ema_length: int = 21
    h4_adx_length: int = 14
    h4_rsi_length: int = 14
    daily_rsi_length: int = 14

    ema_slope_lag: int = 1
    hour_period: int = 24

    garch_roll_window: int = 240
    enable_garch: bool = True

    pvo_fast: int = 12
    pvo_slow: int = 26
    pvo_signal: int = 9

    ny_hours_start: int = 13
    ny_hours_end: int = 17

    scaler_path: str = "janus_m15_scaler.joblib"
    parquet_path: str = "janus_m15_dataset.parquet"
    csv_path: str = "janus_m15_dataset.csv"

    min_required_bars: int = 120
    sl_filter_pct: float = 0.003

    def __post_init__(self) -> None:
        """Initialize default values after dataclass creation."""
        if self.risk_reward_ratios is None:
            self.risk_reward_ratios = [1.5, 2.0, 4.0]
        
        if self.profit_thresholds is None:
            base_sl = self.stop_loss_pct
            self.profit_thresholds = [
                base_sl * rr for rr in self.risk_reward_ratios
            ]
