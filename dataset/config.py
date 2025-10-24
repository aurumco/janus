"""Configuration parameters for dataset creation."""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class DatasetConfig:
    """Configuration for Janus multi-asset dataset creation."""

    dataset_dir: str = "dataset"
    asset_symbols: List[str] = None
    asset_mapping: dict = None
    
    # Mode configuration
    mode: str = "fine-tune"  # "pre-train", "fine-tune", or "both"
    
    # Timeframe configuration
    pretrain_timeframe: str = "1min"
    finetune_timeframe: str = "30min"
    asset: str = "Multi-Asset"
    base_timeframe: str = "30min"
    
    # Date ranges
    pretrain_start: str = "2022-01-01"
    pretrain_end: str = "2025-10-05"
    finetune_start: str = "2023-01-01"
    finetune_end: str = "2025-10-05"
    train_start: str = "2023-01-01"
    train_end: str = "2025-10-05"
    test_start: str = "2025-07-01"

    # Window configuration
    pretrain_input_window: int = 128
    finetune_input_window: int = 96
    input_window_candles: int = 96
    target_window_candles: int = 36  # 18 hours @ 30min
    stop_loss_pct: float = 0.004
    use_atr_stop_loss: bool = True
    atr_multiplier: float = 1.4
    atr_period: int = 14
    profit_thresholds: List[float] = None
    risk_reward_ratios: List[float] = None

    scaler_feature_range: Tuple[float, float] = (-1.0, 1.0)
    round_decimals: int = 5

    primary_rsi_length: int = 14
    primary_atr_length: int = 5
    primary_ema_length: int = 10
    
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

    output_dir: str = "."
    scaler_path: str = "janus_scaler.joblib"
    parquet_path: str = "janus_dataset.parquet"
    csv_path: str = "janus_dataset.csv"
    
    save_parquet: bool = True
    save_csv: bool = False

    min_required_bars: int = 120
    sl_filter_pct: float = 0.003

    def __post_init__(self) -> None:
        """Initialize default values after dataclass creation."""
        if self.asset_symbols is None:
            self.asset_symbols = [
                "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", 
                "XRPUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT",
                "UNIUSDT", "AAVEUSDT", "SUIUSDT", "ARBUSDT",
                "ENAUSDT", "PAXGUSDT", "ZECUSDT"
            ]
        
        if self.asset_mapping is None:
            self.asset_mapping = {symbol: idx for idx, symbol in enumerate(self.asset_symbols)}

        if self.risk_reward_ratios is None:
            self.risk_reward_ratios = [1.5, 2.0, 4.0]
        
        if self.profit_thresholds is None:
            base_sl = self.stop_loss_pct
            self.profit_thresholds = [
                base_sl * rr for rr in self.risk_reward_ratios
            ]
        
        self._configure_paths()
    
    def _configure_paths(self) -> None:
        """Configure output paths based on mode."""
        if self.mode == "pre-train":
            prefix = "janus_pretrain_1min"
            self.base_timeframe = self.pretrain_timeframe
            self.input_window_candles = self.pretrain_input_window
        elif self.mode == "fine-tune":
            prefix = "janus_finetune_30min"
            self.base_timeframe = self.finetune_timeframe
            self.input_window_candles = self.finetune_input_window
        else:  # both
            prefix = "janus_multi"
        
        self.scaler_path = f"{prefix}_scaler.joblib"
        self.parquet_path = f"{prefix}_dataset.parquet"
        self.csv_path = f"{prefix}_dataset.csv"
