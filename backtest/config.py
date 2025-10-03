"""Configuration for backtesting system."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BacktestConfig:
    """Configuration parameters for backtesting."""

    initial_capital_usd: float = 6000000.0
    commission_rate: float = 0.001
    leverage: int = 5
    trades_per_day_min: int = 1
    trades_per_day_max: int = 10
    backtest_start_date: str = '2025-08-01'
    backtest_end_date: str = '2025-09-30'
    risk_per_trade: float = 0.1
    stop_loss_pct: float = 0.01
    risk_reward_ratio: float = 10.0

    slippage_pct: float = 0.0005
    maker_fee: float = 0.0002
    taker_fee: float = 0.0005
    funding_rate_hourly: float = 0.00001

    position_size_pct: float = 0.95
    max_position_size_pct: float = 1.0
    min_position_size_usd: float = 10.0

    allow_pyramiding: bool = True
    max_pyramid_levels: int = 3
    pyramid_profit_threshold: float = 0.01
    pyramid_size_multiplier: float = 1.5

    use_trailing_stop: bool = True
    trailing_stop_activation_pct: float = 0.015
    trailing_stop_distance_pct: float = 0.008

    max_daily_loss_pct: float = 0.05
    max_drawdown_pct: float = 0.20

    close_on_trend_reversal: bool = True
    close_on_neutral: bool = True
    allow_long: bool = True
    allow_short: bool = True

    confidence_threshold: float = 0.6
    min_holding_periods: int = 1
    max_holding_periods: int = 100

    compound_profits: bool = True
    reinvest_profits: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.initial_capital_usd <= 0:
            raise ValueError("Initial capital must be positive")
        if not 0 <= self.commission_rate < 1:
            raise ValueError("Commission rate must be between 0 and 1")
        if self.leverage < 1:
            raise ValueError("Leverage must be at least 1")
