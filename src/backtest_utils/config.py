"""Backtest configuration."""

from dataclasses import dataclass
from typing import Optional

@dataclass
class BacktestConfig:
    """Configuration for backtest run."""
    initial_capital_usd: float
    leverage: int
    backtest_start_date: str
    backtest_end_date: str
    maker_fee: float = 0.0002
    taker_fee: float = 0.0004
    slippage: float = 0.0001
