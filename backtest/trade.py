"""Trade record management."""

from dataclasses import dataclass
from typing import Optional

from .position import PositionSide


@dataclass
class Trade:
    """Represents a completed trade."""

    entry_timestamp: str
    exit_timestamp: str
    side: PositionSide
    entry_price: float
    exit_price: float
    size: float
    leverage: int
    pnl: float
    pnl_percentage: float
    entry_fee: float
    exit_fee: float
    funding_fees: float
    slippage: float
    exit_reason: str
    holding_periods: int
    pyramid_level: int = 0

    @property
    def total_fees(self) -> float:
        """Calculate total fees paid.

        Returns:
            Total fees in USD.
        """
        return self.entry_fee + self.exit_fee + self.funding_fees + self.slippage

    @property
    def net_pnl(self) -> float:
        """Calculate net PnL after all fees.

        Returns:
            Net PnL in USD.
        """
        return self.pnl - self.total_fees

    @property
    def is_profitable(self) -> bool:
        """Check if trade was profitable.

        Returns:
            True if net PnL is positive.
        """
        return self.net_pnl > 0
