"""Position management for futures trading."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PositionSide(Enum):
    """Position side enumeration."""
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Position:
    """Represents a futures trading position."""

    side: PositionSide
    entry_price: float
    size: float
    leverage: int
    entry_timestamp: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    pyramid_level: int = 0
    entry_fee: float = 0.0
    funding_paid: float = 0.0

    @property
    def notional_value(self) -> float:
        """Calculate notional value of position.

        Returns:
            Notional value in USD.
        """
        return self.size * self.entry_price * self.leverage

    @property
    def margin_used(self) -> float:
        """Calculate margin used for position.

        Returns:
            Margin in USD.
        """
        return self.size * self.entry_price

    def calculate_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL.

        Args:
            current_price: Current market price.

        Returns:
            Unrealized PnL in USD.
        """
        if self.side == PositionSide.LONG:
            price_change = current_price - self.entry_price
        else:
            price_change = self.entry_price - current_price

        pnl = price_change * self.size * self.leverage
        return pnl

    def calculate_pnl_percentage(self, current_price: float) -> float:
        """Calculate PnL as percentage of margin.

        Args:
            current_price: Current market price.

        Returns:
            PnL percentage.
        """
        pnl = self.calculate_pnl(current_price)
        return (pnl / self.margin_used) * 100 if self.margin_used > 0 else 0.0

    def update_trailing_stop(self, current_price: float, distance_pct: float) -> None:
        """Update trailing stop loss.

        Args:
            current_price: Current market price.
            distance_pct: Distance from current price as percentage.
        """
        if self.side == PositionSide.LONG:
            new_stop = current_price * (1 - distance_pct)
            if self.trailing_stop is None or new_stop > self.trailing_stop:
                self.trailing_stop = new_stop
        else:
            new_stop = current_price * (1 + distance_pct)
            if self.trailing_stop is None or new_stop < self.trailing_stop:
                self.trailing_stop = new_stop

    def check_stop_loss_hit(self, low: float, high: float) -> bool:
        """Check if stop loss was hit.

        Args:
            low: Period low price.
            high: Period high price.

        Returns:
            True if stop loss was hit.
        """
        if self.stop_loss is None and self.trailing_stop is None:
            return False

        if self.side == PositionSide.LONG:
            stop_price = max(
                self.stop_loss or 0,
                self.trailing_stop or 0
            )
            return low <= stop_price
        else:
            stop_price = min(
                self.stop_loss or float('inf'),
                self.trailing_stop or float('inf')
            )
            return high >= stop_price

    def check_take_profit_hit(self, low: float, high: float) -> bool:
        """Check if take profit was hit.

        Args:
            low: Period low price.
            high: Period high price.

        Returns:
            True if take profit was hit.
        """
        if self.take_profit is None:
            return False

        if self.side == PositionSide.LONG:
            return high >= self.take_profit
        else:
            return low <= self.take_profit
