"""Backtesting engine for futures trading strategies."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pandas_ta_classic as ta
from rich.console import Console
from rich.progress import track

from .config import BacktestConfig
from .position import Position, PositionSide
from .trade import Trade


class BacktestEngine:
    """Executes backtests for futures trading strategies."""

    def __init__(self, config: BacktestConfig) -> None:
        """Initialize backtest engine.

        Args:
            config: Backtest configuration.
        """
        self.config = config
        self.console = Console()
        
        self.capital = config.initial_capital_usd
        self.initial_capital = config.initial_capital_usd
        self.peak_capital = config.initial_capital_usd
        
        self.current_position: Optional[Position] = None
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.timestamps: List[str] = []
        
        self.total_fees_paid = 0.0
        self.total_funding_paid = 0.0
        self.total_slippage_paid = 0.0
        
        self.daily_pnl: Dict[str, float] = {}
        self.max_drawdown = 0.0
        self.max_drawdown_duration = 0

    def calculate_position_size(self, price: float) -> float:
        """Calculate position size based on available capital.

        Args:
            price: Current market price.

        Returns:
            Position size in base currency.
        """
        available_capital = self.capital * self.config.position_size_pct
        max_capital = self.capital * self.config.max_position_size_pct
        
        position_value = min(available_capital, max_capital)
        size = position_value / price
        
        min_size = self.config.min_position_size_usd / price
        return max(size, min_size)

    def calculate_fees(self, size: float, price: float, is_maker: bool = False) -> float:
        """Calculate trading fees.

        Args:
            size: Position size.
            price: Execution price.
            is_maker: Whether order is maker or taker.

        Returns:
            Fee amount in USD.
        """
        notional = size * price
        fee_rate = self.config.maker_fee if is_maker else self.config.taker_fee
        return notional * fee_rate

    def calculate_slippage(self, size: float, price: float) -> float:
        """Calculate slippage cost.

        Args:
            size: Position size.
            price: Execution price.

        Returns:
            Slippage cost in USD.
        """
        notional = size * price
        return notional * self.config.slippage_pct

    def calculate_funding_fee(self, position: Position, periods: int) -> float:
        """Calculate funding fees for holding position.

        Args:
            position: Current position.
            periods: Number of periods held.

        Returns:
            Funding fee in USD.
        """
        return position.notional_value * self.config.funding_rate_hourly * periods

    def open_position(
        self,
        side: PositionSide,
        price: float,
        timestamp: str,
        atr_value: float,
        pyramid_level: int = 0,
    ) -> None:
        """Open a new position.

        Args:
            side: Position side (LONG/SHORT).
            price: Entry price.
            timestamp: Entry timestamp.
            atr_value: Current ATR value for dynamic stops.
            pyramid_level: Pyramid level (0 for initial position).
        """
        size = self.calculate_position_size(price)
        
        if pyramid_level > 0:
            size *= self.config.pyramid_size_multiplier
        
        entry_fee = self.calculate_fees(size, price, is_maker=False)
        slippage = self.calculate_slippage(size, price)
        
        if self.config.use_atr_stop and atr_value > 0:
            stop_distance = atr_value * self.config.atr_multiplier
        else:
            stop_distance = price * self.config.stop_loss_pct
        
        if side == PositionSide.LONG:
            stop_loss = price - stop_distance
            take_profit = price + (stop_distance * self.config.risk_reward_ratio)
        else:
            stop_loss = price + stop_distance
            take_profit = price - (stop_distance * self.config.risk_reward_ratio)
        
        self.current_position = Position(
            side=side,
            entry_price=price,
            size=size,
            leverage=self.config.leverage,
            entry_timestamp=timestamp,
            stop_loss=stop_loss,
            take_profit=take_profit,
            pyramid_level=pyramid_level,
            entry_fee=entry_fee,
        )
        
        self.capital -= (entry_fee + slippage)
        self.total_fees_paid += entry_fee
        self.total_slippage_paid += slippage

    def close_position(
        self,
        price: float,
        timestamp: str,
        reason: str,
        periods_held: int,
    ) -> None:
        """Close current position.

        Args:
            price: Exit price.
            timestamp: Exit timestamp.
            reason: Reason for closing.
            periods_held: Number of periods position was held.
        """
        if self.current_position is None:
            return
        
        pnl = self.current_position.calculate_pnl(price)
        pnl_pct = self.current_position.calculate_pnl_percentage(price)
        
        exit_fee = self.calculate_fees(
            self.current_position.size,
            price,
            is_maker=False
        )
        slippage = self.calculate_slippage(self.current_position.size, price)
        funding_fee = self.calculate_funding_fee(self.current_position, periods_held)
        
        trade = Trade(
            entry_timestamp=self.current_position.entry_timestamp,
            exit_timestamp=timestamp,
            side=self.current_position.side,
            entry_price=self.current_position.entry_price,
            exit_price=price,
            size=self.current_position.size,
            leverage=self.config.leverage,
            pnl=pnl,
            pnl_percentage=pnl_pct,
            entry_fee=self.current_position.entry_fee,
            exit_fee=exit_fee,
            funding_fees=funding_fee,
            slippage=slippage,
            exit_reason=reason,
            holding_periods=periods_held,
            pyramid_level=self.current_position.pyramid_level,
        )
        
        self.capital += pnl - exit_fee - slippage - funding_fee
        self.total_fees_paid += exit_fee
        self.total_slippage_paid += slippage
        self.total_funding_paid += funding_fee
        
        self.trades.append(trade)
        self.current_position = None
        
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital

    def should_close_position(
        self,
        current_signal: int,
        low: float,
        high: float,
    ) -> Tuple[bool, str]:
        """Determine if position should be closed.

        Args:
            current_signal: Current model prediction (0-4).
            low: Period low price.
            high: Period high price.

        Returns:
            Tuple of (should_close, reason).
        """
        if self.current_position is None:
            return False, ""
        
        if self.current_position.check_stop_loss_hit(low, high):
            return True, "Stop Loss"
        
        if self.current_position.check_take_profit_hit(low, high):
            return True, "Take Profit"
        
        if self.config.close_on_trend_reversal:
            if self.current_position.side == PositionSide.LONG:
                if current_signal in [0, 1]:
                    return True, "Trend Reversal (Sell Signal)"
            else:
                if current_signal in [3, 4]:
                    return True, "Trend Reversal (Buy Signal)"
        
        if self.config.close_on_neutral and current_signal == 2:
            return True, "Neutral Signal"
        
        return False, ""

    def should_open_position(self, signal: int) -> Optional[PositionSide]:
        """Determine if a new position should be opened.

        Args:
            signal: Model prediction (0-4).

        Returns:
            Position side to open, or None.
        """
        if signal == 4 and self.config.allow_long:
            return PositionSide.LONG
        elif signal == 3 and self.config.allow_long:
            return PositionSide.LONG
        elif signal == 0 and self.config.allow_short:
            return PositionSide.SHORT
        elif signal == 1 and self.config.allow_short:
            return PositionSide.SHORT
        
        return None

    def check_risk_limits(self) -> bool:
        """Check if risk limits are breached.

        Returns:
            True if trading should continue, False if limits breached.
        """
        current_drawdown = (self.peak_capital - self.capital) / self.peak_capital
        if current_drawdown > self.config.max_drawdown_pct:
            return False
        
        return True

    def run(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
    ) -> Dict:
        """Run backtest on data with predictions.

        Args:
            data: OHLCV DataFrame with columns: open, high, low, close, volume.
            predictions: Model predictions array.

        Returns:
            Dictionary with backtest results.
        """
        self.console.print("\n[bold cyan]Starting Backtest...[/bold cyan]\n")
        
        atr_period = 14
        atr_series = ta.atr(
            high=data['high'],
            low=data['low'],
            close=data['close'],
            length=atr_period
        )
        atr_values = atr_series.bfill().fillna(0.0).values
        
        for i in track(range(len(data)), description="Processing..."):
            row = data.iloc[i]
            timestamp = str(row.name)
            signal = int(predictions[i])
            current_atr = atr_values[i]
            
            self.timestamps.append(timestamp)
            self.equity_curve.append(self.capital)
            
            if not self.check_risk_limits():
                self.console.print("[bold red]Risk limits breached. Stopping backtest.[/bold red]")
                break
            
            if self.current_position is not None:
                if self.config.use_trailing_stop:
                    current_pnl_pct = self.current_position.calculate_pnl_percentage(row['close'])
                    if current_pnl_pct > self.config.trailing_stop_activation_pct * 100:
                        trailing_distance = (current_atr / row['close']) if self.config.use_atr_stop else self.config.trailing_stop_distance_pct
                        self.current_position.update_trailing_stop(
                            row['close'],
                            trailing_distance
                        )
                
                should_close, reason = self.should_close_position(
                    signal,
                    row['low'],
                    row['high']
                )
                
                if should_close:
                    self.close_position(
                        row['close'],
                        timestamp,
                        reason,
                        periods_held=1
                    )
            
            if self.current_position is None:
                side = self.should_open_position(signal)
                if side is not None:
                    self.open_position(side, row['close'], timestamp, current_atr)
        
        if self.current_position is not None:
            self.close_position(
                data.iloc[-1]['close'],
                str(data.iloc[-1].name),
                "End of Backtest",
                periods_held=1
            )
        
        return self._calculate_metrics()

    def _calculate_metrics(self) -> Dict:
        """Calculate performance metrics.

        Returns:
            Dictionary with all metrics.
        """
        metrics = {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_return': self.capital - self.initial_capital,
            'total_return_pct': ((self.capital - self.initial_capital) / self.initial_capital) * 100,
            'total_trades': len(self.trades),
            'winning_trades': sum(1 for t in self.trades if t.is_profitable),
            'losing_trades': sum(1 for t in self.trades if not t.is_profitable),
            'win_rate': (sum(1 for t in self.trades if t.is_profitable) / len(self.trades) * 100) if self.trades else 0,
            'total_fees': self.total_fees_paid,
            'total_funding': self.total_funding_paid,
            'total_slippage': self.total_slippage_paid,
            'avg_win': np.mean([t.net_pnl for t in self.trades if t.is_profitable]) if any(t.is_profitable for t in self.trades) else 0,
            'avg_loss': np.mean([t.net_pnl for t in self.trades if not t.is_profitable]) if any(not t.is_profitable for t in self.trades) else 0,
            'largest_win': max([t.net_pnl for t in self.trades], default=0),
            'largest_loss': min([t.net_pnl for t in self.trades], default=0),
            'profit_factor': abs(sum(t.net_pnl for t in self.trades if t.is_profitable) / sum(t.net_pnl for t in self.trades if not t.is_profitable)) if any(not t.is_profitable for t in self.trades) else float('inf'),
            'max_drawdown': self._calculate_max_drawdown(),
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'timestamps': self.timestamps,
        }
        
        return metrics

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown.

        Returns:
            Maximum drawdown percentage.
        """
        if not self.equity_curve:
            return 0.0
        
        equity = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdown = (running_max - equity) / running_max
        return np.max(drawdown) * 100

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio.

        Returns:
            Sharpe ratio.
        """
        if len(self.equity_curve) < 2:
            return 0.0
        
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        return np.mean(returns) / np.std(returns) * np.sqrt(252)
