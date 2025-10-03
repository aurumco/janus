"""Rich terminal reporting for backtest results."""

from typing import Dict, List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .trade import Trade


class BacktestReporter:
    """Creates rich terminal reports for backtest results."""

    def __init__(self) -> None:
        """Initialize reporter."""
        self.console = Console()

    def print_summary(self, metrics: Dict) -> None:
        """Print backtest summary with rich formatting.

        Args:
            metrics: Dictionary with backtest metrics.
        """
        self.console.print("\n")
        
        title = Text("BACKTEST RESULTS SUMMARY", style="bold white on blue")
        
        summary_table = Table(show_header=False, box=None, padding=(0, 2))
        summary_table.add_column("Metric", style="cyan", width=30)
        summary_table.add_column("Value", style="white", width=20)
        
        initial = f"${metrics['initial_capital']:,.2f}"
        final = f"${metrics['final_capital']:,.2f}"
        total_return = metrics['total_return']
        return_pct = metrics['total_return_pct']
        
        return_color = "green" if total_return > 0 else "red"
        return_text = f"${total_return:,.2f} ({return_pct:+.2f}%)"
        
        summary_table.add_row("Initial Capital", initial)
        summary_table.add_row("Final Capital", final)
        summary_table.add_row("Total Return", Text(return_text, style=f"bold {return_color}"))
        summary_table.add_row("", "")
        summary_table.add_row("Total Trades", str(metrics['total_trades']))
        summary_table.add_row("Winning Trades", Text(str(metrics['winning_trades']), style="green"))
        summary_table.add_row("Losing Trades", Text(str(metrics['losing_trades']), style="red"))
        
        win_rate = metrics['win_rate']
        win_rate_color = "green" if win_rate >= 50 else "yellow" if win_rate >= 40 else "red"
        summary_table.add_row("Win Rate", Text(f"{win_rate:.2f}%", style=f"bold {win_rate_color}"))
        
        panel = Panel(summary_table, title=title, border_style="blue")
        self.console.print(panel)

    def print_performance_metrics(self, metrics: Dict) -> None:
        """Print detailed performance metrics.

        Args:
            metrics: Dictionary with backtest metrics.
        """
        self.console.print("\n")
        
        perf_table = Table(title="Performance Metrics", box=None, show_header=True)
        perf_table.add_column("Metric", style="cyan", width=30)
        perf_table.add_column("Value", style="white", width=25)
        
        avg_win = metrics['avg_win']
        avg_loss = metrics['avg_loss']
        
        perf_table.add_row(
            "Average Win",
            Text(f"${avg_win:,.2f}", style="green")
        )
        perf_table.add_row(
            "Average Loss",
            Text(f"${avg_loss:,.2f}", style="red")
        )
        perf_table.add_row(
            "Largest Win",
            Text(f"${metrics['largest_win']:,.2f}", style="bold green")
        )
        perf_table.add_row(
            "Largest Loss",
            Text(f"${metrics['largest_loss']:,.2f}", style="bold red")
        )
        perf_table.add_row("", "")
        
        profit_factor = metrics['profit_factor']
        pf_color = "green" if profit_factor > 1.5 else "yellow" if profit_factor > 1 else "red"
        perf_table.add_row(
            "Profit Factor",
            Text(f"{profit_factor:.2f}", style=f"bold {pf_color}")
        )
        
        sharpe = metrics['sharpe_ratio']
        sharpe_color = "green" if sharpe > 1 else "yellow" if sharpe > 0 else "red"
        perf_table.add_row(
            "Sharpe Ratio",
            Text(f"{sharpe:.2f}", style=f"bold {sharpe_color}")
        )
        
        max_dd = metrics['max_drawdown']
        dd_color = "green" if max_dd < 10 else "yellow" if max_dd < 20 else "red"
        perf_table.add_row(
            "Max Drawdown",
            Text(f"{max_dd:.2f}%", style=f"bold {dd_color}")
        )
        
        self.console.print(perf_table)

    def print_cost_breakdown(self, metrics: Dict) -> None:
        """Print cost breakdown.

        Args:
            metrics: Dictionary with backtest metrics.
        """
        self.console.print("\n")
        
        cost_table = Table(title="Cost Breakdown", box=None)
        cost_table.add_column("Cost Type", style="cyan", width=30)
        cost_table.add_column("Amount", style="yellow", width=25)
        
        total_fees = metrics['total_fees']
        total_funding = metrics['total_funding']
        total_slippage = metrics['total_slippage']
        total_costs = total_fees + total_funding + total_slippage
        
        cost_table.add_row("Trading Fees", f"${total_fees:,.2f}")
        cost_table.add_row("Funding Fees", f"${total_funding:,.2f}")
        cost_table.add_row("Slippage", f"${total_slippage:,.2f}")
        cost_table.add_row("", "")
        cost_table.add_row(
            "Total Costs",
            Text(f"${total_costs:,.2f}", style="bold yellow")
        )
        
        self.console.print(cost_table)

    def print_all_trades(self, trades: List[Trade]) -> None:
        """Print detailed list of all trades.

        Args:
            trades: List of completed trades.
        """
        if not trades:
            self.console.print("\n[yellow]No trades executed.[/yellow]\n")
            return
        
        self.console.print("\n")
        self.console.print(Panel(
            Text("DETAILED TRADE LOG", justify="center"),
            style="bold white on magenta"
        ))
        self.console.print("\n")
        
        for idx, trade in enumerate(trades, 1):
            self._print_single_trade(idx, trade)

    def _print_single_trade(self, trade_num: int, trade: Trade) -> None:
        """Print details of a single trade.

        Args:
            trade_num: Trade number.
            trade: Trade object.
        """
        side_color = "green" if trade.side.value == "LONG" else "red"
        pnl_color = "green" if trade.is_profitable else "red"
        
        trade_table = Table(
            show_header=False,
            box=None,
            padding=(0, 1),
            width=100
        )
        trade_table.add_column("Field", style="cyan", width=25)
        trade_table.add_column("Value", width=35)
        trade_table.add_column("Field", style="cyan", width=25)
        trade_table.add_column("Value", width=35)
        
        trade_table.add_row(
            "Trade #",
            Text(f"{trade_num}", style="bold white"),
            "Side",
            Text(trade.side.value, style=f"bold {side_color}")
        )
        
        trade_table.add_row(
            "Entry Time",
            trade.entry_timestamp,
            "Exit Time",
            trade.exit_timestamp
        )
        
        trade_table.add_row(
            "Entry Price",
            f"${trade.entry_price:,.2f}",
            "Exit Price",
            f"${trade.exit_price:,.2f}"
        )
        
        trade_table.add_row(
            "Position Size",
            f"{trade.size:.6f} BTC",
            "Leverage",
            f"{trade.leverage}x"
        )
        
        trade_table.add_row(
            "Gross PnL",
            Text(f"${trade.pnl:,.2f}", style=pnl_color),
            "PnL %",
            Text(f"{trade.pnl_percentage:+.2f}%", style=f"bold {pnl_color}")
        )
        
        trade_table.add_row(
            "Total Fees",
            Text(f"${trade.total_fees:,.2f}", style="yellow"),
            "Net PnL",
            Text(f"${trade.net_pnl:,.2f}", style=f"bold {pnl_color}")
        )
        
        trade_table.add_row(
            "Holding Periods",
            f"{trade.holding_periods}",
            "Exit Reason",
            Text(trade.exit_reason, style="italic")
        )
        
        if trade.pyramid_level > 0:
            trade_table.add_row(
                "Pyramid Level",
                Text(f"{trade.pyramid_level}", style="bold yellow"),
                "",
                ""
            )
        
        panel_title = f"Trade #{trade_num}"
        panel_style = "green" if trade.is_profitable else "red"
        
        self.console.print(Panel(
            trade_table,
            title=panel_title,
            border_style=panel_style,
            padding=(1, 2)
        ))
        self.console.print()

    def print_trade_summary_table(self, trades: List[Trade]) -> None:
        """Print condensed summary table of all trades.

        Args:
            trades: List of completed trades.
        """
        if not trades:
            return
        
        self.console.print("\n")
        
        summary_table = Table(
            title="Trade Summary Table",
            show_header=True,
            header_style="bold cyan"
        )
        
        summary_table.add_column("#", style="white", width=5)
        summary_table.add_column("Side", width=6)
        summary_table.add_column("Entry", width=18)
        summary_table.add_column("Exit", width=18)
        summary_table.add_column("Entry $", justify="right", width=12)
        summary_table.add_column("Exit $", justify="right", width=12)
        summary_table.add_column("PnL $", justify="right", width=12)
        summary_table.add_column("PnL %", justify="right", width=10)
        summary_table.add_column("Reason", width=15)
        
        for idx, trade in enumerate(trades, 1):
            side_style = "green" if trade.side.value == "LONG" else "red"
            pnl_style = "green" if trade.is_profitable else "red"
            
            summary_table.add_row(
                str(idx),
                Text(trade.side.value[:4], style=side_style),
                trade.entry_timestamp[:16],
                trade.exit_timestamp[:16],
                f"${trade.entry_price:,.2f}",
                f"${trade.exit_price:,.2f}",
                Text(f"${trade.net_pnl:,.2f}", style=pnl_style),
                Text(f"{trade.pnl_percentage:+.2f}%", style=pnl_style),
                trade.exit_reason[:15]
            )
        
        self.console.print(summary_table)

    def print_complete_report(self, metrics: Dict) -> None:
        """Print complete backtest report.

        Args:
            metrics: Dictionary with all backtest metrics.
        """
        self.console.clear()
        
        self.console.print("\n" * 2)
        self.console.print("="*100, style="bold blue")
        self.console.print(
            Text("JANUS BACKTEST REPORT", justify="center"),
            style="bold white on blue"
        )
        self.console.print("="*100, style="bold blue")
        
        self.print_summary(metrics)
        self.print_performance_metrics(metrics)
        self.print_cost_breakdown(metrics)
        
        self.console.print("\n" * 2)
        self.console.print("="*100, style="bold magenta")
        
        self.print_trade_summary_table(metrics['trades'])
        self.print_all_trades(metrics['trades'])
        
        self.console.print("\n")
        self.console.print("="*100, style="bold blue")
        self.console.print(
            Text("END OF REPORT", justify="center"),
            style="bold white on blue"
        )
        self.console.print("="*100, style="bold blue")
        self.console.print("\n")
