"""Backtest reporter."""

class BacktestReporter:
    def print_complete_report(self, metrics: dict):
        """Print backtest report."""
        print("Backtest Complete")
        print(f"Metrics: {metrics}")
