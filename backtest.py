"""Main backtesting script for Janus Bitcoin trend classifier."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from backtest.config import BacktestConfig
from backtest.engine import BacktestEngine
from backtest.reporter import BacktestReporter
from src.config.config_loader import ConfigLoader
from src.models.mamba_classifier import MambaClassifier
from src.utils.helpers import get_device


def load_model(checkpoint_path: str, config: ConfigLoader, device: torch.device) -> torch.nn.Module:
    """Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint.
        config: Configuration loader.
        device: Device to load model on.

    Returns:
        Loaded model.
    """
    model = MambaClassifier(
        input_dim=config.get('data.num_features'),
        d_model=config.get('model.d_model'),
        d_state=config.get('model.d_state'),
        d_conv=config.get('model.d_conv'),
        n_layers=config.get('model.n_layers'),
        num_classes=config.get('model.num_classes'),
        dropout=config.get('model.dropout'),
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def generate_predictions(
    model: torch.nn.Module,
    data: pd.DataFrame,
    feature_columns: list,
    sequence_length: int,
    device: torch.device,
) -> np.ndarray:
    """Generate predictions for backtest data.

    Args:
        model: Trained model.
        data: DataFrame with features.
        feature_columns: List of feature column names.
        sequence_length: Input sequence length.
        device: Device for inference.

    Returns:
        Array of predictions.
    """
    predictions = []

    with torch.no_grad():
        for i in range(sequence_length - 1, len(data)):
            sequence = data[feature_columns].iloc[i - sequence_length + 1:i + 1].values
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)

            output = model(sequence_tensor)
            pred = output.argmax(dim=1).item()
            predictions.append(pred)

    padding = [2] * (sequence_length - 1)
    return np.array(padding + predictions)


def main() -> None:
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Backtest Janus Bitcoin trend classifier')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to model config')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to backtest data (parquet)')
    parser.add_argument('--start-date', type=str, default='2025-08-01', help='Backtest start date')
    parser.add_argument('--end-date', type=str, default='2025-09-30', help='Backtest end date')
    parser.add_argument('--initial-capital', type=float, default=6000000, help='Initial capital')
    parser.add_argument('--leverage', type=int, default=5, help='Leverage')

    args = parser.parse_args()

    config = ConfigLoader(args.config)
    device = get_device(use_cuda=config.get('device.use_cuda', True))

    print(f"\nLoading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, config, device)

    print(f"Loading data from {args.data}...")
    data = pd.read_parquet(args.data)

    data.index = pd.to_datetime(data.index)
    backtest_data = data[(data.index >= args.start_date) & (data.index <= args.end_date)]

    if len(backtest_data) == 0:
        print(f"No data found between {args.start_date} and {args.end_date}")
        return

    print(f"Generating predictions for {len(backtest_data)} samples...")
    predictions = generate_predictions(
        model,
        backtest_data,
        config.get('data.feature_columns'),
        config.get('data.input_window'),
        device,
    )

    backtest_config = BacktestConfig(
        initial_capital_usd=args.initial_capital,
        leverage=args.leverage,
        backtest_start_date=args.start_date,
        backtest_end_date=args.end_date,
    )

    engine = BacktestEngine(backtest_config)

    ohlcv_data = pd.DataFrame({
        'open': backtest_data.index.to_series().shift(1),
        'high': backtest_data.index.to_series().shift(1),
        'low': backtest_data.index.to_series().shift(1),
        'close': backtest_data.index.to_series(),
    })

    metrics = engine.run(ohlcv_data, predictions)

    reporter = BacktestReporter()
    reporter.print_complete_report(metrics)


if __name__ == '__main__':
    main()
