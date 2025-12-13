"""Main backtesting script for Janus Bitcoin price regressor."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None

from src.backtest_utils.config import BacktestConfig
from src.backtest_utils.engine import BacktestEngine
from src.backtest_utils.reporter import BacktestReporter
from src.config.config_loader import ConfigLoader


from src.backtest_utils.model_loader import load_model_auto, ModelInferenceWrapper


def load_model(checkpoint_path: str, config: ConfigLoader) -> ModelInferenceWrapper:
    """Load trained model from checkpoint (supports PyTorch, TorchScript, ONNX).

    Args:
        checkpoint_path: Path to model checkpoint or directory.
        config: Configuration loader.

    Returns:
        ModelInferenceWrapper instance.
    """
    return load_model_auto(checkpoint_path, config)


def generate_predictions(
    model: ModelInferenceWrapper,
    data: pd.DataFrame,
    feature_columns: list,
    sequence_length: int,
    entry_threshold: float = 0.005,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate regression predictions and convert to trading signals.

    Args:
        model: Model wrapper (supports PyTorch, TorchScript, ONNX).
        data: DataFrame with features.
        feature_columns: List of feature column names.
        sequence_length: Input sequence length.
        entry_threshold: Minimum absolute predicted change to enter (e.g., 0.5%).

    Returns:
        Tuple of (signals, predicted_changes) arrays.
        signals: -1 (short), 0 (neutral), 1 (long)
        predicted_changes: continuous predicted price changes
    """
    from numpy.lib.stride_tricks import sliding_window_view

    # Extract features as numpy array
    features = data[feature_columns].values.astype(np.float32)

    # Create sliding windows view - zero copy
    # shape: (n_samples, sequence_length, n_features)
    # sliding_window_view on axis 0 of shape (N, F) with window W gives (N-W+1, F, W)
    windows = sliding_window_view(features, window_shape=sequence_length, axis=0)

    # We want (n_windows, sequence_length, n_features), so we swap the last two axes
    windows = np.moveaxis(windows, 2, 1)

    # Predict in batches
    batch_size = 1024
    predicted_changes = []

    for i in range(0, len(windows), batch_size):
        batch = windows[i:i + batch_size]
        # Ensure batch is physically contiguous for efficient inference if needed
        # batch = np.ascontiguousarray(batch)
        # model.predict_batch should handle it, but contiguous is safer.
        batch_preds = model.predict_batch(batch)
        predicted_changes.append(batch_preds)
        
    if predicted_changes:
        predicted_changes = np.concatenate(predicted_changes)
    else:
        predicted_changes = np.array([])

    # Vectorized signal generation
    signals = np.zeros_like(predicted_changes, dtype=int)
    signals[predicted_changes > entry_threshold] = 1
    signals[predicted_changes < -entry_threshold] = -1

    # Add padding
    padding_len = sequence_length - 1
    padding_signal = np.zeros(padding_len, dtype=int)
    padding_change = np.zeros(padding_len, dtype=float)
    
    return np.concatenate([padding_signal, signals]), np.concatenate([padding_change, predicted_changes])


def main() -> None:
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Backtest Janus Bitcoin price regressor')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to model config')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to backtest data (parquet)')
    parser.add_argument('--start-date', type=str, default='2025-08-01', help='Backtest start date')
    parser.add_argument('--end-date', type=str, default='2025-09-30', help='Backtest end date')
    parser.add_argument('--initial-capital', type=float, default=6000000, help='Initial capital')
    parser.add_argument('--leverage', type=int, default=5, help='Leverage')
    parser.add_argument('--entry-threshold', type=float, default=0.005, help='Minimum predicted change to enter (0.5%%)')

    args = parser.parse_args()

    config = ConfigLoader(args.config)

    print(f"\nLoading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, config)

    print(f"Loading data from {args.data}...")
    data = pd.read_parquet(args.data)

    data.index = pd.to_datetime(data.index)
    backtest_data = data[(data.index >= args.start_date) & (data.index <= args.end_date)]

    if len(backtest_data) == 0:
        print(f"No data found between {args.start_date} and {args.end_date}")
        return

    print(f"Generating predictions for {len(backtest_data)} samples...")
    print(f"Using entry threshold: {args.entry_threshold:.2%}")
    signals, predicted_changes = generate_predictions(
        model,
        backtest_data,
        config.get('data.feature_columns'),
        config.get('data.input_window'),
        entry_threshold=args.entry_threshold,
    )

    print("Loading OHLCV data for backtest...")
    try:
        ohlcv_raw = pd.read_csv('dataset/BTCUSDT.csv', header=None)
        
        column_names = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                       'taker_buy_quote', 'taker_buy_base', 'quote_volume', 'num_trades']
        ohlcv_raw.columns = column_names[:len(ohlcv_raw.columns)]
        ohlcv_raw = ohlcv_raw[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        timestamps = pd.to_numeric(ohlcv_raw['timestamp'], errors='coerce')
        unit = 'ms' if timestamps.max() > 1e12 else 's'
        ohlcv_raw['timestamp'] = pd.to_datetime(timestamps, unit=unit, errors='coerce')
        ohlcv_raw.set_index('timestamp', inplace=True)
        
        timeframe = config.get('data.base_timeframe', '30min')
        ohlcv_raw = ohlcv_raw.resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        ohlcv_data = ohlcv_raw.loc[backtest_data.index].copy()
        ohlcv_data = ohlcv_data[['open', 'high', 'low', 'close', 'volume']]
        
    except FileNotFoundError:
        print("Warning: BTCUSDT.csv not found, trying alternative paths...")
        try:
            ohlcv_raw = pd.read_csv('dataset/btc.csv', header=None)
            column_names = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            ohlcv_raw.columns = column_names[:len(ohlcv_raw.columns)]
            ohlcv_raw = ohlcv_raw[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            timestamps = pd.to_numeric(ohlcv_raw['timestamp'], errors='coerce')
            unit = 'ms' if timestamps.max() > 1e12 else 's'
            ohlcv_raw['timestamp'] = pd.to_datetime(timestamps, unit=unit, errors='coerce')
            ohlcv_raw.set_index('timestamp', inplace=True)
            
            timeframe = config.get('data.base_timeframe', '30min')
            ohlcv_raw = ohlcv_raw.resample(timeframe).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            ohlcv_data = ohlcv_raw.loc[backtest_data.index].copy()
            ohlcv_data = ohlcv_data[['open', 'high', 'low', 'close', 'volume']]
        except:
            print("Warning: No CSV found, using synthetic OHLCV from dataset index")
            ohlcv_data = pd.DataFrame({
                'open': backtest_data.index.to_series().shift(1),
                'high': backtest_data.index.to_series().shift(1),
                'low': backtest_data.index.to_series().shift(1),
                'close': backtest_data.index.to_series(),
                'volume': 1000.0,
            })

    backtest_config = BacktestConfig(
        initial_capital_usd=args.initial_capital,
        leverage=args.leverage,
        backtest_start_date=args.start_date,
        backtest_end_date=args.end_date,
    )

    engine = BacktestEngine(backtest_config)
    metrics = engine.run(ohlcv_data, signals)

    reporter = BacktestReporter()
    reporter.print_complete_report(metrics)


if __name__ == '__main__':
    main()
