import time
import sys
import os
import signal
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
import ccxt
import onnxruntime as ort
from sklearn.preprocessing import MinMaxScaler
import joblib
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.engine import BacktestEngine
from backtest.config import BacktestConfig
from backtest.position import PositionSide
from dataset.data_processor import MultiTimeframeProcessor
from dataset.config import DatasetConfig

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("live_trading.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("LiveTrader")
warnings.filterwarnings('ignore')

class LiveTrader:
    def __init__(self, symbol: str, model_path: str, coinex_creds: Dict[str, str]):
        self.symbol = symbol
        self.asset_id = 0 # Default to 0 for BTCUSDT
        self.console = Console()

        # Configuration
        self.seq_length = 72 # Based on pretrain config
        self.feature_count = 16
        self.buffer_size = 2000 # Keep 2000 candles for proper feature calc

        # Initialize CoinEx
        self.exchange = ccxt.coinex({
            'apiKey': coinex_creds['access_id'],
            'secret': coinex_creds['secret_key'],
            'enableRateLimit': True,
            'options': {'defaultType': 'swap'}
        })

        # Initialize Model
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.asset_input_name = self.session.get_inputs()[1].name
        self.output_name = self.session.get_outputs()[3].name # Output '476' (Direction Logits)

        # Initialize Data Processor
        self.ds_config = DatasetConfig()
        self.ds_config.enable_garch = True # Ensure consistency with feature count
        self.processor = MultiTimeframeProcessor(self.ds_config)

        # Initialize Backtest Engine (Simulation)
        bt_config = BacktestConfig(
            initial_capital_usd=10000.0,
            leverage=5,
            position_size_pct=0.95,
            maker_fee=0.0003,
            taker_fee=0.0005
        )
        self.engine = BacktestEngine(bt_config)

        # State
        self.buffer_df = pd.DataFrame()
        self.scaler = None
        self.trades_history = []
        self.running = True

        # Setup Signal Handler
        signal.signal(signal.SIGINT, self.shutdown)

    def shutdown(self, signum, frame):
        self.console.print("\n[bold red]Shutting down...[/bold red]")
        self.save_history()
        self.running = False
        sys.exit(0)

    def fetch_initial_data(self):
        """Fetch historical data to fill buffer."""
        logger.info(f"Fetching initial history for {self.symbol}...")
        try:
            # Fetch 2000 candles in batches if necessary, but ccxt/coinex might limit to 1000
            # We'll fetch 1000 which is ~10 days. Should be enough for most features except very long RSIs
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe='15m', limit=1000)

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            self.buffer_df = df
            logger.info(f"Initial buffer filled: {len(self.buffer_df)} candles.")

        except Exception as e:
            logger.error(f"Error fetching initial data: {e}")
            sys.exit(1)

    def update_buffer(self):
        """Fetch latest candle and update buffer."""
        try:
            # Fetch last few candles to ensure we have the latest closed one
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe='15m', limit=5)
            new_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')
            new_df.set_index('timestamp', inplace=True)

            # Combine and remove duplicates
            self.buffer_df = pd.concat([self.buffer_df, new_df])
            self.buffer_df = self.buffer_df[~self.buffer_df.index.duplicated(keep='last')]

            # Keep fixed size
            if len(self.buffer_df) > self.buffer_size:
                self.buffer_df = self.buffer_df.iloc[-self.buffer_size:]

        except Exception as e:
            logger.error(f"Error updating buffer: {e}")

    def prepare_features(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """Generate features and normalize."""
        # We need to simulate the file loading structure for MultiTimeframeProcessor
        # It expects a dictionary of timeframes usually, but here we can manually invoke resampling

        df = self.buffer_df.copy()

        # Resample
        ohlcv_logic = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}

        # M15 (Base)
        # Assuming buffer is already 15m.
        df_m15 = df.copy() # Already 15m

        # H1
        df_h1 = df.resample('1h').apply(ohlcv_logic).dropna()

        # H4
        df_h4 = df.resample('4h').apply(ohlcv_logic).dropna()

        # Daily
        df_daily = df.resample('D').apply(ohlcv_logic).dropna()

        # Calculate Features
        # 1. Higher Timeframe
        try:
            df_h1 = self.processor.calculate_higher_timeframe_features(df_h1, df_h4, df_daily)

            # 2. M15 Features
            df_m15 = self.processor.calculate_m15_features(df_m15)

            # 3. Merge
            df_final = self.processor.merge_timeframes(df_m15, df_h1, df_h4)

            # 4. Select Columns
            feature_cols = self.processor.get_feature_columns()

            # 5. Add Asset ID
            # In training, asset_id is added. We assume BTC=0.
            # We need to verify if we should add it before or after scaling.
            # Usually categorical features are NOT scaled by MinMaxScaler.
            # But if it's part of the feature matrix 'X', it often gets swept up.
            # Given we don't have the original scaler, we will Scale everything EXCEPT asset_id if possible,
            # or just scale everything. Scaling 0 is 0.

            X = df_final[feature_cols].copy()
            X['asset_id'] = self.asset_id

            # Handle NaNs (created by indicators)
            X = X.ffill().bfill().fillna(0)

            # Normalize
            # Fit scaler on current window (Adaptive Normalization)
            scaler = MinMaxScaler(feature_range=(-1, 1))
            X_scaled = scaler.fit_transform(X)

            # Return last sequence
            if len(X_scaled) < self.seq_length:
                logger.warning("Not enough data for sequence length.")
                return None, df_final

            last_sequence = X_scaled[-self.seq_length:]
            return last_sequence, df_final.iloc[-1:]

        except Exception as e:
            logger.error(f"Feature calculation error: {e}")
            return None, None

    def predict(self, sequence: np.ndarray) -> Dict:
        """Run inference."""
        # Input: [1, seq_len, 16]
        x_input = sequence.astype(np.float32).reshape(1, self.seq_length, self.feature_count)
        asset_ids = np.array([self.asset_id], dtype=np.int64)

        outputs = self.session.run(None, {
            self.input_name: x_input,
            self.asset_input_name: asset_ids
        })

        # Parse outputs
        # Output 3 is '476' (Logits: [batch, 2])
        # Output 2 is 'predicted_direction' (Hidden states: [batch, seq, 192])
        # Output 1 is 'predicted_volatility'
        # Output 0 is 'reconstructed_sequence'

        logits = outputs[3][0] # [2]

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        return {
            'prob_down': probs[0],
            'prob_up': probs[1],
            'logits': logits
        }

    def execute_simulation(self, prediction: Dict, current_price: float, timestamp: str):
        """Simulate trading execution."""
        prob_up = prediction['prob_up']

        # Thresholds
        BUY_THRESHOLD = 0.60
        SELL_THRESHOLD = 0.60 # If prob_down > 0.6 (i.e. prob_up < 0.4)

        signal_val = 0
        if prob_up > BUY_THRESHOLD:
            signal_val = 1
        elif (1 - prob_up) > SELL_THRESHOLD:
            signal_val = -1

        # Use Engine Logic
        # Calculate ATR for stop loss (approximate from buffer)
        # Using existing engine ATR calculation might require full DF pass.
        # We can calculate current ATR quickly.
        # ATR 14
        high = self.buffer_df['high'].iloc[-15:]
        low = self.buffer_df['low'].iloc[-15:]
        close = self.buffer_df['close'].iloc[-15:]
        try:
            import pandas_ta_classic as ta
            atr_val = ta.atr(high, low, close, length=14).iloc[-1]
        except:
            atr_val = current_price * 0.01 # Fallback

        # Check Exits
        if self.engine.current_position:
             should_close, reason = self.engine.should_close_position(
                 signal_val,
                 self.buffer_df['low'].iloc[-1],
                 self.buffer_df['high'].iloc[-1]
             )
             if should_close:
                 self.engine.close_position(current_price, timestamp, reason, periods_held=1)
                 logger.info(f"CLOSED POSITION: {reason} @ {current_price}")

        # Check Entries
        if not self.engine.current_position:
            side = self.engine.should_open_position(signal_val)
            if side:
                self.engine.open_position(side, current_price, timestamp, atr_val)
                logger.info(f"OPENED {side} @ {current_price}")

    def save_history(self):
        """Save trade history to JSON."""
        trades_dict = []
        for t in self.engine.trades:
            trades_dict.append({
                'entry_time': str(t.entry_timestamp),
                'exit_time': str(t.exit_timestamp),
                'side': str(t.side),
                'pnl': float(t.net_pnl),
                'pnl_pct': float(t.pnl_percentage)
            })

        with open('live_trades.json', 'w') as f:
            json.dump(trades_dict, f, indent=2)
        logger.info("Trade history saved to live_trades.json")

    def run(self):
        """Main loop."""
        self.fetch_initial_data()
        self.console.print("[bold green]System Initialized. Starting Loop...[/bold green]")

        while self.running:
            try:
                # 1. Update Data
                self.update_buffer()
                current_price = self.buffer_df['close'].iloc[-1]
                timestamp = str(self.buffer_df.index[-1])

                # 2. Prepare Features
                sequence, last_row = self.prepare_features()

                if sequence is not None:
                    # 3. Predict
                    pred = self.predict(sequence)

                    # 4. Execute
                    self.execute_simulation(pred, current_price, timestamp)

                    # 5. Display Status
                    self.display_status(current_price, pred)

                # 6. Wait for next candle?
                # Or run continuously? User said "continuously running loop".
                # But candles update every 15 mins.
                # However, live price updates faster.
                # BUT feature calculation is based on closed candles usually.
                # If we want to trade "live", we can recalc features on partial candle?
                # For safety, let's wait 30 seconds.
                time.sleep(30)

            except Exception as e:
                logger.error(f"Loop error: {e}")
                time.sleep(10)

    def display_status(self, price, pred):
        table = Table(title="Janus Live Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Price", f"${price:,.2f}")
        table.add_row("Prob Up", f"{pred['prob_up']:.2%}")
        table.add_row("Prob Down", f"{pred['prob_down']:.2%}")
        table.add_row("Capital", f"${self.engine.capital:,.2f}")

        pos = self.engine.current_position
        if pos:
            pnl = pos.calculate_pnl(price)
            table.add_row("Position", f"{pos.side} (PnL: ${pnl:.2f})")
        else:
            table.add_row("Position", "None")

        self.console.print(table)

if __name__ == "__main__":
    # Credentials
    creds = {
        'access_id': 'B3832C3C66CA49B4B560C8E33ED72D62',
        'secret_key': '975898DD0906B8757C606D18AF277E71DA5C71B388345682'
    }

    # Model Path
    model_path = "backtest/pretrain_model.onnx"

    trader = LiveTrader('BTC/USDT', model_path, creds)
    trader.run()
