import asyncio
import sys
import os
import signal
import json
import logging
import time
from typing import Optional, Tuple, Any, List
import warnings

import numpy as np
import pandas as pd
import ccxt.async_support as ccxt  # Async CCXT
import onnxruntime as ort
from sklearn.preprocessing import MinMaxScaler

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich import box

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.engine import BacktestEngine
from backtest.config import BacktestConfig
from dataset.data_processor import MultiTimeframeProcessor
from dataset.config import DatasetConfig

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("live_trading.log"),
    ],
)
logger = logging.getLogger("LiveTrader")
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

COINEX_FEES = {
    "maker": 0.0003,  # 0.03%
    "taker": 0.0005,  # 0.05%
}

TRADING_CONFIG = {
    "capital_per_asset": 100.0,
    "entry_size_pct": 0.075,  # 7.5% of allocated capital
    "leverage_base": 5,
    "leverage_max": 10,  # Cap for dynamic leverage
    "confidence_threshold": 0.60,
    "buffer_size": 3000,  # Need ample history for H4/Daily indicators
    "assets": [
        {"symbol": "BTC/USDT", "asset_id": 0},
        {"symbol": "ETH/USDT", "asset_id": 1},
    ],
}

CREDENTIALS = {
    "access_id": "B3832C3C66CA49B4B560C8E33ED72D62",
    "secret_key": "975898DD0906B8757C606D18AF277E71DA5C71B388345682",
}

MODEL_PATH = "backtest/pretrain_model.onnx"


# -----------------------------------------------------------------------------
# Data Processor Adapter
# -----------------------------------------------------------------------------


class LiveDataProcessor(MultiTimeframeProcessor):
    """
    Extends MultiTimeframeProcessor to work with in-memory DataFrames
    instead of loading from CSV.
    """

    def process_live_buffer(
        self, df: pd.DataFrame
    ) -> Tuple[Optional[np.ndarray], pd.DataFrame]:
        """
        Process the buffer DataFrame into model features.
        Returns:
            sequence: (seq_len, num_features) normalized
            latest_row: DataFrame row for inspection
        """
        # Ensure correct types
        df = df.copy()
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        # Resample logic mirrored from load_and_resample
        ohlcv_logic = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }

        # Base M15
        df_m15 = df.resample("15min").apply(ohlcv_logic).dropna()

        # H1
        df_h1 = df.resample("1h").apply(ohlcv_logic).dropna()

        # H4
        df_h4 = df.resample("4h").apply(ohlcv_logic).dropna()

        # Daily
        df_daily = df.resample("D").apply(ohlcv_logic).dropna()

        if len(df_h4) < 25 or len(df_daily) < 15:
            # Not enough data for EMA_21_H4 or RSI_14_Daily
            return None, df_m15.iloc[-1:]

        try:
            # Calculate Features (Exact sequence as training)
            # 1. Higher Timeframe
            df_h1 = self.calculate_higher_timeframe_features(df_h1, df_h4, df_daily)

            # 2. M15 Features
            df_m15 = self.calculate_m15_features(df_m15)

            # 3. Merge
            df_final = self.merge_timeframes(df_m15, df_h1, df_h4)

            # 4. Select Columns
            feature_cols = self.get_feature_columns()

            # 5. Prepare for Model
            X = df_final[feature_cols].copy()

            # Handle NaNs (Indicator warm-up)
            X = X.ffill().bfill().fillna(0)

            # Normalize (Rolling Window approach for live data)
            # We fit on the available history in the buffer to approximate local distribution
            scaler = MinMaxScaler(feature_range=(-1, 1))
            X_scaled = scaler.fit_transform(X)

            # Get last sequence
            seq_len = 72  # Fixed for pretrain model
            if len(X_scaled) < seq_len:
                return None, df_final.iloc[-1:]

            last_sequence = X_scaled[-seq_len:]

            # Attach asset_id (not scaled, handled in model input separately usually,
            # BUT check if model expects it in input tensor or separate.
            # inspect_model.py showed: input_sequence (float), asset_ids (int64).
            # So sequence should NOT contain asset_id.)

            return last_sequence, df_final.iloc[-1:]

        except Exception as e:
            logger.error(f"Feature processing error: {e}")
            return None, df_m15.iloc[-1:]


# -----------------------------------------------------------------------------
# Asset Trader Class
# -----------------------------------------------------------------------------


class AssetTrader:
    def __init__(self, symbol: str, asset_id: int, exchange, session, lock: asyncio.Lock):
        self.symbol = symbol
        self.asset_id = asset_id
        self.exchange = exchange
        self.session = session
        self.lock = (
            lock  # Lock for ONNX session (thread safety if needed, though run is sync)
        )

        self.config = TRADING_CONFIG

        # Initialize Engine
        capital = float(self.config["capital_per_asset"])
        leverage = int(self.config["leverage_base"])

        bt_config = BacktestConfig(
            initial_capital_usd=capital,
            leverage=leverage,
            position_size_pct=0.99,  # Controlled manually via open_position size
            maker_fee=COINEX_FEES["maker"],
            taker_fee=COINEX_FEES["taker"],
            slippage_pct=0.0001,
            use_atr_stop=True,
            atr_multiplier=2.0,
        )
        self.engine = BacktestEngine(bt_config)
        self.engine.capital = capital  # Explicit reset

        # Data
        self.buffer_df = pd.DataFrame()
        self.processor = LiveDataProcessor(DatasetConfig(enable_garch=True))

        # State
        self.status = "Initializing"
        self.last_pred = {"prob_up": 0.0, "prob_down": 0.0}
        self.last_price = 0.0
        self.pnl_history: List[float] = []
        self.errors: List[str] = []

    async def fetch_initial_data(self):
        self.status = "Fetching History"
        try:
            # We need ~3000 candles (approx 31 days) for Daily/H4 indicators
            target_candles = 3000
            limit = 1000  # CoinEx max limit per request
            all_ohlcv = []

            # Fetch latest chunk first

            # Simple loop to fetch backwards isn't standard in CCXT usually,
            # usually we fetch forward from 'since' or backward using 'end' param if supported.
            # CoinEx API supports 'before' or pagination? CCXT usually handles 'since'.
            # To get *latest* N candles, we might need to be clever.
            # Let's try fetching latest 1000, then see timestamp, calculate 'since' for previous batch.
            # Or use a loop fetching from a calculated start time.

            # 15min * 3000 = 45,000 minutes = 750 hours = 31.25 days.
            ms_per_candle = 15 * 60 * 1000
            now = int(time.time() * 1000)
            start_time = now - (target_candles * ms_per_candle)

            current_since = start_time

            while len(all_ohlcv) < target_candles:
                remaining = target_candles - len(all_ohlcv)
                fetch_limit = min(limit, remaining)

                # We fetch forward from current_since
                chunk = await self.exchange.fetch_ohlcv(
                    self.symbol, timeframe="15m", limit=fetch_limit, since=current_since
                )

                if not chunk:
                    break

                all_ohlcv.extend(chunk)

                # Update since to the last timestamp + 1 candle
                last_time = chunk[-1][0]
                current_since = last_time + ms_per_candle

                if current_since > now:
                    break

                # Rate limit safety
                await asyncio.sleep(0.5)

            # Sort and deduplicate just in case
            all_ohlcv.sort(key=lambda x: x[0])

            df = pd.DataFrame(
                all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            # Deduplicate by timestamp
            df.drop_duplicates(subset="timestamp", keep="last", inplace=True)
            df.set_index("timestamp", inplace=True)

            self.buffer_df = df
            logger.info(f"[{self.symbol}] Fetched {len(self.buffer_df)} candles.")

        except Exception as e:
            self.errors.append(f"Init Fetch Error: {str(e)}")
            logger.error(f"[{self.symbol}] Init Fetch Error: {e}")

    async def tick(self):
        """Single simulation step."""
        try:
            self.status = "Updating Data"
            # 1. Fetch latest
            ohlcv = await self.exchange.fetch_ohlcv(
                self.symbol, timeframe="15m", limit=5
            )
            new_df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], unit="ms")
            new_df.set_index("timestamp", inplace=True)

            # Update Buffer
            self.buffer_df = pd.concat([self.buffer_df, new_df])
            self.buffer_df = self.buffer_df[
                ~self.buffer_df.index.duplicated(keep="last")
            ]
            if len(self.buffer_df) > self.config["buffer_size"]: # type: ignore
                self.buffer_df = self.buffer_df.iloc[-self.config["buffer_size"] :] # type: ignore

            self.last_price = self.buffer_df["close"].iloc[-1]
            current_time = self.buffer_df.index[-1]

            # 2. Process Features
            self.status = "Processing Features"
            sequence, _ = self.processor.process_live_buffer(self.buffer_df)

            if sequence is None:
                self.status = "Insufficient Data"
                return

            # 3. Inference
            self.status = "Inference"
            # Prepare inputs
            # Input 1: Sequence [1, 72, 16] float32
            x_input = sequence.astype(np.float32).reshape(1, 72, 16)
            # Input 2: Asset ID [1] int64
            x_asset = np.array([self.asset_id], dtype=np.int64)

            # Run ONNX (Sync call, wrap in thread if blocking too much, but ONNX is fast)
            # Using simple synchronous call here as it's sub-millisecond usually
            input_name = self.session.get_inputs()[0].name
            asset_input_name = self.session.get_inputs()[1].name

            outputs = self.session.run(
                None, {input_name: x_input, asset_input_name: x_asset}
            )

            # Output 3 is Direction Logits [1, 2]
            logits = outputs[3][0]
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()

            self.last_pred = {"prob_down": probs[0], "prob_up": probs[1]}

            # 4. Simulation
            self.status = "Simulating"
            self.simulate_trade(probs, self.last_price, str(current_time))

            self.status = "Idle"

        except Exception as e:
            self.status = "Error"
            self.errors.append(str(e))
            logger.error(f"[{self.symbol}] Tick Error: {e}")

    def simulate_trade(self, probs, price, timestamp):
        prob_up = probs[1]
        prob_down = probs[0]

        # Decide Signal
        signal_val = 0
        confidence = 0.0

        threshold = float(self.config["confidence_threshold"])

        if prob_up > threshold:
            signal_val = 1
            confidence = prob_up
        elif prob_down > threshold:
            signal_val = -1
            confidence = prob_down

        # Dynamic Leverage Calculation
        # Base 5x. If confidence > 80%, scale up to Max (10x).
        leverage_base = float(self.config["leverage_base"])
        leverage_max = float(self.config["leverage_max"])
        leverage = leverage_base

        if confidence > 0.8:
            # Linearly interpolate between 5 and 10 for confidence 0.8 to 1.0
            scale = (confidence - 0.8) / 0.2
            leverage = leverage_base + scale * (
                leverage_max - leverage_base
            )

        self.engine.config.leverage = int(leverage)

        # Close Checks
        if self.engine.current_position:
            should_close, reason = self.engine.should_close_position(
                signal_val,
                self.buffer_df["low"].iloc[-1],  # Approx low/high of current candle
                self.buffer_df["high"].iloc[-1],
            )
            if should_close:
                self.engine.close_position(price, timestamp, reason, periods_held=1)
                logger.info(f"[{self.symbol}] CLOSED {reason} @ {price}")
                self._log_trade()

        # Open Checks
        if not self.engine.current_position and signal_val != 0:
            side = self.engine.should_open_position(signal_val)
            if side:
                # Calculate Size: 7.5% of Capital
                # Real logic: Margin = Capital * 0.075. Size = Margin * Leverage
                entry_pct = float(self.config["entry_size_pct"])

                # Override engine's generic sizing to match specific request
                # We interpret engine.open_position logic.
                # Currently engine calculates size based on config.position_size_pct.
                # We will hack it slightly by setting config dynamically or just calculating here
                # but engine.open_position calls calculate_position_size internally.
                # Let's adjust the config.position_size_pct temporarily?
                # Or better, we can modify engine to accept size, but we can't change engine code now easily.
                # Engine 'calculate_position_size' uses 'position_size_pct' of capital.
                # If we want 7.5% margin usage, that means position value = 7.5% * Leverage.
                # e.g. 5x leverage -> 37.5% of capital as position value.

                target_pos_size_pct = entry_pct * leverage
                # Cap at 1.0 (100% capital used) just in case, though 7.5%*10 = 75%
                self.engine.config.position_size_pct = min(target_pos_size_pct, 0.99)

                # ATR for stop
                try:
                    import pandas_ta_classic as ta

                    atr = ta.atr(
                        self.buffer_df["high"],
                        self.buffer_df["low"],
                        self.buffer_df["close"],
                        length=14,
                    ).iloc[-1]
                except Exception:
                    atr = price * 0.01

                self.engine.open_position(side, price, timestamp, atr)
                logger.info(
                    f"[{self.symbol}] OPENED {side} @ {price} | Lev: {leverage:.1f}x"
                )

    def _log_trade(self):
        if self.engine.trades:
            last_trade = self.engine.trades[-1]
            self.pnl_history.append(last_trade.net_pnl)


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------


class Orchestrator:
    def __init__(self):
        self.console = Console()
        self.exchange = None
        self.session = None
        self.traders = []
        self.running = True

        # Setup Layout
        self.layout = Layout()
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=8),
        )
        self.layout["body"].split_row(Layout(name="btc"), Layout(name="eth"))

    async def setup(self):
        # Init Exchange
        self.exchange = ccxt.coinex(
            {
                "apiKey": CREDENTIALS["access_id"],
                "secret": CREDENTIALS["secret_key"],
                "enableRateLimit": True,
                "options": {"defaultType": "swap"},
            }
        )

        # Load Model
        self.session = ort.InferenceSession(MODEL_PATH)
        lock = asyncio.Lock()

        # Init Traders
        # Type ignored because 'assets' is a list of dicts, verified in logic
        assets: List[Any] = TRADING_CONFIG["assets"] # type: ignore
        for asset in assets:
            t = AssetTrader(
                asset["symbol"], asset["asset_id"], self.exchange, self.session, lock
            )
            self.traders.append(t)

        # Initial Fetch
        tasks = [t.fetch_initial_data() for t in self.traders]
        await asyncio.gather(*tasks)

    def generate_table(self, trader: AssetTrader) -> Table:
        t = Table(title=f"{trader.symbol} Trading", box=box.ROUNDED, expand=True)
        t.add_column("Metric", style="cyan")
        t.add_column("Value", style="yellow")

        t.add_row("Status", trader.status)
        t.add_row("Price", f"${trader.last_price:,.2f}")

        # Signal
        up = trader.last_pred["prob_up"]
        down = trader.last_pred["prob_down"]
        sig_color = "green" if up > down else "red"
        t.add_row("Signal", f"[{sig_color}]↑ {up:.1%} / ↓ {down:.1%}[/]")

        # Capital
        cap = trader.engine.capital
        pnl_color = "green" if cap >= 100 else "red"
        t.add_row("Capital", f"[{pnl_color}]${cap:.2f}[/]")

        # Position
        pos = trader.engine.current_position
        if pos:
            unrealized = pos.calculate_pnl(trader.last_price)
            p_color = "green" if unrealized >= 0 else "red"
            t.add_row("Position", f"{pos.side.value} ({pos.leverage}x)")
            t.add_row("Unrealized PnL", f"[{p_color}]${unrealized:.2f}[/]")
        else:
            t.add_row("Position", "WAITING")
            t.add_row("Unrealized PnL", "-")

        return t

    def update_ui(self):
        # Header
        grid = Table.grid(expand=True)
        grid.add_column(justify="center", ratio=1)
        grid.add_row("[bold magenta]JANUS LIVE SIMULATION TRADER[/bold magenta]")
        self.layout["header"].update(Panel(grid, style="white on blue"))

        # Traders
        if len(self.traders) >= 1:
            self.layout["btc"].update(
                Panel(self.generate_table(self.traders[0]), title="BTC")
            )
        if len(self.traders) >= 2:
            self.layout["eth"].update(
                Panel(self.generate_table(self.traders[1]), title="ETH")
            )

        # Footer
        logs = Table.grid(expand=True)
        logs.add_column(ratio=1)

        total_cap = sum(t.engine.capital for t in self.traders)
        start_cap = sum(float(t.config["capital_per_asset"]) for t in self.traders) # type: ignore
        total_pnl = total_cap - start_cap
        c = "green" if total_pnl >= 0 else "red"

        logs.add_row(
            f"Total Capital: ${total_cap:.2f} | Total PnL: [{c}]${total_pnl:.2f}[/]"
        )

        # Show last error if any
        all_errors = []
        for t in self.traders:
            if t.errors:
                all_errors.append(f"{t.symbol}: {t.errors[-1]}")

        if all_errors:
            logs.add_row(f"[red]Errors: {' | '.join(all_errors)}[/red]")
        else:
            logs.add_row("[green]System Healthy[/green]")

        self.layout["footer"].update(Panel(logs, title="System Status"))

    async def run(self):
        await self.setup()

        # Assigned to 'live' but not used, but Context Manager requires it.
        # Just suppressing the linter error by using it or ignoring it if possible.
        # ruff F841 complains about unused variable.
        with Live(self.layout, refresh_per_second=4, screen=True) as _:
            while self.running:
                # Tick Traders
                tasks = [t.tick() for t in self.traders]
                await asyncio.gather(*tasks)

                # Update UI
                self.update_ui()

                # Sleep (Rate limit safe)
                await asyncio.sleep(15)

    async def shutdown(self):
        self.running = False
        if self.exchange:
            await self.exchange.close()

        # Save History
        all_trades = []
        for t in self.traders:
            for trade in t.engine.trades:
                all_trades.append(
                    {
                        "asset": t.symbol,
                        "entry": str(trade.entry_timestamp),
                        "exit": str(trade.exit_timestamp),
                        "pnl": trade.net_pnl,
                        "side": str(trade.side),
                    }
                )

        with open("live_trades.json", "w") as f:
            json.dump(all_trades, f, indent=2)


# -----------------------------------------------------------------------------
# Main Entry
# -----------------------------------------------------------------------------


async def main():
    orch = Orchestrator()

    # Signal Handlers
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(orch.shutdown()))

    try:
        await orch.run()
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.critical(f"Fatal Error: {e}")
    finally:
        await orch.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
