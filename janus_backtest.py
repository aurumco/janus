import pandas as pd
import numpy as np
import joblib
import os
import warnings
from tqdm import tqdm
import pandas_ta as ta
from arch import arch_model
from statistics import mean

# Variables for configuration (updated per request)
INITIAL_CAPITAL_IRR = 2000000  # Initial capital in IRR (toman)
USD_IRR_RATE = 100000   # USD to IRR exchange rate (toman per USD)
BTC_USD_PRICE = 117160  # Current BTC price in USD
COMMISSION_RATE = 0.002 # Commission per trade side (0.2%)
LEVERAGE = 5            # Leverage factor
TRADES_PER_DAY_MIN = 5  # Min trades per day for simulation
TRADES_PER_DAY_MAX = 15 # Max trades per day for simulation
BACKTEST_START_DATE = '2021-03-01'
BACKTEST_END_DATE = '2021-04-01'
SIMULATION_DAYS = 30    # Days for forward simulation
RISK_PER_TRADE = 0.075  # Risk percentage per trade (7.5%)
STOP_LOSS_PCT = 0.01    # Average stop loss percentage (1%)
RISK_REWARD_RATIO = 20  # Target risk-reward ratio for estimation

# Dynamic trading parameters
PYRAMID_TRIGGER_PCT = 0.01   # Add to position every +1% in favor (fallback if ATR not available)
PYRAMID_ADD_FACTOR = 0.60    # Each add is 60% of initial position size
MAX_PYRAMIDS = 10            # Maximum number of adds per position
TREND_CHANGE_EXIT = True     # Exit immediately on trend reversal signal

# Exchange-like risk params
MAINT_MARGIN_RATE = 0.005    # 0.5% maintenance margin
ATR_LENGTH = 24
ATR_MULT_STRONG = 2.2
ATR_MULT_NORMAL = 1.6
BREAKEVEN_ATR = 0.7          # move stop to breakeven after +0.7*ATR in favor
ADD_STEP_ATR = 0.5           # add a new leg every +0.4*ATR in favor

# Scalping parameters to boost trade count
SCALP_ADD_ATR = 0.25         # add a scalp leg every +0.25*ATR from last add/anchor
SCALP_TP_ATR = 0.5           # scalp take-profit distance in ATR
SCALP_SL_ATR = 0.25          # scalp stop-loss distance in ATR
MAX_SCALP_LEGS = 5           # maximum simultaneous scalp legs within a position

# Fees and execution
MAKER_FEE = COMMISSION_RATE * 0.5
TAKER_FEE = COMMISSION_RATE
SLIPPAGE_BPS = 5             # 0.05% slippage per market order
MARGIN_MODE = 'cross'        # 'cross' or 'isolated'

# Funding (optional, per 8h). Positive means longs pay, negative means shorts pay.
FUNDING_RATE_BPS_8H = 0.0
FUNDING_DIRECTION = 1  # 1: longs pay, -1: shorts pay, 0: disabled

# Exposure cap: limit notional to capital * min(leverage, MAX_EXPOSURE_X)
MAX_EXPOSURE_X = 2

# Derived variables
INITIAL_CAPITAL_USD = INITIAL_CAPITAL_IRR / USD_IRR_RATE

# Setup
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='arch')
tqdm.pandas()

RESULTS_DIR = 'janus_v2_results'
BACKTEST_DATASET = 'janus_v2_dataset.parquet'  # Provided pre-built dataset (normalized features)
UNNORMALIZED_FILE = 'janus_v2_unnormalized.parquet'  # For prices/market columns
MODEL_FILE = os.path.join(RESULTS_DIR, 'janus_v2_model.joblib')
SCALER_FILE = 'janus_v2_scaler.joblib'

def safe_slice_by_date(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp']).set_index('timestamp')
        else:
            raise ValueError("DataFrame must be indexed by datetime or contain a 'timestamp' column")
    return df[(df.index >= pd.to_datetime(start_date)) & (df.index < pd.to_datetime(end_date))]

def map_regime_to_strategy(regime_number: int) -> str:
    strategy_map = {
    # Strong Bullish
    8: "Strong Bullish",
    5: "Strong Bullish",

    # Bullish
    0: "Bullish",
    2: "Bullish",

    # Neutral / Ranging
    1: "Neutral",
    4: "Neutral",

    # Bearish
    6: "Bearish",
    7: "Bearish",

    # Strong Bearish
    3: "Strong Bearish",
}
    return strategy_map.get(regime_number, "Unknown")

def frac_diff(series, d, thres=1e-5):
    weights = [1.]
    for k in range(1, len(series)):
        weight = -weights[-1] * (d - k + 1) / k
        if abs(weight) < thres:
            break
        weights.append(weight)
    weights = np.array(weights[::-1])
    diff_series = np.convolve(series, weights, mode='valid')
    return np.concatenate((np.full(len(series) - len(diff_series), np.nan), diff_series))

def garch_volatility_robust(returns):
    try:
        from arch import arch_model
    except Exception:
        return np.nan
    if returns.isnull().any() or len(returns) < 50:
        return np.nan
    try:
        model = arch_model(returns * 100, vol='Garch', p=1, q=1, rescale=False)
        res = model.fit(disp='off', show_warning=False)
        if getattr(res, 'convergence_flag', 0) == 0:
            forecast = res.forecast(horizon=1)
            return np.sqrt(forecast.variance.iloc[-1, 0])
        return np.nan
    except Exception:
        return np.nan

def build_dataset_for_range(start_date=BACKTEST_START_DATE, end_date=BACKTEST_END_DATE):
    print(f"Building dataset from btc.csv for {start_date} to {end_date}...")
    try:
        raw = pd.read_csv('btc.csv')
    except FileNotFoundError:
        print("Error: btc.csv not found. Cannot build dataset.")
        return False
    raw.columns = raw.columns.str.lower()
    raw['timestamp'] = pd.to_datetime(raw['timestamp'], unit='s', errors='coerce')
    raw = raw.dropna(subset=['timestamp']).set_index('timestamp')
    # Use a warm-up window to compute long lookback indicators
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    warmup_start = (start_dt - pd.Timedelta(days=180)).strftime('%Y-%m-%d')
    raw = safe_slice_by_date(raw, warmup_start, end_date)
    if raw.empty:
        print("No raw data in requested date range.")
        return False
    ohlcv_logic = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    df_h1 = raw.resample('1h').apply(ohlcv_logic).dropna()
    df_h4 = raw.resample('4h').apply(ohlcv_logic).dropna()
    df_d = raw.resample('1D').apply(ohlcv_logic).dropna()
    # Helper EMA & RSI to avoid pandas_ta None returns
    def ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False, min_periods=span).mean()
    def rsi(series: pd.Series, length: int = 14) -> pd.Series:
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
        roll_down = down.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
        rs = roll_up / roll_down.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
    # Compute higher timeframe indicators using robust helpers
    df_h4['EMA_diff_H4'] = ema(df_h4['close'], 6) - ema(df_h4['close'], 25)
    df_h4['RSI_H4'] = rsi(df_h4['close'], 6)
    df_d['EMA_diff_Daily'] = ema(df_d['close'], 10) - ema(df_d['close'], 50)
    df_d['RSI_Daily'] = rsi(df_d['close'], 14)
    # Merge down
    df_h1 = df_h1.join(df_h4[['EMA_diff_H4', 'RSI_H4']]).join(df_d[['EMA_diff_Daily', 'RSI_Daily']])
    df_h1 = df_h1.ffill()
    # H1 features
    df_h1['log_return'] = np.log(df_h1['close'] / df_h1['close'].shift(1))
    df_h1['EMA_diff_H1'] = ema(df_h1['close'], 24) - ema(df_h1['close'], 100)
    df_h1['RSI_H1'] = rsi(df_h1['close'], 24)
    df_h1['ATR_percent_H1'] = (ta.atr(high=df_h1['high'], low=df_h1['low'], close=df_h1['close'], length=24) / df_h1['close']) * 100
    df_h1['Volume_MA_H1'] = df_h1['volume'].rolling(window=24).mean()
    macd = ta.macd(df_h1['close'], fast=12, slow=26, signal=9)
    if isinstance(macd, pd.DataFrame):
        df_h1['MACD_line_H1'] = macd['MACD_12_26_9']
        df_h1['MACD_signal_H1'] = macd['MACDs_12_26_9']
    bb = ta.bbands(df_h1['close'], length=24)
    if isinstance(bb, pd.DataFrame) and 'BBB_24_2.0' in bb.columns:
        df_h1['BB_width_percent_H1'] = bb['BBB_24_2.0']
    df_h1['rolling_skew_H1'] = df_h1['log_return'].rolling(window=48).skew()
    # Advanced
    if len(df_h1) >= 260:  # only compute when enough data
        print("Calculating GARCH volatility (fast mode)...")
        df_h1['garch_volatility'] = df_h1['log_return'].rolling(window=240).apply(garch_volatility_robust, raw=False)
        print("Calculating Fractionally Differentiated series...")
        df_h1['frac_diff'] = frac_diff(df_h1['close'].values, d=0.5)
    else:
        df_h1['garch_volatility'] = np.nan
        df_h1['frac_diff'] = np.nan
    # Restrict to target period after features are ready
    df_h1 = df_h1[(df_h1.index >= start_dt) & (df_h1.index < end_dt)]
    # Impute remaining NaNs for advanced features so scaler can transform
    for c in ['garch_volatility', 'frac_diff', 'MACD_line_H1', 'MACD_signal_H1', 'BB_width_percent_H1', 'rolling_skew_H1',
              'EMA_diff_H4','RSI_H4','EMA_diff_Daily','RSI_Daily','EMA_diff_H1','RSI_H1','ATR_percent_H1','Volume_MA_H1','log_return']:
        if c in df_h1.columns:
            if df_h1[c].isna().any():
                df_h1[c] = df_h1[c].fillna(method='ffill').fillna(method='bfill').fillna(0)
    # Drop rows with missing OHLC only
    df_h1 = df_h1.dropna(subset=['close'])
    # Columns and normalization
    feature_columns = [
        'log_return', 'EMA_diff_H1', 'RSI_H1', 'ATR_percent_H1', 'Volume_MA_H1',
        'MACD_line_H1', 'MACD_signal_H1', 'BB_width_percent_H1', 'rolling_skew_H1',
        'EMA_diff_H4', 'RSI_H4', 'EMA_diff_Daily', 'RSI_Daily',
        'garch_volatility', 'frac_diff'
    ]
    # Persist unnormalized ohlcv for trading
    unnorm_cols = [c for c in ['open','high','low','close','volume'] if c in df_h1.columns]
    df_h1[unnorm_cols].to_parquet(UNNORMALIZED_FILE)
    # Normalize
    janus_df = df_h1[feature_columns].copy()
    if os.path.exists(SCALER_FILE):
        scaler = joblib.load(SCALER_FILE)
        # align columns if needed
        if hasattr(scaler, 'feature_names_in_'):
            expected = list(scaler.feature_names_in_)
            missing = [c for c in expected if c not in janus_df.columns]
            if missing:
                print(f"Warning: Missing expected features {missing}; proceeding with available columns order.")
        normalized = scaler.transform(janus_df.values)
    else:
        print("Scaler not found; fitting a new StandardScaler (note: may differ from training)")
        scaler = StandardScaler()
        normalized = scaler.fit_transform(janus_df.values)
        joblib.dump(scaler, SCALER_FILE)
    if len(janus_df) == 0:
        print("No feature rows available after processing; aborting build.")
        return False
    normalized_df = pd.DataFrame(normalized, index=janus_df.index, columns=janus_df.columns)
    normalized_df.round(5).to_parquet(BACKTEST_DATASET)
    print(f"Dataset built and saved: {BACKTEST_DATASET} with {len(normalized_df)} rows")
    return True

def load_backtest_data(start_date=BACKTEST_START_DATE, end_date=BACKTEST_END_DATE):
    print(f"Loading backtest dataset {BACKTEST_DATASET} and prices from {UNNORMALIZED_FILE}...")
    try:
        features_df = pd.read_parquet(BACKTEST_DATASET)
    except FileNotFoundError:
        print(f"Dataset not found at {BACKTEST_DATASET}. Attempting to build it...")
        if not build_dataset_for_range(start_date, end_date):
            return None, None
        features_df = pd.read_parquet(BACKTEST_DATASET)
    # Load prices (close) and any OHLC if available
    prices_df = None
    if os.path.exists(UNNORMALIZED_FILE):
        try:
            prices_df = pd.read_parquet(UNNORMALIZED_FILE)
        except Exception:
            prices_df = None
    # Ensure datetime index and slice
    features_df = safe_slice_by_date(features_df.copy(), start_date, end_date)
    # If too few rows (e.g., 1), rebuild from raw
    if len(features_df) < 100:
        print(f"Dataset has only {len(features_df)} rows in range; rebuilding...")
        if build_dataset_for_range(start_date, end_date):
            features_df = pd.read_parquet(BACKTEST_DATASET)
            features_df = safe_slice_by_date(features_df.copy(), start_date, end_date)
    if prices_df is not None:
        prices_df = safe_slice_by_date(prices_df.copy(), start_date, end_date)
        # Align to features index
        prices_df = prices_df.reindex(features_df.index).ffill().bfill()
    return features_df, prices_df

def allocate_risk_by_signal(signal: str) -> float:
    # Strong signals use more risk; neutral is skipped in trading engine
    if signal == 'Strong Bullish' or signal == 'Strong Bearish':
        return RISK_PER_TRADE * 1.5
    if signal == 'Bullish' or signal == 'Bearish':
        return RISK_PER_TRADE
    return 0.0

def backtest_engine(index: pd.DatetimeIndex,
                    prices: pd.Series,
                    signals: pd.Series,
                    commission_rate: float = COMMISSION_RATE,
                    leverage: float = LEVERAGE,
                    atr_series: pd.Series | None = None):
    # State
    capital = INITIAL_CAPITAL_USD
    equity_curve = []
    position = None  # dict with: side, legs: [leg...], scalp_legs: [leg...], adds, entry_ts, last_add_price, last_scalp_anchor
    trade_log = []   # list of completed trades (dict)
    funding_total = 0.0

    def current_price(i):
        return float(prices.iloc[i])

    def apply_slippage(px: float, side: str, is_entry: bool = True) -> float:
        slip = SLIPPAGE_BPS / 10000.0
        if is_entry:
            # marketable entry: longs pay up, shorts sell down
            return px * (1 + slip) if side == 'Long' else px * (1 - slip)
        else:
            # exit: longs sell down, shorts buy up
            return px * (1 - slip) if side == 'Long' else px * (1 + slip)

    # Align signals to index
    signals = signals.reindex(index).ffill().bfill()
    for i in range(len(index)):
        ts = index[i]
        price = current_price(i)
        if not np.isfinite(price):
            # Skip bar if price is invalid; carry forward equity
            equity_curve.append({'ts': ts, 'capital_usd': capital})
            continue
        signal = signals.iloc[i]

        # Manage open position (multi-leg)
        if position is not None:
            side = position['side']
            direction = 1 if side == 'Long' else -1
            adds = position['adds']
            # Update trailing stops and check exits for each leg
            remaining_legs = []
            total_notional = 0.0
            leg_exit_reason = None
            # Process main legs exits/updates
            for leg in position['legs']:
                entry = leg['entry_price']
                size = leg['size_usd']
                stop = leg['stop_price']
                take = leg['take_price']
                atr = atr_series.iloc[i] if atr_series is not None else np.nan
                # Update trailing stop if ATR available
                if np.isfinite(atr):
                    mult = ATR_MULT_STRONG if 'Strong' in signal else ATR_MULT_NORMAL
                    if side == 'Long':
                        trail = price - mult * atr
                        # move stop only upwards
                        stop = max(stop, trail)
                        # breakeven
                        if (price - entry) >= BREAKEVEN_ATR * atr:
                            stop = max(stop, entry)
                    else:
                        trail = price + mult * atr
                        stop = min(stop, trail)
                        if (entry - price) >= BREAKEVEN_ATR * atr:
                            stop = min(stop, entry)
                    leg['stop_price'] = stop
                # Liquidation (simplified): if unrealized loss exceeds margin - maintenance
                margin_used = size / max(leverage, 1e-9)
                unreal_pnl = direction * (price / entry - 1) * size
                maint_margin = margin_used * MAINT_MARGIN_RATE
                if -unreal_pnl > (margin_used - maint_margin):
                    # force exit at current price
                    pnl = - (margin_used - maint_margin)
                    exit_comm = commission_rate * size
                    capital += pnl - exit_comm
                    trade_log.append({
                        'side': side,
                        'entry_ts': leg['entry_ts'],
                        'exit_ts': ts,
                        'entry_price': float(entry),
                        'exit_price': float(price),
                        'size_usd': float(size),
                        'adds': int(adds),
                        'reason': 'liquidation',
                        'pnl_usd': float(pnl),
                        'pnl_pct_on_margin': float(pnl / max(1e-9, margin_used) * 100.0),
                        'entry_commission': float(leg.get('entry_commission', 0.0)),
                        'exit_commission': float(exit_comm),
                        'capital_entry': float(leg.get('capital_entry', np.nan)),
                    })
                    continue
                # Check stop/take
                hit_stop = (side == 'Long' and price <= stop) or (side == 'Short' and price >= stop)
                hit_take = (side == 'Long' and price >= take) or (side == 'Short' and price <= take)
                if hit_stop or hit_take:
                    raw_exit_px = stop if hit_stop else take
                    exit_px = apply_slippage(raw_exit_px, side, is_entry=False)
                    pnl = direction * (exit_px / entry - 1) * size
                    exit_comm = TAKER_FEE * size
                    capital += pnl - exit_comm
                    trade_log.append({
                        'side': side,
                        'entry_ts': leg['entry_ts'],
                        'exit_ts': ts,
                        'entry_price': float(entry),
                        'exit_price': float(exit_px),
                        'size_usd': float(size),
                        'adds': int(adds),
                        'reason': 'stop' if hit_stop else 'take',
                        'pnl_usd': float(pnl),
                        'pnl_pct_on_margin': float(pnl / max(1e-9, size / max(leverage,1e-9)) * 100.0),
                        'entry_commission': float(leg.get('entry_commission', 0.0)),
                        'exit_commission': float(exit_comm),
                        'capital_entry': float(leg.get('capital_entry', np.nan)),
                    })
                else:
                    remaining_legs.append(leg)
                    total_notional += size
            position['legs'] = remaining_legs
            # If no legs remain, flat
            if len(position['legs']) == 0:
                position = None
            else:
                # Trend change exit: close all remaining legs
                if TREND_CHANGE_EXIT and ((side == 'Long' and ('Bearish' in signal)) or (side == 'Short' and ('Bullish' in signal)) or (signal == 'Neutral')):
                    for leg in position['legs']:
                        entry = leg['entry_price']
                        size = leg['size_usd']
                        pnl = direction * (price / entry - 1) * size
                        exit_comm = commission_rate * size
                        capital += pnl - exit_comm
                        trade_log.append({
                            'side': side,
                            'entry_ts': leg['entry_ts'],
                            'exit_ts': ts,
                            'entry_price': float(entry),
                            'exit_price': float(price),
                            'size_usd': float(size),
                            'adds': int(position['adds']),
                            'reason': 'trend_change',
                            'pnl_usd': float(pnl),
                            'pnl_pct_on_margin': float(pnl / max(1e-9, size / max(leverage,1e-9)) * 100.0),
                            'entry_commission': float(leg.get('entry_commission', 0.0)),
                            'exit_commission': float(exit_comm),
                            'capital_entry': float(leg.get('capital_entry', np.nan)),
                        })
                    position = None
                else:
                    # Add new leg when in profit by step (ATR or pct)
                    if adds < MAX_PYRAMIDS:
                        anchor = position.get('last_add_price', position['legs'][0]['entry_price'])
                        can_add = False
                        if atr_series is not None and np.isfinite(atr_series.iloc[i]):
                            step = ADD_STEP_ATR * atr_series.iloc[i]
                            can_add = (direction == 1 and price - anchor >= step) or (direction == -1 and anchor - price >= step)
                        else:
                            can_add = (direction == 1 and (price / anchor - 1) >= PYRAMID_TRIGGER_PCT * (adds + 1)) or (direction == -1 and (anchor / price - 1) >= PYRAMID_TRIGGER_PCT * (adds + 1))
                        if can_add:
                            # size based on initial leg size * factor, capped by notional
                            current_notional = sum(l['size_usd'] for l in position['legs'])
                            max_notional = capital * min(leverage, MAX_EXPOSURE_X)
                            init_size = position['legs'][0]['size_usd']
                            add_size = min(init_size * PYRAMID_ADD_FACTOR, max(0.0, max_notional - current_notional))
                            if add_size > 0:
                                stop_pct = STOP_LOSS_PCT
                                entry_comm = commission_rate * add_size
                                capital -= entry_comm
                                leg = {
                                    'entry_ts': ts,
                                    'entry_price': price,
                                    'size_usd': add_size,
                                    'stop_price': price * (1 - stop_pct) if side == 'Long' else price * (1 + stop_pct),
                                    'take_price': price * (1 + stop_pct * RISK_REWARD_RATIO) if side == 'Long' else price * (1 - stop_pct * RISK_REWARD_RATIO),
                                    'entry_commission': entry_comm,
                                    'capital_entry': capital + entry_comm,
                                }
                                position['legs'].append(leg)
                                position['adds'] += 1
                                position['last_add_price'] = price

                    # Add scalp legs more frequently on micro-steps
                    if atr_series is not None and np.isfinite(atr_series.iloc[i]) and len(position.get('scalp_legs', [])) < MAX_SCALP_LEGS:
                        scalp_anchor = position.get('last_scalp_anchor', position['legs'][0]['entry_price'])
                        step = SCALP_ADD_ATR * atr_series.iloc[i]
                        can_scalp = (direction == 1 and price - scalp_anchor >= step) or (direction == -1 and scalp_anchor - price >= step)
                        if can_scalp:
                            current_notional = sum(l['size_usd'] for l in position['legs'] + position.get('scalp_legs', []))
                            max_notional = capital * leverage
                            init_size = position['legs'][0]['size_usd']
                            scalp_size = min(init_size * (PYRAMID_ADD_FACTOR * 0.5), max(0.0, max_notional - current_notional))
                            if scalp_size > 0:
                                entry_px = apply_slippage(price, side, is_entry=True)
                                dist = atr_series.iloc[i]
                                stop_price = entry_px - SCALP_SL_ATR * dist if side == 'Long' else entry_px + SCALP_SL_ATR * dist
                                take_price = entry_px + SCALP_TP_ATR * dist if side == 'Long' else entry_px - SCALP_TP_ATR * dist
                                entry_comm = TAKER_FEE * scalp_size
                                capital -= entry_comm
                                leg = {
                                    'entry_ts': ts,
                                    'entry_price': entry_px,
                                    'size_usd': scalp_size,
                                    'stop_price': stop_price,
                                    'take_price': take_price,
                                    'entry_commission': entry_comm,
                                    'capital_entry': capital + entry_comm,
                                }
                                position.setdefault('scalp_legs', []).append(leg)
                                position['last_scalp_anchor'] = price

        # If flat, consider new entry
        if position is None:
            if signal in ('Bullish', 'Strong Bullish', 'Bearish', 'Strong Bearish'):
                side = 'Long' if 'Bullish' in signal else 'Short'
                risk_pct = allocate_risk_by_signal(signal)
                if risk_pct > 0:
                    # Position sizing based on fixed stop percentage
                    stop_pct = STOP_LOSS_PCT
                    # Notional exposure implied by risk model
                    desired_size = max(0.0, (risk_pct * capital) / stop_pct)
                    max_notional = capital * min(leverage, MAX_EXPOSURE_X)
                    size_usd = min(desired_size, max_notional)
                    if size_usd > 0:
                        entry_px = apply_slippage(price, side, is_entry=True)
                        # initial ATR-based stop/take distances if available
                        if atr_series is not None and np.isfinite(atr_series.iloc[i]):
                            mult = ATR_MULT_STRONG if 'Strong' in signal else ATR_MULT_NORMAL
                            dist = mult * atr_series.iloc[i]
                            stop_price = entry_px - dist if side == 'Long' else entry_px + dist
                            take_price = entry_px + dist * RISK_REWARD_RATIO if side == 'Long' else entry_px - dist * RISK_REWARD_RATIO
                        else:
                            stop_price = entry_px * (1 - stop_pct) if side == 'Long' else entry_px * (1 + stop_pct)
                            take_price = entry_px * (1 + stop_pct * RISK_REWARD_RATIO) if side == 'Long' else entry_px * (1 - stop_pct * RISK_REWARD_RATIO)
                        entry_commission = TAKER_FEE * size_usd
                        capital -= entry_commission  # entry commission
                        position = {
                            'side': side,
                            'legs': [{
                                'entry_ts': ts,
                                'entry_price': entry_px,
                                'size_usd': size_usd,
                                'stop_price': stop_price,
                                'take_price': take_price,
                                'entry_commission': entry_commission,
                                'capital_entry': capital + entry_commission,
                            }],
                            'scalp_legs': [],
                            'adds': 0,
                            'entry_ts': ts,
                            'last_add_price': price,
                            'last_scalp_anchor': price,
                        }

        equity_curve.append({'ts': ts, 'capital_usd': capital})

    # Close any open position at the end at market price
    if position is not None:
        price = prices.iloc[-1]
        side = position['side']
        direction = 1 if side == 'Long' else -1
        for leg in position['legs'] + position.get('scalp_legs', []):
            entry = leg['entry_price']
            size = leg['size_usd']
            exit_px = apply_slippage(price, side, is_entry=False)
            pnl = direction * (exit_px / entry - 1) * size if (np.isfinite(entry) and entry > 0) else 0.0
            exit_commission = TAKER_FEE * size
            capital += pnl - exit_commission
            trade_log.append({
                'side': side,
                'entry_ts': leg.get('entry_ts', index[-1]),
                'exit_ts': index[-1],
                'entry_price': float(entry),
                'exit_price': float(exit_px),
                'size_usd': float(size),
                'adds': int(position['adds']),
                'reason': 'eod',
                'pnl_usd': float(pnl),
                'pnl_pct_on_margin': float(pnl / max(1e-9, size / max(leverage,1e-9)) * 100.0),
                'entry_commission': float(leg.get('entry_commission', 0.0)),
                'exit_commission': float(exit_commission),
                'capital_entry': float(leg.get('capital_entry', np.nan)),
            })

    equity_df = pd.DataFrame(equity_curve).set_index('ts')

    # Metrics
    trades = pd.DataFrame(trade_log)
    if not trades.empty and 'pnl_usd' in trades.columns:
        trades['pnl_usd'] = pd.to_numeric(trades['pnl_usd'], errors='coerce')
    clean_trades = trades.dropna(subset=['pnl_usd']) if not trades.empty else trades
    win_rate = (clean_trades['pnl_usd'] > 0).mean() * 100 if len(clean_trades) else 0.0
    avg_win = clean_trades.loc[clean_trades['pnl_usd'] > 0, 'pnl_usd'].mean() if len(clean_trades) else 0.0
    avg_loss = clean_trades.loc[clean_trades['pnl_usd'] <= 0, 'pnl_usd'].mean() if len(clean_trades) else 0.0

    return capital, equity_df, trades, {
        'Trades Count': int(len(trades)),
        'Win Rate (%)': float(win_rate),
        'Avg Win (USD)': float(0.0 if np.isnan(avg_win) else avg_win),
        'Avg Loss (USD)': float(0.0 if np.isnan(avg_loss) else avg_loss),
    }

def predict_signals(features_df: pd.DataFrame):
    try:
        model = joblib.load(MODEL_FILE)
    except FileNotFoundError:
        print(f"Error: Model not found at {MODEL_FILE}")
        return None
    regimes = model.predict(features_df.values)
    signals = pd.Series(regimes, index=features_df.index).apply(map_regime_to_strategy)
    return signals

def ensure_close_series(prices_df: pd.DataFrame, fallback_features: pd.DataFrame) -> pd.Series:
    # Try common price columns
    for c in ['close', 'Close', 'price', 'Price']:
        if c in prices_df.columns:
            return prices_df[c].astype(float)
    # If not present, attempt to reconstruct from features if any
    for c in ['close', 'Close', 'price', 'Price']:
        if c in fallback_features.columns:
            return fallback_features[c].astype(float)
    raise ValueError("No close price column found in provided dataframes")

def report_results(final_capital_usd: float, trades: pd.DataFrame, equity: pd.DataFrame):
    final_capital_irr = final_capital_usd * USD_IRR_RATE
    growth_pct = ((final_capital_usd / INITIAL_CAPITAL_USD) - 1.0) * 100.0
    trades = trades.copy()
    if not trades.empty and 'pnl_usd' in trades.columns:
        trades['pnl_usd'] = pd.to_numeric(trades['pnl_usd'], errors='coerce')
    win_rate = (trades['pnl_usd'] > 0).mean() * 100 if len(trades) else 0.0
    print("\n=== Janus V2 Backtest (Dynamic Trend-Following) ===")
    print(f"Period: {BACKTEST_START_DATE} to {BACKTEST_END_DATE}")
    print(f"Bars Analyzed: {len(equity)}")
    print(f"Initial Capital: {INITIAL_CAPITAL_IRR:,} IRR ({INITIAL_CAPITAL_USD:.2f} USD)")
    print(f"Final Capital:   {final_capital_irr:,.0f} IRR ({final_capital_usd:.2f} USD)")
    print(f"Total Return:    {growth_pct:.2f}%")
    print(f"Trades: {len(trades)} | Win Rate: {win_rate:.2f}%")
    if len(trades):
        print(f"Avg Win (USD): {trades.loc[trades['pnl_usd']>0,'pnl_usd'].mean():.2f}")
        print(f"Avg Loss (USD): {trades.loc[trades['pnl_usd']<=0,'pnl_usd'].mean():.2f}")
        # Detailed trade log
        print("\n--- Detailed Trades ---")
        cols = ['entry_ts','exit_ts','side','entry_price','exit_price','size_usd','adds','reason','pnl_usd','pnl_pct_on_margin','entry_commission','exit_commission']
        trades_to_show = trades[cols] if all(c in trades.columns for c in cols) else trades
        for _, row in trades_to_show.iterrows():
            print(f"{row.get('entry_ts')} -> {row.get('exit_ts')} | {row.get('side')} | size ${row.get('size_usd'):.2f} | entry {row.get('entry_price'):.2f} -> exit {row.get('exit_price'):.2f} | adds {int(row.get('adds',0))} | reason {row.get('reason')} | pnl ${row.get('pnl_usd'):.2f} ({row.get('pnl_pct_on_margin', np.nan):.2f}%)")

def compute_atr_series(prices_df: pd.DataFrame, length: int = ATR_LENGTH) -> pd.Series:
    if prices_df is None:
        return None
    required = {'high','low','close'}
    if not required.issubset(set(prices_df.columns)):
        return None
    atr = ta.atr(high=prices_df['high'], low=prices_df['low'], close=prices_df['close'], length=length)
    return atr

def save_results(trades: pd.DataFrame, equity: pd.DataFrame, start_date: str, end_date: str):
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        tag = f"{start_date}_to_{end_date}"
        trades_path = os.path.join(RESULTS_DIR, f"trades_{tag}.csv")
        equity_path = os.path.join(RESULTS_DIR, f"equity_{tag}.csv")
        trades.to_csv(trades_path, index=False)
        equity.to_csv(equity_path)
        print(f"Saved trades to {trades_path}")
        print(f"Saved equity to {equity_path}")
    except Exception as e:
        print(f"Warning: failed to save results: {e}")

if __name__ == "__main__":
    # Load features (normalized) and prices
    features_df, prices_df = load_backtest_data()
    if features_df is None:
        exit()
    signals = predict_signals(features_df)
    if signals is None:
        exit()
    # Price series
    if prices_df is None:
        prices_df = features_df.copy()
    close_series = ensure_close_series(prices_df, features_df)
    close_series = close_series.reindex(features_df.index).ffill().bfill()

    # Run dynamic backtest
    final_capital_usd, equity_df, trades_df, metrics = backtest_engine(
        index=features_df.index,
        prices=close_series,
        signals=signals,
        commission_rate=COMMISSION_RATE,
        leverage=LEVERAGE,
    )

    # Report
    report_results(final_capital_usd, trades_df, equity_df)