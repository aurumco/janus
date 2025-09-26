# Janus Dataset Construction

#---------------------------------------------------------------------------
# Core imports
#---------------------------------------------------------------------------
import pandas as pd
import pandas_ta as ta
import numpy as np
import time
import warnings
from sklearn.preprocessing import MinMaxScaler
from arch import arch_model
import joblib

# Diagnostics verbosity
VERBOSE = False

# Suppress convergence warnings from the GARCH model for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='arch')
print("Janus dataset engine initialized.\n")
print("[1/4] Load & resample data...")

#---------------------------------------------------------------------------
# Critical hyperparameters (easy to tune)
#---------------------------------------------------------------------------
ASSET = "BTC/USDT"
BASE_TIMEFRAME = "15min"
TRAIN_START = "2023-01-01"
TRAIN_END = "2025-06-30"
TEST_START = "2025-07-01"

INPUT_WINDOW_CANDLES = 27      # past bars for pattern capture
TARGET_WINDOW_CANDLES = 42      # future bars for quick moves
LOOKAHEAD_CANDLES = TARGET_WINDOW_CANDLES
STOP_LOSS_PCT = 0.003           # 0.3%
PROFIT_THRESHOLDS = [0.02, 0.05, 0.10]  # 2%, 5%, 10%

SCALER_FEATURE_RANGE = (-1.0, 1.0)
ROUND_DECIMALS = 5

# Indicator and window lengths
M15_RSI_LEN = 14
M15_ATR_LEN = 5
M15_EMA_LEN = 10
H1_RSI_LEN = 14
H1_ADX_LEN = 14
H4_EMA_LEN = 21
H4_ADX_LEN = 14
H4_RSI_LEN = 14
DAILY_RSI_LEN = 14

# Slope and time encodings
EMA_SLOPE_LAG = 1
HOUR_PERIOD = 24

# GARCH and rolling
GARCH_ROLL_WINDOW = 240         # ~10 days of H1
ENABLE_GARCH = True            # toggle to enable/disable GARCH

# Volume oscillator (PVO) parameters
PVO_FAST = 12
PVO_SLOW = 26
PVO_SIGNAL = 9

# Session window (UTC)
NY_HOURS_START = 13
NY_HOURS_END = 17               # inclusive

# File names
SCALER_PATH = 'janus_m15_scaler.joblib'
PARQUET_PATH = 'janus_m15_dataset.parquet'
CSV_PATH = 'janus_m15_dataset.csv'

# Data sufficiency
MIN_REQUIRED_BARS = 120

# Labeling SL filter (directional adverse excursion)
SL_FILTER_PCT = STOP_LOSS_PCT   # tie to STOP_LOSS_PCT for consistency

#---------------------------------------------------------------------------
# Step 1: Helper Functions (FracDiff, Robust GARCH, Triple-Barrier Labels)
#---------------------------------------------------------------------------
def frac_diff(series, d, thres=1e-5):
    """Fractional differencing (kept for potential future use)."""
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
    """
    Fits a GARCH(1,1) model robustly, checking for convergence.
    Returns forecasted volatility only if the model converges successfully.
    """
    if returns.isnull().any() or len(returns) < 50: # Need enough data points
        return np.nan
    try:
        # We multiply by 100 as GARCH models work better with percentages
        model = arch_model(returns * 100, vol='Garch', p=1, q=1, rescale=False)
        res = model.fit(disp='off', show_warning=False)
        # CRITICAL: Check if the optimization was successful
        if res.convergence_flag == 0:
            # Forecast 1 step ahead and return the annualized volatility
            forecast = res.forecast(horizon=1)
            return np.sqrt(forecast.variance.iloc[-1, 0])
        else:
            return np.nan
    except Exception:
        return np.nan

def get_triple_barrier_labels(*args, **kwargs):
    """Deprecated in this pipeline."""
    raise NotImplementedError("Use hunter_style_labels() instead.")

def get_triple_barrier_labels_dynamic(*args, **kwargs):
    """Deprecated in this pipeline."""
    raise NotImplementedError("Use hunter_style_labels() instead.")

def hunter_style_labels(*args, **kwargs):
    """Deprecated in this pipeline."""
    raise NotImplementedError("Use hybrid_labels_5c() instead.")

def hybrid_labels_5c(ohlcv_m15: pd.DataFrame, lookahead: int, sl_filter_pct: float) -> pd.Series:
    """Hybrid 5-class labeling on M15.
    Classes: 0=strong sell, 1=sell, 2=neutral, 3=buy, 4=strong buy
    price_change = (close[i+lookahead] - close[i]) / close[i] * 100
    SL filter: if adverse excursion beyond sl_filter_pct in either direction within lookahead -> neutral.
    """
    n = len(ohlcv_m15)
    close = ohlcv_m15['close'].values
    high = ohlcv_m15['high'].values
    low = ohlcv_m15['low'].values
    labels = np.full(n, 2, dtype=np.int8)
    sl_neutralized = 0

    for i in range(0, n - lookahead):
        c0 = close[i]
        cf = close[i + lookahead]
        j0, j1 = i + 1, i + lookahead
        win_high = high[j0:j1+1]
        win_low = low[j0:j1+1]
        if win_high.size == 0:
            continue

        pc = (cf - c0) / c0 * 100.0
        # directional SL filter
        up_excursion = (np.max(win_high) - c0) / c0
        down_excursion = (c0 - np.min(win_low)) / c0
        if pc > 0 and down_excursion > sl_filter_pct:
            labels[i] = 2
            sl_neutralized += 1
            continue
        if pc < 0 and up_excursion > sl_filter_pct:
            labels[i] = 2
            sl_neutralized += 1
            continue
        if pc > 2.0:
            labels[i] = 4
        elif 0.5 < pc <= 2.0:
            labels[i] = 3
        elif -0.5 <= pc <= 0.5:
            labels[i] = 2
        elif -2.0 <= pc < -0.5:
            labels[i] = 1
        elif pc < -2.0:
            labels[i] = 0

    s = pd.Series(labels, index=ohlcv_m15.index, dtype=np.int8)
    s.attrs['sl_neutralized'] = sl_neutralized
    return s

# Load, clean, and prepare multi-timeframe data
if VERBOSE:
    print("Loading data and preparing timeframes (M15 base + H1/H4/Daily)...")
start_time = time.time()
try:
    df = pd.read_csv('btc.csv')
except FileNotFoundError:
    print("Error: btc.csv not found.")
    exit()

df.columns = df.columns.str.lower()
# Auto-detect timestamp unit
ts_raw = pd.to_numeric(df['timestamp'], errors='coerce')
unit = 'ms' if ts_raw.max() > 1e12 else 's'
df['timestamp'] = pd.to_datetime(ts_raw, unit=unit, errors='coerce')
df.dropna(subset=['timestamp'], inplace=True)
df.set_index('timestamp', inplace=True)

df_filtered = df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)]
if df_filtered.empty:
    print("Error: No data found for 2023 or later.")
    exit()

# Create M15, H1, H4, and Daily DataFrames
ohlcv_logic = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
base_timeframe = BASE_TIMEFRAME
df_m15 = df_filtered.resample(base_timeframe).apply(ohlcv_logic).dropna()
df_h1 = df_filtered.resample('1h').apply(ohlcv_logic).dropna()
df_h4 = df_filtered.resample('4h').apply(ohlcv_logic).dropna()
df_daily = df_filtered.resample('D').apply(ohlcv_logic).dropna()

# Ensure strict clamp to the requested training span
df_m15 = df_m15[(df_m15.index >= TRAIN_START) & (df_m15.index <= TRAIN_END)]
df_h1 = df_h1[(df_h1.index >= TRAIN_START) & (df_h1.index <= TRAIN_END)]
df_h4 = df_h4[(df_h4.index >= TRAIN_START) & (df_h4.index <= TRAIN_END)]
df_daily = df_daily[(df_daily.index >= TRAIN_START) & (df_daily.index <= TRAIN_END)]

if VERBOSE:
    print("Timeframes ready.")

# Feature engineering (higher TF features + M15 short-term features)
print("[2/4] Engineer features...")

# Higher TF indicators (H4, Daily)
ema21_h4 = df_h4.ta.ema(length=H4_EMA_LEN)
df_h4['EMA_diff_21_H4pct'] = (df_h4['close'] - ema21_h4) / df_h4['close']
df_h4['RSI_H4'] = df_h4.ta.rsi(length=H4_RSI_LEN)
df_h4['ADX_H4'] = df_h4.ta.adx(length=H4_ADX_LEN)['ADX_14']
df_daily['RSI_Daily'] = df_daily.ta.rsi(length=DAILY_RSI_LEN)

"""H1 features core"""
h4_features = df_h4[['EMA_diff_21_H4pct', 'RSI_H4']]
daily_features = df_daily[['RSI_Daily']]
df_h1 = df_h1.join(h4_features).join(daily_features)
df_h1 = df_h1.ffill()

if VERBOSE:
    print("  H1 core features...")
df_h1['log_return'] = np.log(df_h1['close'] / df_h1['close'].shift(1))
df_h1['RSI_14_H1'] = df_h1.ta.rsi(length=H1_RSI_LEN)
df_h1['ADX_14_H1'] = df_h1.ta.adx(length=H1_ADX_LEN)['ADX_14']

if ENABLE_GARCH:
    if VERBOSE:
        print("  GARCH volatility (this may take a few minutes)...")
    if len(df_h1) >= GARCH_ROLL_WINDOW:
        df_h1['garch_volatility'] = df_h1['log_return'].rolling(window=GARCH_ROLL_WINDOW).apply(garch_volatility_robust, raw=False)
    else:
        df_h1['garch_volatility'] = np.nan
else:
    if VERBOSE:
        print("  GARCH disabled -> skipping volatility computation.")
    df_h1['garch_volatility'] = np.nan
# Do not drop the entire H1 frame; fill essential cols to avoid full-NaN after alignment
essential_h1 = ['log_return', 'RSI_14_H1', 'ADX_14_H1']
df_h1[essential_h1] = df_h1[essential_h1].ffill().bfill()
if ENABLE_GARCH:
    df_h1['garch_volatility'] = df_h1['garch_volatility'].ffill().bfill()

# Fallback: if RSI/ADX remain entirely NaN, compute basic versions again
if df_h1['RSI_14_H1'].isna().all():
    df_h1['RSI_14_H1'] = df_h1.ta.rsi(length=H1_RSI_LEN)
if df_h1['ADX_14_H1'].isna().all():
    df_h1['ADX_14_H1'] = df_h1.ta.adx(length=H1_ADX_LEN)['ADX_14']
if VERBOSE:
    _nn = {k: int(v) for k, v in df_h1[essential_h1].notna().sum().to_dict().items()}
    print(f"  H1 non-null: {_nn}")

if VERBOSE:
    print("  M15 short-term features...")
df_m15['RSI_14_M15'] = df_m15.ta.rsi(length=M15_RSI_LEN)
df_m15['ATR_5_pct_M15'] = (df_m15.ta.atr(length=M15_ATR_LEN) / df_m15['close'])
ema10_m15 = df_m15.ta.ema(length=M15_EMA_LEN)
df_m15['dist_from_ema_10_M15'] = (df_m15['close'] - ema10_m15) / df_m15['close']
df_m15['ema10_slope_M15'] = ema10_m15.diff(EMA_SLOPE_LAG)
pvo = ta.pvo(volume=df_m15['volume'], fast=PVO_FAST, slow=PVO_SLOW, signal=PVO_SIGNAL)
df_m15['volume_oscillator_M15'] = pvo.iloc[:, 0] if pvo is not None else np.nan
df_m15['obv_M15'] = ta.obv(close=df_m15['close'], volume=df_m15['volume'])

"""Time-based features"""
df_m15['hour_of_day'] = df_m15.index.hour
df_m15['hour_sin'] = np.sin(2 * np.pi * df_m15['hour_of_day'] / HOUR_PERIOD)
df_m15['hour_cos'] = np.cos(2 * np.pi * df_m15['hour_of_day'] / HOUR_PERIOD)

if VERBOSE:
    print("  Merging higher timeframe features into M15 base...")
cols_h1 = ['RSI_14_H1', 'ADX_14_H1', 'garch_volatility']
cols_h4 = ['EMA_diff_21_H4pct']

# Fallback: recompute H1 RSI/ADX from M15 if H1 is mostly NaN
h1_nan_ratio_rsi = float(df_h1['RSI_14_H1'].isna().mean()) if 'RSI_14_H1' in df_h1.columns else 1.0
h1_nan_ratio_adx = float(df_h1['ADX_14_H1'].isna().mean()) if 'ADX_14_H1' in df_h1.columns else 1.0
if h1_nan_ratio_rsi > 0.95 or h1_nan_ratio_adx > 0.95:
    df_h1_alt = df_m15[['open','high','low','close','volume']].resample('1h').apply({'open':'first','high':'max','low':'min','close':'last','volume':'sum'})
    df_h1_alt['RSI_14_H1'] = df_h1_alt.ta.rsi(length=H1_RSI_LEN)
    df_h1_alt['ADX_14_H1'] = df_h1_alt.ta.adx(length=H1_ADX_LEN)['ADX_14']
    if ENABLE_GARCH:
        df_h1_alt['log_return'] = np.log(df_h1_alt['close'] / df_h1_alt['close'].shift(1))
        if len(df_h1_alt) >= GARCH_ROLL_WINDOW:
            df_h1_alt['garch_volatility'] = df_h1_alt['log_return'].rolling(window=GARCH_ROLL_WINDOW).apply(garch_volatility_robust, raw=False)
        else:
            df_h1_alt['garch_volatility'] = np.nan
    else:
        df_h1_alt['garch_volatility'] = np.nan
    df_h1_src = df_h1_alt
    print("  Using H1 features recomputed from M15 resample.")
else:
    df_h1_src = df_h1

h1_aligned = df_h1_src[cols_h1].reindex(df_m15.index).ffill().bfill()
h4_aligned = df_h4[cols_h4].reindex(df_m15.index).ffill().bfill()
df_m15 = df_m15.join(h1_aligned).join(h4_aligned)
# quick diagnostic
if VERBOSE:
    dbg_nan_h1 = h1_aligned.isna().sum().to_dict()
    dbg_nan_h4 = h4_aligned.isna().sum().to_dict()
    diag = {**{f"H1_{k}": int(v) for k, v in dbg_nan_h1.items()}, **{f"H4_{k}": int(v) for k, v in dbg_nan_h4.items()}}
    print(diag)

# Session mask (UTC) and interaction feature
ny_hours_mask = df_m15['hour_of_day'].between(NY_HOURS_START, NY_HOURS_END).astype(int)
df_m15['RSI_15_x_NYHours'] = df_m15['RSI_14_M15'] * ny_hours_mask

print(f"[3/4] Labeling: hybrid-5c, lookahead={LOOKAHEAD_CANDLES}, SL={SL_FILTER_PCT*100:.2f}%")
labels_series = hybrid_labels_5c(
    df_m15[['open','high','low','close','volume']],
    lookahead=LOOKAHEAD_CANDLES,
    sl_filter_pct=SL_FILTER_PCT,
)
df_m15['target'] = labels_series
SL_NEUTRALIZED_COUNT = int(labels_series.attrs.get('sl_neutralized', 0))

if VERBOSE:
    print("Feature engineering complete.")

# Normalization and saving
print("[4/4] Scale & save...")

# Define the final feature set for the M15 model
feature_columns_m15 = [
    # M15 core
    'RSI_14_M15', 'ATR_5_pct_M15', 'dist_from_ema_10_M15', 'ema10_slope_M15',
    'volume_oscillator_M15', 'obv_M15',
    'hour_sin', 'hour_cos',
    # Higher TF
    'RSI_14_H1', 'ADX_14_H1', 'EMA_diff_21_H4pct',
    # Interaction
    'RSI_15_x_NYHours',
]

if ENABLE_GARCH:
    feature_columns_m15.append('garch_volatility')

# Prepare final dataset
if len(df_m15) < MIN_REQUIRED_BARS:
    print(f"Error: not enough M15 bars (have {len(df_m15)}, need >= {MIN_REQUIRED_BARS}).")
    exit()

X = df_m15[feature_columns_m15].copy()
y = df_m15['target'].copy()

pre_rows = len(X)
valid_mask = X.notna().all(axis=1) & y.notna()
X = X[valid_mask]
y = y[valid_mask]
post_rows = len(X)
if VERBOSE:
    print(f"Rows before/after NaN filtering: {pre_rows} -> {post_rows}")

if post_rows == 0:
    print("Error: no valid samples after filtering. Check feature availability and NaN handling.")
    missing_counts = df_m15[feature_columns_m15 + ['target']].isna().sum().to_dict()
    print({k:int(v) for k,v in missing_counts.items()})
    exit()

# Normalize features to [-1, 1]
scaler = MinMaxScaler(feature_range=SCALER_FEATURE_RANGE)
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

# Combine normalized features with the target for a complete dataset
final_dataset_m15 = X_scaled_df.join(y)

# Save the new dataset and scaler
joblib.dump(scaler, SCALER_PATH)
parquet_filename = PARQUET_PATH
csv_filename = CSV_PATH
final_dataset_m15.round(ROUND_DECIMALS).to_parquet(parquet_filename)
final_dataset_m15.round(ROUND_DECIMALS).to_csv(csv_filename)

end_time = time.time()
print("\nComplete.")
print(f"Saved: {parquet_filename} | {csv_filename} | Scaler: {SCALER_PATH}")
print(f"Rows: {len(final_dataset_m15)} | Features: {len(feature_columns_m15)} + target | Time: {end_time - start_time:.2f}s")

# Data span summary (raw bars vs final samples)
raw_m15 = len(df_m15)
_days = (pd.to_datetime(TRAIN_END) - pd.to_datetime(TRAIN_START)).days + 1
_expected = _days * 96
print(f"Span: {pd.to_datetime(TRAIN_START).date()}..{pd.to_datetime(TRAIN_END).date()} | Expected M15: {_expected} | Raw M15: {raw_m15} | Samples: {len(X)}")

# Label distribution summary
label_counts = final_dataset_m15['target'].value_counts().sort_index()
strong_sell = int(label_counts.get(0, 0))
sell = int(label_counts.get(1, 0))
neutral = int(label_counts.get(2, 0))
buy = int(label_counts.get(3, 0))
strong_buy = int(label_counts.get(4, 0))
total = strong_sell + sell + neutral + buy + strong_buy

def pct(x):
    return (x / total * 100.0) if total > 0 else 0.0

print("\nLabel summary:")
print(f"  4 strong_buy  : {strong_buy} ({pct(strong_buy):.2f}%)")
print(f"  3 buy         : {buy} ({pct(buy):.2f}%)")
print(f"  2 neutral     : {neutral} ({pct(neutral):.2f}%)")
print(f"  1 sell        : {sell} ({pct(sell):.2f}%)")
print(f"  0 strong_sell : {strong_sell} ({pct(strong_sell):.2f}%)")
print(f"  Totals        : {total} | Buys: {buy + strong_buy} | Sells: {sell + strong_sell} | Neutral: {neutral}")
try:
    print(f"SL-neutralized windows: {SL_NEUTRALIZED_COUNT}")
except NameError:
    pass
