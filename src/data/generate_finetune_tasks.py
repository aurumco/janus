
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import glob

def generate_finetune_tasks(input_path: str, raw_data_dir: str, output_dir: str, sequence_length: int = 72):
    """
    Generates three finetune datasets for different tasks.

    Args:
        input_path: Path to the processed features parquet (Janus Pretrain dataset).
        raw_data_dir: Directory containing raw OHLC pipe-separated CSVs.
        output_dir: Directory to save the task datasets.
        sequence_length: Sequence length for windowing (used for validation, not slicing here).
    """
    print(f"Loading features from {input_path}...")
    try:
        df_features = pd.read_parquet(input_path)
    except Exception as e:
        print(f"Error loading features parquet: {e}")
        return

    # Normalize feature columns
    df_features.columns = [c.lower() for c in df_features.columns]

    # Ensure timestamp exists
    if 'timestamp' not in df_features.columns:
        # Check if index is timestamp
        if isinstance(df_features.index, pd.DatetimeIndex):
            df_features['timestamp'] = df_features.index.astype(np.int64) // 10**6 # ms
        else:
            print("Warning: 'timestamp' column missing in features. Attempting to use index.")
            df_features = df_features.reset_index()
            if 'timestamp' not in df_features.columns:
                 # Try 'date' or 'time'
                 for t_col in ['date', 'time', 'datetime']:
                     if t_col in df_features.columns:
                         df_features.rename(columns={t_col: 'timestamp'}, inplace=True)
                         break

    # Convert feature timestamp to ms int64 if needed
    if pd.api.types.is_datetime64_any_dtype(df_features['timestamp']):
        df_features['timestamp'] = df_features['timestamp'].astype(np.int64) // 10**6
    elif pd.api.types.is_integer_dtype(df_features['timestamp']):
        # Normalize to ms if it looks like seconds
        if df_features['timestamp'].max() < 30000000000:
            df_features['timestamp'] = df_features['timestamp'] * 1000

    print(f"Features loaded: {df_features.shape}")

    # Load Raw OHLC
    print(f"Loading raw OHLC data from {raw_data_dir}...")
    raw_files = glob.glob(os.path.join(raw_data_dir, "*.csv"))
    if not raw_files:
        print(f"No CSV files found in {raw_data_dir}. Looking for files without extension or recursive?")
        # Sometimes Kaggle datasets are just files.
        # Check specific pipe separated pattern
        # Prompt says: pipe-separated CSVs with NO HEADER.
        pass

    # Columns: timestamp|open|high|low|close|volume|quote_vol|trades_count|taker_base_vol|taker_quote_vol
    raw_cols = [
        'timestamp', 'open', 'high', 'low', 'close',
        'volume', 'quote_vol', 'trades_count', 'taker_base_vol', 'taker_quote_vol'
    ]

    df_raw_list = []

    # If no glob match, maybe raw_data_dir IS the file?
    if os.path.isfile(raw_data_dir):
        raw_files = [raw_data_dir]

    for f in raw_files:
        try:
            # Determine separation based on extension or content
            # Prompt says PIPE-SEPARATED
            temp_df = pd.read_csv(f, sep='|', header=None, names=raw_cols)
            df_raw_list.append(temp_df)
        except Exception as e:
            print(f"Failed to read {f}: {e}")

    if not df_raw_list:
        print("Error: No raw data loaded.")
        return

    df_raw = pd.concat(df_raw_list, ignore_index=True)

    # Normalize timestamps to ms
    # Check if timestamp is seconds or ms. Usually crypto data is ms.
    # If max timestamp is small (< 3e10), it's seconds.
    if df_raw['timestamp'].max() < 30000000000:
        df_raw['timestamp'] = df_raw['timestamp'] * 1000

    df_raw['timestamp'] = df_raw['timestamp'].astype(np.int64)

    # Sort and Deduplicate
    df_raw = df_raw.sort_values('timestamp').drop_duplicates('timestamp')
    print(f"Raw data loaded: {df_raw.shape}")

    # Merge
    # We want features + raw prices (Close, High, Low) for target generation
    # Inner join on timestamp
    print("Merging features and raw data...")

    # Fix ambiguity if index has same name as merge column
    if df_features.index.name == 'timestamp':
        df_features.index.name = None
    if df_raw.index.name == 'timestamp':
        df_raw.index.name = None

    df_merged = pd.merge(df_features, df_raw[['timestamp', 'close', 'high', 'low']], on='timestamp', how='inner')
    print(f"Merged data: {df_merged.shape}")

    if df_merged.empty:
        print("Error: Merged dataframe is empty. Timestamp mismatch?")
        print(f"Feature timestamps sample: {df_features['timestamp'].head()}")
        print(f"Raw timestamps sample: {df_raw['timestamp'].head()}")
        return

    # Ensure asset_id exists
    if 'asset_id' not in df_merged.columns:
        print("Warning: asset_id missing. Assigning default 0.")
        df_merged['asset_id'] = 0

    # Sort by timestamp
    df_merged = df_merged.sort_values('timestamp')

    # Generate Targets

    # Task 1: Direction (Binary)
    # Target: 1 if Close[t+1] > Close[t]
    print("Generating Task 1: Directional Classification...")
    df_task1 = df_merged.copy()
    df_task1['target_direction'] = (df_task1['close'].shift(-1) > df_task1['close']).astype(int)

    # Task 2: Volatility (Dynamic TP/SL)
    # Target: [Max_Return_Next_12, Min_Return_Next_12]
    print("Generating Task 2: Dynamic TP/SL...")
    df_task2 = df_merged.copy()
    window = 12
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window)

    high_forward = df_task2['high'].shift(-1)
    low_forward = df_task2['low'].shift(-1)

    max_high = high_forward.rolling(window=indexer, min_periods=window).max()
    min_low = low_forward.rolling(window=indexer, min_periods=window).min()

    df_task2['target_max_return_12'] = (max_high - df_task2['close']) / df_task2['close']
    df_task2['target_min_return_12'] = (min_low - df_task2['close']) / df_task2['close']

    # Task 3: Price Forecasting (Log Return)
    print("Generating Task 3: Price Forecasting...")
    df_task3 = df_merged.copy()
    df_task3['target_log_return'] = np.log(df_task3['close'].shift(-1) / df_task3['close'])

    # Clean NaNs
    df_task1 = df_task1.dropna()
    df_task2 = df_task2.dropna()
    df_task3 = df_task3.dropna()

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving tasks to {output_dir_path}...")
    df_task1.to_parquet(output_dir_path / "finetune_task1_direction.parquet")
    df_task2.to_parquet(output_dir_path / "finetune_task2_volatility.parquet")
    df_task3.to_parquet(output_dir_path / "finetune_task3_price.parquet")

    print("Task generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Defaults set for typical Kaggle paths based on prompt, but overrideable
    parser.add_argument("--input", type=str,
                        default="/kaggle/input/janusds/janus_pretrain_5min/janus_pretrain_5min_dataset.parquet",
                        help="Path to feature parquet")
    parser.add_argument("--raw", type=str,
                        default="/kaggle/input/janusai/",
                        help="Directory containing raw OHLC pipe-separated CSVs")
    parser.add_argument("--output", type=str,
                        default="/kaggle/working/tasks",
                        help="Output directory")

    args = parser.parse_args()

    generate_finetune_tasks(args.input, args.raw, args.output)
