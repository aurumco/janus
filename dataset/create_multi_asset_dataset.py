"""Multi-asset dataset builder with SSL pre-training support."""

import sys
import time
import zipfile
from pathlib import Path
from typing import List, Tuple, Optional
import select

import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

try:
    from .config import DatasetConfig
    from .data_processor import MultiTimeframeProcessor
    from .labeling import PriceLabelingStrategy
except ImportError:
    from config import DatasetConfig
    from data_processor import MultiTimeframeProcessor
    from labeling import PriceLabelingStrategy


class MultiAssetDatasetBuilder:
    """Builds multi-asset dataset with SSL pre-training support."""

    def __init__(self, config: DatasetConfig, verbose: bool = True) -> None:
        """Initialize multi-asset dataset builder.

        Args:
            config: Dataset configuration.
            verbose: Whether to print progress messages.
        """
        self.config = config
        self.verbose = verbose
        self.processor = MultiTimeframeProcessor(config)
        self.labeler = PriceLabelingStrategy(
            lookahead=config.target_window_candles,
            sl_filter_pct=config.sl_filter_pct,
            use_atr_stop=config.use_atr_stop_loss,
            atr_multiplier=config.atr_multiplier,
            atr_period=config.atr_period,
        )

    def _log(self, message: str) -> None:
        """Print message if verbose mode is enabled.

        Args:
            message: Message to print.
        """
        if self.verbose:
            print(message)

    def _detect_environment(self) -> Tuple[bool, List[Path]]:
        """Detect Kaggle environment and search paths.

        Returns:
            Tuple of (is_kaggle, search_paths).
        """
        kaggle_input = Path("/kaggle/input")
        kaggle_working = Path("/kaggle/working")
        
        search_paths = []
        
        if kaggle_input.exists():
            search_paths.append(kaggle_input)
            self._log("✓ Detected Kaggle environment")
        
        if kaggle_working.exists():
            search_paths.append(kaggle_working)
        
        local_dataset = Path(self.config.dataset_dir)
        if local_dataset.exists():
            search_paths.append(local_dataset)
        
        is_kaggle = len(search_paths) > 0 and kaggle_input in search_paths
        
        if not search_paths:
            search_paths = [local_dataset]
        
        return is_kaggle, search_paths

    def _extract_zip_files(self, search_paths: List[Path]) -> None:
        """Extract any janus_dataset.zip files found.

        Args:
            search_paths: Paths to search for zip files.
        """
        for search_path in search_paths:
            zip_files = list(search_path.glob("**/janus_dataset.zip"))
            
            for zip_file in zip_files:
                extract_to = zip_file.parent / "extracted"
                
                if extract_to.exists():
                    self._log(f"  Already extracted: {zip_file.name}")
                    continue
                
                self._log(f"  Extracting: {zip_file}")
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
                self._log(f"    ✓ Extracted to: {extract_to}")

    def _detect_gpu(self) -> Tuple[bool, str]:
        """Detect available GPU.

        Returns:
            Tuple of (has_gpu, gpu_name).
        """
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_count = torch.cuda.device_count()
                return True, f"{gpu_name} (x{gpu_count})"
        except ImportError:
            pass
        
        return False, "CPU"

    def discover_assets(self) -> List[Tuple[str, Path]]:
        """Discover available asset CSV files.

        Returns:
            List of (symbol, path) tuples.
        """
        is_kaggle, search_paths = self._detect_environment()
        
        if is_kaggle:
            self._log("\nSearching in Kaggle paths:")
            for path in search_paths:
                self._log(f"  • {path}")
            
            self._extract_zip_files(search_paths)
        
        assets = []
        seen_symbols = set()
        
        for search_path in search_paths:
            if not search_path.exists():
                self._log(f"  Warning: Path does not exist: {search_path}")
                continue
            
            csv_files = list(search_path.glob("**/*USDT.csv"))
            
            if not csv_files:
                csv_files = list(search_path.glob("**/*.csv"))
                self._log(f"  No *USDT.csv files found in {search_path}, trying all CSV files...")
            
            for csv_file in sorted(csv_files):
                symbol = csv_file.stem
                
                if 'USDT' not in symbol.upper():
                    continue
                
                if symbol not in seen_symbols:
                    assets.append((symbol, csv_file))
                    seen_symbols.add(symbol)
        
        if not assets:
            self._log("\n⚠️  Warning: No asset CSV files found!")
            self._log("Searched in:")
            for path in search_paths:
                self._log(f"  • {path}")
            self._log("\nPlease ensure CSV files are named like: BTCUSDT.csv, ETHUSDT.csv, etc.")
        
        return assets

    def load_and_resample_asset(
        self,
        csv_path: Path,
        symbol: str,
        asset_id: int,
        timeframe: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Load and resample a single asset.

        Args:
            csv_path: Path to CSV file.
            symbol: Asset symbol.
            asset_id: Numeric asset ID.
            timeframe: Target timeframe (e.g., "30min" or "1min").
            start_date: Start date filter.
            end_date: End date filter.

        Returns:
            Resampled DataFrame with asset_id column.
        """
        def _read_with_sep(sep: Optional[str]) -> pd.DataFrame:
            return pd.read_csv(csv_path, header=None, sep=sep, engine='python')

        try_order = ['|', ',', ';', None]
        df = None
        for sep in try_order:
            try:
                df = _read_with_sep(sep)
                break
            except Exception:
                df = None
        if df is None:
            try:
                df = pd.read_csv(csv_path, header=0, sep=None, engine='python')
            except Exception:
                df = pd.read_csv(csv_path)

        if df.shape[1] == 1 and isinstance(df.iloc[0, 0], str):
            raw = df.iloc[:, 0]
            if '|' in raw.iloc[0]:
                df = raw.str.split('|', expand=True)
            elif ',' in raw.iloc[0]:
                df = raw.str.split(',', expand=True)
            elif ';' in raw.iloc[0]:
                df = raw.str.split(';', expand=True)

        cols_lower = [str(c).strip().lower() for c in df.columns]

        if len(cols_lower) >= 6 and all(c.isdigit() for c in cols_lower):
            df.columns = list(range(df.shape[1]))
            df = df[[0, 1, 2, 3, 4, 5]]
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        else:
            alias_map = {
                'timestamp': ['timestamp', 'time', 'open_time', 'opentime', 'date'],
                'open': ['open', 'o'],
                'high': ['high', 'h'],
                'low': ['low', 'l'],
                'close': ['close', 'c'],
                'volume': ['volume', 'v', 'vol']
            }
            name_map = {}
            df_lower = df.copy()
            df_lower.columns = cols_lower
            for target, aliases in alias_map.items():
                for a in aliases:
                    if a in df_lower.columns:
                        name_map[target] = a
                        break
            required = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
            if required.issubset(name_map.keys()):
                df = df_lower[[name_map['timestamp'], name_map['open'], name_map['high'], name_map['low'], name_map['close'], name_map['volume']]].copy()
                df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            elif df.shape[1] >= 6:
                positional = df.iloc[:, :6].copy()
                positional.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                df = positional
            else:
                raise ValueError('Unable to parse OHLCV columns from CSV')

        timestamps = pd.to_numeric(df['timestamp'], errors='coerce')
        unit = 'ms' if timestamps.max() > 1e12 else 's'
        df['timestamp'] = pd.to_datetime(timestamps, unit=unit, errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
        df.set_index('timestamp', inplace=True)

        # Filter by date range
        df = df[(df.index >= start_date) & (df.index <= end_date)]

        if df.empty:
            self._log(f"  Warning: No data for {symbol} in date range")
            return pd.DataFrame()

        # Resample to target timeframe
        ohlcv_logic = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }

        df_resampled = df.resample(timeframe).apply(ohlcv_logic).dropna()
        
        # Add asset_id column
        df_resampled['asset_id'] = asset_id
        df_resampled['symbol'] = symbol
        
        return df_resampled

    def build_multi_asset(
        self,
        mode: str = "fine-tune"
    ) -> Tuple[pd.DataFrame, MinMaxScaler]:
        """Build multi-asset dataset.

        Args:
            mode: Dataset mode ("pre-train", "fine-tune", or "both").

        Returns:
            Tuple of (dataset, scaler).
        """
        start_time = time.time()

        self._log("="*70)
        self._log("JANUS MULTI-ASSET DATASET BUILDER")
        self._log("="*70)
        self._log(f"Mode: {mode.upper()}")
        
        # Discover assets
        assets = self.discover_assets()
        
        if not assets:
            self._log("\n❌ Error: No valid asset data found!")
            self._log("\nTroubleshooting:")
            self._log("1. Ensure CSV files exist in the dataset directory")
            self._log("2. Files should be named like: BTCUSDT.csv, ETHUSDT.csv")
            self._log("3. For Kaggle: Check /kaggle/input/ and /kaggle/working/")
            self._log("4. Use --dataset-dir flag to specify custom directory")
            raise ValueError("No asset CSV files found. Please check file locations.")
        
        self._log(f"\nDiscovered {len(assets)} assets:")
        for symbol, _ in assets:
            self._log(f"  • {symbol}")

        # Configure based on mode
        if mode == "pre-train":
            timeframe = self.config.pretrain_timeframe
            start_date = self.config.pretrain_start
            end_date = self.config.pretrain_end
            self._log(f"\nPre-training mode:")
            self._log(f"  Timeframe: {timeframe}")
            self._log(f"  Date range: {start_date} to {end_date}")
        else:  # fine-tune or both
            timeframe = self.config.finetune_timeframe
            start_date = self.config.finetune_start
            end_date = self.config.finetune_end
            self._log(f"\nFine-tuning mode:")
            self._log(f"  Timeframe: {timeframe}")
            self._log(f"  Date range: {start_date} to {end_date}")

        # Load and resample each asset
        self._log(f"\n[1/5] Loading and resampling assets...")
        asset_dfs = []
        
        skipped = 0
        for idx, (symbol, csv_path) in enumerate(assets):
            self._log(f"  Processing {symbol} ({idx+1}/{len(assets)})...")
            try:
                df_asset = self.load_and_resample_asset(
                    csv_path, symbol, idx, timeframe, start_date, end_date
                )
                if not df_asset.empty:
                    asset_dfs.append(df_asset)
                    self._log(f"    ✓ {len(df_asset):,} candles")
                else:
                    skipped += 1
                    self._log("    Skipped: empty after filtering/resample")
            except Exception as e:
                skipped += 1
                self._log(f"    Skipped {symbol} due to parse error: {e}")

        if not asset_dfs:
            raise ValueError("No valid asset data found after parsing. Check CSV formats/paths.")

        # Concatenate all assets
        self._log(f"\n[2/5] Concatenating {len(asset_dfs)} assets...")
        df_combined = pd.concat(asset_dfs, axis=0)
        
        # Sort by timestamp
        self._log(f"[3/5] Sorting by timestamp...")
        df_combined.sort_index(inplace=True)
        
        self._log(f"  Total samples: {len(df_combined):,}")
        self._log(f"  Date range: {df_combined.index.min()} to {df_combined.index.max()}")

        # Calculate features for fine-tune mode
        if mode == "fine-tune":
            df_final = self._calculate_features_and_labels(df_combined)
        else:  # pre-train mode - minimal features
            df_final = self._calculate_pretrain_features(df_combined)

        # Scale features
        self._log(f"\n[5/5] Scaling features...")
        feature_cols = [col for col in df_final.columns if col not in ['target', 'asset_id', 'symbol']]
        
        scaler = MinMaxScaler(feature_range=self.config.scaler_feature_range)
        df_final[feature_cols] = scaler.fit_transform(df_final[feature_cols])

        elapsed_time = time.time() - start_time
        self._print_summary(df_final, feature_cols, elapsed_time)

        return df_final, scaler

    def _calculate_features_and_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features and labels for fine-tuning.

        Args:
            df: Combined OHLCV DataFrame with asset_id.

        Returns:
            DataFrame with features and labels.
        """
        self._log(f"[4/5] Calculating features and labels...")
        
        all_features = []
        
        # Process each asset separately
        for asset_id in df['asset_id'].unique():
            asset_mask = df['asset_id'] == asset_id
            df_asset = df[asset_mask].copy()
            
            symbol = df_asset['symbol'].iloc[0]
            self._log(f"  Processing features for {symbol}...")
            
            # Calculate indicators (using 30min as base)
            df_asset = self._calculate_technical_indicators(df_asset)
            
            # Calculate labels
            labels = self.labeler.label_regression(
                df_asset[['open', 'high', 'low', 'close', 'volume']]
            )
            df_asset['target'] = labels
            
            # Keep asset_id
            all_features.append(df_asset)
        
        # Recombine and sort
        df_final = pd.concat(all_features, axis=0).sort_index()
        
        # Drop NaN rows
        feature_cols = [col for col in df_final.columns if col not in ['target', 'asset_id', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
        valid_mask = df_final[feature_cols + ['target']].notna().all(axis=1)
        df_final = df_final[valid_mask]
        
        # Keep only features, target, and asset_id
        keep_cols = feature_cols + ['target', 'asset_id']
        df_final = df_final[keep_cols]
        
        return df_final

    def _calculate_pretrain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate minimal features for pre-training (SSL).

        Args:
            df: Combined OHLCV DataFrame with asset_id.

        Returns:
            DataFrame with basic features (no labels).
        """
        self._log(f"[4/5] Calculating pre-training features...")
        
        all_features = []
        
        for asset_id in df['asset_id'].unique():
            asset_mask = df['asset_id'] == asset_id
            df_asset = df[asset_mask].copy()
            
            symbol = df_asset['symbol'].iloc[0]
            self._log(f"  Processing {symbol}...")
            
            # Basic price features
            df_asset['returns'] = df_asset['close'].pct_change()
            df_asset['log_returns'] = np.log(df_asset['close'] / df_asset['close'].shift(1))
            df_asset['high_low_ratio'] = df_asset['high'] / df_asset['low']
            df_asset['close_open_ratio'] = df_asset['close'] / df_asset['open']
            
            # Volume features
            df_asset['volume_change'] = df_asset['volume'].pct_change()
            df_asset['volume_ma_ratio'] = df_asset['volume'] / df_asset['volume'].rolling(20).mean()
            
            # Time features
            df_asset['hour_sin'] = np.sin(2 * np.pi * df_asset.index.hour / 24)
            df_asset['hour_cos'] = np.cos(2 * np.pi * df_asset.index.hour / 24)
            df_asset['day_of_week_sin'] = np.sin(2 * np.pi * df_asset.index.dayofweek / 7)
            df_asset['day_of_week_cos'] = np.cos(2 * np.pi * df_asset.index.dayofweek / 7)
            
            all_features.append(df_asset)
        
        df_final = pd.concat(all_features, axis=0).sort_index()
        
        # Drop OHLCV and symbol, keep only features and asset_id
        feature_cols = ['returns', 'log_returns', 'high_low_ratio', 'close_open_ratio',
                       'volume_change', 'volume_ma_ratio',
                       'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
                       'asset_id']
        
        df_final = df_final[feature_cols].dropna()
        
        return df_final

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for a single asset.

        Args:
            df: OHLCV DataFrame.

        Returns:
            DataFrame with calculated indicators.
        """
        from indicators import IndicatorCalculator
        
        calc = IndicatorCalculator()
        
        # Primary timeframe indicators (30min)
        df['RSI_14_M15'] = calc.calculate_rsi(df['close'], 14)
        atr = calc.calculate_atr(df['high'], df['low'], df['close'], 5)
        df['ATR_5_pct_M15'] = atr / df['close']
        
        ema10 = calc.calculate_ema(df['close'], 10)
        df['dist_from_ema_10_M15'] = (df['close'] - ema10) / df['close']
        df['ema10_slope_M15'] = ema10.diff(1)
        
        df['volume_oscillator_M15'] = calc.calculate_pvo(df['volume'], 12, 26, 9)
        df['obv_M15'] = calc.calculate_obv(df['close'], df['volume'])
        
        # Time features
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        
        # Higher timeframe features (resample internally)
        df_h1 = df[['open', 'high', 'low', 'close', 'volume']].resample('1h').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        
        df_h1['RSI_14_H1'] = calc.calculate_rsi(df_h1['close'], 14)
        df_h1['ADX_14_H1'] = calc.calculate_adx(df_h1['high'], df_h1['low'], df_h1['close'], 14)
        
        df_h4 = df[['open', 'high', 'low', 'close', 'volume']].resample('4h').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        
        ema21_h4 = calc.calculate_ema(df_h4['close'], 21)
        df_h4['EMA_diff_21_H4pct'] = (df_h4['close'] - ema21_h4) / df_h4['close']
        
        # Merge higher timeframes
        df = df.join(df_h1[['RSI_14_H1', 'ADX_14_H1']]).ffill().bfill()
        df = df.join(df_h4[['EMA_diff_21_H4pct']]).ffill().bfill()
        
        # NY hours interaction
        ny_hours_mask = ((df.index.hour >= 13) & (df.index.hour <= 17)).astype(int)
        df['RSI_15_x_NYHours'] = df['RSI_14_M15'] * ny_hours_mask
        
        # GARCH volatility (optional, computationally expensive)
        if self.config.enable_garch:
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            df['garch_volatility'] = df['log_return'].rolling(window=240).apply(
                calc.calculate_garch_volatility, raw=False
            )
            df['garch_volatility'] = df['garch_volatility'].ffill().bfill()
        
        return df

    def _print_summary(
        self,
        dataset: pd.DataFrame,
        features: list,
        elapsed_time: float
    ) -> None:
        """Print dataset creation summary.

        Args:
            dataset: Final dataset.
            features: List of feature names.
            elapsed_time: Time taken to build dataset.
        """
        self._log("\n" + "="*70)
        self._log("DATASET SUMMARY")
        self._log("="*70)
        self._log(f"Total samples: {len(dataset):,}")
        self._log(f"Features: {len(features)}")
        self._log(f"Assets: {dataset['asset_id'].nunique()}")
        self._log(f"Time elapsed: {elapsed_time:.2f}s")

        if 'target' in dataset.columns:
            target_values = dataset['target']
            self._log("\nTarget Distribution (Price Change %):")
            self._log(f"  Mean: {target_values.mean():.4f} ({target_values.mean()*100:.2f}%)")
            self._log(f"  Std:  {target_values.std():.4f} ({target_values.std()*100:.2f}%)")
            self._log(f"  Min:  {target_values.min():.4f} ({target_values.min()*100:.2f}%)")
            self._log(f"  Max:  {target_values.max():.4f} ({target_values.max()*100:.2f}%)")
            
            positive_samples = (target_values > 0).sum()
            negative_samples = (target_values < 0).sum()
            total = len(target_values)
            
            self._log(f"\n  Positive: {positive_samples:6,} ({100*positive_samples/total:5.2f}%)")
            self._log(f"  Negative: {negative_samples:6,} ({100*negative_samples/total:5.2f}%)")

        self._log("="*70 + "\n")

    def save(self, dataset: pd.DataFrame, scaler: MinMaxScaler, mode: str) -> None:
        """Save dataset and scaler to disk.

        Args:
            dataset: Dataset to save.
            scaler: Fitted scaler to save.
            mode: Dataset mode for filename.
        """
        # Update config paths based on mode
        self.config.mode = mode
        self.config._configure_paths()
        
        saved_files = []
        
        if self.config.save_parquet:
            dataset.round(self.config.round_decimals).to_parquet(self.config.parquet_path)
            saved_files.append(self.config.parquet_path)
            self._log(f"Saved: {self.config.parquet_path}")
        
        if self.config.save_csv:
            dataset.round(self.config.round_decimals).to_csv(self.config.csv_path)
            saved_files.append(self.config.csv_path)
            self._log(f"Saved: {self.config.csv_path}")
        
        joblib.dump(scaler, self.config.scaler_path)
        self._log(f"Saved: {self.config.scaler_path}")


def get_user_choice(prompt: str, options: List[str], default: str, timeout: int = 10) -> str:
    """Get user choice with timeout.

    Args:
        prompt: Prompt message.
        options: List of valid options.
        default: Default choice if timeout.
        timeout: Timeout in seconds.

    Returns:
        Selected option.
    """
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    print(f"\nDefault: {default} (timeout in {timeout}s)")
    print("Choice: ", end='', flush=True)
    
    # Use select for timeout on Unix systems
    if sys.platform != 'win32':
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            choice = sys.stdin.readline().strip()
        else:
            print(f"\n⏱ Timeout! Using default: {default}")
            return default
    else:
        # Windows fallback (no timeout)
        try:
            choice = input()
        except:
            choice = ""
    
    # Parse choice
    if not choice:
        return default
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(options):
            return options[idx]
    except ValueError:
        pass
    
    # Check if choice matches an option
    choice_lower = choice.lower()
    for opt in options:
        if opt.lower().startswith(choice_lower):
            return opt
    
    print(f"Invalid choice. Using default: {default}")
    return default


def main() -> None:
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Janus Multi-Asset Dataset Builder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_multi_asset_dataset.py --mode fine-tune --format parquet
  python create_multi_asset_dataset.py --mode both --format csv
  python create_multi_asset_dataset.py  # Interactive mode
        """
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['pre-train', 'fine-tune', 'both'],
        help='Dataset mode (pre-train, fine-tune, or both). If not specified, interactive mode.'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['parquet', 'csv', 'both'],
        help='Output format (parquet, csv, or both). If not specified, interactive mode.'
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='dataset',
        help='Directory containing CSV files (default: dataset)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(" "*20 + "JANUS MULTI-ASSET DATASET BUILDER")
    print("="*70)
    
    if args.mode:
        mode = args.mode
        print(f"\nMode: {mode} (from CLI)")
    else:
        mode_options = ["pre-train", "fine-tune", "both"]
        mode = get_user_choice(
            "Select dataset mode:",
            mode_options,
            default="both",
            timeout=10
        )
    
    if args.format:
        format_choice = args.format
        print(f"Format: {format_choice} (from CLI)")
    else:
        format_options = ["parquet", "csv", "both"]
        format_choice = get_user_choice(
            "Select output format:",
            format_options,
            default="parquet",
            timeout=10
        )
    
    # Configure format settings
    save_parquet = format_choice in ["parquet", "both"]
    save_csv = format_choice in ["csv", "both"]
    
    # Build dataset(s)
    config = DatasetConfig()
    config.dataset_dir = args.dataset_dir
    config.save_parquet = save_parquet
    config.save_csv = save_csv
    
    builder = MultiAssetDatasetBuilder(config, verbose=True)
    
    try:
        if mode == "both":
            # Build pre-train dataset
            print("\n" + "="*70)
            print("BUILDING PRE-TRAIN DATASET")
            print("="*70)
            dataset_pretrain, scaler_pretrain = builder.build_multi_asset(mode="pre-train")
            builder.save(dataset_pretrain, scaler_pretrain, mode="pre-train")
            
            # Build fine-tune dataset
            print("\n" + "="*70)
            print("BUILDING FINE-TUNE DATASET")
            print("="*70)
            dataset_finetune, scaler_finetune = builder.build_multi_asset(mode="fine-tune")
            builder.save(dataset_finetune, scaler_finetune, mode="fine-tune")
        else:
            dataset, scaler = builder.build_multi_asset(mode=mode)
            builder.save(dataset, scaler, mode=mode)
        
        print("\n✅ Dataset creation completed successfully.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
