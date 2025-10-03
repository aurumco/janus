"""Main script for creating Janus Bitcoin trend dataset."""

import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .config import DatasetConfig
from .data_processor import MultiTimeframeProcessor
from .labeling import PriceLabelingStrategy


class DatasetBuilder:
    """Builds the complete Janus dataset."""

    def __init__(self, config: DatasetConfig, verbose: bool = True) -> None:
        """Initialize dataset builder.

        Args:
            config: Dataset configuration.
            verbose: Whether to print progress messages.
        """
        self.config = config
        self.verbose = verbose
        self.processor = MultiTimeframeProcessor(config)
        self.labeler = PriceLabelingStrategy(
            lookahead=config.target_window_candles,
            sl_filter_pct=config.sl_filter_pct
        )

    def _log(self, message: str) -> None:
        """Print message if verbose mode is enabled.

        Args:
            message: Message to print.
        """
        if self.verbose:
            print(message)

    def build(self, csv_path: str) -> pd.DataFrame:
        """Build complete dataset from raw CSV.

        Args:
            csv_path: Path to raw BTC CSV file.

        Returns:
            Complete dataset with features and labels.
        """
        start_time = time.time()

        self._log("="*70)
        self._log("JANUS DATASET BUILDER")
        self._log("="*70)
        self._log(f"[1/5] Loading and resampling data from {csv_path}...")

        timeframes = self.processor.load_and_resample(csv_path)

        self._log(f"[2/5] Calculating higher timeframe features...")
        df_h1 = self.processor.calculate_higher_timeframe_features(
            timeframes['h1'], timeframes['h4'], timeframes['daily']
        )

        self._log(f"[3/5] Calculating M15 features...")
        df_m15 = self.processor.calculate_m15_features(timeframes['m15'])

        self._log(f"[4/5] Merging timeframes...")
        df_m15 = self.processor.merge_timeframes(df_m15, df_h1, timeframes['h4'])

        self._log(f"[5/5] Creating labels (lookahead={self.config.target_window_candles})...")
        labels = self.labeler.label_hybrid_5class(
            df_m15[['open', 'high', 'low', 'close', 'volume']]
        )
        df_m15['target'] = labels

        feature_columns = self.processor.get_feature_columns()
        X = df_m15[feature_columns].copy()
        y = df_m15['target'].copy()

        pre_filter_rows = len(X)
        valid_mask = X.notna().all(axis=1) & y.notna()
        X = X[valid_mask]
        y = y[valid_mask]
        post_filter_rows = len(X)

        self._log(f"\nFiltered rows: {pre_filter_rows} -> {post_filter_rows}")

        if post_filter_rows == 0:
            raise ValueError("No valid samples after filtering. Check data quality.")

        scaler = MinMaxScaler(feature_range=self.config.scaler_feature_range)
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

        final_dataset = X_scaled_df.join(y)

        elapsed_time = time.time() - start_time

        self._print_summary(final_dataset, feature_columns, elapsed_time)

        return final_dataset, scaler

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
        self._log(f"Time elapsed: {elapsed_time:.2f}s")

        label_counts = dataset['target'].value_counts().sort_index()
        total = len(dataset)

        self._log("\nClass Distribution:")
        class_names = ["Strong Sell", "Sell", "Neutral", "Buy", "Strong Buy"]
        for i, name in enumerate(class_names):
            count = label_counts.get(i, 0)
            pct = (count / total * 100) if total > 0 else 0
            self._log(f"  {i} ({name:12s}): {count:6,} ({pct:5.2f}%)")

        sl_neutralized = self.labeler.sl_neutralized_count
        self._log(f"\nSL-neutralized samples: {sl_neutralized:,}")
        self._log("="*70 + "\n")

    def save(self, dataset: pd.DataFrame, scaler: MinMaxScaler) -> None:
        """Save dataset and scaler to disk.

        Args:
            dataset: Dataset to save.
            scaler: Fitted scaler to save.
        """
        dataset.round(self.config.round_decimals).to_parquet(self.config.parquet_path)
        dataset.round(self.config.round_decimals).to_csv(self.config.csv_path)
        joblib.dump(scaler, self.config.scaler_path)

        self._log(f"Saved: {self.config.parquet_path}")
        self._log(f"Saved: {self.config.csv_path}")
        self._log(f"Saved: {self.config.scaler_path}")


def main() -> None:
    """Main execution function."""
    config = DatasetConfig()
    builder = DatasetBuilder(config, verbose=True)

    dataset, scaler = builder.build('btc.csv')
    builder.save(dataset, scaler)


if __name__ == '__main__':
    main()
