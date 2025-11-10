"""Step 1: Data Health & Sanity Check.

This module performs comprehensive data quality checks including:
- Missing values analysis
- Descriptive statistics
- Distribution visualization
- Skewness detection and transformation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple


class DataHealthAnalyzer:
    """Analyzer for data quality and health checks."""
    
    def __init__(self, df: pd.DataFrame, output_dir: str = "analysis/reports"):
        """Initialize analyzer.
        
        Args:
            df: Input DataFrame
            output_dir: Directory for saving reports
        """
        self.df = df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.viz_dir = Path("analysis/visualizations")
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        self.report = []
    
    def log(self, message: str):
        """Add message to report."""
        print(message)
        self.report.append(message)
    
    def analyze_missing_values(self) -> pd.Series:
        """Analyze missing values in dataset.
        
        Returns:
            Series with percentage of missing values per column
        """
        self.log("\n" + "="*80)
        self.log("STEP 1.1: MISSING VALUES ANALYSIS")
        self.log("="*80)
        
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Missing_Count': missing,
            'Missing_Percentage': missing_pct
        }).sort_values('Missing_Percentage', ascending=False)
        
        self.log(f"\nTotal rows: {len(self.df):,}")
        self.log(f"\nColumns with missing values:")
        self.log(missing_df[missing_df['Missing_Count'] > 0].to_string())
        
        total_missing = missing.sum()
        if total_missing == 0:
            self.log("\n✓ NO MISSING VALUES DETECTED!")
        else:
            self.log(f"\n⚠ Total missing values: {total_missing:,}")
        
        # Save to file
        missing_df.to_csv(self.output_dir / "01_missing_values.csv")
        
        return missing_pct
    
    def compute_descriptive_stats(self) -> pd.DataFrame:
        """Compute and save descriptive statistics.
        
        Returns:
            DataFrame with descriptive statistics
        """
        self.log("\n" + "="*80)
        self.log("STEP 1.2: DESCRIPTIVE STATISTICS")
        self.log("="*80)
        
        stats = self.df.describe().T
        
        # Add additional statistics
        stats['skewness'] = self.df.skew()
        stats['kurtosis'] = self.df.kurtosis()
        
        self.log("\nDescriptive Statistics (first 10 columns):")
        self.log(stats.head(10).to_string())
        
        # Save to file
        stats.to_csv(self.output_dir / "02_descriptive_statistics.csv")
        
        return stats
    
    def visualize_distributions(self, columns: List[str] = None):
        """Create histogram visualizations for all features.
        
        Args:
            columns: List of columns to visualize (None = all numeric)
        """
        self.log("\n" + "="*80)
        self.log("STEP 1.3: DISTRIBUTION VISUALIZATION")
        self.log("="*80)
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        n_cols = len(columns)
        self.log(f"\nGenerating histograms for {n_cols} features...")
        
        # Create grid of histograms
        n_rows = (n_cols + 3) // 4
        fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5 * n_rows))
        axes = axes.flatten() if n_cols > 1 else [axes]
        
        for idx, col in enumerate(columns):
            if idx < len(axes):
                ax = axes[idx]
                self.df[col].hist(bins=50, ax=ax, edgecolor='black', alpha=0.7)
                ax.set_title(f'{col}\n(μ={self.df[col].mean():.2f}, σ={self.df[col].std():.2f})',
                           fontsize=10)
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                ax.grid(alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_cols, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / "01_feature_distributions.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log(f"✓ Saved: {self.viz_dir / '01_feature_distributions.png'}")
    
    def analyze_skewness(self, threshold: float = 2.0) -> Tuple[List[str], pd.DataFrame]:
        """Identify and analyze highly skewed features.
        
        Args:
            threshold: Absolute skewness threshold for flagging
            
        Returns:
            Tuple of (list of skewed feature names, DataFrame with skewness stats)
        """
        self.log("\n" + "="*80)
        self.log("STEP 1.4: SKEWNESS ANALYSIS")
        self.log("="*80)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        skewness = self.df[numeric_cols].skew().sort_values(ascending=False)
        
        skew_df = pd.DataFrame({
            'Feature': skewness.index,
            'Skewness': skewness.values,
            'Abs_Skewness': np.abs(skewness.values)
        })
        
        highly_skewed = skew_df[skew_df['Abs_Skewness'] > threshold]['Feature'].tolist()
        
        self.log(f"\nSkewness threshold: |skew| > {threshold}")
        self.log(f"Highly skewed features found: {len(highly_skewed)}")
        
        if highly_skewed:
            self.log("\nHighly Skewed Features:")
            for feat in highly_skewed:
                skew_val = skewness[feat]
                self.log(f"  • {feat}: {skew_val:.3f}")
        
        self.log("\nTop 10 Most Skewed Features:")
        self.log(skew_df.head(10).to_string(index=False))
        
        # Save
        skew_df.to_csv(self.output_dir / "03_skewness_analysis.csv", index=False)
        
        # Visualize skewness
        plt.figure(figsize=(12, 6))
        plt.barh(skew_df.head(15)['Feature'], skew_df.head(15)['Skewness'])
        plt.xlabel('Skewness')
        plt.title('Top 15 Most Skewed Features')
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.3)
        plt.axvline(x=threshold, color='red', linestyle='--', alpha=0.5, label=f'Threshold: ±{threshold}')
        plt.axvline(x=-threshold, color='red', linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.viz_dir / "02_skewness_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log(f"✓ Saved: {self.viz_dir / '02_skewness_analysis.png'}")
        
        return highly_skewed, skew_df
    
    def apply_transformations(self, skewed_features: List[str], target_column: str = None) -> pd.DataFrame:
        """Apply log transformation to skewed features.
        
        Args:
            skewed_features: List of feature names to transform
            target_column: Name of target column (will be excluded from transformation)
            
        Returns:
            DataFrame with new log-transformed columns
        """
        self.log("\n" + "="*80)
        self.log("STEP 1.5: APPLYING TRANSFORMATIONS")
        self.log("="*80)
        
        df_transformed = self.df.copy()
        transformed_cols = []
        skipped_features = []
        
        for feat in skewed_features:
            if target_column and feat == target_column:
                self.log(f"  ⚠ Skipping {feat} (target variable - must not be transformed)")
                skipped_features.append(feat)
                continue
            
            if feat not in df_transformed.columns:
                self.log(f"  ⚠ Skipping {feat} (not found in dataset - may have been dropped)")
                skipped_features.append(feat)
                continue
            
            if feat in df_transformed.columns:
                # Check for non-positive values
                min_val = df_transformed[feat].min()
                
                if min_val <= 0:
                    # Use log1p for values that might be 0 or negative
                    # Add offset if negative
                    offset = abs(min_val) + 1 if min_val < 0 else 0
                    new_col = f'log_{feat}'
                    df_transformed[new_col] = np.log1p(df_transformed[feat] + offset)
                    self.log(f"  • Created {new_col} (with offset={offset})")
                else:
                    new_col = f'log_{feat}'
                    df_transformed[new_col] = np.log(df_transformed[feat])
                    self.log(f"  • Created {new_col}")
                
                transformed_cols.append(new_col)
        
        self.log(f"\n✓ Total new features created: {len(transformed_cols)}")
        if skipped_features:
            self.log(f"⚠ Skipped {len(skipped_features)} feature(s): {skipped_features[:5]}")
        
        return df_transformed
    
    def save_report(self):
        """Save analysis report to file."""
        report_path = self.output_dir / "00_data_health_report.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(self.report))
        
        self.log(f"\n{'='*80}")
        self.log(f"✓ REPORT SAVED: {report_path}")
        self.log(f"{'='*80}")


def run_data_health_check(df: pd.DataFrame, target_column: str = None) -> pd.DataFrame:
    """Run complete data health check pipeline.
    
    Args:
        df: Input DataFrame
        target_column: Name of target column (to exclude from transformations)
        
    Returns:
        Transformed DataFrame with log features
    """
    analyzer = DataHealthAnalyzer(df)
    
    # Step 1.1: Missing values
    analyzer.analyze_missing_values()
    
    # Step 1.2: Descriptive statistics
    analyzer.compute_descriptive_stats()
    
    # Step 1.3: Distribution visualization
    analyzer.visualize_distributions()
    
    # Step 1.4: Skewness analysis
    skewed_features, _ = analyzer.analyze_skewness(threshold=2.0)
    
    # Step 1.5: Apply transformations (exclude target)
    df_transformed = analyzer.apply_transformations(skewed_features, target_column=target_column)
    
    # Save report
    analyzer.save_report()
    
    return df_transformed


if __name__ == "__main__":
    # Test with sample data
    print("Data Health Check Module - Ready for use")
