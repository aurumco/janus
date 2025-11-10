"""Step 4: Lag Analysis - Finding Temporal Patterns.

Analyze whether lagged versions of features have stronger predictive power.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict


class LagAnalyzer:
    """Analyzer for lagged feature correlations."""
    
    def __init__(self, df: pd.DataFrame, target_column: str, 
                 output_dir: str = "analysis/reports"):
        """Initialize analyzer.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            output_dir: Directory for saving reports
        """
        self.df = df
        self.target_column = target_column
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.viz_dir = Path("analysis/visualizations")
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        self.report = []
    
    def log(self, message: str):
        """Add message to report."""
        print(message)
        self.report.append(message)
    
    def create_lagged_features(self, features: List[str], 
                              lag_periods: List[int]) -> pd.DataFrame:
        """Create lagged versions of features.
        
        Args:
            features: List of feature names to lag
            lag_periods: List of lag periods (in timesteps)
            
        Returns:
            DataFrame with original and lagged features
        """
        self.log("\n" + "="*80)
        self.log("STEP 4.1: CREATING LAGGED FEATURES")
        self.log("="*80)
        
        df_lagged = self.df.copy()
        created_features = []
        
        self.log(f"\nCreating lags: {lag_periods}")
        self.log(f"For {len(features)} features...")
        
        for feat in features:
            if feat not in self.df.columns:
                continue
            
            for lag in lag_periods:
                lag_name = f"{feat}_lag_{lag}"
                df_lagged[lag_name] = self.df[feat].shift(lag)
                created_features.append(lag_name)
        
        # Drop rows with NaN from shifting
        df_lagged = df_lagged.dropna()
        
        self.log(f"\n✓ Created {len(created_features)} lagged features")
        self.log(f"  Original shape: {self.df.shape}")
        self.log(f"  New shape: {df_lagged.shape}")
        
        return df_lagged, created_features
    
    def analyze_lag_correlations(self, df_lagged: pd.DataFrame, 
                                 original_features: List[str],
                                 lagged_features: List[str]) -> pd.DataFrame:
        """Analyze correlations of lagged features with target.
        
        Args:
            df_lagged: DataFrame with lagged features
            original_features: List of original feature names
            lagged_features: List of lagged feature names
            
        Returns:
            DataFrame with lag correlation analysis
        """
        self.log("\n" + "="*80)
        self.log("STEP 4.2: LAG CORRELATION ANALYSIS")
        self.log("="*80)
        
        results = []
        
        for orig_feat in original_features:
            if orig_feat not in df_lagged.columns:
                continue
            
            # Original correlation
            orig_corr = df_lagged[orig_feat].corr(df_lagged[self.target_column])
            
            # Lagged correlations
            lag_corrs = {}
            for lag_feat in lagged_features:
                if lag_feat.startswith(orig_feat + "_lag_"):
                    lag_period = int(lag_feat.split("_lag_")[1])
                    corr = df_lagged[lag_feat].corr(df_lagged[self.target_column])
                    lag_corrs[lag_period] = corr
            
            if lag_corrs:
                best_lag = max(lag_corrs.items(), key=lambda x: abs(x[1]))
                
                results.append({
                    'Feature': orig_feat,
                    'Original_Corr': orig_corr,
                    'Best_Lag_Period': best_lag[0],
                    'Best_Lag_Corr': best_lag[1],
                    'Improvement': abs(best_lag[1]) - abs(orig_corr)
                })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Improvement', ascending=False)
        
        # Report findings
        self.log(f"\n📊 LAG ANALYSIS RESULTS:")
        self.log(f"  Features analyzed: {len(results_df)}")
        
        improved = results_df[results_df['Improvement'] > 0]
        self.log(f"  Improved with lag: {len(improved)} ({100*len(improved)/len(results_df):.1f}%)")
        
        if len(improved) > 0:
            self.log(f"\n🔍 TOP 10 IMPROVEMENTS WITH LAG:")
            for _, row in improved.head(10).iterrows():
                self.log(f"  • {row['Feature']:40s}:")
                self.log(f"      Original: {row['Original_Corr']:+.4f}  →  "
                        f"Lag-{int(row['Best_Lag_Period'])}: {row['Best_Lag_Corr']:+.4f}  "
                        f"(+{row['Improvement']:.4f})")
        
        # Save results
        results_df.to_csv(self.output_dir / "07_lag_analysis.csv", index=False)
        self.log(f"\n✓ Saved: {self.output_dir / '07_lag_analysis.csv'}")
        
        return results_df
    
    def visualize_lag_patterns(self, df_lagged: pd.DataFrame, 
                              top_features: List[str], 
                              lag_periods: List[int]):
        """Visualize how correlation changes with lag.
        
        Args:
            df_lagged: DataFrame with lagged features
            top_features: List of top features to visualize
            lag_periods: List of lag periods used
        """
        self.log("\n" + "="*80)
        self.log("STEP 4.3: LAG PATTERN VISUALIZATION")
        self.log("="*80)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for feat in top_features[:10]:
            if feat not in df_lagged.columns:
                continue
            
            correlations = []
            periods = [0] + lag_periods
            
            # Original (lag 0)
            correlations.append(df_lagged[feat].corr(df_lagged[self.target_column]))
            
            # Lagged
            for lag in lag_periods:
                lag_feat = f"{feat}_lag_{lag}"
                if lag_feat in df_lagged.columns:
                    corr = df_lagged[lag_feat].corr(df_lagged[self.target_column])
                    correlations.append(corr)
                else:
                    correlations.append(np.nan)
            
            ax.plot(periods, correlations, marker='o', linewidth=2, label=feat, alpha=0.7)
        
        ax.set_xlabel('Lag Period (timesteps)', fontweight='bold')
        ax.set_ylabel('Correlation with Target', fontweight='bold')
        ax.set_title('How Correlation Changes with Lag\n(Top 10 Features)', 
                    fontweight='bold', fontsize=12)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / "07_lag_patterns.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log(f"✓ Saved: {self.viz_dir / '07_lag_patterns.png'}")
    
    def save_report(self):
        """Save analysis report to file."""
        report_path = self.output_dir / "07_lag_analysis_report.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(self.report))
        
        self.log(f"\n✓ REPORT SAVED: {report_path}")


def run_lag_analysis(df: pd.DataFrame, target_column: str, 
                    top_features: List[str],
                    lag_periods: List[int] = [5, 10, 15, 20, 30]) -> pd.DataFrame:
    """Run complete lag analysis.
    
    Args:
        df: Input DataFrame
        target_column: Name of target column
        top_features: List of top features to analyze
        lag_periods: List of lag periods to test
        
    Returns:
        DataFrame with lag analysis results and lagged features
    """
    analyzer = LagAnalyzer(df, target_column)
    
    # Create lagged features
    df_lagged, lagged_features = analyzer.create_lagged_features(top_features, lag_periods)
    
    # Analyze correlations
    results_df = analyzer.analyze_lag_correlations(df_lagged, top_features, lagged_features)
    
    # Visualize patterns
    analyzer.visualize_lag_patterns(df_lagged, top_features, lag_periods)
    
    # Save report
    analyzer.save_report()
    
    return df_lagged, results_df


if __name__ == "__main__":
    print("Lag Analysis Module - Ready for use")
