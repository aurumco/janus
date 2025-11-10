"""Step 3: Bivariate Analysis - Correlation & Relationships.

This module proves (or disproves) the hypothesis that features have
predictive power for the target variable.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple


class CorrelationAnalyzer:
    """Analyzer for feature-target correlations."""
    
    def __init__(self, df: pd.DataFrame, target_column: str, 
                 feature_columns: List[str], output_dir: str = "analysis/reports"):
        """Initialize analyzer.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            feature_columns: List of feature column names
            output_dir: Directory for saving reports
        """
        self.df = df
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.viz_dir = Path("analysis/visualizations")
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        self.report = []
    
    def log(self, message: str):
        """Add message to report."""
        print(message)
        self.report.append(message)
    
    def compute_correlations(self) -> pd.DataFrame:
        """Compute Pearson correlations between features and target.
        
        Returns:
            DataFrame with correlation values
        """
        self.log("\n" + "="*80)
        self.log("STEP 3.1: LINEAR CORRELATION ANALYSIS")
        self.log("="*80)
        
        # Filter features that exist in dataframe
        valid_features = [f for f in self.feature_columns if f in self.df.columns]
        
        if len(valid_features) < len(self.feature_columns):
            missing = set(self.feature_columns) - set(valid_features)
            self.log(f"\n⚠ Warning: {len(missing)} features not found in dataset:")
            for feat in list(missing)[:5]:
                self.log(f"  • {feat}")
        
        self.log(f"\nAnalyzing {len(valid_features)} features...")
        
        # Compute correlation matrix
        cols_to_analyze = valid_features + [self.target_column]
        corr_matrix = self.df[cols_to_analyze].corr()
        
        # Extract correlations with target
        target_corr = corr_matrix[self.target_column].drop(self.target_column)
        target_corr = target_corr.sort_values(ascending=False)
        
        # Create results dataframe
        corr_df = pd.DataFrame({
            'Feature': target_corr.index,
            'Correlation': target_corr.values,
            'Abs_Correlation': np.abs(target_corr.values)
        })
        
        # Statistics
        self.log(f"\nCorrelation Statistics:")
        self.log(f"  • Mean correlation    : {target_corr.mean():.4f}")
        self.log(f"  • Median correlation  : {target_corr.median():.4f}")
        self.log(f"  • Max correlation     : {target_corr.max():.4f}")
        self.log(f"  • Min correlation     : {target_corr.min():.4f}")
        
        # Count significant correlations
        sig_positive = (target_corr > 0.1).sum()
        sig_negative = (target_corr < -0.1).sum()
        self.log(f"  • Significant (>0.1)  : {sig_positive}")
        self.log(f"  • Significant (<-0.1) : {sig_negative}")
        
        # Top 10 positive and negative
        self.log("\n📊 TOP 10 POSITIVE CORRELATIONS:")
        for feat, corr in target_corr.head(10).items():
            self.log(f"  {feat:40s}: {corr:+.4f}")
        
        self.log("\n📊 TOP 10 NEGATIVE CORRELATIONS:")
        for feat, corr in target_corr.tail(10).items():
            self.log(f"  {feat:40s}: {corr:+.4f}")
        
        # Save full correlation matrix and target correlations
        corr_matrix.to_csv(self.output_dir / "05_correlation_matrix.csv")
        corr_df.to_csv(self.output_dir / "06_target_correlations.csv", index=False)
        
        self.log(f"\n✓ Saved correlation matrix and target correlations")
        
        return corr_df, corr_matrix
    
    def visualize_correlations(self, corr_df: pd.DataFrame, corr_matrix: pd.DataFrame):
        """Create correlation visualizations.
        
        Args:
            corr_df: DataFrame with target correlations
            corr_matrix: Full correlation matrix
        """
        self.log("\n" + "="*80)
        self.log("STEP 3.2: CORRELATION VISUALIZATIONS")
        self.log("="*80)
        
        # 1. Bar chart of top correlations with target
        top_n = 20
        top_features = corr_df.nlargest(top_n, 'Abs_Correlation')
        
        plt.figure(figsize=(12, 8))
        colors = ['green' if x > 0 else 'red' for x in top_features['Correlation']]
        plt.barh(range(len(top_features)), top_features['Correlation'], color=colors, alpha=0.7)
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Correlation with Target')
        plt.title(f'Top {top_n} Features by Absolute Correlation with {self.target_column}',
                 fontweight='bold', fontsize=12)
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        plt.axvline(x=0.1, color='blue', linestyle='--', alpha=0.5, label='±0.1 threshold')
        plt.axvline(x=-0.1, color='blue', linestyle='--', alpha=0.5)
        plt.legend()
        plt.grid(alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(self.viz_dir / "04_target_correlations_bar.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log(f"✓ Saved: {self.viz_dir / '04_target_correlations_bar.png'}")
        
        # 2. Heatmap of full correlation matrix (top features only)
        top_features_list = top_features['Feature'].tolist()[:15]
        matrix_subset = corr_matrix.loc[
            top_features_list + [self.target_column],
            top_features_list + [self.target_column]
        ]
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(matrix_subset, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, vmin=-1, vmax=1, square=True,
                   linewidths=0.5, cbar_kws={'label': 'Correlation'})
        plt.title(f'Correlation Heatmap: Top 15 Features + Target', 
                 fontweight='bold', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.viz_dir / "05_correlation_heatmap.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log(f"✓ Saved: {self.viz_dir / '05_correlation_heatmap.png'}")
    
    def create_scatter_plots(self, top_n: int = 5):
        """Create scatter plots for top correlated features.
        
        Args:
            top_n: Number of top features to plot
        """
        self.log("\n" + "="*80)
        self.log(f"STEP 3.3: SCATTER PLOTS (Top {top_n} Features)")
        self.log("="*80)
        
        # Get top features by absolute correlation
        valid_features = [f for f in self.feature_columns if f in self.df.columns]
        corr_values = self.df[valid_features].corrwith(self.df[self.target_column])
        top_features = corr_values.abs().nlargest(top_n).index.tolist()
        
        # Create scatter plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, feat in enumerate(top_features):
            if idx < len(axes):
                ax = axes[idx]
                
                # Sample data if too large
                if len(self.df) > 10000:
                    sample_df = self.df[[feat, self.target_column]].sample(10000, random_state=42)
                else:
                    sample_df = self.df[[feat, self.target_column]]
                
                # Scatter plot with regression line
                sns.regplot(data=sample_df, x=feat, y=self.target_column, 
                          ax=ax, scatter_kws={'alpha': 0.3, 's': 10},
                          line_kws={'color': 'red', 'linewidth': 2})
                
                corr = self.df[feat].corr(self.df[self.target_column])
                ax.set_title(f'{feat}\nCorrelation: {corr:.4f}', fontweight='bold')
                ax.set_xlabel(feat)
                ax.set_ylabel(self.target_column)
                ax.grid(alpha=0.3)
        
        # Hide unused subplot
        if len(top_features) < len(axes):
            for idx in range(len(top_features), len(axes)):
                axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / "06_scatter_plots.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log(f"✓ Saved: {self.viz_dir / '06_scatter_plots.png'}")
        
        # Analysis
        self.log(f"\n📊 Relationship Analysis:")
        for feat in top_features:
            corr = self.df[feat].corr(self.df[self.target_column])
            if abs(corr) > 0.3:
                relationship = "Strong"
            elif abs(corr) > 0.1:
                relationship = "Moderate"
            else:
                relationship = "Weak"
            
            direction = "positive" if corr > 0 else "negative"
            self.log(f"  • {feat}: {relationship} {direction} relationship (r={corr:.4f})")
    
    def save_report(self):
        """Save analysis report to file."""
        report_path = self.output_dir / "06_correlation_analysis_report.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(self.report))
        
        self.log(f"\n✓ REPORT SAVED: {report_path}")


def run_correlation_analysis(df: pd.DataFrame, target_column: str, 
                            feature_columns: List[str]) -> pd.DataFrame:
    """Run complete correlation analysis.
    
    Args:
        df: Input DataFrame
        target_column: Name of target column
        feature_columns: List of feature column names
        
    Returns:
        DataFrame with correlation values
    """
    analyzer = CorrelationAnalyzer(df, target_column, feature_columns)
    
    # Compute correlations
    corr_df, corr_matrix = analyzer.compute_correlations()
    
    # Visualize
    analyzer.visualize_correlations(corr_df, corr_matrix)
    
    # Scatter plots
    analyzer.create_scatter_plots(top_n=5)
    
    # Save report
    analyzer.save_report()
    
    return corr_df


if __name__ == "__main__":
    print("Correlation Analysis Module - Ready for use")
