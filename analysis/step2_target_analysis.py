"""Step 2: Target Variable Deep Dive.

Comprehensive analysis of the target variable (volatility_target) to understand
what we're trying to predict.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats


class TargetAnalyzer:
    """Analyzer for target variable characteristics."""
    
    def __init__(self, df: pd.DataFrame, target_column: str, output_dir: str = "analysis/reports"):
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
    
    def analyze_target_distribution(self):
        """Comprehensive analysis of target variable distribution."""
        self.log("\n" + "="*80)
        self.log("STEP 2: TARGET VARIABLE ANALYSIS")
        self.log("="*80)
        
        target = self.df[self.target_column].dropna()
        
        # Basic statistics
        self.log(f"\nTarget Variable: {self.target_column}")
        self.log(f"  • Count        : {len(target):,}")
        self.log(f"  • Mean         : {target.mean():.6f}")
        self.log(f"  • Median       : {target.median():.6f}")
        self.log(f"  • Std Dev      : {target.std():.6f}")
        self.log(f"  • Min          : {target.min():.6f}")
        self.log(f"  • Max          : {target.max():.6f}")
        self.log(f"  • Range        : {target.max() - target.min():.6f}")
        
        # Quartiles
        q25, q75 = target.quantile([0.25, 0.75])
        iqr = q75 - q25
        self.log(f"\n  Quartiles:")
        self.log(f"  • Q1 (25%)     : {q25:.6f}")
        self.log(f"  • Q3 (75%)     : {q75:.6f}")
        self.log(f"  • IQR          : {iqr:.6f}")
        
        # Shape metrics
        skew = target.skew()
        kurt = target.kurtosis()
        self.log(f"\n  Shape Metrics:")
        self.log(f"  • Skewness     : {skew:.3f} {'(right-skewed)' if skew > 0 else '(left-skewed)'}")
        self.log(f"  • Kurtosis     : {kurt:.3f} {'(heavy-tailed)' if kurt > 0 else '(light-tailed)'}")
        
        # Zero/near-zero analysis
        zeros = (target == 0).sum()
        near_zeros = ((target > -0.01) & (target < 0.01)).sum()
        self.log(f"\n  Zero Analysis:")
        self.log(f"  • Exact zeros  : {zeros:,} ({100*zeros/len(target):.2f}%)")
        self.log(f"  • Near-zeros   : {near_zeros:,} ({100*near_zeros/len(target):.2f}%)")
        
        # Outlier detection
        lower_fence = q25 - 1.5 * iqr
        upper_fence = q75 + 1.5 * iqr
        outliers = ((target < lower_fence) | (target > upper_fence)).sum()
        self.log(f"\n  Outliers (IQR method):")
        self.log(f"  • Count        : {outliers:,} ({100*outliers/len(target):.2f}%)")
        self.log(f"  • Lower fence  : {lower_fence:.6f}")
        self.log(f"  • Upper fence  : {upper_fence:.6f}")
        
        # Normality test
        if len(target) < 5000:
            statistic, p_value = stats.shapiro(target)
            test_name = "Shapiro-Wilk"
        else:
            statistic, p_value = stats.kstest(target, 'norm')
            test_name = "Kolmogorov-Smirnov"
        
        self.log(f"\n  Normality Test ({test_name}):")
        self.log(f"  • Test statistic: {statistic:.6f}")
        self.log(f"  • P-value      : {p_value:.6f}")
        self.log(f"  • Distribution : {'NOT NORMAL (p < 0.05)' if p_value < 0.05 else 'NORMAL (p >= 0.05)'}")
        
        # Save statistics
        stats_df = pd.DataFrame({
            'Metric': ['Count', 'Mean', 'Median', 'Std', 'Min', 'Max', 'Range',
                      'Q1', 'Q3', 'IQR', 'Skewness', 'Kurtosis', 'Zeros_pct', 'Outliers_pct'],
            'Value': [len(target), target.mean(), target.median(), target.std(),
                     target.min(), target.max(), target.max() - target.min(),
                     q25, q75, iqr, skew, kurt,
                     100*zeros/len(target), 100*outliers/len(target)]
        })
        stats_df.to_csv(self.output_dir / "04_target_statistics.csv", index=False)
    
    def visualize_target(self):
        """Create comprehensive visualizations for target variable."""
        self.log("\n" + "="*80)
        self.log("STEP 2.1: TARGET VISUALIZATIONS")
        self.log("="*80)
        
        target = self.df[self.target_column].dropna()
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Histogram with KDE
        ax1 = fig.add_subplot(gs[0, :])
        ax1.hist(target, bins=100, edgecolor='black', alpha=0.7, density=True, label='Histogram')
        target.plot(kind='kde', ax=ax1, color='red', linewidth=2, label='KDE')
        ax1.set_xlabel('Target Value')
        ax1.set_ylabel('Density')
        ax1.set_title(f'Distribution of {self.target_column}\n(Mean: {target.mean():.4f}, Median: {target.median():.4f}, Std: {target.std():.4f})',
                     fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. Box plot
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.boxplot(target, vert=True)
        ax2.set_ylabel('Target Value')
        ax2.set_title('Box Plot (Outlier Detection)', fontweight='bold')
        ax2.grid(alpha=0.3)
        
        # 3. Q-Q plot
        ax3 = fig.add_subplot(gs[1, 1])
        stats.probplot(target, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot (Normality Check)', fontweight='bold')
        ax3.grid(alpha=0.3)
        
        # 4. Cumulative distribution
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.hist(target, bins=100, cumulative=True, density=True, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Target Value')
        ax4.set_ylabel('Cumulative Probability')
        ax4.set_title('Cumulative Distribution Function', fontweight='bold')
        ax4.grid(alpha=0.3)
        
        # 5. Percentile plot
        ax5 = fig.add_subplot(gs[2, 1])
        percentiles = np.percentile(target, range(0, 101))
        ax5.plot(range(0, 101), percentiles, linewidth=2)
        ax5.set_xlabel('Percentile')
        ax5.set_ylabel('Target Value')
        ax5.set_title('Percentile Plot', fontweight='bold')
        ax5.grid(alpha=0.3)
        
        plt.savefig(self.viz_dir / "03_target_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log(f"✓ Saved: {self.viz_dir / '03_target_analysis.png'}")
        
        # Additional: Log-scale histogram if heavily skewed
        if abs(target.skew()) > 2:
            self.log("\n⚠ Target is heavily skewed - creating log-scale visualization...")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Original scale
            ax1.hist(target, bins=100, edgecolor='black', alpha=0.7)
            ax1.set_xlabel('Target Value')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Original Scale', fontweight='bold')
            ax1.grid(alpha=0.3)
            
            # Log scale
            target_positive = target[target > 0]
            if len(target_positive) > 0:
                ax2.hist(np.log1p(target_positive), bins=100, edgecolor='black', alpha=0.7)
                ax2.set_xlabel('log(Target + 1)')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Log Scale', fontweight='bold')
                ax2.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.viz_dir / "03b_target_log_scale.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            self.log(f"✓ Saved: {self.viz_dir / '03b_target_log_scale.png'}")
    
    def save_report(self):
        """Save analysis report to file."""
        report_path = self.output_dir / "04_target_analysis_report.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(self.report))
        
        self.log(f"\n✓ REPORT SAVED: {report_path}")


def run_target_analysis(df: pd.DataFrame, target_column: str):
    """Run complete target variable analysis.
    
    Args:
        df: Input DataFrame
        target_column: Name of target column
    """
    analyzer = TargetAnalyzer(df, target_column)
    
    # Analyze distribution
    analyzer.analyze_target_distribution()
    
    # Create visualizations
    analyzer.visualize_target()
    
    # Save report
    analyzer.save_report()


if __name__ == "__main__":
    print("Target Analysis Module - Ready for use")
