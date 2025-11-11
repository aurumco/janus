"""Main Entry Point: Complete Data Analysis Pipeline.

This script runs a comprehensive analysis of the cryptocurrency dataset
to validate the hypothesis that features have predictive power for volatility.

Usage:
    python run_analysis.py

For Kaggle:
    1. Upload this folder as a dataset or notebook
    2. Set INPUT_PATH to your Parquet file path
    3. Run this script
"""

import pandas as pd
import numpy as np
import sys
import warnings
from pathlib import Path
from datetime import datetime

# Import analysis modules
from step1_data_health import run_data_health_check
from step2_target_analysis import run_target_analysis
from step3_correlation_analysis import run_correlation_analysis
from step4_lag_analysis import run_lag_analysis
from step5_baseline_models import run_baseline_modeling

warnings.filterwarnings('ignore')


def print_banner(text: str):
    """Print a formatted banner."""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")


def load_dataset(file_path: str, sample_size: int = None) -> pd.DataFrame:
    """Load dataset from Parquet file.
    
    Args:
        file_path: Path to Parquet file
        sample_size: If specified, sample this many rows for faster analysis
        
    Returns:
        Loaded DataFrame
    """
    print_banner("LOADING DATASET")
    
    print(f"📂 Loading from: {file_path}")
    
    try:
        # Load full dataset
        df = pd.read_parquet(file_path)
        
        print(f"✓ Loaded successfully!")
        print(f"  • Shape: {df.shape}")
        print(f"  • Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        print(f"  • Columns: {len(df.columns)}")
        
        # Sample if requested
        if sample_size and len(df) > sample_size:
            print(f"\n⚠ Sampling {sample_size:,} rows for faster analysis...")
            df = df.sample(sample_size, random_state=42).reset_index(drop=True)
            print(f"  • New shape: {df.shape}")
        
        # Display columns
        print(f"\n📋 Available columns:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2d}. {col}")
        
        return df
        
    except FileNotFoundError:
        print(f"❌ ERROR: File not found: {file_path}")
        print("\n💡 For Kaggle, use:")
        print("   '/kaggle/input/your-dataset/file.parquet'")
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: Failed to load file")
        print(f"   {str(e)}")
        sys.exit(1)


def detect_target_column(df: pd.DataFrame) -> str:
    """Auto-detect target column (volatility).
    
    Args:
        df: Input DataFrame
        
    Returns:
        Name of target column
    """
    # Look for common volatility column names
    candidates = [
        'volatility_target',
        'volatility',
        'target_volatility',
        'vol_target',
        'future_volatility'
    ]
    
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    
    # If not found, look for anything with 'volatility' or 'target'
    for col in df.columns:
        if 'volatility' in col.lower() or 'target' in col.lower():
            return col
    
    raise ValueError("Could not detect target column. Please specify manually.")


def detect_feature_columns(df: pd.DataFrame, target_column: str) -> list:
    """Auto-detect feature columns.
    
    Args:
        df: Input DataFrame
        target_column: Name of target column to exclude
        
    Returns:
        List of feature column names
    """
    # Exclude non-features
    exclude_patterns = ['timestamp', 'date', 'time', 'symbol', 'asset', 'id']
    
    feature_cols = []
    for col in df.columns:
        # Skip target
        if col == target_column:
            continue
        
        # Skip if matches exclude pattern
        if any(pattern in col.lower() for pattern in exclude_patterns):
            continue
        
        # Skip if not numeric
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        
        feature_cols.append(col)
    
    return feature_cols


def create_final_summary(results: dict, output_dir: str = "analysis/reports"):
    """Create final summary report.
    
    Args:
        results: Dictionary with all analysis results
        output_dir: Directory for saving report
    """
    print_banner("CREATING FINAL SUMMARY")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    summary = []
    summary.append("="*80)
    summary.append("  JANUS V5: COMPREHENSIVE DATA ANALYSIS - FINAL SUMMARY")
    summary.append("="*80)
    summary.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Data Health
    summary.append("\n" + "-"*80)
    summary.append("1. DATA HEALTH")
    summary.append("-"*80)
    summary.append(f"  • Total samples analyzed: {results['n_samples']:,}")
    summary.append(f"  • Features analyzed: {results['n_features']}")
    summary.append(f"  • Highly skewed features: {results['n_skewed']}")
    summary.append(f"  • Log-transformed features created: {results['n_log_features']}")
    
    # Target Analysis
    summary.append("\n" + "-"*80)
    summary.append("2. TARGET VARIABLE (Volatility)")
    summary.append("-"*80)
    summary.append(f"  • Mean: {results['target_mean']:.6f}")
    summary.append(f"  • Std Dev: {results['target_std']:.6f}")
    summary.append(f"  • Skewness: {results['target_skew']:.3f}")
    
    # Correlation Analysis
    summary.append("\n" + "-"*80)
    summary.append("3. CORRELATION ANALYSIS")
    summary.append("-"*80)
    summary.append(f"  • Features with |corr| > 0.1: {results['n_corr_sig']}")
    summary.append(f"  • Top correlated feature: {results['top_corr_feature']}")
    summary.append(f"  • Top correlation value: {results['top_corr_value']:.4f}")
    
    # Lag Analysis
    if 'lag_results' in results:
        summary.append("\n" + "-"*80)
        summary.append("4. LAG ANALYSIS")
        summary.append("-"*80)
        summary.append(f"  • Features improved with lag: {results['n_lag_improved']}")
        if results['n_lag_improved'] > 0:
            summary.append(f"  • Best improvement: {results['best_lag_improvement']:.4f}")
    
    # Baseline Models
    summary.append("\n" + "-"*80)
    summary.append("5. BASELINE MODELS")
    summary.append("-"*80)
    summary.append(f"\n  Linear Regression:")
    summary.append(f"    • Test R²: {results['lr_r2']:.4f}")
    summary.append(f"    • Test MAE: {results['lr_mae']:.6f}")
    
    summary.append(f"\n  Random Forest:")
    summary.append(f"    • Test R²: {results['rf_r2']:.4f}")
    summary.append(f"    • Test MAE: {results['rf_mae']:.6f}")
    
    summary.append(f"\n  Top 5 Most Important Features:")
    for i, feat in enumerate(results['top5_features'], 1):
        summary.append(f"    {i}. {feat}")
    
    # Final Verdict
    summary.append("\n" + "="*80)
    summary.append("6. FINAL VERDICT")
    summary.append("="*80)
    
    if results['rf_r2'] > 0.05:
        summary.append("\n  ✓ HYPOTHESIS SUPPORTED")
        summary.append("    The features show predictive power for volatility.")
        summary.append(f"    Random Forest achieved R² = {results['rf_r2']:.4f}")
        
        if results['rf_r2'] > 0.15:
            summary.append("\n  ✓ STRONG SIGNAL DETECTED")
            summary.append("    Proceed with confidence to Mamba-SSM model.")
        elif results['rf_r2'] > 0.05:
            summary.append("\n  ⚠ MODERATE SIGNAL")
            summary.append("    Proceed with caution. Consider feature engineering.")
        
        summary.append(f"\n  📋 RECOMMENDED FEATURES FOR MAMBA MODEL:")
        summary.append(f"    Use the top {min(20, len(results['top5_features']))} features from importance ranking.")
        
    else:
        summary.append("\n  ✗ HYPOTHESIS NOT SUPPORTED")
        summary.append("    Features show weak/no predictive power.")
        summary.append("    Consider:")
        summary.append("      • Different features")
        summary.append("      • Different target definition")
        summary.append("      • More feature engineering")
    
    summary.append("\n" + "="*80)
    summary.append("  END OF ANALYSIS")
    summary.append("="*80)
    
    # Save
    summary_text = '\n'.join(summary)
    summary_path = output_path / "00_FINAL_SUMMARY.txt"
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    
    # Print
    print(summary_text)
    print(f"\n✓ SUMMARY SAVED: {summary_path}")


def main():
    """Main analysis pipeline."""
    print("\n")
    print("="*80)
    print("  JANUS V5: COMPREHENSIVE CRYPTOCURRENCY DATA ANALYSIS")
    print("  Hypothesis Validation Before Deploying Mamba-SSM Model")
    print("="*80)
    print("\n")
    
    # Configuration
    # Modify this path for your environment
    INPUT_PATH = "/kaggle/input/janusds/janus_pretrain_5min/janus_pretrain_5min_dataset.parquet"
    
    # For faster testing, set a sample size (or None for full dataset)
    SAMPLE_SIZE = None  # Set to 500000 for quick testing
    
    # Lag periods to test (in minutes for 1-min data)
    LAG_PERIODS = [5, 10, 15, 20, 30, 60]
    
    print(f"⚙ Configuration:")
    print(f"  • Input: {INPUT_PATH}")
    print(f"  • Sample size: {'Full dataset' if SAMPLE_SIZE is None else f'{SAMPLE_SIZE:,} rows'}")
    print(f"  • Lag periods: {LAG_PERIODS}")
    
    # Step 0: Load dataset
    df = load_dataset(INPUT_PATH, sample_size=SAMPLE_SIZE)
    
    print_banner("REMOVING CONTAMINATED FEATURES (Data Leakage Prevention)")
    
    CONTAMINATED_FEATURES = [
        'garch_volatility',     # The target itself - will be detected separately
        'ATR_5_pct_M15',        # Leaks current volatility (corr: 0.87+)
        'log_return'            # Direct ingredient for GARCH calculation
    ]
    
    # Drop contaminated features (except target which we need)
    features_to_drop = [col for col in CONTAMINATED_FEATURES if col in df.columns and col != 'garch_volatility']
    if features_to_drop:
        print(f"🚫 Dropping contaminated features: {features_to_drop}")
        df = df.drop(columns=features_to_drop)
        print(f"  ✓ Dropped {len(features_to_drop)} feature(s)")
        print(f"  • New shape: {df.shape}")
    else:
        print("  ✓ No contaminated features found (or already removed)")
    
    # Detect target and features
    print_banner("DETECTING TARGET & FEATURES")
    
    target_column = detect_target_column(df)
    print(f"✓ Target column detected: {target_column}")
    
    feature_columns = detect_feature_columns(df, target_column)
    
    original_len = len(feature_columns)
    feature_columns = list(dict.fromkeys(feature_columns))
    if len(feature_columns) < original_len:
        print(f"⚠ Removed {original_len - len(feature_columns)} duplicate feature(s)")
    
    print(f"✓ Feature columns detected: {len(feature_columns)}")
    print(f"  First 10: {feature_columns[:10]}")
    
    # Initialize results dictionary
    results = {
        'n_samples': len(df),
        'n_features': len(feature_columns)
    }
    
    # Step 1: Data Health Check
    print_banner("STEP 1: DATA HEALTH CHECK")
    df_clean = run_data_health_check(df, target_column=target_column)
    
    # Update feature list with new log features (excluding log transforms of target)
    new_features = [col for col in df_clean.columns 
                   if col.startswith('log_') and col != f'log_{target_column}']
    if new_features:
        print(f"\n📊 Adding {len(new_features)} log-transformed features to feature list")
        feature_columns.extend(new_features)
    
    # Remove duplicates again after adding log features
    feature_columns = list(dict.fromkeys(feature_columns))
    
    results['n_skewed'] = len([col for col in df.columns if f'log_{col}' in df_clean.columns])
    results['n_log_features'] = len(new_features)
    
    # Step 2: Target Analysis
    print_banner("STEP 2: TARGET VARIABLE ANALYSIS")
    run_target_analysis(df_clean, target_column)
    
    target_stats = df_clean[target_column].describe()
    results['target_mean'] = target_stats['mean']
    results['target_std'] = target_stats['std']
    results['target_skew'] = df_clean[target_column].skew()
    
    # Step 3: Correlation Analysis
    print_banner("STEP 3: CORRELATION ANALYSIS")
    corr_df = run_correlation_analysis(df_clean, target_column, feature_columns)
    
    sig_corr = corr_df[corr_df['Abs_Correlation'] > 0.1]
    results['n_corr_sig'] = len(sig_corr)
    results['top_corr_feature'] = corr_df.iloc[0]['Feature']
    results['top_corr_value'] = corr_df.iloc[0]['Correlation']
    
    # Step 4: Lag Analysis
    print_banner("STEP 4: LAG ANALYSIS")
    top_features = corr_df.head(10)['Feature'].tolist()
    df_lagged, lag_results = run_lag_analysis(df_clean, target_column, top_features, LAG_PERIODS)
    
    improved = lag_results[lag_results['Improvement'] > 0]
    results['n_lag_improved'] = len(improved)
    if len(improved) > 0:
        results['best_lag_improvement'] = improved['Improvement'].max()
    
    # Step 5: Baseline Modeling
    print_banner("STEP 5: BASELINE MODELING")
    model_results = run_baseline_modeling(df_lagged, target_column, feature_columns, use_gpu=True)
    
    results['lr_r2'] = model_results['lr_results']['test_r2']
    results['lr_mae'] = model_results['lr_results']['test_mae']
    results['rf_r2'] = model_results['rf_results']['test_r2']
    results['rf_mae'] = model_results['rf_results']['test_mae']
    results['top5_features'] = model_results['importance_df'].head(5)['Feature'].tolist()
    
    # Final Summary
    create_final_summary(results)
    
    print_banner("ANALYSIS COMPLETE!")
    print("📁 All reports saved in: analysis/reports/")
    print("📊 All visualizations saved in: analysis/visualizations/")
    print("🤖 Models saved in: analysis/models/")
    print("\n✓ You can now review the results and decide on next steps.")
    print("\n")


if __name__ == "__main__":
    main()
