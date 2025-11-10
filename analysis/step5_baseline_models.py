"""Step 5: Baseline Modeling - Proof of Concept.

Train simple models to validate hypothesis and extract feature importance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib


class BaselineModeler:
    """Baseline model trainer and evaluator."""
    
    def __init__(self, df: pd.DataFrame, target_column: str, 
                 feature_columns: List[str], output_dir: str = "analysis/reports"):
        """Initialize modeler.
        
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
        
        self.model_dir = Path("analysis/models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.report = []
    
    def log(self, message: str):
        """Add message to report."""
        print(message)
        self.report.append(message)
    
    def prepare_data(self, test_size: float = 0.2, 
                    random_state: int = 42) -> Tuple:
        """Prepare train/test split.
        
        Args:
            test_size: Fraction of data for testing
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        self.log("\n" + "="*80)
        self.log("STEP 5.1: DATA PREPARATION")
        self.log("="*80)
        
        # Filter valid features
        valid_features = [f for f in self.feature_columns if f in self.df.columns]
        
        X = self.df[valid_features].copy()
        y = self.df[self.target_column].copy()
        
        # Drop NaN
        valid_idx = X.notna().all(axis=1) & y.notna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        self.log(f"\nDataset prepared:")
        self.log(f"  • Total samples: {len(X):,}")
        self.log(f"  • Features: {len(valid_features)}")
        self.log(f"  • Test size: {test_size*100}%")
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        self.log(f"\nSplit sizes:")
        self.log(f"  • Train: {len(X_train):,}")
        self.log(f"  • Test: {len(X_test):,}")
        
        return X_train, X_test, y_train, y_test
    
    def train_linear_regression(self, X_train, X_test, y_train, y_test) -> Dict:
        """Train and evaluate Linear Regression model.
        
        Args:
            X_train, X_test, y_train, y_test: Train/test data
            
        Returns:
            Dictionary with results
        """
        self.log("\n" + "="*80)
        self.log("STEP 5.2: LINEAR REGRESSION (Simple Baseline)")
        self.log("="*80)
        
        # Train
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        
        # Predict
        y_train_pred = lr_model.predict(X_train)
        y_test_pred = lr_model.predict(X_test)
        
        # Metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        self.log(f"\n📊 LINEAR REGRESSION RESULTS:")
        self.log(f"\n  Training Set:")
        self.log(f"    • R² Score: {train_r2:.4f}")
        self.log(f"    • MAE: {train_mae:.6f}")
        self.log(f"    • RMSE: {train_rmse:.6f}")
        
        self.log(f"\n  Test Set:")
        self.log(f"    • R² Score: {test_r2:.4f}")
        self.log(f"    • MAE: {test_mae:.6f}")
        self.log(f"    • RMSE: {test_rmse:.6f}")
        
        # Interpretation
        if test_r2 > 0:
            self.log(f"\n  ✓ POSITIVE R²: Model has predictive power!")
        else:
            self.log(f"\n  ✗ NEGATIVE R²: Model worse than mean baseline")
        
        if test_r2 > 0.1:
            self.log(f"  ✓ R² > 0.1: Meaningful linear relationship exists")
        
        # Save model
        joblib.dump(lr_model, self.model_dir / "linear_regression.pkl")
        
        return {
            'model': lr_model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'y_test': y_test,
            'y_test_pred': y_test_pred
        }
    
    def train_random_forest(self, X_train, X_test, y_train, y_test) -> Dict:
        """Train and evaluate Random Forest model.
        
        Args:
            X_train, X_test, y_train, y_test: Train/test data
            
        Returns:
            Dictionary with results
        """
        self.log("\n" + "="*80)
        self.log("STEP 5.3: RANDOM FOREST (Non-linear Baseline)")
        self.log("="*80)
        
        # Train with limited depth to prevent overfitting
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        
        self.log("\nTraining Random Forest (100 trees, max_depth=10)...")
        rf_model.fit(X_train, y_train)
        
        # Predict
        y_train_pred = rf_model.predict(X_train)
        y_test_pred = rf_model.predict(X_test)
        
        # Metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        self.log(f"\n📊 RANDOM FOREST RESULTS:")
        self.log(f"\n  Training Set:")
        self.log(f"    • R² Score: {train_r2:.4f}")
        self.log(f"    • MAE: {train_mae:.6f}")
        self.log(f"    • RMSE: {train_rmse:.6f}")
        
        self.log(f"\n  Test Set:")
        self.log(f"    • R² Score: {test_r2:.4f}")
        self.log(f"    • MAE: {test_mae:.6f}")
        self.log(f"    • RMSE: {test_rmse:.6f}")
        
        # Comparison
        self.log(f"\n  Overfitting Check:")
        self.log(f"    • Train R² - Test R²: {train_r2 - test_r2:.4f}")
        if train_r2 - test_r2 < 0.1:
            self.log(f"    ✓ Minimal overfitting")
        else:
            self.log(f"    ⚠ Some overfitting present")
        
        # Save model
        joblib.dump(rf_model, self.model_dir / "random_forest.pkl")
        
        return {
            'model': rf_model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'y_test': y_test,
            'y_test_pred': y_test_pred
        }
    
    def analyze_feature_importance(self, rf_results: Dict, X_train) -> pd.DataFrame:
        """Extract and analyze feature importance from Random Forest.
        
        Args:
            rf_results: Results from Random Forest training
            X_train: Training features for column names
            
        Returns:
            DataFrame with feature importances
        """
        self.log("\n" + "="*80)
        self.log("STEP 5.4: FEATURE IMPORTANCE ANALYSIS")
        self.log("="*80)
        
        rf_model = rf_results['model']
        
        importance_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        self.log(f"\n🔍 TOP 20 MOST IMPORTANT FEATURES:")
        for idx, row in importance_df.head(20).iterrows():
            self.log(f"  {row['Feature']:40s}: {row['Importance']:.6f}")
        
        # Save
        importance_df.to_csv(self.output_dir / "08_feature_importance.csv", index=False)
        self.log(f"\n✓ Saved: {self.output_dir / '08_feature_importance.csv'}")
        
        return importance_df
    
    def visualize_results(self, lr_results: Dict, rf_results: Dict, 
                         importance_df: pd.DataFrame):
        """Create visualizations for model results.
        
        Args:
            lr_results: Linear Regression results
            rf_results: Random Forest results
            importance_df: Feature importance DataFrame
        """
        self.log("\n" + "="*80)
        self.log("STEP 5.5: RESULT VISUALIZATIONS")
        self.log("="*80)
        
        # 1. Model Comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # LR predictions
        ax1 = axes[0, 0]
        ax1.scatter(lr_results['y_test'], lr_results['y_test_pred'], 
                   alpha=0.3, s=10)
        ax1.plot([lr_results['y_test'].min(), lr_results['y_test'].max()],
                [lr_results['y_test'].min(), lr_results['y_test'].max()],
                'r--', linewidth=2)
        ax1.set_xlabel('Actual')
        ax1.set_ylabel('Predicted')
        ax1.set_title(f'Linear Regression\nTest R² = {lr_results["test_r2"]:.4f}',
                     fontweight='bold')
        ax1.grid(alpha=0.3)
        
        # RF predictions
        ax2 = axes[0, 1]
        ax2.scatter(rf_results['y_test'], rf_results['y_test_pred'], 
                   alpha=0.3, s=10)
        ax2.plot([rf_results['y_test'].min(), rf_results['y_test'].max()],
                [rf_results['y_test'].min(), rf_results['y_test'].max()],
                'r--', linewidth=2)
        ax2.set_xlabel('Actual')
        ax2.set_ylabel('Predicted')
        ax2.set_title(f'Random Forest\nTest R² = {rf_results["test_r2"]:.4f}',
                     fontweight='bold')
        ax2.grid(alpha=0.3)
        
        # Metrics comparison
        ax3 = axes[1, 0]
        metrics = ['R²', 'MAE', 'RMSE']
        lr_vals = [lr_results['test_r2'], lr_results['test_mae'], lr_results['test_rmse']]
        rf_vals = [rf_results['test_r2'], rf_results['test_mae'], rf_results['test_rmse']]
        
        x = np.arange(len(metrics))
        width = 0.35
        ax3.bar(x - width/2, lr_vals, width, label='Linear Regression', alpha=0.8)
        ax3.bar(x + width/2, rf_vals, width, label='Random Forest', alpha=0.8)
        ax3.set_xlabel('Metric')
        ax3.set_ylabel('Value')
        ax3.set_title('Model Comparison (Test Set)', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics)
        ax3.legend()
        ax3.grid(alpha=0.3, axis='y')
        
        # Feature importance
        ax4 = axes[1, 1]
        top_features = importance_df.head(15)
        ax4.barh(range(len(top_features)), top_features['Importance'], alpha=0.7)
        ax4.set_yticks(range(len(top_features)))
        ax4.set_yticklabels(top_features['Feature'], fontsize=8)
        ax4.set_xlabel('Importance')
        ax4.set_title('Top 15 Feature Importance (Random Forest)', fontweight='bold')
        ax4.grid(alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / "08_baseline_models.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log(f"✓ Saved: {self.viz_dir / '08_baseline_models.png'}")
    
    def save_report(self):
        """Save analysis report to file."""
        report_path = self.output_dir / "08_baseline_models_report.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(self.report))
        
        self.log(f"\n✓ REPORT SAVED: {report_path}")


def run_baseline_modeling(df: pd.DataFrame, target_column: str, 
                         feature_columns: List[str]) -> Dict:
    """Run complete baseline modeling pipeline.
    
    Args:
        df: Input DataFrame
        target_column: Name of target column
        feature_columns: List of feature column names
        
    Returns:
        Dictionary with model results and feature importance
    """
    modeler = BaselineModeler(df, target_column, feature_columns)
    
    # Prepare data
    X_train, X_test, y_train, y_test = modeler.prepare_data()
    
    # Train Linear Regression
    lr_results = modeler.train_linear_regression(X_train, X_test, y_train, y_test)
    
    # Train Random Forest
    rf_results = modeler.train_random_forest(X_train, X_test, y_train, y_test)
    
    # Feature importance
    importance_df = modeler.analyze_feature_importance(rf_results, X_train)
    
    # Visualize
    modeler.visualize_results(lr_results, rf_results, importance_df)
    
    # Save report
    modeler.save_report()
    
    return {
        'lr_results': lr_results,
        'rf_results': rf_results,
        'importance_df': importance_df
    }


if __name__ == "__main__":
    print("Baseline Modeling Module - Ready for use")
