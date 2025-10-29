"""Visualization utilities for model evaluation."""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class MetricsVisualizer:
    """Creates visualizations for model evaluation metrics."""

    def __init__(self) -> None:
        """Initialize visualizer."""
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)


    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        output_dir: Path,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        if not history.get('train_loss') or not history.get('val_loss'):
            return

        epochs = range(1, len(history['train_loss']) + 1)
        marker_style = 'o' if len(epochs) <= 10 else None
        marker_size = 8 if len(epochs) <= 10 else 4

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', 
                     linewidth=2, marker=marker_style, markersize=marker_size)
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', 
                     linewidth=2, marker=marker_style, markersize=marker_size)
        axes[0].set_title('Loss Curves', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, history['learning_rate'], 'g-', linewidth=2, 
                     marker=marker_style, markersize=marker_size)
        axes[1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Learning Rate', fontsize=12)
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_path: Path,
    ) -> None:
        """Plot predicted vs actual values.

        Args:
            y_true: True values.
            y_pred: Predicted values.
            output_path: Path to save the plot.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Scatter plot: Predicted vs Actual
        axes[0].scatter(y_true, y_pred, alpha=0.5, s=10)
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0].set_xlabel('True Values', fontsize=12)
        axes[0].set_ylabel('Predicted Values', fontsize=12)
        axes[0].set_title('Predicted vs Actual', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Histogram of errors
        errors = y_pred - y_true
        axes[1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Prediction Error', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Error Distribution', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_residuals(
        self,
        residuals: np.ndarray,
        output_path: Path,
    ) -> None:
        """Plot residual analysis.

        Args:
            residuals: Residuals (true - predicted).
            output_path: Path to save the plot.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Residual plot
        axes[0, 0].scatter(range(len(residuals)), residuals, alpha=0.5, s=10)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Sample Index', fontsize=12)
        axes[0, 0].set_ylabel('Residual', fontsize=12)
        axes[0, 0].set_title('Residual Plot', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)

        # Residual histogram
        axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Residual Value', fontsize=12)
        axes[0, 1].set_ylabel('Frequency', fontsize=12)
        axes[0, 1].set_title('Residual Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

        # Absolute residuals
        axes[1, 1].scatter(range(len(residuals)), np.abs(residuals), alpha=0.5, s=10)
        axes[1, 1].set_xlabel('Sample Index', fontsize=12)
        axes[1, 1].set_ylabel('|Residual|', fontsize=12)
        axes[1, 1].set_title('Absolute Residuals', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
