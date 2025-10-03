"""Visualization utilities for model evaluation."""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class MetricsVisualizer:
    """Creates visualizations for model evaluation metrics."""

    def __init__(self, class_names: List[str]) -> None:
        """Initialize visualizer.

        Args:
            class_names: List of class names.
        """
        self.class_names = class_names
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)

    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        output_path: Path,
        normalize: bool = False,
    ) -> None:
        """Plot confusion matrix as a heatmap.

        Args:
            confusion_matrix: Confusion matrix array.
            output_path: Path to save the plot.
            normalize: Whether to normalize the matrix.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if normalize:
            cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            cm = confusion_matrix
            fmt = 'd'
            title = 'Confusion Matrix'

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
        )
        plt.title(title, fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        output_dir: Path,
    ) -> None:
        """Plot training and validation curves.

        Args:
            history: Training history dictionary.
            output_dir: Directory to save plots.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        epochs = range(1, len(history['train_loss']) + 1)

        axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0].set_title('Loss Curves', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        axes[1].set_title('Accuracy Curves', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history['learning_rate'], 'g-', linewidth=2)
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'learning_rate.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_roc_curves(
        self,
        roc_curves: Dict[str, Dict[str, np.ndarray]],
        output_path: Path,
    ) -> None:
        """Plot ROC curves for all classes.

        Args:
            roc_curves: Dictionary with ROC curve data for each class.
            output_path: Path to save the plot.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 8))

        colors = plt.cm.Set3(np.linspace(0, 1, len(roc_curves)))

        for (class_name, roc_data), color in zip(roc_curves.items(), colors):
            fpr = roc_data['fpr']
            tpr = roc_data['tpr']
            auc = np.trapz(tpr, fpr)

            plt.plot(
                fpr,
                tpr,
                color=color,
                linewidth=2,
                label=f'{class_name} (AUC = {auc:.3f})'
            )

        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_class_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_path: Path,
    ) -> None:
        """Plot class distribution comparison.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            output_path: Path to save the plot.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        true_counts = np.bincount(y_true, minlength=len(self.class_names))
        pred_counts = np.bincount(y_pred, minlength=len(self.class_names))

        x = np.arange(len(self.class_names))
        width = 0.35

        axes[0].bar(x, true_counts, width, label='True', color='skyblue')
        axes[0].set_xlabel('Class', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title('True Class Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(self.class_names, rotation=45, ha='right')
        axes[0].grid(True, alpha=0.3, axis='y')

        axes[1].bar(x, pred_counts, width, label='Predicted', color='lightcoral')
        axes[1].set_xlabel('Class', fontsize=12)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].set_title('Predicted Class Distribution', fontsize=14, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(self.class_names, rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
