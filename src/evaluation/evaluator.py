"""Model evaluation module with comprehensive metrics."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


class ModelEvaluator:
    """Evaluates model performance with comprehensive metrics."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        class_names: List[str],
    ) -> None:
        """Initialize evaluator.

        Args:
            model: Model to evaluate.
            device: Device to run evaluation on.
            class_names: List of class names for reporting.
        """
        self.model = model
        self.device = device
        self.class_names = class_names
        self.num_classes = len(class_names)

    def evaluate(self, data_loader: DataLoader) -> Dict:
        """Evaluate model on a dataset.

        Args:
            data_loader: DataLoader for evaluation.

        Returns:
            Dictionary containing all evaluation metrics.
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []

        with torch.no_grad():
            for inputs, targets in tqdm(data_loader, desc="Evaluating"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = outputs.max(1)

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)

        metrics = self._compute_metrics(
            all_targets,
            all_predictions,
            all_probabilities
        )

        return metrics

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
    ) -> Dict:
        """Compute comprehensive evaluation metrics.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_prob: Predicted probabilities.

        Returns:
            Dictionary with all metrics.
        """
        metrics = {}

        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

        metrics['classification_report'] = classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            zero_division=0
        )

        try:
            metrics['roc_auc_ovr'] = roc_auc_score(
                y_true,
                y_prob,
                multi_class='ovr',
                average='macro'
            )
            metrics['roc_auc_ovo'] = roc_auc_score(
                y_true,
                y_prob,
                multi_class='ovo',
                average='macro'
            )
        except ValueError:
            metrics['roc_auc_ovr'] = 0.0
            metrics['roc_auc_ovo'] = 0.0

        metrics['roc_curves'] = {}
        for i in range(self.num_classes):
            y_true_binary = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, i])
            metrics['roc_curves'][self.class_names[i]] = {
                'fpr': fpr,
                'tpr': tpr,
            }

        metrics['per_class_accuracy'] = {}
        for i in range(self.num_classes):
            mask = y_true == i
            if mask.sum() > 0:
                class_acc = (y_pred[mask] == y_true[mask]).mean()
                metrics['per_class_accuracy'][self.class_names[i]] = class_acc
            else:
                metrics['per_class_accuracy'][self.class_names[i]] = 0.0

        return metrics

    def print_metrics(self, metrics: Dict) -> None:
        """Print evaluation metrics in a formatted way.

        Args:
            metrics: Dictionary of metrics from evaluate().
        """
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)

        print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro F1-Score: {metrics['f1_macro']:.4f}")
        print(f"Weighted F1-Score: {metrics['f1_weighted']:.4f}")
        print(f"ROC AUC (OvR): {metrics['roc_auc_ovr']:.4f}")
        print(f"ROC AUC (OvO): {metrics['roc_auc_ovo']:.4f}")

        print("\nPer-Class Accuracy:")
        for class_name, acc in metrics['per_class_accuracy'].items():
            print(f"  {class_name}: {acc:.4f}")

        print("\n" + "-"*70)
        print("CLASSIFICATION REPORT")
        print("-"*70)
        print(metrics['classification_report'])

        print("\n" + "-"*70)
        print("CONFUSION MATRIX")
        print("-"*70)
        print(metrics['confusion_matrix'])
        print("="*70 + "\n")

    def save_metrics(self, metrics: Dict, output_path: Path) -> None:
        """Save metrics to a text file.

        Args:
            metrics: Dictionary of metrics.
            output_path: Path to save the metrics file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("EVALUATION RESULTS\n")
            f.write("="*70 + "\n\n")

            f.write(f"Overall Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Macro F1-Score: {metrics['f1_macro']:.4f}\n")
            f.write(f"Weighted F1-Score: {metrics['f1_weighted']:.4f}\n")
            f.write(f"ROC AUC (OvR): {metrics['roc_auc_ovr']:.4f}\n")
            f.write(f"ROC AUC (OvO): {metrics['roc_auc_ovo']:.4f}\n\n")

            f.write("Per-Class Accuracy:\n")
            for class_name, acc in metrics['per_class_accuracy'].items():
                f.write(f"  {class_name}: {acc:.4f}\n")

            f.write("\n" + "-"*70 + "\n")
            f.write("CLASSIFICATION REPORT\n")
            f.write("-"*70 + "\n")
            f.write(metrics['classification_report'])
            f.write("\n")

            f.write("\n" + "-"*70 + "\n")
            f.write("CONFUSION MATRIX\n")
            f.write("-"*70 + "\n")
            f.write(str(metrics['confusion_matrix']))
            f.write("\n" + "="*70 + "\n")
