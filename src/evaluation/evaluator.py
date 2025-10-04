"""Model evaluation module with comprehensive metrics.

This module prefers scikit-learn for metrics, but falls back to
NumPy-based implementations if scikit-learn is not available or is
incompatible in the current environment (e.g., Kaggle images with
mixed versions). The public API remains the same.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Attempt to import metrics from scikit-learn. If unavailable or broken,
# provide light-weight NumPy fallbacks for the metrics we use.
try:
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
    _SKLEARN_AVAILABLE = True
except Exception:  # noqa: B902 - broad to catch env import errors
    _SKLEARN_AVAILABLE = False

    def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        correct = (y_true == y_pred).sum()
        return float(correct) / max(1, y_true.size)

    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        classes = np.unique(np.concatenate([y_true, y_pred]))
        n = classes.size
        idx = {c: i for i, c in enumerate(classes)}
        cm = np.zeros((n, n), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[idx[t], idx[p]] += 1
        return cm

    def _precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, average: str) -> Dict[str, float]:
        classes = np.unique(np.concatenate([y_true, y_pred]))
        tp = []
        fp = []
        fn = []
        supports = []
        for c in classes:
            yt = (y_true == c)
            yp = (y_pred == c)
            tp_i = np.logical_and(yt, yp).sum()
            fp_i = np.logical_and(~yt, yp).sum()
            fn_i = np.logical_and(yt, ~yp).sum()
            sup_i = yt.sum()
            tp.append(tp_i)
            fp.append(fp_i)
            fn.append(fn_i)
            supports.append(sup_i)
        tp = np.asarray(tp, dtype=float)
        fp = np.asarray(fp, dtype=float)
        fn = np.asarray(fn, dtype=float)
        supports = np.asarray(supports, dtype=float)

        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)

        if average == "macro":
            return {
                "precision": float(precision.mean()),
                "recall": float(recall.mean()),
                "f1": float(f1.mean()),
            }
        elif average == "weighted":
            w = np.divide(supports, supports.sum(), out=np.zeros_like(supports), where=supports.sum() > 0)
            return {
                "precision": float((precision * w).sum()),
                "recall": float((recall * w).sum()),
                "f1": float((f1 * w).sum()),
            }
        else:
            raise ValueError("Unsupported average type for fallback metrics")

    def precision_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro", zero_division: int = 0) -> float:
        return _precision_recall_f1(y_true, y_pred, average)["precision"]

    def recall_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro", zero_division: int = 0) -> float:
        return _precision_recall_f1(y_true, y_pred, average)["recall"]

    def f1_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro", zero_division: int = 0) -> float:
        return _precision_recall_f1(y_true, y_pred, average)["f1"]

    def classification_report(y_true: np.ndarray, y_pred: np.ndarray, target_names: List[str], zero_division: int = 0) -> str:
        # Minimal text report summarizing precision/recall/f1 per class
        classes = np.unique(np.concatenate([y_true, y_pred]))
        lines = ["precision  recall  f1-score  support"]
        for i, c in enumerate(classes):
            yt = (y_true == c)
            yp = (y_pred == c)
            tp = np.logical_and(yt, yp).sum()
            fp = np.logical_and(~yt, yp).sum()
            fn = np.logical_and(yt, ~yp).sum()
            sup = yt.sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            name = target_names[i] if i < len(target_names) else str(c)
            lines.append(f"{name:>10s}  {prec:7.3f}  {rec:6.3f}  {f1:8.3f}  {sup:7d}")
        return "\n".join(lines)

    def roc_curve(y_true_binary: np.ndarray, y_score: np.ndarray):
        # y_true_binary in {0,1}
        order = np.argsort(-y_score)
        y_true_sorted = y_true_binary[order]
        y_score_sorted = y_score[order]
        # thresholds are unique scores
        thresholds = np.r_[np.inf, np.unique(y_score_sorted)[::-1]]
        tps = np.cumsum(y_true_sorted)
        fps = np.cumsum(1 - y_true_sorted)
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        P = y_true_binary.sum()
        N = y_true_binary.size - P
        tpr = np.divide(tps, P, out=np.zeros_like(tps, dtype=float), where=P > 0)
        fpr = np.divide(fps, N, out=np.zeros_like(fps, dtype=float), where=N > 0)
        return fpr, tpr, thresholds

    def roc_auc_score(y_true: np.ndarray, y_prob: np.ndarray, multi_class: str = "ovr", average: str = "macro") -> float:
        # One-vs-rest AUC per class, macro average
        classes = np.unique(y_true)
        aucs = []
        for c in classes:
            yb = (y_true == c).astype(int)
            scores = y_prob[:, int(c)]
            fpr, tpr, _ = roc_curve(yb, scores)
            # trapezoidal rule
            auc = float(np.trapz(tpr, fpr))
            aucs.append(auc)
        return float(np.mean(aucs)) if len(aucs) > 0 else 0.0


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
