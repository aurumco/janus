"""Model evaluation module for regression metrics."""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


class ModelEvaluator:
    """Evaluates regression model performance with comprehensive metrics."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
    ) -> None:
        """Initialize evaluator.

        Args:
            model: Model to evaluate.
            device: Device to run evaluation on.
        """
        self.model = model
        self.device = device

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

        with torch.no_grad():
            for inputs, targets in tqdm(data_loader, desc="Evaluating"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)

                # Handle quantile outputs: (batch, num_quantiles)
                if outputs.ndim == 2 and outputs.size(1) > 1:
                    median_idx = outputs.size(1) // 2
                    median_pred = outputs[:, median_idx]
                    all_predictions.extend(median_pred.detach().cpu().numpy())
                else:
                    all_predictions.extend(outputs.squeeze(-1).detach().cpu().numpy())

                all_targets.extend(targets.squeeze(-1).detach().cpu().numpy())

        all_predictions = np.array(all_predictions).flatten()
        all_targets = np.array(all_targets).flatten()

        metrics = self._compute_metrics(all_targets, all_predictions)
        metrics['y_true'] = all_targets
        metrics['y_pred'] = all_predictions

        return metrics

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict:
        """Compute comprehensive regression evaluation metrics.

        Args:
            y_true: True values.
            y_pred: Predicted values.

        Returns:
            Dictionary with all metrics.
        """
        metrics = {}

        # Standard regression metrics
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(y_true, y_pred)

        # Residuals
        residuals = y_true - y_pred
        metrics['residuals'] = residuals
        metrics['mean_residual'] = np.mean(residuals)
        metrics['std_residual'] = np.std(residuals)

        # Sign accuracy (direction prediction)
        # Predicting if price will go up or down
        true_signs = np.sign(y_true)
        pred_signs = np.sign(y_pred)
        metrics['sign_accuracy'] = np.mean(true_signs == pred_signs)

        # Mean Absolute Percentage Error (MAPE)
        # Only calculate for non-zero true values
        non_zero_mask = y_true != 0
        if non_zero_mask.sum() > 0:
            metrics['mape'] = np.mean(
                np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])
            ) * 100
        else:
            metrics['mape'] = 0.0

        # Additional statistics
        metrics['mean_true'] = np.mean(y_true)
        metrics['mean_pred'] = np.mean(y_pred)
        metrics['std_true'] = np.std(y_true)
        metrics['std_pred'] = np.std(y_pred)

        # Correlation
        metrics['correlation'] = np.corrcoef(y_true, y_pred)[0, 1]

        return metrics

    def print_metrics(self, metrics: Dict) -> None:
        """Print evaluation metrics in a formatted way.

        Args:
            metrics: Dictionary of metrics from evaluate().
        """
        print("\n" + "="*70)
        print("REGRESSION EVALUATION RESULTS")
        print("="*70)

        print("\n--- Error Metrics ---")
        print(f"MAE (Mean Absolute Error):  {metrics['mae']:.6f}")
        print(f"MSE (Mean Squared Error):   {metrics['mse']:.6f}")
        print(f"RMSE (Root Mean Squared):   {metrics['rmse']:.6f}")
        print(f"MAPE (Mean Abs % Error):    {metrics['mape']:.2f}%")

        print("\n--- Goodness of Fit ---")
        print(f"R² Score:                   {metrics['r2']:.4f}")
        print(f"Correlation:                {metrics['correlation']:.4f}")

        print("\n--- Direction Accuracy ---")
        print(f"Sign Accuracy:              {metrics['sign_accuracy']:.2%}")

        print("\n--- Residual Statistics ---")
        print(f"Mean Residual:              {metrics['mean_residual']:.6f}")
        print(f"Std Residual:               {metrics['std_residual']:.6f}")

        print("\n--- Distribution Statistics ---")
        print(f"Mean (True):                {metrics['mean_true']:.6f} ({metrics['mean_true']*100:.2f}%)")
        print(f"Mean (Predicted):           {metrics['mean_pred']:.6f} ({metrics['mean_pred']*100:.2f}%)")
        print(f"Std (True):                 {metrics['std_true']:.6f}")
        print(f"Std (Predicted):            {metrics['std_pred']:.6f}")

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
            f.write("REGRESSION EVALUATION RESULTS\n")
            f.write("="*70 + "\n\n")

            f.write("--- Error Metrics ---\n")
            f.write(f"MAE (Mean Absolute Error):  {metrics['mae']:.6f}\n")
            f.write(f"MSE (Mean Squared Error):   {metrics['mse']:.6f}\n")
            f.write(f"RMSE (Root Mean Squared):   {metrics['rmse']:.6f}\n")
            f.write(f"MAPE (Mean Abs % Error):    {metrics['mape']:.2f}%\n\n")

            f.write("--- Goodness of Fit ---\n")
            f.write(f"R² Score:                   {metrics['r2']:.4f}\n")
            f.write(f"Correlation:                {metrics['correlation']:.4f}\n\n")

            f.write("--- Direction Accuracy ---\n")
            f.write(f"Sign Accuracy:              {metrics['sign_accuracy']:.2%}\n\n")

            f.write("--- Residual Statistics ---\n")
            f.write(f"Mean Residual:              {metrics['mean_residual']:.6f}\n")
            f.write(f"Std Residual:               {metrics['std_residual']:.6f}\n\n")

            f.write("--- Distribution Statistics ---\n")
            f.write(f"Mean (True):                {metrics['mean_true']:.6f} ({metrics['mean_true']*100:.2f}%)\n")
            f.write(f"Mean (Predicted):           {metrics['mean_pred']:.6f} ({metrics['mean_pred']*100:.2f}%)\n")
            f.write(f"Std (True):                 {metrics['std_true']:.6f}\n")
            f.write(f"Std (Predicted):            {metrics['std_pred']:.6f}\n")

            f.write("\n" + "="*70 + "\n")
