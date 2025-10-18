"""Evaluator for multi-task regression models."""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class MultiTaskEvaluator:
    """Evaluates multi-task regression models."""

    def __init__(self, device: torch.device) -> None:
        """Initialize evaluator.

        Args:
            device: Device to run evaluation on.
        """
        self.device = device

    @torch.no_grad()
    def evaluate(
        self, 
        model: nn.Module, 
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate model on dataset.

        Args:
            model: Model to evaluate.
            dataloader: Data loader for evaluation.

        Returns:
            Dictionary of evaluation metrics.
        """
        model.eval()
        
        all_predictions = []
        all_targets = []
        all_direction_preds = []
        all_magnitudes = []
        all_confidences = []

        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = model(inputs)
            
            # Extract outputs
            predictions = outputs['regression'].cpu().numpy()
            direction_logits = outputs['direction_logits'].cpu().numpy()
            magnitudes = outputs['magnitude'].cpu().numpy()
            confidences = outputs['confidence'].cpu().numpy()
            
            targets_np = targets.cpu().numpy()

            all_predictions.append(predictions)
            all_targets.append(targets_np)
            all_direction_preds.append(direction_logits)
            all_magnitudes.append(magnitudes)
            all_confidences.append(confidences)

        # Concatenate all batches
        predictions = np.concatenate(all_predictions, axis=0).squeeze()
        targets = np.concatenate(all_targets, axis=0).squeeze()
        direction_logits = np.concatenate(all_direction_preds, axis=0)
        magnitudes = np.concatenate(all_magnitudes, axis=0).squeeze()
        confidences = np.concatenate(all_confidences, axis=0).squeeze()

        # Calculate metrics
        metrics = self._calculate_metrics(
            predictions, 
            targets,
            direction_logits,
            magnitudes,
            confidences
        )

        return metrics

    def _calculate_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        direction_logits: np.ndarray,
        magnitudes: np.ndarray,
        confidences: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate evaluation metrics.

        Args:
            predictions: Model predictions.
            targets: True target values.
            direction_logits: Direction classification logits.
            magnitudes: Magnitude predictions.
            confidences: Confidence scores.

        Returns:
            Dictionary of metrics.
        """
        # Basic regression metrics
        mae = np.mean(np.abs(predictions - targets))
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        
        # MAPE (handle near-zero targets)
        mask = np.abs(targets) > 1e-6
        if mask.sum() > 0:
            mape = np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100
        else:
            mape = float('inf')

        # R² score
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        # Correlation
        correlation = np.corrcoef(predictions, targets)[0, 1]

        # Direction accuracy (sign accuracy)
        pred_signs = np.sign(predictions)
        target_signs = np.sign(targets)
        sign_accuracy = np.mean(pred_signs == target_signs) * 100

        # Direction classification accuracy
        direction_preds = np.argmax(direction_logits, axis=-1)
        direction_targets = self._targets_to_classes(targets)
        direction_class_accuracy = np.mean(direction_preds == direction_targets) * 100

        # Magnitude metrics
        target_magnitudes = np.abs(targets)
        magnitude_mae = np.mean(np.abs(magnitudes - target_magnitudes))
        magnitude_mse = np.mean((magnitudes - target_magnitudes) ** 2)

        # Confidence calibration
        prediction_errors = np.abs(predictions - targets)
        # Check if high confidence correlates with low error
        high_conf_mask = confidences > np.median(confidences)
        if high_conf_mask.sum() > 0 and (~high_conf_mask).sum() > 0:
            high_conf_error = np.mean(prediction_errors[high_conf_mask])
            low_conf_error = np.mean(prediction_errors[~high_conf_mask])
            confidence_ratio = low_conf_error / (high_conf_error + 1e-8)
        else:
            confidence_ratio = 1.0

        # Mean confidence
        mean_confidence = np.mean(confidences)

        # Distribution statistics
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        mean_target = np.mean(targets)
        std_target = np.std(targets)

        return {
            # Regression metrics
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2_score': float(r2),
            'correlation': float(correlation),
            
            # Direction metrics
            'sign_accuracy': float(sign_accuracy),
            'direction_class_accuracy': float(direction_class_accuracy),
            
            # Magnitude metrics
            'magnitude_mae': float(magnitude_mae),
            'magnitude_mse': float(magnitude_mse),
            
            # Confidence metrics
            'mean_confidence': float(mean_confidence),
            'confidence_calibration_ratio': float(confidence_ratio),
            
            # Distribution statistics
            'mean_prediction': float(mean_pred),
            'std_prediction': float(std_pred),
            'mean_target': float(mean_target),
            'std_target': float(std_target),
            'variance_ratio': float(std_pred / (std_target + 1e-8)),
        }

    def _targets_to_classes(self, targets: np.ndarray) -> np.ndarray:
        """Convert targets to direction classes.

        Args:
            targets: Continuous target values.

        Returns:
            Class labels: 0=negative, 1=neutral, 2=positive.
        """
        neutral_threshold = 0.001
        classes = np.ones_like(targets, dtype=np.int64)  # Default to neutral
        classes[targets < -neutral_threshold] = 0
        classes[targets > neutral_threshold] = 2
        return classes

    def print_metrics(self, metrics: Dict[str, float]) -> None:
        """Print formatted metrics.

        Args:
            metrics: Dictionary of evaluation metrics.
        """
        print("\n" + "="*70)
        print("MULTI-TASK REGRESSION EVALUATION RESULTS")
        print("="*70)
        
        print("\n--- Primary Regression Metrics ---")
        print(f"MAE (Mean Absolute Error):  {metrics['mae']:.6f}")
        print(f"MSE (Mean Squared Error):   {metrics['mse']:.6f}")
        print(f"RMSE (Root Mean Squared):   {metrics['rmse']:.6f}")
        print(f"MAPE (Mean Abs % Error):    {metrics['mape']:.2f}%")
        
        print("\n--- Goodness of Fit ---")
        print(f"R² Score:                   {metrics['r2_score']:.4f}")
        print(f"Correlation:                {metrics['correlation']:.4f}")
        
        print("\n--- Direction Prediction ---")
        print(f"Sign Accuracy:              {metrics['sign_accuracy']:.2f}%")
        print(f"Direction Class Accuracy:   {metrics['direction_class_accuracy']:.2f}%")
        
        print("\n--- Magnitude Prediction ---")
        print(f"Magnitude MAE:              {metrics['magnitude_mae']:.6f}")
        print(f"Magnitude MSE:              {metrics['magnitude_mse']:.6f}")
        
        print("\n--- Confidence Calibration ---")
        print(f"Mean Confidence:            {metrics['mean_confidence']:.4f}")
        print(f"Calibration Ratio:          {metrics['confidence_calibration_ratio']:.4f}")
        print(f"  (>1.0 = well calibrated, high conf → low error)")
        
        print("\n--- Distribution Statistics ---")
        print(f"Mean (True):                {metrics['mean_target']:.6f} ({metrics['mean_target']*100:.2f}%)")
        print(f"Mean (Predicted):           {metrics['mean_prediction']:.6f} ({metrics['mean_prediction']*100:.2f}%)")
        print(f"Std (True):                 {metrics['std_target']:.6f}")
        print(f"Std (Predicted):            {metrics['std_prediction']:.6f}")
        print(f"Variance Ratio (Pred/True): {metrics['variance_ratio']:.4f}")
        print("="*70 + "\n")
