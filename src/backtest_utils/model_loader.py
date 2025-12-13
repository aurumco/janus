"""Model loader wrapper for backtesting."""

import os
from pathlib import Path
from typing import Optional, Union, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None

from src.config.config_loader import ConfigLoader
from src.models.mamba_regressor import MambaRegressor


class ModelInferenceWrapper:
    """Wrapper for model inference supporting different backends."""

    def __init__(
        self,
        model: Union[nn.Module, 'ort.InferenceSession'],
        backend: str = 'pytorch',
        device: str = 'cpu',
        input_window: int = 96,
    ):
        """Initialize wrapper.

        Args:
            model: Loaded model instance.
            backend: 'pytorch', 'torchscript', or 'onnx'.
            device: Device to run inference on.
            input_window: Input sequence length.
        """
        self.model = model
        self.backend = backend
        self.device = device
        self.input_window = input_window

    def predict(self, sequence: np.ndarray) -> float:
        """Predict price change for a single sequence.

        Args:
            sequence: Input sequence of shape (seq_len, n_features).

        Returns:
            Predicted price change.
        """
        if self.backend == 'pytorch':
            return self._predict_pytorch(sequence)
        elif self.backend == 'torchscript':
            return self._predict_torchscript(sequence)
        elif self.backend == 'onnx':
            return self._predict_onnx(sequence)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def predict_batch(self, sequences: np.ndarray) -> np.ndarray:
        """Predict price changes for a batch of sequences.

        Args:
            sequences: Input sequences of shape (batch_size, seq_len, n_features).

        Returns:
            Predicted price changes of shape (batch_size,).
        """
        if self.backend == 'pytorch':
            return self._predict_batch_pytorch(sequences)
        elif self.backend == 'torchscript':
            return self._predict_batch_torchscript(sequences)
        elif self.backend == 'onnx':
            return self._predict_batch_onnx(sequences)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _predict_pytorch(self, sequence: np.ndarray) -> float:
        """Run PyTorch inference."""
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
            output = self.model(x)
            return output.item()

    def _predict_batch_pytorch(self, sequences: np.ndarray) -> np.ndarray:
        """Run PyTorch inference in batch."""
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(sequences, dtype=torch.float32).to(self.device)
            output = self.model(x)
            return output.cpu().numpy().flatten()

    def _predict_torchscript(self, sequence: np.ndarray) -> float:
        """Run TorchScript inference."""
        with torch.no_grad():
            x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
            output = self.model(x)
            return output.item()

    def _predict_batch_torchscript(self, sequences: np.ndarray) -> np.ndarray:
        """Run TorchScript inference in batch."""
        with torch.no_grad():
            x = torch.tensor(sequences, dtype=torch.float32).to(self.device)
            output = self.model(x)
            return output.cpu().numpy().flatten()

    def _predict_onnx(self, sequence: np.ndarray) -> float:
        """Run ONNX inference."""
        ort_inputs = {self.model.get_inputs()[0].name: sequence.astype(np.float32)[np.newaxis, ...]}
        ort_outs = self.model.run(None, ort_inputs)
        return float(ort_outs[0][0])

    def _predict_batch_onnx(self, sequences: np.ndarray) -> np.ndarray:
        """Run ONNX inference in batch."""
        ort_inputs = {self.model.get_inputs()[0].name: sequences.astype(np.float32)}
        ort_outs = self.model.run(None, ort_inputs)
        return ort_outs[0].flatten()


def load_model_auto(checkpoint_path: str, config: ConfigLoader) -> ModelInferenceWrapper:
    """Load model automatically based on file extension.

    Args:
        checkpoint_path: Path to checkpoint file.
        config: Configuration loader.

    Returns:
        ModelInferenceWrapper.
    """
    path = Path(checkpoint_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if path.suffix == '.onnx':
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime not installed")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        session = ort.InferenceSession(str(path), providers=providers)
        return ModelInferenceWrapper(session, backend='onnx', device='cpu', input_window=config.get('data.input_window'))

    elif path.suffix == '.pt' and 'traced' in path.name:
        model = torch.jit.load(str(path), map_location=device)
        return ModelInferenceWrapper(model, backend='torchscript', device=device, input_window=config.get('data.input_window'))

    else:
        # Assume PyTorch checkpoint
        model = MambaRegressor(
            input_dim=config.get('data.num_features'),
            d_model=config.get('model.d_model'),
            d_state=config.get('model.d_state'),
            d_conv=config.get('model.d_conv'),
            n_layers=config.get('model.n_layers'),
            output_dim=config.get('model.output_dim', 1),
            dropout=config.get('model.dropout'),
            pretrained_checkpoint_path=None
        )

        checkpoint = torch.load(str(path), map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        return ModelInferenceWrapper(model, backend='pytorch', device=device, input_window=config.get('data.input_window'))
