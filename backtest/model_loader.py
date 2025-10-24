"""Model loader supporting multiple formats: PyTorch, TorchScript, ONNX."""

from pathlib import Path
from typing import Union, Any
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None


class ModelInferenceWrapper:
    """Unified interface for model inference across formats."""

    def __init__(self, model, model_type: str):
        """Initialize wrapper.

        Args:
            model: Loaded model object.
            model_type: Type of model ('pytorch', 'torchscript', 'onnx').
        """
        self.model = model
        self.model_type = model_type

    def predict(self, x: np.ndarray) -> float:
        """Run inference on input.

        Args:
            x: Input array of shape (sequence_length, num_features).

        Returns:
            Predicted value.
        """
        if self.model_type == 'onnx':
            x_input = x.astype(np.float32).reshape(1, x.shape[0], x.shape[1])
            input_name = self.model.get_inputs()[0].name
            output_name = self.model.get_outputs()[0].name
            result = self.model.run([output_name], {input_name: x_input})
            return float(result[0][0][0])
        
        elif self.model_type in ['pytorch', 'torchscript']:
            if not TORCH_AVAILABLE:
                raise RuntimeError("PyTorch not available")
            
            x_tensor = torch.FloatTensor(x).unsqueeze(0)
            
            if hasattr(self.model, 'device'):
                x_tensor = x_tensor.to(self.model.device)
            
            with torch.no_grad():
                output = self.model(x_tensor)
            return output.item()
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")


def detect_model_format(model_path: Path) -> str:
    """Detect model format from file extension.

    Args:
        model_path: Path to model file.

    Returns:
        Model format: 'pytorch', 'torchscript', or 'onnx'.
    """
    suffix = model_path.suffix.lower()
    
    if suffix == '.onnx':
        return 'onnx'
    elif suffix == '.pt':
        if 'traced' in model_path.stem or 'script' in model_path.stem:
            return 'torchscript'
        return 'pytorch'
    elif suffix == '.pth':
        return 'pytorch'
    else:
        raise ValueError(f"Unknown model format: {suffix}")


def load_onnx_model(model_path: Path) -> Any:
    """Load ONNX model.

    Args:
        model_path: Path to .onnx file.

    Returns:
        ONNX inference session.
    """
    if not ONNX_AVAILABLE:
        raise RuntimeError("onnxruntime not installed. Install with: pip install onnxruntime")
    
    session = ort.InferenceSession(str(model_path))
    return session


def load_pytorch_model(model_path: Path, config) -> Any:
    """Load PyTorch model.

    Args:
        model_path: Path to .pt or .pth file.
        config: Configuration object.

    Returns:
        Loaded PyTorch model.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not installed. Install with: pip install torch")
    
    from src.models.mamba_regressor import MambaRegressor
    from src.utils.helpers import get_device
    
    device = get_device(use_cuda=config.get('device.use_cuda', False))
    
    model = MambaRegressor(
        input_dim=config.get('data.num_features'),
        d_model=config.get('model.d_model'),
        d_state=config.get('model.d_state'),
        d_conv=config.get('model.d_conv'),
        n_layers=config.get('model.n_layers'),
        output_dim=config.get('model.num_classes', 1),
        dropout=config.get('model.dropout'),
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    model.device = device
    
    return model


def load_torchscript_model(model_path: Path) -> Any:
    """Load TorchScript model.

    Args:
        model_path: Path to traced .pt file.

    Returns:
        Loaded TorchScript model.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not installed. Install with: pip install torch")
    
    model = torch.jit.load(str(model_path))
    model.eval()
    return model


def load_model_auto(model_path: Union[str, Path], config=None) -> ModelInferenceWrapper:
    """Automatically load model based on file format.

    Model format priority:
    1. model.onnx - ONNX format (no torch required)
    2. model_traced.pt - TorchScript format
    3. model_state_dict.pth - PyTorch state dict
    4. best_model.pt - PyTorch checkpoint

    Args:
        model_path: Path to model file or directory.
        config: Configuration object (required for PyTorch models).

    Returns:
        ModelInferenceWrapper instance.
    """
    model_path = Path(model_path)
    
    if model_path.is_dir():
        candidates = [
            (model_path / 'model.onnx', 'onnx'),
            (model_path / 'model_traced.pt', 'torchscript'),
            (model_path / 'model_state_dict.pth', 'pytorch'),
            (model_path / 'best_model.pt', 'pytorch'),
        ]
        
        for candidate_path, expected_type in candidates:
            if candidate_path.exists():
                model_path = candidate_path
                break
        else:
            raise FileNotFoundError(f"No model found in {model_path}")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model_type = detect_model_format(model_path)
    
    print(f"Loading {model_type.upper()} model from: {model_path}")
    
    if model_type == 'onnx':
        model = load_onnx_model(model_path)
    elif model_type == 'torchscript':
        model = load_torchscript_model(model_path)
    elif model_type == 'pytorch':
        if config is None:
            raise ValueError("Config required for PyTorch models")
        model = load_pytorch_model(model_path, config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return ModelInferenceWrapper(model, model_type)
