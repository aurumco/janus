"""Utility helper functions."""

import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(use_cuda: bool = True, device_id: int = 0) -> torch.device:
    """Get the appropriate device for training.

    Args:
        use_cuda: Whether to use CUDA if available.
        device_id: CUDA device ID.

    Returns:
        PyTorch device object.
    """
    if use_cuda and torch.cuda.is_available():
        device = torch.device(f'cuda:{device_id}')
        print(f"Using device: {device} ({torch.cuda.get_device_name(device_id)})")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device}")

    return device


def save_model_architecture(model: torch.nn.Module, output_path: Path) -> None:
    """Save model architecture summary to file.

    Args:
        model: PyTorch model.
        output_path: Path to save the architecture file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MODEL ARCHITECTURE\n")
        f.write("="*70 + "\n\n")
        f.write(str(model))
        f.write("\n\n" + "="*70 + "\n")
        f.write("PARAMETER COUNT\n")
        f.write("="*70 + "\n")

        params = model.get_num_parameters()
        f.write(f"Total parameters: {params['total']:,}\n")
        f.write(f"Trainable parameters: {params['trainable']:,}\n")
        f.write("="*70 + "\n")


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary with parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable,
    }
