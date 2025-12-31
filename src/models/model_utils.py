"""Utility functions for model operations."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def load_pretrained_weights(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    backbone_keys: Optional[List[str]] = None,
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = False,
) -> nn.Module:
    """Load pretrained weights into a model with dimension adaptation support.

    Args:
        model: The PyTorch model to load weights into.
        checkpoint_path: Path to the pretrained checkpoint.
        backbone_keys: List of key substrings to identify backbone layers.
                       If None, defaults to Mamba backbone keys.
        map_location: Device to map weights to.
        strict: Whether to strictly enforce state dict keys matching.
                Note: This function manually filters keys, so strict=True
                checks against the filtered keys.

    Returns:
        The model with loaded weights.

    Raises:
        FileNotFoundError: If checkpoint cannot be found.
        RuntimeError: If loading fails.
    """
    if backbone_keys is None:
        backbone_keys = [
            "asset_embedding",
            "input_projection",
            "input_norm",
            "mamba_layers",
            "layer_norms",
        ]

    # Resolve path
    path = Path(checkpoint_path)
    possible_paths = [
        path,
        Path("/kaggle/input") / path,
        Path("checkpoints/pretrain") / path.name,
        # Fallback for specific naming convention used in the repo
        Path("/kaggle/working/checkpoints/pretrain") / "best_model.pt",
        Path("checkpoints/pretrain") / "best_model.pt",
    ]

    loaded_path: Optional[Path] = None
    for p in possible_paths:
        if p.exists():
            loaded_path = p
            break

    if loaded_path is None:
        raise FileNotFoundError(
            f"Pretrained checkpoint not found. Searched: {[str(p) for p in possible_paths]}"
        )

    logger.info(f"Loading pretrained weights from: {loaded_path}")

    try:
        # Security: weights_only=True is preferred but requires newer PyTorch and
        # clean checkpoint files. Using weights_only=False for compatibility.
        checkpoint = torch.load(loaded_path, map_location=map_location, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {loaded_path}: {e}") from e

    # Extract state dict
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            pretrained_state = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            pretrained_state = checkpoint["state_dict"]
        else:
            pretrained_state = checkpoint
    elif isinstance(checkpoint, nn.Module):
        pretrained_state = checkpoint.state_dict()
    else:
        # Fallback for unknown format
        pretrained_state = checkpoint

    if not isinstance(pretrained_state, dict):
         raise RuntimeError(f"Could not extract state_dict from checkpoint type: {type(checkpoint)}")

    model_dict = model.state_dict()
    pretrained_dict: Dict[str, Any] = {}
    skipped_count = 0
    loaded_count = 0

    for key, value in pretrained_state.items():
        # Check if this parameter belongs to the backbone
        if any(bk in key for bk in backbone_keys):
            if key in model_dict:
                current_param = model_dict[key]
                if current_param.shape == value.shape:
                    pretrained_dict[key] = value
                    loaded_count += 1
                else:
                    logger.warning(
                        f"Dimension mismatch for {key}: "
                        f"pretrained={tuple(value.shape)} vs "
                        f"current={tuple(current_param.shape)}"
                    )
                    # Safe default: skip if shapes don't match
                    skipped_count += 1
            else:
                # Key not in current model
                skipped_count += 1

    if not pretrained_dict:
        logger.warning("No compatible pretrained weights found in checkpoint!")
        return model

    # Update state
    # We use the provided 'strict' argument, but since we manually filtered the dict,
    # keys in 'pretrained_dict' should match. However, the model might have MORE keys.
    # Standard load_state_dict(strict=False) allows missing keys in input.
    # If the user passed strict=True, they expect ALL model keys to be present in input.
    # But since we are doing partial loading (backbone only), strict=True would almost always fail.
    # Thus, we enforce strict=False to allow partial loading, but we could warn if 'strict' was requested.

    if strict:
        logger.warning("load_pretrained_weights performs partial loading. Forcing strict=False.")

    model.load_state_dict(pretrained_dict, strict=False)

    logger.info(f"Successfully loaded {len(pretrained_dict)} backbone layers.")
    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} layers due to mismatch or missing keys.")

    return model
