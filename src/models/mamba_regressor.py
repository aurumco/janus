"""Mamba-based regressor for Bitcoin price change prediction."""

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from .mamba_block import MambaBlock


class MambaRegressor(nn.Module):
    """Mamba-based sequence regressor for continuous price change prediction."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        d_state: int,
        d_conv: int,
        n_layers: int,
        output_dim: int = 1,
        dropout: float = 0.1,
        pretrained_checkpoint_path: Optional[str] = None,
    ) -> None:
        """Initialize Mamba regressor.

        Args:
            input_dim: Input feature dimension.
            d_model: Model dimension.
            d_state: SSM state dimension.
            d_conv: Convolution kernel size.
            n_layers: Number of Mamba blocks.
            output_dim: Output dimension (1 for single value regression).
            dropout: Dropout probability.
            pretrained_checkpoint_path: Path to pretrained model checkpoint.
        """
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim

        self.input_projection = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        self.mamba_layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])

        # Regression head: outputs continuous value
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim),
        )

        if pretrained_checkpoint_path:
            self._load_pretrained_weights(pretrained_checkpoint_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the regressor.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).

        Returns:
            Predictions tensor of shape (batch, output_dim).
        """
        x = self.input_projection(x)
        x = self.input_norm(x)

        for mamba_layer, layer_norm in zip(self.mamba_layers, self.layer_norms):
            residual = x
            x = mamba_layer(x)
            x = layer_norm(x + residual)

        # Use last sequence position for prediction
        x = x[:, -1, :]

        prediction = self.regression_head(x)

        return prediction

    def get_num_parameters(self) -> Dict[str, int]:
        """Get the number of parameters in the model.

        Returns:
            Dictionary with total and trainable parameter counts.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total": total_params,
            "trainable": trainable_params,
        }

    def _load_pretrained_weights(self, checkpoint_path: str) -> None:
        """Load pretrained weights from pre-training checkpoint.

        Args:
            checkpoint_path: Path to pretrained model checkpoint.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            print(f"Warning: Pretrained checkpoint not found: {checkpoint_path}")
            return

        print(f"Loading pretrained weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if "model_state_dict" in checkpoint:
            pretrained_state = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            pretrained_state = checkpoint["state_dict"]
        else:
            pretrained_state = checkpoint

        if hasattr(pretrained_state, "module"):
            pretrained_state = pretrained_state.module.state_dict()

        backbone_keys = [
            "input_projection",
            "input_norm",
            "mamba_layers",
            "layer_norms",
        ]

        model_dict = self.state_dict()
        pretrained_dict = {}

        for key, value in pretrained_state.items():
            if any(bk in key for bk in backbone_keys):
                if key in model_dict and model_dict[key].shape == value.shape:
                    pretrained_dict[key] = value
                    print(f"  ✓ Loaded: {key}")
                else:
                    print(f"  ✗ Skipped: {key} (shape mismatch or not found)")

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=False)

        print(f"Successfully loaded {len(pretrained_dict)} pretrained layers")
