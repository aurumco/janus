"""Mamba-based regressor for Bitcoin price change prediction.

This module defines the `MambaRegressor` with two interchangeable heads:

- "mlp" (default): the original MLP regression head
- "factorized": factorized head that estimates sign and magnitude separately

It also exposes utility methods used by training-time regularization guards:

- `reinit_head()` to reinitialize the prediction head weights
- `freeze_backbone()` / `unfreeze_backbone()` to control which parts train
"""

from typing import Dict, Optional

import math
import torch
import torch.nn as nn

from .mamba_block import MambaBlock


class MambaRegressor(nn.Module):
    """Mamba-based sequence regressor for continuous price change prediction.

    The architecture consists of an input projection, a stack of Mamba blocks,
    and a configurable prediction head.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        d_state: int,
        d_conv: int,
        n_layers: int,
        output_dim: int = 1,
        dropout: float = 0.1,
        head_type: str = "mlp",
        max_scale: Optional[float] = None,
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
            head_type: Prediction head type, either "mlp" or "factorized".
            max_scale: Optional clamp for factorized scale output.
        """
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim
        self.head_type = head_type.lower()
        self.max_scale = max_scale

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

        # Regression heads
        if self.head_type == "factorized":
            # Separate paths for sign and magnitude with implicit recombination
            self.sign_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )
            self.scale_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )
        else:
            # Default MLP regression head
            self.regression_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, output_dim),
            )

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

        if self.head_type == "factorized":
            # Sign in [-1, 1], Scale positive. Final pred = sign * scale
            sign = torch.tanh(self.sign_head(x))
            scale = torch.nn.functional.softplus(self.scale_head(x))
            if self.max_scale is not None:
                scale = torch.clamp(scale, max=self.max_scale)
            prediction = sign * scale
        else:
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

    # --- Utilities for training-time regularization guards ---

    def reinit_head(self) -> None:
        """Reinitialize parameters of the prediction head.

        Uses Kaiming uniform for linear layers to reset the last mapping,
        which helps when overfitting is detected.
        """
        def _reset_linear(lin: nn.Linear) -> None:
            nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5))
            if lin.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(lin.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(lin.bias, -bound, bound)

        if self.head_type == "factorized":
            for m in self.sign_head.modules():
                if isinstance(m, nn.Linear):
                    _reset_linear(m)
            for m in self.scale_head.modules():
                if isinstance(m, nn.Linear):
                    _reset_linear(m)
        else:
            for m in self.regression_head.modules():
                if isinstance(m, nn.Linear):
                    _reset_linear(m)

    def freeze_backbone(self) -> None:
        """Freeze backbone (everything except prediction head)."""
        modules_to_freeze = [
            self.input_projection,
            self.input_norm,
            *self.mamba_layers,
            *self.layer_norms,
        ]
        for module in modules_to_freeze:
            for p in module.parameters():
                p.requires_grad = False

        # Ensure head stays trainable
        head_modules = (
            list(self.sign_head.parameters()) + list(self.scale_head.parameters())
            if self.head_type == "factorized"
            else list(self.regression_head.parameters())
        )
        for p in head_modules:
            p.requires_grad = True

    def unfreeze_backbone(self) -> None:
        """Unfreeze the entire model for normal training."""
        for p in self.parameters():
            p.requires_grad = True
