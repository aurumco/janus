"""Mamba-based regressor for Bitcoin price change prediction."""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba_block import MambaBlock


class MambaRegressor(nn.Module):
    """Mamba-based regressor for time series prediction.

    Uses state space models for efficient sequence processing.
    Supports both single-output and multi-quantile regression.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        n_layers: int = 4,
        dropout: float = 0.2,
        num_features: int = 13,
        num_classes: int = 1,
        quantiles: List[float] | None = None,
        use_attention: bool = False,
    ) -> None:
        """Initialize MambaRegressor.

        Args:
            d_model: Dimension of the model.
            d_state: State space dimension.
            d_conv: Convolution kernel size.
            n_layers: Number of Mamba layers.
            dropout: Dropout rate.
            num_features: Number of input features.
            num_classes: Number of output classes (1 for regression).
            quantiles: List of quantiles to predict (e.g., [0.1, 0.5, 0.9]).
                      If None, uses standard single-output regression.
            use_attention: Whether to add attention layer before output.
        """
        super().__init__()

        self.d_model = d_model
        self.num_classes = num_classes
        self.quantiles = quantiles if quantiles is not None else [0.5]
        self.use_attention = use_attention
        self.num_quantiles = len(self.quantiles)

        self.input_projection = nn.Linear(num_features, d_model)
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

        if self.use_attention:
            self.attention = nn.MultiHeadAttention(
                embed_dim=d_model,
                num_heads=8,
                dropout=dropout,
            )
            self.attention_norm = nn.LayerNorm(d_model)

        if self.num_quantiles > 1:
            self.quantile_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model // 2, 1),
                )
                for _ in range(self.num_quantiles)
            ])
        else:
            self.output_projection = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features).

        Returns:
            Predictions of shape (batch_size, num_quantiles) for quantile regression,
            or (batch_size, num_classes) for standard regression.
        """
        # Input projection
        x = self.input_projection(x)
        x = self.input_norm(x)

        # Mamba layers
        for mamba_layer, layer_norm in zip(self.mamba_layers, self.layer_norms):
            residual = x
            x = mamba_layer(x)
            x = layer_norm(x + residual)

        # Optional attention
        if self.use_attention:
            attn_out, _ = self.attention(x, x, x)
            x = self.attention_norm(x + attn_out)
        
        # Take last timestep
        x = x[:, -1, :]
        
        # Multi-quantile output
        if self.num_quantiles > 1:
            quantile_preds = [head(x) for head in self.quantile_heads]
            x = torch.cat(quantile_preds, dim=1)  # (batch, num_quantiles)
        else:
            x = self.output_projection(x)

        return x

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
