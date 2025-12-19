"""Mamba-based classifier for Bitcoin trend prediction."""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .mamba_block import MambaBlock


class MambaClassifier(nn.Module):
    """Mamba-based sequence classifier for multi-class classification."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        d_state: int,
        d_conv: int,
        n_layers: int,
        num_classes: int,
        dropout: float = 0.1,
        num_assets: int = 15,
        asset_embedding_dim: int = 32,
    ) -> None:
        """Initialize Mamba classifier.

        Args:
            input_dim: Input feature dimension.
            d_model: Model dimension.
            d_state: SSM state dimension.
            d_conv: Convolution kernel size.
            n_layers: Number of Mamba blocks.
            num_classes: Number of output classes.
            dropout: Dropout probability.
            num_assets: Number of unique assets.
            asset_embedding_dim: Asset embedding dimension.
        """
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.num_classes = num_classes
        self.num_assets = num_assets
        self.asset_embedding_dim = asset_embedding_dim

        if asset_embedding_dim > 0:
            self.asset_embedding = nn.Embedding(num_assets, asset_embedding_dim)
            self.asset_projection = nn.Linear(asset_embedding_dim, d_model, bias=False)
        else:
            self.asset_embedding = None
            self.asset_projection = None

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

        self.classifier_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor, asset_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the classifier.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).
            asset_ids: Asset ID tensor of shape (batch,) if using embeddings.

        Returns:
            Logits tensor of shape (batch, num_classes).
        """
        # Project inputs: (batch, seq_len, input_dim) -> (batch, seq_len, d_model)
        x_proj = self.input_projection(x)

        # Add asset embedding projection if available
        if self.asset_embedding is not None and asset_ids is not None:
            # (batch, asset_dim)
            asset_emb = self.asset_embedding(asset_ids)
            # (batch, d_model)
            asset_proj = self.asset_projection(asset_emb)
            # Broadcast add: (batch, seq_len, d_model) + (batch, 1, d_model)
            x = x_proj + asset_proj.unsqueeze(1)
        else:
            x = x_proj

        x = self.input_norm(x)

        for mamba_layer, layer_norm in zip(self.mamba_layers, self.layer_norms):
            residual = x
            x = mamba_layer(x)
            x = layer_norm(x + residual)

        x = x[:, -1, :]

        logits = self.classifier_head(x)

        return logits

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
