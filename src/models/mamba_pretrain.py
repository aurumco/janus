"""Mamba-based model for SSL pre-training."""

from typing import Dict

import torch
import torch.nn as nn

from .mamba_block import MambaBlock


class MambaPretrainModel(nn.Module):
    """Mamba-based foundation model for self-supervised pre-training."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        d_state: int,
        d_conv: int,
        n_layers: int,
        reconstruction_head_dim: int,
        volatility_head_dim: int = 1,
        dropout: float = 0.1,
        num_assets: int = 15,
        asset_embedding_dim: int = 16,
    ) -> None:
        """Initialize Mamba pre-training model.

        Args:
            input_dim: Input feature dimension.
            d_model: Model dimension.
            d_state: SSM state dimension.
            d_conv: Convolution kernel size.
            n_layers: Number of Mamba blocks.
            reconstruction_head_dim: Output dimension for reconstruction head.
            volatility_head_dim: Output dimension for volatility head.
            dropout: Dropout probability.
            num_assets: Number of unique assets.
            asset_embedding_dim: Asset embedding dimension.
        """
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.num_assets = num_assets
        self.asset_embedding_dim = asset_embedding_dim

        if asset_embedding_dim > 0:
            self.asset_embedding = nn.Embedding(num_assets, asset_embedding_dim)
            total_input_dim = input_dim + asset_embedding_dim
        else:
            self.asset_embedding = None
            total_input_dim = input_dim

        self.input_projection = nn.Linear(total_input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        self.mamba_layers = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(n_layers)]
        )

        self.reconstruction_head = nn.Linear(d_model, reconstruction_head_dim)

        self.volatility_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, volatility_head_dim),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor, asset_ids: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Forward pass through pre-training model.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).
            asset_ids: Asset ID tensor of shape (batch,) if using embeddings.

        Returns:
            Dictionary with reconstructed sequence and predicted volatility.
        """
        if self.asset_embedding is not None and asset_ids is not None:
            asset_emb = self.asset_embedding(asset_ids)
            asset_emb = asset_emb.unsqueeze(1).expand(-1, x.size(1), -1)
            x = torch.cat([x, asset_emb], dim=-1)

        x = self.input_projection(x)
        x = self.input_norm(x)

        for mamba_layer, layer_norm in zip(self.mamba_layers, self.layer_norms):
            residual = x
            x = mamba_layer(x)
            x = layer_norm(x + residual)

        reconstructed_sequence = self.reconstruction_head(x)

        last_hidden = x[:, -1, :]
        predicted_volatility = self.volatility_head(last_hidden)

        return {
            "reconstructed_sequence": reconstructed_sequence,
            "predicted_volatility": predicted_volatility,
        }

    def get_num_parameters(self) -> Dict[str, int]:
        """Get the number of parameters in the model.

        Returns:
            Dictionary with total and trainable parameter counts.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

        return {
            "total": total_params,
            "trainable": trainable_params,
        }
