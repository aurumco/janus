"""Mamba-based regressor for Bitcoin price change prediction."""

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from .mamba_block import MambaBlock
from .normalization import RMSNorm


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
        num_assets: int = 15,
        asset_embedding_dim: int = 32,
        pretrained_checkpoint_path: Optional[str] = None,
        use_gradient_checkpointing: bool = False,
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
            num_assets: Number of unique assets.
            asset_embedding_dim: Asset embedding dimension.
            pretrained_checkpoint_path: Path to pretrained model checkpoint.
            use_gradient_checkpointing: Whether to use gradient checkpointing.
        """
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim
        self.num_assets = num_assets
        self.asset_embedding_dim = asset_embedding_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing

        if asset_embedding_dim > 0:
            self.asset_embedding = nn.Embedding(num_assets, asset_embedding_dim)
            self.asset_projection = nn.Linear(asset_embedding_dim, d_model, bias=False)
        else:
            self.asset_embedding = None
            self.asset_projection = None

        self.input_projection = nn.Linear(input_dim, d_model)
        self.input_norm = RMSNorm(d_model)

        self.mamba_layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                dropout=dropout,
            )
            for _ in range(n_layers)
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

    def forward(self, x: torch.Tensor, asset_ids: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through the regressor.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).
            asset_ids: Asset ID tensor of shape (batch,) if using embeddings.

        Returns:
            Predictions tensor of shape (batch, output_dim).
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

        for mamba_layer in self.mamba_layers:
            if self.training and self.use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    mamba_layer, x, use_reentrant=False
                )
            else:
                x = mamba_layer(x)

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
        """Load pretrained weights with dimension adaptation support.

        Args:
            checkpoint_path: Path to pretrained model checkpoint.
        """
        possible_paths = [
            Path(checkpoint_path),
            Path("/kaggle/input") / checkpoint_path,
            Path("/kaggle/working/checkpoints/pretrain") / "best_model.pt",
            Path("checkpoints/pretrain") / "best_model.pt",
        ]

        loaded_path = None
        for path in possible_paths:
            if path.exists():
                loaded_path = path
                break

        if loaded_path is None:
            print(f"Warning: No pretrained checkpoint found at {checkpoint_path}")
            return

        print(f"\n{'='*60}")
        print(f"Loading pretrained weights from: {loaded_path}")
        print(f"{'='*60}")

        try:
            checkpoint = torch.load(loaded_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return

        if "model_state_dict" in checkpoint:
            pretrained_state = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            pretrained_state = checkpoint["state_dict"]
        else:
            pretrained_state = checkpoint

        if isinstance(pretrained_state, torch.nn.Module):
            pretrained_state = pretrained_state.state_dict()

        backbone_keys = [
            "asset_embedding",
            "input_projection",
            "input_norm",
            "mamba_layers",
            "layer_norms",
        ]

        model_dict = self.state_dict()
        pretrained_dict = {}
        skipped_count = 0
        adapted_count = 0

        for key, value in pretrained_state.items():
            if any(bk in key for bk in backbone_keys):
                if key in model_dict:
                    if model_dict[key].shape == value.shape:
                        pretrained_dict[key] = value
                        print(f"  ✓ Loaded: {key} {tuple(value.shape)}")
                    else:
                        print(
                            f"  ⚠ Dimension mismatch: {key} "
                            f"pretrained={tuple(value.shape)} vs "
                            f"current={tuple(model_dict[key].shape)}"
                        )
                        
                        if "input_projection" in key or "input_norm" in key:
                            if value.shape[0] != model_dict[key].shape[0]:
                                print(f"    → Cannot adapt input dimensions, skipping")
                                skipped_count += 1
                            else:
                                pretrained_dict[key] = value
                                adapted_count += 1
                        else:
                            skipped_count += 1
                else:
                    print(f"  ✗ Not found in current model: {key}")
                    skipped_count += 1

        if pretrained_dict:
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)
            
            print(f"\n{'='*60}")
            print(f"✓ Successfully loaded {len(pretrained_dict)} layers")
            if adapted_count > 0:
                print(f"⚠ Adapted {adapted_count} layers with dimension mismatches")
            if skipped_count > 0:
                print(f"✗ Skipped {skipped_count} incompatible layers")
            print(f"{'='*60}\n")
        else:
            print("\nWarning: No compatible pretrained weights found!")
