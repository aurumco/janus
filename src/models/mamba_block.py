"""Mamba SSM block wrapper using official mamba-ssm library."""

import torch
import torch.nn as nn

from .normalization import RMSNorm

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
    _MAMBA_IMPORT_ERROR: str | None = None
except Exception as e:  # ImportError or lower-level load errors (e.g., CUDA libs)
    MAMBA_AVAILABLE = False
    _MAMBA_IMPORT_ERROR = str(e)
    print(
        "Warning: failed to import mamba-ssm. This can be due to it not being installed "
        "or missing CUDA libraries at runtime. Original error: " + _MAMBA_IMPORT_ERROR
    )


class MambaBlock(nn.Module):
    """Wrapper for official Mamba block from mamba-ssm library.

    This uses the optimized CUDA implementation from the official library
    for maximum performance and efficiency.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        **kwargs
    ) -> None:
        """Initialize Mamba block wrapper.

        Args:
            d_model: Model dimension.
            d_state: SSM state expansion factor.
            d_conv: Local convolution width.
            expand: Block expansion factor.
            dropout: Dropout probability.
            **kwargs: Additional arguments (ignored for compatibility).
        """
        super().__init__()

        if not MAMBA_AVAILABLE:
            hint = (
                "mamba-ssm import failed. Ensure it is installed and CUDA runtime libraries are discoverable. "
                "If using conda/mamba, propagate LD_LIBRARY_PATH to include $CONDA_PREFIX/lib. "
                "Original error: " + (_MAMBA_IMPORT_ERROR or "unknown")
            )
            raise ImportError(hint)

        self.norm = RMSNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Mamba block with residual connection.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        x = residual + x
        return x
