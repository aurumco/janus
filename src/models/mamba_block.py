"""Mamba SSM block wrapper using official mamba-ssm library."""

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba-ssm not installed. Install with: pip install mamba-ssm")


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
            raise ImportError(
                "mamba-ssm is required but not installed. "
                "Install with: pip install mamba-ssm"
            )

        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Mamba block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        x = self.mamba(x)
        x = self.dropout(x)
        return x
