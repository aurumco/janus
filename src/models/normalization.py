"""Normalization layers for Mamba models.

Implements RMSNorm as used in the official Mamba architecture and Llama.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py
    and the Llama implementation.
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        """Initialize RMSNorm.

        Args:
            d_model: Input dimension.
            eps: Epsilon for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Normalized tensor.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalization."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return f"{self.weight.shape[0]}, eps={self.eps}"
