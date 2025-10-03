"""Mamba SSM (State Space Model) block implementation."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    """Mamba block implementing selective state space model.

    This implementation is based on the Mamba architecture for efficient
    sequence modeling with linear-time complexity.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        """Initialize Mamba block.

        Args:
            d_model: Model dimension.
            d_state: SSM state dimension.
            d_conv: Local convolution width.
            expand: Expansion factor for inner dimension.
            dt_rank: Rank of delta projection. If None, uses d_model // 16.
            dropout: Dropout probability.
        """
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = dt_rank or max(1, d_model // 16)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Mamba block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        batch, seq_len, _ = x.shape

        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        x_conv = x_inner.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = x_conv.transpose(1, 2)

        x_ssm = F.silu(x_conv)

        x_proj_out = self.x_proj(x_ssm)
        delta, B, C = torch.split(
            x_proj_out,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1
        )

        delta = F.softplus(self.dt_proj(delta))

        A = -torch.exp(self.A_log.float())

        y = self._selective_scan(x_ssm, delta, A, B, C, self.D)

        y = y * F.silu(z)

        output = self.out_proj(y)
        output = self.dropout(output)

        return output

    def _selective_scan(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        """Selective scan operation (simplified implementation).

        Args:
            u: Input tensor (batch, seq_len, d_inner).
            delta: Time step tensor (batch, seq_len, d_inner).
            A: State transition matrix (d_inner, d_state).
            B: Input projection (batch, seq_len, d_state).
            C: Output projection (batch, seq_len, d_state).
            D: Skip connection parameter (d_inner,).

        Returns:
            Output tensor (batch, seq_len, d_inner).
        """
        batch, seq_len, d_inner = u.shape
        d_state = A.shape[1]

        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)

        x = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
        ys = []

        for i in range(seq_len):
            x = deltaA[:, i] * x + deltaB[:, i] * u[:, i].unsqueeze(-1)
            y = torch.einsum('bdn,bn->bd', x, C[:, i])
            ys.append(y)

        y = torch.stack(ys, dim=1)
        y = y + u * D.unsqueeze(0).unsqueeze(0)

        return y
