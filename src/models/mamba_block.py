"""Mamba SSM block wrapper using official mamba-ssm library with PyTorch fallback."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
    _MAMBA_IMPORT_ERROR: Optional[str] = None
except Exception as e:
    MAMBA_AVAILABLE = False
    _MAMBA_IMPORT_ERROR = str(e)


class MinimalMamba(nn.Module):
    """A pure PyTorch implementation of Mamba for CPU/fallback usage.

    This is functionally equivalent to the official Mamba implementation but
    without the hardware-aware selective scan kernel, making it significantly slower
    but compatible with non-CUDA environments.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        bias: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias, device=device, dtype=dtype)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            device=device,
            dtype=dtype,
        )

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, device=device, dtype=dtype
        )

        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, device=device, dtype=dtype)

        # Initialize special parameters A and D
        # A: (d_inner, d_state)
        A = torch.repeat_interleave(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device).unsqueeze(0),
            self.d_inner,
            dim=0,
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias, device=device, dtype=dtype)

        # Initialize dt bias
        dt = torch.exp(
            torch.rand(self.d_inner, device=device) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # Initialize A_log (S4D-Lin initialization)
        self.A_log._no_weight_decay = True
        self.D._no_weight_decay = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape

        # Project input
        x_and_res = self.in_proj(x)  # (B, L, 2*d_inner)
        x_val, res = x_and_res.chunk(2, dim=-1)

        # Conv1d expects (B, C, L)
        x_val = x_val.transpose(1, 2)
        x_val = self.conv1d(x_val)[:, :, :seq_len]
        x_val = x_val.transpose(1, 2)

        x_val = F.silu(x_val)

        # SSM
        y = self.ssm(x_val)

        # Gating
        y = y * F.silu(res)

        return self.out_proj(y)

    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        """Runs the SSM.

        Args:
            x: shape (batch, seq_len, d_inner)
        """
        d_inner, d_state = self.d_inner, self.d_state
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        D = self.D.float()

        # Delta, B, C projections
        # x_dbl shape: (B, L, dt_rank + 2*d_state)
        x_dbl = self.x_proj(x)

        delta, B, C = torch.split(
            x_dbl, [self.dt_rank, d_state, d_state], dim=-1
        )

        delta = F.softplus(self.dt_proj(delta))  # (B, L, d_inner)

        # Discretize
        # dt: (B, L, d_inner)
        # A: (d_inner, d_state)
        # dA = exp(delta * A) -> (B, L, d_inner, d_state)
        dA = torch.exp(torch.einsum('bld,dn->bldn', delta, A))

        # dB = delta * B -> (B, L, d_inner, d_state)
        # B: (B, L, d_state)
        # delta: (B, L, d_inner)
        # We need to broadcast B to d_inner? No, B is (B, L, d_state)
        # Mamba paper: B is input-dependent.
        # usually B is (B, L, d_state).
        # But we need (B, L, d_inner, d_state) for the state update
        # standard discretization: bar_B = (exp(delta A) - I) A^{-1} delta B
        # approximate: bar_B = delta * B
        # We need to repeat B for each d_inner channel?
        # In Mamba, B is shared across d_inner? No, B is (B, L, N).
        # We broadcast B to (B, L, 1, N) and multiply by delta (B, L, D, 1)?
        # Actually ssm parameters are B (batch, L, N), C (batch, L, N).

        dB = torch.einsum('bld,bln->bldn', delta, B)

        # Scan
        # h_t = dA * h_{t-1} + dB * x_t
        # x: (B, L, D)
        # h: (B, L, D, N)

        h = torch.zeros(x.size(0), self.d_inner, self.d_state, device=x.device)
        ys = []

        for t in range(x.size(1)):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            ys.append(h)

        y_stack = torch.stack(ys, dim=1) # (B, L, D, N)

        # y = C * h
        # C: (B, L, N)
        # y: (B, L, D)
        y = torch.einsum('bldn,bln->bld', y_stack, C)

        return y + x * D


class MambaBlock(nn.Module):
    """Wrapper for Mamba block with automatic fallback to pure PyTorch implementation.

    If mamba-ssm is installed and CUDA is available, it uses the optimized implementation.
    Otherwise, it falls back to a pure PyTorch implementation (MinimalMamba).
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
            **kwargs: Additional arguments passed to Mamba implementation.
        """
        super().__init__()

        self.use_fallback = not MAMBA_AVAILABLE

        self.norm = nn.LayerNorm(d_model)

        if not self.use_fallback:
            try:
                self.mamba = Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
            except Exception as e:
                print(f"Warning: Mamba initialization failed ({e}), using fallback.")
                self.use_fallback = True

        if self.use_fallback:
            if not getattr(self, "_logged_fallback", False):
                print("Notice: Using pure PyTorch Mamba fallback (slower, no CUDA req).")
                self._logged_fallback = True

            self.mamba = MinimalMamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                **kwargs
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
