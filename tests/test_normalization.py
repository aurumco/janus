
import pytest
import torch
import torch.nn as nn
from src.models.normalization import RMSNorm
from src.models.mamba_block import MambaBlock

def test_rmsnorm_forward():
    batch_size, seq_len, d_model = 2, 10, 32
    x = torch.randn(batch_size, seq_len, d_model)
    norm = RMSNorm(d_model)
    output = norm(x)

    assert output.shape == x.shape
    assert not torch.isnan(output).any()

    # Check if norm is close to 1 (RMSNorm makes RMS=1)
    # RMS = sqrt(mean(x^2))
    rms = torch.sqrt(output.pow(2).mean(dim=-1))
    # It should be close to 1.0, but remember we multiply by weight (ones)
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)

def test_mamba_block_integration():
    # Mock Mamba availability if needed, but MambaBlock handles import error gracefully-ish
    # We need to mock mamba_ssm.Mamba if not available
    try:
        import mamba_ssm
    except ImportError:
        pytest.skip("mamba_ssm not installed")

    d_model = 32
    block = MambaBlock(d_model=d_model, d_state=16, d_conv=4, expand=2)

    x = torch.randn(2, 10, d_model).cuda() if torch.cuda.is_available() else torch.randn(2, 10, d_model)
    block = block.to(x.device)

    output = block(x)
    assert output.shape == x.shape

    # Verify we are using RMSNorm
    assert isinstance(block.norm, RMSNorm)
