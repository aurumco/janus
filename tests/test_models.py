"""Unit tests for model architectures."""

import sys
from unittest.mock import MagicMock
import torch
import torch.nn as nn
import pytest

# Mock mamba_ssm
mamba_ssm_mock = MagicMock()
class MockMamba(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
    def forward(self, x):
        return self.linear(x)

mamba_ssm_mock.Mamba = MockMamba
sys.modules["mamba_ssm"] = mamba_ssm_mock

from src.models.mamba_pretrain import MambaPretrainModel
from src.models.mamba_regressor import MambaRegressor


@pytest.fixture
def device():
    """Fixture to get available device (CUDA if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_mamba_pretrain_model_forward(device):
    """Test MambaPretrainModel forward pass."""
    batch_size = 4
    seq_len = 256
    input_dim = 16
    
    model = MambaPretrainModel(
        input_dim=input_dim,
        d_model=128,
        d_state=16,
        d_conv=4,
        n_layers=2,
        reconstruction_head_dim=16,
        volatility_head_dim=1,
        num_assets=15,
        asset_embedding_dim=32,
    ).to(device)
    
    x = torch.randn(batch_size, seq_len, input_dim).to(device)
    asset_ids = torch.randint(0, 15, (batch_size,)).to(device)
    
    outputs = model(x, asset_ids)
    
    assert "reconstructed_sequence" in outputs
    assert "predicted_volatility" in outputs
    assert outputs["reconstructed_sequence"].shape == (batch_size, seq_len, 16)
    assert outputs["predicted_volatility"].shape == (batch_size, 1)
    assert outputs["reconstructed_sequence"].device.type == device.type


def test_mamba_regressor_forward(device):
    """Test MambaRegressor forward pass."""
    batch_size = 4
    seq_len = 96
    input_dim = 16
    
    model = MambaRegressor(
        input_dim=input_dim,
        d_model=128,
        d_state=16,
        d_conv=4,
        n_layers=2,
        output_dim=1,
    ).to(device)
    
    x = torch.randn(batch_size, seq_len, input_dim).to(device)
    
    outputs = model(x)
    
    assert outputs.shape == (batch_size, 1)
    assert outputs.device.type == device.type


def test_model_parameter_count():
    """Test get_num_parameters method."""
    model = MambaRegressor(
        input_dim=16,
        d_model=64,
        d_state=16,
        d_conv=4,
        n_layers=2,
        output_dim=1,
    )
    
    params = model.get_num_parameters()
    
    assert "total" in params
    assert "trainable" in params
    assert params["total"] > 0
    assert params["trainable"] == params["total"]


def test_gradient_checkpointing(device):
    """Test gradient checkpointing flag."""
    model = MambaPretrainModel(
        input_dim=16,
        d_model=64,
        d_state=16,
        d_conv=4,
        n_layers=2,
        reconstruction_head_dim=16,
        use_gradient_checkpointing=True,
    ).to(device)
    
    assert model.use_gradient_checkpointing is True
    
    model.eval()
    x = torch.randn(2, 128, 16).to(device)
    asset_ids = torch.zeros(2, dtype=torch.long).to(device)
    outputs = model(x, asset_ids)
    
    assert outputs is not None
