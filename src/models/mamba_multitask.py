"""Multi-task Mamba model for simultaneous direction and magnitude prediction."""

from typing import Dict

import torch
import torch.nn as nn

from mamba_ssm import Mamba


class MambaMultitaskRegressor(nn.Module):
    """Mamba-based multi-task regressor.
    
    Predicts both:
    1. Direction (sign classification): -1, 0, +1
    2. Magnitude (regression): absolute value of change
    
    This architecture prevents mode collapse by explicitly separating
    direction prediction from magnitude estimation.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        n_layers: int = 4,
        dropout: float = 0.3,
        num_features: int = 13,
        sequence_length: int = 64,
    ) -> None:
        """Initialize multi-task Mamba regressor.

        Args:
            d_model: Model dimension.
            d_state: SSM state dimension.
            d_conv: Convolution kernel size.
            n_layers: Number of Mamba layers.
            dropout: Dropout probability.
            num_features: Number of input features.
            sequence_length: Length of input sequences.
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_features = num_features
        self.sequence_length = sequence_length

        # Input projection
        self.input_proj = nn.Linear(num_features, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        
        # Mamba backbone layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'mamba': Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=2,
                ),
                'norm': nn.LayerNorm(d_model),
                'dropout': nn.Dropout(dropout),
            })
            for _ in range(n_layers)
        ])

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Direction classification head
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3),  # 3 classes: negative, neutral, positive
        )

        # Magnitude regression head
        self.magnitude_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Softplus(),  # Ensure positive output
        )
        
        # Confidence head (predicts reliability of magnitude estimate)
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights with Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with multi-task outputs.

        Args:
            x: Input tensor of shape (batch, sequence_length, num_features).

        Returns:
            Dictionary containing:
                - 'regression': Combined signed magnitude prediction
                - 'direction_logits': Raw logits for direction classification
                - 'magnitude': Absolute magnitude prediction
                - 'confidence': Confidence score for prediction
        """
        batch_size = x.size(0)
        
        # Input projection: (batch, seq, features) -> (batch, seq, d_model)
        x = self.input_proj(x)
        x = self.input_norm(x)

        # Pass through Mamba layers
        for layer_dict in self.layers:
            mamba = layer_dict['mamba']
            norm = layer_dict['norm']
            dropout = layer_dict['dropout']
            
            # Mamba expects (batch, seq, d_model)
            residual = x
            x = mamba(x) + residual
            x = norm(x)
            x = dropout(x)

        # Global pooling: (batch, seq, d_model) -> (batch, d_model)
        x = x.transpose(1, 2)  # (batch, d_model, seq)
        x = self.pool(x).squeeze(-1)  # (batch, d_model)

        # Multi-task predictions
        direction_logits = self.direction_head(x)  # (batch, 3)
        magnitude = self.magnitude_head(x)  # (batch, 1)
        confidence = self.confidence_head(x)  # (batch, 1)

        # Convert direction logits to sign
        direction_probs = torch.softmax(direction_logits, dim=-1)
        # [-1, 0, +1] weighted by probabilities
        direction_values = torch.tensor(
            [-1.0, 0.0, 1.0], 
            device=x.device, 
            dtype=x.dtype
        ).view(1, 3)
        predicted_sign = torch.sum(direction_probs * direction_values, dim=-1, keepdim=True)

        # Combine sign and magnitude
        regression = predicted_sign * magnitude * confidence

        return {
            'regression': regression,
            'direction_logits': direction_logits,
            'magnitude': magnitude,
            'confidence': confidence,
        }

    def get_num_parameters(self) -> Dict[str, int]:
        """Get number of parameters in model.

        Returns:
            Dictionary with total and trainable parameter counts.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
        }
