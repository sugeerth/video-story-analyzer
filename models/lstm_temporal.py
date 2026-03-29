"""LSTM Temporal Model for sequence modeling across video frames."""

import torch
import torch.nn as nn


class LSTMTemporal(nn.Module):
    """Models temporal dependencies across a sequence of frame-level features."""

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 10,
        bidirectional: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        direction_factor = 2 if bidirectional else 1
        self.output_dim = hidden_dim * direction_factor

        self.attention = nn.Sequential(
            nn.Linear(self.output_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.classifier = nn.Linear(self.output_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Temporal features via attention-weighted LSTM. Input: (B, T, input_dim) -> (B, output_dim)."""
        lstm_out, _ = self.lstm(x)  # (B, T, output_dim)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (B, T, 1)
        context = (lstm_out * attn_weights).sum(dim=1)  # (B, output_dim)
        return self.dropout(context)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, features). Input: (B, T, input_dim)."""
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits, features
