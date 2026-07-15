"""
face_rater.py — Shared FaceRatingMLP architecture.

Centralised here so both training (train_mlp.py) and inference
(scoring.py, predict.py) import from the same definition.
Avoids architecture drift between train and serve.
"""
import torch
import torch.nn as nn


class FaceRatingMLP(nn.Module):
    """
    4-layer MLP: input_dim -> 512 -> 256 -> 64 -> 1

    Design decisions:
    - LayerNorm: per-sample normalisation, stable with any batch size
      and compatible with Dropout (unlike BatchNorm).
    - GELU: smooth gradient, better than ReLU for regression.
    - Dropout: 0.35 / 0.20 for regularisation on small datasets.
    - Final layer xavier-initialised near zero for stable early training.
    """

    def __init__(self, input_dim: int = 427):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.35),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.20),

            nn.Linear(256, 64),
            nn.GELU(),

            nn.Linear(64, 1),
        )
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
