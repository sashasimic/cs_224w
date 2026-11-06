"""The GRU history encoder, shared by gat (checkpoint 3) and hetero (checkpoint 4b).

HistoryGRU(n_in, hidden) : (batch, window, n_in) -> (batch, hidden)
Separate instances are used for zone nodes and constraint nodes; the class is the same.
"""
import torch
from torch import nn


class HistoryGRU(nn.Module):
    def __init__(self, n_in: int, hidden: int = 64, layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return h[-1]
