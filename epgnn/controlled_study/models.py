"""Classical and graph heads, optionally sharing a temporal GRU encoder."""

import torch
from torch import nn

from epgnn.experiment_suite.neural import PriceGAT


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


class MLP(nn.Module):
    def __init__(self, features, width):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(features, width),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(width, width),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(width, 1),
        )

    def forward(self, x, edges, active):
        return self.layers(x).squeeze(-1)


def matched_width(features):
    # Width is selected by parameter count only, never by validation/test loss.
    with torch.random.fork_rng(devices=[]):
        target = parameter_count(PriceGAT(features))
    return min(
        range(1, 257),
        key=lambda width: abs(width**2 + (features + 3) * width + 1 - target),
    )


class Forecaster(nn.Module):
    def __init__(self, kind, features, history_hidden=32):
        super().__init__()
        self.temporal = kind.startswith("gru_")
        inputs = features + history_hidden if self.temporal else features
        self.encoder = (
            nn.GRU(2, history_hidden, batch_first=True) if self.temporal else None
        )
        self.width = matched_width(inputs) if kind.endswith("mlp") else 64
        self.head = (
            MLP(inputs, self.width) if kind.endswith("mlp") else PriceGAT(inputs)
        )

    def forward(self, x, edges, active, history=None):
        if self.temporal:
            if history is None:
                raise ValueError("Temporal model requires timestamp-aligned history")
            values, inverse = history
            days, zones, window, features = values.shape
            _, state = self.encoder(values.reshape(days * zones, window, features))
            encoded = state[-1].reshape(days, zones, -1)[inverse]
            x = torch.cat([x, encoded], dim=-1)
        return self.head(x, edges, active)


def make_model(kind, features, config):
    if kind not in {"mlp", "gat", "gru_mlp", "gru_gat"}:
        raise ValueError(f"Unknown neural architecture: {kind}")
    return Forecaster(kind, features, config["history_hidden"])
