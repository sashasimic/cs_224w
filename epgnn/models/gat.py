"""Checkpoints 3, 3', 4a: GRU + GATv2 on the zone graph.

One graph per delivery day: zone nodes with HistoryGRU summaries + day-D inputs, interconnector edges
(epgnn.data.graph.zone_graph; `graph: shuffled` for the control) with EDGE_FEATURES, GATv2Conv layers,
MLP head -> H hourly residual targets per zone (epgnn.data.scaling.residual_target), masked loss +
spread term (epgnn.train).

Config keys: feature_set ("own" | "own+jao" for 4a), graph ("interconnects" | "shuffled"), history_window,
hidden, layers, heads, epochs, lr, pretrain ({splits: [pretrain]} then fine-tune on train), seed.
"""
import numpy as np


def run(cfg: dict, fit_splits: list[str], eval_splits: list[str]) -> dict[str, np.ndarray]:
    raise NotImplementedError("checkpoint 3: see module docstring; owner B")
