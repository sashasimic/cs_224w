"""Checkpoint 4b: the full heterogeneous model.

Node types: zone (as in gat.py) and constraint (one per (line, contingency, direction) of day D, from
epgnn.data.graph.constraint_graph, HistoryGRU over its hourly [ram, fmax, ram_fmax]).
Relations: zone-zone "interconnects" (as in gat.py), zone->constraint "affects" and constraint->zone
"rev_affects" with the signed PTDF as edge feature. torch_geometric HeteroConv with one GATv2Conv per
relation; same head, loss and schedule as gat.py (pretrain zone-only, then fine-tune with constraints
on the Core period). Non-Core zones and pre-2022-06-09 days simply have no constraint edges.

Config keys: as gat.py plus ptdf_min (default 0.005), constraint_hidden.
"""
import numpy as np


def run(cfg: dict, fit_splits: list[str], eval_splits: list[str]) -> dict[str, np.ndarray]:
    raise NotImplementedError("checkpoint 4b: see module docstring; owner C")
