"""Checkpoints 3 and 3': GRU + GATv2 on the zone graph, without constraint nodes.

The same model as epgnn.models.hetero with `constraints: none`; kept as a model name so the configs read
naturally. `graph: interconnects` is checkpoint 3, `graph: shuffled` the rewired control.
"""
import numpy as np

from epgnn.models import hetero


def run(cfg: dict, fit_splits: list[str], eval_splits: list[str]) -> dict[str, np.ndarray]:
    out = hetero.run({**cfg, "constraints": "none"}, fit_splits, eval_splits)
    run.info = hetero.run.info  # type: ignore[attr-defined]
    return out
