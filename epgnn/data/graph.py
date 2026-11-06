"""Per-day graphs.

zone_graph        the static zone-zone graph (interconnects) with its per-day edge features; owner: checkpoint 3
constraint_graph  constraint nodes of day D + affects / rev_affects PTDF relations; owner: checkpoint 4b

Both return plain numpy so the model files decide how to build torch_geometric Data / HeteroData.
"""
import numpy as np
import pandas as pd

from epgnn.data import features as F
from epgnn.data import tables


def zone_index(zones: list[str]) -> dict[str, int]:
    return {z: i for i, z in enumerate(zones)}


def zone_graph(zones: list[str], shuffled: bool = False, seed: int = 0) -> np.ndarray:
    """Directed edge_index (2, E) over `zones` from the interconnector list. `shuffled` keeps the degree
    count but rewires endpoints at random (the checkpoint-3 control)."""
    zi = zone_index(zones)
    p = tables.edge_pairs()
    p = p[p.from_zone.isin(zi) & p.to_zone.isin(zi)]
    ei = np.array([[zi[a] for a in p.from_zone], [zi[b] for b in p.to_zone]], dtype="int64")
    if shuffled:
        # keep every zone's out-degree, randomise where the edges go
        ei[1] = np.random.default_rng(seed).permutation(ei[1])
    return ei


def zone_edge_features(zones: list[str], ts: pd.Series, cols: list[str] | None = None) -> np.ndarray:
    """Edge features for the hours `ts`, aligned with zone_graph(zones): (len(ts), E, len(cols))."""
    cols = cols or F.EDGE_FEATURES
    zi = zone_index(zones)
    e = tables.edges()
    e = e[e.from_zone.isin(zi) & e.to_zone.isin(zi) & e.ts_utc.isin(ts)]
    pairs = tables.edge_pairs()
    pairs = pairs[pairs.from_zone.isin(zi) & pairs.to_zone.isin(zi)]
    wide = e.set_index(["ts_utc", "from_zone", "to_zone"])[cols]
    idx = pd.MultiIndex.from_tuples([(t, a, b) for t in ts for a, b in zip(pairs.from_zone, pairs.to_zone)])
    return wide.reindex(idx).to_numpy(dtype="float32").reshape(len(ts), len(pairs), len(cols))


def constraint_graph(day: str, zones: list[str], ptdf_min: float = 0.005) -> dict:
    """Constraint nodes of delivery day `day` and their PTDF edges to zones.

    Returns {"node_key": (C,) ids, "x": (C, 25, 3) hourly [ram, fmax, ram_fmax] (NaN-padded),
             "affects": (2, E) [zone_idx, constraint_idx], "ptdf": (E,) signed weights}.
    A constraint node is one (line, contingency, direction) for the whole day; jao_cnec_hourly needs a
    stable within-day id for that (`cnec_id`, planned for v1.1). Until then this raises.
    """
    raise NotImplementedError("needs jao_cnec_hourly v1.1 with cnec_id; see the dataset doc")
