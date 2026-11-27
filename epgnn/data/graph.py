"""Per-day graphs.

zone_graph            the static zone-zone graph (interconnects) with its per-day edge features; checkpoint 3
laplacian_pe          Laplacian-eigenvector positional encoding of the zones (config zone_pe)
constraint_graph      constraint nodes of day D + affects / rev_affects PTDF relations; owner: checkpoint 4b (stub)

Both return plain numpy so the model files decide how to build the torch_geometric batch.
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


def laplacian_pe(zones: list[str], k: int = 8) -> np.ndarray:
    """Zone positional encoding (lectures 7, 8, 13), (len(zones), k): the eigenvectors of L = D - A of the
    undirected interconnector graph with the k smallest non-zero eigenvalues (the constant one is dropped).
    Row i says where zone i sits: centre vs periphery, and which region. Signs are arbitrary, so the model
    flips them at random in training. Always built from the real graph, also for the shuffled and no-graph
    controls, so those controls still differ from checkpoint 3 only in the messages."""
    ei = zone_graph(zones)
    a = np.zeros((len(zones), len(zones)))
    a[ei[0], ei[1]] = 1.0
    a = np.maximum(a, a.T)
    _, vec = np.linalg.eigh(np.diag(a.sum(1)) - a)
    return vec[:, 1:k + 1].astype("float32")


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
