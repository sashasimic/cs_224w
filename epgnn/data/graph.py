"""Per-day graphs.

zone_graph            the static zone-zone graph (interconnects) with its per-day edge features; checkpoint 3
constraint_snapshots  constraint nodes per delivery hour with their PTDF coefficients; checkpoint 4b

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


HOUR_NS = 3_600_000_000_000
CONSTRAINT_FEATURES = ["log_ram", "log_fmax", "ram_fmax", "is_core"]


def constraint_snapshots(hours: np.ndarray, zones: list[str], max_constraints: int = 64,
                         lag_hours: int = 0) -> dict:
    """Constraint nodes for each delivery hour in `hours` (sorted unique int64 ns).

    One node per physical constraint (row of jao_cnec_hourly) of the hour; with `lag_hours` > 0 the rows of
    the hour that long before delivery are used instead (the safe choice if the day-D domain turned out
    to be published after the cutoff). Only the `max_constraints` tightest rows (lowest ram/fmax) of an
    hour are kept: the proposal's attention should look at the constraints near their limit, and the
    tail of slack lines only adds cost.

    Returns CSR-style arrays, rows grouped by hour in the order of `hours`:
      offsets  (n_hours + 1,)  rows of hour i are offsets[i]:offsets[i+1]
      x        (C, 4) float32  CONSTRAINT_FEATURES: log1p(ram), log1p(fmax), ram/fmax clipped to [0, 2], Core flag
      coef     (C, 12) float32 signed PTDF per epgnn.jao.ZONES; NaN where the zone is outside the domain
      core_idx (12,) int64     index of each PTDF zone in `zones`, -1 if that zone is not modelled
    """
    from epgnn import jao
    df = tables.cnec()
    key = pd.DatetimeIndex(df.ts_utc).tz_convert("UTC").as_unit("ns").asi8     # the table stores microseconds
    need = hours - lag_hours * HOUR_NS
    cols = [f"ptdf_{z}" for z in jao.ZONES]
    sub = df.loc[np.isin(key, need), ["ram", "fmax", "ram_fmax", "regime"] + cols].copy()
    sub["t"] = key[np.isin(key, need)]
    sub = sub.sort_values(["t", "ram_fmax"]).groupby("t").head(max_constraints)
    t = sub.t.to_numpy()
    counts = np.searchsorted(t, need, side="right") - np.searchsorted(t, need, side="left")
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype("int64")
    x = np.stack([np.log1p(sub.ram.clip(lower=0)), np.log1p(sub.fmax.clip(lower=0)),
                  sub.ram_fmax.clip(0, 2), sub.regime.eq("Core").astype(float)], axis=1).astype("float32")
    coef = sub[cols].to_numpy(dtype="float32")
    core_idx = np.array([zones.index(z) if z in zones else -1 for z in jao.ZONES], dtype="int64")
    return {"offsets": offsets, "x": x, "coef": coef, "core_idx": core_idx}
