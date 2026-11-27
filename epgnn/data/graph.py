"""Per-day graphs.

zone_graph            the static zone-zone graph (interconnects) with its per-day edge features; checkpoint 3
rewire                the shuffled control graph: same number of links per zone, random partners
laplacian_pe          Laplacian-eigenvector positional encoding of the zones (config zone_pe)
edge_inputs           lagged flows on the links, per delivery hour (config edge_feats: flows)
constraint_graph      constraint nodes of day D + affects / rev_affects PTDF relations; owner: checkpoint 4b (stub)

Both return plain numpy so the model files decide how to build the torch_geometric batch.
"""
import numpy as np
import pandas as pd

HOUR_NS = 3_600_000_000_000

from epgnn.data import features as F
from epgnn.data import tables


def zone_index(zones: list[str]) -> dict[str, int]:
    return {z: i for i, z in enumerate(zones)}


def zone_graph(zones: list[str], shuffled: bool = False, seed: int = 0) -> np.ndarray:
    """Directed edge_index (2, E) over `zones` from the interconnector list: every link in both directions.
    `shuffled` is the control graph: the links are rewired at random (see rewire), so only *which* zones are
    connected changes."""
    zi = zone_index(zones)
    p = tables.edge_pairs()
    p = p[p.from_zone.isin(zi) & p.to_zone.isin(zi)]
    ei = np.array([[zi[a] for a in p.from_zone], [zi[b] for b in p.to_zone]], dtype="int64")
    return rewire(ei, seed) if shuffled else ei


def rewire(ei: np.ndarray, seed: int = 0, swaps_per_link: int = 10) -> np.ndarray:
    """Random graph with the same number of links at every node as `ei` (double-edge swaps: links a-b and
    c-d become a-d and c-b). Links stay two-way, and no self-link or duplicate is created, so the result
    differs from the real graph only in which nodes are connected."""
    links = sorted({(min(a, b), max(a, b)) for a, b in ei.T.tolist()})
    assert 2 * len(links) == ei.shape[1] and all(a != b for a, b in links), "expects every link once per direction"
    rng = np.random.default_rng(seed)
    have, done, tries, target = set(links), 0, 0, swaps_per_link * len(links)
    while done < target and tries < 200 * target:
        tries += 1
        i, j = rng.choice(len(links), 2, replace=False)
        (a, b), (c, d) = links[i], links[j]
        if rng.random() < 0.5:
            c, d = d, c
        new1, new2 = (min(a, d), max(a, d)), (min(c, b), max(c, b))
        if len({a, b, c, d}) < 4 or new1 in have or new2 in have:
            continue
        have -= {links[i], links[j]}
        have |= {new1, new2}
        links[i], links[j] = new1, new2
        done += 1
    one_way = np.array(sorted(have), dtype="int64").T
    return np.concatenate([one_way, one_way[::-1]], axis=1)


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


def edge_inputs(zones: list[str], hours: np.ndarray, shuffled: bool = False, seed: int = 0) -> np.ndarray:
    """Link features for the delivery hours `hours` (int64 ns), aligned with zone_graph(zones, shuffled, seed):
    (len(hours), E, 2 * len(F.EDGE_KNOWN_AT_CUTOFF)), float32, NaN where the source has no value.

    For the directed link a -> b the first half is the lagged flow a -> b and the second half the lagged flow
    b -> a, both as log1p(MW). All columns are lags that are known at the cutoff (features.EDGE_KNOWN_AT_CUTOFF).
    On the 25-hour day of the autumn clock change the last hour is 24 elapsed hours after the first, so its
    flow_sched_lag24 would be that day's own schedule, an auction outcome; it takes the previous hour's value.

    With `shuffled` the graph is the rewired control. Its links do not exist physically, so each one is given
    the whole feature series of one real link, drawn without replacement: the link data stays, only its
    attachment to zones is destroyed.
    """
    cols = F.EDGE_KNOWN_AT_CUTOFF
    zi = zone_index(zones)
    real = zone_graph(zones)
    n_e = real.shape[1]
    slot = np.full((len(zones), len(zones)), -1, dtype="int64")
    slot[real[0], real[1]] = np.arange(n_e)
    e = tables.edges()
    e = e[e.from_zone.isin(zi) & e.to_zone.isin(zi)]
    t = pd.DatetimeIndex(e.ts_utc).tz_convert("UTC").as_unit("ns").asi8
    all_hours = np.unique(t)
    k = slot[e.from_zone.map(zi).to_numpy(), e.to_zone.map(zi).to_numpy()]
    cube = np.full((len(all_hours), n_e, len(cols)), np.nan, dtype="float32")
    cube[np.searchsorted(all_hours, t)[k >= 0], k[k >= 0]] = e[cols].to_numpy(dtype="float32")[k >= 0]
    if "flow_sched_lag24" in cols:
        local = pd.DatetimeIndex(all_hours, tz="UTC").tz_convert("Europe/Brussels")
        earlier = pd.DatetimeIndex(all_hours - 24 * HOUR_NS, tz="UTC").tz_convert("Europe/Brussels")
        same_day = np.flatnonzero(local.normalize() == earlier.normalize())      # last hour of a 25-hour day
        same_day = same_day[same_day > 0]
        j = cols.index("flow_sched_lag24")
        cube[same_day, :, j] = cube[same_day - 1, :, j]
    both = np.log1p(np.clip(np.concatenate([cube, cube[:, slot[real[1], real[0]]]], axis=2), 0, None))
    take = np.arange(n_e)
    if shuffled:
        links = np.array(sorted({(min(a, b), max(a, b)) for a, b in real.T.tolist()}), dtype="int64")
        perm = np.random.default_rng([seed, 1]).permutation(len(links))
        take = np.concatenate([slot[links[perm, 0], links[perm, 1]], slot[links[perm, 1], links[perm, 0]]])
    rows = np.searchsorted(all_hours, hours).clip(max=len(all_hours) - 1)
    out = both[rows][:, take]
    out[all_hours[rows] != hours] = np.nan
    return out


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
