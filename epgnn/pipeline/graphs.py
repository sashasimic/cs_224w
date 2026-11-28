"""Zone-level graph structure: which zones talk to which.

Three neighbourhoods for the same GATv2 (the attention-scope experiment):
  self           no edges; GATv2's self-loop only (a zone sees itself)
  interconnects  the physical interconnectors of zone_edges_hourly that exist in the current zone layout:
                 174 directed edges (87 links). `active_pairs` drops pairs with no physical or scheduled
                 flow since ACTIVE_SINCE: DE_LU-HU, DE_LU-SI, DE_LU-IT_NORD, which are Austria's borders
                 inherited from the joint DE-AT-LU zone (mapped to DE_LU before Oct 2018)
  full           every ordered pair of distinct zones, 45 x 44 = 1,980 (a Transformer over zones)

Edges are (2, E) arrays of zone indices, row 0 = source, row 1 = destination, without self-loops
(the GATv2 layer adds them). Zone order is tables.zones() (sorted codes) everywhere.
"""
from functools import lru_cache

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from epgnn.data import tables
from epgnn.paths import dataset_path

GRAPHS = ("self", "interconnects", "full")
ACTIVE_SINCE = "2018-10-01"   # first delivery day of the DE-LU / AT split; every pair in use has flows after it


@lru_cache(maxsize=1)
def active_pairs() -> pd.DataFrame:
    """Directed (from_zone, to_zone) pairs of zone_edges_hourly with any physical or scheduled flow since
    ACTIVE_SINCE. Pairs whose flows all predate it belong to a zone layout that no longer exists."""
    t = pq.read_table(dataset_path("zone_edges_hourly", "zone_edges_hourly.parquet"),
                      columns=["from_zone", "to_zone", "flow_phys", "flow_sched"],
                      filters=[("ts_utc", ">=", pd.Timestamp(ACTIVE_SINCE, tz="UTC"))]).to_pandas()
    t = t[t.flow_phys.notna() | t.flow_sched.notna()]
    return t[["from_zone", "to_zone"]].drop_duplicates().sort_values(["from_zone", "to_zone"]).reset_index(drop=True)


def dropped_pairs() -> pd.DataFrame:
    """Pairs of the edge table that active_pairs leaves out (diagnostic)."""
    m = tables.edge_pairs().merge(active_pairs(), how="left", indicator=True)
    return m[m._merge == "left_only"].drop(columns="_merge").reset_index(drop=True)


def zones() -> list[str]:
    return tables.zones()


def interconnect_edges(zone_list: list[str]) -> np.ndarray:
    zi = {z: i for i, z in enumerate(zone_list)}
    p = active_pairs()
    p = p[p.from_zone.isin(zi) & p.to_zone.isin(zi) & (p.from_zone != p.to_zone)]
    return np.array([[zi[a] for a in p.from_zone], [zi[b] for b in p.to_zone]], dtype=np.int64)


def full_edges(n: int) -> np.ndarray:
    src, dst = np.nonzero(~np.eye(n, dtype=bool))
    return np.stack([src, dst]).astype(np.int64)


def edges_for(graph: str, zone_list: list[str]) -> np.ndarray:
    if graph == "self":
        return np.zeros((2, 0), dtype=np.int64)
    if graph == "interconnects":
        return interconnect_edges(zone_list)
    if graph == "full":
        return full_edges(len(zone_list))
    raise ValueError(f"graph must be one of {GRAPHS}, got {graph!r}")


def adjacency(zone_list: list[str]) -> np.ndarray:
    """Symmetric 0/1 interconnector adjacency (Z, Z)."""
    ei = interconnect_edges(zone_list)
    a = np.zeros((len(zone_list), len(zone_list)), dtype=np.float64)
    a[ei[0], ei[1]] = 1.0
    return np.maximum(a, a.T)


def spread_pairs(zone_list: list[str]) -> np.ndarray:
    """Each interconnected pair once, (2, P) with a < b: used by the spread loss and the spread metric."""
    a = np.triu(adjacency(zone_list), k=1)
    i, j = np.nonzero(a)
    return np.stack([i, j]).astype(np.int64)


def laplacian_pe(zone_list: list[str], k: int = 8) -> np.ndarray:
    """Eigenvectors 2..k+1 of L = D - A on the interconnector graph, (Z, k). Signs are arbitrary
    (flip them at random during training). Ties and disconnected parts make the basis non-unique; this
    is the usual graph-transformer positional encoding, not an identity."""
    a = adjacency(zone_list)
    lap = np.diag(a.sum(1)) - a
    vals, vecs = np.linalg.eigh(lap)
    return vecs[:, 1:k + 1].astype(np.float32)


def components(zone_list: list[str]) -> list[list[str]]:
    """Connected components of the interconnector graph (diagnostic)."""
    a = adjacency(zone_list)
    seen, comps = set(), []
    for s in range(len(zone_list)):
        if s in seen:
            continue
        stack, comp = [s], []
        seen.add(s)
        while stack:
            u = stack.pop()
            comp.append(zone_list[u])
            for v in np.flatnonzero(a[u]):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(sorted(comp))
    return comps
