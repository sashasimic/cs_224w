"""Load the pinned model tables, once per process."""
from functools import lru_cache

import pandas as pd

from epgnn.paths import dataset_path


@lru_cache(maxsize=1)
def nodes() -> pd.DataFrame:
    """zone_nodes_hourly: one row per (zone, hour), sorted by zone then ts_utc."""
    df = pd.read_parquet(dataset_path("zone_nodes_hourly", "zone_nodes_hourly.parquet"))
    return df.sort_values(["zone", "ts_utc"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def edges() -> pd.DataFrame:
    """zone_edges_hourly: one row per (from_zone, to_zone, hour)."""
    return pd.read_parquet(dataset_path("zone_edges_hourly", "zone_edges_hourly.parquet"))


@lru_cache(maxsize=1)
def edge_pairs() -> pd.DataFrame:
    """The directed interconnector list: distinct (from_zone, to_zone)."""
    return edges()[["from_zone", "to_zone"]].drop_duplicates().reset_index(drop=True)


@lru_cache(maxsize=1)
def cnec() -> pd.DataFrame:
    """jao_cnec_hourly: one row per physical constraint per hour (constraint nodes). Must be pinned:
    scripts/datasets/datasets.py pull jao_cnec_hourly@v1.0 --pin"""
    try:
        return pd.read_parquet(dataset_path("jao_cnec_hourly", "jao_cnec_hourly.parquet"))
    except KeyError as e:
        raise KeyError("jao_cnec_hourly is not pinned; run: scripts/datasets/datasets.py pull jao_cnec_hourly@v1.0 --pin") from e


@lru_cache(maxsize=1)
def zones() -> list[str]:
    return sorted(nodes().zone.unique())
