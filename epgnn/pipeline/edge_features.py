"""Engineered (from_zone, to_zone, hour) columns on top of zone_edges_hourly."""
import numpy as np
import pandas as pd

CORE = {"AT", "BE", "CZ", "DE_LU", "FR", "HR", "HU", "NL", "PL", "RO", "SI", "SK"}


def build(edges: pd.DataFrame) -> pd.DataFrame:
    e = edges.sort_values(["from_zone", "to_zone", "ts_utc"]).reset_index(drop=True).copy()
    g = e.groupby(["from_zone", "to_zone"], sort=False)
    assert g.size().nunique() == 1, "edge table is not a complete (pair, hour) grid"
    e["maxbex_lag24"] = g.maxbex.shift(24).astype("float32")
    rev = e[["ts_utc", "from_zone", "to_zone", "flow_phys_lag48"]].rename(
        columns={"from_zone": "to_zone", "to_zone": "from_zone", "flow_phys_lag48": "_rev"})
    e = e.merge(rev, on=["ts_utc", "from_zone", "to_zone"], how="left")
    e["net_flow_lag48"] = (e.flow_phys_lag48 - e._rev.fillna(0)).astype("float32")
    e = e.drop(columns="_rev")
    with np.errstate(invalid="ignore", divide="ignore"):
        u = e.flow_phys_lag48 / e.maxbex_lag24
    e["util_lag48"] = u.where(e.maxbex_lag24 > 0).astype("float32")
    e["is_core_border"] = (e.from_zone.isin(CORE) & e.to_zone.isin(CORE)).astype("float32")
    return e.sort_values(["from_zone", "to_zone", "ts_utc"]).reset_index(drop=True)
