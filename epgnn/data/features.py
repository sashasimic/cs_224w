"""Named feature sets and the gate-closure rule.

Forecasts for delivery day D are made at noon on D-1. A column may be used as an input for day D only if
its value is known by then. This module is the single place that encodes which columns those are; model
code asks for a feature set by name and never lists columns itself.
"""
import numpy as np
import pandas as pd

# ---- zone (node) columns of zone_nodes_hourly -------------------------------------------------------
CALENDAR = ["hour_local", "dow", "month"]
# Known at gate closure for every hour of day D.
OWN = ["price_lag24", "price_lag168", "load_fc",
       "temp_pw_lag24", "temp_anom_lag24", "wind_cf_lag24", "solar_cf_lag24",
       "net_import_lag48"] + CALENDAR
# JAO Core summaries (zone-level); null outside Core, see the zone_nodes_hourly doc.
JAO_SUMMARIES = ["n_presolved", "min_ram_ratio", "in_core"]
# Same-hour weather: observed, so only allowed as the "perfect forecast" ablation.
WEATHER_PERFECT = ["temp_pw", "temp_anom", "wind_cf", "low_wind", "solar_cf"]
# History columns a GRU may see for hours up to the cutoff (realised values are fine in the past).
HISTORY = ["price_lag24", "price_lag168", "load_fc", "net_import", "temp_pw", "temp_anom", "wind_cf", "solar_cf"]
# Never an input for day D: outcomes of day D itself.
OUTCOMES = ["price", "net_import", "has_price", "price_filled"]

FEATURE_SETS: dict[str, list[str]] = {
    "calendar": CALENDAR,
    "own": OWN,
    "own+jao": OWN + JAO_SUMMARIES,
    "own+weather_perfect": OWN + WEATHER_PERFECT,
}

# ---- edge columns of zone_edges_hourly --------------------------------------------------------------
EDGE_FEATURES = ["flow_phys_lag48", "flow_phys_lag168", "flow_sched_lag24", "maxbex"]
EDGE_OUTCOMES = ["flow_phys", "flow_sched"]


def day_d_columns(feature_set: str) -> list[str]:
    """Columns allowed as day-D inputs for the named set (raises on an unknown name)."""
    cols = FEATURE_SETS[feature_set]
    bad = [c for c in cols if c in OUTCOMES]
    assert not bad, f"feature set {feature_set!r} contains day-D outcomes {bad}"
    return cols


def neighbour_means(nodes: pd.DataFrame, pairs: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Mean of `cols` over each zone's interconnected neighbours, per hour (checkpoint 2 features).
    Returns a frame keyed like `nodes` on (zone, ts_utc) with columns `nb_<col>`."""
    nb = pairs.rename(columns={"to_zone": "zone", "from_zone": "nb"})
    src = nodes[["zone", "ts_utc"] + cols].rename(columns={"zone": "nb"})
    j = nb.merge(src, on="nb")                       # one row per (zone, neighbour, hour)
    out = j.groupby(["zone", "ts_utc"])[cols].mean(numeric_only=True)
    return out.rename(columns=lambda c: f"nb_{c}").reset_index()


def as_float(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Columns to a float32 matrix; booleans and nullable ints become 0/1 with NaN preserved."""
    return df[cols].astype("float32").to_numpy()
