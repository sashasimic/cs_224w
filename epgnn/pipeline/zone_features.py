"""Engineered (zone, hour) columns on top of zone_nodes_hourly.

The node table is a complete grid: every zone has a row for every hour, sorted by zone then ts_utc, so a
shift of k rows inside a zone is a lag of k hours. Every new column is listed in registry.ZONE with its
timing; nothing here uses day-D outcomes for a day-D input.
"""
import numpy as np
import pandas as pd

from epgnn.pipeline import constraints, graphs


def _lag(df: pd.DataFrame, col: str, hours: int) -> pd.Series:
    return df.groupby("zone", sort=False)[col].shift(hours)


def add_calendar(n: pd.DataFrame) -> None:
    two_pi = 2 * np.pi
    n["hour_sin"] = np.sin(two_pi * n.hour_local / 24).astype("float32")
    n["hour_cos"] = np.cos(two_pi * n.hour_local / 24).astype("float32")
    n["dow_sin"] = np.sin(two_pi * n.dow / 7).astype("float32")
    n["dow_cos"] = np.cos(two_pi * n.dow / 7).astype("float32")
    n["month_sin"] = np.sin(two_pi * (n.month - 1) / 12).astype("float32")
    n["month_cos"] = np.cos(two_pi * (n.month - 1) / 12).astype("float32")
    n["is_weekend"] = (n.dow >= 5).astype("float32")


def add_price(n: pd.DataFrame, zone_list: list[str]) -> None:
    n["price_wk_change"] = n.price_lag24 - n.price_lag168
    g = n.groupby(["zone", "delivery_date"], sort=False).price_lag24
    for stat in ("mean", "min", "max", "std"):
        n[f"price_d1_{stat}"] = g.transform(stat).astype("float32")
    n["europe_price_lag24"] = n.groupby("ts_utc").price_lag24.transform("mean").astype("float32")
    n["price_lag24_vs_europe"] = n.price_lag24 - n.europe_price_lag24
    # neighbour mean over the interconnector graph, done as a matrix product on the (hour, zone) grid
    z, t = len(zone_list), len(n) // len(zone_list)
    p = n.price_lag24.to_numpy(dtype=np.float64).reshape(z, t).T          # (T, Z)
    ok = np.isfinite(p)
    a = graphs.adjacency(zone_list)                                        # a[i, j] = 1 if j neighbours i
    num = np.where(ok, p, 0.0) @ a.T
    den = ok.astype(np.float64) @ a.T
    with np.errstate(invalid="ignore", divide="ignore"):
        nb = np.where(den > 0, num / den, np.nan)
    n["nb_price_lag24"] = nb.T.reshape(-1).astype("float32")


def add_load(n: pd.DataFrame) -> None:
    n["load_fc_lag168"] = _lag(n, "load_fc", 168)
    with np.errstate(invalid="ignore", divide="ignore"):
        n["load_fc_wk_change"] = (n.load_fc / n.load_fc_lag168 - 1).astype("float32")
        n["load_fc_ramp"] = (n.load_fc - _lag(n, "load_fc", 1)).astype("float32")
        day_mean = n.groupby(["zone", "delivery_date"], sort=False).load_fc.transform("mean")
        n["load_fc_day_rel"] = (n.load_fc / day_mean).astype("float32")
    for c in ("load_fc_wk_change", "load_fc_day_rel"):
        n[c] = n[c].replace([np.inf, -np.inf], np.nan)


WEATHER = ("temp_pw", "temp_anom", "wind_cf", "solar_cf")
CUTOFF_UTC = 10   # the forecast for delivery day D is made at D-1 10:00 UTC (epgnn.data.samples.CUTOFF_UTC)


def lag24_is_observed(ts_utc: pd.Series, delivery_date: pd.Series) -> np.ndarray:
    """True where the value 24 h before hour ts_utc of delivery day D is already observed at the cutoff
    D-1 10:00 UTC, i.e. ts_utc - 24 h <= D-1 10:00 UTC  <=>  ts_utc <= D 10:00 UTC. False for roughly the
    last 14 hours of every day: their lag-24 value lies on D-1 after the cutoff."""
    days = pd.Series(pd.to_datetime(delivery_date.unique(), utc=True), index=delivery_date.unique())
    limit = delivery_date.map(days) + pd.Timedelta(hours=CUTOFF_UTC)
    return (ts_utc <= limit).to_numpy()


def add_weather(n: pd.DataFrame) -> None:
    """Observed weather that is legal as a day-D input.

    `<c>_lag24` (from the node table) is NOT: for hours after 10:00 UTC of day D it is the observation from
    D-1 after the cutoff, a future value at forecast time. `<c>_asof` is the same-clock-hour observation
    that is known at the cutoff: lag 24 where that hour of D-1 is already past, lag 48 otherwise.
    heat_stress is built on the as-of anomaly."""
    for c in WEATHER:
        n[f"{c}_lag48"] = _lag(n, c, 48).astype("float32")
    ok = lag24_is_observed(n.ts_utc, n.delivery_date)
    for c in WEATHER:
        n[f"{c}_asof"] = np.where(ok, n[f"{c}_lag24"], n[f"{c}_lag48"]).astype("float32")
    n["heat_stress"] = n.temp_anom_asof.clip(lower=0).astype("float32")


def add_jao(n: pd.DataFrame, exposure: pd.DataFrame | None) -> pd.DataFrame:
    cols = constraints.EXPOSURE_FEATURES
    if exposure is None:
        for c in cols + [f"{c}_lag24" for c in cols]:
            n[c] = np.float32(np.nan)
        return n
    n = n.merge(exposure, on=["ts_utc", "zone"], how="left", validate="one_to_one")
    n = n.sort_values(["zone", "ts_utc"]).reset_index(drop=True)
    for c in cols:
        n[f"{c}_lag24"] = _lag(n, c, 24).astype("float32")
    return n


def build(nodes: pd.DataFrame, exposure: pd.DataFrame | None = None) -> pd.DataFrame:
    """zone_nodes_hourly + engineered columns, same row order (zone, ts_utc)."""
    n = nodes.sort_values(["zone", "ts_utc"]).reset_index(drop=True).copy()
    zone_list = sorted(n.zone.unique())
    counts = n.groupby("zone").size()
    assert counts.nunique() == 1, "node table is not a complete (zone, hour) grid"
    add_calendar(n)
    add_price(n, zone_list)
    add_load(n)
    add_weather(n)
    n = add_jao(n, exposure)
    return n
