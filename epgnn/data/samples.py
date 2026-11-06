"""Samples: one per (zone, delivery day). Every model is trained and scored on the same samples.

For a delivery day D (Brussels calendar day) a sample provides
  - targets: the hourly prices of D, as a (H,) vector with a validity mask (H = 25 slots: 23/24/25-hour days)
  - day-D inputs: per-hour features known at gate closure, (H, n_features)
  - history: the W hours ending at the cutoff (D-1 at CUTOFF_UTC), (W, n_history_cols)
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from epgnn.data import features as F
from epgnn.data import tables
from epgnn.splits import splits

H = 25                       # slots per delivery day (DST days have 23 or 25 hours)
CUTOFF_UTC = 10              # D-1 10:00 UTC = noon CEST / 11:00 CET, the day-ahead gate closure


@dataclass
class Samples:
    zone: np.ndarray            # (n,) zone code
    day: np.ndarray             # (n,) delivery day, YYYY-MM-DD
    y: np.ndarray               # (n, H) price EUR/MWh, NaN where masked
    mask: np.ndarray            # (n, H) True where the hour exists and has a valid price
    hours: np.ndarray           # (n, H) ts_utc as int64 ns, -1 where the slot does not exist

    def __len__(self) -> int:
        return len(self.zone)


def days_of(split: str) -> list[str]:
    """Delivery days of a split that exist in the node table."""
    a, b = splits()[split]
    d = tables.nodes().delivery_date.unique()
    return sorted(x for x in d if a <= x <= b)


def _slot(nodes: pd.DataFrame) -> pd.Series:
    """Position of each hour within its (zone, delivery day): 0..24."""
    return nodes.groupby(["zone", "delivery_date"]).cumcount()


def build(split: str, zones: list[str] | None = None) -> Samples:
    """Targets and masks for every (zone, day) of the split."""
    n = tables.nodes()
    days = days_of(split)
    zones = zones or tables.zones()
    sub = n[n.delivery_date.isin(days) & n.zone.isin(zones)].copy()
    sub["slot"] = _slot(sub)
    idx = pd.MultiIndex.from_product([zones, days], names=["zone", "delivery_date"])
    y = sub.pivot_table(index=["zone", "delivery_date"], columns="slot", values="price", aggfunc="first").reindex(idx)
    y = y.reindex(columns=range(H)).to_numpy(dtype="float32")
    ok = sub.pivot_table(index=["zone", "delivery_date"], columns="slot", values="has_price", aggfunc="first").reindex(idx)
    mask = ok.reindex(columns=range(H)).eq(True).to_numpy(dtype=bool)
    ts = sub.assign(t=sub.ts_utc.astype("int64")).pivot_table(index=["zone", "delivery_date"], columns="slot", values="t", aggfunc="first").reindex(idx)
    hours = ts.reindex(columns=range(H)).fillna(-1).to_numpy(dtype="int64")
    return Samples(zone=idx.get_level_values(0).to_numpy(), day=idx.get_level_values(1).to_numpy(),
                   y=y, mask=mask, hours=hours)


def day_inputs(s: Samples, cols: list[str]) -> np.ndarray:
    """Per-hour day-D inputs, (n, H, len(cols)); NaN where the slot does not exist."""
    n = tables.nodes().set_index(["zone", "ts_utc"])
    out = np.full((len(s), H, len(cols)), np.nan, dtype="float32")
    zone_rep = np.repeat(s.zone, H)
    ts = pd.to_datetime(s.hours.ravel(), utc=True)
    valid = s.hours.ravel() >= 0
    keys = pd.MultiIndex.from_arrays([zone_rep[valid], ts[valid]])
    vals = F.as_float(n.reindex(keys), cols)
    flat = out.reshape(-1, len(cols))
    flat[valid] = vals
    return flat.reshape(len(s), H, len(cols))


def history(s: Samples, cols: list[str], window: int = 168) -> np.ndarray:
    """The `window` hours ending at the cutoff (D-1 CUTOFF_UTC inclusive), (n, window, len(cols))."""
    n = tables.nodes()
    out = np.full((len(s), window, len(cols)), np.nan, dtype="float32")
    cutoff = (pd.DatetimeIndex(pd.to_datetime(s.day, utc=True)) - pd.Timedelta(days=1) + pd.Timedelta(hours=CUTOFF_UTC)).asi8
    for z in np.unique(s.zone):
        g = n[n.zone == z]
        t = g.ts_utc.astype("int64").to_numpy()
        arr = F.as_float(g, cols)
        rows = np.flatnonzero(s.zone == z)
        end = np.searchsorted(t, cutoff[rows], side="right")          # position after the cutoff hour
        idx = end[:, None] + np.arange(-window, 0)[None, :]
        ok = idx >= 0
        safe = np.where(ok, idx, 0)
        win = arr[safe]
        win[~ok] = np.nan
        out[rows] = win
    return out
