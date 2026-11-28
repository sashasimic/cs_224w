"""Dense arrays for the torch models, built in memory from features/v1/zone_features_hourly.parquet.

    zt = load(["price_lag24", "load_fc", ...])
    zt.x[t, z, f]      raw (unscaled) feature f of zone z at hour t           (T, Z, F) float32
    zt.price[t, z]     target price, NaN where invalid                          (T, Z)
    zt.has_price[t, z] the loss mask                                            (T, Z) bool
    zt.base[t, z]      residual reference: price_lag168, else price_lag24, else 0
    zt.days            delivery days (Brussels), sorted                         (D,)
    zt.slots[d, k]     hour index of slot k of day d, -1 if the day has fewer slots (D, 25)
    zt.cutoff[d]       hour index of the forecast cutoff, D-1 10:00 UTC          (D,)

A history window for day d is hours cutoff[d]-W+1 .. cutoff[d] (inclusive), exactly as
epgnn.data.samples.history; day slots follow epgnn.data.samples.build, so predictions reshape directly
into the Samples layout that epgnn.metrics scores.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from epgnn.paths import ROOT
from epgnn.splits import splits

FEATURES_DIR = ROOT / "features" / "v1"
H = 25
CUTOFF_UTC = 10


@dataclass
class ZoneTensors:
    hours: np.ndarray
    zones: list[str]
    cols: list[str]
    x: np.ndarray
    price: np.ndarray
    has_price: np.ndarray
    base: np.ndarray
    days: np.ndarray
    slots: np.ndarray
    cutoff: np.ndarray

    def day_indices(self, split: str) -> np.ndarray:
        a, b = splits()[split]
        return np.flatnonzero((self.days >= a) & (self.days <= b))

    def hours_of(self, day_idx: np.ndarray) -> np.ndarray:
        s = self.slots[day_idx].ravel()
        return s[s >= 0]


def load(cols: list[str], features_dir: Path = FEATURES_DIR) -> ZoneTensors:
    need = ["zone", "ts_utc", "delivery_date", "price", "has_price", "price_lag24", "price_lag168"]
    df = pd.read_parquet(features_dir / "zone_features_hourly.parquet", columns=list(dict.fromkeys(need + cols)))
    df = df.sort_values(["zone", "ts_utc"]).reset_index(drop=True)
    zones = sorted(df.zone.unique())
    z = len(zones)
    t = len(df) // z
    assert t * z == len(df), "not a complete (zone, hour) grid"
    hours = df.ts_utc.astype("int64").to_numpy()[:t]
    assert (df.ts_utc.astype("int64").to_numpy().reshape(z, t) == hours).all(), "zones do not share the hour grid"

    def grid(c: str, dtype=np.float32) -> np.ndarray:
        return df[c].to_numpy(dtype=dtype, na_value=np.nan if dtype == np.float32 else False).reshape(z, t).T

    x = np.stack([grid(c) for c in cols], axis=-1) if cols else np.zeros((t, z, 0), np.float32)
    price = grid("price")
    has_price = grid("has_price", bool) & np.isfinite(price)
    base = grid("price_lag168")
    lag24 = grid("price_lag24")
    base = np.where(np.isnan(base), lag24, base)
    base = np.where(np.isnan(base), 0.0, base).astype(np.float32)

    dd = df.delivery_date.to_numpy()[:t]
    days, first = np.unique(dd, return_index=True)
    slots = np.full((len(days), H), -1, dtype=np.int64)
    counts = np.diff(np.append(first, t))
    for i, (f, c) in enumerate(zip(first, counts)):
        slots[i, :c] = np.arange(f, f + c)
    cut = (pd.DatetimeIndex(pd.to_datetime(days, utc=True)) - pd.Timedelta(days=1) + pd.Timedelta(hours=CUTOFF_UTC)).asi8
    cutoff = np.searchsorted(hours, cut, side="right") - 1
    return ZoneTensors(hours=hours, zones=zones, cols=list(cols), x=x, price=price, has_price=has_price,
                       base=base, days=days.astype(str), slots=slots, cutoff=cutoff)
