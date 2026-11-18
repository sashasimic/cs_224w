"""Immutable source panels, exact forecast windows and directional CORE features."""

import shutil
import tempfile
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from epgnn.dataset_build.contracts import forecast_origin
from epgnn.dataset_build.storage import ROOT, digest, json_write
from epgnn.experiment_suite import data
from epgnn.relational_forecast import catalog, graph

HOUR = 3_600_000_000_000
CORE = catalog.CORE
FIELDS = {
    "weather": ("temp_pw", "temp_anom", "wind_cf", "solar_cf"),
    "load": ("value",),
    "jao": ("n_presolved", "min_ram_ratio", "domain_present"),
    "flow": ("value",),
}
QUERIES = {
    "weather": """
        SELECT epoch_ns(effective_utc) AS hour, zone, temp_pw, temp_anom, wind_cf, solar_cf,
               epoch_ns(published_utc) AS published FROM weather
        WHERE zone <> 'DE_LU' AND effective_utc >= ? AND effective_utc < ?
    """,
    "load": """
        SELECT epoch_ns(effective_utc) AS hour, zone,
               CASE WHEN IsValidValue THEN LoadForecastMW END AS value,
               epoch_ns(published_utc) AS published FROM load_forecast
        WHERE compatibility = 'exact_source_name_identity'
          AND effective_utc >= ? AND effective_utc < ?
    """,
    "jao": """
        SELECT epoch_ns(d.ts_utc) AS hour, s.zone, s.n_presolved, s.min_ram_ratio,
               CAST(d.domain_present AS DOUBLE) AS domain_present,
               epoch_ns(coalesce(s.published_utc, d.published_utc)) AS published
        FROM jao_domain_hour d JOIN jao_zone_summary s USING (regime, ts_utc)
        WHERE d.ts_utc >= ? AND d.ts_utc < ?
    """,
    "flow": """
        SELECT epoch_ns(ts_utc) AS hour, from_zone, to_zone,
               CASE WHEN source_valid THEN value_mw END AS value,
               epoch_ns(published_utc) AS published
        FROM border_hour
        WHERE kind = 'physical_flow' AND ts_utc >= ? AND ts_utc < ?
    """,
}


def windows(times, start, panel_size):
    """Exact elapsed-hour indexes, with one window shared by every delivery-day hour."""
    origins = forecast_origin(pd.Series(pd.to_datetime(times, utc=True))).array.asi8
    unique, inverse = np.unique(origins, return_inverse=True)
    positions = (unique[:, None] - start) // HOUR - np.arange(168, 0, -1)
    if (positions < 0).any() or (positions >= panel_size).any():
        raise ValueError("Source calendar does not cover the requested histories")
    return unique, inverse, positions


def source_panel(db, name, start, size, zones, pairs):
    """Stream source rows into hourly panels. Presence stays separate from validity."""
    length = len(pairs) if name == "flow" else len(zones)
    values = np.full((size, length, len(FIELDS[name])), np.nan, np.float32)
    presence = np.zeros((size, length), dtype=bool)
    published = np.full((size, length), np.iinfo(np.int64).min, dtype=np.int64)
    lower, upper = pd.to_datetime([start, start + size * HOUR], utc=True)
    zone_ids = {z: i for i, z in enumerate(zones)}
    pair_ids = {a * len(zones) + b: i for i, (a, b) in enumerate(pairs)}
    for batch in db.execute(QUERIES[name], [lower, upper]).to_arrow_reader(262144):
        frame = batch.to_pandas(types_mapper=pd.ArrowDtype)
        if name == "flow":
            a = frame.from_zone.map(zone_ids)
            b = frame.to_zone.map(zone_ids)
            keys = (a * len(zones) + b).map(pair_ids)
        else:
            keys = frame.zone.map(zone_ids)
        good = keys.notna().to_numpy()
        col = keys[good].to_numpy(dtype=np.int64)
        sub = frame.loc[good]
        hour = (sub.hour.to_numpy(dtype=np.int64) - start) // HOUR
        flat = hour * length + col
        if len(np.unique(flat)) != len(flat) or presence[hour, col].any():
            raise ValueError(f"Duplicate canonical fact in {name}")
        presence[hour, col] = True
        values[hour, col] = sub[list(FIELDS[name])].to_numpy(
            dtype=np.float32, na_value=np.nan
        )
        published[hour, col] = sub.published.to_numpy(
            dtype=np.int64, na_value=np.iinfo(np.int64).min
        )
    return values, presence, published


def select(values, presence, published, positions, origins):
    valid = presence[positions] & (published[positions] <= origins[:, None, None])
    return np.where(valid[..., None], values[positions], np.nan), valid


def quantile(values, mask):
    eligible = np.isfinite(values) & mask
    n = eligible.sum(axis=0)
    sorted_values = np.sort(np.where(eligible, values, np.inf), axis=0)
    index = np.maximum(n - 1, 0) * 0.1
    lo = np.floor(index).astype(int)
    hi = np.ceil(index).astype(int)
    a = np.take_along_axis(sorted_values, lo[None], axis=0)[0]
    b = np.take_along_axis(sorted_values, hi[None], axis=0)[0]
    a = np.where(n > 0, a, 0)
    b = np.where(n > 0, b, 0)
    return np.where(n > 0, a + (index - lo) * (b - a), np.nan)


def directional_margin(ptdf, attributes):
    """Summaries for transfers from row zone to column zone."""
    delta = ptdf[:, :, None] - ptdf[:, None, :]
    eligible = np.isfinite(delta)
    tight = eligible & (delta > 0.005)
    ram, fmax, ratio = attributes.T[:, :, None, None]
    features = np.stack(
        [
            tight.sum(axis=0),
            np.max(np.where(tight, delta, 0), axis=0),
            quantile(ram, tight),
            quantile(ratio, tight & (fmax > 0)),
        ],
        axis=-1,
    ).astype(np.float32)
    features[~eligible.any(axis=0)] = np.nan
    features[np.arange(len(ptdf.T)), np.arange(len(ptdf.T))] = np.nan
    return features


def preparation(settings):
    base = catalog.load_settings(settings["base_config"])
    source = catalog.build(base)
    cache_marker = Path(base["benchmark_arrays"]) / "complete.json"
    artifacts = catalog.read_json(cache_marker)
    for name, sha in artifacts.items():
        if digest(cache_marker.parent / name)["sha256"] != sha:
            raise ValueError(f"Benchmark artifacts changed: {name}")
    meta, arrays = data.load(cache_marker.parent)
    if meta["dataset_fingerprint"] != base["dataset_fingerprint"]:
        raise ValueError("Unexpected benchmark dataset")
    manifest = {
        "catalog": source.name,
        "dataset": base["dataset_fingerprint"],
        "benchmark": digest(cache_marker)["sha256"],
        "inputs_code": digest(__file__)["sha256"],
    }
    root = ROOT / settings["cache"]
    final = root / catalog.identity(manifest)
    if final.exists():
        return catalog.check_generation(final)
    root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".prepare-", dir=root))
    try:
        fit = (arrays["split"] == 0) | (arrays["split"] == 1)
        roster = np.array(meta["eligible"])
        zones = [meta["zones"][i] for i in roster]
        times = arrays["times"][fit]
        prices = arrays["y"][fit][:, roster]
        active = arrays["active"][fit][:, roster]
        recent = arrays["x"][fit][:, roster, : len(data.RECENT)]
        index = {int(orig): i for i, orig in enumerate(roster)}
        edges = np.array(
            [
                (index[a], index[b])
                for a, b in arrays["edges"].T
                if a in index and b in index
            ],
            np.int64,
        )
        start = int(times[0] - 216 * HOUR)
        size = len(times) + 216
        origins, day, historical = windows(times, start, size)
        if not np.array_equal(np.diff(times), np.full(len(times) - 1, HOUR)):
            raise ValueError("Target calendar is not a contiguous UTC hourly grid")
        np.savez(
            work / "grid.npz",
            times=times,
            prices=prices,
            active=active,
            recent=recent,
            edges=edges,
            origins=origins,
            day=day,
            historical=historical,
        )
        json_write(work / "zones.json", zones)
        database = source / "forecast.duckdb"
        price = np.full((size, len(zones), 1), np.nan, np.float32)
        price[216:, :, 0] = prices
        np.save(work / "price_values.npy", price)
        np.save(work / "price_presence.npy", np.isfinite(price[..., 0]))
        with closing(catalog.connect(database)) as connection:
            for name in FIELDS:
                print(f"Preparing {name} source rows", flush=True)
                panel = source_panel(connection, name, start, size, zones, edges)
                for suffix, value in zip(
                    ("values", "presence", "published"), panel, strict=True
                ):
                    np.save(work / f"{name}_{suffix}.npy", value)
        days = pd.to_datetime(times, utc=True).tz_convert("Europe/Brussels")
        margin_hours = np.flatnonzero(days >= pd.Timestamp("2022-06-09", tz=days.tz))
        print("Preparing directional CORE margins", flush=True)
        cnec = graph.observations(database, times[margin_hours], zones)
        offsets = cnec["offsets"]
        margin = np.full(
            (len(offsets) - 1, len(CORE), len(CORE), 4), np.nan, np.float32
        )
        for hour in np.flatnonzero(np.diff(offsets)):
            rows = slice(offsets[hour], offsets[hour + 1])
            margin[hour] = directional_margin(
                cnec["coefficients"][rows], cnec["attributes"][rows]
            )
        np.save(work / "margin.npy", margin)
        np.save(work / "margin_hours.npy", margin_hours)
        manifest["source_rows"] = {
            name: int(np.load(work / f"{name}_presence.npy", mmap_mode="r").sum())
            for name in ("price", *FIELDS)
        }
        json_write(work / "manifest.json", manifest)
        json_write(
            work / "complete.json",
            {p.name: digest(p)["sha256"] for p in work.iterdir()},
        )
        try:
            work.rename(final)
        except OSError:
            if not final.exists():
                raise
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return catalog.check_generation(final)


class Inputs:
    def __init__(self, directory):
        self.directory = directory
        with np.load(directory / "grid.npz") as grid:
            self.__dict__.update(dict(grid))
        self.zones = catalog.read_json(directory / "zones.json")
        self.day_active = np.zeros((len(self.origins), len(self.zones)), dtype=bool)
        np.logical_or.at(self.day_active, self.day, self.active)
        self.dates = (
            pd.to_datetime(self.times, utc=True)
            .tz_convert("Europe/Brussels")
            .strftime("%Y-%m-%d")
        )
        self.margin = np.load(directory / "margin.npy", mmap_mode="r")
        self.margin_hours = np.load(directory / "margin_hours.npy")
        self.core = np.array([self.zones.index(zone) for zone in CORE])
        self.panels = {
            name: tuple(
                np.load(directory / f"{name}_{attr}.npy", mmap_mode="r")
                for attr in (
                    ("values", "presence")
                    if name == "price"
                    else ("values", "presence", "published")
                )
            )
            for name in ("price", *FIELDS)
        }

    def history(self, name, days):
        positions = self.historical[days]
        if name == "price":
            values, mask = self.panels[name]
            values, valid = values[positions], mask[positions]
        else:
            values, valid = select(
                *self.panels[name], positions[:, -72:-24], self.origins[days]
            )
        active = self.day_active[days]
        if name == "flow":
            source, dest = self.edges.T
            active = active[:, source] & active[:, dest]
        valid &= active[:, None, :]
        return np.where(valid[..., None], values, np.nan), valid
