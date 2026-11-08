"""Audited, timestamp-aligned tensors shared by all five experiments."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from epgnn.boosting import (
    _check_artifacts,
    extend_constraint_history,
    join_constraint_history,
    JAO_LAG_FEATURES,
)
from epgnn.dataset_build.contracts import NODE_KEYS, TZ, forecast_origin, unique, utc
from epgnn.dataset_build.storage import digest, json_write

RECENT = ["hour_local", "dow", "month", "price_lag24_guarded", "price_lag168"]
RICH = (
    RECENT
    + ["load_fc", "temp_pw_lag48", "temp_anom_lag48", "wind_cf_lag48", "solar_cf_lag48"]
    + list(JAO_LAG_FEATURES)
)
NEIGHBOURS = [
    "neighbour_price24",
    "neighbour_price168",
    "neighbour_count24",
    "neighbour_count168",
]
FEATURES = RICH + NEIGHBOURS


def guard_price24(frame):
    source_day = (
        (frame.ts_utc - pd.Timedelta(hours=24))
        .dt.tz_convert(TZ)
        .dt.strftime("%Y-%m-%d")
    )
    return frame.price_lag24.where(source_day < frame.delivery_date)


def shuffle_edges(edges, seed=224):
    """Directed double-edge swaps: preserve both degrees, forbid loops/duplicates."""
    rng = np.random.default_rng(seed)
    out = edges.T.copy()
    present = set(map(tuple, out))
    for _ in range(max(100, len(out) * 30)):
        i, j = rng.choice(len(out), 2, replace=False)
        a, b = out[i]
        c, d = out[j]
        if a == d or c == b or (a, d) in present or (c, b) in present:
            continue
        present.remove((a, b))
        present.remove((c, d))
        out[i] = a, d
        out[j] = c, b
        present.add((a, d))
        present.add((c, b))
    return out.T


def neighbour_features(x, active, edges):
    """Incoming neighbour prices; no-neighbour means NaN, not a published zero."""
    result = np.full((*x.shape[:2], 4), np.nan, np.float32)
    for dest in range(x.shape[1]):
        sources = edges[0, edges[1] == dest]
        for k, column in enumerate(
            [FEATURES.index("price_lag24_guarded"), FEATURES.index("price_lag168")]
        ):
            values = x[:, sources, column]
            usable = np.isfinite(values) & active[:, sources] & active[:, dest, None]
            count = usable.sum(axis=1)
            total = np.where(usable, values, 0).sum(axis=1)
            result[:, dest, k] = np.divide(
                total, count, out=np.full(len(x), np.nan), where=count > 0
            )
            result[:, dest, k + 2] = count
    return result


def prepare(directory, output):
    """Read one zone at a time, then save fixed-width memory-mapped arrays."""
    directory, output = Path(directory).resolve(), Path(output)
    complete = json.loads((directory / "complete.json").read_text())
    hashes = _check_artifacts(
        directory, complete, ["nodes", "edges", "source_manifest.json"]
    )
    nodes = ds.dataset(directory / "nodes")
    first = nodes.to_table(columns=["zone", "ts_utc", "split"]).to_pandas()
    zones = sorted(first.zone.unique())
    times = pd.DatetimeIndex(sorted(first.ts_utc.unique())).as_unit("ns")
    labels = first[first.zone.eq(zones[0])].set_index("ts_utc").reindex(times)["split"]
    split = labels.map(
        {"pretrain": 0, "train": 0, "val": 1, "test": 2, "unassigned": 3}
    ).to_numpy(np.int8)
    if np.any(np.diff(split) < 0):
        raise ValueError("Nonchronological split calendar")
    del first
    output.mkdir(parents=True, exist_ok=True)
    shape = (len(times), len(zones))
    x = np.lib.format.open_memmap(
        output / "x.npy", mode="w+", dtype="float32", shape=(*shape, len(FEATURES))
    )
    x[:] = np.nan
    y = np.lib.format.open_memmap(
        output / "y.npy", mode="w+", dtype="float32", shape=shape
    )
    active = np.lib.format.open_memmap(
        output / "active.npy", mode="w+", dtype="bool", shape=shape
    )
    columns = list(
        dict.fromkeys(
            NODE_KEYS
            + [
                "delivery_date",
                "forecast_origin_utc",
                "split",
                "price",
                "target_valid",
                "price_structurally_available",
                "price_lag24",
                "in_cwe",
                "in_core",
            ]
            + [c for c in RICH if c not in ["price_lag24_guarded", *JAO_LAG_FEATURES]]
            + list(JAO_LAG_FEATURES.values())
        )
    )
    eligible, excluded, target_counts = [], {}, {}
    for j, zone in enumerate(zones):
        f = nodes.to_table(columns=columns, filter=ds.field("zone") == zone).to_pandas()
        f = f.sort_values("ts_utc").reset_index(drop=True)
        unique(f, NODE_KEYS, "Suite input")
        if not pd.DatetimeIndex(f.ts_utc).equals(times):
            raise ValueError("Noncanonical zone-hour grid")
        if not np.array_equal(f.target_valid, np.isfinite(f.price)):
            raise ValueError("Target validity differs from raw price finiteness")
        if not np.array_equal(f.forecast_origin_utc, forecast_origin(f.ts_utc)):
            raise ValueError("Forecast origin mismatch")
        f["price_lag24_guarded"] = guard_price24(f)
        history = f[NODE_KEYS + list(JAO_LAG_FEATURES.values()) + ["in_cwe", "in_core"]]
        history = extend_constraint_history(directory, zone, history, times[0])
        f = join_constraint_history(f, history, list(JAO_LAG_FEATURES))
        x[:, j, : len(RICH)] = f[RICH].to_numpy(np.float32)
        y[:, j] = f.price.to_numpy(np.float32)
        active[:, j] = f.price_structurally_available.to_numpy(bool)
        counts = [int(np.isfinite(y[split == s, j]).sum()) for s in range(3)]
        target_counts[zone] = dict(zip(["fit", "val", "test"], counts))
        if min(counts) > 0:
            eligible.append(j)
        else:
            excluded[zone] = "No valid targets in at least one of fit/validation/test"
        print(f"Prepared {zone}: targets {counts}", flush=True)
    # Only physically observed training edges determine topology, never test coverage.
    pairs = set()
    edge_data = ds.dataset(directory / "edges")
    filt = ds.field("split").isin(["pretrain", "train"]) & (
        ds.field("physical_row_present") == True
    )
    for batch in edge_data.scanner(
        columns=["from_zone", "to_zone"], filter=filt, batch_size=131072
    ).to_batches():
        pairs.update(zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()))
    zi = {z: j for j, z in enumerate(zones)}
    edges = np.array(
        [(zi[a], zi[b]) for a, b in sorted(pairs) if a in zi and b in zi and a != b],
        dtype=np.int64,
    ).T
    if edges.shape[0] != 2 or edges.shape[1] < 2:
        raise ValueError("Insufficient training topology")
    x[:, :, len(RICH) :] = neighbour_features(x, active, edges)
    x[~active] = np.nan
    for value in [x, y, active]:
        value.flush()
    np.save(output / "times.npy", times.asi8)
    np.save(output / "split.npy", split)
    np.save(output / "edges.npy", edges)
    np.save(output / "shuffled_edges.npy", shuffle_edges(edges))
    json_write(
        output / "metadata.json",
        {
            "zones": zones,
            "features": FEATURES,
            "eligible": eligible,
            "excluded": excluded,
            "target_counts": target_counts,
            "dataset_fingerprint": complete["fingerprint"],
            "source_artifacts": hashes,
            "geography": "Canonical identities; inactive historical nodes cannot send messages",
            "availability": "conditional: auction convention; load/weather/JAO publication vintages unverified",
            "topology": "Directed physical pairs observed in training, masked by dated node lifetimes",
            "feature_timing": {
                "price_lag24_guarded": "t-24h, previous delivery date only; auction publication convention",
                "price_lag168": "exact same-zone t-168h",
                "load_fc": "delivery-hour forecast, vintage unverified",
                "weather": "exact same-identity t-48h observations, vintage unverified",
                "JAO": "same-zone regional domain t-48h; published zeros retained, gaps remain NaN",
                "neighbours": "incoming training-topology means of guarded same-hour recent price features",
            },
            "normalization": "None for XGBoost; neural statistics fit on training rows only",
        },
    )
    json_write(
        output / "complete.json",
        {p.name: digest(p)["sha256"] for p in output.iterdir() if p.is_file()},
    )


def load(directory):
    directory = Path(directory)
    meta = json.loads((directory / "metadata.json").read_text())
    arrays = {
        name: np.load(directory / (name + ".npy"), mmap_mode="r")
        for name in ["x", "y", "active", "times", "split", "edges", "shuffled_edges"]
    }
    return meta, arrays
