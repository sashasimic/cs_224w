"""Canonical day batches, source-preserving multivariate histories and scalers."""

import fcntl
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from epgnn.controlled_study.history import history_index
from epgnn.dataset_build.contracts import forecast_origin
from epgnn.dataset_build.storage import ROOT, canonical, digest
from epgnn.experiment_suite.artifacts import write_json
from epgnn.experiment_suite.data import FEATURES
from epgnn.experiment_suite.runner import checked_cache

HISTORY = [
    "price_lag24_guarded",
    "price_lag168",
    "load_fc",
    "net_import_training_pairs",
    "temp_pw",
    "temp_anom",
    "wind_cf",
    "solar_cf",
]
DAY = [
    "price_lag24_guarded",
    "price_lag168",
    "load_fc",
    "temp_pw_lag48",
    "temp_anom_lag48",
    "wind_cf_lag48",
    "solar_cf_lag48",
    "net_import_training_pairs_lag48",
    "hour_local",
    "dow",
    "month",
]


def partner_identity(config):
    directory = Path(config["partner_directory"])
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=directory, text=True
    ).strip()
    if commit != config["partner_commit"]:
        raise ValueError("Partner source differs from pinned commit")
    names = [
        "epgnn/models/gru_mlp.py",
        "epgnn/models/encoders.py",
        "epgnn/train.py",
        "epgnn/data/scaling.py",
        "configs/experiments/c25_gru_mlp.yaml",
    ]
    hashes = {}
    for name in names:
        payload = subprocess.check_output(
            ["git", "show", f"{commit}:{name}"], cwd=directory
        )
        hashes[name] = hashlib.sha256(payload).hexdigest()
    return {"commit": commit, "files": hashes}


def calendar(times, split):
    times = pd.DatetimeIndex(times).as_unit("ns")
    if not times.is_unique or not times.is_monotonic_increasing:
        raise ValueError("Unique sorted canonical hours required")
    dates = times.tz_convert("Europe/Brussels").strftime("%Y-%m-%d")
    days = sorted(set(dates))
    index = np.full((len(days), 25), -1, np.int64)
    day_split = np.empty(len(days), np.int8)
    for i, day in enumerate(days):
        positions = np.flatnonzero(dates == day)
        if len(positions) > 25 or len(set(split[positions])) != 1:
            raise ValueError("Invalid delivery-day calendar")
        index[i, : len(positions)] = positions
        day_split[i] = split[positions[0]]
    origins = pd.DatetimeIndex(forecast_origin(times))
    required, present, inverse = history_index(times)
    first = index[:, 0]
    if not np.array_equal(inverse[first], np.arange(len(days))):
        raise ValueError("Origin/day ordering mismatch")
    return {
        "days": days,
        "slots": index,
        "split": day_split,
        "origins": origins[first].asi8,
        "history_positions": required,
        "history_present": present,
    }


def net_imports(times, zones, active, pairs, batches):
    """Net inflow over pinned training pairs; incomplete incident flows stay NaN."""
    shape = active.shape
    values = np.zeros(shape, np.float64)
    count = np.zeros(shape, np.int16)
    expected = np.zeros(shape, np.int16)
    zi = {z: j for j, z in enumerate(zones)}
    pairs = set(map(tuple, pairs.T.tolist()))
    for a, b in pairs:
        both = active[:, a] & active[:, b]
        expected[:, a] += both
        expected[:, b] += both
    for batch in batches:
        frame = batch.to_pandas()
        stamp = pd.DatetimeIndex(frame.ts_utc).as_unit("ns").asi8
        pos = np.searchsorted(times, stamp)
        for (source, dest), part in frame.groupby(["from_zone", "to_zone"], sort=False):
            a, b = zi.get(source), zi.get(dest)
            if (a, b) not in pairs:
                continue
            ix = part.index.to_numpy()
            p = pos[ix]
            good = (p < len(times)) & np.isfinite(part.flow_phys.to_numpy(float))
            ix, p = ix[good], p[good]
            good = (times[p] == stamp[ix]) & active[p, a] & active[p, b]
            ix, p = ix[good], p[good]
            flow = frame.flow_phys.to_numpy(float)[ix]
            np.add.at(values[:, a], p, -flow)
            np.add.at(values[:, b], p, flow)
            np.add.at(count[:, a], p, 1)
            np.add.at(count[:, b], p, 1)
    good = (expected > 0) & (count == expected) & active
    return np.where(good, values, np.nan).astype(np.float32), good


def prepare(config):
    parent, meta, base = checked_cache(config)
    identity = {
        "config": config,
        "parent_complete": json.loads((parent / "complete.json").read_text()),
        "partner": partner_identity(config),
        "code": {
            p.name: digest(p)["sha256"] for p in Path(__file__).parent.glob("*.py")
        },
    }
    token = hashlib.sha256(canonical(identity)).hexdigest()
    output = ROOT / "staging/partner_gru" / token
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (output / "complete.json").exists():
            temporary = output.with_name("." + token + ".building")
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary.mkdir()
            shape = base["y"].shape
            hist = np.lib.format.open_memmap(
                temporary / "history.npy",
                mode="w+",
                dtype=np.float32,
                shape=(*shape, len(HISTORY)),
            )
            hist[:] = np.nan
            for k, name in enumerate(HISTORY[:3]):
                hist[:, :, k] = base["x"][:, :, FEATURES.index(name)]
            nodes = ds.dataset(ROOT / config["dataset_directory"] / "nodes")
            for j, zone in enumerate(meta["zones"]):
                frame = (
                    nodes.to_table(
                        columns=["ts_utc", *HISTORY[4:]],
                        filter=ds.field("zone") == zone,
                    )
                    .to_pandas()
                    .sort_values("ts_utc")
                )
                np.testing.assert_array_equal(
                    pd.DatetimeIndex(frame.ts_utc).as_unit("ns").asi8, base["times"]
                )
                hist[:, j, 4:] = frame[HISTORY[4:]].to_numpy(np.float32)
            print(
                "Preparing conservative net imports from original physical observations",
                flush=True,
            )
            edges = ds.dataset(ROOT / config["dataset_directory"] / "edges")
            net, good = net_imports(
                np.asarray(base["times"]),
                meta["zones"],
                np.asarray(base["active"]),
                np.asarray(base["edges"]),
                edges.scanner(
                    columns=["ts_utc", "from_zone", "to_zone", "flow_phys"],
                    batch_size=131072,
                ).to_batches(),
            )
            hist[:, :, 3] = net
            hist[~np.asarray(base["active"])] = np.nan
            hist.flush()
            day = np.lib.format.open_memmap(
                temporary / "day.npy",
                mode="w+",
                dtype=np.float32,
                shape=(*shape, len(DAY)),
            )
            day[:] = np.nan
            for k, name in enumerate(DAY):
                if name in FEATURES:
                    day[:, :, k] = base["x"][:, :, FEATURES.index(name)]
            required = np.asarray(base["times"]) - 48 * 3600_000_000_000
            pos = np.searchsorted(base["times"], required)
            good_time = (pos < len(required)) & (
                np.asarray(base["times"])[np.minimum(pos, len(required) - 1)]
                == required
            )
            day[good_time, :, 7] = net[pos[good_time]]
            day[~np.asarray(base["active"])] = np.nan
            day.flush()
            write_json(
                temporary / "metadata.json",
                {
                    **meta,
                    "history_features": HISTORY,
                    "day_features": DAY,
                    "net_import_complete_cells": int(good.sum()),
                    "history_finite_cells": {
                        name: int(np.isfinite(hist[:, :, k]).sum())
                        for k, name in enumerate(HISTORY)
                    },
                    "availability": "conditional retrospective ERA5, load/flow archive vintages unresolved; no post-origin history intervals",
                    "differences_from_partner": [
                        "Canonical 62 identities/49 scored, no German splice or upstream fills",
                        "Literal Brussels noon and timestamp joins, 23/24/25 slots",
                        "Day weather lag48 rather than lag24; known availability conflict remains",
                        "Net inflow over training physical pairs only, missing incident flows => NaN",
                        "Index targets before arithmetic to avoid NaN masking defect",
                    ],
                },
            )
            write_json(temporary / "identity.json", identity)
            write_json(
                temporary / "complete.json",
                {
                    p.name: digest(p)["sha256"]
                    for p in temporary.iterdir()
                    if p.is_file()
                },
            )
            temporary.rename(output)
    for name, checksum in json.loads((output / "complete.json").read_text()).items():
        if digest(output / name)["sha256"] != checksum:
            raise ValueError("Partner-GRU cache changed: " + name)
    meta = json.loads((output / "metadata.json").read_text())
    arrays = {
        **base,
        "hist": np.load(output / "history.npy", mmap_mode="r"),
        "day": np.load(output / "day.npy", mmap_mode="r"),
    }
    return output, token, meta, arrays


class Days:
    def __init__(self, arrays, zones, history_hours=168):
        if history_hours != 168:
            raise ValueError("This experiment pins 168-hour histories")
        self.arrays, self.zones = arrays, zones
        times = pd.to_datetime(arrays["times"], unit="ns", utc=True)
        self.calendar = calendar(times, arrays["split"])
        self.pairs = (
            np.array(
                sorted(
                    {
                        tuple(sorted((a, b)))
                        for a, b in arrays["edges"].T.tolist()
                        if a in zones and b in zones and a != b
                    }
                ),
                np.int64,
            )
            .reshape(-1, 2)
            .T
        )

    def raw(self, days):
        cal, arrays = self.calendar, self.arrays
        positions = cal["slots"][days]
        exists = positions >= 0
        safe = np.maximum(positions, 0)
        x = np.asarray(arrays["day"][safe]).transpose(0, 2, 1, 3).copy()
        x[~np.broadcast_to(exists[:, None, :, None], x.shape)] = np.nan
        y = np.asarray(arrays["y"][safe]).transpose(0, 2, 1).copy()
        mask = np.isfinite(y) & exists[:, None, :]
        eligible = np.zeros(arrays["y"].shape[1], bool)
        eligible[self.zones] = True
        mask &= eligible[None, :, None]
        weekly, recent = x[:, :, :, 1], x[:, :, :, 0]
        base = np.where(
            np.isfinite(weekly), weekly, np.where(np.isfinite(recent), recent, 0)
        )
        residual = np.arcsinh(y / 50) - np.arcsinh(base / 50)
        pos = cal["history_positions"][days]
        h = np.asarray(arrays["hist"][pos]).transpose(0, 2, 1, 3).copy()
        present = (
            cal["history_present"][days][:, None, :, None]
            & np.asarray(arrays["active"][pos]).transpose(0, 2, 1)[:, :, :, None]
        )
        h[~np.broadcast_to(present, h.shape)] = np.nan
        return {
            "hist": h,
            "xday": x,
            "target": residual,
            "mask": mask,
            "base": base,
            "slots": positions,
        }

    def fit_scalers(self, days):
        result = {}
        for key in ["hist", "xday"]:
            count = total = squares = None
            for start in range(0, len(days), 32):
                values = self.raw(days[start : start + 32])[key][:, self.zones].astype(
                    np.float64
                )
                good = np.isfinite(values)
                axes = tuple(range(values.ndim - 1))
                n = good.sum(axis=axes)
                s = np.where(good, values, 0).sum(axis=axes)
                ss = np.square(np.where(good, values, 0)).sum(axis=axes)
                count = n if count is None else count + n
                total = s if total is None else total + s
                squares = ss if squares is None else squares + ss
            mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
            variance = (
                np.divide(squares, count, out=np.zeros_like(total), where=count > 0)
                - mean**2
            )
            result[key] = {
                "mean": mean.tolist(),
                "std": np.where(
                    count > 0, np.sqrt(np.maximum(variance, 0)) + 1e-6, 1.0
                ).tolist(),
                "count": count.tolist(),
            }
        return result

    def batch(self, days, scalers, device):
        import torch

        raw = self.raw(days)
        result = {}
        for key in ["hist", "xday"]:
            values = (raw[key] - np.asarray(scalers[key]["mean"])) / np.asarray(
                scalers[key]["std"]
            )
            result[key] = torch.as_tensor(
                np.where(
                    np.isfinite(values) & (np.asarray(scalers[key]["count"]) > 0),
                    values,
                    0,
                ).astype(np.float32),
                device=device,
            )
        result["target"] = torch.as_tensor(raw["target"], device=device)
        result["mask"] = torch.as_tensor(raw["mask"], device=device)
        return result, raw
