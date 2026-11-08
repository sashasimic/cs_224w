"""Bounded-memory airport feature cache, preserving the canonical target grid."""

import fcntl
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from epgnn.dataset_build.contracts import forecast_origin
from epgnn.dataset_build.storage import ROOT, canonical, digest
from epgnn.experiment_suite.artifacts import write_json
from epgnn.experiment_suite.data import FEATURES, RECENT
from epgnn.experiment_suite.runner import checked_cache

from .features import TEMPERATURE_FEATURES, at_origins, country_proxy, normalize
from .sources import checked_sources

ALL_FEATURES = FEATURES + TEMPERATURE_FEATURES
GROUPS = {
    "recent": RECENT,
    "airport_latest": RECENT + TEMPERATURE_FEATURES[:2],
    "airport_history": RECENT + TEMPERATURE_FEATURES,
}


def indices(group):
    return [ALL_FEATURES.index(name) for name in GROUPS[group]]


def load_cache(path, base):
    complete = json.loads((path / "complete.json").read_text())
    for name, sha in complete.items():
        if digest(path / name)["sha256"] != sha:
            raise ValueError(f"Airport cache checksum mismatch: {name}")
    arrays = {**base, "x": np.load(path / "x.npy", mmap_mode="r")}
    return json.loads((path / "metadata.json").read_text()), arrays


def prepare(config):
    manifest = checked_sources(config)
    parent, metadata, base = checked_cache(config)
    identity = {
        "config": config,
        "sources": manifest,
        "parent_complete": json.loads((parent / "complete.json").read_text()),
        "code": {
            name: digest(Path(__file__).with_name(name))["sha256"]
            for name in ["data.py", "features.py", "sources.py"]
        },
    }
    token = hashlib.sha256(canonical(identity)).hexdigest()
    cache = ROOT / "staging/airport_temperature" / token
    cache.parent.mkdir(parents=True, exist_ok=True)
    with (cache.parent / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (cache / "complete.json").exists():
            temporary = cache.with_name("." + token + ".building")
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary.mkdir()
            build_cache(config, manifest, metadata, base, identity, temporary)
            temporary.rename(cache)
    metadata, arrays = load_cache(cache, base)
    return cache, metadata, arrays


def build_cache(config, manifest, metadata, base, identity, output):
    shape = (*base["x"].shape[:2], len(ALL_FEATURES))
    x = np.lib.format.open_memmap(
        output / "x.npy", mode="w+", dtype=np.float32, shape=shape
    )
    x[:] = np.nan
    for name in RECENT:
        column = FEATURES.index(name)
        x[:, :, column] = base["x"][:, :, column]
    hourly_origins = forecast_origin(pd.to_datetime(base["times"], utc=True))
    unique, inverse = np.unique(
        pd.DatetimeIndex(hourly_origins).asi8, return_inverse=True
    )
    origins = pd.to_datetime(unique, utc=True)
    mapping = pd.read_csv(ROOT / config["airport_mapping"]).set_index("country")
    station_features, findings = {}, []
    source = ROOT / config["source_directory"]
    normalized = output / "observations"
    normalized.mkdir()
    for row in mapping.itertuples():
        raw = pd.read_csv(
            source / "raw" / f"{row.station}.csv.gz", comment="#", na_values=["M"]
        )
        observations, audit = normalize(raw)
        station_features[row.Index] = at_origins(observations, origins, config)
        observations.to_parquet(normalized / f"{row.station}.parquet", index=False)
        findings.append({"country": row.Index, "station": row.station, **audit})
        print(f"AUDIT {row.station}: {audit}", flush=True)
    coverage, geography = [], []
    months = (
        pd.to_datetime(base["times"], utc=True)
        .tz_convert("Europe/Brussels")
        .strftime("%Y-%m")
    )
    for j, zone in enumerate(metadata["zones"]):
        country, geographic_status = country_proxy(zone)
        if country not in mapping.index and geographic_status != "unresolved_geography":
            geographic_status = "country_without_pinned_airport"
        if country in station_features:
            x[:, j, len(FEATURES) :] = station_features[country][inverse]
        x[~np.asarray(base["active"][:, j]), j, :] = np.nan
        station = (
            str(mapping.loc[country, "station"]) if country in mapping.index else None
        )
        geography.append(
            {
                "zone": zone,
                "country_proxy": country,
                "station": station,
                "geographic_status": geographic_status,
            }
        )
        frame = pd.DataFrame(
            {
                "month": months,
                "split": np.asarray(base["split"]),
                "active": np.asarray(base["active"][:, j]),
                "targets": np.isfinite(base["y"][:, j]),
                "latest_present": np.isfinite(x[:, j, len(FEATURES)]),
                "history_present": np.isfinite(x[:, j, len(FEATURES) + 2]),
            }
        )
        for (month, split), group in frame.groupby(["month", "split"]):
            target = group.targets.to_numpy()
            coverage.append(
                {
                    "zone": zone,
                    "station": station,
                    "geographic_status": geographic_status,
                    "scored_zone": j in metadata["eligible"],
                    "month": month,
                    "split": {0: "fit", 1: "val", 2: "test", 3: "unassigned"}[
                        int(split)
                    ],
                    "rows": len(group),
                    "structurally_active": int(group.active.sum()),
                    "target_hours": int(target.sum()),
                    "latest_available_target_hours": int(
                        (target & group.latest_present).sum()
                    ),
                    "history_available_target_hours": int(
                        (target & group.history_present).sum()
                    ),
                }
            )
    x.flush()
    coverage = pd.DataFrame(coverage)
    coverage.to_csv(output / "coverage.csv", index=False)
    pd.DataFrame(findings).to_csv(output / "source_findings.csv", index=False)
    pd.DataFrame(geography).to_csv(output / "geography.csv", index=False)
    pd.DataFrame(
        {
            "forecast_origin_utc": origins,
            "observation_cutoff_utc": origins
            - pd.Timedelta(hours=config["reporting_buffer_hours"]),
        }
    ).to_parquet(output / "origins.parquet", index=False)
    summary = (
        coverage[coverage.scored_zone]
        .groupby("split")[
            [
                "rows",
                "target_hours",
                "latest_available_target_hours",
                "history_available_target_hours",
            ]
        ]
        .sum()
    )
    write_json(
        output / "metadata.json",
        {
            **metadata,
            "features": ALL_FEATURES,
            "feature_groups": GROUPS,
            "parent_features_excluded": [
                name for name in FEATURES if name not in RECENT
            ],
            "coverage_summary": summary.to_dict("index"),
            "availability": "conditional: observations buffered two hours; original receipt/revisions unresolved",
            "geography": geography,
        },
    )
    registry = []
    for name in RECENT:
        registry.append(
            {
                "feature": name,
                "source": "canonical price/calendar cache",
                "status": "conditional" if name.startswith("price_") else "admitted",
                "rule": "auction convention and historical identity guard; archive revisions unresolved"
                if name.startswith("price_")
                else "deterministic delivery calendar",
            }
        )
    for name in TEMPERATURE_FEATURES:
        registry.append(
            {
                "feature": name,
                "source": "raw IEM METAR temperature observations",
                "status": "conditional",
                "geography": "fixed national airport proxy, not zone average",
                "rule": "observed <= origin-2h; receipt <= origin when recorded; no original vintage proof",
                "transformation": "Fahrenheit to Celsius; age; one report per hourly bin for 24h history",
                "parameters": {
                    key: config[key]
                    for key in [
                        "reporting_buffer_hours",
                        "max_observation_age_hours",
                        "mean_window_hours",
                        "mean_min_hours",
                    ]
                },
            }
        )
    write_json(output / "feature_registry.json", registry)
    write_json(output / "provenance.json", identity)
    write_json(
        output / "complete.json",
        {
            str(path.relative_to(output)): digest(path)["sha256"]
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
    )
