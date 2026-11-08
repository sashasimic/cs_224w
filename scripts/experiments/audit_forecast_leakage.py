"""Read-only model/data checks; write separate leakage findings, never refit models."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch

from epgnn.controlled_study.design import FEATURE_SETS, jobs, feature_indices
from epgnn.controlled_study.history import Histories, history_index
from epgnn.controlled_study.models import make_model
from epgnn.dataset_build.contracts import forecast_origin, TZ
from epgnn.dataset_build.storage import ROOT, digest
from epgnn.experiment_suite.data import load, neighbour_features, guard_price24
from epgnn.experiment_suite.artifacts import write_json


def main():
    config = json.loads(
        (ROOT / "configs/experiments/controlled_study.json").read_text()
    )
    summary = json.loads((ROOT / "results/controlled_study_full.json").read_text())
    directory = ROOT / "runs/controlled_study" / summary["fingerprint"] / "full"
    provenance = json.loads((directory / "provenance.json").read_text())
    meta, arrays = load(provenance["cache"])
    cache_directory = Path(provenance["cache"])
    for name, sha in json.loads(
        (cache_directory / "complete.json").read_text()
    ).items():
        if digest(cache_directory / name)["sha256"] != sha:
            raise ValueError("Input cache artifact changed: " + name)
    calendar = pd.to_datetime(np.asarray(arrays["times"]), unit="ns", utc=True)
    origins = pd.DatetimeIndex(forecast_origin(pd.Series(calendar)))
    train, val, test = [
        np.flatnonzero(arrays["split"] == number) for number in range(3)
    ]
    result = {
        "study_fingerprint": summary["fingerprint"],
        "cache": provenance["cache"],
        "code_identity": provenance["study_code"],
        "checks": {},
        "source_risks": {},
    }
    for name, sha in {
        **provenance["suite_provenance"]["code"],
        **provenance["study_code"],
    }.items():
        if digest(ROOT / name)["sha256"] != sha:
            raise ValueError(
                "Current implementation differs from the studied source: " + name
            )
    result["checks"]["source_matches_trained_provenance"] = True
    result["checks"]["cache_artifact_checksums_match"] = True
    result["checks"]["split"] = {
        "train_last": str(calendar[train[-1]]),
        "val_first": str(calendar[val[0]]),
        "val_last": str(calendar[val[-1]]),
        "test_first": str(calendar[test[0]]),
        "disjoint_ordered": bool(train[-1] < val[0] and val[-1] < test[0]),
        "unassigned_not_fitted": bool(
            not np.intersect1d(train, np.flatnonzero(arrays["split"] == 3)).size
        ),
    }
    local_dates = calendar.tz_convert(TZ).strftime("%Y-%m-%d").to_numpy()
    active = np.asarray(arrays["active"])
    for hours, feature in [(24, "price_lag24_guarded"), (168, "price_lag168")]:
        required = calendar.asi8 - hours * 3_600_000_000_000
        pos = np.searchsorted(calendar.asi8, required)
        safe = np.minimum(pos, len(calendar) - 1)
        present = (pos < len(calendar)) & (calendar.asi8[safe] == required)
        expected = np.asarray(arrays["y"][safe]).copy()
        expected[~present] = np.nan
        if hours == 24:
            source_dates = (
                pd.to_datetime(required, unit="ns", utc=True)
                .tz_convert(TZ)
                .strftime("%Y-%m-%d")
                .to_numpy()
            )
            expected[source_dates >= local_dates] = np.nan
        expected[~active] = np.nan
        actual = np.asarray(arrays["x"][:, :, meta["features"].index(feature)])
        unequal = ~np.isclose(actual, expected, rtol=0, atol=1e-5, equal_nan=True)
        result["checks"][feature] = {
            "mismatches": int(unequal.sum()),
            "finite_values": int(np.isfinite(actual).sum()),
        }
    weather_age = (
        origins.asi8 - (calendar.asi8 - 48 * 3_600_000_000_000)
    ) / 3_600_000_000_000
    weather_checks = {}
    for name in ["temp_pw_lag48", "temp_anom_lag48", "wind_cf_lag48", "solar_cf_lag48"]:
        values = np.asarray(arrays["x"][test, :, meta["features"].index(name)])
        finite = np.isfinite(values)
        age = np.broadcast_to(weather_age[test, None], finite.shape)
        weather_checks[name] = {
            "finite_test_values": int(finite.sum()),
            "younger_than_5_days": int((finite & (age < 120)).sum()),
            "minimum_age_hours": float(age[finite].min()),
            "maximum_age_hours": float(age[finite].max()),
        }
    result["source_risks"]["weather_5_day_latency_screen"] = weather_checks
    result["source_risks"]["jobs_using_weather"] = sum(
        any(
            "temp_" in f or "wind_" in f or "solar_" in f
            for f in FEATURE_SETS[job.features]
        )
        for job in jobs(config)
    )
    nodes = ds.dataset(ROOT / config["dataset_directory"] / "nodes")
    fields = [
        "zone",
        "ts_utc",
        "load_source_IsAvailabilityVerified",
        "load_source_SourceUpdateTimeMinUTC",
        "load_source_SourceUpdateTimeMaxUTC",
        "load_fc",
    ]
    load_counts = {
        "finite": 0,
        "availability_verified": 0,
        "nonmissing_update_max": 0,
        "update_max_after_origin": 0,
        "update_min_after_origin": 0,
        "examples": [],
    }
    for batch in nodes.scanner(
        columns=fields,
        filter=ds.field("split").isin(["val", "test"]),
        batch_size=131072,
    ).to_batches():
        frame = batch.to_pandas()
        finite = frame.load_fc.notna()
        stamp = pd.to_datetime(frame.load_source_SourceUpdateTimeMaxUTC, utc=True)
        first_stamp = pd.to_datetime(frame.load_source_SourceUpdateTimeMinUTC, utc=True)
        origin = forecast_origin(frame.ts_utc)
        verified = (
            frame.load_source_IsAvailabilityVerified.astype("string")
            .str.lower()
            .isin(["true", "1"])
        )
        load_counts["finite"] += int(finite.sum())
        load_counts["availability_verified"] += int((finite & verified).sum())
        load_counts["nonmissing_update_max"] += int((finite & stamp.notna()).sum())
        load_counts["update_max_after_origin"] += int((finite & (stamp > origin)).sum())
        load_counts["update_min_after_origin"] += int(
            (finite & (first_stamp > origin)).sum()
        )
        if len(load_counts["examples"]) < 5:
            for i in frame.index[finite & (stamp > origin)][
                : 5 - len(load_counts["examples"])
            ]:
                load_counts["examples"].append(
                    {
                        "zone": str(frame.loc[i, "zone"]),
                        "delivery": str(frame.loc[i, "ts_utc"]),
                        "origin": str(origin.loc[i]),
                        "update_min": str(first_stamp.loc[i]),
                        "update_max": str(stamp.loc[i]),
                    }
                )
    result["source_risks"]["load"] = load_counts
    prep_checks = []
    temporal_stats = Histories(arrays).fit(train).state()
    for group in ["recent", "load", "weather", "jao", "rich"]:
        columns = [meta["features"].index(name) for name in FEATURE_SETS[group]]
        values = np.asarray(arrays["x"][train])[:, :, columns][active[train]]
        median = np.array(
            [
                np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0
                for c in values.T
            ],
            np.float32,
        )
        filled = np.where(np.isfinite(values), values, median)
        mean = filled.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = filled.std(axis=0, dtype=np.float64).astype(np.float32)
        scale[scale < 1e-6] = 1
        for job in jobs(config):
            if job.features != group or job.model in {"xgb", "pooled_xgb"}:
                continue
            state = torch.load(
                directory / job.name / "best.pt", map_location="cpu", weights_only=False
            )
            for key, expected in [("median", median), ("mean", mean), ("scale", scale)]:
                np.testing.assert_allclose(
                    state["preprocessing"][key], expected, rtol=0, atol=1e-6
                )
            if job.model.startswith("gru_"):
                for key, value in temporal_stats.items():
                    np.testing.assert_allclose(
                        state["history"][key], value, rtol=0, atol=1e-9
                    )
            prep_checks.append(job.name)
    result["checks"]["checkpoint_preprocessing"] = {
        "checked_neural_checkpoints": len(prep_checks),
        "all_match_training_only_statistics": True,
        "temporal_statistics_match_training_histories": True,
    }
    positions, present, inverse = history_index(calendar)
    last = calendar.asi8[positions[:, -1]] + 3_600_000_000_000
    unique_origins = np.unique(origins.asi8)
    result["checks"]["histories"] = {
        "unique_origins": len(unique_origins),
        "completion_after_origin": int(
            (last[present[:, -1]] > unique_origins[present[:, -1]]).sum()
        ),
        "missing_source_intervals": int((~present).sum()),
    }
    # Counterfactuals exercise the real feature functions on a small UTC fixture.
    times = pd.date_range("2024-03-20", periods=24 * 15, freq="h", tz="UTC").as_unit(
        "ns"
    )
    fake = {
        "times": times.asi8,
        "y": np.arange(len(times) * 2, dtype=np.float32).reshape(-1, 2),
        "active": np.ones((len(times), 2), bool),
    }
    earlier = Histories(fake).fit(np.arange(24 * 10))
    before, _ = earlier.batch([24 * 12])
    fake["y"][24 * 12 - 12 :] = 999999
    later = Histories(fake).fit(np.arange(24 * 10))
    after, _ = later.batch([24 * 12])
    np.testing.assert_array_equal(before, after)
    assert earlier.state() == later.state()
    torch.manual_seed(224)
    model = make_model("gat", 26, config["neural"]).eval()
    x = torch.randn(2, 3, 26)
    edges = torch.tensor([[0, 1, 2], [1, 2, 0]])
    alive = torch.ones(2, 3, dtype=torch.bool)
    with torch.no_grad():
        before = model(x, edges, alive)
        x[1] = 99999
        after = model(x, edges, alive)
    torch.testing.assert_close(before[0], after[0])
    result["checks"]["counterfactuals"] = {
        "future_prices_do_not_change_past_histories_or_statistics": True,
        "future_graph_in_batch_cannot_change_earlier_graph_prediction": True,
    }
    cache_edges = set(map(tuple, np.asarray(arrays["edges"]).T))
    zi = {zone: index for index, zone in enumerate(meta["zones"])}
    observed = set()
    edge_data = ds.dataset(ROOT / config["dataset_directory"] / "edges")
    for batch in edge_data.scanner(
        columns=["from_zone", "to_zone"],
        filter=ds.field("split").isin(["pretrain", "train"])
        & (ds.field("physical_row_present") == True),
        batch_size=131072,
    ).to_batches():
        observed.update(
            (zi[a], zi[b])
            for a, b in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist())
            if a in zi and b in zi and a != b
        )
    result["checks"]["topology"] = {
        "edges": len(cache_edges),
        "exactly_training_observed_pairs": observed == cache_edges,
        "test_observations_used_for_edge_selection": False,
    }
    result["source_risks"]["cohort"] = {
        "eligible_zones": len(meta["eligible"]),
        "uses_validation_and_test_target_presence_to_define_reported_cohort": True,
    }
    result["classification"] = {
        "confirmed_product_timing_problem": "lag-48 ERA5-derived values are younger than the earliest approximately five-day ERA5T availability; exact export lineage remains unverified",
        "unresolved": "load/JAO publication vintages, revised historical prices, weather-derived anomaly climatology and capacity-factor calibration",
        "research_protocol": "test period was repeatedly inspected and influenced subsequent experiment design; no fresh confirmatory holdout",
        "not_found": "direct same-day target input, cross-split preprocessing fit, future GRU intervals, cross-time graph edges, test-derived physical topology",
    }
    output = ROOT / "results/forecast_leakage_audit.json"
    write_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
