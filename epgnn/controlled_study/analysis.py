"""Training-defined regimes, seed summaries and spatially paired week bootstrap."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from epgnn.dataset_build.contracts import TZ
from epgnn.experiment_suite.artifacts import write_json


def clean_records(frame):
    return frame.astype(object).where(pd.notna(frame), None).to_dict("records")


def country(zone):
    return zone.split("_")[0] if zone.startswith(("IT_", "NO_", "SE_", "DK_")) else zone


def diagnostics(predictions_path, arrays, metadata, zones):
    """Use evaluation outcomes only to score predeclared groups, never as inputs."""
    frame = pd.read_parquet(predictions_path)
    local = frame.ts_utc.dt.tz_convert(TZ)
    frame["week"] = local.dt.tz_localize(None).dt.to_period("W-SUN").astype(str)
    frame["season"] = local.dt.month.map(
        {
            12: "DJF",
            1: "DJF",
            2: "DJF",
            3: "MAM",
            4: "MAM",
            5: "MAM",
            6: "JJA",
            7: "JJA",
            8: "JJA",
            9: "SON",
            10: "SON",
            11: "SON",
        }
    )
    valid = frame.target_valid.to_numpy(bool)
    residual = frame.prediction.to_numpy(float) - frame.price.to_numpy(float)
    frame["absolute_error"] = np.where(valid, np.abs(residual), np.nan)
    frame["squared_error"] = np.where(valid, residual**2, np.nan)
    paired = valid & np.isfinite(frame.price_lag168.to_numpy(float))
    weekly = (
        frame.loc[paired]
        .groupby("week")
        .absolute_error.agg(["sum", "size"])
        .reset_index()
    )
    rows, thresholds = [], {}
    for index in zones:
        zone = metadata["zones"][index]
        training = np.asarray(arrays["y"][arrays["split"] == 0, index])
        finite = training[np.isfinite(training)]
        low, high = np.quantile(finite, [0.05, 0.95])
        thresholds[zone] = {"q05": float(low), "q95": float(high)}
        selected = frame.zone.eq(zone).to_numpy() & valid
        masks = {
            "all": selected,
            "negative": selected & frame.price.lt(0).to_numpy(),
            "lower_tail": selected & frame.price.lt(low).to_numpy(),
            "upper_tail": selected & frame.price.gt(high).to_numpy(),
        }
        masks.update(
            {
                season: selected & frame.season.eq(season).to_numpy()
                for season in ["DJF", "MAM", "JJA", "SON"]
            }
        )
        for label, mask in masks.items():
            rows.append(
                {
                    "zone": zone,
                    "country": country(zone),
                    "slice": label,
                    "hours": int(mask.sum()),
                    "absolute_error_sum": float(np.abs(residual[mask]).sum()),
                    "squared_error_sum": float(np.square(residual[mask]).sum()),
                }
            )
    return rows, weekly, thresholds


def week_interval(before, after, repetitions=2000, seed=224):
    if not before.week.equals(after.week) or not np.array_equal(
        before["size"], after["size"]
    ):
        raise ValueError("Paired bootstrap requires identical validation weeks/counts")
    difference = before["sum"].to_numpy(float) - after["sum"].to_numpy(float)
    counts = before["size"].to_numpy(float)
    if not counts.sum():
        raise ValueError("No paired validation observations")
    draws = np.random.default_rng(seed).integers(
        0, len(counts), size=(repetitions, len(counts))
    )
    improvement = difference[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {
        "mean_mae_improvement": float(difference.sum() / counts.sum()),
        "ci95_lower": float(np.quantile(improvement, 0.025)),
        "ci95_upper": float(np.quantile(improvement, 0.975)),
        "blocks": len(counts),
        "repetitions": repetitions,
        "interpretation": "Paired calendar-week uncertainty conditional on all fitted models; weeks shared across countries",
    }


def mean_weekly(frames):
    first = frames[0].copy()
    for frame in frames[1:]:
        if not first.week.equals(frame.week) or not np.array_equal(
            first["size"], frame["size"]
        ):
            raise ValueError("Models or seeds have different target masks")
    first["sum"] = np.stack([frame["sum"].to_numpy() for frame in frames]).mean(axis=0)
    return first


def summarize(directory, results, config, output_prefix, token, smoke=False):
    """Aggregate fixed-model scores, never average predictions or select test seeds."""
    directory = Path(directory)
    aggregate_rows, zone_rows, slice_rows = [], [], []
    for result in results:
        tags = {"variant": result["variant"], **result["job"]}
        aggregate_rows.extend({**row, **tags} for row in result["aggregate"])
        zone_rows.extend({**row, **tags} for row in result["rows"])
        slice_rows.extend({**row, **tags} for row in result["slices"])
    aggregate = pd.DataFrame(aggregate_rows)
    measures = ["mae", "rmse", "paired_mae", "persistence_mae", "relative_mae"]
    group_keys = ["variant", "split", "coverage", "scope", "country"]
    if "country" not in aggregate:
        aggregate["country"] = "ALL"
    aggregate["country"] = aggregate.country.fillna("ALL")
    means = (
        aggregate.groupby(group_keys, dropna=False)[measures]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    means.columns = [
        "_".join(filter(None, column)) if isinstance(column, tuple) else column
        for column in means.columns
    ]
    slices = pd.DataFrame(slice_rows)
    pooled = []
    for keys in [
        ["variant", "seed", "split", "slice"],
        ["variant", "seed", "split", "slice", "country"],
        ["variant", "seed", "split", "slice", "zone"],
    ]:
        frame = (
            slices.groupby(keys)[["hours", "absolute_error_sum", "squared_error_sum"]]
            .sum()
            .reset_index()
        )
        frame["mae"] = frame.absolute_error_sum / frame.hours.replace(0, np.nan)
        frame["rmse"] = np.sqrt(
            frame.squared_error_sum / frame.hours.replace(0, np.nan)
        )
        frame["scope"] = (
            "zone" if "zone" in keys else "country" if "country" in keys else "overall"
        )
        pooled.append(frame)
    slices = pd.concat(pooled, ignore_index=True)
    intervals = {}
    contrasts = [
        ("mlp_rich_none", "gat_rich_real"),
        ("gat_rich_self", "gat_rich_real"),
        ("gru_mlp_rich_none", "gru_gat_rich_real"),
    ]
    contrasts += [
        (f"gat_rich_shuffled_rewire{seed}", "gat_rich_real")
        for seed in config["rewiring_seeds"]
    ]
    contrasts += [
        (f"{model}_recent_{graph}", f"{model}_{group}_{graph}")
        for model, graph in [("mlp", "none"), ("gat", "real")]
        for group in ["load", "weather", "jao", "rich"]
    ]
    variants = {item["variant"] for item in results}
    for before, after in contrasts:
        if before not in variants or after not in variants:
            continue
        first = mean_weekly(
            [
                pd.read_parquet(directory / item["name"] / "weekly_val.parquet")
                for item in results
                if item["variant"] == before
            ]
        )
        second = mean_weekly(
            [
                pd.read_parquet(directory / item["name"] / "weekly_val.parquet")
                for item in results
                if item["variant"] == after
            ]
        )
        intervals[f"{before}_to_{after}"] = week_interval(
            first,
            second,
            config["bootstrap"]["repetitions"],
            config["bootstrap"]["seed"],
        )
    shuffled = aggregate[
        (aggregate.graph == "shuffled")
        & (aggregate.scope == "overall")
        & (aggregate.coverage == "all")
    ]
    rewiring = []
    for split, frame in shuffled.groupby("split"):
        matrix = frame.pivot(index="seed", columns="rewiring_seed", values="mae")
        paired_matrix = frame.pivot(
            index="seed", columns="rewiring_seed", values="paired_mae"
        )
        rewiring.append(
            {
                "split": split,
                "mean_mae": float(matrix.to_numpy().mean()),
                "paired_mae_mean": float(paired_matrix.to_numpy().mean()),
                "paired_mae_training_sd": (
                    float(paired_matrix.mean(axis=1).std())
                    if len(paired_matrix) > 1
                    else None
                ),
                "training_seed_sd_after_averaging_rewirings": (
                    float(matrix.mean(axis=1).std()) if len(matrix) > 1 else None
                ),
                "mean_rewiring_sd_within_training_seed": float(
                    matrix.std(axis=1).mean()
                ),
            }
        )
    summary = {
        "fingerprint": token,
        "phase": "smoke" if smoke else "full",
        "complete": True,
        "job_count": len(results),
        "seeds": sorted({r["job"]["seed"] for r in results}),
        "jobs": [
            {k: r[k] for k in ["name", "variant", "job", "seconds", "details"]}
            for r in results
        ],
        "scores": clean_records(means),
        "validation_intervals": intervals,
        "rewiring_variability": rewiring,
        "limitations": [
            "Exploratory test period inspected previously",
            "Conditional source publication vintages",
            "Weekly bootstrap conditions on fitted models; seed/rewiring summaries are separate",
            "Means are scores, not ensembles",
        ],
    }
    write_json(directory / "summary.json", summary)
    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)
    write_json(str(output_prefix) + ".json", summary)
    means.to_csv(str(output_prefix) + "_scores.csv", index=False)
    pd.DataFrame(zone_rows).to_csv(str(output_prefix) + "_by_zone.csv", index=False)
    regime_keys = ["variant", "split", "slice", "scope", "country", "zone"]
    regimes = (
        slices.groupby(regime_keys, dropna=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            hours_per_seed=("hours", "first"),
            seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    regimes.to_csv(str(output_prefix) + "_regimes.csv", index=False)
    slices.to_csv(directory / "regime_seeds.csv", index=False)
    return summary
