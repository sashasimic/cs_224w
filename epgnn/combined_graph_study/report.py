"""Summarize matched validation forecasts and publish paired comparisons."""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from epgnn.dataset_build.storage import ROOT, json_write

from .runner import CONTRASTS
from .training import TRAIN_START, VARIANTS


def intervals(frame, predictions, repetitions):
    outcomes = frame.price.to_numpy()
    weeks = (
        frame.ts_utc.dt.tz_convert("Europe/Brussels")
        .dt.tz_localize(None)
        .dt.to_period("W-SUN")
    )
    results = []
    for scope, keep in (
        ("all", frame.target_valid.to_numpy()),
        ("core", (frame.target_valid & frame.core_covered).to_numpy()),
    ):
        week = weeks[keep].to_numpy()
        labels = outcomes[keep]
        loss = {
            model: np.mean(np.abs(np.asarray(values)[:, keep] - labels), axis=0)
            for model, values in predictions.items()
        }
        for model, control in CONTRASTS:
            delta = loss[control] - loss[model]
            grouped = pd.Series(delta).groupby(week).agg(["sum", "size"])
            rng = np.random.default_rng(224)
            sample = rng.integers(len(grouped), size=(repetitions, len(grouped)))
            draws = grouped["sum"].to_numpy()[sample].sum(axis=1) / grouped[
                "size"
            ].to_numpy()[sample].sum(axis=1)
            ci = np.quantile(draws, [0.025, 0.975])
            results.append(
                {
                    "scope": scope,
                    "model": model,
                    "control": control,
                    "improvement": float(delta.mean()),
                    "ci95": ci.tolist(),
                    "n": int(keep.sum()),
                    "weeks": len(grouped),
                }
            )
    return results


def link_summary(directory, data, results):
    rows = []
    for variant in ("physical", "uniform", "learned", "margin", "shuffled_margin"):
        matched = [job for job in results if job["variant"] == variant]
        seeds = [job["seed"] for job in matched]
        weights = np.stack(
            [
                np.load(directory / f"{variant}_{seed}" / "link_weights.npy")
                for seed in seeds
            ]
        )
        for target, zone in enumerate(data.zones):
            edges = []
            for seed_weights in weights[:, :, target]:
                rank = np.argsort(-seed_weights, kind="stable")
                edges.append(set(rank[seed_weights[rank] > 0][:5]))
            agreement = [
                len(a & b) / len(a | b)
                for a, b in itertools.combinations(edges, 2)
                if a or b
            ]
            avg = weights[:, :, target].mean(axis=0)
            ordered = np.argsort(-avg, kind="stable")
            leaders = [i for i in ordered[:5] if avg[i] > 0]
            rows.append(
                {
                    "variant": variant,
                    "zone": zone,
                    "top_neighbors": [data.zones[i] for i in leaders],
                    "top_weights": [float(avg[i]) for i in leaders],
                    "mean_seed_jaccard": float(np.mean(agreement))
                    if agreement
                    else None,
                }
            )
    return rows


def publish(settings, data, output, common, results, predictions):
    contrast = intervals(common, predictions, settings["bootstrap_repetitions"])
    links = link_summary(output, data, results)
    records = []
    for result in results:
        for scope, score in result["scores"].items():
            records.append(
                {
                    "variant": result["variant"],
                    "family": result["family"],
                    "seed": result["seed"],
                    "scope": scope,
                    "spread_mae": result["spread_mae"] if scope == "all" else np.nan,
                    **score,
                }
            )
    metrics = pd.DataFrame(records)
    path = ROOT / "results/combined_graph_validation"
    metrics.to_csv(path.with_suffix(".csv"), index=False, float_format="%.7f")
    peak = max(
        epoch["peak_vram_gib"]
        for j in results
        if j["family"] != "xgb"
        for phase in ("calibration", "refit")
        for epoch in j["training"][phase]
    )
    json_write(
        path.with_suffix(".json"),
        {
            "run": output.name,
            "jobs": len(results),
            "keys_per_job": len(common),
            "contrasts": contrast,
            "links": links,
            "peak_vram_gib": peak,
            "manifests": [
                {
                    "variant": job["variant"],
                    "seed": job["seed"],
                    "train_start": job["train_start"],
                    "train_end": job["train_end"],
                    "prediction_sha256": job["prediction_sha256"],
                    "selected": job["training"]["selected"],
                }
                for job in results
            ],
        },
    )
    manifest = Path(output) / "manifest.json"
    path.with_name(path.name + "_manifest.json").write_text(manifest.read_text())
    group = metrics.groupby(["variant", "scope"])
    means, sd = group.mean(numeric_only=True), group.mae.std()
    lines = [
        "# Combined graph validation",
        "",
        (
            f"{len(results)} jobs used the same {len(common):,} zone-hour keys from April 2024 through March 2025. "
            "MAE is in EUR/MWh and averaged over five seeds. Skill is the reduction in MAE versus weekly persistence on matched rows. "
            "JAO, weather, and archived load inputs are conditional."
        ),
        "",
        "| Model | Fit begins | All MAE | CORE MAE | France MAE | All skill | CORE skill | Spread MAE |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in VARIANTS:
        family = metrics.loc[metrics.variant.eq(model), "family"].iat[0]
        overall = means.loc[(model, "all"), :]
        core = means.loc[(model, "core"), :]
        fr = means.loc[(model, "FR"), "mae"]
        lines.append(
            f"| {model} | {TRAIN_START[family]} | {overall.mae:.3f} ± {sd.loc[(model, 'all')]:.3f} | "
            f"{core.mae:.3f} | {fr:.3f} | {overall.persistence_skill:.1%} | "
            f"{core.persistence_skill:.1%} | {overall.spread_mae:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Paired contrasts",
            "",
            "Positive values favor the first model. Intervals resample Brussels calendar weeks jointly across markets.",
            "",
            "| Cohort | Model | Control | MAE improvement | 95% interval |",
            "|---|---|---|---:|---|",
        ]
    )
    for c in contrast:
        a, b = c["ci95"]
        lines.append(
            f"| {c['scope']} | {c['model']} | {c['control']} | {c['improvement']:+.4f} | [{a:+.4f}, {b:+.4f}] |"
        )
    lines.extend(
        [
            "",
            f"Peak allocated GPU tensors were {peak:.2f} GiB.",
            "Machine-readable summaries include mean weights and top-five neighbor stability across seeds.",
            "",
        ]
    )
    (ROOT / "docs/combined_graph_validation.md").write_text("\n".join(lines))
    print(path.with_suffix(".json"), flush=True)
