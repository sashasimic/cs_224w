#!/usr/bin/env python3
"""Verify a completed CORE validation run and publish compact tracked results."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from epgnn.dataset_build.storage import digest, json_write

LABELS = {
    "interconnector": "Interconnector GAT",
    "flat_jao": "Interconnector GAT + flat JAO",
    "bipartite": "Bipartite GAT + flat JAO",
    "shuffled": "Bipartite GAT + flat JAO, shuffled links",
}
KEY_COLUMNS = ["ts_utc", "zone", "price", "target_valid", "weekly", "core_covered"]


def verify_predictions(directory, jobs):
    reference = None
    for job in jobs:
        stem = f"{job['variant']}_{job['seed']}"
        frame = pd.read_parquet(directory / stem / "predictions.parquet")
        if frame.duplicated(["ts_utc", "zone"]).any():
            raise ValueError(f"Duplicate target keys in {stem}")
        if not np.isfinite(frame.prediction).all():
            raise ValueError(f"Missing predictions in {stem}")
        if not np.array_equal(frame.target_valid, np.isfinite(frame.price)):
            raise ValueError(f"Inconsistent target validity in {stem}")
        if reference is None:
            reference = frame[KEY_COLUMNS]
        elif not reference.equals(frame[KEY_COLUMNS]):
            raise ValueError(f"Target rows or masks differ in {stem}")
    return len(reference)


def report(directory, output):
    result_path = directory / "results.json"
    result = json.loads(result_path.read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    expected = {
        (model, seed) for model in LABELS for seed in manifest["settings"]["seeds"]
    }
    jobs = result["jobs"]
    if (
        manifest["smoke"]
        or len(jobs) != len(expected)
        or {(j["variant"], j["seed"]) for j in jobs} != expected
    ):
        raise ValueError("A completed full validation run is required")
    row_count = verify_predictions(directory, jobs)
    metrics = pd.DataFrame(result["metrics"])
    group = metrics.groupby(["variant", "scope"], sort=False)
    counts = group[["n", "paired_n"]]
    if (counts.nunique() != 1).any().any():
        raise ValueError("Evaluation support differs across seeds")
    summary = group.agg(
        hours=("n", "first"),
        paired_hours=("paired_n", "first"),
        mae=("mae", "mean"),
        seed_sd=("mae", "std"),
        rmse=("rmse", "mean"),
        paired_mae=("paired_mae", "mean"),
        persistence_mae=("persistence_mae", "mean"),
        persistence_skill=("persistence_skill", "mean"),
    ).reset_index()
    peak = max(epoch["peak_vram_gib"] for j in jobs for epoch in j["epochs"])
    json_write(
        output / "results/core_constraint_validation.json",
        {
            "run_fingerprint": directory.name,
            "catalog_fingerprint": manifest["catalog"],
            "dataset_fingerprint": manifest["settings"]["dataset_fingerprint"],
            "source_results_sha256": digest(result_path)["sha256"],
            "seed_count": len(manifest["settings"]["seeds"]),
            "jobs": len(jobs),
            "matched_prediction_rows_per_job": row_count,
            "publication_admission": result["admission"],
            "contrasts": result["contrasts"],
            "peak_vram_gib": peak,
        },
    )
    summary.to_csv(
        output / "results/core_constraint_validation.csv",
        index=False,
        float_format="%.6f",
    )
    aligned = summary.set_index(["variant", "scope"])
    lines = [
        "# CORE constraint GAT validation results",
        "",
        "[Schema and experimental protocol](relational_forecast.md)",
        "",
        (
            f"Four variants and {len(manifest['settings']['seeds'])} seeds used the same {row_count:,} "
            "zone-hour keys from April 2024 through March 2025. Values are mean validation MAE "
            "across seeds in EUR/MWh. Skill uses the same valid hours as weekly persistence. "
            "JAO publication vintages remain unverified."
        ),
        "",
        "| Variant | All markets MAE | CORE-covered MAE | France MAE | CORE skill vs. persistence |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, label in LABELS.items():
        overall, core, france = (
            aligned.loc[model, scope] for scope in ("all", "core_covered", "FR")
        )
        lines.append(
            f"| {label} | {overall.mae:.2f} ± {overall.seed_sd:.2f} | "
            f"{core.mae:.2f} ± {core.seed_sd:.2f} | {france.mae:.2f} | {core.persistence_skill:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Paired CORE comparisons",
            "",
            (
                "Positive differences favor the first model. Intervals use a paired Brussels calendar-week "
                "bootstrap and condition on the fitted models."
            ),
            "",
        ]
    )
    for comparison in result["contrasts"]:
        if comparison["scope"] != "core_covered":
            continue
        lo, hi = comparison["ci95"]
        lines.append(
            f"- {LABELS[comparison['model']]} versus {LABELS[comparison['control']]} "
            f"{comparison['mae_improvement']:+.3f} EUR/MWh with 95% interval [{lo:+.3f}, {hi:+.3f}]."
        )
    core = aligned.loc["bipartite", "core_covered"]
    overall = aligned.loc["bipartite", "all"]
    lines.extend(
        [
            "",
            (
                f"The CORE-covered cohort has {int(core.hours):,} valid targets. Across all markets there "
                f"are {int(overall.hours):,} valid targets. Weekly persistence MAE on its matched hours is "
                f"{core.persistence_mae:.2f} on CORE coverage and {overall.persistence_mae:.2f} across all markets. "
                f"Peak GPU allocation was {peak:.2f} GiB."
            ),
            "",
            (
                "Detailed [zone scores](../results/core_constraint_validation.csv) and "
                "[comparison metadata](../results/core_constraint_validation.json) are available."
            ),
            "",
        ]
    )
    (output / "docs/core_constraint_validation.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation_run", type=Path)
    parser.add_argument("--output", type=Path, default=Path.cwd())
    args = parser.parse_args()
    report(args.validation_run.resolve(), args.output)
