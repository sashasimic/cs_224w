"""Review frozen airport results; no fitting, model selection or GPU access.

Compute prespecified geography subgroups and post-hoc market contributions.
Keep the trained package unchanged so its fingerprint remains executable.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from epgnn.dataset_build.storage import digest
from epgnn.experiment_suite.artifacts import write_json

ROOT = Path(__file__).resolve().parents[2]
PREFIX = ROOT / "results/airport_temperature_full"
MARKER = "## Reviewed interpretation"
GROUPS = ["recent", "airport_latest", "airport_history"]
MODELS = [("pooled_xgb", "none"), ("mlp", "none"), ("gat", "real")]


def review():
    summary = json.loads(PREFIX.with_suffix(".json").read_text())
    verification = json.loads(Path(str(PREFIX) + "_verification.json").read_text())
    report_sources = json.loads(Path(str(PREFIX) + "_report_sources.json").read_text())
    if (
        summary["fingerprint"]
        != "8be91e76858aa68c606def353b59030b5b96c6ef94904da15287b663779e4d6b"
        or not verification["passed"]
        or verification["job_count"] != 45
        or verification["fingerprint"] != summary["fingerprint"]
        or digest(PREFIX.with_suffix(".json"))["sha256"]
        != report_sources["summary_sha256"]
    ):
        raise ValueError("Matching verified 45-job results required")
    frame = pd.read_csv(str(PREFIX) + "_by_zone.csv")
    frame = frame[frame.coverage.eq("all")].copy()
    geography = pd.read_csv(ROOT / "results/airport_temperature_geography.csv")
    mapped = set(geography.loc[geography.station.notna(), "zone"])
    frame["airport_mapped"] = frame.zone.isin(mapped)
    cache = Path(report_sources["cache"])
    metadata = json.loads((cache / "metadata.json").read_text())
    values = np.load(cache / "x.npy", mmap_mode="r")
    weather = [
        metadata["features"].index(name)
        for name in [
            "airport_temp_c",
            "airport_temp_age_h",
            "airport_temp_mean24_c",
            "airport_temp_hours24",
        ]
    ]
    unmapped = sorted(set(frame.zone) - mapped)
    for zone in unmapped:
        j = metadata["zones"].index(zone)
        if not np.isnan(np.asarray(values[:, j, :])[:, weather]).all():
            raise ValueError(
                f"Unexpected local airport data for unmapped market {zone}"
            )
    subgroup_rows = []
    for label, mask in [
        ("all", np.ones(len(frame), bool)),
        ("airport_mapped", frame.airport_mapped.to_numpy()),
        ("airport_unmapped", ~frame.airport_mapped.to_numpy()),
    ]:
        selected = frame.loc[mask].copy()
        selected["absolute_error_sum"] = selected.mae * selected.hours
        for (variant, split, seed), group in selected.groupby(
            ["variant", "split", "seed"]
        ):
            subgroup_rows.append(
                {
                    "variant": variant,
                    "split": split,
                    "seed": seed,
                    "subgroup": label,
                    "hours": int(group.hours.sum()),
                    "mae": float(group.absolute_error_sum.sum() / group.hours.sum()),
                }
            )
    subgroup = pd.DataFrame(subgroup_rows)
    means = (
        subgroup.groupby(["variant", "split", "subgroup"])
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            seeds=("seed", "nunique"),
            hours_per_seed=("hours", "first"),
        )
        .reset_index()
    )
    means.to_csv(str(PREFIX) + "_geographic_subgroups.csv", index=False)
    keys = ["zone", "split", "seed", "model"]
    baseline = frame[frame.features.eq("recent")][keys + ["hours", "mae"]]
    additions = frame[~frame.features.eq("recent")][keys + ["features", "hours", "mae"]]
    paired = additions.merge(
        baseline, on=keys, suffixes=("_after", "_before"), validate="many_to_one"
    )
    np.testing.assert_array_equal(paired.hours_after, paired.hours_before)
    totals = (
        baseline.groupby(["split", "seed", "model"]).hours.sum().rename("all_hours")
    )
    paired = paired.join(totals, on=["split", "seed", "model"])
    paired["mae_improvement"] = paired.mae_before - paired.mae_after
    paired["contribution_to_overall_mae"] = (
        paired.mae_improvement * paired.hours_before / paired.all_hours
    )
    contributions = (
        paired.groupby(["zone", "split", "model", "features"])
        .agg(
            mae_improvement_mean=("mae_improvement", "mean"),
            contribution_to_overall_mae=("contribution_to_overall_mae", "mean"),
            hours_per_seed=("hours_before", "first"),
            seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    contributions["airport_mapped"] = contributions.zone.isin(mapped)
    contributions.to_csv(str(PREFIX) + "_market_contributions.csv", index=False)

    def metric(variant, subgroup="all", split="test"):
        return float(
            means.loc[
                (means.variant == variant)
                & (means.subgroup == subgroup)
                & (means.split == split),
                "mae_mean",
            ].iloc[0]
        )

    for recorded in summary["scores"]:
        if recorded["scope"] == "overall" and recorded["coverage"] == "all":
            np.testing.assert_allclose(
                metric(recorded["variant"], split=recorded["split"]),
                recorded["mae_mean"],
                rtol=0,
                atol=1e-9,
            )
    for (split, model, features), group in contributions.groupby(
        ["split", "model", "features"]
    ):
        graph = "real" if model == "gat" else "none"
        expected = metric(f"{model}_recent_{graph}", split=split) - metric(
            f"{model}_{features}_{graph}", split=split
        )
        np.testing.assert_allclose(
            group.contribution_to_overall_mae.sum(), expected, rtol=0, atol=1e-9
        )

    test_xgb = contributions[
        (contributions.split == "test")
        & contributions.model.eq("pooled_xgb")
        & contributions.features.eq("airport_latest")
    ]
    biggest = test_xgb.sort_values("contribution_to_overall_mae", ascending=False).iloc[
        0
    ]
    total_improvement = metric("pooled_xgb_recent_none") - metric(
        "pooled_xgb_airport_latest_none"
    )
    if biggest.airport_mapped:
        raise ValueError(
            "The frozen review interpretation requires an unmapped largest contributor"
        )
    share = float(biggest.contribution_to_overall_mae / total_improvement)
    lines = [
        MARKER,
        "",
        (
            "The full run finished on 8 November 2026 at 20:48 Europe/Zurich, "
            "with exit code 0, after approximately 64 minutes. All 45 fits passed saved-artifact verification. "
            "The following subgroup and contribution checks are descriptive post-hoc analyses; "
            "they do not change the evaluation cohort, models or hyperparameters."
        ),
        "",
        (
            "**The airport proxy gives small, inconsistent improvements.** "
            "For latest temperature, validation-week intervals include zero for pooled XGBoost and MLP. "
            "GAT's interval is narrowly positive, but these exploratory intervals are unadjusted for multiple comparisons. "
            "Adding the trailing mean does not establish a clear advantage over the latest reading."
        ),
        "",
        (
            "Test MAE below uses all valid labels within each fixed geography subgroup. "
            "Each number is a five-seed mean, weighted by target hours within a seed."
        ),
        "",
        "| Model | Subgroup | Price-only | + Latest | + History |",
        "|---|---|---:|---:|---:|",
    ]
    for model, graph in MODELS:
        for label in ["all", "airport_mapped", "airport_unmapped"]:
            numbers = [metric(f"{model}_{group}_{graph}", label) for group in GROUPS]
            lines.append(
                f"| {model} | {label} | "
                + " | ".join(f"{value:.4f}" for value in numbers)
                + " |"
            )
    lines += [
        "",
        (
            f"**Why the pooled XGBoost headline needs care:** `{biggest.zone}` accounts for "
            f"{biggest.contribution_to_overall_mae:.4f} EUR/MWh of its {total_improvement:.4f} "
            f"overall test improvement with latest temperature—about {100 * share:.1f}%. "
            "This market has no resolved airport mapping, and all four of its local airport inputs "
            "were verified to be missing throughout the cached calendar. "
            "Its changed predictions therefore cannot be attributed to a local temperature reading: "
            "adding features changes the jointly fitted pooled model, including its learned missing-value routes. "
            "Unmapped-node GAT predictions can additionally change through messages from mapped neighbours."
        ),
        "",
        (
            "France worsens with this airport proxy for all three models. Germany/Luxembourg shows small improvements "
            "with latest temperature; effects vary across the other markets. "
            "These results do not establish that national or regional temperature is useless, "
            "nor that a single airport adequately represents a decentralised country."
        ),
        "",
        (
            "GAT remains ahead of the matched pooled MLP overall with and without airport features. "
            "France remains a counterexample. The earlier rich-feature results used additional conditional sources "
            "and are context, not a direct test of observed temperature versus ERA5. "
            "No training was rerun or selected using these post-hoc checks."
        ),
        "",
        (
            "The review tables are saved as `airport_temperature_full_geographic_subgroups.csv` "
            "and `airport_temperature_full_market_contributions.csv`; "
            "`scripts/experiments/review_airport_results.py` reproduces them."
        ),
        "",
    ]
    report = ROOT / "docs/airport_temperature_results.md"
    original = report.read_text()
    if MARKER in original:
        first, remainder = original.split(MARKER, 1)
        original = (
            first + "## Overall errors" + remainder.split("## Overall errors", 1)[1]
        )
    original = original.replace(
        "for the causal feature comparison", "for the matched feature comparison"
    )
    original = original.replace(
        "## Overall errors", "\n".join(lines) + "\n## Overall errors", 1
    )
    report.write_text(original)
    evidence = {
        "fingerprint": summary["fingerprint"],
        "all_aggregate_scores_and_contribution_sums_reconciled": True,
        "reviewed_report_sha256": digest(report)["sha256"],
        "training_rerun": False,
        "analysis": "post-hoc descriptive geography/contribution review; no model selection",
        "summary_sha256": digest(PREFIX.with_suffix(".json"))["sha256"],
        "zone_scores_sha256": digest(Path(str(PREFIX) + "_by_zone.csv"))["sha256"],
        "script_sha256": digest(Path(__file__))["sha256"],
        "unmapped_markets_all_local_temperature_inputs_missing": unmapped,
        "largest_xgb_latest_test_contributor": str(biggest.zone),
        "contribution_eur_mwh": float(biggest.contribution_to_overall_mae),
        "overall_improvement_eur_mwh": total_improvement,
        "share_of_overall_improvement": share,
    }
    write_json(str(PREFIX) + "_review.json", evidence)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    review()
