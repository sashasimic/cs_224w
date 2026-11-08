"""Publish a standalone report without changing previous results or milestones."""

import json
import shutil
from pathlib import Path

import pandas as pd

from epgnn.controlled_study.analysis import mean_weekly, week_interval
from epgnn.dataset_build.storage import ROOT, digest
from epgnn.experiment_suite.artifacts import write_json

from .data import GROUPS


def publish(root, directory, results, config, token, smoke):
    phase = "smoke" if smoke else "full"
    prefix = ROOT / f"results/airport_temperature_{phase}"
    summary = json.loads((directory / "summary.json").read_text())
    contrasts = []
    for model, graph in [("pooled_xgb", "none"), ("mlp", "none"), ("gat", "real")]:
        contrasts.extend(
            (f"{model}_recent_{graph}", f"{model}_{group}_{graph}")
            for group in ["airport_latest", "airport_history"]
        )
        contrasts.append(
            (f"{model}_airport_latest_{graph}", f"{model}_airport_history_{graph}")
        )
    contrasts.extend((f"mlp_{group}_none", f"gat_{group}_real") for group in GROUPS)
    intervals = {}
    for before, after in contrasts:
        frames = []
        for variant in [before, after]:
            frames.append(
                mean_weekly(
                    [
                        pd.read_parquet(directory / item["name"] / "weekly_val.parquet")
                        for item in results
                        if item["variant"] == variant
                    ]
                )
            )
        intervals[f"{before}_to_{after}"] = week_interval(
            *frames, **config["bootstrap"]
        )
    summary["validation_intervals"] = intervals
    summary["limitations"] = [
        "Fixed national airport proxies do not represent regional or population-weighted weather",
        "Two-hour buffer is an assumption; original METAR receipt/revision vintages unresolved",
        "Price archive revision vintages unresolved; fixed training-union graph is retrospective",
        "Previously explored test period is not a fresh holdout",
        "Five-seed means average scores, not predictions; week bootstrap conditions on fitted models",
    ]
    write_json(directory / "summary.json", summary)
    write_json(str(prefix) + ".json", summary)
    provenance = json.loads((directory / "provenance.json").read_text())
    cache = Path(provenance["cache"])
    for name in [
        "coverage.csv",
        "source_findings.csv",
        "geography.csv",
        "feature_registry.json",
    ]:
        shutil.copyfile(cache / name, ROOT / f"results/airport_temperature_{name}")
    source = ROOT / config["source_directory"]
    shutil.copyfile(
        source / "manifest.json", ROOT / "results/airport_temperature_sources.json"
    )
    write_json(
        ROOT / f"results/airport_temperature_{phase}_gpu.json",
        json.loads((root / "gpu_preflight.json").read_text()),
    )
    scores = pd.DataFrame(summary["scores"])
    overall = scores[(scores.scope == "overall") & scores.coverage.eq("all")]
    lines = [
        f"# Airport temperature experiment — {phase}",
        "",
        f"Fingerprint: `{token}`. {len(results)} verified fits; "
        + (
            "France supervision, short training windows."
            if smoke
            else "five seeds across the unchanged 49-market cohort."
        ),
        "",
        (
            "Each country's largest airport by 2023 annual passengers is fixed before fitting. "
            "Temperature is a local proxy, not a national or bidding-zone average. "
            "Regional zones share the national airport; DE_LU and IE_SEM are explicitly combined-market proxies. "
            "DE_AT_LU and unresolved EIC identities retain missing airport inputs."
        ),
        "",
        (
            "At literal D−1 noon Europe/Brussels, use observations at least two hours old and at most six hours old for latest temperature. "
            "When receipt timestamps exist they must also precede origin. "
            "The IEM export used here lacks original receipt and revision timestamps: **all weather variants remain conditional**. "
            "The same inputs are frozen across every delivery-day hour, including DST days."
        ),
        "",
        (
            "Features: calendar and guarded 24h/168h prices; optionally latest temperature in Celsius plus age; "
            "optionally trailing 24h hourly-sampled mean plus coverage count (mean requires 18 hours). "
            "No load, ERA5, JAO or realized-flow predictors. XGBoost is pooled, without country IDs; MLP and GAT share the same feature groups. "
            "Target asinh(price/50), L1 objective, train-only neural preprocessing, and validation early stopping match the controlled study."
        ),
        "",
        (
            "Fit: January 2016–March 2024; validation: April 2024–March 2025; test: April 2025–September 2026. "
            "October 2026 remains unassigned. Missing weather never removes a target. "
            "The test period was explored before this experiment; results are not a fresh blind evaluation."
        ),
        "",
        "## Overall errors",
        "",
        (
            "MAE/RMSE are EUR/MWh; ± is sample standard deviation across training seeds. "
            "Primary metrics retain every valid price target. Paired MAE additionally requires weekly persistence history."
        ),
        "",
        "| Variant | Validation MAE | Test MAE | Test RMSE | Test paired MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in sorted(overall.variant.unique()):
        val = overall[(overall.variant == variant) & overall.split.eq("val")].iloc[0]
        test = overall[(overall.variant == variant) & overall.split.eq("test")].iloc[0]

        def metric(row, name):
            sd = row.get(name + "_std")
            return f"{row[name + '_mean']:.4f}" + (
                f" ± {sd:.4f}" if pd.notna(sd) else ""
            )

        lines.append(
            f"| {variant} | {metric(val, 'mae')} | {metric(test, 'mae')} | {metric(test, 'rmse')} | {metric(test, 'paired_mae')} |"
        )
    lines += [
        "",
        "## Paired validation-week comparisons",
        "",
        (
            "Positive means the second variant improves MAE. Bootstrap resamples shared calendar weeks across countries; "
            "it conditions on the fitted models and does not include retraining uncertainty."
        ),
        "",
        "| Comparison | Improvement | 95% interval |",
        "|---|---:|---:|",
    ]
    for contrast, interval in intervals.items():
        lines.append(
            f"| {contrast} | {interval['mean_mae_improvement']:.4f} | [{interval['ci95_lower']:.4f}, {interval['ci95_upper']:.4f}] |"
        )
    lines += [
        "",
        "## Country results",
        "",
        (
            "Each entry is test MAE averaged across seeds. "
            "Multi-zone countries pool absolute errors over their target hours; prices are never averaged into a national target."
        ),
        "",
        "| Country/market | Model | Price-only | + Latest | + History |",
        "|---|---|---:|---:|---:|",
    ]
    countries = scores[
        (scores.scope == "country")
        & scores.split.eq("test")
        & scores.coverage.eq("all")
    ]
    for country in sorted(countries.country.unique()):
        for model, graph in [("pooled_xgb", "none"), ("mlp", "none"), ("gat", "real")]:
            values = []
            for group in GROUPS:
                row = countries[
                    (countries.country == country)
                    & countries.variant.eq(f"{model}_{group}_{graph}")
                ]
                values.append(f"{row.iloc[0].mae_mean:.4f}")
            lines.append(f"| {country} | {model} | " + " | ".join(values) + " |")
    lines += [
        "",
        "## Coverage and interpretation",
        "",
        (
            "See `results/airport_temperature_coverage.csv` for target-hour availability by zone, month and split; "
            "`airport_temperature_source_findings.csv` for duplicates, conflicts, corrections and missing values; "
            "and `airport_temperature_geography.csv` for every mapping, including unresolved identities. "
            "Corrected reports without original versions and conflicting timestamps are quarantined. "
            "No airport was changed because of coverage or prediction scores."
        ),
        "",
        (
            "A benefit supports this inexpensive proxy. A null or negative result does not rule out regional temperature, "
            "population-weighted weather or genuine archived weather forecasts. "
            "This matrix tests the existing real graph against MLP; it adds no new self-loop or shuffled-graph controls."
        ),
        "",
    ]
    if not smoke:
        prior = json.loads((ROOT / "results/controlled_study_full.json").read_text())
        lines += [
            "## Previous controlled study",
            "",
            (
                "These are contextual paired test MAEs, not matched weather-source ablations. "
                "The earlier rich variants included ERA5, load and JAO with unresolved publication availability. "
                "Use the new price-only controls for the causal feature comparison."
            ),
            "",
            "| Previous variant | Paired test MAE |",
            "|---|---:|",
        ]
        names = [
            "mlp_recent_none",
            "gat_recent_real",
            "mlp_rich_none",
            "gat_rich_real",
            "pooled_xgb_rich_none",
        ]
        for row in prior["scores"]:
            if (
                row["scope"] == "overall"
                and row["coverage"] == "all"
                and row["split"] == "test"
                and row["variant"] in names
            ):
                lines.append(f"| {row['variant']} | {row['paired_mae_mean']:.4f} |")
        lines += [
            "",
            "Full per-zone and country metrics, seed dispersions and regime diagnostics are saved alongside the JSON summary.",
            "",
        ]
    lines += [
        "## Sources",
        "",
        (
            "- [Eurostat annual airport passenger transport](https://ec.europa.eu/eurostat/databrowser/view/avia_paoa/default/table?lang=en). "
            "Exact 2023 query, passenger counts and station identifiers are pinned in the mapping CSV."
        ),
        "- [Montenegro airport operator's 2023 national totals](https://montenegroairports.com/novosti/finalne-brojke-tokom-2023-godine-opsluzeno-preko-dva-i-po-miliona-putnika/).",
        (
            "- [IEM observation archive](https://mesonet.agron.iastate.edu/request/download.phtml) and "
            "[API documentation](https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?help=)."
        ),
        "",
    ]
    name = "airport_temperature_smoke.md" if smoke else "airport_temperature_results.md"
    (ROOT / "docs" / name).write_text("\n".join(lines))
    write_json(
        ROOT / f"results/airport_temperature_{phase}_report_sources.json",
        {
            "fingerprint": token,
            "summary_sha256": digest(directory / "summary.json")["sha256"],
            "verification_sha256": digest(directory / "verification.json")["sha256"],
            "mapping_sha256": digest(ROOT / config["airport_mapping"])["sha256"],
            "cache": str(cache),
            "seconds": sum(item["seconds"] for item in results),
        },
    )
    print(f"REPORT docs/{name}", flush=True)
