"""Build and execute the guided CPU notebook, then export a self-contained HTML."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[2]


def notebook(load_local_details=False):
    cells = []

    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text))

    def code(text):
        cells.append(nbf.v4.new_code_cell(text))

    md("""# Forecasting European electricity prices: an investigation

A guided recap of the data, pipeline changes and completed experiments. **MAE is an error in EUR/MWh, not classification accuracy.** Lower is better.

Read top to bottom for the project story; edit the next cell to investigate a country and validation week. This notebook fits no models and downloads no data. Published test scores are exploratory: that period has been inspected repeatedly.

**November 9 correction:** the original GRU scores are affected by a fused ROCm discrepancy on shuffled training-shaped batches. Consecutive inference and saved-score replay pass, but this does not validate training. Their old negative temporal conclusion is withdrawn. The later native-GPU residual GRU is a different accepted recipe. Read the complete experiment record (`docs/experiments.md`) for all programmes and both backend audits.

**Contents:** 1. Data and timing · 2. Models and pipeline · 3. Competitive baselines · 4. Graphs, features and histories · 5. Airport observations · 6. Country investigations · 7. Conclusions and appendices
""")
    code("""from pathlib import Path
import sys
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "epgnn").is_dir())
sys.path.insert(0, str(ROOT))
get_ipython().run_line_magic("matplotlib", "inline")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from IPython.display import display, Markdown, HTML
from epgnn.reporting.investigation import Report, heatmap, plot_window
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
pd.set_option("display.max_columns", 12)
report = Report(ROOT)
""")
    code(f"""# Editable investigation parameters. Detailed outcomes stay in validation.
ZONES = ["FR", "DE_LU", "BE"]
COUNTRIES = ["FR", "DE_LU", "BE", "NL", "AT", "ES", "PT", "CZ", "CH", "NO", "IT"]
START = "2024-07-08T00:00:00+02:00"
END = "2024-07-15T00:00:00+02:00"
SEED = 224  # Illustrative seed, not a selected best seed.
MODELS = ["Per-zone XGBoost", "Pooled XGBoost", "MLP", "GAT"]
LOAD_LOCAL_DETAILS = {load_local_details}
display(pd.Series({{"Markets": ", ".join(ZONES), "Validation window": START + " to " + END, "Illustrative seed": SEED, "Models": ", ".join(MODELS), "Local details": LOAD_LOCAL_DETAILS}}).to_frame("Report selections"))
""")
    md("""## 1. What are we forecasting, and from what data?

At **12:00 Europe/Brussels on D−1**, predict every delivery-hour price for day D. Winter/summer noon and 23/25-hour delivery days matter. Observation time, interval completion and original publication time are different clocks.

The master preserves **62 market identities**, of which **49 have labels in all evaluation partitions**. DE_AT_LU and DE_LU remain separate. Missing prices are never filled; missing predictors do not remove targets. October 2026 remains unassigned.

Source classes: ENTSO-E prices and flows, archived load forecasts, retrospective ERA5-lineage weather, JAO summaries, and the newer airport observations. Price revisions, load/JAO vintages and airport receipt times remain unresolved. Recent ERA5 has a documented product-latency conflict with the noon forecast contract. None of these experiments is certified as an operational leakage-free forecast.
""")
    code("""fig, ax = plt.subplots(figsize=(12, 2.4))
for label, start, end, color in [
    ("Fit: pretrain + train", "2016-01-01", "2024-04-01", "#347a99"),
    ("Validation", "2024-04-01", "2025-04-01", "#e5a940"),
    ("Previously explored test", "2025-04-01", "2026-10-01", "#b2768d"),
    ("Unassigned", "2026-10-01", "2026-11-01", "#b9c1c8")]:
    ax.barh(label, pd.Timestamp(end) - pd.Timestamp(start), left=pd.Timestamp(start), color=color)
ax.set_title("Partitions by Brussels delivery date (inclusive dates described in the text)")
fig.autofmt_xdate(); plt.tight_layout(); plt.show()
coverage = pd.read_csv(ROOT / "results/airport_temperature_coverage.csv")
selected = coverage[coverage.scored_zone & coverage.zone.isin(ZONES)].copy()
selected["latest_missing_fraction"] = 1 - selected.latest_available_target_hours / selected.target_hours.replace(0, np.nan)
matrix = selected.pivot_table(index="zone", columns="month", values="latest_missing_fraction", aggfunc="mean")
heatmap(matrix.iloc[:, ::3], "Airport-input missingness on valid targets: selected zones, every third month")
plt.show()
display(coverage[coverage.scored_zone].groupby("split")[["rows", "target_hours", "latest_available_target_hours", "history_available_target_hours"]].sum())
""")
    code("""# Training/validation variability only. Read just selected zones and columns.
master = ROOT / "staging/forecast_dataset_v1.0/current/nodes"
if LOAD_LOCAL_DETAILS and master.exists():
    if len(ZONES) > 5:
        raise ValueError("Use at most five zones for bounded distribution inspection")
    filt = ds.field("zone").isin(ZONES) & ds.field("split").isin(["pretrain", "train", "val"])
    prices = ds.dataset(master).to_table(columns=["zone", "split", "price"], filter=filt).to_pandas()
    prices["partition"] = np.where(prices.split.eq("val"), "Validation", "Fit")
    stats = prices.groupby(["zone", "partition"]).price.agg(hours="count", mean="mean", variance="var", standard_deviation="std", median="median")
    stats["IQR"] = prices.groupby(["zone", "partition"]).price.quantile(.75) - prices.groupby(["zone", "partition"]).price.quantile(.25)
    display(stats.round(2))
    fig, axes = plt.subplots(1, len(ZONES), figsize=(13, 3), squeeze=False)
    for ax, zone in zip(axes[0], ZONES):
        for label, group in prices[prices.zone.eq(zone)].groupby("partition"):
            ax.hist(group.price.dropna(), bins=100, histtype="step", density=True, label=label)
        ax.set(title=zone, xlabel="Price (EUR/MWh)", ylabel="Density"); ax.legend()
    plt.tight_layout(); plt.show()
else:
    display(Markdown("Local variability panel is optional. Enable LOAD_LOCAL_DETAILS with the pinned local dataset present; no test-target distribution is produced."))
""")
    md("""**Reading the data:** coverage is not the same as volatility. The missingness panel retains structural absences; standard deviation and IQR describe finite fit/validation prices. Variance is in (EUR/MWh)². These distributions explain differing difficulty, not which model to select on test.

## 2. What did we train, and what changed?

We have **seven forecasting setups**: weekly persistence; per-zone and pooled XGBoost; pooled MLP; GAT; GRU+MLP; GRU+GAT. The two five-seed studies contain **27 distinct variant names across 145 experiment jobs**. A per-zone XGBoost job fits 49 market models. Two price-only baselines were repeated across studies; they are not ten independent seeds.

Current models learn **asinh(price/50)** with L1 loss and invert with **50·sinh(prediction)** before price scoring. This preserves negative and zero prices and compresses extremes. Transformed-space L1 is not the same objective as MAE in EUR/MWh. Neural imputation and standardization use training data only.

**Main story:** recent-price baselines, controlled feature/topology comparisons, and airport observations. **Affected historical evidence:** the original GRUs require corrected training because of a fused-backend discrepancy. **Historical context:** earlier raw-price versus asinh/L1, France pipeline diagnostics, partner reproduction and the first single-seed suite. **Not scientific results:** smoke fits. CNEC/heterogeneous-graph YAMLs are scaffolds, not completed experiments. JAO constraints here are flat summary features, not extra graph nodes.
""")
    code("""inventory = report.inventory()
assert inventory.jobs.sum() == 145 and inventory.variant.nunique() == 27
display(inventory.groupby("study")[["jobs"]].sum())
display(HTML("<details><summary>Full 29-row study inventory (two repeated names)</summary>" + inventory.to_html(index=False) + "</details>"))
fig, ax = plt.subplots(figsize=(7, 2.5))
p = np.linspace(-300, 300, 400)
ax.plot(p, np.arcsinh(p / 50)); ax.set(xlabel="Original price (EUR/MWh)", ylabel="Training target", title="The fixed target transformation"); ax.axhline(0, color="grey", linewidth=.5)
plt.tight_layout(); plt.show()
""")
    md("""## 3. Which baselines are competitive?

The headline comparison holds recent-price/calendar inputs fixed. The per-zone tree comes from the controlled study; pooled trees, MLP and GAT use the airport study's price-only controls. They share the dataset, partitions and seeds.

The main chart uses **the same valid target hours with finite weekly history**, so persistence is comparable. The table separately retains all-target MAE/RMSE. No weather-complete-case filtering is applied. Error bars show sample SD over five training seeds; persistence is deterministic. They are not forecast prediction intervals.
""")
    code("""display(Markdown(report.compare_signatures()))
main = report.main_table()
display(main.drop(columns="job_seconds").round(4))
report.main_figure(); plt.show()
report.country_figure(COUNTRIES); plt.show()
fig, ax = plt.subplots(figsize=(9, 2.8))
f = main[main.split.eq("val") & main.job_seconds.notna()]
ax.barh(f.model, f.job_seconds, color="#347a99")
ax.set(xlabel="Mean end-to-end job seconds (fit, prediction and scoring)", title="Local runtime: a per-zone tree job includes 49 models; neural jobs use ROCm")
plt.tight_layout(); plt.show()
""")
    md("""**Interpretation:** pooling is a major control: a graph should be compared with pooled classical models, not only independent country trees. Country improvements vary, and France is an important counterexample. Country/market scores pool errors over target hours; prices are never averaged into a national target.

## 4. What do graphs, additional features and histories contribute?

The next panels use the original five-seed controlled study. **All richer-input panels are retrospective/conditional:** equal inputs do not guarantee that operational rankings survive removal of unavailable inputs.

The rich features include load forecasts, four ERA5-lineage variables and flat JAO summaries. Self-loop and degree-preserving shuffled GATs hold the architecture fixed. Three rewiring seeds are shown separately across the same five training seeds; they are not 15 independent training seeds. We have no matching price-only self-loop/shuffled experiment.
""")
    code("""report.family_figure(["mlp_rich_none", "pooled_xgb_rich_none", "gat_rich_real", "gat_rich_self", "gat_rich_shuffled_rewire901", "gat_rich_shuffled_rewire902", "gat_rich_shuffled_rewire903"], "Rich-input graph controls — retrospective/conditional")
plt.show()
ablations = []
for model, graph in [("mlp", "none"), ("gat", "real")]:
    for group in ["load", "weather", "jao", "rich"]:
        ablations.append({"addition": model + " + " + group, **{split: report.score("controlled", f"{model}_recent_{graph}", split).paired_mae_mean - report.score("controlled", f"{model}_{group}_{graph}", split).paired_mae_mean for split in ["val", "test"]}})
heatmap(pd.DataFrame(ablations).set_index("addition"), "Matched MAE reduction from feature additions (EUR/MWh); positive helps", center=True)
plt.show()
report.family_figure(["mlp_rich_none", "gat_rich_real", "gru_mlp_rich_none", "gru_gat_rich_real"], "Historical GRU scores: fused training path affected")
plt.show()
intervals = report.summaries["controlled"]["validation_intervals"]
display(pd.DataFrame([{ "contrast": key, "improvement": value["mean_mae_improvement"], "lower": value["ci95_lower"], "upper": value["ci95_upper"]} for key, value in intervals.items() if "rich" in key]).round(4))
""")
    md("""**Interpretation:** rich real GAT improves on MLP and self-loop controls overall, but one shuffled-control validation interval spans zero. The historical GRU scores and their contrast intervals involve an affected fused training backend and cannot establish a temporal disadvantage or GRU graph advantage. Validation-week intervals resample shared weeks across countries, condition on fixed fitted models and are unadjusted for multiple comparisons. Feature additions are mixed; do not choose the weather-only GAT retrospectively because it has the smallest old test MAE.

## 5. Did current airport observations help?

For each country we froze its largest airport by 2023 passenger traffic. Regional zones share that proxy. At D−1 noon, latest temperature must be at least two and at most six hours old; history uses the preceding 24-hour window ending two hours before origin and requires 18 hourly samples for a mean. Inputs are constant across delivery day D.

These are actual airport reports rather than retrospective ERA5, but **historical receipt/revision timestamps remain unavailable**. The two-hour buffer is an assumption, so the experiment remains conditional. ERA5, airport observations and the earlier perfect-weather oracle are different experiments, not interchangeable weather products.
""")
    code("""report.weather_figure(); plt.show()
subgroups = pd.read_csv(ROOT / "results/airport_temperature_full_geographic_subgroups.csv")
display(subgroups[subgroups.split.eq("test")].pivot(index="variant", columns="subgroup", values="mae_mean").round(4))
contributions = pd.read_csv(ROOT / "results/airport_temperature_full_market_contributions.csv")
f = contributions[contributions.split.eq("test") & contributions.model.eq("pooled_xgb") & contributions.features.eq("airport_latest")].copy()
f = f.loc[f.contribution_to_overall_mae.abs().nlargest(10).index].sort_values("contribution_to_overall_mae")
fig, ax = plt.subplots(figsize=(12, 4))
ax.barh(f.zone, f.contribution_to_overall_mae, color=np.where(f.airport_mapped, "#347a99", "#d48b3b"))
ax.axvline(0, color="black", linewidth=.6); ax.set(xlabel="Contribution to overall MAE reduction (EUR/MWh)", title="Post-hoc XGBoost contribution check: orange = no local airport mapping")
plt.tight_layout(); plt.show()
review = report.root / "results/airport_temperature_full_review.json"
import json
review = json.loads(review.read_text())
display(Markdown(f"**{review['share_of_overall_improvement']:.1%} of XGBoost's headline test improvement comes from `{review['largest_xgb_latest_test_contributor']}`, whose local temperature inputs are all missing.** Adding features changes the jointly fitted pooled model; this is not evidence of a local temperature signal. GAT can also transmit weather information from mapped neighbours."))
""")
    md("""**Interpretation:** gains are small and inconsistent on markets with airport coverage. France worsens for all three models; Germany/Luxembourg shows small latest-temperature gains. Validation intervals include zero for most weather additions. These airport proxies cannot answer whether regional heatwaves or proper temperature forecasts would help. The mapped/unmapped contribution check is post-hoc description, not a changed evaluation cohort.

## 6. Investigate a country and week

The next panels use the selectors at the top. The default week is fixed in advance, not chosen for a large error. Seed 224 is illustrative, not the best test seed. Full-series metrics above use all five seeds. Detailed target exploration is limited to validation and at most 31 days; no target gaps are filled.
""")
    code("""if LOAD_LOCAL_DETAILS:
    for zone in ZONES:
        frame, message = report.drilldown(zone, START, END, SEED, MODELS)
        display(Markdown(f"### {zone} — {message}"))
        if frame is not None:
            plot_window(frame, zone); plt.show()
            weather = report.weather_window(zone, START, END)
            if weather is not None:
                display(weather.round(2))
                fig, ax = plt.subplots(figsize=(10, 2.2))
                ax.plot(weather.delivery_date, weather.temperature_C, marker="o", label="Latest observed at origin")
                ax.plot(weather.delivery_date, weather.mean24_C, marker="o", label="Trailing mean")
                ax.set(ylabel="Celsius", title=f"{zone}: one frozen temperature per delivery day"); ax.legend()
                plt.tight_layout(); plt.show()
            else:
                display(Markdown("Pinned local airport cache unavailable; temperature timeline omitted."))
else:
    display(Markdown("Summary mode: enable LOAD_LOCAL_DETAILS for bounded local prediction and temperature timelines."))
regimes = pd.read_csv(ROOT / "results/airport_temperature_full_regimes.csv")
f = regimes[regimes.scope.eq("overall") & regimes.split.eq("val") & regimes.variant.isin(["mlp_recent_none", "gat_recent_real", "pooled_xgb_recent_none"]) & regimes["slice"].isin(["DJF", "MAM", "JJA", "SON", "negative", "lower_tail", "upper_tail"])]
heatmap(f.pivot(index="slice", columns="variant", values="mae_mean"), "Validation error by season and training-defined price regime (all-valid MAE)")
plt.show()
""")
    md("""**Reading the investigations:** residuals show timing and magnitude errors; compare the same timestamps across models. Error on negative-price and tail slices is a descriptive diagnostic. Tail thresholds come from training only, and the slices overlap. Temperature values above are features from the pinned cache, not future delivery-day weather. Observation age and history coverage make stale/missing data visible.

## 7. What can we conclude?

| Question | What this evidence supports | What remains open |
|---|---|---|
| Do learned methods beat weekly persistence? | Recent-price models improve the matched aggregate substantially | Performance varies by market and regime |
| Does pooling matter? | Pooled XGBoost is a necessary strong reference | Isolating geography/identity effects needs dedicated controls |
| Do graphs add value? | Modest overall GAT gains over matched MLP; rich topology controls offer additional evidence | Price-only topology controls, operational availability and country counterexamples |
| Does temperature help? | Small mixed gains for the airport experiment | Regional representativeness, receipt vintages and archived weather forecasts |
| Did temporal histories help? | The original GRU scores involve an affected fused training path | Corrected original-family fits are needed; the later residual GRU is a different recipe |
| Were constraints graph nodes? | No: tested JAO summaries were flat node features | Detailed CNEC graph identities and models remain untested |

**Retain all archived runs.** Use the matched baselines in the main narrative; retain rich conditional studies and negative results with their limitations. No best seed or new model is selected from test performance here.

### Appendix A. Earlier pipeline development

The earlier experiments answer narrower questions on different populations. In the fixed-learner France sequence, adding the prior-day auction price was a much larger improvement than changing the target transformation alone. The ten-country comparison changed **both transformation and loss**. Partner reproduction uses legacy geography/filled inputs and a different cohort; it is not another row in the 49-market ranking.
""")
    code("""diagnostics = json.loads((ROOT / "results/pipeline_diagnostics_france.json").read_text())
display(pd.DataFrame(diagnostics["rows"])[["profile", "split", "mae", "rmse", "hours"]].round(3))
transform = pd.read_csv(ROOT / "results/c1_xgb_countries_asinh_l1_comparison.csv")
display(transform[transform.split.eq("test")][["zone", "xgb_mae_raw_mse", "xgb_mae_asinh_l1", "persistence_mae_asinh_l1"]].round(3))
""")
    md("""### Appendix B. Context, not additional multi-seed evidence

- The original five-model single-seed suite is preserved in `results/vibing_suite_full.json`; its baseline implementations/sampler history differ from later repeats.
- The partner's reproduced score uses 43 scored zones; compare within its own protocol.
- The old France “current temperature” diagnostic was retrospective ERA5. Its perfect delivery-hour weather companion deliberately supplies future information and is an **oracle**, not a deployable candidate or a guaranteed performance bound.
- Smoke fits and unimplemented heterogeneous-graph configuration files are excluded from scientific comparisons.

### Appendix C. Provenance and reproducibility

The main evidence is the verified 100-job controlled study and 45-job airport study. The table below pins their fingerprints. The reporting helpers recompute aggregate errors from seed/zone summaries; local detail panels additionally check identical target keys/values/masks. When local Parquets are absent the summaries still execute and unavailable detail is stated explicitly.

For deeper source auditing see `docs/leakage_audit.md`, `docs/controlled_study_comparison.md`, `docs/airport_temperature_results.md` and the existing dataset inspection notebook. Repeated test inspection and unresolved source publication vintages limit the claims throughout.
""")
    code("""display(pd.DataFrame(report.sources))
assert "torch" not in sys.modules and "xgboost" not in sys.modules
plt.close("all")
print("Completed: verified summaries, CPU-only execution, no training or downloads.")
""")
    md("""### Appendix D. Learning behaviour and the first single-seed suite

These learning curves use illustrative seed 224. They show validation MAE in transformed target units, not EUR/MWh. Early stopping uses validation only. The old single-seed results below provide historical context; do not pool them with the five-seed means above.
""")
    code("""fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
for ax, study, variants in [
    (axes[0], "airport", ["mlp_recent_none", "gat_recent_real"]),
    (axes[1], "controlled", ["gru_mlp_rich_none", "gru_gat_rich_real"])]:
    for variant in variants:
        job = next(j for j in report.summaries[study]["jobs"] if j["variant"] == variant and j["job"]["seed"] == 224)
        curve = pd.DataFrame(job["details"]["learning_curve"])
        ax.plot(curve.epoch, curve.val_transformed_mae, label=variant)
    ax.set(xlabel="Epoch", ylabel="Validation transformed MAE", title=study + " — illustrative seed 224"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
old = pd.DataFrame(json.loads((ROOT / "results/vibing_suite_full.json").read_text())["aggregate"])
display(old[old.scope.eq("overall") & old.coverage.eq("all")][["experiment", "split", "paired_hours", "paired_mae"]].round(4))
plt.close("all")
""")

    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python (project environment)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    nb.metadata["title"] = "Forecasting European electricity prices: an investigation"
    return nb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", choices=["root", "notebooks"], default="root")
    parser.add_argument("--with-details", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "notebooks/forecasting_results_investigation.ipynb",
    )
    args = parser.parse_args()
    audit = json.loads((ROOT / "results/controlled_gru_backend_audit.json").read_text())
    if (audit["checkpoint_count"] != 10 or audit["cases_per_checkpoint"] != 4
            or not audit["native_reference_passed"] or not audit["saved_replay_passed"]):
        raise ValueError("Complete original-GRU backend audit required")
    nb = notebook(args.with_details)
    env = {
        **os.environ,
        "PATH": str(Path(sys.executable).parent)
        + os.pathsep
        + os.environ.get("PATH", ""),
        "MPLBACKEND": "Agg",
    }
    cwd = ROOT if args.cwd == "root" else ROOT / "notebooks"
    NotebookClient(
        nb,
        timeout=240,
        kernel_name="python3",
        resources={"metadata": {"path": str(cwd)}},
    ).execute(env=env)
    nbf.validate(nb)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, args.output)
    exporter = HTMLExporter(
        template_name="lab",
        exclude_input=True,
        raw_template=(ROOT / "scripts/reports/static_report.html.j2").read_text(),
    )
    html, _ = exporter.from_notebook_node(nb)
    html_path = (
        ROOT / "reports/investigation/forecasting_results.html"
        if args.output == ROOT / "notebooks/forecasting_results_investigation.ipynb"
        else args.output.with_suffix(".html")
    )
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html)
    manifest = {
        "executed": True,
        "cwd": str(cwd),
        "local_details": args.with_details,
        "code_cells": sum(cell.cell_type == "code" for cell in nb.cells),
        "figures": sum(
            "image/png" in output.get("data", {})
            for cell in nb.cells
            if cell.cell_type == "code"
            for output in cell.outputs
        ),
        "python": sys.version,
        "artifacts": {
            str(p.relative_to(ROOT))
            if p.is_relative_to(ROOT)
            else str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                args.output.resolve(),
                html_path.resolve(),
                Path(__file__),
                ROOT / "epgnn/reporting/investigation.py",
                ROOT / "scripts/reports/static_report.html.j2",
                ROOT / "results/controlled_gru_backend_audit.json",
            ]
        },
        "study_fingerprints": {
            name: json.loads((ROOT / "results" / (prefix + ".json")).read_text())[
                "fingerprint"
            ]
            for name, prefix in [
                ("controlled", "controlled_study_full"),
                ("airport", "airport_temperature_full"),
            ]
        },
    }
    html_path.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Executed {len(nb.cells)} cells from {cwd}; saved {args.output} and {html_path}"
    )


if __name__ == "__main__":
    main()
