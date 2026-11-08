"""Preserve milestone v1, compile v2, and compare verified multi-seed results.

Uses saved scores only. No model fitting, GPU access, or test-based selection.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cs224w-milestone-v2-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from epgnn.reporting.milestone import FUTURE_WORK, render_table

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "reports/milestone_v1"
V2 = ROOT / "reports/milestone_v2"


def load(path):
    return json.loads((ROOT / path).read_text())


def score(summary, variant, split="test", field="paired_mae_mean", country="ALL"):
    scope = "overall" if country == "ALL" else "country"
    return next(
        row[field]
        for row in summary["scores"]
        if row["variant"] == variant
        and row["split"] == split
        and row["coverage"] == "all"
        and row["scope"] == scope
        and row["country"] == country
    )


def replace_section(text, start, end, replacement):
    first, last = text.index(start), text.index(end)
    return text[:first] + replacement + "\n" + text[last:]


def archive_v1():
    # Snapshot the explicitly pinned pre-study report, not the auto-updated PDF.
    if not (V1 / "milestone.pdf").exists():
        names = subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", "efe0e66", "reports/milestone"],
            cwd=ROOT,
            text=True,
        ).splitlines()
        for name in names:
            relative = Path(name).relative_to("reports/milestone")
            target = V1 / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(
                subprocess.check_output(["git", "show", f"efe0e66:{name}"], cwd=ROOT)
            )
        (V1 / "README.md").write_text(
            "# Milestone v1\n\nFrozen pre-study report from commit `efe0e66`, covering the first single-seed suite and earlier France diagnostics.\n\n[PDF](milestone.pdf) · [LaTeX](milestone.tex) · [Milestone v2](../milestone_v2/milestone.pdf)\n"
        )


def build():
    summary = load("results/controlled_study_full.json")
    verified = load("results/controlled_study_full_verification.json")
    digest = hashlib.sha256(
        (ROOT / "results/controlled_study_full.json").read_bytes()
    ).hexdigest()
    if (
        not verified["passed"]
        or verified["summary_sha256"] != digest
        or summary["job_count"] != 100
    ):
        raise ValueError(
            "All 100 jobs must be independently verified before publication"
        )
    archive_v1()
    V2.mkdir(parents=True, exist_ok=True)
    for name in [
        "milestone.tex",
        "neurips_2019.sty",
        "numbers.tex",
        "suite_numbers.tex",
        "suite_results.tex",
        "controlled_study.tex",
        "controlled_sources.json",
        "sources.json",
    ]:
        shutil.copy(ROOT / "reports/milestone" / name, V2 / name)
    template = (
        (V2 / "milestone.tex")
        .read_text()
        .replace("Project Milestone --- Draft", "Project Milestone v2 --- Draft")
    )
    template = template.replace(
        "Both GATs add self-loops", "All GAT variants add self-loops"
    )
    template = template.replace(r"\newpage", "")
    (V2 / "milestone.tex").write_text(template)
    gat, mlp, pooled, self_only = [
        score(summary, name)
        for name in [
            "gat_rich_real",
            "mlp_rich_none",
            "pooled_xgb_rich_none",
            "gat_rich_self",
        ]
    ]
    text = (V2 / "controlled_study.tex").read_text()
    zone_scores = pd.read_csv(ROOT / "results/controlled_study_full_by_zone.csv")
    counts = {}
    for split in ["val", "test"]:
        selected = zone_scores[
            (zone_scores.variant == "mlp_rich_none")
            & (zone_scores.seed == 224)
            & (zone_scores.split == split)
            & (zone_scores.coverage == "all")
        ]
        counts[split] = int(selected.paired_hours.sum())
    text = text.replace(
        "The richer-input controls below use common finite weekly-history hours.",
        f"The richer-input controls use {counts['val']:,} validation and {counts['test']:,} test zone-hours with valid targets and finite weekly history.",
    )
    abstract = (
        r"\newcommand{\ControlledAbstract}{Across five seeds and 49 market identities, a pooled MLP scores "
        + f"{mlp:.2f}"
        + r" EUR/MWh test MAE versus "
        + f"{gat:.2f}"
        + r" for real-graph GAT. Self-loop GAT and pooled XGBoost each score "
        + f"{pooled:.2f}"
        + r". Graph gains are smaller after controlling for pooled learning. Feature effects are mixed and GRU histories worsen scores. These are conditional exploratory results with unresolved publication vintages.}"
    )
    text = replace_section(
        text,
        r"\newcommand{\ControlledAbstract}",
        r"\newcommand{\ControlledMethods}",
        abstract,
    )
    contrast = summary["validation_intervals"]["mlp_rich_none_to_gat_rich_real"]
    discussion = (
        r"\paragraph{Graph contribution.} Real GAT improves test MAE by "
        + f"{mlp-gat:.2f} ({100*(1-gat/mlp):.2f}\\%)"
        + r" against MLP, but only "
        + f"{self_only-gat:.2f}"
        + r" against self-loop GAT and "
        + f"{pooled-gat:.2f}"
        + r" against pooled XGBoost. Pooled XGBoost has the best validation score among richer-input controls; the small test advantage does not justify selecting GAT retrospectively. MLP-to-GAT validation improvement is "
        + f"{contrast['mean_mae_improvement']:.2f}"
        + r" (95\% paired week interval ["
        + f"{contrast['ci95_lower']:.2f}, {contrast['ci95_upper']:.2f}"
        + r"]). Weeks are shared across countries, conditional on fitted models. Real GAT beats all three shuffled means on test; one shuffled-control validation interval includes zero."
    )
    text = replace_section(
        text,
        r"\paragraph{Graph contribution.}",
        r"\paragraph{Earlier evidence.}",
        discussion,
    )
    earlier = (
        r"\paragraph{Comparison with earlier experiments.} The earlier single-seed GAT/shuffled scores of 20.83/21.54 become 20.97/21.45 across seeds and rewirings. Pooled richer XGBoost scores 21.06 versus 22.11 for the new per-zone reference, supporting the need for sharing controls. Gains vary by market: France GAT scores "
        + f"{score(summary,'gat_rich_real',country='FR'):.2f}"
        + r" versus MLP's "
        + f"{score(summary,'mlp_rich_none',country='FR'):.2f}"
        + r". France's earlier 32.09-to-22.20 recent-price gain is a separate input diagnostic."
    )
    earlier += r"\paragraph{Feature and temporal extensions.} Weather-only GAT has test MAE 20.90, but richer GAT has better validation MAE (20.98 versus 21.06). Load and JAO additions are mixed. GRU--MLP/GAT score 24.50/23.59, worse than static controls; graph messaging still helps within the GRU pair. These fixed settings do not establish that temporal models or detailed constraints are generally unhelpful."
    text = replace_section(
        text,
        r"\paragraph{Earlier evidence.}",
        r"\paragraph{Timing and limitations.}",
        earlier,
    )
    first = text.index(r"\begin{table}")
    last = text.index(r"\end{table}", first) + len(r"\end{table}")
    text = text[:first] + render_table(summary) + text[last:]
    first = text.index(r"\section{Future work}")
    text = text[:first] + FUTURE_WORK + "\n}\n"
    methods = r"""\newcommand{\ControlledMethods}{\paragraph{Controlled models.} The 100 jobs cross five seeds (224--228): per-zone/pooled trees, MLP/GAT feature ablations, self-loop and three degree-preserving rewired graphs, and GRU--MLP/GAT pairs. Rich static MLP/GAT have 12,001/12,097 parameters; neither uses zone embeddings. Neural models share training-only preprocessing, asinh(price/50), L1 loss, Adam 0.001, maximum 30 epochs and validation patience five. GAT has two layers, 64 channels and four heads. The 32-unit GRU encodes 168 same-identity completed price intervals ending at Brussels noon, with missingness. Trees use depth 6 and at most 600 rounds.}
"""
    text = replace_section(text, r"\newcommand{\ControlledMethods}", r"\newcommand{\ControlledResults}", methods)
    earlier = r"""\paragraph{Earlier and extended experiments.} The first single-seed real/shuffled GAT scores of 20.83/21.54 become 20.97/21.45 across seeds and rewirings. Pooled rich XGBoost scores 21.06 versus 22.11 for per-zone models. Gains vary: France GAT scores 25.48 versus MLP's 22.00. Weather-only GAT has test MAE 20.90, but richer GAT has better validation MAE (20.98 versus 21.06). Load and JAO effects are mixed. GRU--MLP/GAT score 24.50/23.59, worse than static controls; these settings do not rule out other temporal models or detailed constraints.
"""
    text = replace_section(text, r"\paragraph{Comparison with earlier experiments.}", r"\paragraph{Timing and limitations.}", earlier)
    limits = r"""\paragraph{Timing and limitations.} Prior-day auction prices are normally published on $D-2$ \cite{auction}; DST guards exclude lag-24 target-day prices. ERA5-derived lag-48 inputs conflict with documented publication latency \cite{era5}; load/JAO vintages remain unresolved. Temporal histories end by noon. Static training-only topology does not reconstruct commissioning dates. All models retain identical target keys/masks and missing features; targets are never filled. Previously inspected test data make these conditional exploratory comparisons, not confirmatory evidence. Multiple feature comparisons do not establish causal effects.
"""
    text = replace_section(text, r"\paragraph{Timing and limitations.}", r"\section{Future work}", limits)
    (V2 / "controlled_study.tex").write_text(text)
    template = template.replace(
        "Weekly persistence copies the same identity's price",
        "A 1 MWh quantity is 1,000 kWh; 21 EUR/MWh error is 2.1 euro cents/kWh. "
        "Relative MAE divides model MAE by weekly-persistence MAE on identical hours; "
        "0.72 means 28\\% less error, not 28\\% price error. "
        "Weekly persistence copies the same identity's price",
    )
    (V2 / "milestone.tex").write_text(template)
    env = dict(os.environ, TEXMFVAR="/tmp/cs224w-milestone-v2-texmf")
    # Editor auto-builds may run in V2 concurrently. Validate in isolation and
    # publish only a complete two-page PDF, without touching the editor process.
    with tempfile.TemporaryDirectory(prefix="cs224w-milestone-") as temporary:
        build_dir = Path(temporary)
        for source in V2.iterdir():
            if source.suffix in {".tex", ".sty"}:
                shutil.copyfile(source, build_dir / source.name)
        for _ in range(2):
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "milestone.tex"],
                cwd=build_dir, env=env, capture_output=True, text=True,
            )
            if result.returncode:
                raise RuntimeError(result.stdout[-4000:])
        info = subprocess.check_output(["pdfinfo", str(build_dir / "milestone.pdf")], text=True)
        if not any(line.split() == ["Pages:", "2"] for line in info.splitlines()):
            raise ValueError("Milestone v2 must fit two pages:\n" + info)
        staged = V2 / "milestone.pdf.tmp"
        staged.write_bytes((build_dir / "milestone.pdf").read_bytes())
        staged.replace(V2 / "milestone.pdf")
    (V2 / "README.md").write_text(
        "# Milestone v2\n\n[Two-page PDF](milestone.pdf) · [LaTeX](milestone.tex) · [Frozen v1](../milestone_v1/milestone.pdf) · [Detailed comparison](../../docs/controlled_study_comparison.md)\n\nRebuild with `.venv/bin/python scripts/experiments/build_milestone_v2.py`. All 100 jobs must pass artifact verification first. No models are fitted. Author details remain placeholders.\n"
    )
    with (V2 / "README.md").open("a") as handle:
        handle.write("\nPresentation revision: Table 1 groups methods by class, includes weekly persistence and relative MAE, and bolds the smallest mean in each column. Future work now proposes a daily prospective forecasting service. Numerical model results are unchanged. The [availability audit](../../docs/leakage_audit.md) still applies; this is a conditional retrospective report. See [metrics and prospective deployment](../../docs/metrics_and_live_forecasting.md).\n")
    (V1 / ".gitignore").write_text("*.aux\n*.log\n*.out\n")
    (V2 / ".gitignore").write_text("*.aux\n*.log\n*.out\n*.fls\n*.fdb_latexmk\n*.synctex.gz\n")
    manifest = {
        "study_fingerprint": summary["fingerprint"],
        "summary_sha256": digest,
        "verification": verified,
        "v1_source_commit": "efe0e66",
        "v1_pdf_sha256": hashlib.sha256(
            (V1 / "milestone.pdf").read_bytes()
        ).hexdigest(),
    }
    manifest["presentation_code_sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [Path(__file__), ROOT / "epgnn/reporting/milestone.py"]
    }
    manifest["comparison_inputs_sha256"] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in [
            "results/controlled_study_full.json",
            "results/controlled_study_full_by_zone.csv",
            "results/controlled_study_full_regimes.csv",
            "results/vibing_suite_full.json",
            "results/pipeline_diagnostics_france.json",
            "results/partner_c1_xgb_zone_by_country.json",
        ]
    }
    (V2 / "comparison_sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_comparison(summary)
    print(info)


def write_comparison(summary):
    lines = [
        "# Multi-seed controls: comparison with earlier experiments",
        "",
        "All 100 jobs completed and passed independent artifact, target-mask and score verification. Models use the same 49-market cohort and chronological splits. Scores below use common finite weekly-history hours, averaged across training seeds (not ensembled predictions).",
        "",
        "## Main controls",
        "",
        "| Variant | Validation MAE ± SD | Test MAE ± SD |",
        "|---|---:|---:|",
    ]
    for variant in sorted({row["variant"] for row in summary["scores"]}):
        cells = [
            f"{score(summary,variant,split):.4f} ± {score(summary,variant,split,'paired_mae_std'):.4f}"
            for split in ["val", "test"]
        ]
        lines.append(f"| {variant} | {cells[0]} | {cells[1]} |")
    lines += [
        "",
        "## What changes the interpretation",
        "",
        "- Real GAT has a modest advantage over the parameter-matched pooled MLP and self-loop GAT. Pooled XGBoost is nearly tied on test and has better validation MAE than richer GAT. These controls reduce the gap that can be attributed specifically to topology; the original report already identified pooling as a confound.",
        "- Real GAT beats the three shuffled graph means on test. One validation block-bootstrap interval against a shuffled graph spans zero; evidence is not uniformly decisive across rewiring controls.",
        "- Weather-only GAT has the smallest observed test MAE, but richer GAT is preferred by validation. Do not select the weather-only model retrospectively using test.",
        "- The specified GRUs degrade absolute scores. GRU-GAT improves on GRU-MLP, but neither beats the static controls; this does not rule out other temporal architectures/settings.",
        "- Gains are not universal: France is substantially worse under richer GAT than MLP or pooled XGBoost. These market-level patterns warrant further validation, not per-country selection using test.",
        "",
        "## Earlier cohorts and models",
        "",
        "| Experiment | Test MAE | Interpretation |",
        "|---|---:|---|",
        "| Earlier 49-market real GAT, one seed | 20.8279 | Paired hours; earlier sampler/seed implementation, not a multi-seed estimate |",
        "| Earlier shuffled GAT, one seed | 21.5421 | Paired hours; one rewiring, versus three graphs/five seeds now |",
        "| Earlier recent-price per-zone XGBoost | 21.8654 | Paired hours; same cohort, with pooling now an additional control |",
        "| Earlier richer per-zone XGBoost | 22.1323 | Paired hours; new per-zone mean 22.1096 versus pooled mean 21.0584 |",
        "| France: weekly/calendar → recent-price | 32.0921 → 22.1988 | One-country diagnostic, different population |",
        "| Partner reproduction, all 43 zones | 21.2539 | Full-target MAE; different geography/features/population, not directly ranked |",
        "",
        "## Familiar country/market comparisons",
        "",
        "| Market | Pooled MLP | Real GAT | Pooled XGBoost |",
        "|---|---:|---:|---:|",
    ]
    for country in ["BE", "FR", "NL", "DE_LU", "AT", "ES", "PT", "PL", "CZ", "CH"]:
        values = [
            score(summary, variant, country=country)
            for variant in ["mlp_rich_none", "gat_rich_real", "pooled_xgb_rich_none"]
        ]
        lines.append(
            f"| {country} | {values[0]:.3f} | {values[1]:.3f} | {values[2]:.3f} |"
        )
    lines += [
        "",
        "## Seasons and training-defined price regimes",
        "",
        "All-valid-target MAE means; this population differs from the paired table. Tails use training-only 5th/95th percentiles. These diagnostic slices overlap and are not model-selection criteria.",
        "",
        "| Slice | MLP | Real GAT | Pooled XGBoost |",
        "|---|---:|---:|---:|",
    ]
    regimes = pd.read_csv(ROOT / "results/controlled_study_full_regimes.csv")
    for label in ["DJF", "MAM", "JJA", "SON", "negative", "lower_tail", "upper_tail"]:
        values = [
            regimes.loc[
                (regimes.variant == variant)
                & (regimes.split == "test")
                & (regimes.scope == "overall")
                & (regimes["slice"] == label),
                "mae_mean",
            ].iloc[0]
            for variant in ["mlp_rich_none", "gat_rich_real", "pooled_xgb_rich_none"]
        ]
        lines.append(
            f"| {label} | {values[0]:.3f} | {values[1]:.3f} | {values[2]:.3f} |"
        )
    lines += [
        "",
        "## Uncertainty and limits",
        "",
        "Paired validation-week intervals condition on fitted models, with weeks resampled jointly across countries. Seed variability is distinct from temporal loss uncertainty and graph rewiring variability. There is no fresh blind test claim, multiple-comparison significance claim or verified historical publication availability.",
        "",
    ]
    for name, interval in summary["validation_intervals"].items():
        lines.append(
            f"- {name}: validation MAE improvement {interval['mean_mae_improvement']:.4f}, interval [{interval['ci95_lower']:.4f}, {interval['ci95_upper']:.4f}]."
        )
    lines += [
        "",
        "[Milestone v2](../reports/milestone_v2/milestone.pdf) · [Milestone v1](../reports/milestone_v1/milestone.pdf) · [Raw multi-seed scores](../results/controlled_study_full_scores.csv)",
    ]
    (ROOT / "docs/controlled_study_comparison.md").write_text("\n".join(lines) + "\n")
    figure_dir = V2 / "figures"
    figure_dir.mkdir(exist_ok=True)
    variants = [
        "pooled_xgb_rich_none",
        "mlp_rich_none",
        "gat_rich_self",
        "gat_rich_real",
        "gru_mlp_rich_none",
        "gru_gat_rich_real",
    ]
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    figure, axis = plt.subplots(figsize=(7, 3))
    axis.barh(
        np.arange(len(variants)),
        [score(summary, v) for v in variants],
        xerr=[score(summary, v, field="paired_mae_std") for v in variants],
        capsize=3,
    )
    axis.set_yticks(
        np.arange(len(variants)),
        [
            "Pooled XGBoost",
            "Pooled MLP",
            "Self-loop GAT",
            "Real GAT",
            "GRU + MLP",
            "GRU + GAT",
        ],
    )
    axis.invert_yaxis()
    axis.set_xlim(0, 28)
    axis.set_xlabel("Test MAE (EUR/MWh), mean ± SD across five seeds")
    figure.tight_layout()
    for extension in ["png", "pdf"]:
        figure.savefig(figure_dir / f"neural_controls.{extension}", dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    build()
