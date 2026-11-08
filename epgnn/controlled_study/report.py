"""Measured runtime estimates and publication after all predefined jobs complete."""

from pathlib import Path
import subprocess
import sys

import numpy as np

from epgnn.dataset_build.storage import ROOT, digest
from epgnn.experiment_suite.artifacts import write_json
from .design import jobs


def estimate(config, smoke, arrays, preflight):
    prototypes = {item["variant"]: item for item in smoke["jobs"]}
    full_hours = int((arrays["split"] == 0).sum())
    smoke_hours = min(full_hours, config["smoke"]["training_days"] * 24)
    estimates = []
    for job in jobs(config):
        prototype = prototypes[job.variant]
        if job.model in {"xgb", "pooled_xgb"}:
            seconds = prototype["seconds"] * len(smoke.get("eligible_zones", range(49)))
        else:
            curves = prototype["details"]["learning_curve"]
            warm = np.mean(
                [row["seconds"] for row in curves[1:]] or [curves[0]["seconds"]]
            )
            scaled = warm * full_hours / smoke_hours * config["neural"]["epochs"]
            check = next(
                row
                for row in preflight["checks"]
                if row["model"] == job.model and row["graph"] == job.graph
            )
            worst = (
                check["warm_step_seconds"]
                * np.ceil(full_hours / check["batch_size"])
                * config["neural"]["epochs"]
            )
            seconds = max(scaled, worst) + prototype["seconds"]
        estimates.append({"job": job.name, "estimated_seconds": float(seconds)})
    hours = sum(row["estimated_seconds"] for row in estimates) / 3600
    return {
        "maximum_epoch_extrapolation_hours": hours,
        "planning_range_hours": [hours * 0.6, hours * 1.8],
        "jobs": estimates,
        "method": "France warm epochs plus worst-case full-node GPU batches; all jobs sequential",
        "caveats": "Not a guarantee. Early stopping reduces work; pooled XGBoost/data I/O can scale differently.",
    }


def metric(summary, variant, split="test", column="paired_mae_mean"):
    return next(
        row[column]
        for row in summary["scores"]
        if row["variant"] == variant
        and row["split"] == split
        and row["coverage"] == "all"
        and row["scope"] == "overall"
    )


def controlled_tex(summary):
    """Render a short primary comparison; every extension stays in the appendix."""
    rows = [
        ("pooled_xgb_rich_none", "Pooled XGBoost"),
        ("mlp_rich_none", "Pooled MLP"),
        ("gat_rich_self", "Self-loop GAT"),
        ("gat_rich_real", "Real-graph GAT"),
        ("gru_mlp_rich_none", "GRU + MLP"),
        ("gru_gat_rich_real", "GRU + real GAT"),
    ]
    table = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Model (richer inputs) & Validation MAE & Test MAE \\",
        r"\midrule",
    ]
    for variant, label in rows:
        cells = [
            f"{metric(summary,variant,split):.2f} $\\pm$ {metric(summary,variant,split,'paired_mae_std'):.2f}"
            for split in ["val", "test"]
        ]
        table.append(f"{label} & {cells[0]} & {cells[1]}" + r" \\")
    cells = []
    for split in ["val", "test"]:
        rewired = next(
            row for row in summary["rewiring_variability"] if row["split"] == split
        )
        cells.append(
            f"{rewired['paired_mae_mean']:.2f} $\\pm$ {rewired['paired_mae_training_sd']:.2f}"
        )
    table.append(f"Shuffled GAT (3 graphs) & {cells[0]} & {cells[1]}" + r" \\")
    table.extend([r"\bottomrule", r"\end{tabular}"])
    contrast = summary["validation_intervals"]["mlp_rich_none_to_gat_rich_real"]
    difference = metric(summary, "gat_rich_real") - metric(summary, "mlp_rich_none")
    text = (
        r"""\def\ControlledStudyComplete{1}
\newcommand{\ControlledAbstract}{We evaluate pooled classical neural, graph and temporal controls across five training seeds and 49 market identities. A parameter-matched MLP, self-loop and shuffled graphs separate pooling from cross-zone messaging. Feature ablations and 168-hour GRU histories test additional explanations. Historical publication vintages remain unverified; results are exploratory.}
\newcommand{\ControlledMethods}{\paragraph{Controlled models.} The prespecified study has 100 experiment/seed/topology combinations: four tabular references; MLP and real GAT with recent-price/calendar inputs, plus load alone, weather alone, JAO alone, or all groups; self-loop GAT and three fixed degree-preserving shuffled graphs; and GRU--MLP/GAT temporal pairs. Seeds are 224--228. MLP has two ELU/dropout layers, with width selected by parameter count only (96 for richer inputs, versus GAT's 64 channels/four heads). Models have no zone embeddings. Neural heads share training-only preprocessing, asinh(price/50), L1 loss, Adam 0.001, maximum 30 epochs and validation patience five. Histories contain 168 same-identity completed price intervals ending at Brussels noon, with explicit missingness. GRU has 32 units; both temporal heads share its inputs. XGBoost retains the prior depth-6/600-tree settings.}
\newcommand{\ControlledResults}{\paragraph{Multi-seed comparison.} The richer-input controls below use common finite weekly-history hours. Values are mean $\pm$ sample standard deviation across five seeds, not prediction ensembles. Other feature sets, per-zone/country results, training-defined price-tail/season slices and rewiring variability are in the generated study appendix.
\begin{table}[!ht]\centering\small
"""
        + "\n".join(table)
        + r"""
\caption{Matched MAE (EUR/MWh), five seeds. Lower is better.}\end{table}
\paragraph{Graph contribution.} Real-GAT minus pooled-MLP test MAE is """
        + f"{difference:+.2f}"
        + r""" EUR/MWh; negative favours GAT. The paired validation MAE improvement for MLP to GAT is """
        + f"{contrast['mean_mae_improvement']:.2f}"
        + r""" (95\% calendar-week bootstrap interval ["""
        + f"{contrast['ci95_lower']:.2f}, {contrast['ci95_upper']:.2f}"
        + r"""]). Weeks are sampled jointly across countries after averaging fitted-model losses across seeds. This interval conditions on fitted models; seed and rewiring variability are reported separately. Self-loop GAT controls the architecture, pooled XGBoost/MLP control shared learning, and temporal pairs compare graph use with the same history encoder.
\paragraph{Earlier evidence.} The earlier single-seed real GAT scored 20.83 versus 21.54 for shuffled GAT on matched test hours. France diagnostics motivated recent-price inputs: adding prior-day auction prices reduced MAE from 32.09 to 22.20. These earlier cohorts and learners are separate from this replication.
\paragraph{Timing and limitations.} Previous-day prices are normally auctioned on $D-2$ \cite{auction}; delivery is not publication time. DST guards exclude lag-24 target-day prices. Load, weather and JAO revision/publication vintages remain unverified, including retrospective ERA5 \cite{era5}. No observation after the noon origin enters temporal histories. Training-only topology is static and need not reconstruct historical commissioning dates. Models use identical target keys/masks, retain missing features and never fill targets. Test has been inspected previously: these are conditional exploratory comparisons, not a fresh confirmatory holdout. Multiple feature comparisons do not establish causal effects.
\section{Future work} Validate conclusions on a genuinely new chronological period and archived operational forecasts. Investigate detailed JAO constraint nodes and edge attributes after assessing the pooled, topology, feature and temporal controls, keeping physical flow, scheduled exchanges and MaxBex distinct.
}
"""
    )
    return text


def publish(summary, config):
    if (
        summary["phase"] != "full"
        or summary["job_count"] != len(jobs(config))
        or summary["seeds"] != config["seeds"]
    ):
        raise ValueError("Refusing to publish an incomplete multi-seed study")
    lines = [
        "# Controlled multi-seed study",
        "",
        "All 100 prespecified jobs completed. Scores are averaged across seeds, not ensembled predictions.",
        "",
        "| Variant | Validation paired MAE | Test paired MAE | Test SD |",
        "|---|---:|---:|---:|",
    ]
    variants = sorted({row["variant"] for row in summary["scores"]})
    for variant in variants:
        lines.append(
            f"| {variant} | {metric(summary,variant,'val'):.4f} | {metric(summary,variant):.4f} | {metric(summary,variant,column='paired_mae_std'):.4f} |"
        )
    lines += [
        "",
        "## Validation uncertainty",
        "",
        "Paired calendar-week blocks are resampled jointly across countries; intervals condition on fitted models.",
        "",
    ]
    for name, interval in summary["validation_intervals"].items():
        lines.append(
            f"- {name}: improvement {interval['mean_mae_improvement']:.4f}, 95% interval [{interval['ci95_lower']:.4f}, {interval['ci95_upper']:.4f}]."
        )
    lines += [
        "",
        "## Limits",
        "",
        *["- " + text for text in summary["limitations"]],
        "",
        "Regime thresholds use each zone's training-only 5th/95th percentiles. Country grouping retains unresolved identities. Raw predictions, complete job metadata and hashes remain under runs/controlled_study/.",
    ]
    (ROOT / "docs/controlled_study_results.md").write_text("\n".join(lines) + "\n")
    report = ROOT / "reports/milestone"
    (report / "controlled_study.tex").write_text(controlled_tex(summary))
    write_json(
        report / "controlled_sources.json",
        {
            "fingerprint": summary["fingerprint"],
            "jobs": summary["job_count"],
            "summary_sha256": digest(ROOT / "results/controlled_study_full.json")[
                "sha256"
            ],
        },
    )
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/experiments/build_milestone_report.py")],
        cwd=ROOT,
        check=True,
    )
