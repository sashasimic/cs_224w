"""Rebuild the consolidated experiment record from saved evidence, without fitting."""

import json
import re
import sys
from html import escape
from pathlib import Path

import mistune
import numpy as np
import pandas as pd

from epgnn.dataset_build.storage import digest
from epgnn.reporting.investigation import Report

ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / "docs/experiments.md"
OUTPUT = ROOT / "reports/experiments"


class Sources:
    def __init__(self):
        self.paths = set()

    def json(self, name):
        path = ROOT / "results" / name
        self.paths.add(path)
        return json.loads(path.read_text())

    def csv(self, name):
        path = ROOT / "results" / name
        self.paths.add(path)
        return pd.read_csv(path)


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(map(str, row)) + " |" for row in rows),
        ]
    )


def study_table(report, study):
    variants = report.inventory()
    variants = variants[variants.study.eq(study)].sort_values(
        ["model", "features", "graph", "variant"]
    )
    records = [
        (
            r.variant,
            report.score(study, r.variant, "val"),
            report.score(study, r.variant, "test"),
        )
        for r in variants.itertuples()
    ]
    if len(records) != {"controlled": 20, "airport": 9}[study]:
        raise ValueError("Unexpected number of variants")
    eligible = [r for r in records if not r[0].startswith("gru_")]
    best_val = min(v.paired_mae_mean for _, v, _ in eligible)
    best_test = min(t.paired_mae_mean for _, _, t in eligible)
    rows = []
    for variant, val, test in records:
        affected = variant.startswith("gru_")

        def number(value, deviation, best, affected=affected):
            result = f"{value:.4f} ± {deviation:.4f}"
            return f"**{result}**" if not affected and value == best else result

        rows.append(
            [
                variant + (" †" if affected else ""),
                number(val.paired_mae_mean, val.paired_mae_std, best_val),
                number(test.paired_mae_mean, test.paired_mae_std, best_test),
                f"{test.mae_mean:.4f}",
                f"{test.rmse_mean:.4f}",
            ]
        )
    result = table(
        [
            "Variant",
            "Validation paired MAE",
            "Test paired MAE",
            "Test all-target MAE",
            "Test all-target RMSE",
        ],
        rows,
    )
    if study == "controlled":
        result += "\n\n† Affected fused-GRU training path; retain the numbers as history and exclude them from architecture conclusions. See the next section."
    else:
        result += "\n\nFive seeds per row. Bold is descriptive within this programme; latest/history are alternative airport feature groups."
    return result


def main():
    sources = Sources()
    report = Report(ROOT)  # Verifies both five-seed studies and reconciles zone scores.
    for prefix in ["controlled_study", "airport_temperature"]:
        sources.json(prefix + "_full.json")
        sources.json(prefix + "_full_verification.json")
        sources.csv(prefix + "_full_by_zone.csv")
    sources.json("airport_temperature_full_report_sources.json")
    sources.json("airport_temperature_full_review.json")
    gru = sources.json("partner_gru_full.json")
    verified = sources.json("partner_gru_full_verification.json")
    if (
        not verified["passed"]
        or verified["fingerprint"] != gru["fingerprint"]
        or digest(ROOT / "results/partner_gru_full.json")["sha256"]
        != verified["summary_sha256"]
    ):
        raise ValueError("Verified residual GRU required")
    audit = sources.json("controlled_gru_backend_audit.json")
    sources.json("partner_gru_backend_audit.json")
    sources.json("c1_persistence.json")
    sources.json("partner_c1_xgb_zone_by_country.json")
    sources.json("c1_xgb_be_strict.json")
    if (
        audit["fingerprint"] != report.summaries["controlled"]["fingerprint"]
        or audit["checkpoint_count"] != 10
        or audit["cases_per_checkpoint"] != 4
        or not audit["native_reference_passed"]
        or not audit["saved_replay_passed"]
    ):
        raise ValueError("Complete native/replay backend audit required")
    blocks = {study: study_table(report, study) for study in ["controlled", "airport"]}
    audit_rows = []
    for family in ["gru_mlp", "gru_gat"]:
        checkpoints = [r for r in audit["checkpoints"] if r["job"].startswith(family)]
        if len(checkpoints) != 5:
            raise ValueError("Missing GRU seed")
        inference = [
            c for r in checkpoints for c in r["cases"] if c["split"] != "fit_shuffled"
        ]
        fit = [
            c for r in checkpoints for c in r["cases"] if c["split"] == "fit_shuffled"
        ]
        audit_rows.append(
            [
                family,
                "5",
                f"{max(c['fused_gpu_vs_cpu_float64']['max_eur_mwh'] for c in inference):.6f}",
                f"{min(c['fused_gpu_vs_cpu_float64']['max_eur_mwh'] for c in fit):.2f}–{max(c['fused_gpu_vs_cpu_float64']['max_eur_mwh'] for c in fit):.2f}",
                f"{max(c['native_gpu_vs_cpu_float64']['max_eur_mwh'] for r in checkpoints for c in r['cases']):.6f}",
            ]
        )
    failures = sum(not r["fused_reference_passed"] for r in audit["checkpoints"])
    blocks["gru"] = table(
        [
            "Family",
            "Checkpoints",
            "Largest fused inference difference",
            "Shuffled-input max-difference range",
            "Largest native GPU difference",
        ],
        audit_rows,
    )
    blocks["gru"] += (
        f"\n\nDifferences are EUR/MWh versus the same weights on CPU float64. **{failures}/10 checkpoints failed the fused reference gate** across four batches each (three chronological evaluation batches and one shuffled training-input batch). Tolerances: max ≤0.01, mean ≤0.0001 EUR/MWh. All native and saved-inference replay gates passed."
    )
    temporal_rows = []
    for variant in ["gru_mlp_rich_none", "gru_gat_rich_real"]:
        val, test = (
            report.score("controlled", variant, split) for split in ["val", "test"]
        )
        temporal_rows.append(
            [
                variant,
                "5; affected",
                f"{val.paired_mae_mean:.4f}",
                f"{test.paired_mae_mean:.4f}",
                f"{test.mae_mean:.4f}",
                f"{test.rmse_mean:.4f}",
            ]
        )
    val, test = (
        next(
            r
            for r in gru["aggregate"]
            if r["scope"] == "overall" and r["split"] == split
        )
        for split in ["val", "test"]
    )
    temporal_rows.append(
        [
            "partner_residual_gru_conditional",
            "1; native checked",
            f"{val['paired_mae']:.4f}",
            f"{test['paired_mae']:.4f}",
            f"{test['mae']:.4f}",
            f"{test['rmse']:.4f}",
        ]
    )
    blocks["gru"] += "\n\n" + table(
        [
            "Recipe",
            "Seeds/status",
            "Validation paired MAE",
            "Test paired MAE",
            "Test all-target MAE",
            "Test all-target RMSE",
        ],
        temporal_rows,
    )
    suite = sources.csv("vibing_suite_full_aggregate.csv")
    suite = suite[suite.coverage.eq("all") & suite.scope.eq("overall")]
    rows = []
    for model in [
        "recent_xgb",
        "rich_xgb",
        "neighbour_xgb",
        "real_gat",
        "shuffled_gat",
    ]:
        val = suite[suite.experiment.eq(model) & suite.split.eq("val")].iloc[0]
        test = suite[suite.experiment.eq(model) & suite.split.eq("test")].iloc[0]
        rows.append(
            [
                model,
                f"{val.paired_mae:.4f}",
                f"{test.paired_mae:.4f}",
                f"{test.mae:.4f}",
                f"{test.rmse:.4f}",
            ]
        )
    blocks["suite"] = table(
        [
            "Model",
            "Validation paired MAE",
            "Test paired MAE",
            "Test all-target MAE",
            "Test all-target RMSE",
        ],
        rows,
    )
    raw = sources.csv("c1_xgb_countries_weekly_conditional.csv")
    transformed = sources.csv("c1_xgb_countries_weekly_asinh_l1_conditional.csv")
    partner = sources.csv("partner_vs_audited_weekly_country_comparison.csv")
    early = (
        raw[raw.split.eq("test")]
        .merge(
            transformed[transformed.split.eq("test")],
            on="zone",
            suffixes=("_raw", "_asinh"),
            validate="one_to_one",
        )
        .merge(partner[partner.split.eq("test")], on="zone", validate="one_to_one")
    )
    if len(early) != 10:
        raise ValueError("Ten-country cohort changed")
    np.testing.assert_array_equal(early.paired_hours_raw, early.paired_hours_asinh)
    blocks["countries"] = table(
        [
            "Market",
            "Raw weekly XGB",
            "Asinh/L1 weekly XGB",
            "Partner pipeline",
            "Persistence",
        ],
        [
            [
                r.zone,
                f"{r.xgb_mae_raw:.4f}",
                f"{r.xgb_mae_asinh:.4f}",
                f"{r.partner_pipeline_mae:.4f}",
                f"{r.persistence_mae_raw:.4f}",
            ]
            for r in early.itertuples()
        ],
    )
    jao = sources.csv("c1_xgb_countries_jao48_incremental.csv")
    blocks["jao"] = table(
        ["Market", "Weekly raw XGB", "+ JAO summaries", "Change in MAE"],
        [
            [
                r.zone,
                f"{r.xgb_mae_weekly:.4f}",
                f"{r.xgb_mae_jao:.4f}",
                f"{r.xgb_mae_jao - r.xgb_mae_weekly:+.4f}",
            ]
            for r in jao[jao.split.eq("test")].itertuples()
        ],
    )
    temperature = sources.csv("fr_temperature_experiments.csv")
    rows = []
    for model, group in temperature.groupby("experiment", sort=False):
        val, test = (
            group[group.split.eq("val")].iloc[0],
            group[group.split.eq("test")].iloc[0],
        )
        rows.append(
            [
                model,
                test.policy,
                f"{val.mae:.4f}",
                f"{test.mae:.4f}",
                f"{test.rmse:.4f}",
            ]
        )
    blocks["temperature"] = table(
        ["Profile", "Status", "Validation MAE", "Test MAE", "Test RMSE"], rows
    )
    france = sources.csv("pipeline_diagnostics_france.csv")
    rows = []
    for model, group in france.groupby("profile", sort=False):
        val, test = (
            group[group.split.eq("val")].iloc[0],
            group[group.split.eq("test")].iloc[0],
        )
        rows.append(
            [model, f"{val.mae:.4f}", f"{test.mae:.4f}", str(test.selected_rounds)]
        )
    blocks["france"] = table(
        ["Profile", "Validation MAE", "Test MAE", "Selected rounds"], rows
    )
    document = DOCUMENT.read_text()
    for name, content in blocks.items():
        pattern = rf"(<!-- generated:{name} -->).*?(<!-- end:{name} -->)"
        document, count = re.subn(
            pattern,
            lambda m, content=content: m[1] + "\n\n" + content + "\n\n" + m[2],
            document,
            flags=re.DOTALL,
        )
        if count != 1:
            raise ValueError("Non-unique report section: " + name)
    if "torch" in sys.modules or "xgboost" in sys.modules:
        raise ValueError("Report compilation must not import training frameworks")
    DOCUMENT.write_text(document)
    OUTPUT.mkdir(exist_ok=True)
    rendered = mistune.create_markdown(escape=False, plugins=["table"])(document)
    # HTML lives two levels below the root; Markdown links are relative to docs/.
    rendered = re.sub(
        r'href="(?![a-z]+:|#|/)([^"]+)"',
        lambda m: 'href="../../docs/' + m[1] + '"',
        rendered,
    )
    contents = []

    def heading(match):
        title = match[1]
        slug = re.sub(r"[^a-z0-9 -]", "", title.lower()).replace(" ", "-")
        contents.append(f'<a href="#{slug}">{title}</a>')
        return f'<h2 id="{slug}">{title}</h2>'

    rendered = re.sub(r"<h2>(.*?)</h2>", heading, rendered)
    navigation = '<nav aria-label="Contents">' + " · ".join(contents) + "</nav>"
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape("Electricity forecasting — experiment record")}</title>
<style>body{{font:17px/1.65 system-ui,sans-serif;color:#20313d;background:#f5f6f8;margin:0}}
main{{max-width:1180px;margin:auto;background:white;padding:40px}}h1,h2{{line-height:1.2;color:#123d54}}
h2{{margin-top:2em;border-top:1px solid #ddd;padding-top:1em}}a{{color:#17688a}}table{{font-size:14px;display:block;overflow:auto;border-collapse:collapse;margin:1.5em 0}}
th,td{{text-align:left;border-bottom:1px solid #dde3e7;padding:8px 12px}}th{{background:#edf3f7}}tr:nth-child(even){{background:#fafbfd}}
code,pre{{background:#eef2f5;font-size:.88em}}pre{{padding:16px;overflow:auto}}@media(max-width:700px){{main{{padding:18px}}}}
@media print{{body{{background:white}}main{{padding:0}}table{{font-size:10px}}h2{{break-after:avoid}}}}</style></head>
<body><main>{navigation}{rendered}</main></body></html>"""
    destination = OUTPUT / "experiments.html"
    destination.write_text(html)
    sources.paths.add(Path(__file__))
    sources.paths.add(ROOT / "epgnn/reporting/investigation.py")
    sources.paths.add(ROOT / "scripts/experiments/check_controlled_gru_backend.py")
    manifest = {
        "training_performed": False,
        "gpu_used": False,
        "controlled_jobs": 100,
        "airport_jobs": 45,
        "controlled_gru_checkpoints_audited": audit["checkpoint_count"],
        "sources": {
            str(p.relative_to(ROOT)): digest(p)["sha256"] for p in sorted(sources.paths)
        },
        "artifacts": {
            str(p.relative_to(ROOT)): digest(p)["sha256"]
            for p in [DOCUMENT, destination]
        },
    }
    (OUTPUT / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Compiled {len(blocks)} result sections from {len(sources.paths)} sources; no training or GPU imports"
    )


if __name__ == "__main__":
    main()
