"""Compile seven verified forecasting references into milestone v3; no fitting."""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/milestone_v3"
STATIC = [
    ("xgb_rich_none", "Per-zone XGBoost"),
    ("pooled_xgb_rich_none", "Pooled XGBoost"),
    ("mlp_rich_none", "Static pooled MLP"),
    ("gat_rich_self", "Self-loop GAT"),
    ("gat_rich_real", "Real-graph GAT"),
]
COUNTS = {"val": (428427, 428213), "test": (634578, 633885)}
INPUTS = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(name):
    path = ROOT / name
    INPUTS[str(path.relative_to(ROOT))] = sha(path)
    return json.loads(path.read_text())


def score(summary, variant, split):
    selected = [
        r
        for r in summary["scores"]
        if r["variant"] == variant
        and r["split"] == split
        and r["scope"] == "overall"
        and r["coverage"] == "all"
    ]
    if len(selected) != 1:
        raise ValueError("Missing/duplicate overall score")
    return selected[0]


def inputs():
    summary = read_json("results/controlled_study_full.json")
    verified = read_json("results/controlled_study_full_verification.json")
    gru = read_json("results/partner_gru_full.json")
    gru_verified = read_json("results/partner_gru_full_verification.json")
    for record, evidence, path in [
        (summary, verified, "results/controlled_study_full.json"),
        (gru, gru_verified, "results/partner_gru_full.json"),
    ]:
        if (
            not evidence["passed"]
            or record["fingerprint"] != evidence["fingerprint"]
            or INPUTS[path] != evidence["summary_sha256"]
        ):
            raise ValueError("Verified score generation required: " + path)
    if summary["job_count"] != 100 or gru["phase"] != "full" or not gru["verified"]:
        raise ValueError("Completed full evaluations required")
    audit = read_json("results/controlled_gru_backend_audit.json")
    read_json("results/partner_gru_backend_audit.json")
    config = read_json("configs/experiments/partner_gru.json")
    read_json("configs/experiments/controlled_study.json")
    if (
        config["rnn_backend"] != "native_gpu_verified_against_cpu"
        or not audit["native_reference_passed"]
        or audit["fused_reference_passed"]
    ):
        raise ValueError("Current backend correction must be represented")
    zones_path = ROOT / "results/controlled_study_full_by_zone.csv"
    INPUTS[str(zones_path.relative_to(ROOT))] = sha(zones_path)
    zones = pd.read_csv(zones_path)
    rows = [
        {"variant": "weekly_persistence", "label": "Weekly persistence", "seeds": 0}
    ]
    for variant, label in STATIC:
        row = {"variant": variant, "label": label, "seeds": 5}
        for split, counts in COUNTS.items():
            saved = score(summary, variant, split)
            selected = zones[
                zones.variant.eq(variant)
                & zones.split.eq(split)
                & zones.coverage.eq("all")
            ]
            values = []
            for seed, group in selected.groupby("seed"):
                if (
                    group.zone.nunique() != 49
                    or len(group) != 49
                    or (int(group.hours.sum()), int(group.paired_hours.sum())) != counts
                ):
                    raise ValueError("Changed scored cohort/counts")
                job = f"{variant}_seed{seed}"
                directory = (
                    Path("runs/controlled_study")
                    / summary["fingerprint"]
                    / "full"
                    / job
                )
                result = read_json(directory / "result.json")
                complete = read_json(directory / "complete.json")
                if (
                    complete["result.json"] != INPUTS[str(directory / "result.json")]
                    or result["target_signatures"] != gru["target_signatures"]
                    or result["identity"] != summary["fingerprint"]
                ):
                    raise ValueError("Selected model target/source identity changed")
                values.append(np.average(group.paired_mae, weights=group.paired_hours))
            if set(selected.seed) != {224, 225, 226, 227, 228}:
                raise ValueError("Expected five training seeds")
            np.testing.assert_allclose(
                [np.mean(values), np.std(values, ddof=1)],
                [saved["paired_mae_mean"], saved["paired_mae_std"]],
                rtol=0,
                atol=1e-9,
            )
            row[split], row[split + "_sd"] = (
                saved["paired_mae_mean"],
                saved["paired_mae_std"],
            )
            baseline = score(summary, "gat_rich_real", split)["persistence_mae_mean"]
            np.testing.assert_allclose(
                saved["persistence_mae_mean"], baseline, rtol=0, atol=1e-9
            )
            rows[0][split], rows[0][split + "_sd"] = baseline, None
        rows.append(row)
    residual = {
        "variant": "partner_residual_gru_conditional",
        "label": r"GRU + MLP (partner)$^\dagger$",
        "seeds": 1,
    }
    for split, counts in COUNTS.items():
        saved = next(
            r
            for r in gru["aggregate"]
            if r["scope"] == "overall" and r["split"] == split
        )
        if (saved["hours"], saved["paired_hours"]) != counts:
            raise ValueError("GRU target counts differ")
        np.testing.assert_allclose(
            saved["persistence_mae"], rows[0][split], rtol=0, atol=1e-9
        )
        check = gru["cpu_reference_checks"][split]
        if (
            check["max_absolute_eur_mwh"] >= 0.0001
            or check["mean_absolute_eur_mwh"] > check["mean_allowed_eur_mwh"]
        ):
            raise ValueError("GRU reference check no longer supports report claim")
        residual[split], residual[split + "_sd"] = saved["paired_mae"], None
    if gru_verified["target_signatures"] != gru["target_signatures"]:
        raise ValueError("GRU canonical verification mismatch")
    rows.insert(4, residual)
    for row in rows:
        row["test_rmae"] = row["test"] / rows[0]["test"]
    return summary, gru, rows


def table(rows):
    best = {split: min(r[split] for r in rows) for split in ["val", "test"]}
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Forecasting method & Seeds & Val. MAE & Test MAE & Test rMAE \\",
        r"\midrule",
    ]
    for i, row in enumerate(rows):
        if i in [1, 3, 5]:
            lines.append(r"\midrule")
        cells = []
        for split in ["val", "test"]:
            cell = f"{row[split]:.2f}"
            if row[split + "_sd"] is not None:
                cell += f" $\\pm$ {row[split + '_sd']:.2f}"
            cells.append(
                r"\textbf{" + cell + "}" if row[split] == best[split] else cell
            )
        relative = f"{row['test_rmae']:.4f}"
        if row["test"] == best["test"]:
            relative = r"\textbf{" + relative + "}"
        lines.append(
            " & ".join(
                [
                    row["label"],
                    str(row["seeds"]) if row["seeds"] else "--",
                    *cells,
                    relative,
                ]
            )
            + r" \\"
        )
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}"]) + "\n"


def numbers(summary, gru, rows):
    by_name = {r["variant"]: r for r in rows}
    mapping = {
        "ZoneTest": "xgb_rich_none",
        "PooledTest": "pooled_xgb_rich_none",
        "MlpTest": "mlp_rich_none",
        "GatTest": "gat_rich_real",
        "GruTest": "partner_residual_gru_conditional",
    }
    values = {macro: f"{by_name[name]['test']:.2f}" for macro, name in mapping.items()}
    values.update(
        {
            "ValPaired": "428,213",
            "TestPaired": "633,885",
            "ValAll": "428,427",
            "TestAll": "634,578",
            "MlpGatDifference": f"{by_name['mlp_rich_none']['test'] - by_name['gat_rich_real']['test']:.2f}",
            "SelfGatDifference": f"{by_name['gat_rich_self']['test'] - by_name['gat_rich_real']['test']:.2f}",
            "GatRmse": f"{score(summary, 'gat_rich_real', 'test')['rmse_mean']:.2f}",
            "GruRmse": f"{next(r for r in gru['aggregate'] if r['scope'] == 'overall' and r['split'] == 'test')['rmse']:.2f}",
            "GruBestEpoch": str(
                min(gru["curves"], key=lambda r: r["val_residual_plus_spread"])["epoch"]
            ),
            "GruEpochs": str(len(gru["curves"])),
        }
    )
    for prefix, contrast in [
        ("Mlp", "mlp_rich_none_to_gat_rich_real"),
        ("Self", "gat_rich_self_to_gat_rich_real"),
    ]:
        interval = summary["validation_intervals"][contrast]
        values[prefix + "CiLower"], values[prefix + "CiUpper"] = (
            f"{interval['ci95_lower']:.2f}",
            f"{interval['ci95_upper']:.2f}",
        )
    return (
        "\n".join(
            "\\newcommand{\\" + macro + "}{" + value + "}"
            for macro, value in values.items()
        )
        + "\n"
    )


def main():
    summary, gru, rows = inputs()
    preserved = {
        f"reports/milestone_v{version}/milestone.pdf": sha(
            ROOT / f"reports/milestone_v{version}/milestone.pdf"
        )
        for version in [1, 2]
    }
    REPORT.mkdir(exist_ok=True)
    shutil.copyfile(
        ROOT / "reports/milestone_v2/neurips_2019.sty", REPORT / "neurips_2019.sty"
    )
    (REPORT / "results.tex").write_text(table(rows))
    (REPORT / "numbers.tex").write_text(numbers(summary, gru, rows))
    (REPORT / "results.json").write_text(
        json.dumps(
            {
                "selection": "Requested model families; common richer static controls, verified partner-inspired GRU. No test-minimum variant or seed selection.",
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    env = dict(
        os.environ,
        TEXMFVAR="/tmp/cs224w-milestone-v3-texmf",
        SOURCE_DATE_EPOCH=str(
            int(datetime(2026, 11, 9, tzinfo=timezone.utc).timestamp())
        ),
        FORCE_SOURCE_DATE="1",
    )
    with tempfile.TemporaryDirectory(prefix="cs224w-milestone-v3-") as temporary:
        build = Path(temporary)
        for path in REPORT.iterdir():
            if path.suffix in {".tex", ".sty"}:
                shutil.copyfile(path, build / path.name)
        for _ in range(2):
            result = subprocess.run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "milestone.tex",
                ],
                cwd=build,
                check=False,
                env=env,
                text=True,
                capture_output=True,
            )
            (REPORT / "build.log").write_text(result.stdout)
            if result.returncode:
                raise RuntimeError(result.stdout[-4000:])
        info = subprocess.check_output(
            ["pdfinfo", str(build / "milestone.pdf")], text=True
        )
        if not any(line.split() == ["Pages:", "2"] for line in info.splitlines()):
            raise ValueError("Milestone v3 must have exactly two pages:\n" + info)
        text = subprocess.check_output(
            ["pdftotext", str(build / "milestone.pdf"), "-"], text=True
        )
        if (
            "GRU + MLP" not in text
            or "Real-graph GAT" not in text
            or "Future work" not in text
        ):
            raise ValueError("Expected methods/sections missing from PDF")
        if "Overfull \\hbox" in result.stdout or "Overfull \\vbox" in result.stdout:
            raise ValueError("Milestone has overflowing content; inspect build.log")
        staged = REPORT / "milestone.pdf.tmp"
        shutil.copyfile(build / "milestone.pdf", staged)
        staged.replace(REPORT / "milestone.pdf")
    if any(sha(ROOT / path) != checksum for path, checksum in preserved.items()):
        raise ValueError("Earlier milestone snapshot changed")
    artifacts = [
        REPORT / name
        for name in [
            "milestone.tex",
            "neurips_2019.sty",
            "numbers.tex",
            "results.tex",
            "results.json",
            "milestone.pdf",
        ]
    ]
    INPUTS[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    manifest = {
        "training_performed": False,
        "gpu_used": False,
        "pages": 2,
        "controlled_fingerprint": summary["fingerprint"],
        "gru_fingerprint": gru["fingerprint"],
        "target_signatures": gru["target_signatures"],
        "counts": COUNTS,
        "sources": INPUTS,
        "preserved_milestones": preserved,
        "artifacts": {str(p.relative_to(ROOT)): sha(p) for p in artifacts},
    }
    (REPORT / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(info)


if __name__ == "__main__":
    main()
