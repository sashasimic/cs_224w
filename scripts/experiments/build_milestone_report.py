"""Generate milestone tables/plots directly from saved metrics, then compile PDF."""

import json
import hashlib
import os
from pathlib import Path
import subprocess

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cs224w-milestone-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/milestone"
SOURCE_HASHES = {}


def load(path):
    content = (ROOT / path).read_bytes()
    SOURCE_HASHES[path] = hashlib.sha256(content).hexdigest()
    return json.loads(content)


def build_suite_assets():
    """Use the verified full run, keeping its cohort separate from earlier studies."""
    path = ROOT / "results/vibing_suite_full.json"
    suite = load("results/vibing_suite_full.json")
    verification = load("results/vibing_suite_full_verification.json")
    if (
        not verification["passed"]
        or verification["summary_sha256"]
        != hashlib.sha256(path.read_bytes()).hexdigest()
    ):
        raise ValueError("Full-suite verification does not match the saved result")
    if suite["fingerprint"] != verification["fingerprint"]:
        raise ValueError("Full-suite fingerprints differ")
    rows = {
        (row["experiment"], row["split"]): row
        for row in suite["aggregate"]
        if row["scope"] == "overall" and row["coverage"] == "all"
    }
    names = [
        ("recent_xgb", "Recent-price XGBoost"),
        ("rich_xgb", "Richer XGBoost"),
        ("neighbour_xgb", "Neighbour XGBoost"),
        ("real_gat", "Real-graph GAT"),
        ("shuffled_gat", "Shuffled-graph GAT"),
    ]
    for split in ["val", "test"]:
        zone_sets = [
            set(
                row["zone"]
                for row in item["rows"]
                if row["split"] == split and row["coverage"] == "all"
            )
            for item in suite["experiments"]
        ]
        if (
            len(zone_sets) != 5
            or any(zones != zone_sets[0] for zones in zone_sets)
            or len(zone_sets[0]) != 49
        ):
            raise ValueError("Full-suite evaluation cohorts differ")
        if (
            len(
                {
                    (rows[name, split]["hours"], rows[name, split]["paired_hours"])
                    for name, _ in names
                }
            )
            != 1
        ):
            raise ValueError("Full-suite comparison populations differ")
    table = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Method (49 market identities) & Validation MAE & Test MAE \\",
        r"\midrule",
    ]
    table.append(
        f"Weekly persistence & {rows['real_gat', 'val']['persistence_mae']:.2f} & {rows['real_gat', 'test']['persistence_mae']:.2f}"
        + r" \\"
    )
    for name, label in names:
        val, test = rows[name, "val"]["paired_mae"], rows[name, "test"]["paired_mae"]
        table.append(f"{label} & {val:.2f} & {test:.2f}" + r" \\")
    table.extend([r"\bottomrule", r"\end{tabular}"])
    (REPORT / "suite_results.tex").write_text("\n".join(table) + "\n")
    graph, shuffled, recent = [
        rows[name, "test"] for name in ["real_gat", "shuffled_gat", "recent_xgb"]
    ]
    values = {
        "SuiteGraphTest": f"{graph['paired_mae']:.2f}",
        "SuiteShuffleGain": f"{100*(1-graph['paired_mae']/shuffled['paired_mae']):.2f}",
        "SuiteXgbGain": f"{100*(1-graph['paired_mae']/recent['paired_mae']):.2f}",
        "SuitePersistenceGain": f"{100*(1-graph['relative_mae']):.2f}",
        "SuiteGraphRmse": f"{graph['rmse']:.2f}",
        "SuiteRecentRmse": f"{recent['rmse']:.2f}",
        "SuiteValHours": f"{rows['real_gat', 'val']['paired_hours']:,}",
        "SuiteTestHours": f"{graph['paired_hours']:,}",
        "SuiteAllValHours": f"{rows['real_gat', 'val']['hours']:,}",
        "SuiteAllTestHours": f"{graph['hours']:,}",
    }
    (REPORT / "suite_numbers.tex").write_text(
        "\n".join(
            f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in values.items()
        )
        + "\n"
    )
    return suite


def main():
    suite = build_suite_assets()
    diagnostic = load("results/pipeline_diagnostics_france.json")
    audit = load("results/milestone_pipeline_audit.json")
    partner = load("results/partner_c1_xgb_zone_by_country.json")
    lookup = {(r["profile"], r["split"]): r for r in diagnostic["rows"]}
    scores = {
        (r["model"], r["split"]): r["mae"] for r in audit["matched_ten_country_scores"]
    }
    if audit["france_diagnostic_fingerprint"] != diagnostic["fingerprint"]:
        raise ValueError("The milestone audit refers to different France diagnostics")
    # Refuse a stale hand-collated table when any underlying run changes.
    raw = load("results/c1_xgb_countries_weekly_conditional.json")
    transformed = load("results/c1_xgb_countries_weekly_asinh_l1_conditional.json")
    matched = load("results/partner_vs_audited_weekly_country_comparison.json")
    for split in ["val", "test"]:
        rows = [r for r in matched["rows"] if r["split"] == split]
        if len(rows) != 10 or not all(
            r["target_masks_match"] and r["target_prices_match_within_float32_rounding"]
            for r in rows
        ):
            raise ValueError("The ten-zone comparison has unmatched targets")
        expected = [
            raw["aggregate"][split]["persistence_mae"],
            raw["aggregate"][split]["xgb_mae"],
            transformed["aggregate"][split]["xgb_mae"],
            np.average(
                [r["partner_pipeline_mae"] for r in rows],
                weights=[r["hours"] for r in rows],
            ),
        ]
        models = [
            "Weekly persistence",
            "Raw-price squared-error XGBoost",
            "Asinh/L1 weekly XGBoost",
            "Full partner pipeline",
        ]
        np.testing.assert_allclose(
            [scores[(m, split)] for m in models], expected, rtol=0, atol=1e-9
        )
    names = [
        "Weekly persistence",
        "Raw-price squared-error XGBoost",
        "Asinh/L1 weekly XGBoost",
        "Full partner pipeline",
    ]
    labels = [
        "Weekly persistence",
        "Calendar + weekly, raw-price MSE",
        "Same inputs, asinh + L1",
        "Reproduced partner, richer inputs",
    ]
    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Method (same ten zones) & Validation MAE & Test MAE \\",
        r"\midrule",
    ]
    for name, label in zip(names, labels):
        lines.append(
            f"{label} & {scores[(name, 'val')]:.2f} & {scores[(name, 'test')]:.2f}"
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (REPORT / "matched_results.tex").write_text("\n".join(lines) + "\n")
    values = {
        "WeeklyCalendarTest": lookup[("weekly_calendar", "test")]["mae"],
        "RecentPriceTest": lookup[("recent_price", "test")]["mae"],
        "PastWeatherTest": lookup[("weather_past24", "test")]["mae"],
        "AllWeatherTest": lookup[("weather_all24", "test")]["mae"],
        "WeatherFortyEightTest": lookup[("weather48", "test")]["mae"],
        "PartnerFranceTest": next(
            r["mae"]
            for r in partner["country_scores"]
            if r["zone"] == "FR" and r["split"] == "test"
        ),
        "PriceBootstrapLower": diagnostic["validation_week_block_bootstrap"][
            "weekly_calendar_to_recent_price"
        ]["ci95_lower"],
        "PriceBootstrapUpper": diagnostic["validation_week_block_bootstrap"][
            "weekly_calendar_to_recent_price"
        ]["ci95_upper"],
    }
    (REPORT / "numbers.tex").write_text(
        "\n".join(
            f"\\newcommand{{\\{name}}}{{{value:.2f}}}" for name, value in values.items()
        )
        + "\n"
    )
    profiles = [
        "weekly_flags",
        "weekly_calendar",
        "recent_price",
        "load",
        "flows",
        "weather_past24",
        "weather_all24",
    ]
    stage_labels = [
        "Weekly + era flags",
        "Weekly + calendar",
        "+ Prior-day auction price",
        "+ Load forecast",
        "+ Lagged net imports",
        "+ Past-time weather",
        "+ Post-origin weather",
    ]
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.25), sharey=True)
    for ax, split in zip(axes, ["val", "test"]):
        y = np.arange(len(profiles))
        metric = [lookup[(p, split)]["mae"] for p in profiles]
        colors = [
            "#8d99a6",
            "#8d99a6",
            "#1a6489",
            "#448fb2",
            "#448fb2",
            "#448fb2",
            "#cf7e35",
        ]
        ax.barh(y, metric, height=0.66, color=colors)
        for index, value in enumerate(metric):
            ax.text(value + 0.35, index, f"{value:.2f}", va="center", fontsize=7)
        ax.set_xlim(0, 42)
        ax.set_xticks([0, 10, 20, 30, 40])
        ax.set_xlabel("MAE (EUR/MWh)")
        ax.set_title("Validation" if split == "val" else "Test", fontsize=9)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", alpha=0.18)
        ax.set_axisbelow(True)
    axes[0].set_yticks(np.arange(len(profiles)), stage_labels)
    axes[0].invert_yaxis()
    fig.subplots_adjust(left=0.34, right=0.98, bottom=0.19, top=0.87, wspace=0.16)
    for ext in ["pdf", "png"]:
        fig.savefig(
            REPORT / "figures" / f"france_ablation.{ext}",
            dpi=240,
            bbox_inches="tight",
            pad_inches=0.02,
        )
    plt.close(fig)
    (REPORT / "sources.json").write_text(
        json.dumps(
            {
                "full_suite_fingerprint": suite["fingerprint"],
                "france_diagnostic_fingerprint": diagnostic["fingerprint"],
                "inputs_sha256": SOURCE_HASHES,
                "main_table_population": "49 identities, common finite weekly-history target hours",
                "rmse_population": "all valid test targets, distinct from the paired MAE table",
            },
            indent=2,
        )
        + "\n"
    )
    env = dict(os.environ, TEXMFVAR="/tmp/cs224w-milestone-texmf")
    for _ in range(2):
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "milestone.tex"],
            cwd=REPORT,
            env=env,
            text=True,
            capture_output=True,
        )
        if result.returncode:
            raise RuntimeError(result.stdout[-7000:] + result.stderr[-1000:])
    info = subprocess.run(
        ["pdfinfo", str(REPORT / "milestone.pdf")],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if not any(line.split() == ["Pages:", "2"] for line in info.splitlines()):
        raise ValueError("Milestone must fit exactly two pages:\n" + info)
    print(info)


if __name__ == "__main__":
    main()
