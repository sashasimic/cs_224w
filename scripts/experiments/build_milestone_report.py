"""Generate milestone tables/plots directly from saved metrics, then compile PDF."""
import json
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


def load(path):
    return json.loads((ROOT / path).read_text())


def main():
    diagnostic = load("results/pipeline_diagnostics_france.json")
    audit = load("results/milestone_pipeline_audit.json")
    partner = load("results/partner_c1_xgb_zone_by_country.json")
    lookup = {(r["profile"], r["split"]): r for r in diagnostic["rows"]}
    scores = {(r["model"], r["split"]): r["mae"] for r in audit["matched_ten_country_scores"]}
    if audit["france_diagnostic_fingerprint"] != diagnostic["fingerprint"]:
        raise ValueError("The milestone audit refers to different France diagnostics")
    # Refuse a stale hand-collated table when any underlying run changes.
    raw = load("results/c1_xgb_countries_weekly_conditional.json")
    transformed = load("results/c1_xgb_countries_weekly_asinh_l1_conditional.json")
    matched = load("results/partner_vs_audited_weekly_country_comparison.json")
    for split in ["val", "test"]:
        rows = [r for r in matched["rows"] if r["split"] == split]
        if len(rows) != 10 or not all(r["target_masks_match"] and r["target_prices_match_within_float32_rounding"] for r in rows):
            raise ValueError("The ten-zone comparison has unmatched targets")
        expected = [raw["aggregate"][split]["persistence_mae"], raw["aggregate"][split]["xgb_mae"],
                    transformed["aggregate"][split]["xgb_mae"],
                    np.average([r["partner_pipeline_mae"] for r in rows], weights=[r["hours"] for r in rows])]
        models = ["Weekly persistence", "Raw-price squared-error XGBoost", "Asinh/L1 weekly XGBoost", "Full partner pipeline"]
        np.testing.assert_allclose([scores[(m, split)] for m in models], expected, rtol=0, atol=1e-9)
    names = ["Weekly persistence", "Raw-price squared-error XGBoost", "Asinh/L1 weekly XGBoost", "Full partner pipeline"]
    labels = ["Weekly persistence", "Calendar + weekly, raw-price MSE", "Same inputs, asinh + L1", "Reproduced partner, richer inputs"]
    lines = [r"\begin{tabular}{lrr}", r"\toprule", r"Method (same ten zones) & Validation MAE & Test MAE \\", r"\midrule"]
    for name, label in zip(names, labels):
        lines.append(f"{label} & {scores[(name, 'val')]:.2f} & {scores[(name, 'test')]:.2f}" + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (REPORT / "matched_results.tex").write_text("\n".join(lines) + "\n")
    values = {"WeeklyCalendarTest": lookup[("weekly_calendar", "test")]["mae"],
              "RecentPriceTest": lookup[("recent_price", "test")]["mae"],
              "PastWeatherTest": lookup[("weather_past24", "test")]["mae"],
              "AllWeatherTest": lookup[("weather_all24", "test")]["mae"],
              "WeatherFortyEightTest": lookup[("weather48", "test")]["mae"],
              "PartnerFranceTest": next(r["mae"] for r in partner["country_scores"] if r["zone"] == "FR" and r["split"] == "test"),
              "PriceBootstrapLower": diagnostic["validation_week_block_bootstrap"]["weekly_calendar_to_recent_price"]["ci95_lower"],
              "PriceBootstrapUpper": diagnostic["validation_week_block_bootstrap"]["weekly_calendar_to_recent_price"]["ci95_upper"]}
    (REPORT / "numbers.tex").write_text("\n".join(f"\\newcommand{{\\{name}}}{{{value:.2f}}}" for name, value in values.items()) + "\n")
    profiles = ["weekly_flags", "weekly_calendar", "recent_price", "load", "flows", "weather_past24", "weather_all24"]
    stage_labels = ["Weekly + era flags", "Weekly + calendar", "+ Prior-day auction price", "+ Load forecast", "+ Lagged net imports", "+ Past-time weather", "+ Post-origin weather"]
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.25), sharey=True)
    for ax, split in zip(axes, ["val", "test"]):
        y = np.arange(len(profiles))
        metric = [lookup[(p, split)]["mae"] for p in profiles]
        colors = ["#8d99a6", "#8d99a6", "#1a6489", "#448fb2", "#448fb2", "#448fb2", "#cf7e35"]
        ax.barh(y, metric, height=.66, color=colors)
        for index, value in enumerate(metric):
            ax.text(value + .35, index, f"{value:.2f}", va="center", fontsize=7)
        ax.set_xlim(0, 42)
        ax.set_xticks([0, 10, 20, 30, 40])
        ax.set_xlabel("MAE (EUR/MWh)")
        ax.set_title("Validation" if split == "val" else "Test", fontsize=9)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", alpha=.18)
        ax.set_axisbelow(True)
    axes[0].set_yticks(np.arange(len(profiles)), stage_labels)
    axes[0].invert_yaxis()
    fig.subplots_adjust(left=.34, right=.98, bottom=.19, top=.87, wspace=.16)
    for ext in ["pdf", "png"]:
        fig.savefig(REPORT / "figures" / f"france_ablation.{ext}", dpi=240, bbox_inches="tight", pad_inches=.02)
    plt.close(fig)
    env = dict(os.environ, TEXMFVAR="/tmp/cs224w-milestone-texmf")
    for _ in range(2):
        result = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "milestone.tex"],
                                cwd=REPORT, env=env, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(result.stdout[-7000:] + result.stderr[-1000:])
    info = subprocess.run(["pdfinfo", str(REPORT / "milestone.pdf")], capture_output=True, text=True, check=True).stdout
    if not any(line.split() == ["Pages:", "2"] for line in info.splitlines()):
        raise ValueError("Milestone must fit exactly two pages:\n" + info)
    print(info)


if __name__ == "__main__":
    main()
