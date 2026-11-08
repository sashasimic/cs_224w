"""Presentation of frozen milestone scores; no fitting or model selection."""

ROWS = [
    (None, "Weekly persistence"),
    ("pooled_xgb_rich_none", "Pooled XGBoost"),
    ("mlp_rich_none", "Pooled MLP"),
    ("gat_rich_real", "Real-graph GAT"),
    ("gat_rich_self", "Self-loop GAT"),
    ("shuffled", "Shuffled GAT (3 graphs)"),
    ("gru_mlp_rich_none", "GRU + MLP"),
    ("gru_gat_rich_real", "GRU + real GAT"),
]


def score(summary, variant, split):
    return next(
        r
        for r in summary["scores"]
        if r["variant"] == variant
        and r["split"] == split
        and r["coverage"] == "all"
        and r["scope"] == "overall"
    )


def comparison_rows(summary):
    rows = []
    for variant, label in ROWS:
        row = {"variant": variant, "label": label}
        for split in ["val", "test"]:
            baseline = score(summary, "gat_rich_real", split)["persistence_mae_mean"]
            if baseline <= 0:
                raise ValueError("Relative MAE needs a positive benchmark error")
            if variant is None:
                mean, sd = baseline, None
            elif variant == "shuffled":
                item = next(
                    r for r in summary["rewiring_variability"] if r["split"] == split
                )
                mean, sd = item["paired_mae_mean"], item["paired_mae_training_sd"]
            else:
                item = score(summary, variant, split)
                if abs(item["persistence_mae_mean"] - baseline) > 1e-9:
                    raise ValueError("Benchmark denominators differ across rows")
                mean, sd = item["paired_mae_mean"], item["paired_mae_std"]
            row[split], row[split + "_sd"] = mean, sd
            row[split + "_rmae"] = mean / baseline
        rows.append(row)
    return rows


def render_table(summary):
    rows = comparison_rows(summary)
    best = {split: min(row[split] for row in rows) for split in ["val", "test"]}
    lines = [
        r"\begin{table}[!ht]\centering\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Forecasting method & Validation MAE & Test MAE & Test rMAE \\",
        r"\midrule",
    ]
    for i, row in enumerate(rows):
        if i in [1, 3, 6]:
            lines.append(r"\midrule")
        cells = []
        for split in ["val", "test"]:
            cell = f"{row[split]:.2f}"
            if row[split + "_sd"] is not None:
                cell += f" $\\pm$ {row[split + '_sd']:.2f}"
            if row[split] == best[split]:
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        relative = f"{row['test_rmae']:.3f}"
        if row["test"] == best["test"]:
            relative = r"\textbf{" + relative + "}"
        lines.append(" & ".join([row["label"], *cells, relative]) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Grouped by model class; learned models use richer inputs. MAE in EUR/MWh; rMAE divides test MAE by weekly-persistence MAE on the same hours (1 = baseline). Bold denotes the smallest mean per column, not statistical significance.}\label{tab:controlled}\end{table}",
    ]
    return "\n".join(lines)


FUTURE_WORK = r"""\section{Future work} Build a daily forecast service that saves predictions before Brussels noon, together with received inputs and model versions. Score forecasts when auction results arrive; monitor rolling rMAE, data completeness, deadline misses and fallback use. Start with price-only pooled XGBoost and persistence, then compare MLP/GAT and timely station observations. This prospective experiment measures accuracy, drift and operational reliability. Assess detailed constraint graphs after establishing the baseline service."""
