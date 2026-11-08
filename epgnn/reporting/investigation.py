"""Load verified result summaries and bounded local diagnostics, without fitting."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from epgnn.dataset_build.contracts import forecast_origin

ROOT = Path(__file__).resolve().parents[2]
CORE = [
    ("controlled", "xgb_recent_none", "Per-zone XGBoost"),
    ("airport", "pooled_xgb_recent_none", "Pooled XGBoost"),
    ("airport", "mlp_recent_none", "MLP"),
    ("airport", "gat_recent_real", "GAT"),
]
PREFIX = {"controlled": "controlled_study_full", "airport": "airport_temperature_full"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


class Report:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.summaries, self.scores, self.zones = {}, {}, {}
        self.sources = []
        for study, prefix in PREFIX.items():
            path = self.root / "results" / (prefix + ".json")
            evidence = read_json(
                self.root / "results" / (prefix + "_verification.json")
            )
            provenance = (
                evidence
                if study == "controlled"
                else read_json(
                    self.root / "results" / (prefix + "_report_sources.json")
                )
            )
            if not evidence["passed"] or sha(path) != provenance["summary_sha256"]:
                raise ValueError("Result verification mismatch: " + prefix)
            summary = read_json(path)
            if evidence["fingerprint"] != summary["fingerprint"]:
                raise ValueError("Result fingerprint mismatch")
            self.summaries[study] = summary
            self.scores[study] = pd.DataFrame(summary["scores"])
            self.zones[study] = pd.read_csv(
                self.root / "results" / (prefix + "_by_zone.csv")
            )
            self.sources.append(
                {
                    "study": study,
                    "fingerprint": summary["fingerprint"],
                    "summary_sha256": sha(path),
                    "jobs": summary["job_count"],
                }
            )
        review = read_json(self.root / "results/airport_temperature_full_review.json")
        if (
            sha(self.root / "results/airport_temperature_full_by_zone.csv")
            != review["zone_scores_sha256"]
        ):
            raise ValueError("Airport zone scores changed")
        if [s["job_count"] for s in self.summaries.values()] != [100, 45]:
            raise ValueError("Unexpected study matrix")
        self.validate_scores()

    def validate_scores(self):
        """Reconcile seed-level zone errors with published overall summaries."""
        for study, frame in self.zones.items():
            frame = frame[frame.coverage.eq("all")].copy()
            if frame.duplicated(["variant", "seed", "split", "zone"]).any():
                raise ValueError("Duplicate score keys")
            for (variant, split), group in frame.groupby(["variant", "split"]):
                per_seed = []
                for _, seed in group.groupby("seed"):
                    per_seed.append(
                        [
                            np.average(seed.mae, weights=seed.hours),
                            np.sqrt(np.average(seed.rmse**2, weights=seed.hours)),
                            np.average(seed.paired_mae, weights=seed.paired_hours),
                        ]
                    )
                saved = self.score(study, variant, split)
                np.testing.assert_allclose(
                    np.mean(per_seed, axis=0),
                    [saved.mae_mean, saved.rmse_mean, saved.paired_mae_mean],
                    rtol=0,
                    atol=1e-9,
                )

    def score(self, study, variant, split="val", country="ALL"):
        frame = self.scores[study]
        scope = "overall" if country == "ALL" else "country"
        rows = frame[
            frame.variant.eq(variant)
            & frame.split.eq(split)
            & frame.coverage.eq("all")
            & frame.scope.eq(scope)
            & frame.country.eq(country)
        ]
        if len(rows) != 1:
            raise ValueError("Non-unique or missing score")
        return rows.iloc[0]

    def inventory(self):
        records = []
        for study, summary in self.summaries.items():
            for job in summary["jobs"]:
                records.append(
                    {"study": study, "variant": job["variant"], **job["job"]}
                )
        frame = pd.DataFrame(records)
        return (
            frame.groupby(
                ["study", "variant", "model", "features", "graph"], dropna=False
            )
            .seed.agg(seeds="nunique", jobs="size")
            .reset_index()
        )

    def main_table(self):
        rows = []
        for split in ["val", "test"]:
            for study, variant, label in CORE:
                score = self.score(study, variant, split)
                frame = self.zones[study]
                frame = frame[
                    frame.variant.eq(variant)
                    & frame.split.eq(split)
                    & frame.coverage.eq("all")
                    & frame.seed.eq(224)
                ]
                jobs = [
                    j for j in self.summaries[study]["jobs"] if j["variant"] == variant
                ]
                rows.append(
                    {
                        "model": label,
                        "split": split,
                        "matched_MAE": score.paired_mae_mean,
                        "seed_SD": score.paired_mae_std,
                        "all_target_MAE": score.mae_mean,
                        "all_target_RMSE": score.rmse_mean,
                        "matched_hours": int(frame.paired_hours.sum()),
                        "all_target_hours": int(frame.hours.sum()),
                        "job_seconds": np.mean([j["seconds"] for j in jobs]),
                    }
                )
            reference = self.score("airport", "mlp_recent_none", split)
            frame = self.zones["airport"]
            frame = frame[
                frame.variant.eq("mlp_recent_none")
                & frame.split.eq(split)
                & frame.seed.eq(224)
                & frame.coverage.eq("all")
            ]
            rows.append(
                {
                    "model": "Weekly persistence",
                    "split": split,
                    "matched_MAE": reference.persistence_mae_mean,
                    "seed_SD": 0.0,
                    "all_target_MAE": np.nan,
                    "all_target_RMSE": np.nan,
                    "matched_hours": int(frame.paired_hours.sum()),
                    "all_target_hours": int(frame.hours.sum()),
                    "job_seconds": np.nan,
                }
            )
        table = pd.DataFrame(rows)
        if table.groupby("split").matched_hours.nunique().max() != 1:
            raise ValueError("Headline populations differ")
        return table

    def main_figure(self):
        frame = self.main_table()
        order = ["Weekly persistence"] + [item[2] for item in CORE]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
        for ax, split in zip(axes, ["val", "test"], strict=True):
            f = frame[frame.split.eq(split)].set_index("model").loc[order]
            ax.barh(
                f.index,
                f.matched_MAE,
                xerr=f.seed_SD,
                capsize=3,
                color=["#a3adb8"] + ["#347a99"] * 4,
            )
            ax.invert_yaxis()
            ax.set(
                title={"val": "Validation", "test": "Previously explored test"}[split],
                xlabel="Matched MAE (EUR/MWh); bars ± seed SD",
            )
        fig.tight_layout()
        return fig

    def country_figure(self, countries):
        data = {}
        for study, variant, label in CORE:
            data[label] = [
                100
                * (
                    1
                    - self.score(study, variant, "test", c).paired_mae_mean
                    / self.score(study, variant, "test", c).persistence_mae_mean
                )
                for c in countries
            ]
        table = pd.DataFrame(data, index=countries)
        return heatmap(
            table, "Test MAE improvement over weekly persistence (%)", center=True
        )

    def family_figure(self, variants, title, field="paired_mae"):
        fig, axes = plt.subplots(
            1, 2, figsize=(13, max(3.5, len(variants) * 0.36)), sharey=True
        )
        for ax, split in zip(axes, ["val", "test"], strict=True):
            rows = [self.score("controlled", name, split) for name in variants]
            ax.errorbar(
                [r[field + "_mean"] for r in rows],
                range(len(rows)),
                xerr=[r[field + "_std"] for r in rows],
                fmt="o",
                capsize=3,
            )
            labels = {
                "mlp_rich_none": "MLP: rich inputs",
                "pooled_xgb_rich_none": "Pooled XGBoost: rich inputs",
                "gat_rich_real": "GAT: real graph",
                "gat_rich_self": "GAT: self-loops only",
                "gru_mlp_rich_none": "GRU + MLP",
                "gru_gat_rich_real": "GRU + GAT",
            }
            ax.set_yticks(
                range(len(rows)),
                [
                    labels.get(v, "GAT: shuffled graph " + v.rsplit("rewire", 1)[-1])
                    for v in variants
                ],
            )
            ax.set(title=split, xlabel="Matched MAE ± training-seed SD (EUR/MWh)")
            if ax is axes[0]:
                ax.invert_yaxis()
        fig.suptitle(title)
        fig.tight_layout()
        return fig

    def weather_figure(self):
        variants = [
            f"{m}_{g}_{graph}"
            for m, graph in [("pooled_xgb", "none"), ("mlp", "none"), ("gat", "real")]
            for g in ["recent", "airport_latest", "airport_history"]
        ]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for ax, split in zip(axes, ["val", "test"], strict=True):
            for m, graph, label in [
                ("pooled_xgb", "none", "Pooled XGBoost"),
                ("mlp", "none", "MLP"),
                ("gat", "real", "GAT"),
            ]:
                rows = [
                    self.score("airport", v, split)
                    for v in variants
                    if v.startswith(m + "_")
                ]
                ax.errorbar(
                    range(3),
                    [r.mae_mean for r in rows],
                    yerr=[r.mae_std for r in rows],
                    marker="o",
                    capsize=3,
                    label=label,
                )
            ax.set(
                xticks=range(3),
                xticklabels=["Price-only", "+ Latest", "+ History"],
                ylabel="All-target MAE ± seed SD (EUR/MWh)",
                title=split,
            )
            ax.legend()
        fig.tight_layout()
        return fig

    def local_paths(self):
        return {
            study: self.root
            / "runs"
            / ("controlled_study" if study == "controlled" else "airport_temperature")
            / summary["fingerprint"]
            / "full"
            for study, summary in self.summaries.items()
        }

    def compare_signatures(self):
        signatures = []
        for study, variant, _ in CORE:
            path = self.local_paths()[study] / (variant + "_seed224") / "result.json"
            if not path.exists():
                return "Local key signatures unavailable; tracked summaries are verified separately. Cross-study detail comparison requires local artifacts."
            signatures.append(read_json(path)["target_signatures"])
        if any(item != signatures[0] for item in signatures):
            raise ValueError("Cross-study canonical target signatures differ")
        return "Exact full validation/test target-key, value and mask signatures match across all four learned headline models."

    def drilldown(self, zone, start, end, seed=224, models=None):
        a, b = bounds(start, end)
        if models is not None and (
            not models or set(models) - {item[2] for item in CORE}
        ):
            raise ValueError("Choose one or more recognised headline model labels")
        frames = []
        for study, variant, label in CORE:
            path = (
                self.local_paths()[study]
                / (variant + f"_seed{seed}")
                / "predictions_val.parquet"
            )
            if not path.exists():
                return None, "Local predictions unavailable: " + str(path)
            dataset = ds.dataset(path)
            filt = (
                (ds.field("zone") == zone)
                & (ds.field("ts_utc") >= a.to_pydatetime())
                & (ds.field("ts_utc") < b.to_pydatetime())
            )
            frame = (
                dataset.to_table(
                    columns=[
                        "zone",
                        "ts_utc",
                        "price",
                        "target_valid",
                        "prediction",
                        "price_lag168",
                    ],
                    filter=filt,
                )
                .to_pandas()
                .sort_values("ts_utc")
                .reset_index(drop=True)
            )
            if frame.empty:
                return None, "No validation observations in selected window."
            if frames:
                pd.testing.assert_frame_equal(
                    frames[0][["zone", "ts_utc", "price", "target_valid"]],
                    frame[["zone", "ts_utc", "price", "target_valid"]],
                )
            frame["model"] = label
            frames.append(frame)
        return pd.concat(
            frames, ignore_index=True
        ), "Aligned validation-only window; no missing targets filled."

    def weather_window(self, zone, start, end):
        a, b = bounds(start, end)
        provenance = self.local_paths()["airport"] / "provenance.json"
        if not provenance.exists():
            return None
        cache = Path(read_json(provenance)["cache"])
        if not cache.exists():
            return None
        metadata = read_json(cache / "metadata.json")
        # Origins table is small; feature array is memory-mapped, not fully loaded.
        origins = pd.read_parquet(cache / "origins.parquet")
        days = pd.date_range(
            a.tz_convert("Europe/Brussels").normalize(),
            b.tz_convert("Europe/Brussels").normalize(),
            freq="D",
            inclusive="left",
        )
        origins = origins[
            origins.forecast_origin_utc.isin(forecast_origin(days))
        ].copy()
        parent = self.root / "staging/vibing_suite"
        # The canonical hour axis comes from the airport cache's pinned parent.
        identity = read_json(cache / "provenance.json")
        candidates = [
            p
            for p in parent.glob("*/complete.json")
            if read_json(p) == identity["parent_complete"]
        ]
        if not candidates:
            return None
        times = np.load(candidates[0].parent / "times.npy", mmap_mode="r")
        x = np.load(cache / "x.npy", mmap_mode="r")
        j = metadata["zones"].index(zone)
        samples = []
        for origin in origins.forecast_origin_utc:
            delivery = origin.tz_convert("Europe/Brussels").normalize() + pd.DateOffset(
                days=1
            )
            index = int(np.searchsorted(times, delivery.tz_convert("UTC").value))
            values = x[index, j, -4:]
            samples.append(
                {
                    "delivery_date": delivery.date(),
                    "origin_utc": origin,
                    "temperature_C": values[0],
                    "age_h": values[1],
                    "mean24_C": values[2],
                    "hours_present": values[3],
                }
            )
        return pd.DataFrame(samples)


def bounds(start, end):
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    if a.tzinfo is None or b.tzinfo is None or b <= a or b - a > pd.Timedelta(days=31):
        raise ValueError(
            "Choose a positive window of at most 31 days with timezone-aware bounds"
        )
    if a < pd.Timestamp("2024-03-31 22:00Z") or b > pd.Timestamp("2025-03-31 22:00Z"):
        raise ValueError("Detailed outcomes are restricted to validation")
    return a.tz_convert("UTC"), b.tz_convert("UTC")


def heatmap(table, title, center=False):
    fig, ax = plt.subplots(
        figsize=(min(14, max(7, len(table.columns) * 1.3)), max(3, len(table) * 0.34))
    )
    extent = np.nanmax(np.abs(table.to_numpy())) if center else None
    im = ax.imshow(
        table.to_numpy(float),
        aspect="auto",
        cmap="RdBu" if center else "viridis",
        vmin=-extent if center else None,
        vmax=extent,
    )
    ax.set(
        xticks=range(
            0, len(table.columns), max(1, int(np.ceil(len(table.columns) / 12)))
        ),
        xticklabels=table.columns[:: max(1, int(np.ceil(len(table.columns) / 12)))],
        yticks=range(len(table)),
        yticklabels=table.index,
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def plot_window(frame, zone):
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    reference = frame[frame.model.eq(frame.model.iloc[0])]
    axes[0].plot(
        reference.ts_utc, reference.price, color="black", linewidth=2, label="Observed"
    )
    axes[0].plot(
        reference.ts_utc,
        reference.price_lag168,
        color="grey",
        alpha=0.7,
        label="Weekly persistence",
    )
    for label, group in frame.groupby("model", sort=False):
        axes[0].plot(group.ts_utc, group.prediction, alpha=0.75, label=label)
        axes[1].plot(
            group.ts_utc,
            (group.prediction - group.price).where(group.target_valid),
            alpha=0.7,
            label=label,
        )
    axes[0].legend(ncol=3)
    axes[0].set(
        title=f"{zone}: fixed validation window, seed selected for illustration",
        ylabel="EUR/MWh",
    )
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set(ylabel="Prediction minus observed (EUR/MWh)", xlabel="UTC")
    fig.tight_layout()
    return fig
