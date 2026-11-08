"""Verify frozen GRU predictions and compare scores without training or GPU use."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from epgnn.dataset_build.storage import digest
from epgnn.experiment_suite.artifacts import write_json

ROOT = Path(__file__).resolve().parents[2]
PREFIX = ROOT / "results/partner_gru_full"


def verify(record):
    directory = ROOT / "runs/partner_gru" / record["fingerprint"] / "full"
    hashes = json.loads((directory / "complete.json").read_text())
    for name, checksum in hashes.items():
        if digest(directory / name)["sha256"] != checksum:
            raise ValueError(f"Completed artifact changed: {name}")
    if json.loads((directory / "result.json").read_text()) != record:
        raise ValueError("Published result differs from completed run")
    pd.testing.assert_frame_equal(
        pd.DataFrame(record["rows"]),
        pd.read_csv(str(PREFIX) + "_by_zone.csv"),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    reference = json.loads(
        (ROOT / "results/airport_temperature_full_verification.json").read_text()
    )
    if not reference["passed"] or not record["verified"]:
        raise ValueError("Verified run and canonical reference required")
    provenance = json.loads((directory / "provenance.json").read_text())
    if provenance["config"]["rnn_backend"] != "native_gpu_verified_against_cpu":
        raise ValueError("Corrected native backend required")
    errors, counts = [], {}
    for split in ["val", "test"]:
        frame = pd.read_parquet(directory / f"predictions_{split}.parquet")
        signature = hashlib.sha256(
            pd.util.hash_pandas_object(
                frame[["zone", "ts_utc", "price", "target_valid"]], index=False
            )
            .to_numpy()
            .tobytes()
        ).hexdigest()
        if (
            signature != reference["target_signatures"][split]
            or signature != record["target_signatures"][split]
        ):
            raise ValueError("Canonical target keys/values/masks changed")
        if frame.duplicated(["zone", "ts_utc"]).any():
            raise ValueError("Duplicate prediction keys")
        if not np.isfinite(frame.prediction).all() or frame.zone.nunique() != 49:
            raise ValueError("Incomplete predictions or changed cohort")
        np.testing.assert_array_equal(frame.target_valid, np.isfinite(frame.price))
        specifications = [
            (row, frame[frame.zone.eq(row["zone"])])
            for row in record["rows"]
            if row["split"] == split
        ]
        overall = next(
            r
            for r in record["aggregate"]
            if r["scope"] == "overall" and r["split"] == split
        )
        specifications.append((overall, frame))
        for expected, selected in specifications:
            y = selected.price.to_numpy(float)
            p = selected.prediction.to_numpy(float)
            baseline = selected.price_lag168.to_numpy(float)
            valid = np.isfinite(y)
            paired = valid & np.isfinite(baseline)
            mae = np.abs(p[paired] - y[paired]).mean()
            persistence = np.abs(baseline[paired] - y[paired]).mean()
            metrics = {
                "mae": np.abs(p[valid] - y[valid]).mean(),
                "rmse": np.sqrt(np.square(p[valid] - y[valid]).mean()),
                "paired_mae": mae,
                "persistence_mae": persistence,
                "relative_mae": mae / persistence,
            }
            if (
                expected["hours"] != valid.sum()
                or expected["paired_hours"] != paired.sum()
            ):
                raise ValueError("Target counts changed")
            errors.extend(
                abs(value - expected[name]) for name, value in metrics.items()
            )
        check = record["cpu_reference_checks"][split]
        if (
            check["max_absolute_eur_mwh"] > check["max_allowed_eur_mwh"]
            or check["mean_absolute_eur_mwh"] > check["mean_allowed_eur_mwh"]
        ):
            raise ValueError("CPU reference check failed")
        counts[split] = {
            "valid_targets": overall["hours"],
            "paired_targets": overall["paired_hours"],
            "prediction_rows": len(frame),
            "scored_zones": frame.zone.nunique(),
        }
    if max(errors) > 1e-9:
        raise ValueError("Saved prediction metrics differ from summary")
    verification = {
        "passed": True,
        "fingerprint": record["fingerprint"],
        "maximum_metric_error": float(max(errors)),
        "counts": counts,
        "target_signatures": record["target_signatures"],
        "cpu_reference_checks": record["cpu_reference_checks"],
        "summary_sha256": digest(PREFIX.with_suffix(".json"))["sha256"],
        "completed_artifact_count": len(hashes),
        "checks": [
            "completed artifact hashes",
            "canonical unique target keys/values/masks",
            "49-zone cohort",
            "finite predictions",
            "recomputed zone and overall metrics",
            "recorded CPU float64 reference gates",
        ],
    }
    write_json(str(PREFIX) + "_verification.json", verification)
    return verification


def main():
    record = json.loads(PREFIX.with_suffix(".json").read_text())
    verification = verify(record)
    sources = [
        PREFIX.with_suffix(".json"),
        ROOT / "results/controlled_gru_backend_audit.json",
    ]
    rows = []
    for metric in record["aggregate"]:
        if metric["scope"] == "overall":
            rows.append(
                {
                    "variant": "partner_residual_gru_conditional",
                    "split": metric["split"],
                    "seeds": 1,
                    "mae": metric["mae"],
                    "rmse": metric["rmse"],
                    "paired_mae": metric["paired_mae"],
                }
            )
    comparisons = {
        "controlled_study": [
            "pooled_xgb_rich_none",
            "mlp_rich_none",
            "gat_rich_real",
            "gat_weather_real",
        ],
        "airport_temperature": ["gat_airport_history_real"],
    }
    for name, variants in comparisons.items():
        path = ROOT / f"results/{name}_full_scores.csv"
        sources.append(path)
        frame = pd.read_csv(path)
        for _, metric in frame[
            frame.variant.isin(variants)
            & frame.scope.eq("overall")
            & frame.coverage.eq("all")
        ].iterrows():
            rows.append(
                {
                    "variant": metric.variant,
                    "split": metric.split,
                    "seeds": int(metric.mae_count),
                    "mae": metric.mae_mean,
                    "rmse": metric.rmse_mean,
                    "paired_mae": metric.paired_mae_mean,
                }
            )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(str(PREFIX) + "_comparison.csv", index=False)
    lines = [
        "# Residual GRU compared with previous experiments",
        "",
        "All rows use the same canonical target cohort and chronological splits. Earlier rows are five-seed means of scores, not ensembles; the new GRU is seed 0 only. Features, architecture, training objective and checkpoint-selection objective differ, so this is context rather than a controlled architecture or weather ablation.",
        "",
        "| Model | Seeds | Validation MAE | Test MAE | Test paired MAE | Test RMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant, group in comparison.groupby("variant", sort=False):
        val = group[group.split.eq("val")].iloc[0]
        test = group[group.split.eq("test")].iloc[0]
        lines.append(
            f"| {variant} | {int(test.seeds)} | {val.mae:.4f} | {test.mae:.4f} | {test.paired_mae:.4f} | {test.rmse:.4f} |"
        )
    overall = {r["split"]: r for r in record["aggregate"] if r["scope"] == "overall"}
    test = overall["test"]
    best = min(record["curves"], key=lambda r: r["val_residual_plus_spread"])
    lines += [
        "",
        f"The GRU reduces matched test MAE versus weekly persistence by {100 * (1 - test['relative_mae']):.1f}% ({test['paired_mae']:.4f} versus {test['persistence_mae']:.4f} EUR/MWh). Its test MAE is close to the existing pooled XGBoost and static GAT results, while its higher RMSE shows larger squared errors. Its lower validation MAE does not establish a test improvement or a general architecture advantage.",
        "",
        f"Training stopped after {len(record['curves'])} epochs; validation residual-plus-spread loss selected epoch {best['epoch']}. End-to-end fit/evaluation took {record['seconds'] / 60:.1f} minutes. Test physical price-spread MAE is {record['spread_metrics']['test']['price_spread_mae_eur_mwh']:.4f} EUR/MWh across {record['spread_metrics']['test']['pairs']} training graph pairs. This physical metric differs from the transformed residual-contrast training loss.",
        "",
        "The partner's published 20.40 MAE uses a different 43-zone population and pipeline; it cannot be directly ranked here. Our earlier controlled GRU+MLP score (24.50 paired test MAE) used a different recipe and fused RNN backend. A separate audit of all ten earlier temporal checkpoints found CPU disagreement on shuffled training-shaped inputs, despite passing consecutive inference checks. Those affected scores cannot support temporal-architecture conclusions until corrected training. See results/controlled_gru_backend_audit.json and the consolidated experiment record. Static MLP/GAT and XGBoost do not use this RNN path.",
        "",
        "ERA5 is retrospective reanalysis, not archived weather forecasts. Load/flow vintages and anomaly climatology remain unresolved. This is a conditional experiment; it does not isolate a temperature benefit or certify deployment-time availability. Test results were previously explored, so this is not a fresh blind test.",
        "",
        f"Verification passed: {verification['counts']['test']['valid_targets']:,} test labels, {verification['counts']['test']['paired_targets']:,} paired labels, unchanged canonical target signatures, finite predictions, artifact hashes and recomputed zone/overall metrics. CPU float64 comparisons passed on actual validation/test batches; the full-run maximum difference was {max(c['max_absolute_eur_mwh'] for c in record['cpu_reference_checks'].values()):.8f} EUR/MWh. Original rejected fused-run artifacts remain archived.",
        "",
        "[Full run and learning curve](partner_gru_results.md) · [Method and backend audit](partner_gru.md). Reproduce this review with `.venv/bin/python scripts/experiments/review_partner_gru.py`. Per-zone scores are in `results/partner_gru_full_by_zone.csv`.",
    ]
    (ROOT / "docs/partner_gru_comparison.md").write_text("\n".join(lines) + "\n")
    write_json(
        str(PREFIX) + "_review_sources.json",
        {str(p.relative_to(ROOT)): digest(p)["sha256"] for p in sources},
    )
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
