"""Run fixed, separate country models and report paired persistence comparisons.

python -m epgnn.boosting_batch --config configs/experiments/c1_xgb_countries_weekly_conditional.json
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd

from epgnn import boosting
from epgnn.dataset_build.storage import ROOT, canonical, digest, json_write


def comparison_rows(records):
    """Use exactly the same target hours for each model/persistence pair."""
    rows = []
    for record in records:
        for split, block in record["metrics"].items():
            common = block["weekly_persistence_conditional_common_hours"]
            model, weekly = common["model"], common["persistence"]
            if model["n_hours"] != weekly["n_hours"]:
                raise ValueError("Paired model/persistence sample counts differ")
            rows.append({"zone": record["zone"], "split": split, "run": record["name"],
                "selected_rounds": record["selected_rounds"], "fit_rows": record["fit_rows"],
                "valid_targets": block["valid_targets"], "paired_hours": model["n_hours"],
                "valid_targets_without_persistence": common["valid_targets_without_persistence"],
                "xgb_mae": model["mae"], "persistence_mae": weekly["mae"],
                "xgb_rmse": model["rmse"], "persistence_rmse": weekly["rmse"],
                "mae_improvement_percent": 100 * (1 - common["relative_mae"]) if common["relative_mae"] is not None else None,
                "full_target_xgb_mae": block["model"]["mae"],
                "admission": record["feature_policy"]})
    return pd.DataFrame(rows)


def aggregate(rows):
    """Hour-weighted errors across country-hour pairs; RMSE pools squared errors."""
    summaries = {}
    for split, group in rows.groupby("split", sort=True):
        g = group.loc[group.paired_hours > 0]
        if g.empty:
            raise ValueError(f"No paired evaluation hours in {split}")
        weight = g.paired_hours.to_numpy(dtype=float)
        model = float(np.average(g.xgb_mae, weights=weight))
        weekly = float(np.average(g.persistence_mae, weights=weight))
        summaries[split] = {"countries": len(g), "paired_country_hours": int(weight.sum()),
            "xgb_mae": model, "persistence_mae": weekly,
            "xgb_rmse": float(np.sqrt(np.average(np.square(g.xgb_rmse), weights=weight))),
            "persistence_rmse": float(np.sqrt(np.average(np.square(g.persistence_rmse), weights=weight))),
            "mae_improvement_percent": 100 * (1 - model / weekly) if weekly else None,
            "countries_xgb_better": int((g.xgb_mae < g.persistence_mae).sum()),
            "countries_xgb_worse": int((g.xgb_mae > g.persistence_mae).sum())}
    return summaries


def run(config_path):
    batch = json.loads(Path(config_path).read_text())
    if not re.fullmatch(r"[A-Za-z0-9_-]+", batch["name"]):
        raise ValueError("Invalid batch name")
    zones = batch["zones"]
    if not zones or len(set(zones)) != len(zones) or any(not re.fullmatch(r"[A-Z0-9_]+", z) for z in zones):
        raise ValueError("Select unique canonical zones")
    base_path = Path(batch["base_config"])
    if not base_path.is_absolute():
        base_path = ROOT / base_path
    base = json.loads(base_path.read_text())
    if "price_lag168" not in base.get("conditional_features", []):
        raise ValueError("Country comparisons require the conditional weekly-price input")
    boosting.check_features(base["features"], boosting.experiment_registry(
        json.loads((boosting._path(base["dataset_directory"]) / "feature_registry.json").read_text())),
        base["conditional_features"])
    prefix = base["name"].split(f"_{base['zone'].lower()}_", 1)
    if len(prefix) != 2:
        raise ValueError("Base run name must include its zone token")
    configs = []
    # Materialize reviewable configs; refuse to overwrite an existing experiment.
    for zone in zones:
        cfg = deepcopy(base)
        cfg.update(zone=zone, name=f"{prefix[0]}_{zone.lower()}_{prefix[1]}")
        p = base_path.parent / f"{cfg['name']}.json"
        if p.exists():
            if json.loads(p.read_text()) != cfg:
                raise ValueError(f"Existing experiment differs: {p}")
        else:
            json_write(p, cfg)
        configs.append(p)
    records = []
    for zone, p in zip(zones, configs):
        print(f"Country model {zone}", flush=True)
        records.append(boosting.run(p))
    # One configuration per country; no country or parameter selection by test score.
    rows = comparison_rows(records)
    for record in records:
        if (record["dataset_fingerprint"] != base["dataset_fingerprint"] or
                record["config"]["params"] != base["params"] or
                record["config"]["features"] != base["features"] or
                record["config"]["fit_splits"] != base["fit_splits"] or
                record["config"].get("target_transform", "identity") != base.get("target_transform", "identity") or
                record["config"].get("target_scale_eur_mwh", 50.0) != base.get("target_scale_eur_mwh", 50.0)):
            raise ValueError("Country experiment definitions do not match")
    summary = {"name": batch["name"], "config": batch, "dataset_fingerprint": base["dataset_fingerprint"],
        "feature_policy": "conditional", "features": base["features"],
        "target_transform": base.get("target_transform", "identity"),
        "target_scale_eur_mwh": base.get("target_scale_eur_mwh", 50.0),
        "early_stopping_metric_units": records[0]["early_stopping_metric_units"],
        "parameters": base["params"], "training": "pretrain + train; 2016-01-01 through 2024-03-31 where structurally available",
        "validation": "2024-04-01 through 2025-03-31; early stopping only",
        "test": "2025-04-01 through 2026-09-30; one evaluation per new country model",
        "comparison": "Exact finite common target hours for each country's model and weekly persistence",
        "limitation": "Historical publication vintages remain unverified; conditional inputs: " +
            ", ".join(base["conditional_features"]),
        "aggregate": aggregate(rows), "countries": rows.to_dict("records"),
        "batch_runner_sha256": digest(Path(__file__))["sha256"],
        "experiments": [{"zone": r["zone"], "name": r["name"], "fingerprint": r["experiment_fingerprint"],
                         "report_sha256": digest(ROOT / "results" / f"{r['name']}.json")["sha256"]} for r in records]}
    summary["batch_fingerprint"] = hashlib.sha256(canonical(summary)).hexdigest()
    result_dir = ROOT / "results"
    json_write(result_dir / f"{batch['name']}.json", summary)
    fd, temp = tempfile.mkstemp(prefix=".country-comparison-", suffix=".csv", dir=result_dir)
    os.close(fd)
    try:
        rows.to_csv(temp, index=False)
        os.replace(temp, result_dir / f"{batch['name']}.csv")
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    try:
        summary, rows = run(args.config)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.exit(2, f"boosting_batch: {exc}\n")
    print(rows.loc[rows.split == "test", ["zone", "xgb_mae", "persistence_mae", "mae_improvement_percent", "selected_rounds", "paired_hours"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(json.dumps(summary["aggregate"], indent=2))
    print(f"Reports: results/{summary['name']}.json and .csv")


if __name__ == "__main__":
    main()
