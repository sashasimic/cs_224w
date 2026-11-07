#!/usr/bin/env python3
"""Run a pinned partner checkout unchanged, then split its predictions by zone.

Run this script in a fresh Python process: it deliberately imports epgnn from
--partner-root, not the audited experiment checkout containing this script.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def aggregate_countries(zone_rows):
    """Pool errors over a country's bidding-zone hours, not averaged prices.

    DE_LU and IE_SEM remain explicitly combined markets. Do not invent national
    prices or split observations between Germany/Luxembourg or Ireland/NI.
    """
    import math
    groups = {}
    for row in zone_rows:
        zone = row["zone"]
        country = zone.split("_", 1)[0] if zone.startswith(("DK_", "NO_", "SE_", "IT_")) else zone
        groups.setdefault((country, row["split"]), []).append(row)
    out = []
    for (country, split), rows in sorted(groups.items()):
        active = [r for r in rows if r["n_hours"]]
        count = sum(r["n_hours"] for r in active)
        mae = sum(r["mae"] * r["n_hours"] for r in active) / count if count else None
        persistence = sum(r["persistence_mae"] * r["n_hours"] for r in active) / count if count else None
        out.append({"country": country, "split": split, "zones": sorted(r["zone"] for r in rows),
            "n_zone_hours": count, "mae": mae,
            "rmse": math.sqrt(sum(r["rmse"] ** 2 * r["n_hours"] for r in active) / count) if count else None,
            "persistence_mae": persistence, "relative_mae": mae / persistence if persistence else None,
            "status": "scored" if count else "no_valid_targets"})
    return out


def file_hash(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partner-root", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--reuse-predictions", action="store_true", help="verify the saved reproduction and reaggregate without fitting")
    args = parser.parse_args()
    root, output = args.partner_root.resolve(), args.output_root.resolve()
    if args.threads <= 0:
        parser.error("--threads must be positive")
    for name in ["OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ[name] = str(args.threads)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    if commit != args.expected_commit:
        raise ValueError(f"Partner commit mismatch: {commit}")
    # Verify that all tracked partner source/config/result files are unchanged.
    subprocess.run(["git", "diff", "--exit-code", "HEAD"], cwd=root, check=True)
    sys.path.insert(0, str(root))
    import base64
    import numpy as np
    import pandas as pd
    import yaml
    import epgnn
    from epgnn import metrics
    from epgnn.data import samples as S
    from epgnn.models import persistence
    from epgnn.run import main as partner_main
    if not Path(epgnn.__file__).resolve().is_relative_to(root):
        raise ValueError("Wrong epgnn checkout imported")
    pin = json.loads((root / "configs/datasets.json").read_text())
    sources = {}
    for name, spec in pin["datasets"].items():
        for filename, meta in spec["files"].items():
            p = Path(filename)
            path = root / "datasets" / f"{p.stem}_{spec['version']}{p.suffix}"
            md5 = base64.b64encode(bytes.fromhex(file_hash(path, "md5"))).decode()
            if md5 != meta["md5"] or path.stat().st_size != meta["size"]:
                raise ValueError(f"Pinned input changed: {path}")
            sources[str(path.relative_to(root))] = {"sha256": file_hash(path), "md5": md5,
                "size": path.stat().st_size, "resolved_path": str(path.resolve())}
    published = json.loads((root / "results/c1_xgb_zone.json").read_text())
    cfg = yaml.safe_load((root / "configs/experiments/c1_xgb_zone.yaml").read_text())
    original_cfg = dict(cfg)
    run_name = "partner_c1_xgb_zone_reproduction"
    artifact_dir = root / "runs" / run_name
    prior = None
    if artifact_dir.exists() or (root / "results" / f"{run_name}.json").exists():
        if not args.reuse_predictions:
            raise ValueError("A reproduction already exists; use --reuse-predictions instead of refitting")
        for rel, sha in json.loads((artifact_dir / "reaggregation_checksums.json").read_text()).items():
            if file_hash(artifact_dir / rel) != sha:
                raise ValueError(f"Saved reproduction artifact changed: {rel}")
        prior = json.loads((artifact_dir / "country_reaggregation.json").read_text())
        if prior["partner_commit"] != commit or prior["input_files"] != sources:
            raise ValueError("Saved reproduction source identities differ")
        if any(file_hash(root / p) != sha for p, sha in prior["source_hashes"].items()):
            raise ValueError("Partner source files changed since the saved reproduction")
    elif args.reuse_predictions:
        raise ValueError("No saved reproduction to reuse")
    cfg["name"] = run_name  # output identifier only; no modelling parameter changes
    config_file = root / "tmp" / f"{run_name}.yaml"
    config_file.parent.mkdir(exist_ok=True)
    config_file.write_text(yaml.safe_dump(cfg, sort_keys=False))
    started = time.monotonic()
    if prior:
        if prior["config"] != cfg:
            raise ValueError("Saved reproduction config differs")
        print("Verified saved predictions; reaggregating without fitting or predicting again", flush=True)
        record = prior["pipeline_record"]
    else:
        print(f"Run exact partner pipeline at {commit}; OpenMP thread limit={args.threads}", flush=True)
        record = partner_main(str(config_file))
    rows, reconciliation = [], {}
    for split in cfg["eval_splits"]:
        sample = S.build(split, cfg.get("zones"))
        pred = np.load(artifact_dir / f"pred_{split}.npy")
        weekly = persistence.predict(sample)
        if pred.shape != sample.y.shape or weekly.shape != sample.y.shape:
            raise ValueError("Prediction/sample shape mismatch")
        mask = sample.mask
        if not np.isfinite(sample.y[mask]).all() or not np.isfinite(pred[mask]).all():
            raise ValueError("Nonfinite values inside the partner's target mask")
        positions = np.nonzero(sample.hours >= 0)
        frame = pd.DataFrame({"zone": sample.zone[positions[0]], "delivery_date": sample.day[positions[0]],
            "ts_utc": pd.to_datetime(sample.hours[positions], utc=True, unit="ns"),
            "split": split, "price": sample.y[positions], "target_valid": mask[positions],
            "prediction": pred[positions], "weekly_persistence": weekly[positions]})
        if frame.duplicated(["zone", "ts_utc"]).any():
            raise ValueError("Duplicate zone/hour predictions")
        frame.to_parquet(artifact_dir / f"predictions_{split}_keyed.parquet", index=False)
        # Independent per-zone reaggregation, using precisely the partner mask.
        for zone in np.unique(sample.zone):
            selected = sample.zone == zone
            m = mask[selected]
            n = int(m.sum())
            if not n:
                rows.append({"zone": str(zone), "split": split, "n_hours": 0,
                    "mae": None, "rmse": None, "persistence_mae": None, "relative_mae": None,
                    "status": "no_valid_targets", "trees_used": record["info"]["trees_used"].get(str(zone))})
                continue
            y, p, q = sample.y[selected][m], pred[selected][m], weekly[selected][m]
            if not np.isfinite(q).all():
                raise ValueError("Partner persistence is missing on valid target hours")
            # Float64 accumulators reconcile country-weighted errors independently;
            # the original pipeline uses float32 reductions, so allow tiny rounding.
            error = p.astype("float64") - y.astype("float64")
            weekly_error = q.astype("float64") - y.astype("float64")
            mae = float(np.abs(error).mean())
            persistence_mae = float(np.abs(weekly_error).mean())
            rows.append({"zone": str(zone), "split": split, "n_hours": n,
                "mae": mae, "rmse": float(np.sqrt(np.square(error).mean())),
                "persistence_mae": persistence_mae,
                "relative_mae": mae / persistence_mae if persistence_mae else None,
                "status": "scored", "trees_used": record["info"]["trees_used"].get(str(zone))})
        split_rows = [r for r in rows if r["split"] == split and r["n_hours"]]
        count = sum(r["n_hours"] for r in split_rows)
        weighted_mae = sum(r["mae"] * r["n_hours"] for r in split_rows) / count
        pooled_rmse = float(np.sqrt(sum(r["rmse"] ** 2 * r["n_hours"] for r in split_rows) / count))
        pipeline = record["metrics"][split]["all"]
        np.testing.assert_allclose(weighted_mae, pipeline["mae"], rtol=2e-6, atol=1e-5)
        np.testing.assert_allclose(pooled_rmse, pipeline["rmse"], rtol=2e-6, atol=1e-5)
        if count != pipeline["n_hours"]:
            raise ValueError("Country target counts do not sum to the pipeline aggregate")
        reconciliation[split] = {"country_hours": count, "active_zones": len(split_rows),
            "weighted_country_mae": weighted_mae, "pooled_country_rmse": pooled_rmse,
            "pipeline_mae": pipeline["mae"], "published_mae": published["metrics"][split]["all"]["mae"],
            "reproduction_minus_published_mae": pipeline["mae"] - published["metrics"][split]["all"]["mae"]}
        print(f"Reaggregated {split}: {len(split_rows)} active zones; MAE={weighted_mae:.6f}", flush=True)
    source_files = ["epgnn/models/xgb.py", "epgnn/data/features.py", "epgnn/data/samples.py", "epgnn/data/scaling.py",
        "epgnn/data/tables.py", "epgnn/metrics.py", "epgnn/run.py", "configs/splits.json", "configs/datasets.json"]
    summary = {"partner_commit": commit, "published_run_git": published["git"], "config": cfg,
        "original_config": original_cfg, "only_config_override": "name (output identifier)",
        "source_checkout": str(root), "source_hashes": {p: file_hash(root / p) for p in source_files},
        "input_files": sources, "versions": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "pyarrow", "xgboost", "scikit-learn"]},
        "python": sys.version.split()[0], "openmp_thread_limit": args.threads,
        "elapsed_seconds": prior["elapsed_seconds"] if prior else time.monotonic() - started,
        "ran_at": prior["ran_at"] if prior else datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reaggregated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reused_predictions": bool(prior), "reaggregation_script_sha256": file_hash(__file__),
        "reconciliation": reconciliation, "country_scores": rows, "country_aggregates": aggregate_countries(rows),
        "aggregation_definition": "Per-zone scores plus country scores pooling bidding-zone-hour errors; not errors of a constructed national average price",
        "pipeline_record": record,
        "protocol": "Unmodified partner source, v1.0 source tables, original features/targets/early-stopping/scoring; country scores from saved predictions",
        "availability_status": "Reproduction of partner protocol, not a strict-noon availability certification",
        "artifact_directory": str(artifact_dir)}
    # Keep a complete archival report with the saved prediction arrays.
    write_json(artifact_dir / "country_reaggregation.json", summary)
    frame = pd.DataFrame(rows)
    frame.to_csv(artifact_dir / "country_scores.csv", index=False)
    write_json(artifact_dir / "reaggregation_checksums.json", {p.name: file_hash(p) for p in artifact_dir.iterdir()
        if p.is_file() and p.name != "reaggregation_checksums.json"})
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "partner_c1_xgb_zone_by_country.json", summary)
    frame.to_csv(output / "partner_c1_xgb_zone_by_country.csv", index=False)
    frame.to_csv(output / "partner_c1_xgb_zone_by_zone.csv", index=False)
    country_frame = pd.DataFrame(summary["country_aggregates"])
    country_frame["zones"] = country_frame.zones.map(lambda x: ",".join(x))
    country_frame.to_csv(output / "partner_c1_xgb_zone_country_aggregates.csv", index=False)
    write_json(output / "partner_c1_xgb_zone_country_aggregates.json", {
        "partner_commit": commit, "aggregation_definition": summary["aggregation_definition"],
        "country_scores": summary["country_aggregates"]})
    subprocess.run(["git", "diff", "--exit-code", "HEAD"], cwd=root, check=True)
    print(frame.loc[frame.split == "test", ["zone", "n_hours", "mae", "persistence_mae", "relative_mae"]].to_string(index=False), flush=True)
    print(f"Reports: {output}/partner_c1_xgb_zone_by_country.json and .csv", flush=True)


if __name__ == "__main__":
    main()
