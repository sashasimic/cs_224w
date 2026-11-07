"""Auditable per-zone XGBoost with explicit strict/conditional feature selection.

python -m epgnn.boosting --config configs/experiments/c1_xgb_be_strict.json

This runner deliberately reads the new audited artifacts directly. The legacy
day/sample/scoring path and its known blockers are not used or silently fixed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.dataset as ds
import xgboost as xgb

from epgnn.dataset_build.contracts import NODE_KEYS, forecast_origin, unique, utc
from epgnn.dataset_build.storage import ROOT, canonical, digest, json_write

JAO_LAG_FEATURES = {
    "n_presolved_lag48": "n_presolved",
    "min_ram_ratio_lag48": "min_ram_ratio",
    "jao_domain_present_lag48": "jao_domain_present",
}
TEMPERATURE_CURRENT = "temp_pw_current_origin"
TEMPERATURE_PERFECT = "temp_pw_perfect"


def experiment_registry(registry):
    """Derive an experiment-local registry; never modify the dataset registry."""
    rows = [dict(row) for row in registry]
    by_name = {r["feature"]: r for r in rows if r["table"] == "node"}
    for name, source in JAO_LAG_FEATURES.items():
        if name not in by_name and source in by_name:
            rows.append({**by_name[source], "feature": name, "admission": "conditional",
                "transformation": f"exact zone/UTC join of {source} at t-48h; regional members only; NaNs retained",
                "timing_rule": "source domain hour and interval completion precede forecast origin; historical publication time unverified"})
    if "temp_pw" in by_name:
        definitions = {
            TEMPERATURE_CURRENT: ("conditional", "exact zone/time lookup at forecast_origin_utc minus 1h, frozen across the delivery day",
                "previous calendar date at 11:00 Brussels; hourly interval ends at origin; ERA5 publication vintages unverified"),
            TEMPERATURE_PERFECT: ("excluded", "exact zone/delivery-hour lookup of actual reanalysis temperature",
                "future weather intentionally supplied; explicit oracle-only diagnostic, never a realistic forecasting input"),
        }
        for name, (status, transform, timing) in definitions.items():
            if name not in by_name:
                rows.append({**by_name["temp_pw"], "feature": name, "admission": status,
                    "transformation": transform, "timing_rule": timing})
    return rows


def _path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def check_features(features, registry, conditional_features=(), oracle_features=()):
    if not features or len(features) != len(set(features)):
        raise ValueError("Select a nonempty, unique feature list")
    conditional = set(conditional_features)
    if (len(conditional) != len(conditional_features) or not conditional <= set(features)
            or not conditional <= {"price_lag168", *JAO_LAG_FEATURES, TEMPERATURE_CURRENT}):
        raise ValueError("Unsupported conditional feature opt-in")
    oracle = set(oracle_features)
    if (len(oracle) != len(oracle_features) or not oracle <= set(features)
            or not oracle <= {TEMPERATURE_PERFECT} or conditional & oracle):
        raise ValueError("Oracle opt-in supports only the explicit perfect-temperature diagnostic")
    registry = experiment_registry(registry)
    admission = {r["feature"]: r["admission"] for r in registry if r["table"] == "node"}
    bad = {f: admission.get(f, "unregistered") for f in features
           if admission.get(f) != "admitted" and not (f in conditional and admission.get(f) == "conditional")
           and not (f in oracle and f == TEMPERATURE_PERFECT and admission.get(f) == "excluded")}
    if bad:
        raise ValueError(f"Strict experiment rejects non-admitted inputs: {bad}")


def masks(frame, cfg):
    """Targets alone determine label validity; missing features are never dropped."""
    fit_splits = cfg["fit_splits"]
    val_split = cfg["validation_split"]
    eval_splits = cfg["evaluation_splits"]
    if set(fit_splits) - {"pretrain", "train"}:
        raise ValueError("Only pretrain/train may supply fit labels")
    if val_split != "val" or set(fit_splits) & set(eval_splits):
        raise ValueError("Validation/evaluation roles must be disjoint from fit labels")
    if not set(eval_splits) <= {"val", "test"} or len(eval_splits) != len(set(eval_splits)):
        raise ValueError("Only unique val/test evaluation splits are supported")
    valid = frame.target_valid.astype(bool).to_numpy()
    y = frame.price.to_numpy(dtype="float64")
    if not np.array_equal(valid, np.isfinite(y)):
        raise ValueError("Target validity does not match finite unfilled prices")
    fit = frame.split.isin(fit_splits).to_numpy() & valid
    val = frame.split.eq(val_split).to_numpy() & valid
    if not fit.any() or not val.any():
        raise ValueError("Training and validation must have valid targets")
    if frame.loc[fit, "delivery_date"].max() >= frame.loc[val, "delivery_date"].min():
        raise ValueError("Training must precede validation chronologically")
    test = frame.split.eq("test").to_numpy()
    if test.any() and frame.loc[val, "delivery_date"].max() >= frame.loc[test, "delivery_date"].min():
        raise ValueError("Validation must precede test chronologically")
    return fit, val


def transform_target(values, kind="identity", scale=50.0, *, inverse=False):
    """Partner's fixed asinh(price/50) transform; no fitted target statistics."""
    values = np.asarray(values, dtype="float64")
    if kind == "identity":
        return values
    if kind != "asinh" or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Target transform must be identity or asinh with a positive finite scale")
    with np.errstate(over="raise", invalid="raise"):
        return scale * np.sinh(values) if inverse else np.arcsinh(values / scale)


def errors(y, prediction, mask):
    """Index finite valid pairs before arithmetic; never multiply errors by masks."""
    y, prediction, mask = np.asarray(y), np.asarray(prediction), np.asarray(mask, dtype=bool)
    used = mask & np.isfinite(y) & np.isfinite(prediction)
    if not used.any():
        return {"n_hours": 0, "mae": None, "rmse": None}
    residual = prediction[used].astype("float64") - y[used].astype("float64")
    return {"n_hours": int(used.sum()), "mae": float(np.abs(residual).mean()),
            "rmse": float(np.sqrt(np.square(residual).mean()))}


def score(frame, prediction, persistence):
    valid = frame.target_valid.to_numpy(dtype=bool)
    y = frame.price.to_numpy(dtype="float64")
    prediction, persistence = np.asarray(prediction), np.asarray(persistence)
    if len(prediction) != len(frame) or len(persistence) != len(frame):
        raise ValueError("Predictions must align with the exact zone-hour rows")
    if not np.isfinite(prediction).all():
        raise ValueError("The model must predict every evaluation row")
    common = valid & np.isfinite(persistence)
    model_all = errors(y, prediction, valid)
    model_common = errors(y, prediction, common)
    weekly = errors(y, persistence, common)
    relative = model_common["mae"] / weekly["mae"] if weekly["mae"] else None
    return {"rows": len(frame), "valid_targets": int(valid.sum()), "missing_targets": int((~valid).sum()),
        "model": model_all,
        "weekly_persistence_conditional_common_hours": {
            "admission": "conditional; historical publication vintages unverified",
            "mask": "target_valid AND finite(model) AND finite(price_lag168), exact UTC/zone keys",
            "model": model_common, "persistence": weekly, "relative_mae": relative,
            "valid_targets_without_persistence": int((valid & ~np.isfinite(persistence)).sum())}}


def _check_artifacts(directory, complete, names):
    hashes = {}
    for name in names:
        paths = sorted((directory / name).glob("*.parquet")) if (directory / name).is_dir() else [directory / name]
        if not paths:
            raise ValueError(f"Missing artifact: {name}")
        expected = {k for k in complete["artifacts"] if k.startswith(name + "/")} if (directory / name).is_dir() else {name}
        actual = {str(p.relative_to(directory)) for p in paths}
        if expected != actual:
            raise ValueError(f"Artifact file list changed: {name}")
        for p in paths:
            rel = str(p.relative_to(directory))
            sha = digest(p)["sha256"]
            if sha != complete["artifacts"].get(rel):
                raise ValueError(f"Artifact checksum mismatch: {rel}")
            hashes[rel] = sha
    return hashes


def load_strict(directory, cfg, registry):
    check_features(cfg["features"], registry)
    cols = NODE_KEYS + ["delivery_date", "forecast_origin_utc", "split", "price", "target_valid"] + cfg["features"]
    f = ds.dataset(directory / "strict", format="parquet").to_table(
        columns=cols, filter=ds.field("zone") == cfg["zone"]).to_pandas()
    if f.empty:
        raise ValueError(f"Zone absent from strict data: {cfg['zone']}")
    f = f.sort_values(NODE_KEYS).reset_index(drop=True)
    unique(f, NODE_KEYS, "boosting input")
    if not f.forecast_origin_utc.equals(forecast_origin(f.ts_utc)):
        raise ValueError("Forecast origin mismatch")
    # Admitted lag features, when configured, must also satisfy row-level gating.
    eligibility = [c + "_eligible" for c in cfg["features"] if "_lag" in c]
    if eligibility:
        gated = ds.dataset(directory / "strict").to_table(columns=NODE_KEYS + eligibility,
            filter=ds.field("zone") == cfg["zone"]).to_pandas()
        f = f.merge(gated, on=NODE_KEYS, validate="one_to_one")
        for c in eligibility:
            if f.loc[~f[c], c.removesuffix("_eligible")].notna().any():
                raise ValueError(f"Unsupported per-row feature availability: {c}")
    return f


def load_inputs(directory, cfg, registry):
    """Opt-in lag values keep their conditional status; strict artifacts stay intact."""
    conditional = cfg.get("conditional_features", [])
    oracle = cfg.get("oracle_features", [])
    check_features(cfg["features"], registry, conditional, oracle)
    strict_cfg = {**cfg, "features": [f for f in cfg["features"] if f not in conditional + oracle]}
    frame = load_strict(directory, strict_cfg, registry)
    if not conditional and not oracle:
        return frame
    lagged_prices = [f for f in conditional if f == "price_lag168"]
    jao = [f for f in conditional if f in JAO_LAG_FEATURES]
    columns = NODE_KEYS + [c for f in lagged_prices for c in
        [f, f + "_observation_utc", f + "_completion_utc"]]
    source = ds.dataset(directory / "nodes").to_table(columns=columns,
        filter=ds.field("zone") == cfg["zone"]).to_pandas()
    frame = frame.merge(source, on=NODE_KEYS, how="left", validate="one_to_one", sort=False)
    for f in lagged_prices:
        # This verifies observation/interval timing, not historical publication.
        present = frame[f].notna()
        observation = frame[f + "_observation_utc"]
        completion = frame[f + "_completion_utc"]
        expected = frame.ts_utc - pd.Timedelta(hours=168)
        invalid = present & ((observation != expected) |
            (completion != observation + pd.Timedelta(hours=1)) |
            ~(observation < frame.forecast_origin_utc) |
            ~(completion <= frame.forecast_origin_utc))
        if invalid.any():
            raise ValueError(f"Conditional lag violates exact history/forecast timing: {f}")
    if jao:
        columns = NODE_KEYS + [JAO_LAG_FEATURES[f] for f in jao] + ["in_cwe", "in_core"]
        source = ds.dataset(directory / "nodes").to_table(columns=columns,
            filter=ds.field("zone") == cfg["zone"]).to_pandas()
        source = extend_constraint_history(directory, cfg["zone"], source, frame.ts_utc.min())
        frame = join_constraint_history(frame, source, jao)
    temperature = [f for f in conditional + oracle if f in [TEMPERATURE_CURRENT, TEMPERATURE_PERFECT]]
    if temperature:
        manifest = json.loads((directory / "source_manifest.json").read_text())
        info = manifest["weather_hourly"]
        if digest(info["path"])["sha256"] != info["sha256"]:
            raise ValueError("Original weather history checksum mismatch")
        source = ds.dataset(info["path"]).to_table(columns=["zone", "time_utc", "temp_pw"],
            filter=ds.field("zone") == cfg["zone"]).to_pandas().rename(columns={"time_utc": "ts_utc"})
        # Original weather starts in 2010, allowing lookup before the output window.
        source["ts_utc"] = utc(source.ts_utc)
        frame = join_temperature_history(frame, source, temperature)
    return frame


def join_temperature_history(frame, source, features):
    """Exact lookups; no nearest-time substitution, interpolation or future fallback."""
    if not set(features) <= {TEMPERATURE_CURRENT, TEMPERATURE_PERFECT}:
        raise ValueError("Unknown temperature feature")
    unique(source, NODE_KEYS, "Temperature history")
    values = source.set_index(NODE_KEYS).temp_pw
    out = frame.copy()
    for feature in features:
        stamp = out.ts_utc if feature == TEMPERATURE_PERFECT else out.forecast_origin_utc - pd.Timedelta(hours=1)
        key = pd.MultiIndex.from_arrays([out.zone, stamp], names=NODE_KEYS)
        out[feature] = values.reindex(key).to_numpy(dtype="float64")
        out[feature + "_source_utc"] = stamp
        if feature == TEMPERATURE_CURRENT:
            present = out[feature].notna()
            if not ((stamp[present] < out.loc[present, "forecast_origin_utc"]) &
                    (stamp[present] + pd.Timedelta(hours=1) <= out.loc[present, "forecast_origin_utc"])).all():
                raise ValueError("Current-temperature source interval is after forecast origin")
    return out


def extend_constraint_history(directory, zone, source, first_target):
    """Read original JAO observations before the master calendar when available."""
    manifest = json.loads((directory / "source_manifest.json").read_text())
    inputs = {}
    for name in ["jao_hourly", "jao_metadata/coverage.csv"]:
        info = manifest[name]
        if digest(info["path"])["sha256"] != info["sha256"]:
            raise ValueError(f"Original JAO history checksum mismatch: {name}")
        inputs[name] = Path(info["path"])
    start = first_target - pd.Timedelta(hours=48)
    first_source = source.ts_utc.min()
    before = ds.dataset(inputs["jao_hourly"]).to_table(
        columns=["zone", "ts_utc", "n_presolved", "min_ram_ratio", "regime"],
        filter=(ds.field("zone") == zone) &
            (ds.field("ts_utc") >= start.to_pydatetime()) &
            (ds.field("ts_utc") < first_source.to_pydatetime())).to_pandas()
    if before.empty:
        return source
    before["ts_utc"] = utc(before.ts_utc)
    before["in_cwe"] = before.regime.eq("CWE")
    before["in_core"] = before.regime.eq("CORE")
    if "jao_domain_present" in source:
        coverage = pd.read_csv(inputs["jao_metadata/coverage.csv"], usecols=["ts_utc", "domain_present"])
        coverage["ts_utc"] = utc(coverage.ts_utc)
        unique(coverage, ["ts_utc"], "Original JAO coverage")
        before = before.merge(coverage.rename(columns={"domain_present": "jao_domain_present"}),
            on="ts_utc", how="left", validate="many_to_one")
    return pd.concat([before[source.columns], source], ignore_index=True)


def join_constraint_history(frame, source, features):
    """Use only same-zone historical domains; no country/era substitution or fill.

    Domain effective time is distinct from a publication timestamp. This check
    does not certify that an archived domain was published by the forecast origin.
    """
    unique(source, NODE_KEYS, "JAO history")
    h = source.copy()
    member = h.in_cwe | h.in_core
    for feature in features:
        raw = JAO_LAG_FEATURES[feature]
        h[feature] = h[raw].astype("float64").where(member)
    h["jao_source_domain_utc"] = h.ts_utc
    h["ts_utc"] = h.ts_utc + pd.Timedelta(hours=48)
    frame = frame.merge(h[NODE_KEYS + features + ["jao_source_domain_utc"]], on=NODE_KEYS,
        how="left", validate="one_to_one", sort=False)
    present = frame[features].notna().any(axis=1)
    domain = frame.jao_source_domain_utc
    invalid = present & ((domain != frame.ts_utc - pd.Timedelta(hours=48)) |
        ~(domain < frame.forecast_origin_utc) |
        ~(domain + pd.Timedelta(hours=1) <= frame.forecast_origin_utc))
    if invalid.any():
        raise ValueError("JAO source-domain interval violates historical forecast timing")
    return frame


def run(config_path):
    cfg = json.loads(Path(config_path).read_text())
    if not re.fullmatch(r"[A-Za-z0-9_-]+", cfg["name"]):
        raise ValueError("Run name must contain only letters, digits, underscores or hyphens")
    directory = _path(cfg["dataset_directory"]).resolve()
    complete = json.loads((directory / "complete.json").read_text())
    if cfg["dataset_fingerprint"] != complete["fingerprint"]:
        raise ValueError("Dataset fingerprint differs from experiment pin")
    hashes = _check_artifacts(directory, complete, ["strict", "feature_registry.json", "construction_config.json", "build_fingerprint.json"])
    registry = experiment_registry(json.loads((directory / "feature_registry.json").read_text()))
    conditional = cfg.get("conditional_features", [])
    oracle = cfg.get("oracle_features", [])
    check_features(cfg["features"], registry, conditional, oracle)
    if conditional or oracle:
        hashes.update(_check_artifacts(directory, complete, ["nodes"]))
    if set(conditional) & set(JAO_LAG_FEATURES):
        hashes.update(_check_artifacts(directory, complete, ["source_manifest.json"]))
        source_manifest = json.loads((directory / "source_manifest.json").read_text())
        for name in ["jao_hourly", "jao_metadata/coverage.csv"]:
            hashes[f"original_history/{name}"] = source_manifest[name]["sha256"]
    if TEMPERATURE_CURRENT in conditional or TEMPERATURE_PERFECT in oracle:
        hashes.update(_check_artifacts(directory, complete, ["source_manifest.json"]))
        source_manifest = json.loads((directory / "source_manifest.json").read_text())
        hashes["original_history/weather_hourly"] = source_manifest["weather_hourly"]["sha256"]
    versions = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                "pyarrow": pyarrow.__version__, "xgboost": xgb.__version__}
    code_sha = digest(Path(__file__))["sha256"]
    identity = hashlib.sha256(canonical({"config": cfg, "code_sha256": code_sha, "versions": versions})).hexdigest()
    final = ROOT / "runs" / cfg["name"]
    result_file = ROOT / "results" / f"{cfg['name']}.json"
    if final.exists():
        previous = json.loads((final / "result.json").read_text())
        if previous["experiment_fingerprint"] != identity:
            raise ValueError("Existing run has a different configuration/code/runtime; choose a new run name")
        outputs = json.loads((final / "complete.json").read_text())
        for rel, sha in outputs.items():
            if digest(final / rel)["sha256"] != sha:
                raise ValueError(f"Saved run checksum mismatch: {rel}")
        print("Reusing the completed run; test is not reevaluated.", flush=True)
        return previous
    if result_file.exists():
        raise ValueError("A result already exists without its saved model; choose a new run name")
    started = time.monotonic()
    frame = load_inputs(directory, cfg, registry)
    fit, val = masks(frame, cfg)
    x = frame[cfg["features"]].astype("float32")
    if np.isinf(x.to_numpy()).any():
        raise ValueError("Infinite feature values are not supported")
    y = frame.price.to_numpy(dtype="float64")
    params = cfg["params"]
    target_kind = cfg.get("target_transform", "identity")
    target_scale = cfg.get("target_scale_eur_mwh", 50.0)
    if params.get("eval_metric") != "mae" or params.get("objective") not in {"reg:squarederror", "reg:absoluteerror"}:
        raise ValueError("Supported objectives are squared/absolute error with validation MAE")
    transformed_y = transform_target(y, target_kind, target_scale)
    if not np.isfinite(transformed_y[fit | val]).all():
        raise ValueError("Transformed fit/validation labels must be finite")
    model = xgb.XGBRegressor(**params, early_stopping_rounds=cfg["early_stopping_rounds"])
    print(f"Fit one XGBoost model: zone={cfg['zone']}, features={cfg['features']}, train={int(fit.sum())}, val={int(val.sum())}", flush=True)
    # Test data is never supplied to fit, early stopping, or any transform.
    model.fit(x.loc[fit], transformed_y[fit], eval_set=[(x.loc[val], transformed_y[val])], verbose=100)
    print(f"Selected {model.best_iteration + 1} rounds using validation MAE only", flush=True)
    # The conditional comparator is read only after model fitting is finished.
    hashes.update(_check_artifacts(directory, complete, ["nodes"]))
    history = ds.dataset(directory / "nodes").to_table(columns=NODE_KEYS + ["price_lag168"],
        filter=ds.field("zone") == cfg["zone"]).to_pandas()
    history = frame[NODE_KEYS].merge(history, on=NODE_KEYS, how="left", validate="one_to_one")
    metrics = {}
    final.parent.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{cfg['name']}-", dir=final.parent))
    try:
        for split in cfg["evaluation_splits"]:
            selected = frame.split.eq(split).to_numpy()
            if not selected.any():
                raise ValueError(f"No evaluation rows for {split}")
            block = frame.loc[selected].reset_index(drop=True)
            transformed_prediction = model.predict(x.loc[selected])  # automatically uses best_iteration
            prediction = transform_target(transformed_prediction, target_kind, target_scale, inverse=True)
            weekly = history.loc[selected, "price_lag168"].to_numpy(dtype="float64")
            metrics[split] = score(block, prediction, weekly)
            # Fixed, descriptive evaluation slices; no tuning or distribution plots.
            slice_defs = {
                "pre_2025_10": block.delivery_date < "2025-10-01",
                "post_2025_10": block.delivery_date >= "2025-10-01",
                "summer": pd.to_datetime(block.delivery_date).dt.month.isin([6, 7, 8])}
            metrics[split]["slices"] = {name: score(block.loc[mask].reset_index(drop=True), prediction[mask], weekly[mask])
                for name, mask in slice_defs.items() if mask.any()}
            output = block[NODE_KEYS + ["delivery_date", "forecast_origin_utc", "split", "price", "target_valid"]].copy()
            output["prediction"] = prediction
            if target_kind != "identity":
                output["prediction_transformed"] = transformed_prediction
            output["weekly_persistence_conditional"] = weekly
            for feature in conditional + oracle:
                output[feature] = block[feature]
                if feature + "_source_utc" in block:
                    output[feature + "_source_utc"] = block[feature + "_source_utc"]
            if "jao_source_domain_utc" in block:
                output["jao_source_domain_utc"] = block.jao_source_domain_utc
            output["common_comparison_mask"] = output.target_valid & np.isfinite(weekly)
            output.to_parquet(work / f"predictions_{split}.parquet", index=False)
        model.save_model(work / "model.ubj")
        loaded = xgb.XGBRegressor()
        loaded.load_model(work / "model.ubj")
        np.testing.assert_array_equal(model.predict(x.loc[val]), loaded.predict(x.loc[val]))
        json_write(work / "learning_curve.json", model.evals_result())
        json_write(work / "config.json", cfg)
        json_write(work / "input_artifact_hashes.json", hashes)
        json_write(work / "feature_registry.json", [r for r in registry if r["table"] == "node" and r["feature"] in cfg["features"]])
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
        result = {"name": cfg["name"], "zone": cfg["zone"], "config": cfg, "metrics": metrics,
            "experiment_fingerprint": identity, "dataset_fingerprint": complete["fingerprint"],
            "runner_sha256": code_sha, "versions": versions, "git_commit": git, "git_dirty": bool(dirty),
            "normalization": "none", "feature_missingness": "NaNs retained; native XGBoost missing-value handling",
            "target_transform": f"asinh(price/{target_scale:g}); inverse {target_scale:g}*sinh(prediction)"
                if target_kind == "asinh" else "none; EUR/MWh including negative prices",
            "target_scale_eur_mwh": target_scale if target_kind == "asinh" else None,
            "early_stopping_metric_units": "asinh(price/scale) units" if target_kind == "asinh" else "EUR/MWh",
            "prediction_units": "EUR/MWh after inverse transformation", "fit_rows": int(fit.sum()),
            "validation_rows": int(val.sum()), "early_stopping_split": cfg["validation_split"],
            "best_iteration_zero_based": model.best_iteration, "selected_rounds": model.best_iteration + 1,
            "best_validation_mae": model.best_score, "elapsed_seconds": time.monotonic() - started,
            "feature_importance_gain_normalized": dict(zip(cfg["features"], map(float, model.feature_importances_))),
            "feature_policy": "oracle_perfect_weather" if oracle else "conditional" if conditional else "strict",
            "conditional_features": conditional,
            "oracle_features": oracle,
            "missing_feature_rows_by_split": {str(split): {feature: int(g[feature].isna().sum())
                for feature in cfg["features"]} for split, g in frame.groupby("split")},
            "unassigned_rows_not_used": int(frame.split.eq("unassigned").sum()),
            "forecast_contract": "12:00 Europe/Brussels on previous calendar date; " +
                ("deliberate future-temperature oracle diagnostic" if oracle else
                 "explicit conditional historical inputs" if conditional else "strict admitted features only"),
            "input_availability_limitation": "Future delivery-hour reanalysis temperature intentionally supplied; not a deployable forecast"
                if oracle else "Historical price/domain/temperature interval timing checked; publication vintages unverified"
                if conditional else None,
            "comparison_limitation": "Weekly persistence is a conditional archived-price benchmark, not a certified strict feature set",
            "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        json_write(work / "result.json", result)
        outputs = {str(p.relative_to(work)): digest(p)["sha256"] for p in work.iterdir() if p.is_file()}
        json_write(work / "complete.json", outputs)
        work.chmod(work.stat().st_mode | 0o050)
        os.rename(work, final)
        json_write(result_file, result)
        return result
    finally:
        if work.exists():
            shutil.rmtree(work)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    try:
        result = run(args.config)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.exit(2, f"boosting: {exc}\n")
    for split, m in result["metrics"].items():
        common = m["weekly_persistence_conditional_common_hours"]
        print(f"{split}: XGB MAE={m['model']['mae']:.2f}, RMSE={m['model']['rmse']:.2f}; "
              f"conditional weekly persistence MAE={common['persistence']['mae']:.2f}; "
              f"common n={common['model']['n_hours']:,}")
    print(f"Model and predictions: runs/{result['name']}; report: results/{result['name']}.json")


if __name__ == "__main__":
    main()
