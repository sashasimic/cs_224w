"""Prespecified France feature diagnostics for the milestone; no parameter search.

Each new model is trained on the same v1.0 target rows with fixed partner learner
settings. Availability masks apply to features, never to evaluation targets.
"""
import argparse
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import xgboost as xgb

from epgnn.boosting import masks, score
from epgnn.dataset_build.contracts import CORE_START, NODE_KEYS, TZ, forecast_origin, unique, utc
from epgnn.dataset_build.storage import ROOT, canonical, digest, json_write

CALENDAR = ["hour_local", "dow", "month"]
WEATHER = ["temp_pw", "temp_anom", "wind_cf", "solar_cf"]
PARTNER_ORDER = ["price_lag24", "price_lag168", "load_fc"] + [f + "_lag24" for f in WEATHER] + ["net_import_lag48"] + CALENDAR


def prepare_features(frame, weather):
    f = frame.copy().sort_values(NODE_KEYS).reset_index(drop=True)
    f["ts_utc"] = utc(f.ts_utc)
    f["forecast_origin_utc"] = forecast_origin(f.ts_utc)
    f["target_valid"] = f.has_price.astype(bool)
    f["price"] = f.price.astype("float64")
    local = f.ts_utc.dt.tz_convert(TZ)
    f["utc_offset_hours"] = (local.dt.tz_localize(None) - f.ts_utc.dt.tz_localize(None)).dt.total_seconds() / 3600
    # This diagnostic is deliberately France-only; these regional identities are
    # identical to the audited builder and do not reuse v1.0's static in_core flag.
    if not f.zone.eq("FR").all():
        raise ValueError("The prespecified sequential feature diagnostic is France-only")
    f["audit_in_cwe"] = f.ts_utc < CORE_START
    f["audit_in_core"] = f.ts_utc >= CORE_START
    source24 = f.ts_utc - pd.Timedelta(hours=24)
    source24_day = source24.dt.tz_convert(TZ).dt.strftime("%Y-%m-%d")
    # Auction prices are generally known before delivery; do not reject a price
    # merely because its delivery hour is after origin. Guard the fall-DST case
    # where t-24h can land in the target delivery day itself.
    f["price_lag24_guarded"] = f.price_lag24.where(source24_day < f.delivery_date)
    f["weather24_interval_completed"] = source24 + pd.Timedelta(hours=1) <= f.forecast_origin_utc
    history = weather.copy().rename(columns={"time_utc": "ts_utc"})
    history["ts_utc"] = utc(history.ts_utc)
    unique(history, NODE_KEYS, "Weather diagnostic source")
    history = history.set_index(NODE_KEYS)
    key48 = pd.MultiIndex.from_arrays([f.zone, f.ts_utc - pd.Timedelta(hours=48)], names=NODE_KEYS)
    for c in WEATHER:
        f[c + "_lag24_past"] = f[c + "_lag24"].where(f.weather24_interval_completed)
        f[c + "_lag48"] = history.reindex(key48)[c].to_numpy()
    unique(f, NODE_KEYS, "Diagnostic node grid")
    return f


def profile_features(name):
    if name == "weekly_flags":
        return CALENDAR + ["utc_offset_hours", "audit_in_cwe", "audit_in_core", "price_lag168"]
    if name == "weekly_calendar":
        return ["price_lag168"] + CALENDAR
    groups = {"recent_price": ["price_lag24", "price_lag168"],
        "load": ["price_lag24", "price_lag168", "load_fc"],
        "flows": ["price_lag24", "price_lag168", "load_fc", "net_import_lag48"]}
    if name in groups:
        return ["price_lag24_guarded" if c == "price_lag24" else c for c in PARTNER_ORDER if c in groups[name] or c in CALENDAR]
    if name not in {"weather_past24", "weather48", "weather_all24"}:
        raise ValueError(f"Unknown diagnostic profile: {name}")
    columns = []
    for c in PARTNER_ORDER:
        if c == "price_lag24":
            c = "price_lag24_guarded"
        elif c in [x + "_lag24" for x in WEATHER]:
            if name == "weather_past24":
                c += "_past"
            elif name == "weather48":
                c = c.removesuffix("24") + "48"
        columns.append(c)
    return columns


def validation_block_interval(before, after, repetitions=2000):
    """Paired calendar-week bootstrap of validation losses; no test selection."""
    keys = ["zone", "ts_utc", "price", "target_valid"]
    pd.testing.assert_frame_equal(before[keys], after[keys], check_dtype=False)
    valid = before.target_valid.to_numpy(bool)
    b = before.loc[valid].copy()
    delta = (np.abs(before.loc[valid, "prediction"].to_numpy(float) - b.price.to_numpy(float)) -
             np.abs(after.loc[valid, "prediction"].to_numpy(float) - b.price.to_numpy(float)))
    weeks = pd.to_datetime(b.delivery_date).dt.to_period("W-SUN")
    grouped = pd.DataFrame({"week": weeks, "improvement": delta}).groupby("week").improvement.agg(["sum", "size"])
    rng = np.random.default_rng(224)
    draws = rng.integers(0, len(grouped), size=(repetitions, len(grouped)))
    estimates = grouped["sum"].to_numpy()[draws].sum(axis=1) / grouped["size"].to_numpy()[draws].sum(axis=1)
    return {"mean_mae_improvement": float(delta.mean()), "ci95_lower": float(np.quantile(estimates, .025)),
        "ci95_upper": float(np.quantile(estimates, .975)), "calendar_week_blocks": len(grouped),
        "bootstrap_repetitions": repetitions,
        "interpretation": "descriptive paired validation-loss uncertainty conditional on fitted models; no retraining or causal attribution"}


def run(config_path):
    cfg = json.loads(Path(config_path).read_text())
    if cfg["zone"] != "FR":
        raise ValueError("This diagnostic plan fixes the country to France")
    data = ROOT / cfg["audited_data_directory"]
    published = json.loads((data / "complete.json").read_text())
    if published["fingerprint"] != cfg["dataset_fingerprint"]:
        raise ValueError("Dataset fingerprint mismatch")
    source_manifest = data / "source_manifest.json"
    if digest(source_manifest)["sha256"] != published["artifacts"]["source_manifest.json"]:
        raise ValueError("Source manifest checksum mismatch")
    manifest = json.loads(source_manifest.read_text())
    legacy = ROOT / cfg["legacy_path"]
    expected = manifest["reference/nodes"]
    if digest(legacy)["sha256"] != expected["sha256"]:
        raise ValueError("Legacy node source checksum mismatch")
    weather_info = manifest["weather_hourly"]
    if digest(weather_info["path"])["sha256"] != weather_info["sha256"]:
        raise ValueError("Weather source checksum mismatch")
    reference = json.loads((ROOT / cfg["partner_reference"]).read_text())
    if reference["partner_commit"] != cfg["partner_commit"]:
        raise ValueError("Partner reference commit mismatch")
    columns = list(dict.fromkeys(NODE_KEYS + ["delivery_date", "split", "price", "has_price", "in_core"] +
        PARTNER_ORDER + WEATHER))
    raw = ds.dataset(legacy).to_table(columns=columns, filter=ds.field("zone") == "FR").to_pandas()
    weather = ds.dataset(weather_info["path"]).to_table(columns=["zone", "time_utc"] + WEATHER,
        filter=ds.field("zone") == "FR").to_pandas()
    frame = prepare_features(raw, weather)
    fit, val = masks(frame, cfg)
    y = frame.price.to_numpy(dtype="float32")
    target = np.arcsinh(y / np.float32(50))
    row_records = []
    key = {"config": cfg, "code": digest(__file__)["sha256"], "legacy_sha256": expected["sha256"],
        "weather_sha256": weather_info["sha256"], "xgboost": xgb.__version__, "numpy": np.__version__, "pandas": pd.__version__}
    token = hashlib.sha256(canonical(key)).hexdigest()
    directory = ROOT / "runs" / cfg["name"] / token
    directory.parent.mkdir(exist_ok=True)
    if (directory / "complete.json").exists():
        for rel, sha in json.loads((directory / "complete.json").read_text()).items():
            if digest(directory / rel)["sha256"] != sha:
                raise ValueError("Diagnostic artifact changed")
        return json.loads((directory / "summary.json").read_text())
    work = Path(tempfile.mkdtemp(prefix=".diagnostic-", dir=directory.parent))
    try:
        for name in cfg["profiles"]:
            features = profile_features(name)
            x = frame[features].astype("float32")
            model = xgb.XGBRegressor(**cfg["params"], early_stopping_rounds=cfg["early_stopping_rounds"])
            print(f"Fit France diagnostic {name}: {len(features)} features, {int(fit.sum())} targets", flush=True)
            model.fit(x.loc[fit], target[fit], eval_set=[(x.loc[val], target[val])], verbose=False)
            outdir = work / name
            outdir.mkdir()
            model.save_model(outdir / "model.ubj")
            reload = xgb.XGBRegressor()
            reload.load_model(outdir / "model.ubj")
            np.testing.assert_array_equal(model.predict(x.loc[val]), reload.predict(x.loc[val]))
            for split in cfg["evaluation_splits"]:
                selected = frame.split.eq(split).to_numpy()
                block = frame.loc[selected].reset_index(drop=True)
                prediction = (np.sinh(model.predict(x.loc[selected])) * np.float32(50)).astype("float32")
                metrics = score(block, prediction, block.price_lag168.to_numpy(float))
                row_records.append({"profile": name, "split": split, "mae": metrics["model"]["mae"],
                    "rmse": metrics["model"]["rmse"], "hours": metrics["model"]["n_hours"],
                    "persistence_mae": metrics["weekly_persistence_conditional_common_hours"]["persistence"]["mae"],
                    "selected_rounds": model.best_iteration + 1, "features": features,
                    "availability": "retrospective weather includes future effective hours" if name == "weather_all24" else
                        "effective-time restrictions applied where relevant; historical vintages still unverified"})
                out = block[NODE_KEYS + ["delivery_date", "forecast_origin_utc", "price", "target_valid"]].copy()
                out["prediction"] = prediction
                out["price_lag168"] = block.price_lag168
                out["price24_masked_same_delivery_day"] = block.price_lag24.notna() & block.price_lag24_guarded.isna()
                out["weather24_interval_completed"] = block.weather24_interval_completed
                out.to_parquet(outdir / f"predictions_{split}.parquet", index=False)
                print(f"  {split}: MAE={metrics['model']['mae']:.3f}; rounds={model.best_iteration + 1}", flush=True)
            json_write(outdir / "learning_curve.json", model.evals_result())
            json_write(outdir / "feature_importance.json", dict(zip(features, map(float, model.feature_importances_))))
        intervals = {}
        contrasts = [("weekly_calendar", "recent_price"), ("recent_price", "load"), ("load", "flows"),
                     ("flows", "weather_past24"), ("weather_past24", "weather_all24"), ("weather48", "weather_all24")]
        for before, after in contrasts:
            intervals[f"{before}_to_{after}"] = validation_block_interval(
                pd.read_parquet(work / before / "predictions_val.parquet"), pd.read_parquet(work / after / "predictions_val.parquet"))
        timing = {}
        for split, g in frame.groupby("split"):
            valid = g.target_valid
            timing[str(split)] = {"valid_targets": int(valid.sum()),
                "weather24_post_origin_intervals": int((valid & ~g.weather24_interval_completed).sum()),
                "price24_same_delivery_day": int((valid & g.price_lag24.notna() & g.price_lag24_guarded.isna()).sum())}
        summary = {"fingerprint": token, "inputs": key, "config": cfg, "rows": row_records,
            "validation_week_block_bootstrap": intervals, "timing_audit": timing,
            "fitted_targets": int(fit.sum()), "artifact_directory": str(directory),
            "limitations": ["Sequential one-seed feature additions are order-dependent, not causal effects",
                "Past effective time does not establish publication availability; reanalysis and load vintages are unverified",
                "Test has been examined across earlier exploratory runs; these are milestone diagnostics, not a fresh confirmatory holdout",
                "Legacy features retain original input fills and source geography; evaluation target rows are never filtered by feature availability"]}
        json_write(work / "summary.json", summary)
        pd.DataFrame(row_records).to_csv(work / "scores.csv", index=False)
        json_write(work / "complete.json", {str(p.relative_to(work)): digest(p)["sha256"] for p in work.rglob("*") if p.is_file()})
        work.chmod(work.stat().st_mode | 0o050)
        os.rename(work, directory)
        json_write(ROOT / "results" / f"{cfg['name']}.json", summary)
        pd.DataFrame(row_records).to_csv(ROOT / "results" / f"{cfg['name']}.csv", index=False)
        return summary
    finally:
        if work.exists():
            shutil.rmtree(work)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    run(p.parse_args().config)
