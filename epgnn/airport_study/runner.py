"""Isolated, resumable matched-model experiments and saved-artifact verification."""

import fcntl
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from epgnn.controlled_study import training
from epgnn.controlled_study.analysis import diagnostics, summarize
from epgnn.controlled_study.runner import (
    check_complete,
    pooled_xgb,
    signatures,
    study_identity,
)
from epgnn.dataset_build.storage import ROOT, canonical, digest
from epgnn.experiment_suite.artifacts import write_json
from epgnn.experiment_suite.evaluation import evaluate, grouped_scores
from epgnn.experiment_suite.neural import require_gpu

from .data import GROUPS, indices, prepare
from .design import jobs
from .sources import checked_sources


def identity(config):
    parent, provenance = study_identity(config)
    value = {
        "parent_identity": parent,
        "parent": provenance,
        "airport_code": {
            path.name: digest(path)["sha256"]
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "sources": checked_sources(config),
    }
    token = hashlib.sha256(canonical(value)).hexdigest()
    return token, value


def root_for(config):
    token, provenance = identity(config)
    return ROOT / "runs/airport_temperature" / token, token, provenance


def predict_tree(job, arrays, columns, zones, directory, config, smoke):
    selected_arrays = arrays
    if smoke:
        # Exactly the same short fit/validation windows used by neural smoke.
        split = np.array(arrays["split"], copy=True)
        fit, val = np.flatnonzero(split == 0), np.flatnonzero(split == 1)
        split[fit[: -24 * config["smoke"]["training_days"]]] = 3
        split[val[24 * config["smoke"]["validation_days"] :]] = 3
        selected_arrays = {**arrays, "split": split}
    _, details = pooled_xgb(job, selected_arrays, columns, zones, directory, config)
    model = xgb.XGBRegressor()
    model.load_model(directory / "pooled.ubj")
    predictions = {}
    for name, code in [("val", 1), ("test", 2)]:
        selected = np.flatnonzero(arrays["split"] == code)
        blocks = []
        for start in range(0, len(selected), 1024):
            raw = np.asarray(arrays["x"][selected[start : start + 1024]])[:, zones, :][
                :, :, columns
            ]
            blocks.append(
                (50 * np.sinh(model.predict(raw.reshape(-1, len(columns))))).reshape(
                    -1, len(zones)
                )
            )
        predictions[name] = np.concatenate(blocks)
    return predictions, details


def execute(job, arrays, metadata, zones, directory, config, token, smoke):
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "complete.json").exists():
        return check_complete(directory)
    started = time.monotonic()
    columns = indices(job.features)
    write_json(
        directory / "job.json",
        {"identity": token, "job": job.record(), "features": GROUPS[job.features]},
    )
    if job.model == "pooled_xgb":
        predictions, details = predict_tree(
            job, arrays, columns, zones, directory, config, smoke
        )
    else:
        fitted = training.fit(
            job,
            arrays,
            columns,
            zones,
            directory,
            config,
            token,
            config["neural"]["batch_size"],
            smoke=smoke,
        )
        restored = training.reload_fitted(job, columns, arrays, directory, config)
        selected = np.flatnonzero(arrays["split"] == 1)[:64]
        np.testing.assert_allclose(
            training.predict(fitted, arrays, selected, columns, require_gpu()),
            training.predict(restored, arrays, selected, columns, require_gpu()),
            rtol=1e-5,
            atol=1e-4,
        )
        predictions = {
            split: training.predict(
                fitted,
                arrays,
                np.flatnonzero(arrays["split"] == code),
                columns,
                require_gpu(),
            )[:, zones]
            for split, code in [("val", 1), ("test", 2)]
        }
        details = {
            "learning_curve": fitted.curves,
            "parameters": sum(p.numel() for p in fitted.model.parameters()),
            "head_width": fitted.model.width,
            "batch_size": fitted.batch_size,
            "peak_vram_gib": max(r["peak_vram_gib"] for r in fitted.curves),
        }
        del fitted, restored
        torch.cuda.empty_cache()
    rows, slices = [], []
    for split, prediction in predictions.items():
        rows.extend(
            r
            for r in evaluate(
                prediction, arrays, metadata, zones, job.name, split, directory
            )
            if r["coverage"] == "all"
        )
        group, weekly, _thresholds = diagnostics(
            directory / f"predictions_{split}.parquet", arrays, metadata, zones
        )
        slices.extend({**r, "split": split} for r in group)
        if split == "val":
            weekly.to_parquet(directory / "weekly_val.parquet", index=False)
    result = {
        "name": job.name,
        "variant": job.variant,
        "job": job.record(),
        "identity": token,
        "seconds": time.monotonic() - started,
        "details": details,
        "rows": rows,
        "slices": slices,
        "aggregate": grouped_scores(rows),
        "target_signatures": signatures(directory),
    }
    write_json(directory / "result.json", result)
    write_json(
        directory / "complete.json",
        {
            path.name: digest(path)["sha256"]
            for path in directory.iterdir()
            if path.is_file() and path.name != "complete.json"
        },
    )
    print(f"COMPLETE {job.name}: {result['seconds']:.1f}s", flush=True)
    return result


def run(config, smoke=False):
    root, token, provenance = root_for(config)
    phase = "smoke" if smoke else "full"
    directory = root / phase
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This airport-study phase is already running") from error
        try:
            gpu_path = root / "gpu_preflight.json"
            if smoke:
                from .gpu import run as gpu_run

                if not gpu_path.exists():
                    gpu_run(config, gpu_path, token)
            gate = json.loads(gpu_path.read_text())
            if not gate["passed"] or gate["fingerprint"] != token:
                raise ValueError("Matching ROCm preflight is required")
            if not smoke:
                evidence = json.loads((root / "smoke/verification.json").read_text())
                if (
                    not evidence["passed"]
                    or evidence["fingerprint"] != token
                    or evidence["job_count"] != len(jobs(config, True))
                ):
                    raise ValueError("All nine France smoke prototypes must pass first")
            cache, metadata, arrays = prepare(config)
            zones = [metadata["zones"].index("FR")] if smoke else metadata["eligible"]
            write_json(
                directory / "provenance.json",
                {
                    **provenance,
                    "cache": str(cache),
                    "zones": [metadata["zones"][z] for z in zones],
                    "phase": phase,
                    "feature_groups": GROUPS,
                    "availability": metadata["availability"],
                    "excluded": metadata["excluded"],
                },
            )
            results = []
            selected = jobs(config, smoke)
            for number, job in enumerate(selected):
                write_json(
                    directory / "status.json",
                    {
                        "state": "running",
                        "phase": phase,
                        "fingerprint": token,
                        "job": job.name,
                        "completed": number,
                        "total": len(selected),
                    },
                )
                result = execute(
                    job,
                    arrays,
                    metadata,
                    zones,
                    directory / job.name,
                    config,
                    token,
                    smoke,
                )
                if (
                    results
                    and result["target_signatures"] != results[0]["target_signatures"]
                ):
                    raise ValueError("Target keys/values/masks differ across models")
                results.append(result)
            prefix = ROOT / f"results/airport_temperature_{phase}"
            evidence = verify(directory, selected, arrays, metadata, zones, token)
            summarize(directory, results, config, prefix, token, smoke)
            write_json(directory / "verification.json", evidence)
            write_json(str(prefix) + "_verification.json", evidence)
            from .report import publish

            publish(root, directory, results, config, token, smoke)
            write_json(
                directory / "status.json",
                {
                    "state": "complete",
                    "phase": phase,
                    "fingerprint": token,
                    "jobs": len(results),
                },
            )
        except BaseException:
            write_json(
                directory / "status.json",
                {
                    "state": "failed",
                    "phase": phase,
                    "fingerprint": token,
                    "traceback": traceback.format_exc(),
                },
            )
            raise


def verify(directory, planned, arrays, metadata, zones, token):
    reference = {}
    maxima = []
    for job in planned:
        result = check_complete(directory / job.name)
        if result["identity"] != token:
            raise ValueError("Job identity mismatch")
        for split, code in [("val", 1), ("test", 2)]:
            frame = pd.read_parquet(
                directory / job.name / f"predictions_{split}.parquet"
            )
            keys = frame[["zone", "ts_utc", "price", "target_valid"]]
            if frame.duplicated(["zone", "ts_utc"]).any():
                raise ValueError("Duplicate saved prediction keys")
            if split in reference:
                pd.testing.assert_frame_equal(reference[split], keys)
            else:
                reference[split] = keys
            selected = np.flatnonzero(arrays["split"] == code)
            expected = np.asarray(arrays["y"][selected])[:, zones].T.reshape(-1)
            np.testing.assert_array_equal(frame.price.to_numpy(), expected)
            np.testing.assert_array_equal(
                frame.zone.to_numpy(),
                np.repeat([metadata["zones"][j] for j in zones], len(selected)),
            )
            np.testing.assert_array_equal(
                frame.ts_utc.to_numpy(),
                np.tile(
                    pd.to_datetime(arrays["times"][selected], utc=True).to_numpy(),
                    len(zones),
                ),
            )
            price, prediction = (
                frame.price.to_numpy(float),
                frame.prediction.to_numpy(float),
            )
            valid = frame.target_valid.to_numpy(bool)
            np.testing.assert_array_equal(valid, np.isfinite(price))
            if not np.isfinite(prediction).all():
                raise ValueError("Nonfinite saved prediction")
            paired = valid & np.isfinite(frame.price_lag168.to_numpy(float))
            residual = prediction - price
            actual = [
                np.abs(residual[valid]).mean(),
                np.sqrt(np.square(residual[valid]).mean()),
                np.abs(residual[paired]).mean(),
            ]
            saved = next(
                r
                for r in result["aggregate"]
                if r["split"] == split and r["scope"] == "overall"
            )
            error = np.abs(
                np.asarray(actual) - [saved["mae"], saved["rmse"], saved["paired_mae"]]
            )
            maxima.extend(error)
            np.testing.assert_allclose(error, 0, rtol=0, atol=1e-9)
        print(
            f"VERIFY {job.name}: hashes, canonical targets, identical masks, finite predictions, recomputed metrics",
            flush=True,
        )
    return {
        "passed": True,
        "fingerprint": token,
        "job_count": len(planned),
        "maximum_metric_error": float(max(maxima)),
        "target_signatures": result["target_signatures"],
        "checks": [
            "artifact hashes",
            "canonical unique target keys/values/masks",
            "finite predictions",
            "recomputed MAE/RMSE/paired MAE",
        ],
    }


def status(config):
    root, token, _ = root_for(config)
    result = {"fingerprint": token, "directory": str(root)}
    for phase in ["smoke", "full"]:
        path = root / phase / "status.json"
        result[phase] = (
            json.loads(path.read_text()) if path.exists() else {"state": "not_started"}
        )
    print(json.dumps(result, indent=2))


def report_saved(config):
    from .report import publish

    root, token, _ = root_for(config)
    directory = root / "full"
    evidence = json.loads((directory / "verification.json").read_text())
    if not evidence["passed"] or evidence["fingerprint"] != token:
        raise ValueError("Matching completed verification required")
    results = [check_complete(directory / job.name) for job in jobs(config)]
    publish(root, directory, results, config, token, False)


def verify_saved(config):
    root, token, _ = root_for(config)
    _, metadata, arrays = prepare(config)
    evidence = verify(
        root / "full", jobs(config), arrays, metadata, metadata["eligible"], token
    )
    write_json(root / "full/verification.json", evidence)
    print(json.dumps(evidence, indent=2))
