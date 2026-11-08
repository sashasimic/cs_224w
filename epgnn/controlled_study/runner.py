"""Fingerprint-separated, locked and resumable orchestration of the full study."""

import fcntl
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from epgnn.dataset_build.storage import ROOT, canonical, digest
from epgnn.experiment_suite.artifacts import write_json
from epgnn.experiment_suite.evaluation import evaluate, grouped_scores
from epgnn.experiment_suite.runner import checked_cache, identity, xgb_experiment
from epgnn.experiment_suite.neural import require_gpu
from .analysis import diagnostics, summarize
from .design import FEATURE_SETS, feature_indices, jobs
from .history import Histories
from . import preflight, training
from .models import parameter_count


def study_identity(config):
    parent_token, parent = identity(config)
    source = {
        str(p.relative_to(ROOT)): digest(p)["sha256"]
        for p in sorted(Path(__file__).parent.glob("*.py"))
    }
    provenance = {
        "suite_identity": parent_token,
        "suite_provenance": parent,
        "study_code": source,
        "config": config,
    }
    return hashlib.sha256(canonical(provenance)).hexdigest(), provenance


def pooled_xgb(job, arrays, columns, zones, directory, config):
    values = np.asarray(arrays["x"][:, zones, :])[:, :, columns]
    target = np.asarray(arrays["y"][:, zones])
    fit, val = arrays["split"] == 0, arrays["split"] == 1
    filename = directory / "pooled.ubj"
    if filename.exists():
        model = xgb.XGBRegressor()
        model.load_model(filename)
    else:
        training_mask = np.isfinite(target[fit].reshape(-1))
        validation_mask = np.isfinite(target[val].reshape(-1))
        tx = values[fit].reshape(-1, len(columns))[training_mask]
        ty = np.arcsinh(target[fit].reshape(-1)[training_mask] / 50)
        vx = values[val].reshape(-1, len(columns))[validation_mask]
        vy = np.arcsinh(target[val].reshape(-1)[validation_mask] / 50)
        model = xgb.XGBRegressor(
            **config["xgb"], random_state=job.seed, early_stopping_rounds=50
        )
        model.fit(tx, ty, eval_set=[(vx, vy)], verbose=False)
        temporary = filename.with_suffix(".tmp.ubj")
        model.save_model(temporary)
        temporary.rename(filename)
        restored = xgb.XGBRegressor()
        restored.load_model(filename)
        np.testing.assert_array_equal(
            model.predict(vx[:128]), restored.predict(vx[:128])
        )
    predictions = {}
    for split, index in [("val", 1), ("test", 2)]:
        selected = arrays["split"] == index
        flat = values[selected].reshape(-1, len(columns))
        predictions[split] = (50 * np.sinh(model.predict(flat))).reshape(
            int(selected.sum()), len(zones)
        )
    return predictions, {
        "selected_rounds": model.best_iteration + 1,
        "country_identity_feature": False,
    }


def signatures(directory):
    result = {}
    for split in ["val", "test"]:
        frame = pd.read_parquet(
            directory / f"predictions_{split}.parquet",
            columns=["zone", "ts_utc", "price", "target_valid"],
        )
        result[split] = hashlib.sha256(
            pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()
        ).hexdigest()
    return result


def check_complete(directory):
    for name, sha in json.loads((directory / "complete.json").read_text()).items():
        if digest(directory / name)["sha256"] != sha:
            raise ValueError(f"Completed artifact changed: {directory/name}")
    return json.loads((directory / "result.json").read_text())


def execute(
    job, arrays, metadata, zones, directory, config, token, report, histories, smoke
):
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "complete.json").exists():
        return check_complete(directory)
    started = time.monotonic()
    columns = feature_indices(job)
    write_json(
        directory / "job.json",
        {
            "identity": token,
            "job": job.record(),
            "features": FEATURE_SETS[job.features],
        },
    )
    if job.model == "xgb":
        local = {**config, "xgb": {**config["xgb"], "random_state": job.seed}}
        predictions, details = xgb_experiment(arrays, columns, zones, directory, local)
    elif job.model == "pooled_xgb":
        predictions, details = pooled_xgb(
            job, arrays, columns, zones, directory, config
        )
    else:
        batch_size = report["batch_sizes"][
            "temporal" if job.model.startswith("gru_") else "static"
        ]
        fitted = training.fit(
            job,
            arrays,
            columns,
            zones,
            directory,
            config,
            token,
            batch_size,
            histories,
            smoke,
        )
        restored = training.reload_fitted(
            job, columns, arrays, directory, config, histories
        )
        check_indices = np.flatnonzero(arrays["split"] == 1)[:batch_size]
        actual = training.predict(fitted, arrays, check_indices, columns, require_gpu())
        expected = training.predict(
            restored, arrays, check_indices, columns, require_gpu()
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-4)
        predictions = {
            split: training.predict(
                fitted,
                arrays,
                np.flatnonzero(arrays["split"] == index),
                columns,
                require_gpu(),
            )[:, zones]
            for split, index in [("val", 1), ("test", 2)]
        }
        details = {
            "learning_curve": fitted.curves,
            "parameters": parameter_count(fitted.model),
            "head_width": fitted.model.width,
            "batch_size": batch_size,
            "peak_vram_gib": max(row["peak_vram_gib"] for row in fitted.curves),
            "history": fitted.history.state() if fitted.history else None,
        }
        del fitted, restored
        torch.cuda.empty_cache()
    rows, slices = [], []
    thresholds = {}
    for split, prediction in predictions.items():
        rows.extend(
            evaluate(prediction, arrays, metadata, zones, job.name, split, directory)
        )
        group, weekly, thresholds = diagnostics(
            directory / f"predictions_{split}.parquet", arrays, metadata, zones
        )
        slices.extend({**row, "split": split} for row in group)
        if split == "val":
            weekly.to_parquet(directory / "weekly_val.parquet", index=False)
    write_json(directory / "regime_thresholds.json", thresholds)
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
    token, provenance = study_identity(config)
    root = ROOT / "runs/controlled_study" / token
    phase = "smoke" if smoke else "full"
    directory = root / phase
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This study phase is already running") from error
        try:
            if smoke:
                try:
                    report = preflight.checked(config)
                except (FileNotFoundError, ValueError):
                    report = preflight.run(config)
            else:
                report = preflight.checked(config)
                gate = json.loads((root / "smoke/summary.json").read_text())
                expected = [
                    job.name for job in jobs(config) if job.seed == config["seeds"][0]
                ]
                if gate["fingerprint"] != token or sorted(
                    row["name"] for row in gate["jobs"]
                ) != sorted(expected):
                    raise ValueError(
                        "Every prototype must pass France smoke before full execution"
                    )
            write_json(
                directory / "status.json",
                {"state": "preparing", "fingerprint": token, "phase": phase},
            )
            cache, metadata, arrays = checked_cache(config)
            zones = [metadata["zones"].index("FR")] if smoke else metadata["eligible"]
            histories = Histories(arrays, config["neural"]["history_hours"])
            write_json(
                directory / "provenance.json",
                {
                    **provenance,
                    "cache": str(cache),
                    "phase": phase,
                    "zones": [metadata["zones"][z] for z in zones],
                    "excluded": metadata["excluded"],
                    "availability": "conditional; historical source publication vintages unresolved",
                    "history": "same-identity hourly intervals ending by literal local-noon origin",
                    "smoke_training": config["smoke"] if smoke else None,
                },
            )
            selected = [
                job
                for job in jobs(config)
                if not smoke or job.seed == config["seeds"][0]
            ]
            results = []
            for index, job in enumerate(selected):
                write_json(
                    directory / "status.json",
                    {
                        "state": "running",
                        "phase": phase,
                        "fingerprint": token,
                        "job": job.name,
                        "completed": index,
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
                    report,
                    histories,
                    smoke,
                )
                if (
                    results
                    and result["target_signatures"] != results[0]["target_signatures"]
                ):
                    raise ValueError("Job target keys, values or validity masks differ")
                results.append(result)
            prefix = ROOT / f"results/controlled_study_{phase}"
            summary = summarize(directory, results, config, prefix, token, smoke)
            write_json(
                directory / "verification.json",
                {
                    "passed": True,
                    "jobs": len(results),
                    "target_signatures": results[0]["target_signatures"],
                    "all_predictions_finite": True,
                },
            )
            if smoke:
                from .report import estimate

                write_json(
                    root / "estimate.json", estimate(config, summary, arrays, report)
                )
            else:
                from .report import publish

                publish(summary, config)
            write_json(
                directory / "status.json",
                {
                    "state": "complete",
                    "phase": phase,
                    "fingerprint": token,
                    "jobs": len(results),
                },
            )
            return summary
        except BaseException as error:
            write_json(
                directory / "status.json",
                {
                    "state": (
                        "interrupted"
                        if isinstance(error, KeyboardInterrupt)
                        else "failed"
                    ),
                    "phase": phase,
                    "fingerprint": token,
                    "traceback": traceback.format_exc(),
                },
            )
            raise


def status(config):
    token, _ = study_identity(config)
    root = ROOT / "runs/controlled_study" / token
    record = {"fingerprint": token, "directory": str(root)}
    for phase in ["smoke", "full"]:
        path = root / phase / "status.json"
        record[phase] = (
            json.loads(path.read_text()) if path.exists() else {"state": "not_started"}
        )
    if (root / "estimate.json").exists():
        estimate = json.loads((root / "estimate.json").read_text())
        record["estimate"] = {
            key: value for key, value in estimate.items() if key != "jobs"
        }
        record["estimate"]["per_job_file"] = str(root / "estimate.json")
    print(json.dumps(record, indent=2))
