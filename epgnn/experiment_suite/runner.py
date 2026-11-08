"""Five matched experiments: one data contract, restartable jobs, auditable scores."""

import hashlib
import fcntl
import json
import re
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from epgnn.dataset_build.storage import ROOT, canonical, digest
from . import data, neural
from .artifacts import write_json
from .evaluation import evaluate, grouped_scores
from .gpu import gpu_smoke

EXPERIMENTS = ["recent_xgb", "rich_xgb", "neighbour_xgb", "real_gat", "shuffled_gat"]
DEFAULT_CONFIG = ROOT / "configs/experiments/vibing_suite.json"


def identity(config):
    code = {
        str(p.relative_to(ROOT)): digest(p)["sha256"]
        for p in sorted((ROOT / "epgnn/experiment_suite").glob("*.py"))
    }
    code["epgnn/train.py"] = digest(ROOT / "epgnn/train.py")["sha256"]
    code["epgnn/boosting.py"] = digest(ROOT / "epgnn/boosting.py")["sha256"]
    value = {
        "config": config,
        "code": code,
        "versions": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xgboost": xgb.__version__,
        },
    }
    return hashlib.sha256(canonical(value)).hexdigest(), value


def checked_cache(config):
    source = ROOT / config["dataset_directory"]
    complete = json.loads((source / "complete.json").read_text())
    if complete["fingerprint"] != config["dataset_fingerprint"]:
        raise ValueError("Dataset differs from pinned construction")
    cache_identity = hashlib.sha256(
        canonical(
            {
                "source": complete,
                "data_code": digest(data.__file__)["sha256"],
                "boosting_code": digest(ROOT / "epgnn/boosting.py")["sha256"],
            }
        )
    ).hexdigest()
    cache = ROOT / "staging/vibing_suite" / cache_identity
    if not (cache / "complete.json").exists():
        temporary = cache.with_name("." + cache.name + ".building")
        if temporary.exists():
            import shutil

            shutil.rmtree(temporary)
        data.prepare(source, temporary)
        temporary.rename(cache)
    for name, sha in json.loads((cache / "complete.json").read_text()).items():
        if digest(cache / name)["sha256"] != sha:
            raise ValueError("Cached input checksum mismatch: " + name)
    return cache, *data.load(cache)


def xgb_experiment(arrays, features, zones, directory, config):
    fit = np.flatnonzero(arrays["split"] == 0)
    val = np.flatnonzero(arrays["split"] == 1)
    evaluation = {
        s: np.flatnonzero(arrays["split"] == k) for s, k in [("val", 1), ("test", 2)]
    }
    predictions = {
        s: np.empty((len(v), len(zones)), np.float32) for s, v in evaluation.items()
    }
    selected = {}
    for k, j in enumerate(zones):
        filename = directory / f"zone_{j}.ubj"
        x = np.asarray(arrays["x"][:, j, :])[:, features]
        y = np.asarray(arrays["y"][:, j])
        if filename.exists():
            model = xgb.XGBRegressor()
            model.load_model(filename)
        else:
            model = xgb.XGBRegressor(**config["xgb"], early_stopping_rounds=50)
            valid_fit, valid_val = fit[np.isfinite(y[fit])], val[np.isfinite(y[val])]
            model.fit(
                x[valid_fit],
                np.arcsinh(y[valid_fit] / 50),
                eval_set=[(x[valid_val], np.arcsinh(y[valid_val] / 50))],
                verbose=False,
            )
            temporary = filename.with_suffix(".tmp.ubj")
            model.save_model(temporary)
            temporary.rename(filename)
            reloaded = xgb.XGBRegressor()
            reloaded.load_model(filename)
            np.testing.assert_array_equal(
                model.predict(x[valid_val[:128]]), reloaded.predict(x[valid_val[:128]])
            )
        selected[str(j)] = model.best_iteration + 1
        for s, idx in evaluation.items():
            predictions[s][:, k] = 50 * np.sinh(model.predict(x[idx]))
        print(f"XGBoost zone index {j}: {model.best_iteration+1} rounds", flush=True)
    return predictions, {"selected_rounds": selected}


def run(config_path=DEFAULT_CONFIG, smoke=False):
    """Reject duplicate launches targeting the same experiment generation."""
    config = json.loads(Path(config_path).read_text())
    token, _ = identity(config)
    phase = "france" if smoke else "full"
    directory = ROOT / "runs/vibing_suite" / token / phase
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This suite generation is already running") from error
        return _run(config_path, smoke)


def _run(config_path, smoke):
    config = json.loads(Path(config_path).read_text())
    token, provenance = identity(config)
    root = ROOT / "runs/vibing_suite" / token
    phase = "france" if smoke else "full"
    directory = root / phase
    directory.mkdir(parents=True, exist_ok=True)
    status = directory / "status.json"
    try:
        write_json(status, {"state": "preparing", "phase": phase, "fingerprint": token})
        gpu_report = json.loads(
            (ROOT / "runs/vibing_gpu_smoke/report.json").read_text()
        )
        if (
            not gpu_report["passed"]
            or gpu_report["neural_code_sha256"] != digest(neural.__file__)["sha256"]
            or gpu_report["loss_code_sha256"]
            != digest(ROOT / "epgnn/train.py")["sha256"]
        ):
            raise ValueError(
                "Run GPU smoke for the current implementation before starting"
            )
        if not smoke:
            gate = json.loads((root / "france/summary.json").read_text())
            if gate["fingerprint"] != token or len(gate["experiments"]) != 5:
                raise ValueError(
                    "All France experiments must pass before full execution"
                )
        cache, meta, arrays = checked_cache(config)
        if meta["dataset_fingerprint"] != config["dataset_fingerprint"]:
            raise ValueError("Cache refers to wrong dataset")
        zones = [meta["zones"].index("FR")] if smoke else meta["eligible"]
        if not all(j in meta["eligible"] for j in zones):
            raise ValueError("Smoke zone is ineligible")
        write_json(
            directory / "provenance.json",
            {
                **provenance,
                "fingerprint": token,
                "cache": str(cache),
                "training": {
                    "xgboost": "all finite pretrain/train labels",
                    "neural": (
                        "last 90 training days, 3 epochs, first 14 validation days"
                        if smoke
                        else "all finite pretrain/train labels, full validation period"
                    ),
                },
                "limitations": [
                    "Conditional publication availability",
                    "Single-seed exploratory test evaluation",
                    "GAT pools zones whereas XGBoost fits per zone; real-vs-shuffled isolates graph topology",
                ],
                "excluded": meta["excluded"],
                "zones": [meta["zones"][j] for j in zones],
            },
        )
        all_rows, experiments = [], []
        for experiment in EXPERIMENTS:
            work = directory / experiment
            work.mkdir(exist_ok=True)
            write_json(
                status,
                {
                    "state": "running",
                    "phase": phase,
                    "experiment": experiment,
                    "fingerprint": token,
                },
            )
            if (work / "complete.json").exists():
                for name, sha in json.loads(
                    (work / "complete.json").read_text()
                ).items():
                    if digest(work / name)["sha256"] != sha:
                        raise ValueError("Saved experiment checksum mismatch")
                saved = json.loads((work / "result.json").read_text())
                all_rows.extend(saved["rows"])
                experiments.append(saved)
                continue
            features = (
                data.RECENT
                if experiment == "recent_xgb"
                else data.FEATURES if experiment == "neighbour_xgb" else data.RICH
            )
            columns = [data.FEATURES.index(f) for f in features]
            started = time.monotonic()
            if experiment.endswith("xgb"):
                predictions, details = xgb_experiment(
                    arrays, columns, zones, work, config
                )
            else:
                neural_cfg = {
                    **config["neural"],
                    "identity": token + ":" + phase + ":" + experiment,
                    "batch_size": gpu_report["batch_size"],
                }
                model, prep, edges, curves, batch_size = neural.train(
                    arrays,
                    columns,
                    zones,
                    work,
                    neural_cfg,
                    shuffled=experiment == "shuffled_gat",
                    smoke=smoke,
                )
                device = neural.require_gpu()
                # Verify actual checkpoint reload against the selected model on real data.
                reload = neural.PriceGAT(len(columns) * 2).to(device)
                state = torch.load(
                    work / "best.pt", map_location=device, weights_only=False
                )
                reload.load_state_dict(state["model"])
                idx = np.flatnonzero(arrays["split"] == 1)[:24]
                before = neural.infer(
                    model, arrays, idx, columns, prep, edges, batch_size, device
                )
                after = neural.infer(
                    reload, arrays, idx, columns, prep, edges, batch_size, device
                )
                np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-5)
                predictions = {
                    s: neural.infer(
                        model,
                        arrays,
                        np.flatnonzero(arrays["split"] == k),
                        columns,
                        prep,
                        edges,
                        batch_size,
                        device,
                    )[:, zones]
                    for s, k in [("val", 1), ("test", 2)]
                }
                details = {
                    "learning_curve": curves,
                    "preprocessor": prep.state(),
                    "peak_vram_gib": max(c["peak_vram_gib"] for c in curves),
                    "batch_size": batch_size,
                }
                del model, reload
                torch.cuda.empty_cache()
            rows = []
            for split, prediction in predictions.items():
                rows.extend(
                    evaluate(prediction, arrays, meta, zones, experiment, split, work)
                )
            saved = {
                "experiment": experiment,
                "fingerprint": token,
                "features": features,
                "seconds": time.monotonic() - started,
                "rows": rows,
                "details": details,
            }
            write_json(work / "result.json", saved)
            write_json(
                work / "complete.json",
                {
                    p.name: digest(p)["sha256"]
                    for p in work.iterdir()
                    if p.is_file() and p.name != "complete.json"
                },
            )
            all_rows.extend(rows)
            experiments.append(saved)
            print(
                f'Completed {phase}/{experiment}: {saved["seconds"]:.1f}s', flush=True
            )
        # Every method must score exactly the same target population.
        counts = pd.DataFrame(all_rows).pivot(
            index=["zone", "split", "coverage"], columns="experiment", values="hours"
        )
        if counts.isna().any().any() or counts.nunique(axis=1).max() != 1:
            raise ValueError("Experiment target masks/counts differ")
        summary = {
            "fingerprint": token,
            "phase": phase,
            "experiments": experiments,
            "aggregate": grouped_scores(all_rows),
            "excluded_zones": meta["excluded"],
        }
        write_json(directory / "summary.json", summary)
        pd.DataFrame(all_rows).to_csv(directory / "by_zone.csv", index=False)
        pd.DataFrame(summary["aggregate"]).to_csv(
            directory / "aggregate.csv", index=False
        )
        destination = ROOT / f"results/vibing_suite_{phase}.json"
        write_json(destination, summary)
        pd.DataFrame(all_rows).to_csv(
            ROOT / f"results/vibing_suite_{phase}_by_zone.csv", index=False
        )
        pd.DataFrame(summary["aggregate"]).to_csv(
            ROOT / f"results/vibing_suite_{phase}_aggregate.csv", index=False
        )
        write_json(status, {"state": "complete", "phase": phase, "fingerprint": token})
        return summary
    except BaseException as error:
        state = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        write_json(
            status,
            {
                "state": state,
                "phase": phase,
                "fingerprint": token,
                "traceback": traceback.format_exc(),
            },
        )
        raise


def status(config_path=DEFAULT_CONFIG, saved=False):
    """Inspect current-code runs, or explicitly select the last saved full run."""
    config = json.loads(Path(config_path).read_text())
    current_token, _ = identity(config)
    token = current_token
    if saved:
        summary = json.loads((ROOT / "results/vibing_suite_full.json").read_text())
        token = summary["fingerprint"]
        if not isinstance(token, str) or not re.fullmatch(r"[0-9a-f]{64}", token):
            raise ValueError("Saved result contains an invalid fingerprint")
    root = ROOT / "runs/vibing_suite" / token
    result = {
        "fingerprint": token,
        "artifact_directory": str(root),
        "current_code_fingerprint": current_token,
        "matches_current_code": token == current_token,
    }
    for phase in ["france", "full"]:
        path = root / phase / "status.json"
        absent = "not_available_locally" if saved else "not_started"
        result[phase] = (
            json.loads(path.read_text()) if path.exists() else {"state": absent}
        )
    print(json.dumps(result, indent=2))
