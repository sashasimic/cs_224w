"""Suite scheduling, verified prediction publication and report entry points."""

import fcntl
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from epgnn.dataset_build.storage import ROOT, digest, json_write
from epgnn.experiment_suite.neural import require_gpu
from epgnn.relational_forecast.catalog import identity, read_json
from epgnn.relational_forecast.experiment import scores

from . import training
from .inputs import Inputs

CONTRASTS = (
    ("relational_cross", "sql_full"),
    ("relational_cross", "relational_local"),
    ("learned", "physical"),
    ("learned", "uniform"),
    ("margin", "learned"),
    ("margin", "shuffled_margin"),
)
KEYS = ["ts_utc", "zone", "price", "target_valid", "weekly", "core_covered"]


def generation(settings, cache, smoke):
    code = {
        name: digest(Path(__file__).with_name(name))["sha256"]
        for name in (
            "inputs.py",
            "models.py",
            "sql_features.py",
            "training.py",
            "runner.py",
        )
    }
    manifest = {
        "settings": settings,
        "cache": cache.name,
        "lineage": read_json(cache / "manifest.json"),
        "mode": "smoke" if smoke else "validation",
        "code": code,
        "torch": torch.__version__,
        "hip": torch.version.hip,
    }
    output = ROOT / settings["output"] / manifest["mode"] / identity(manifest)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".manifest.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        if not (output / "manifest.json").exists():
            json_write(output / "manifest.json", manifest)
    return output


def common_frame(data, validation):
    n = len(data.zones)
    shape = (len(validation), n)
    core = np.zeros(shape, dtype=bool)
    edge = np.isfinite(
        data.margin[np.searchsorted(data.margin_hours, validation), ..., 0]
    )
    core[:, data.core] = edge.any(axis=1) & data.active[np.ix_(validation, data.core)]
    prices = data.prices[validation].ravel()
    return pd.DataFrame(
        {
            "ts_utc": np.repeat(pd.to_datetime(data.times[validation], utc=True), n),
            "zone": np.tile(data.zones, len(validation)),
            "price": prices,
            "target_valid": np.isfinite(prices),
            "weekly": data.recent[validation, :, 4].ravel(),
            "core_covered": core.ravel(),
        }
    )


def pairs(data):
    return np.unique(np.sort(data.edges, axis=1), axis=0).T


def spread_mae(pred, actual, data):
    a, b = pairs(data)
    forecast = pred[:, a] - pred[:, b]
    spread = actual[:, a] - actual[:, b]
    keep = np.isfinite(spread)
    return float(np.abs(forecast[keep] - spread[keep]).mean())


def run(settings, data, output, subset, requested_seed, smoke):
    chosen = training.FAMILIES if subset == "all" else [subset]
    seeds = (
        [requested_seed]
        if requested_seed is not None
        else settings["seeds"][:1]
        if smoke
        else settings["seeds"]
    )
    if any(seed not in settings["seeds"] for seed in seeds):
        raise ValueError("Seed is outside the prespecified suite")
    print(f"Writing {subset} runs to {output}", flush=True)
    device = require_gpu() if any(family != "xgb" for family in chosen) else None
    torch.set_num_threads(4)
    for family in chosen:
        jobs = list(itertools.product(training.FAMILIES[family], seeds))
        fit, calibration, validation = training.split(data, family, smoke)
        common = common_frame(data, validation)
        build_job = None
        for variant, seed in jobs:
            dest = output / f"{variant}_{seed}"
            dest.mkdir(exist_ok=True)
            with (dest / ".job.lock").open("a") as guard:
                fcntl.flock(guard, fcntl.LOCK_EX)
                marker = dest / "result.json"
                prediction_file = dest / "predictions.parquet"
                if marker.exists():
                    if (
                        read_json(marker)["prediction_sha256"]
                        != digest(prediction_file)["sha256"]
                    ):
                        raise ValueError(f"Completed job predictions changed: {dest}")
                    continue
                if build_job is None:
                    if family == "xgb":
                        build_job = training.xgb_jobs(
                            data, fit, calibration, validation, settings, smoke, output
                        )
                    else:
                        build_job = training.neural_jobs(
                            data,
                            family,
                            fit,
                            calibration,
                            validation,
                            settings,
                            smoke,
                            device,
                        )
                prediction, metrics = build_job(variant, seed, dest)
                if (
                    prediction.shape != (len(validation), len(data.zones))
                    or not np.isfinite(prediction).all()
                ):
                    raise ValueError("Model must predict every validation key")
                frame = common.assign(prediction=prediction.ravel())
                temporary = dest / "predictions.partial.parquet"
                frame.to_parquet(temporary, index=False)
                temporary.replace(prediction_file)
                result = {
                    "variant": variant,
                    "family": family,
                    "seed": seed,
                    "train_start": data.dates[fit[0]],
                    "train_end": data.dates[calibration[-1]],
                    "spread_mae": spread_mae(prediction, data.prices[validation], data),
                    "scores": {
                        "all": scores(frame),
                        "core": scores(frame[frame.core_covered]),
                        "FR": scores(frame[frame.zone.eq("FR")]),
                    },
                    "prediction_sha256": digest(prediction_file)["sha256"],
                    "training": metrics,
                }
                if family == "adaptive":
                    result["links_sha256"] = digest(dest / "link_weights.npy")["sha256"]
                json_write(marker, result)
                print(
                    f"Completed {variant} seed {seed}. France MAE {result['scores']['FR']['mae']:.3f}",
                    flush=True,
                )
        del build_job
        if device is not None:
            torch.cuda.empty_cache()


def validate(settings, data, output, smoke):
    seeds = settings["seeds"][:1] if smoke else settings["seeds"]
    validation = training.split(data, "relational", smoke)[2]
    expected = common_frame(data, validation)
    if not smoke and len(expected) != 429_240:
        raise ValueError("Unexpected benchmark cohort size")
    results, predictions = [], {}
    for variant, seed in itertools.product(training.VARIANTS, seeds):
        dest = output / f"{variant}_{seed}"
        result = read_json(dest / "result.json")
        if (result["variant"], result["seed"]) != (variant, seed):
            raise ValueError(f"Wrong run identity in {dest}")
        source = dest / "predictions.parquet"
        if digest(source)["sha256"] != result["prediction_sha256"]:
            raise ValueError(f"Prediction checksum mismatch in {dest}")
        frame = pd.read_parquet(source)
        if not frame[KEYS].equals(expected) or not np.isfinite(frame.prediction).all():
            raise ValueError(f"Unexpected prediction keys, masks or values in {dest}")
        if (
            "links_sha256" in result
            and digest(dest / "link_weights.npy")["sha256"] != result["links_sha256"]
        ):
            raise ValueError(f"Link weights changed in {dest}")
        predictions.setdefault(variant, []).append(frame.prediction.to_numpy())
        results.append(result)
    print(f"Verified {len(results)} jobs on {len(expected):,} matched keys", flush=True)
    return expected, results, predictions


def dispatch(args, settings, prepared):
    data = Inputs(prepared)
    smoke = settings["smoke"] if args.command == "smoke" else None
    output = generation(settings, prepared, smoke)
    if args.command in {"smoke", "run"}:
        run(settings, data, output, args.family, args.seed, smoke)
        if smoke and args.family == "all" and args.seed is None:
            validate(settings, data, output, smoke)
    elif args.command == "validate":
        validate(settings, data, output, smoke)
    else:
        from .report import publish

        publish(settings, data, output, *validate(settings, data, output, smoke))
