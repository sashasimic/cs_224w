"""Separate frozen TimesFM comparisons on the exploratory price-only folds."""

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import models
from .contracts import PRIVATE_MODELS, atomic_write, dump, file_hash
from .service import load_spec
from .training import (
    FOLDS,
    VALIDATION_DAYS,
    dataset,
    load_archive,
    metrics,
    read_artifact,
)


def evaluate(state, archive, model, artifact):
    t0, data = load_archive(archive)
    forecaster = models.timesfm_forecaster(read_artifact(state, artifact))
    scores = {}
    for day in FOLDS:
        rows = dataset(t0, data, day, VALIDATION_DAYS)
        y = np.concatenate([y for _, y in rows])
        predictions = np.concatenate(
            [
                models.predict_timesfm(forecaster, p, model == "timesfm_joint")
                for p, _ in rows
            ]
        )
        baseline = np.concatenate([models.weekly(p) for p, _ in rows])
        missing = ~np.isfinite(predictions)
        predictions = np.where(missing, baseline, predictions)
        scores[day.isoformat()] = {
            **metrics(y, predictions, baseline),
            "fallback_hours": int((missing & np.isfinite(baseline)).sum()),
        }
    return scores


def run(state, archive):
    spec = load_spec(state)
    gpu = json.loads((state / "gpu.json").read_bytes())
    if not gpu["validated"] or spec["artifacts"]["timesfm_joint"] != gpu["artifact"]:
        raise ValueError("Run the ROCm benchmark against this candidate first")
    backtest = json.loads((state / "backtest.json").read_bytes())
    if any(
        file_hash(archive / name) != sha
        for name, sha in backtest["archive_hashes"].items()
    ):
        raise ValueError(
            "Private comparisons must use the same pinned price panel as the core backtest"
        )
    count = gpu["gpu_workers"]
    with ProcessPoolExecutor(
        max_workers=count, mp_context=multiprocessing.get_context("spawn")
    ) as workers:
        pending = {
            model: workers.submit(
                evaluate, state, archive, model, spec["artifacts"][model]
            )
            for model in PRIVATE_MODELS
        }
        values = {name: job.result() for name, job in pending.items()}
    result = {
        "scope": "private_research",
        "classification": backtest["classification"],
        "caveat": "Checkpoint pretraining overlap with archived electricity prices is unverified.",
        "artifact": gpu["artifact"],
        "core_backtest_sha256": spec["backtest_sha256"],
        "validation_days_per_fold": VALIDATION_DAYS,
        "scores": values,
    }
    atomic_write(state / "backtest-private.json", dump(result))
    return result
