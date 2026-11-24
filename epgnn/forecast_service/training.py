"""Chronological exploratory backtest and artifact fitting."""

import concurrent.futures
import importlib.metadata
import json
import multiprocessing
import pickle
import sys
from datetime import date, datetime

import numpy as np

from . import models
from .contracts import (
    BRUSSELS,
    COMPONENTS,
    CPU_WORKERS,
    DAY,
    HISTORY,
    HOUR,
    UTC,
    ZONE_NAMES,
    atomic_write,
    configure_threads,
    digest,
    dump,
    file_hash,
    hour_grid,
    local_at,
)

FOLDS = (date(2024, 10, 14), date(2025, 1, 13), date(2025, 3, 10))
VALIDATION_DAYS = 14
TRAIN_DAYS = 730
CANDIDATES = {
    "pooled_xgb": (80, 180),
    "zone_xgb": (80, 180),
    "lear": (0.5, 2.0),
    "factor": (3, 8),
    "sequence": (48, 80),
}
PINNED_CHECKPOINT = {
    "config.json": "ff17bbc07b792c5a904cca265b8468579d736a4fe84981da25eb871b0a125bc6",
    "model.safetensors": "a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da",
}
UPSTREAM = "918df02c9d366b2145bc5876135fa2131d3ed2a6"


def model_version():
    from pathlib import Path

    root = Path(models.__file__).parent
    return digest(
        dump(
            {
                name: file_hash(root / name)
                for name in ("models.py", "contracts.py", "zones.json")
            }
        )
    )


def runtime():
    return {
        "python": list(sys.version_info[:3]),
        **{
            p: importlib.metadata.version(p)
            for p in ("numpy", "scipy", "scikit-learn", "torch", "xgboost")
        },
    }


def store_artifact(state, value):
    data = pickle.dumps(value, protocol=5)
    sha = digest(data)
    atomic_write(state / "artifacts" / sha, data)
    return sha


def read_artifact(state, sha):
    if (
        not isinstance(sha, str)
        or len(sha) != 64
        or any(c not in "0123456789abcdef" for c in sha)
    ):
        raise ValueError("Invalid artifact digest")
    data = (state / "artifacts" / sha).read_bytes()
    if digest(data) != sha:
        raise ValueError("Model artifact hash mismatch")
    # The state directory contains only locally fitted or explicitly pinned assets.
    return pickle.loads(data)


def prepare_timesfm(state, timesfm_repo):
    import subprocess

    upstream = timesfm_repo.resolve() / "upstream-timesfm"
    package = upstream / "src" / "timesfm3"
    path = timesfm_repo.resolve() / "runs/runtime"
    git = ["git", "-C", str(upstream)]
    if (
        subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
        != UPSTREAM
    ):
        raise ValueError("TimesFM upstream revision differs from the benchmark pin")
    if subprocess.check_output(
        [*git, "status", "--porcelain", "--untracked-files=all", "src/timesfm3"],
        text=True,
    ):
        raise ValueError("TimesFM source has local edits")
    hashes = {
        str(path / "checkpoint" / name): sha for name, sha in PINNED_CHECKPOINT.items()
    }
    if any(file_hash(name) != sha for name, sha in hashes.items()):
        raise ValueError("TimesFM checkpoint differs from the benchmark pin")
    hashes.update({str(p): file_hash(p) for p in sorted(package.rglob("*.py"))})
    vendor = path / "python"
    for name, version in (("huggingface_hub", "1.16.4"), ("safetensors", "0.8.0")):
        metadata = vendor / f"{name}-{version}.dist-info" / "METADATA"
        hashes[str(metadata)] = file_hash(metadata)
        for pattern in ("*.py", "*.so"):
            hashes.update(
                {str(p): file_hash(p) for p in sorted((vendor / name).rglob(pattern))}
            )
    return store_artifact(
        state,
        {
            "checkpoint": str(path / "checkpoint"),
            "hashes": hashes,
            "python_path": [str(upstream / "src"), str(path / "python")],
            "device": "cuda",
            "upstream": UPSTREAM,
        },
    )


def load_archive(path, verify=False):
    if verify:
        manifest = json.loads((path / "complete.json").read_bytes())
        for name in ("y.npy", "times.npy", "metadata.json"):
            if file_hash(path / name) != manifest[name]:
                raise ValueError(f"Archive hash mismatch for {name}")
    meta = json.loads((path / "metadata.json").read_bytes())
    if tuple(meta["zones"][i] for i in meta["eligible"]) != ZONE_NAMES:
        raise ValueError("Archive does not match the 49-zone benchmark roster")
    times = np.load(path / "times.npy")
    if not np.all(np.diff(times) == 3_600_000_000_000):
        raise ValueError("Archive timestamps must be consecutive UTC hours")
    t0 = datetime.fromtimestamp(int(times[0]) / 1e9, UTC)
    y = np.load(path / "y.npy", mmap_mode="r")[:, meta["eligible"]]
    return t0, y


def dataset(t0, y, first, days):
    result = []
    for i in range(days):
        day = first + i * DAY
        grid = hour_grid(day)
        k = (grid[0] - t0) // HOUR
        if k < HISTORY or k + len(grid) > len(y):
            raise ValueError(
                f"Archive does not cover all inputs and outcomes for {day}"
            )
        result.append(
            (models.daily_inputs(day, y[k - HISTORY : k]), y[k : k + len(grid)])
        )
    return result


def fit_candidate(archive, boundary, validate, name, candidate, indices):
    t0, y = load_archive(archive)
    # D-1 auction outcomes can become known around or after this cutoff. Omit D-1
    # targets from fitting and start validation at D.
    train = dataset(t0, y, boundary - (TRAIN_DAYS + 1) * DAY, TRAIN_DAYS)
    val = dataset(t0, y, boundary, VALIDATION_DAYS) if validate else []
    if name in ("pooled_xgb", "zone_xgb"):
        x = np.concatenate([models.tree_features(p) for p, _ in train])
        target = np.concatenate([y for _, y in train])
        if name == "pooled_xgb":
            fit = models.fit_tree(
                x.reshape(-1, x.shape[-1]), target.reshape(-1), candidate
            )
            pred = [models.predict_tree(fit, models.tree_features(p)) for p, _ in val]
        else:
            fit = {
                z: models.fit_tree(x[:, z, :-1], target[:, z], candidate)
                for z in indices
            }
            pred = [
                np.column_stack(
                    [
                        models.predict_tree(fit[z], models.tree_features(p)[:, z, :-1])
                        for z in indices
                    ]
                )
                for p, _ in val
            ]
    else:
        train_fn, infer_fn = {
            "lear": (models.fit_lear, models.predict_lear),
            "factor": (models.fit_factor, models.predict_factor),
            "sequence": (models.fit_sequence, models.predict_sequence),
        }[name]
        fit = train_fn(train, candidate)
        pred = [infer_fn(fit, p) for p, _ in val]
    return np.concatenate(pred) if validate else fit


def run_fits(executor, archive, boundary, candidates, validate):
    tasks = {}
    for name, choices in candidates.items():
        for candidate in choices:
            batches = (
                np.array_split(np.arange(len(ZONE_NAMES)), CPU_WORKERS)
                if name == "zone_xgb"
                else [()]
            )
            tasks[name, candidate] = [
                (
                    executor.submit(
                        fit_candidate,
                        archive,
                        boundary,
                        validate,
                        name,
                        candidate,
                        indices,
                    ),
                    indices,
                )
                for indices in batches
            ]
    result = {}
    for (name, candidate), jobs in tasks.items():
        if name == "zone_xgb" and validate:
            result[name, candidate] = np.concatenate(
                [job.result() for job, _ in jobs], axis=1
            )
        elif name == "zone_xgb":
            result[name, candidate] = {
                z: fit for job, _ in jobs for z, fit in job.result().items()
            }
        else:
            result[name, candidate] = jobs[0][0].result()
    return result


def metrics(y, p, weekly):
    valid = np.isfinite(y) & np.isfinite(p)
    matched = valid & np.isfinite(weekly)
    mae = float(np.abs(y[valid] - p[valid]).mean()) if valid.any() else None
    baseline = (
        float(np.abs(y[matched] - weekly[matched]).mean()) if matched.any() else None
    )
    matched_mae = (
        float(np.abs(y[matched] - p[matched]).mean()) if matched.any() else None
    )
    return {
        "mae": mae,
        "n": int(valid.sum()),
        "matched": int(matched.sum()),
        "matched_mae": matched_mae,
        "weekly_mae": baseline,
        "skill": 1 - matched_mae / baseline if baseline else None,
    }


def fit_blend(y, predictions):
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix, eye, hstack, vstack

    stack = np.column_stack([predictions[n].ravel() for n in COMPONENTS])
    values = y.ravel()
    valid = np.isfinite(values) & np.isfinite(stack).all(axis=1)
    x, values = stack[valid], values[valid]
    if not len(x):
        raise ValueError("No common out-of-fold predictions for blend training")
    # Optimize absolute error with nonnegative weights that sum to one.
    n, k = x.shape
    identity = eye(n, format="csr")
    a = vstack(
        [hstack([csr_matrix(x), -identity]), hstack([-csr_matrix(x), -identity])]
    )
    eq = csr_matrix((np.ones(k), (np.zeros(k, int), np.arange(k))), shape=(1, n + k))
    result = linprog(
        np.r_[np.zeros(k), np.ones(n)],
        A_ub=a,
        b_ub=np.r_[values, -values],
        A_eq=eq,
        b_eq=[1],
        method="highs",
    )
    if not result.success:
        raise RuntimeError("Blend weight fitting failed")
    return result.x[:k]


def select(candidates, rows):
    picked = {}
    for name, values in candidates.items():
        losses = []
        for v in values:
            total = count = 0
            for row in rows:
                y = row["y"]
                pred = row["predictions"][name, v]
                base = row["weekly"]
                # Compare each candidate on identical rows.
                good = np.isfinite(y) & np.isfinite(base)
                for other in values:
                    good &= np.isfinite(row["predictions"][name, other])
                total += np.abs(y[good] - pred[good]).sum()
                count += good.sum()
            losses.append(total / count if count else np.inf)
        if not np.isfinite(losses).any():
            raise ValueError(f"No matched validation scores for {name}")
        picked[name] = values[int(np.argmin(losses))]
    return picked


def stack_oof(rows):
    y = np.concatenate([row["y"] for row in rows])
    predictions = {"weekly": np.concatenate([row["weekly"] for row in rows])}
    for name in CANDIDATES:
        predictions[name] = np.concatenate(
            [row["predictions"][name, row["selected"][name]] for row in rows]
        )
    return y, predictions


def train(state, archive, timesfm_repo=None):
    configure_threads()
    t0, panel = load_archive(archive, verify=True)
    end = (t0 + len(panel) * HOUR).astimezone(BRUSSELS).date()
    if local_at(end) >= t0 + len(panel) * HOUR:
        end -= DAY
    boundary = (
        end + DAY
    )  # Training through the day preceding the possibly partial final day.
    rows = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=CPU_WORKERS, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        for day in FOLDS:
            val = dataset(t0, panel, day, VALIDATION_DAYS)
            runs = run_fits(executor, archive, day, CANDIDATES, True)
            # Fix the first fold a priori. Later folds select only on earlier validation.
            chosen = (
                select(CANDIDATES, rows)
                if rows
                else {k: v[0] for k, v in CANDIDATES.items()}
            )
            rows.append(
                {
                    "day": day.isoformat(),
                    "selected": chosen,
                    "predictions": runs,
                    "y": np.concatenate([y for _, y in val]),
                    "weekly": np.concatenate([models.weekly(p) for p, _ in val]),
                }
            )
            print(f"Completed exploratory fold {day}", flush=True)
        final = select(CANDIDATES, rows)
        fits = run_fits(
            executor, archive, boundary, {k: (v,) for k, v in final.items()}, False
        )
    report = {
        "classification": "exploratory_unverified_archive_vintages",
        "folds": [],
        "training_code_sha256": file_hash(__file__),
        "model_version": model_version(),
        "runtime": runtime(),
        "archive": str(archive.resolve()),
        "archive_hashes": {
            n: file_hash(archive / n) for n in ("times.npy", "y.npy", "metadata.json")
        },
        "validation_days_per_fold": VALIDATION_DAYS,
        "training_days_per_fit": TRAIN_DAYS,
        "final_training_end_exclusive": (boundary - DAY).isoformat(),
        "selected": final,
    }
    for i, row in enumerate(rows):
        y, predictions = stack_oof([row])
        scores = {name: metrics(y, p, row["weekly"]) for name, p in predictions.items()}
        if i:
            weights = fit_blend(*stack_oof(rows[:i]))
            blended = models.blend_predictions(weights, predictions, y.shape)
            scores["blend"] = metrics(y, blended, row["weekly"])
        report["folds"].append(
            {"start": row["day"], "selected": row["selected"], "scores": scores}
        )
    weights = fit_blend(*stack_oof(rows))
    report["blend_weights"] = dict(zip(COMPONENTS, weights.tolist()))
    report_bytes = dump(report)
    atomic_write(state / "backtest.json", report_bytes)
    artifacts = {
        "weekly": store_artifact(state, {"algorithm": "weekly_168h"}),
        "blend": store_artifact(state, weights),
    }
    artifacts.update({k: store_artifact(state, fits[k, v]) for k, v in final.items()})
    if timesfm_repo:
        frozen = prepare_timesfm(state, timesfm_repo)
        artifacts.update(timesfm_joint=frozen, timesfm_independent=frozen)
    spec = {
        "artifacts": artifacts,
        "model_version": model_version(),
        "runtime": runtime(),
        "backtest_sha256": digest(report_bytes),
    }
    atomic_write(state / "candidate.json", dump(spec))
    return report
