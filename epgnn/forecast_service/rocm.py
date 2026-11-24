"""Measure total device VRAM while running the full 49-market TimesFM input."""

import json
import tempfile
import time
from datetime import date
from pathlib import Path

import numpy as np

from .contracts import (
    HISTORY,
    PRIVATE_MODELS,
    ZONE_NAMES,
    atomic_write,
    cutoff,
    deadline,
    dump,
    hour_grid,
)
from .ledger import Ledger
from .service import encoded, load_spec, spawn_worker
from .training import load_archive

BUDGET_SECONDS = 290
VRAM_LIMIT = 12 * 2**30


def measure(state, scratch, day, sha, artifact, device_vram, parallel):
    start = time.monotonic()
    peak = int(device_vram.read_text())
    records = {}
    active = {}
    pending = list(PRIVATE_MODELS)
    try:
        while pending or active:
            while pending and len(active) < parallel:
                model = pending.pop(0)
                launched = time.monotonic()
                output = scratch / model
                active[model] = (
                    spawn_worker(state, day, model, sha, artifact, output),
                    output,
                    launched,
                )
            peak = max(peak, int(device_vram.read_text()))
            for model, (process, output, launched) in list(active.items()):
                if process.poll() is None:
                    continue
                completed = time.monotonic()
                valid = False
                if process.returncode == 0:
                    result = json.loads(output.read_bytes())
                    values = np.asarray(result["values"], np.float32)
                    valid = (
                        result["snapshot"] == sha
                        and result["artifact"] == artifact
                        and values.shape[1] == len(ZONE_NAMES)
                        and np.isfinite(values).all()
                    )
                records[model] = {
                    "seconds": completed - launched,
                    "finish_seconds": completed - start,
                    "exit": process.returncode,
                    "complete_49_zone_forecast": bool(valid),
                }
                del active[model]
            if time.monotonic() - start >= BUDGET_SECONDS:
                break
            if pending or active:
                time.sleep(0.05)
    finally:
        for process, _, _ in active.values():
            process.kill()
            process.wait()
    on_time = len(records) == len(PRIVATE_MODELS) and all(
        r["complete_49_zone_forecast"] and r["finish_seconds"] < BUDGET_SECONDS
        for r in records.values()
    )
    return {
        "workers": parallel,
        "peak_total_vram_bytes": peak,
        "workers_finished": records,
        "within_deadline": on_time,
    }


def benchmark(state, archive, device_vram):
    spec = load_spec(state)
    artifact = spec["artifacts"]["timesfm_joint"]
    if spec["artifacts"]["timesfm_independent"] != artifact:
        raise ValueError("Private modes must use the same pinned TimesFM checkpoint")
    _, y = load_archive(archive, verify=True)
    hist = np.full((HISTORY, len(ZONE_NAMES)), np.nan, np.float32)
    for z in range(len(ZONE_NAMES)):
        observed = np.flatnonzero(np.isfinite(y[:, z]))
        if not len(observed):
            raise ValueError(
                f"Archive contains no GPU load fixture prices for {ZONE_NAMES[z]}"
            )
        end = observed[-1] + 1
        window = y[max(0, end - HISTORY) : end, z]
        hist[-len(window) :, z] = window

    day = date(2026, 10, 25)  # Stress the largest supported horizon.
    key = day.isoformat()
    snapshot = {
        "day": key,
        "zones": list(ZONE_NAMES),
        "hours": [h.isoformat() for h in hour_grid(day)],
        "cutoff": cutoff(day).isoformat(),
        "history": encoded(hist),
    }
    with tempfile.TemporaryDirectory(dir=state) as root_name:
        root = Path(root_name)
        with Ledger(root) as ledger:
            sha = ledger.save_snapshot(
                snapshot, {n: artifact for n in PRIVATE_MODELS}, deadline(day)
            )
        (root / "artifacts").symlink_to(
            state.resolve() / "artifacts", target_is_directory=True
        )
        results = []
        for parallel in (2, 1):
            # Workers open their own inherited ledger / artifact paths. Each run is isolated.
            with tempfile.TemporaryDirectory(dir=root) as name:
                scratch = Path(name)
                # A nested scratch dir is used for workers and inherits the same data.
                results.append(
                    measure(root, scratch, key, sha, artifact, device_vram, parallel)
                )
            if results[-1]["within_deadline"] and (
                parallel == 1 or results[-1]["peak_total_vram_bytes"] < VRAM_LIMIT
            ):
                break
    last = results[-1]
    validated = last["within_deadline"] and (
        last["workers"] == 1 or last["peak_total_vram_bytes"] < VRAM_LIMIT
    )
    result = {
        "artifact": artifact,
        "fixture": "49 nonempty archived market contexts, 25 forecast hours",
        "zone_count": len(ZONE_NAMES),
        "horizon_hours": len(hour_grid(day)),
        "budget_seconds": BUDGET_SECONDS,
        "vram_limit_bytes": VRAM_LIMIT,
        "vram_counter": str(device_vram.resolve()),
        "runs": results,
        "validated": validated,
        "gpu_workers": last["workers"] if validated else 1,
    }
    atomic_write(state / "gpu.json", dump(result))
    return result
