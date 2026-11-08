"""Named ROCm checks for every architecture, graph mode and epoch resume path."""

import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from epgnn.dataset_build.storage import ROOT, digest
from epgnn.experiment_suite.artifacts import write_json
from epgnn.experiment_suite.data import FEATURES, shuffle_edges
from epgnn.experiment_suite.neural import require_gpu
from epgnn.train import masked_loss
from .design import Job
from .history import Histories
from .models import make_model, parameter_count
from . import training

DEFAULT_CONFIG = ROOT / "configs/experiments/controlled_study.json"
REPORT = ROOT / "runs/controlled_study_preflight/report.json"


def source_hashes():
    files = [
        Path(__file__).parent / (name + ".py")
        for name in ["models", "history", "training", "preflight"]
    ]
    files += [ROOT / "epgnn/experiment_suite/neural.py", ROOT / "epgnn/train.py"]
    return {str(path.relative_to(ROOT)): digest(path)["sha256"] for path in files}


def exercise(kind, graph, batch_size, config):
    device = require_gpu()
    torch.manual_seed(224)
    torch.cuda.manual_seed_all(224)
    model = make_model(kind, 26, config["neural"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    pairs = [(a, b) for a in range(62) for b in range(62) if a != b]
    rng = np.random.default_rng(224)
    edges = np.array(
        [pairs[i] for i in rng.choice(len(pairs), 520, replace=False)], np.int64
    ).T
    if graph in {"none", "self"}:
        edges = np.empty((2, 0), np.int64)
    elif graph == "shuffled":
        edges = shuffle_edges(edges, 901)
    edges = torch.tensor(edges.copy(), device=device)
    x = torch.randn(batch_size, 62, 26, device=device)
    active = torch.ones(batch_size, 62, dtype=torch.bool, device=device)
    active[:, -3:] = False
    target = torch.randn(batch_size, 62, device=device)
    target[::2, 0] = float("nan")
    sequence = None
    if kind.startswith("gru_"):
        sequence = (
            torch.randn(
                batch_size, 62, config["neural"]["history_hours"], 2, device=device
            ),
            torch.arange(batch_size, device=device),
        )
    durations = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        torch.cuda.synchronize()
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x, edges, active, sequence)
        loss = masked_loss(prediction, target, torch.isfinite(target))
        if not torch.isfinite(loss):
            raise RuntimeError("GPU smoke loss is not finite")
        loss.backward()
        if any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in model.parameters()
        ):
            raise RuntimeError("GPU smoke gradients are nonfinite")
        optimizer.step()
        torch.cuda.synchronize()
        durations.append(time.monotonic() - started)
    peak = torch.cuda.max_memory_allocated() / 2**30
    if peak > 12:
        raise torch.OutOfMemoryError("Above the 12 GiB safety budget")
    return {
        "model": kind,
        "graph": graph,
        "batch_size": batch_size,
        "parameters": parameter_count(model),
        "head_width": model.width,
        "warm_step_seconds": float(np.mean(durations[1:])),
        "peak_vram_gib": peak,
    }


def resume_check(config, output):
    rng = np.random.default_rng(224)
    total, zones = 24 * 12, 4
    fixture = {
        "x": rng.normal(size=(total, zones, len(FEATURES))).astype(np.float32),
        "y": rng.normal(50, 10, size=(total, zones)).astype(np.float32),
        "active": np.ones((total, zones), bool),
        "split": np.array([0] * (24 * 8) + [1] * (24 * 2) + [2] * (24 * 2)),
        "times": np.arange(total, dtype=np.int64) * 3_600_000_000_000
        + 1_704_067_200_000_000_000,
        "edges": np.array([[0, 1, 2, 3], [1, 2, 3, 0]], np.int64),
    }
    fixture["y"][0, 0] = np.nan
    local = json.loads(json.dumps(config))
    local["neural"]["epochs"] = 1
    columns = list(range(13))
    for kind in ["mlp", "gat", "gru_mlp", "gru_gat"]:
        job = Job(kind, "rich", 224, "real" if kind.endswith("gat") else "none")
        histories = Histories(fixture)
        with tempfile.TemporaryDirectory(dir=output) as temporary:
            resumed, continuous = (
                Path(temporary) / "resumed",
                Path(temporary) / "continuous",
            )
            training.fit(
                job,
                fixture,
                columns,
                list(range(zones)),
                resumed,
                local,
                "fixture",
                64,
                histories,
            )
            local["neural"]["epochs"] = 2
            fitted = training.fit(
                job,
                fixture,
                columns,
                list(range(zones)),
                resumed,
                local,
                "fixture",
                64,
                histories,
            )
            training.fit(
                job,
                fixture,
                columns,
                list(range(zones)),
                continuous,
                local,
                "fixture",
                64,
                histories,
            )
            actual = torch.load(
                resumed / "latest.pt", map_location="cpu", weights_only=False
            )
            expected = torch.load(
                continuous / "latest.pt", map_location="cpu", weights_only=False
            )
            if [item["epoch"] for item in actual["curves"]] != [1, 2]:
                raise RuntimeError("Resume repeated or skipped an epoch")
            for name in actual["model"]:
                torch.testing.assert_close(
                    actual["model"][name], expected["model"][name], rtol=1e-4, atol=1e-5
                )
            reloaded = training.reload_fitted(
                job, columns, fixture, resumed, local, histories
            )
            indices = np.flatnonzero(fixture["split"] == 1)
            before = training.predict(fitted, fixture, indices, columns, require_gpu())
            after = training.predict(reloaded, fixture, indices, columns, require_gpu())
            np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-4)
            local["neural"]["epochs"] = 1
        print(
            f"PASS {kind}: actual epoch resume and checkpoint predictions", flush=True
        )


def run(config):
    require_gpu()
    torch.set_num_threads(4)
    settings = {
        "static": config["neural"]["batch_size"],
        "temporal": config["neural"]["batch_size"],
    }
    checks = []
    for family, cases in [
        (
            "static",
            [("mlp", "none"), ("gat", "real"), ("gat", "self"), ("gat", "shuffled")],
        ),
        ("temporal", [("gru_mlp", "none"), ("gru_gat", "real")]),
    ]:
        while True:
            try:
                results = [
                    exercise(kind, graph, settings[family], config)
                    for kind, graph in cases
                ]
                checks.extend(results)
                for result in results:
                    print("GPU PASS:", result, flush=True)
                break
            except torch.OutOfMemoryError:
                if settings[family] <= 8:
                    raise
                settings[family] //= 2
                torch.cuda.empty_cache()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    resume_check(config, REPORT.parent)
    result = {
        "passed": True,
        "gpu": torch.cuda.get_device_name(),
        "hip": torch.version.hip,
        "config": config,
        "source_hashes": source_hashes(),
        "batch_sizes": settings,
        "checks": checks,
        "epoch_resume_verified": True,
    }
    write_json(REPORT, result)
    return result


def checked(config):
    result = json.loads(REPORT.read_text())
    if (
        not result["passed"]
        or result["source_hashes"] != source_hashes()
        or result["config"] != config
    ):
        raise ValueError(
            "GPU preflight must pass for the current code and configuration"
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run(json.loads(parser.parse_args().config.read_text()))
