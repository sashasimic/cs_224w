"""GPU preflight: graph operators, masked losses and resumable training."""

from pathlib import Path
import random
import time

import numpy as np
import torch

from epgnn.dataset_build.storage import ROOT, digest
from . import data, neural
from .artifacts import write_json


def gpu_smoke(config, output):
    device = neural.require_gpu()
    torch.set_num_threads(4)
    random.seed(224)
    np.random.seed(224)
    torch.manual_seed(224)
    torch.cuda.manual_seed_all(224)
    report = {
        "gpu": torch.cuda.get_device_name(),
        "hip": torch.version.hip,
        "torch": torch.__version__,
        "checks": [],
        "peak_vram_gib": 0.0,
        "batch_size": config["neural"]["batch_size"],
        "neural_code_sha256": digest(neural.__file__)["sha256"],
        "loss_code_sha256": digest(ROOT / "epgnn/train.py")["sha256"],
    }
    zones = 62
    pairs = [(a, b) for a in range(zones) for b in range(zones) if a != b]
    chosen = np.random.default_rng(224).choice(len(pairs), 520, replace=False)
    real = np.array([pairs[k] for k in chosen], np.int64).T
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for shuffled, graph in [(False, real), (True, data.shuffle_edges(real))]:
        edges = torch.tensor(graph.copy(), device=device)
        batch_size = report["batch_size"]
        while True:
            try:
                torch.cuda.reset_peak_memory_stats()
                prep = neural.Preprocessor().fit(
                    np.random.default_rng(224)
                    .normal(size=(100, len(data.RICH)))
                    .astype(np.float32)
                )
                model = neural.PriceGAT(len(data.RICH) * 2).to(device)
                opt = torch.optim.Adam(model.parameters(), lr=0.001)
                x = torch.randn(batch_size, zones, len(data.RICH) * 2, device=device)
                active = torch.ones(batch_size, zones, dtype=torch.bool, device=device)
                active[:, -3:] = False
                y = torch.randn(batch_size, zones, device=device)
                y[::2, 0] = float("nan")
                valid = torch.isfinite(y)
                started = time.monotonic()
                for _ in range(3):
                    opt.zero_grad(set_to_none=True)
                    prediction = model(x, edges, active)
                    loss = neural.masked_loss(prediction, y, valid)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Smoke loss is not finite")
                    loss.backward()
                    if any(
                        p.grad is None or not torch.isfinite(p.grad).all()
                        for p in model.parameters()
                    ):
                        raise RuntimeError("Smoke gradients are missing/nonfinite")
                    opt.step()
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() / 2**30
                if peak > 12:
                    raise torch.OutOfMemoryError("Above 12 GiB safety budget")
                break
            except torch.OutOfMemoryError:
                if batch_size <= 8:
                    raise
                batch_size //= 2
                del model, opt, x, y
                torch.cuda.empty_cache()
        checkpoint = output / ("shuffled.pt" if shuffled else "real.pt")
        model.eval()
        with torch.no_grad():
            before = model(x, edges, active).cpu()
        neural.atomic_checkpoint(
            checkpoint,
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "prep": prep.state(),
            },
        )
        saved = torch.load(checkpoint, weights_only=False, map_location=device)
        restored = neural.PriceGAT(len(data.RICH) * 2).to(device)
        restored.load_state_dict(saved["model"])
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(
                before, restored(x, edges, active).cpu(), rtol=1e-5, atol=1e-6
            )
        restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.001)
        restored_optimizer.load_state_dict(saved["optimizer"])
        restored_optimizer.zero_grad(set_to_none=True)
        neural.masked_loss(restored(x, edges, active), y, valid).backward()
        restored_optimizer.step()
        if not all(torch.isfinite(p).all() for p in restored.parameters()):
            raise RuntimeError("Resumed optimizer produced invalid weights")
        report["checks"].append(
            {
                "graph": "shuffled" if shuffled else "real",
                "training_steps": 3,
                "checks": [
                    "forward",
                    "NaN-masked loss",
                    "backward",
                    "Adam",
                    "validation",
                    "checkpoint reload",
                    "optimizer resume",
                ],
                "seconds": time.monotonic() - started,
                "peak_vram_gib": peak,
            }
        )
        report["peak_vram_gib"] = max(report["peak_vram_gib"], peak)
        report["batch_size"] = min(report["batch_size"], batch_size)
        print("GPU smoke PASS:", report["checks"][-1], flush=True)
    # Exercise the actual epoch/checkpoint loop, not just an isolated optimizer.
    rng = np.random.default_rng(224)
    fixture = {
        "x": rng.normal(size=(128, 8, len(data.FEATURES))).astype(np.float32),
        "y": rng.normal(50, 10, size=(128, 8)).astype(np.float32),
        "active": np.ones((128, 8), bool),
        "split": np.array([0] * 96 + [1] * 16 + [2] * 16),
        "edges": np.array(
            [[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 0]], np.int64
        ),
    }
    fixture["shuffled_edges"] = data.shuffle_edges(fixture["edges"])
    fixture["y"][0, 0] = np.nan
    fixture_cfg = {**config["neural"], "epochs": 1, "identity": "gpu-resume-fixture"}
    import tempfile

    with tempfile.TemporaryDirectory(dir=output) as temp:
        resumed = Path(temp) / "resumed"
        continuous = Path(temp) / "continuous"
        neural.train(
            fixture, list(range(len(data.RICH))), list(range(8)), resumed, fixture_cfg
        )
        fixture_cfg["epochs"] = 2
        neural.train(
            fixture, list(range(len(data.RICH))), list(range(8)), resumed, fixture_cfg
        )
        neural.train(
            fixture,
            list(range(len(data.RICH))),
            list(range(8)),
            continuous,
            fixture_cfg,
        )
        actual = torch.load(
            resumed / "latest.pt", map_location="cpu", weights_only=False
        )
        expected = torch.load(
            continuous / "latest.pt", map_location="cpu", weights_only=False
        )
        if [c["epoch"] for c in actual["curves"]] != [1, 2]:
            raise RuntimeError("Epoch resume repeated or skipped training")
        for name in actual["model"]:
            torch.testing.assert_close(
                actual["model"][name], expected["model"][name], rtol=1e-4, atol=1e-5
            )
    report["epoch_resume_matches_continuous_training"] = True
    report["passed"] = True
    write_json(output / "report.json", report)
    return report
