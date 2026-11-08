"""ROCm checks for the actual static-model features and epoch resume path."""

import tempfile
from copy import deepcopy

import numpy as np
import torch

from epgnn.controlled_study import training
from epgnn.controlled_study.design import Job
from epgnn.controlled_study.models import parameter_count
from epgnn.controlled_study.preflight import exercise
from epgnn.experiment_suite.artifacts import write_json
from epgnn.experiment_suite.neural import require_gpu

from .data import ALL_FEATURES, indices


def run(config, path, token):
    require_gpu()
    torch.set_num_threads(4)
    checks = [
        exercise(model, graph, config["neural"]["batch_size"], config)
        for model, graph in [("mlp", "none"), ("gat", "real")]
    ]
    rng = np.random.default_rng(224)
    arrays = {
        "x": rng.normal(size=(288, 4, len(ALL_FEATURES))).astype(np.float32),
        "y": rng.normal(50, 10, size=(288, 4)).astype(np.float32),
        "active": np.ones((288, 4), bool),
        "split": np.array([0] * 192 + [1] * 48 + [2] * 48),
        "edges": np.array([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=np.int64),
    }
    arrays["x"][0, 0, -4:] = np.nan
    arrays["y"][0, 0] = np.nan
    columns = indices("airport_history")
    local = deepcopy(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    for model in ["mlp", "gat"]:
        job = Job(model, "airport_history", 224, "real" if model == "gat" else "none")
        with tempfile.TemporaryDirectory(dir=path.parent) as temp:
            from pathlib import Path

            resumed, continuous = Path(temp) / "resumed", Path(temp) / "continuous"
            local["neural"]["epochs"] = 1
            training.fit(job, arrays, columns, [0, 1, 2, 3], resumed, local, token, 64)
            local["neural"]["epochs"] = 2
            fitted = training.fit(
                job, arrays, columns, [0, 1, 2, 3], resumed, local, token, 64
            )
            training.fit(
                job, arrays, columns, [0, 1, 2, 3], continuous, local, token, 64
            )
            actual = torch.load(
                resumed / "latest.pt", weights_only=False, map_location="cpu"
            )
            expected = torch.load(
                continuous / "latest.pt", weights_only=False, map_location="cpu"
            )
            if [r["epoch"] for r in actual["curves"]] != [1, 2]:
                raise ValueError("GPU resume skipped/repeated an epoch")
            for name in actual["model"]:
                torch.testing.assert_close(
                    actual["model"][name], expected["model"][name], rtol=1e-4, atol=1e-5
                )
            restored = training.reload_fitted(job, columns, arrays, resumed, local)
            selected = np.flatnonzero(arrays["split"] == 1)
            before = training.predict(fitted, arrays, selected, columns, require_gpu())
            after = training.predict(restored, arrays, selected, columns, require_gpu())
            np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-4)
            print(
                f"GPU PASS {model}: actual airport features, masked loss, backward, epoch resume, reload; {parameter_count(fitted.model)} parameters",
                flush=True,
            )
    evidence = {
        "passed": True,
        "fingerprint": token,
        "gpu": torch.cuda.get_device_name(),
        "hip": torch.version.hip,
        "checks": checks,
        "epoch_resume_verified": True,
        "batch_sizes": {"static": config["neural"]["batch_size"]},
    }
    write_json(path, evidence)
    return evidence
