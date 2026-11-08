"""Audit ten frozen temporal checkpoints on real batches; no fitting or overwrites."""

import json

import numpy as np
import pandas as pd
import torch

from epgnn.controlled_study import training
from epgnn.controlled_study.design import feature_indices, jobs
from epgnn.controlled_study.history import Histories
from epgnn.controlled_study.models import make_model
from epgnn.controlled_study.runner import check_complete, study_identity
from epgnn.dataset_build.storage import ROOT, digest
from epgnn.experiment_suite.artifacts import write_json
from epgnn.experiment_suite.neural import Preprocessor, require_gpu
from epgnn.experiment_suite.runner import checked_cache


def errors(a, b, mask):
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))[mask]
    if not len(delta) or not np.isfinite(delta).all():
        raise ValueError("Empty or nonfinite reference comparison")
    return {"max_eur_mwh": float(delta.max()), "mean_eur_mwh": float(delta.mean())}


def main():
    config = json.loads(
        (ROOT / "configs/experiments/controlled_study.json").read_text()
    )
    summary = json.loads((ROOT / "results/controlled_study_full.json").read_text())
    token, _ = study_identity(config)
    if token != summary["fingerprint"] or not summary["complete"]:
        raise ValueError("Frozen controlled-study identity changed")
    _, metadata, arrays = checked_cache(config)
    history = Histories(arrays)
    zones = metadata["eligible"]
    device = require_gpu()
    torch.set_num_threads(4)
    records = []
    for job in [j for j in jobs(config) if j.model.startswith("gru_")]:
        directory = ROOT / "runs/controlled_study" / token / "full" / job.name
        check_complete(directory)
        state = torch.load(
            directory / "best.pt", map_location="cpu", weights_only=False
        )
        columns = feature_indices(job)
        gpu = (
            make_model(job.model, len(columns) * 2, config["neural"]).to(device).eval()
        )
        cpu = make_model(job.model, len(columns) * 2, config["neural"]).double().eval()
        gpu.load_state_dict(state["model"])
        cpu.load_state_dict({k: v.double() for k, v in state["model"].items()})
        for name, value in gpu.state_dict().items():
            torch.testing.assert_close(
                value.cpu().double(), cpu.state_dict()[name], rtol=0, atol=0
            )
        preprocessing = Preprocessor.restore(state["preprocessing"])
        history.restore(state["history"])
        edges = torch.from_numpy(training.graph_for(job, arrays))
        cases = []
        for split, code, offset in [
            ("val", 1, 0),
            ("val", 1, 24 * 336),
            ("test", 2, 0),
            ("fit_shuffled", 0, 0),
        ]:
            indices = np.flatnonzero(arrays["split"] == code)
            if code == 0:
                indices = np.random.default_rng(job.seed + state["epoch"]).permutation(
                    indices
                )
            selected = indices[offset : offset + state["batch_size"]]
            x, _, active, sequence = training.tensors(
                arrays, selected, columns, preprocessing, history, torch.device("cpu")
            )
            values, inverse = sequence
            with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
                independent = cpu(
                    x.double(), edges, active, (values.double(), inverse)
                ).numpy()
                native = (
                    gpu(
                        x.to(device),
                        edges.to(device),
                        active.to(device),
                        (values.to(device), inverse.to(device)),
                    )
                    .cpu()
                    .numpy()
                )
            with torch.no_grad(), torch.backends.cudnn.flags(enabled=True):
                fused = (
                    gpu(
                        x.to(device),
                        edges.to(device),
                        active.to(device),
                        (values.to(device), inverse.to(device)),
                    )
                    .cpu()
                    .numpy()
                )
            price = [
                50 * np.sinh(p.astype(np.float64))[:, zones]
                for p in [independent, native, fused]
            ]
            mask = np.isfinite(np.asarray(arrays["y"][selected])[:, zones])
            times = pd.to_datetime(np.asarray(arrays["times"][selected]), utc=True)
            replay = None
            if code != 0:
                saved = pd.read_parquet(
                    directory / f"predictions_{split}.parquet",
                    filters=[("ts_utc", ">=", times[0]), ("ts_utc", "<=", times[-1])],
                )
                expected = (
                    saved.pivot(index="ts_utc", columns="zone", values="prediction")
                    .reindex(index=times, columns=[metadata["zones"][j] for j in zones])
                    .to_numpy()
                )
                replay = errors(price[2], expected, mask)
            native_error = errors(price[0], price[1], mask)
            fused_error = errors(price[0], price[2], mask)
            cases.append(
                {
                    "split": split,
                    "hour_offset": offset,
                    "batch_hours": len(selected),
                    "native_gpu_vs_cpu_float64": native_error,
                    "fused_gpu_vs_cpu_float64": fused_error,
                    "unique_history_origins": len(values),
                    "fused_gpu_vs_saved_predictions": replay,
                }
            )
        native_pass = all(
            c["native_gpu_vs_cpu_float64"]["max_eur_mwh"] <= 0.01
            and c["native_gpu_vs_cpu_float64"]["mean_eur_mwh"] <= 0.0001
            for c in cases
        )
        fused_pass = all(
            c["fused_gpu_vs_cpu_float64"]["max_eur_mwh"] <= 0.01
            and c["fused_gpu_vs_cpu_float64"]["mean_eur_mwh"] <= 0.0001
            for c in cases
        )
        record = {
            "job": job.name,
            "checkpoint_sha256": digest(directory / "best.pt")["sha256"],
            "same_weights": True,
            "native_reference_passed": native_pass,
            "fused_reference_passed": fused_pass,
            "saved_replay_passed": all(
                c["fused_gpu_vs_saved_predictions"] is None
                or (
                    c["fused_gpu_vs_saved_predictions"]["max_eur_mwh"] <= 0.01
                    and c["fused_gpu_vs_saved_predictions"]["mean_eur_mwh"] <= 0.0001
                )
                for c in cases
            ),
            "cases": cases,
        }
        print(json.dumps(record), flush=True)
        records.append(record)
    result = {
        "fingerprint": token,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name(),
        "training_rerun": False,
        "checkpoint_count": len(records),
        "cases_per_checkpoint": 4,
        "tolerance": {"max_eur_mwh": 0.01, "mean_eur_mwh": 0.0001},
        "native_reference_passed": all(r["native_reference_passed"] for r in records),
        "fused_reference_passed": all(r["fused_reference_passed"] for r in records),
        "saved_replay_passed": all(r["saved_replay_passed"] for r in records),
        "scope": "Three actual inference batches and one shuffled training-input batch per checkpoint, all ten selected models. All forwards use evaluation mode with dropout disabled. This is a sampled forward backend check, not full-cohort CPU reevaluation or a backward-pass audit. Original weights, scores and prediction files are unchanged; native inference alone does not correct training performed with a discrepant backend.",
        "checkpoints": records,
    }
    write_json(ROOT / "results/controlled_gru_backend_audit.json", result)


if __name__ == "__main__":
    main()
